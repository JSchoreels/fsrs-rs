use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::{FSRS5_DEFAULT_DECAY, MemoryState, Parameters};
use crate::parameter_clipper::clip_parameters;
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

const FSRS6_PARAM_LEN: usize = 21;
const FSRS7_PARAM_LEN: usize = 35;
const FSRS7_S90_TARGET: f32 = 0.9;
const FSRS7_LUT_SIZE: usize = 256;
const FSRS7_BISECTION_ITERS: usize = 50;
const FSRS7_BRENT_ITERS: usize = 10;
const FSRS7_BRENT_TOL: f64 = 1e-3;
const FSRS7_DR_MIN: f32 = 0.0001;
const FSRS7_DR_MAX: f32 = 0.9999;
const FSRS7_LUT_S_MIN: f32 = 0.03;
const FSRS7_LUT_S_MAX: f32 = 3000.0;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

fn tensor_min<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().greater(b.clone()), b)
}

fn tensor_max<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().lower(b.clone()), b)
}

fn fsrs7_forgetting_curve_scalar(w: &[f32], t: f32, s: f32) -> f32 {
    let s = s.max(S_MIN);
    let t_over_s = t.max(0.0) / s;

    let decay1 = -w[27];
    let decay2 = -w[28];
    let base1 = w[29];
    let base2 = w[30];

    let factor1 = base1.powf(1.0 / decay1) - 1.0;
    let factor2 = base2.powf(1.0 / decay2) - 1.0;
    let r1 = (1.0 + factor1 * t_over_s).powf(decay1);
    let r2 = (1.0 + factor2 * t_over_s).powf(decay2);

    let weight1 = w[31] * s.powf(-w[33]);
    let weight2 = w[32] * s.powf(w[34]);

    (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
}

#[derive(Debug)]
struct Fsrs7S90Lut {
    log_s_min: f32,
    log_s_step: f32,
    t90_grid: Vec<f32>,
}

impl Fsrs7S90Lut {
    fn build(w: &[f32]) -> Self {
        let log_s_min = FSRS7_LUT_S_MIN.max(S_MIN).ln();
        let log_s_max = FSRS7_LUT_S_MAX.min(S_MAX).ln();
        let log_s_step = (log_s_max - log_s_min) / (FSRS7_LUT_SIZE - 1) as f32;
        let mut t90_grid = Vec::with_capacity(FSRS7_LUT_SIZE);
        for i in 0..FSRS7_LUT_SIZE {
            let log_s = log_s_min + i as f32 * log_s_step;
            let s = log_s.exp();
            t90_grid.push(fsrs7_next_interval_bisection_scalar(
                w,
                s,
                FSRS7_S90_TARGET,
                None,
            ));
        }
        Self {
            log_s_min,
            log_s_step,
            t90_grid,
        }
    }

    fn interpolate_t90(&self, stability: f32) -> f32 {
        if self.t90_grid.len() == 1 {
            return self.t90_grid[0];
        }
        let s = stability
            .max(FSRS7_LUT_S_MIN.max(S_MIN))
            .min(FSRS7_LUT_S_MAX.min(S_MAX));
        let max_index = (self.t90_grid.len() - 1) as f32;
        let position = ((s.ln() - self.log_s_min) / self.log_s_step).clamp(0.0, max_index);
        let left = position.floor() as usize;
        let right = (left + 1).min(self.t90_grid.len() - 1);
        if left == right {
            return self.t90_grid[left];
        }
        let weight = position - left as f32;
        self.t90_grid[left] + weight * (self.t90_grid[right] - self.t90_grid[left])
    }
}

static FSRS7_S90_LUT_CACHE: OnceLock<Mutex<HashMap<u64, Arc<Fsrs7S90Lut>>>> = OnceLock::new();

fn fsrs7_params_hash(w: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    w.iter()
        .take(FSRS7_PARAM_LEN)
        .for_each(|x| x.to_bits().hash(&mut hasher));
    hasher.finish()
}

fn fsrs7_s90_lut(w: &[f32]) -> Arc<Fsrs7S90Lut> {
    let key = fsrs7_params_hash(w);
    let cache = FSRS7_S90_LUT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(lut) = cache
        .lock()
        .expect("fsrs7 lut cache lock poisoned")
        .get(&key)
        .cloned()
    {
        return lut;
    }
    let built = Arc::new(Fsrs7S90Lut::build(w));
    let mut guard = cache.lock().expect("fsrs7 lut cache lock poisoned");
    guard.entry(key).or_insert_with(|| built.clone()).clone()
}

fn brent_root_f64<F>(mut f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64>
where
    F: FnMut(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);
    if !fa.is_finite() || !fb.is_finite() || fa * fb > 0.0 {
        return None;
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = c;
    let mut mflag = true;

    for _ in 0..max_iter {
        let mut s = if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else if (fb - fa).abs() > f64::EPSILON {
            b - fb * (b - a) / (fb - fa)
        } else {
            (a + b) * 0.5
        };

        let lower = ((3.0 * a) + b) * 0.25;
        let upper = b;
        let cond1 = (s <= lower && s <= upper) || (s >= lower && s >= upper);
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() * 0.5;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() * 0.5;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;
        if cond1 || cond2 || cond3 || cond4 || cond5 {
            s = (a + b) * 0.5;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        if fb.abs() <= tol {
            return Some(b);
        }
    }
    None
}

fn fsrs7_next_interval_bisection_scalar(
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    high_hint: Option<f32>,
) -> f32 {
    let desired_retention = desired_retention.clamp(FSRS7_DR_MIN, FSRS7_DR_MAX);
    let stability = stability.max(S_MIN);

    if desired_retention >= FSRS7_DR_MAX {
        return 0.0;
    }

    let mut low = 0.0;
    let mut high = high_hint
        .unwrap_or_else(|| stability.max(1.0))
        .clamp(0.0, S_MAX)
        .max(S_MIN);

    while fsrs7_forgetting_curve_scalar(w, high, stability) > desired_retention && high < S_MAX {
        high = (high * 2.0).min(S_MAX);
        if (high - S_MAX).abs() < f32::EPSILON {
            break;
        }
    }

    for _ in 0..FSRS7_BISECTION_ITERS {
        let mid = (low + high) / 2.0;
        let r = fsrs7_forgetting_curve_scalar(w, mid, stability);
        if r > desired_retention {
            low = mid;
        } else {
            high = mid;
        }
    }

    ((low + high) / 2.0).clamp(0.0, S_MAX)
}

fn fsrs7_next_interval_scalar(
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    lut: &Fsrs7S90Lut,
) -> f32 {
    let desired_retention = desired_retention.clamp(FSRS7_DR_MIN, FSRS7_DR_MAX);
    let stability = stability.max(S_MIN);
    if desired_retention >= FSRS7_DR_MAX {
        return 0.0;
    }

    let f = |t: f32| fsrs7_forgetting_curve_scalar(w, t, stability) - desired_retention;
    let mut high = lut.interpolate_t90(stability).clamp(0.0, S_MAX).max(S_MIN);
    let mut f_high = f(high);
    if !f_high.is_finite() {
        return fsrs7_next_interval_bisection_scalar(w, stability, desired_retention, Some(high));
    }
    while f_high > 0.0 && high < S_MAX {
        high = (high * 2.0).min(S_MAX);
        f_high = f(high);
    }
    if !f_high.is_finite() {
        return fsrs7_next_interval_bisection_scalar(w, stability, desired_retention, Some(high));
    }
    if f_high > 0.0 {
        return S_MAX;
    }

    let brent = brent_root_f64(
        |x| f(x as f32) as f64,
        0.0,
        high as f64,
        FSRS7_BRENT_TOL,
        FSRS7_BRENT_ITERS,
    );
    if let Some(root) = brent {
        root as f32
    } else {
        fsrs7_next_interval_bisection_scalar(w, stability, desired_retention, Some(high))
    }
}

impl<B: Backend> Model<B> {
    pub fn new(config: ModelConfig) -> Self {
        Self::new_with_device(config, &B::Device::default())
    }

    pub fn new_with_device(config: ModelConfig, device: &B::Device) -> Self {
        let mut initial_params = DEFAULT_PARAMETERS.to_vec();
        if let Some(initial_stability) = config.initial_stability {
            initial_params[0..4].copy_from_slice(&initial_stability);
        }
        if let Some(initial_forgetting_curve) = config.initial_forgetting_curve {
            initial_params[27..35].copy_from_slice(&initial_forgetting_curve);
        }
        if config.freeze_short_term_stability {
            // Legacy (FSRS-6) short-term terms only.
            initial_params[17] = 0.0;
            initial_params[18] = 0.0;
            initial_params[19] = 0.0;
        }

        Self {
            w: Param::from_tensor(Tensor::from_floats(
                TensorData::new(
                    initial_params.clone(),
                    Shape {
                        dims: vec![initial_params.len()],
                    },
                ),
                device,
            )),
        }
    }

    fn is_fsrs7(&self) -> bool {
        self.w.val().dims()[0] >= FSRS7_PARAM_LEN
    }

    fn power_forgetting_curve_fsrs6(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let decay = -self.w.get(20);
        let factor = decay.clone().powi_scalar(-1).mul_scalar(0.9f32.ln()).exp() - 1.0;
        (t / s * factor + 1.0).powf(decay)
    }

    fn power_forgetting_curve_fsrs7(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let t_over_s = t / s.clone();

        let decay1 = -self.w.get(27);
        let decay2 = -self.w.get(28);
        let base1 = self.w.get(29);
        let base2 = self.w.get(30);

        let factor1 = base1.clone().powf(decay1.clone().powi_scalar(-1)) - 1.0;
        let factor2 = base2.clone().powf(decay2.clone().powi_scalar(-1)) - 1.0;

        let r1 = (t_over_s.clone() * factor1 + 1.0).powf(decay1);
        let r2 = (t_over_s * factor2 + 1.0).powf(decay2);

        let weight1 = self.w.get(31) * s.clone().powf(-self.w.get(33));
        let weight2 = self.w.get(32) * s.powf(self.w.get(34));

        (weight1.clone() * r1 + weight2.clone() * r2) / (weight1 + weight2)
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        if self.is_fsrs7() {
            self.power_forgetting_curve_fsrs7(t, s)
        } else {
            self.power_forgetting_curve_fsrs6(t, s)
        }
    }

    fn next_interval_fsrs6(
        &self,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let decay = -self.w.get(20);
        let factor = decay.clone().powi_scalar(-1).mul_scalar(0.9f32.ln()).exp() - 1.0;
        stability / factor * (desired_retention.powf(decay.powi_scalar(-1)) - 1.0)
    }

    fn next_interval_fsrs7(
        &self,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let device = stability.device();
        let w = self.w.val().to_data().to_vec::<f32>().unwrap();
        let lut = fsrs7_s90_lut(&w);
        let stabilities = stability.to_data().to_vec::<f32>().unwrap();
        let desired = desired_retention.to_data().to_vec::<f32>().unwrap();

        let intervals: Vec<f32> = stabilities
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let dr = desired.get(i).copied().unwrap_or_else(|| desired[0]);
                fsrs7_next_interval_scalar(&w, s, dr, lut.as_ref())
            })
            .collect();
        Tensor::from_floats(intervals.as_slice(), &device)
    }

    pub fn next_interval(
        &self,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        if self.is_fsrs7() {
            self.next_interval_fsrs7(stability, desired_retention)
        } else {
            self.next_interval_fsrs6(stability, desired_retention)
        }
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
        rating: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let batch_size = rating.dims()[0];
        let device = rating.device();
        let hard_penalty = Tensor::ones([batch_size], &device)
            .mask_where(rating.clone().equal_elem(2), self.w.get(15));
        let easy_bonus =
            Tensor::ones([batch_size], &device).mask_where(rating.equal_elem(4), self.w.get(16));

        last_s.clone()
            * (self.w.get(8).exp()
                * (-last_d + 11)
                * (last_s.powf(-self.w.get(9)))
                * (((-r + 1) * self.w.get(10)).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1)
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let new_s = self.w.get(11)
            * last_d.powf(-self.w.get(12))
            * ((last_s.clone() + 1).powf(self.w.get(13)) - 1)
            * ((-r + 1) * self.w.get(14)).exp();
        let new_s_min = last_s / (self.w.get(17) * self.w.get(18)).exp();
        new_s
            .clone()
            .mask_where(new_s_min.clone().lower(new_s), new_s_min)
    }

    fn stability_short_term(&self, last_s: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let sinc = (self.w.get(17) * (rating.clone() - 3 + self.w.get(18))).exp()
            * last_s.clone().powf(-self.w.get(19));

        last_s
            * sinc
                .clone()
                .mask_where(rating.greater_equal_elem(2), sinc.clamp_min(1.0))
    }

    fn mean_reversion(&self, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
        let device = new_d.device();
        let rating = Tensor::from_floats([4.0], &device);
        self.w.get(7) * (self.init_difficulty(rating) - new_d.clone()) + new_d
    }

    fn fsrs7_stability_for_set(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        start: usize,
    ) -> Tensor<B, 1> {
        let batch_size = rating.dims()[0];
        let device = rating.device();
        let hard_penalty = Tensor::ones([batch_size], &device)
            .mask_where(rating.clone().equal_elem(2), self.w.get(start + 7));
        let easy_bonus = Tensor::ones([batch_size], &device)
            .mask_where(rating.clone().equal_elem(4), self.w.get(start + 8));

        let new_s_fail = self.w.get(start + 3)
            * last_d.clone().powf(-self.w.get(start + 4))
            * ((last_s.clone() + 1).powf(self.w.get(start + 5)) - 1)
            * ((-r.clone() + 1) * self.w.get(start + 6)).exp();
        let pls = tensor_min(last_s.clone(), new_s_fail);

        let sinc = self.w.get(start).add_scalar(-1.5).exp()
            * last_d.neg().add_scalar(11.0)
            * last_s.clone().powf(-self.w.get(start + 1))
            * (((-r + 1) * self.w.get(start + 2)).exp() - 1)
            * hard_penalty
            * easy_bonus
            + 1;
        let new_s_success = tensor_max(pls.clone(), last_s * sinc);
        let success = rating.greater_elem(1);
        pls.mask_where(success, new_s_success)
    }

    fn fsrs7_transition_function(&self, delta_t: Tensor<B, 1>) -> Tensor<B, 1> {
        (self.w.get(26) * (-self.w.get(25) * delta_t).exp())
            .neg()
            .add_scalar(1.0)
    }

    fn fsrs7_mean_reversion(&self, init: Tensor<B, 1>, current: Tensor<B, 1>) -> Tensor<B, 1> {
        init.mul_scalar(0.01) + current.mul_scalar(0.99)
    }

    fn fsrs7_next_difficulty(
        &self,
        difficulty: Tensor<B, 1>,
        rating: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let delta_d = -self.w.get(6) * (rating - 3);
        let new_d = difficulty.clone() + self.linear_damping(delta_d, difficulty);
        let device = new_d.device();
        let init = self.init_difficulty(Tensor::from_floats([4.0], &device));
        self.fsrs7_mean_reversion(init, new_d).clamp(D_MIN, D_MAX)
    }

    pub(crate) fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.val().select(0, rating.int() - 1)
    }

    fn init_difficulty(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(4) - (self.w.get(5) * (rating - 1)).exp() + 1
    }

    fn linear_damping(&self, delta_d: Tensor<B, 1>, old_d: Tensor<B, 1>) -> Tensor<B, 1> {
        old_d.neg().add_scalar(10.0) * delta_d.div_scalar(9.0)
    }

    fn next_difficulty(&self, difficulty: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let delta_d = -self.w.get(6) * (rating - 3);
        difficulty.clone() + self.linear_damping(delta_d, difficulty)
    }

    pub(crate) fn step(
        &self,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: MemoryStateTensors<B>,
        nth: usize,
    ) -> MemoryStateTensors<B> {
        let last_s = state.stability.clone().clamp(S_MIN, S_MAX);
        let last_d = state.difficulty.clone().clamp(D_MIN, D_MAX);

        let mut new_s;
        let mut new_d;

        if self.is_fsrs7() {
            let retrievability = self.power_forgetting_curve(delta_t.clone(), last_s.clone());
            let new_s_long_term = self.fsrs7_stability_for_set(
                last_s.clone(),
                last_d.clone(),
                retrievability.clone(),
                rating.clone(),
                7,
            );
            let new_s_short_term = self.fsrs7_stability_for_set(
                last_s.clone(),
                last_d.clone(),
                retrievability,
                rating.clone(),
                16,
            );
            let coefficient = self.fsrs7_transition_function(delta_t.clone());
            let short_weight = coefficient.clone().neg().add_scalar(1.0);
            new_s = coefficient * new_s_long_term + short_weight * new_s_short_term;
            new_d = self.fsrs7_next_difficulty(last_d.clone(), rating.clone());
        } else {
            let retrievability = self.power_forgetting_curve(delta_t.clone(), last_s.clone());
            let stability_after_success = self.stability_after_success(
                last_s.clone(),
                last_d.clone(),
                retrievability.clone(),
                rating.clone(),
            );
            let stability_after_failure =
                self.stability_after_failure(last_s.clone(), last_d.clone(), retrievability);
            let stability_short_term = self.stability_short_term(last_s.clone(), rating.clone());
            new_s = stability_after_success
                .mask_where(rating.clone().equal_elem(1), stability_after_failure);
            new_s = new_s.mask_where(delta_t.equal_elem(0), stability_short_term);

            new_d = self.next_difficulty(last_d.clone(), rating.clone());
            new_d = self.mean_reversion(new_d).clamp(D_MIN, D_MAX);
        }

        if nth == 0 {
            let is_initial = state.stability.equal_elem(0.0);
            let init_s = self.init_stability(rating.clone().clamp(1, 4));
            let init_d = self
                .init_difficulty(rating.clone().clamp(1, 4))
                .clamp(D_MIN, D_MAX);
            new_s = new_s.mask_where(is_initial.clone(), init_s);
            new_d = new_d.mask_where(is_initial, init_d);
        }

        // mask padding zeros for rating
        new_s = new_s.mask_where(rating.clone().equal_elem(0), last_s);
        new_d = new_d.mask_where(rating.equal_elem(0), last_d);
        MemoryStateTensors {
            stability: new_s.clamp(S_MIN, S_MAX),
            difficulty: new_d,
        }
    }

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2>,
        starting_state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, batch_size] = delta_ts.dims();
        let mut state = if let Some(state) = starting_state {
            state
        } else {
            MemoryStateTensors::zeros(batch_size)
        };
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            let rating = ratings.get(i).squeeze(0);
            state = self.step(delta_t, rating, state, i);
        }
        state
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
}

impl<B: Backend> MemoryStateTensors<B> {
    pub(crate) fn zeros(batch_size: usize) -> MemoryStateTensors<B> {
        let device = B::Device::default();
        MemoryStateTensors {
            stability: Tensor::zeros([batch_size], &device),
            difficulty: Tensor::zeros([batch_size], &device),
        }
    }

    pub(crate) fn from_state(state: MemoryState) -> Self {
        let device = B::Device::default();
        Self {
            stability: Tensor::from_floats([state.stability], &device),
            difficulty: Tensor::from_floats([state.difficulty], &device),
        }
    }
}

#[derive(Config, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    pub initial_forgetting_curve: Option<[f32; 8]>,
    #[config(default = false)]
    pub freeze_short_term_stability: bool,
    #[config(default = 1)]
    pub num_relearning_steps: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArray> {
    model: Model<B>,
}

impl Default for FSRS<NdArray> {
    fn default() -> Self {
        Self::new(&[]).expect("Default parameters should be valid")
    }
}

impl FSRS<NdArray> {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: &Parameters) -> Result<Self> {
        Self::new_with_backend(parameters, &NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        parameters: &Parameters,
        device: &B2::Device,
    ) -> Result<FSRS<B2>> {
        let parameters = check_and_fill_parameters(parameters)?;
        let model = parameters_to_model::<B2>(&parameters, device);

        Ok(FSRS { model })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        &self.model
    }

    pub(crate) fn device(&self) -> B::Device {
        self.model().w.device()
    }
}

pub(crate) fn parameters_to_model<B: Backend>(
    parameters: &Parameters,
    device: &B::Device,
) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::new_with_device(config.clone(), device);
    let clipped = clip_parameters(parameters, config.num_relearning_steps, Default::default());
    model.w = Param::from_tensor(Tensor::from_floats(
        TensorData::new(
            clipped.clone(),
            Shape {
                dims: vec![clipped.len()],
            },
        ),
        device,
    ));
    model
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = match parameters.len() {
        0 => DEFAULT_PARAMETERS.to_vec(),
        17 => {
            let mut parameters = parameters.to_vec();
            parameters[4] = parameters[5].mul_add(2.0, parameters[4]);
            parameters[5] = parameters[5].mul_add(3.0, 1.0).ln() / 3.0;
            parameters[6] += 0.5;
            parameters.extend_from_slice(&[0.0, 0.0, 0.0, FSRS5_DEFAULT_DECAY]);
            parameters
        }
        19 => {
            let mut parameters = parameters.to_vec();
            parameters.extend_from_slice(&[0.0, FSRS5_DEFAULT_DECAY]);
            parameters
        }
        FSRS6_PARAM_LEN | FSRS7_PARAM_LEN => parameters.to_vec(),
        _ => return Err(FSRSError::InvalidParameters),
    };
    if parameters.iter().any(|&w| !w.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FSRS6_DEFAULT_PARAMETERS;
    use crate::test_helpers::TestHelper;
    use crate::test_helpers::{Model, Tensor};
    use burn::tensor::{TensorData, Tolerance};
    static DEVICE: burn::backend::ndarray::NdArrayDevice = NdArrayDevice::Cpu;

    fn fsrs6_model() -> Model {
        parameters_to_model::<crate::test_helpers::NdArrayAutodiff>(
            &FSRS6_DEFAULT_PARAMETERS,
            &DEVICE,
        )
    }

    #[test]
    fn test_w() {
        let model = Model::new(ModelConfig::default());
        assert_eq!(
            model.w.val().to_data(),
            TensorData::new(DEFAULT_PARAMETERS.to_vec(), Shape { dims: vec![35] })
        )
    }

    #[test]
    fn test_convert_parameters() {
        let fsrs4dot5_param = vec![
            0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26,
            0.29, 2.61,
        ];
        let fsrs5_param = check_and_fill_parameters(&fsrs4dot5_param).unwrap();
        assert_eq!(
            fsrs5_param,
            vec![
                0.4, 0.6, 2.4, 5.8, 6.81, 0.44675013, 1.36, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05,
                0.34, 1.26, 0.29, 2.61, 0.0, 0.0, 0.0, 0.5
            ]
        )
    }

    #[test]
    fn test_power_forgetting_curve() {
        let model = fsrs6_model();
        let delta_t = Tensor::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &DEVICE);
        let stability = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 4.0, 2.0], &DEVICE);
        let retrievability = model.power_forgetting_curve(delta_t, stability);

        retrievability.to_data().assert_approx_eq::<f32>(
            &TensorData::from([1.0, 0.9403443, 0.9253786, 0.9185229, 0.9, 0.8261359]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_init_stability() {
        let model = fsrs6_model();
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &DEVICE);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            TensorData::from([
                FSRS6_DEFAULT_PARAMETERS[0],
                FSRS6_DEFAULT_PARAMETERS[1],
                FSRS6_DEFAULT_PARAMETERS[2],
                FSRS6_DEFAULT_PARAMETERS[3],
                FSRS6_DEFAULT_PARAMETERS[0],
                FSRS6_DEFAULT_PARAMETERS[1]
            ])
        )
    }

    #[test]
    fn test_init_difficulty() {
        let model = fsrs6_model();
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &DEVICE);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            TensorData::from([
                FSRS6_DEFAULT_PARAMETERS[4],
                FSRS6_DEFAULT_PARAMETERS[4] - FSRS6_DEFAULT_PARAMETERS[5].exp() + 1.0,
                FSRS6_DEFAULT_PARAMETERS[4] - (2.0 * FSRS6_DEFAULT_PARAMETERS[5]).exp() + 1.0,
                FSRS6_DEFAULT_PARAMETERS[4] - (3.0 * FSRS6_DEFAULT_PARAMETERS[5]).exp() + 1.0,
                FSRS6_DEFAULT_PARAMETERS[4],
                FSRS6_DEFAULT_PARAMETERS[4] - FSRS6_DEFAULT_PARAMETERS[5].exp() + 1.0,
            ])
        )
    }

    #[test]
    fn test_forward() {
        let model = fsrs6_model();
        let delta_ts = Tensor::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ],
            &DEVICE,
        );
        let ratings = Tensor::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            ],
            &DEVICE,
        );
        let state = model.forward(delta_ts, ratings, None);
        let stability = state.stability.to_data();
        let difficulty = state.difficulty.to_data();

        stability.to_vec::<f32>().unwrap().assert_approx_eq([
            0.10088589,
            3.2494123,
            7.3153,
            18.014914,
            0.112798266,
            4.4694576,
        ]);

        difficulty
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.806304, 6.7404594, 2.1112142, 1.0, 8.806304, 6.7404594]);
    }

    #[test]
    fn test_next_difficulty() {
        let model = fsrs6_model();
        let difficulty = Tensor::from_floats([5.0; 4], &DEVICE);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &DEVICE);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();

        next_difficulty
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.354889, 6.6774445, 5.0, 3.3225555]);
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();

        next_difficulty
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.341763, 6.6659956, 4.990228, 3.3144615]);
    }

    #[test]
    fn test_next_stability() {
        let model = fsrs6_model();
        let stability = Tensor::from_floats([5.0; 4], &DEVICE);
        let difficulty = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &DEVICE);
        let retrievability = Tensor::from_floats([0.9, 0.8, 0.7, 0.6], &DEVICE);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &DEVICE);
        let s_recall = model.stability_after_success(
            stability.clone(),
            difficulty.clone(),
            retrievability.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();

        s_recall
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([25.602541, 28.226582, 58.656002, 127.226685]);
        let s_forget = model.stability_after_failure(stability.clone(), difficulty, retrievability);
        s_forget.clone().backward();

        s_forget
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.0525396, 1.1894329, 1.3680838, 1.584989]);
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();

        next_stability
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.0525396, 28.226582, 58.656002, 127.226685]);
        let next_stability = model.stability_short_term(stability, rating);

        next_stability
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.596818, 5.0, 5.0, 8.12961]);
    }

    #[test]
    fn test_fsrs() {
        FSRS::default()
            .model()
            .w
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq(DEFAULT_PARAMETERS);
        assert!(FSRS::new(&[]).is_ok());
        assert!(FSRS::new(&[1.]).is_err());
        assert!(FSRS::new(DEFAULT_PARAMETERS.as_slice()).is_ok());
        assert!(FSRS::new(&FSRS6_DEFAULT_PARAMETERS[..17]).is_ok());
        assert!(FSRS::new(&FSRS6_DEFAULT_PARAMETERS).is_ok());
    }

    #[test]
    fn test_fsrs7_path_produces_finite_interval() {
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::from_floats([[0.0], [3.0]], &DEVICE);
        let ratings = Tensor::from_floats([[3.0], [3.0]], &DEVICE);
        let state = model.forward(delta_ts, ratings, None);
        let stability: f32 = state.stability.into_scalar();
        assert!(stability.is_finite());
        assert!(stability > 0.0);

        let interval = model
            .next_interval(
                Tensor::from_floats([stability], &DEVICE),
                Tensor::from_floats([0.9], &DEVICE),
            )
            .into_scalar();
        assert!(interval.is_finite());
        assert!(interval >= 0.0);
    }

    #[test]
    fn test_fsrs7_scalar_interval_hits_target_retrievability() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);
        for stability in [0.2, 1.0, 10.0, 100.0, 1000.0] {
            for desired in [0.5, 0.7, 0.8, 0.9, 0.95] {
                let t = fsrs7_next_interval_scalar(&w, stability, desired, &lut);
                let r = fsrs7_forgetting_curve_scalar(&w, t, stability);
                assert!(
                    (r - desired).abs() <= 1e-3 || (t - S_MAX).abs() < 1e-4,
                    "stability={stability}, desired={desired}, t={t}, r={r}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_scalar_interval_matches_bisection_baseline() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);
        for stability in [0.2, 1.0, 10.0, 100.0, 1000.0] {
            for desired in [0.4, 0.6, 0.8, 0.9, 0.95] {
                let baseline = fsrs7_next_interval_bisection_scalar(&w, stability, desired, None);
                let optimized = fsrs7_next_interval_scalar(&w, stability, desired, &lut);
                let baseline_r = fsrs7_forgetting_curve_scalar(&w, baseline, stability);
                let optimized_r = fsrs7_forgetting_curve_scalar(&w, optimized, stability);
                assert!(
                    (optimized_r - baseline_r).abs() <= 1e-3,
                    "stability={stability}, desired={desired}, baseline={baseline}, optimized={optimized}, baseline_r={baseline_r}, optimized_r={optimized_r}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_s90_lut_cache_reuse() {
        let w = DEFAULT_PARAMETERS;
        let lut_a = fsrs7_s90_lut(&w);
        let lut_b = fsrs7_s90_lut(&w);
        assert!(std::sync::Arc::ptr_eq(&lut_a, &lut_b));

        let mut w2 = DEFAULT_PARAMETERS;
        w2[27] += 0.001;
        let lut_c = fsrs7_s90_lut(&w2);
        assert!(!std::sync::Arc::ptr_eq(&lut_a, &lut_c));
    }

    #[test]
    fn test_fsrs7_scalar_interval_monotonicity() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);

        for stability in [0.2, 1.0, 10.0, 100.0, 1000.0] {
            let desireds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95];
            let intervals: Vec<f32> = desireds
                .iter()
                .map(|&dr| fsrs7_next_interval_scalar(&w, stability, dr, &lut))
                .collect();
            for pair in intervals.windows(2) {
                assert!(
                    pair[0] >= pair[1],
                    "expected interval to decrease with higher retention: stability={stability}, intervals={intervals:?}"
                );
            }
        }

        for desired in [0.5, 0.7, 0.9] {
            let stabilities = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0];
            let intervals: Vec<f32> = stabilities
                .iter()
                .map(|&s| fsrs7_next_interval_scalar(&w, s, desired, &lut))
                .collect();
            for pair in intervals.windows(2) {
                assert!(
                    pair[1] >= pair[0],
                    "expected interval to increase with stability: desired={desired}, intervals={intervals:?}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_scalar_interval_matches_bisection_dense_grid() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);
        let desireds = [0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 0.98];

        for i in 0..25 {
            let p = i as f32 / 24.0;
            let stability = (S_MIN.ln() + (S_MAX.ln() - S_MIN.ln()) * p).exp();
            for desired in desireds {
                let baseline = fsrs7_next_interval_bisection_scalar(&w, stability, desired, None);
                let optimized = fsrs7_next_interval_scalar(&w, stability, desired, &lut);
                let baseline_r = fsrs7_forgetting_curve_scalar(&w, baseline, stability);
                let optimized_r = fsrs7_forgetting_curve_scalar(&w, optimized, stability);
                assert!(
                    (optimized_r - baseline_r).abs() <= 1e-3,
                    "stability={stability}, desired={desired}, baseline={baseline}, optimized={optimized}, baseline_r={baseline_r}, optimized_r={optimized_r}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_model_next_interval_vectorized_matches_scalar() {
        let model = Model::new(ModelConfig::default());
        let w = model.w.val().to_data().to_vec::<f32>().unwrap();
        let lut = fsrs7_s90_lut(&w);

        let stabilities = [0.2, 0.8, 3.0, 12.0, 40.0, 200.0];
        let desired = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5];

        let model_out = model
            .next_interval(
                Tensor::from_floats(stabilities, &DEVICE),
                Tensor::from_floats(desired, &DEVICE),
            )
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let scalar_out: Vec<f32> = stabilities
            .iter()
            .zip(desired.iter())
            .map(|(&s, &dr)| fsrs7_next_interval_scalar(&w, s, dr, lut.as_ref()))
            .collect();

        for (model_value, scalar_value) in model_out.iter().zip(scalar_out.iter()) {
            assert!((model_value - scalar_value).abs() < 1e-4);
        }
    }
}
