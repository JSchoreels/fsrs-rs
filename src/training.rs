use crate::batch_shuffle::{BatchTensorDataset, ShuffleDataLoader};
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    FSRSDataset, FSRSItem, WeightedFSRSItem, prepare_training_data, recency_weighted_fsrs_items,
};
use crate::error::Result;
use crate::model::{Model, ModelConfig};
use crate::parameter_clipper::parameter_clipper;
use crate::parameter_initialization::{initialize_parameters, smooth_and_fill};
use crate::simulation::{S_MAX, S_MIN};
use crate::{DEFAULT_PARAMETERS, FSRSError};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::nn::loss::Reduction;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::cast::ToElement;
use burn::train::TrainingInterrupter;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::{config::Config, tensor::backend::AutodiffBackend};
use core::marker::PhantomData;
use log::info;

use std::sync::{Arc, Mutex};

#[cfg(test)]
use crate::model::Get;

type B = NdArray<f32>;

static PARAMS_STDDEV: [f32; 35] = [
    9999.0, 9999.0, 9999.0, 9999.0, 0.523, 0.2528, 0.4329, 0.2966, 0.2139, 0.2889, 0.1862, 0.0829,
    0.175, 0.3812, 0.3013, 0.9104, 0.3234, 0.2448, 0.3273, 0.1842, 0.1542, 0.1735, 0.4608, 0.311,
    0.864, 0.4053, 0.162, 0.0418, 0.2596, 0.0798, 0.0682, 0.1282, 0.1397, 0.1407, 0.1489,
];

const FSRS7_PARAM_LEN: usize = 35;
const FSRS7_PENALTY_W_1: f64 = 0.5;
const FSRS7_PENALTY_W_2: f64 = 0.0015;
const FSRS7_PENALTY_W_L2: f64 = 0.5;
const FSRS7_PENALTY_N_REVIEWS: usize = 10;
const FSRS7_PENALTY_TARGET_DR: f32 = 0.90;
const FSRS7_PENALTY_TARGET_DRS: [f32; 1] = [0.99];
const FSRS7_PENALTY_N_NEWTON: usize = 4;
const FSRS7_MIN_T: f32 = 1.0 / 86400.0;
const FSRS7_MAX_T: f32 = 36500.0;
const FSRS7_ONE_DAY: f32 = 1.0;
const FSRS7_SHORT_C: f32 = 600.0 / 86400.0;
const FSRS7_INV_C: f32 = 1.0 / FSRS7_SHORT_C;
const FSRS7_GRAD_LEN: usize = 35;

#[derive(Clone, Copy, Debug)]
struct Dual35 {
    value: f64,
    grad: [f64; FSRS7_GRAD_LEN],
}

impl Dual35 {
    fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; FSRS7_GRAD_LEN],
        }
    }

    fn variable(value: f64, idx: usize) -> Self {
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        if idx < FSRS7_GRAD_LEN {
            grad[idx] = 1.0;
        }
        Self { value, grad }
    }

    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] + rhs.grad[i];
        }
        Self {
            value: self.value + rhs.value,
            grad,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] - rhs.grad[i];
        }
        Self {
            value: self.value - rhs.value,
            grad,
        }
    }

    fn neg(self) -> Self {
        self.mul_const(-1.0)
    }

    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] * rhs.value + rhs.grad[i] * self.value;
        }
        Self {
            value: self.value * rhs.value,
            grad,
        }
    }

    fn div(self, rhs: Self) -> Self {
        let denom = (rhs.value * rhs.value).max(1e-18);
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = (self.grad[i] * rhs.value - self.value * rhs.grad[i]) / denom;
        }
        Self {
            value: self.value / rhs.value,
            grad,
        }
    }

    fn add_const(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            grad: self.grad,
        }
    }

    fn sub_const(self, rhs: f64) -> Self {
        Self {
            value: self.value - rhs,
            grad: self.grad,
        }
    }

    fn const_sub(self, lhs: f64) -> Self {
        self.neg().add_const(lhs)
    }

    fn mul_const(self, rhs: f64) -> Self {
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] * rhs;
        }
        Self {
            value: self.value * rhs,
            grad,
        }
    }

    fn div_const(self, rhs: f64) -> Self {
        self.mul_const(1.0 / rhs)
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] * value;
        }
        Self { value, grad }
    }

    fn log(self) -> Self {
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] / self.value;
        }
        Self {
            value: self.value.ln(),
            grad,
        }
    }

    fn powf(self, exp: f64) -> Self {
        let value = self.value.powf(exp);
        let coeff = exp * self.value.powf(exp - 1.0);
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] * coeff;
        }
        Self { value, grad }
    }

    fn powi(self, exp: i32) -> Self {
        let value = self.value.powi(exp);
        let coeff = (exp as f64) * self.value.powi(exp - 1);
        let mut grad = [0.0; FSRS7_GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(FSRS7_GRAD_LEN) {
            *item = self.grad[i] * coeff;
        }
        Self { value, grad }
    }

    fn pow(self, exp: Self) -> Self {
        let base = self.clamp_min(1e-12);
        exp.mul(base.log()).exp()
    }

    fn clamp_min(self, min: f64) -> Self {
        if self.value < min {
            Self::constant(min)
        } else {
            self
        }
    }

    fn clamp_max(self, max: f64) -> Self {
        if self.value > max {
            Self::constant(max)
        } else {
            self
        }
    }

    fn clamp(self, min: f64, max: f64) -> Self {
        self.clamp_min(min).clamp_max(max)
    }

    fn min(self, rhs: Self) -> Self {
        if self.value <= rhs.value { self } else { rhs }
    }

    fn max(self, rhs: Self) -> Self {
        if self.value >= rhs.value { self } else { rhs }
    }
}

fn dual_weights(w: &[f32]) -> [Dual35; FSRS7_GRAD_LEN] {
    std::array::from_fn(|i| Dual35::variable(w[i] as f64, i))
}

fn fsrs7_fc_r_and_drdt_scalar(t: f64, s: f64, w: &[f32]) -> (f64, f64) {
    let s_safe = s.max(1e-12);
    let decay1 = -(w[27] as f64);
    let decay2 = -(w[28] as f64);
    let base1 = (w[29] as f64).max(1e-4);
    let base2 = (w[30] as f64).max(1e-4);
    let bw1 = (w[31] as f64).max(1e-4);
    let bw2 = (w[32] as f64).max(1e-4);
    let swp1 = w[33] as f64;
    let swp2 = w[34] as f64;

    let c1 = base1.powf(1.0 / decay1) - 1.0;
    let c2 = base2.powf(1.0 / decay2) - 1.0;
    let tos = t / s_safe;
    let inner1 = (1.0 + c1 * tos).max(1e-9);
    let inner2 = (1.0 + c2 * tos).max(1e-9);
    let r1 = inner1.powf(decay1);
    let r2 = inner2.powf(decay2);

    let wt1 = bw1 * s_safe.powf(-swp1);
    let wt2 = bw2 * s_safe.powf(swp2);
    let wt_sum = (wt1 + wt2).max(1e-9);
    let r = ((wt1 * r1 + wt2 * r2) / wt_sum).clamp(0.0, 1.0);

    let dr1_dt = decay1 * inner1.powf(decay1 - 1.0) * (c1 / s_safe);
    let dr2_dt = decay2 * inner2.powf(decay2 - 1.0) * (c2 / s_safe);
    let dr_dt = ((wt1 * dr1_dt + wt2 * dr2_dt) / wt_sum).clamp(-1e9, 0.0);
    (r, dr_dt)
}

fn fsrs7_fc_r_dual(t: Dual35, s: Dual35, w: &[Dual35; FSRS7_GRAD_LEN]) -> Dual35 {
    let decay1 = w[27].neg();
    let decay2 = w[28].neg();
    let base1 = w[29].clamp_min(1e-4);
    let base2 = w[30].clamp_min(1e-4);
    let bw1 = w[31].clamp_min(1e-4);
    let bw2 = w[32].clamp_min(1e-4);
    let swp1 = w[33];
    let swp2 = w[34];

    let c1 = base1.pow(decay1.powi(-1)).sub_const(1.0);
    let c2 = base2.pow(decay2.powi(-1)).sub_const(1.0);
    let tos = t.div(s);
    let inner1 = c1.mul(tos).add_const(1.0).clamp_min(1e-9);
    let inner2 = c2.mul(tos).add_const(1.0).clamp_min(1e-9);

    let r1 = inner1.pow(decay1);
    let r2 = inner2.pow(decay2);

    let wt1 = bw1.mul(s.pow(swp1.neg()));
    let wt2 = bw2.mul(s.pow(swp2));
    let wt_sum = wt1.add(wt2).clamp_min(1e-9);
    wt1.mul(r1).add(wt2.mul(r2)).div(wt_sum).clamp(0.0, 1.0)
}

fn fsrs7_init_d_dual(rating: f64, w: &[Dual35; FSRS7_GRAD_LEN]) -> Dual35 {
    w[4].sub(w[5].mul_const(rating - 1.0).exp())
        .add_const(1.0)
        .clamp(1.0, 10.0)
}

fn fsrs7_next_d_good_dual(d: Dual35, init_d4: Dual35) -> Dual35 {
    init_d4
        .mul_const(0.01)
        .add(d.mul_const(0.99))
        .clamp(1.0, 10.0)
}

fn fsrs7_s_fail_long_dual(s: Dual35, d: Dual35, r: Dual35, w: &[Dual35; FSRS7_GRAD_LEN]) -> Dual35 {
    let raw = w[10]
        .mul(d.pow(w[11].neg()))
        .mul(s.add_const(1.0).pow(w[12]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[13]).exp());
    s.min(raw)
}

fn fsrs7_s_fail_short_dual(
    s: Dual35,
    d: Dual35,
    r: Dual35,
    w: &[Dual35; FSRS7_GRAD_LEN],
) -> Dual35 {
    let raw = w[19]
        .mul(d.pow(w[20].neg()))
        .mul(s.add_const(1.0).pow(w[21]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[22]).exp());
    s.min(raw)
}

fn fsrs7_next_s_good_dual(
    s: Dual35,
    d: Dual35,
    delta_t: Dual35,
    w: &[Dual35; FSRS7_GRAD_LEN],
) -> Dual35 {
    let r = fsrs7_fc_r_dual(delta_t, s, w).clamp(0.0001, 0.9999);

    let sf_l = fsrs7_s_fail_long_dual(s, d, r, w);
    let si_l = w[7]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[8].neg()))
        .mul(
            r.const_sub(1.0)
                .mul(w[9])
                .clamp_max(30.0)
                .exp()
                .sub_const(1.0),
        )
        .add_const(1.0);
    let s_lng = sf_l.max(s.mul(si_l));

    let sf_sh = fsrs7_s_fail_short_dual(s, d, r, w);
    let si_sh = w[16]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[17].neg()))
        .mul(
            r.const_sub(1.0)
                .mul(w[18])
                .clamp_max(30.0)
                .exp()
                .sub_const(1.0),
        )
        .add_const(1.0);
    let s_sht = sf_sh.max(s.mul(si_sh));

    let coef = Dual35::constant(1.0)
        .sub(w[26].mul(w[25].neg().mul(delta_t).exp()))
        .clamp(0.0, 1.0);
    coef.mul(s_lng)
        .add(Dual35::constant(1.0).sub(coef).mul(s_sht))
        .clamp(S_MIN as f64, S_MAX as f64)
}

fn fsrs7_interval_differentiable_dual(
    s: Dual35,
    target: f64,
    n_newton: usize,
    w: &[f32],
    w_dual: &[Dual35; FSRS7_GRAD_LEN],
) -> Dual35 {
    let s_f = s.value.max(1e-10);
    let d1 = -(w[27] as f64);
    let d2 = -(w[28] as f64);
    let b1 = (w[29] as f64).max(1e-4);
    let b2 = (w[30] as f64).max(1e-4);
    let bw1 = (w[31] as f64).max(1e-4);
    let bw2 = (w[32] as f64).max(1e-4);
    let sw1 = w[33] as f64;
    let sw2 = w[34] as f64;

    let c1 = b1.powf(1.0 / d1) - 1.0;
    let c2 = b2.powf(1.0 / d2) - 1.0;
    let wt1 = bw1 * s_f.powf(-sw1);
    let wt2 = bw2 * s_f.powf(sw2);
    let wts = (wt1 + wt2).max(1e-9);

    let mut u = s_f.ln();
    for _ in 0..n_newton {
        u = u.clamp((FSRS7_MIN_T as f64).ln(), (FSRS7_MAX_T as f64).ln());
        let t = u.exp().clamp(FSRS7_MIN_T as f64, FSRS7_MAX_T as f64);
        let tos = t / s_f;
        let i1 = (1.0 + c1 * tos).max(1e-9);
        let i2 = (1.0 + c2 * tos).max(1e-9);
        let r = (wt1 * i1.powf(d1) + wt2 * i2.powf(d2)) / wts;
        let dr1 = d1 * i1.powf(d1 - 1.0) * c1 / s_f;
        let dr2 = d2 * i2.powf(d2 - 1.0) * c2 / s_f;
        let drdt = (wt1 * dr1 + wt2 * dr2) / wts;
        let dfdu = (drdt * t).min(-1e-12);
        u -= (r - target) / dfdu;
    }

    let t_star = u.exp().clamp(FSRS7_MIN_T as f64, FSRS7_MAX_T as f64);
    let residual = fsrs7_fc_r_dual(Dual35::constant(t_star), s, w_dual).sub_const(target);
    let (_, drdt_s) = fsrs7_fc_r_and_drdt_scalar(t_star, s.value, w);
    let dfdu_s = (drdt_s * t_star).clamp(-1e9, -1e-9);
    Dual35::constant(t_star.ln())
        .sub(residual.div_const(dfdu_s))
        .clamp((FSRS7_MIN_T as f64).ln(), (FSRS7_MAX_T as f64).ln())
        .exp()
}

fn fsrs7_interval_growth_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35; FSRS7_GRAD_LEN],
    n_reviews: usize,
    target_dr: f64,
    n_newton: usize,
) -> Dual35 {
    let mut s = w_dual[2].clamp(S_MIN as f64, S_MAX as f64);
    let init_d4 = fsrs7_init_d_dual(4.0, w_dual);
    let mut d = fsrs7_init_d_dual(3.0, w_dual);
    let mut prev_interval: Option<Dual35> = None;
    let mut best_ratio: Option<Dual35> = None;
    let mut best_val = f64::NEG_INFINITY;
    for _ in 0..n_reviews {
        let t = fsrs7_interval_differentiable_dual(s, target_dr, n_newton, w, w_dual);
        if let Some(prev) = prev_interval {
            if prev.value >= FSRS7_ONE_DAY as f64 {
                let ratio = t.div(prev);
                if ratio.value > best_val {
                    best_val = ratio.value;
                    best_ratio = Some(ratio);
                }
            }
        }
        prev_interval = Some(t);
        s = fsrs7_next_s_good_dual(s, d, t, w_dual);
        d = fsrs7_next_d_good_dual(d, init_d4);
    }
    if let Some(ratio) = best_ratio {
        ratio.powf(2.0)
    } else {
        Dual35::constant(0.0)
    }
}

fn fsrs7_short_interval_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35; FSRS7_GRAD_LEN],
    n_reviews: usize,
    n_newton: usize,
    target_drs: &[f32],
) -> Dual35 {
    let mut penalty_sum = Dual35::constant(0.0);
    let mut penalty_count = 0usize;
    for &target_dr in target_drs {
        let mut s = w_dual[2].clamp(S_MIN as f64, S_MAX as f64);
        let init_d4 = fsrs7_init_d_dual(4.0, w_dual);
        let mut d = fsrs7_init_d_dual(3.0, w_dual);
        let mut short_sum = Dual35::constant(0.0);
        let mut short_count = 0usize;
        for _ in 0..n_reviews {
            let t = fsrs7_interval_differentiable_dual(s, target_dr as f64, n_newton, w, w_dual);
            if t.value < FSRS7_ONE_DAY as f64 {
                short_sum = short_sum.add(t);
                short_count += 1;
            }
            s = fsrs7_next_s_good_dual(s, d, t, w_dual);
            d = fsrs7_next_d_good_dual(d, init_d4);
        }
        if short_count == 0 {
            continue;
        }
        let avg_t = short_sum
            .div_const(short_count as f64)
            .clamp_min(FSRS7_MIN_T as f64);
        let inv_x = avg_t.powf(-1.0);
        let penalty = inv_x
            .clamp_min(FSRS7_INV_C as f64)
            .sub_const(FSRS7_INV_C as f64);
        penalty_sum = penalty_sum.add(penalty);
        penalty_count += 1;
    }
    if penalty_count == 0 {
        Dual35::constant(0.0)
    } else {
        penalty_sum.div_const(penalty_count as f64)
    }
}

fn fsrs7_schedule_penalty_value_and_grad(
    w: &[f32],
    batch_size: usize,
) -> (f64, [f64; FSRS7_GRAD_LEN]) {
    if w.len() < FSRS7_PARAM_LEN {
        return (0.0, [0.0; FSRS7_GRAD_LEN]);
    }
    let w_dual = dual_weights(w);
    let mut p1 = fsrs7_interval_growth_penalty_dual(
        w,
        &w_dual,
        FSRS7_PENALTY_N_REVIEWS,
        FSRS7_PENALTY_TARGET_DR as f64,
        FSRS7_PENALTY_N_NEWTON,
    );
    if !p1.value.is_finite() {
        p1 = Dual35::constant(0.0);
    }
    let mut p2 = fsrs7_short_interval_penalty_dual(
        w,
        &w_dual,
        FSRS7_PENALTY_N_REVIEWS,
        FSRS7_PENALTY_N_NEWTON,
        &FSRS7_PENALTY_TARGET_DRS,
    );
    if !p2.value.is_finite() {
        p2 = Dual35::constant(0.0);
    }
    let penalty = p1
        .mul_const(FSRS7_PENALTY_W_1)
        .add(p2.mul_const(FSRS7_PENALTY_W_2))
        .mul_const(batch_size as f64);
    if !penalty.value.is_finite() {
        return (0.0, [0.0; FSRS7_GRAD_LEN]);
    }
    let mut grad = penalty.grad;
    for g in &mut grad {
        if !g.is_finite() {
            *g = 0.0;
        }
    }
    (penalty.value, grad)
}

fn l2_penalty_value_and_grad(
    w: &[f32],
    init_w: &[f32],
    batch_size: usize,
    total_size: usize,
    l2_weight: f64,
) -> (f64, Vec<f32>) {
    let mut grad = vec![0.0f32; w.len()];
    if total_size == 0 {
        return (0.0, grad);
    }
    let size = w.len().min(init_w.len()).min(PARAMS_STDDEV.len());
    let scale = l2_weight * batch_size as f64 / total_size as f64;
    let mut penalty_sum = 0.0f64;
    for i in 0..size {
        let sigma = PARAMS_STDDEV[i] as f64;
        let denom = sigma * sigma;
        let diff = w[i] as f64 - init_w[i] as f64;
        penalty_sum += diff * diff / denom;
        grad[i] = (2.0 * diff / denom * scale) as f32;
    }
    let penalty = penalty_sum * scale;
    if !penalty.is_finite() {
        return (0.0, vec![0.0; w.len()]);
    }
    for g in &mut grad {
        if !g.is_finite() {
            *g = 0.0;
        }
    }
    (penalty, grad)
}

pub struct BCELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
    pub fn forward(
        &self,
        retrievability: Tensor<B, 1>,
        labels: Tensor<B, 1>,
        weights: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss = (labels.clone() * retrievability.clone().log()
            + (-labels + 1) * (-retrievability + 1).log())
            * weights.clone();
        // info!("loss: {}", &loss);
        match mean {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => (loss.sum() / weights.sum()).neg(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
        weights: Tensor<B, 1>,
        reduce: Reduction,
    ) -> Tensor<B, 1> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let state = self.forward(t_historys, r_historys, None);
        let retrievability = self.power_forgetting_curve(delta_ts, state.stability);
        BCELoss::new().forward(retrievability, labels.float(), weights, reduce)
    }

    #[cfg(test)]
    pub(crate) fn l2_regularization(
        &self,
        init_w: Tensor<B, 1>,
        params_stddev: Tensor<B, 1>,
        batch_size: usize,
        total_size: usize,
        l2_weight: f64,
    ) -> Tensor<B, 1> {
        (self.w.val() - init_w)
            .powi_scalar(2)
            .div(params_stddev.powi_scalar(2))
            .sum()
            .mul_scalar(l2_weight * batch_size as f64 / total_size as f64)
    }

    #[cfg(test)]
    fn fsrs7_fc_r_and_drdt(
        &self,
        t: Tensor<B, 1>,
        s: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let w = self.w.val();
        let decay1 = -w.get(27);
        let decay2 = -w.get(28);
        let base1 = w.get(29).clamp_min(1e-4);
        let base2 = w.get(30).clamp_min(1e-4);
        let bw1 = w.get(31).clamp_min(1e-4);
        let bw2 = w.get(32).clamp_min(1e-4);
        let swp1 = w.get(33);
        let swp2 = w.get(34);

        let c1 = base1.powf(decay1.clone().powi_scalar(-1)) - 1.0;
        let c2 = base2.powf(decay2.clone().powi_scalar(-1)) - 1.0;
        let tos = t / s.clone();
        let inner1 = (c1.clone() * tos.clone() + 1.0).clamp_min(1e-9);
        let inner2 = (c2.clone() * tos + 1.0).clamp_min(1e-9);

        let r1 = inner1.clone().powf(decay1.clone());
        let r2 = inner2.clone().powf(decay2.clone());

        let wt1 = bw1 * s.clone().powf(-swp1);
        let wt2 = bw2 * s.clone().powf(swp2);
        let wt_sum = (wt1.clone() + wt2.clone()).clamp_min(1e-9);

        let r = ((wt1.clone() * r1 + wt2.clone() * r2) / wt_sum.clone()).clamp(0.0, 1.0);

        let dr1_dt = decay1.clone() * inner1.powf(decay1 - 1.0) * (c1 / s.clone());
        let dr2_dt = decay2.clone() * inner2.powf(decay2 - 1.0) * (c2 / s);
        let dr_dt = ((wt1 * dr1_dt + wt2 * dr2_dt) / wt_sum).clamp(-1e9, 0.0);
        (r, dr_dt)
    }

    #[cfg(test)]
    fn fsrs7_init_d(&self, rating: f32) -> Tensor<B, 1> {
        let w = self.w.val();
        (w.get(4) - (w.get(5) * (rating - 1.0)).exp() + 1.0).clamp(1.0, 10.0)
    }

    #[cfg(test)]
    fn fsrs7_next_d_good(&self, d: Tensor<B, 1>, init_d4: Tensor<B, 1>) -> Tensor<B, 1> {
        (init_d4.mul_scalar(0.01) + d.mul_scalar(0.99)).clamp(1.0, 10.0)
    }

    #[cfg(test)]
    fn fsrs7_s_fail_long(&self, s: Tensor<B, 1>, d: Tensor<B, 1>, r: Tensor<B, 1>) -> Tensor<B, 1> {
        let w = self.w.val();
        let raw = w.get(10)
            * d.powf(-w.get(11))
            * ((s.clone() + 1.0).powf(w.get(12)) - 1.0)
            * ((r.neg() + 1.0) * w.get(13)).exp();
        s.clone().mask_where(s.clone().greater(raw.clone()), raw)
    }

    #[cfg(test)]
    fn fsrs7_s_fail_short(
        &self,
        s: Tensor<B, 1>,
        d: Tensor<B, 1>,
        r: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let w = self.w.val();
        let raw = w.get(19)
            * d.powf(-w.get(20))
            * ((s.clone() + 1.0).powf(w.get(21)) - 1.0)
            * ((r.neg() + 1.0) * w.get(22)).exp();
        s.clone().mask_where(s.clone().greater(raw.clone()), raw)
    }

    #[cfg(test)]
    fn fsrs7_next_s_good(
        &self,
        s: Tensor<B, 1>,
        d: Tensor<B, 1>,
        delta_t: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let w = self.w.val();
        let r = self
            .fsrs7_fc_r_and_drdt(delta_t.clone(), s.clone())
            .0
            .clamp(0.0001, 0.9999);

        let sf_l = self.fsrs7_s_fail_long(s.clone(), d.clone(), r.clone());
        let si_l = (w.get(7) - 1.5).exp()
            * (d.clone().neg() + 11.0)
            * s.clone().powf(-w.get(8))
            * (((r.clone().neg() + 1.0) * w.get(9)).clamp(-1e9, 30.0).exp() - 1.0)
            + 1.0;
        let s_lng = sf_l.clone().mask_where(
            sf_l.clone().lower((s.clone() * si_l.clone()).clone()),
            s.clone() * si_l,
        );

        let sf_sh = self.fsrs7_s_fail_short(s.clone(), d.clone(), r.clone());
        let si_sh = (w.get(16) - 1.5).exp()
            * (d.clone().neg() + 11.0)
            * s.clone().powf(-w.get(17))
            * (((r.neg() + 1.0) * w.get(18)).clamp(-1e9, 30.0).exp() - 1.0)
            + 1.0;
        let s_sht = sf_sh.clone().mask_where(
            sf_sh.clone().lower((s.clone() * si_sh.clone()).clone()),
            s.clone() * si_sh,
        );

        let coef = ((w.get(26) * (-w.get(25) * delta_t).exp()).neg() + 1.0).clamp(0.0, 1.0);
        (coef.clone() * s_lng + (coef.neg() + 1.0) * s_sht).clamp(S_MIN, S_MAX)
    }

    #[cfg(test)]
    fn fsrs7_interval_differentiable(
        &self,
        s: Tensor<B, 1>,
        target: f32,
        n_newton: usize,
        w_vec: &[f32],
    ) -> Tensor<B, 1> {
        let s_f = s.clone().into_scalar().to_f32() as f64;
        let d1 = -w_vec[27] as f64;
        let d2 = -w_vec[28] as f64;
        let b1 = (w_vec[29].max(1e-4)) as f64;
        let b2 = (w_vec[30].max(1e-4)) as f64;
        let bw1 = (w_vec[31].max(1e-4)) as f64;
        let bw2 = (w_vec[32].max(1e-4)) as f64;
        let sw1 = w_vec[33] as f64;
        let sw2 = w_vec[34] as f64;

        let c1 = b1.powf(1.0 / d1) - 1.0;
        let c2 = b2.powf(1.0 / d2) - 1.0;
        let wt1 = bw1 * s_f.powf(-sw1);
        let wt2 = bw2 * s_f.powf(sw2);
        let wts = (wt1 + wt2).max(1e-9);

        let mut u = s_f.max(1e-10).ln();
        for _ in 0..n_newton {
            u = u.clamp((FSRS7_MIN_T as f64).ln(), (FSRS7_MAX_T as f64).ln());
            let t = u.exp().clamp(FSRS7_MIN_T as f64, FSRS7_MAX_T as f64);
            let tos = t / s_f.max(1e-10);
            let i1 = (1.0 + c1 * tos).max(1e-9);
            let i2 = (1.0 + c2 * tos).max(1e-9);
            let r = (wt1 * i1.powf(d1) + wt2 * i2.powf(d2)) / wts;
            let dr1 = d1 * i1.powf(d1 - 1.0) * c1 / s_f.max(1e-10);
            let dr2 = d2 * i2.powf(d2 - 1.0) * c2 / s_f.max(1e-10);
            let drdt = (wt1 * dr1 + wt2 * dr2) / wts;
            let dfdu = (drdt * t).min(-1e-12);
            u -= (r - target as f64) / dfdu;
        }

        let t_star = u.exp().clamp(FSRS7_MIN_T as f64, FSRS7_MAX_T as f64) as f32;
        let device = self.w.val().device();
        let t_d = Tensor::from_floats([t_star], &device).detach();
        let (r_s, drdt_s) = self.fsrs7_fc_r_and_drdt(t_d.clone(), s);
        let residual = r_s - target;
        let dfdu_s = (drdt_s * t_d.clone()).detach().clamp(-1e9, -1e-9);
        let u_lifted = t_d.log() - residual / dfdu_s;
        u_lifted.clamp(FSRS7_MIN_T.ln(), FSRS7_MAX_T.ln()).exp()
    }

    #[cfg(test)]
    fn fsrs7_interval_growth_penalty(
        &self,
        n_reviews: usize,
        target_dr: f32,
        n_newton: usize,
        w_vec: &[f32],
    ) -> Tensor<B, 1> {
        let mut s = self.w.val().get(2).clamp(S_MIN, S_MAX);
        let init_d4 = self.fsrs7_init_d(4.0);
        let mut d = self.fsrs7_init_d(3.0);
        let mut prev_interval: Option<Tensor<B, 1>> = None;
        let mut best_ratio: Option<Tensor<B, 1>> = None;
        let mut best_val = f32::NEG_INFINITY;
        for _ in 0..n_reviews {
            let t = self.fsrs7_interval_differentiable(s.clone(), target_dr, n_newton, w_vec);
            if let Some(prev) = &prev_interval {
                let prev_val = prev.clone().detach().into_scalar().to_f32();
                if prev_val >= FSRS7_ONE_DAY {
                    let ratio = t.clone() / prev.clone();
                    let ratio_val = ratio.clone().detach().into_scalar().to_f32();
                    if ratio_val > best_val {
                        best_val = ratio_val;
                        best_ratio = Some(ratio);
                    }
                }
            }
            prev_interval = Some(t.clone());
            s = self.fsrs7_next_s_good(s, d.clone(), t);
            d = self.fsrs7_next_d_good(d, init_d4.clone());
        }
        if let Some(ratio) = best_ratio {
            ratio.powi_scalar(2)
        } else {
            Tensor::zeros([1], &self.w.device())
        }
    }

    #[cfg(test)]
    fn fsrs7_short_interval_penalty(
        &self,
        n_reviews: usize,
        n_newton: usize,
        target_drs: &[f32],
        w_vec: &[f32],
    ) -> Tensor<B, 1> {
        let device = self.w.val().device();
        let mut penalties: Vec<Tensor<B, 1>> = Vec::with_capacity(target_drs.len());

        for &target_dr in target_drs {
            let mut s = self.w.val().get(2).clamp(S_MIN, S_MAX);
            let init_d4 = self.fsrs7_init_d(4.0);
            let mut d = self.fsrs7_init_d(3.0);
            let mut short_sum: Option<Tensor<B, 1>> = None;
            let mut short_count = 0usize;

            for _ in 0..n_reviews {
                let t = self.fsrs7_interval_differentiable(s.clone(), target_dr, n_newton, w_vec);
                if t.clone().detach().into_scalar().to_f32() < FSRS7_ONE_DAY {
                    short_sum = Some(match short_sum {
                        Some(current) => current + t.clone(),
                        None => t.clone(),
                    });
                    short_count += 1;
                }
                s = self.fsrs7_next_s_good(s, d.clone(), t);
                d = self.fsrs7_next_d_good(d, init_d4.clone());
            }

            if short_count == 0 {
                continue;
            }

            let avg_t = short_sum
                .expect("short_sum must exist when short_count > 0")
                .div_scalar(short_count as f64)
                .clamp_min(FSRS7_MIN_T);
            let inv_x = avg_t.powi_scalar(-1);
            penalties.push(inv_x.clamp_min(FSRS7_INV_C) - FSRS7_INV_C);
        }

        if penalties.is_empty() {
            Tensor::zeros([1], &device)
        } else {
            Tensor::cat(penalties, 0).mean()
        }
    }

    #[cfg(test)]
    pub(crate) fn fsrs7_schedule_penalty(&self, batch_size: usize) -> Tensor<B, 1> {
        let device = self.w.val().device();
        if self.w.val().dims()[0] < FSRS7_PARAM_LEN {
            return Tensor::zeros([1], &device);
        }
        let w_vec = self.w.val().to_data().to_vec::<f32>().unwrap();
        let p1 = self.fsrs7_interval_growth_penalty(
            FSRS7_PENALTY_N_REVIEWS,
            FSRS7_PENALTY_TARGET_DR,
            FSRS7_PENALTY_N_NEWTON,
            &w_vec,
        );
        let p1 = if p1.clone().detach().into_scalar().to_f32().is_finite() {
            p1
        } else {
            Tensor::zeros([1], &device)
        };
        let p2 = self.fsrs7_short_interval_penalty(
            FSRS7_PENALTY_N_REVIEWS,
            FSRS7_PENALTY_N_NEWTON,
            &FSRS7_PENALTY_TARGET_DRS,
            &w_vec,
        );
        let p2 = if p2.clone().detach().into_scalar().to_f32().is_finite() {
            p2
        } else {
            Tensor::zeros([1], &device)
        };
        (p1.mul_scalar(FSRS7_PENALTY_W_1) + p2.mul_scalar(FSRS7_PENALTY_W_2))
            .mul_scalar(batch_size as f64)
    }
}

impl<B: AutodiffBackend> Model<B> {
    fn add_manual_weight_gradient(
        &self,
        mut gradients: B::Gradients,
        manual_grad: &[f32],
    ) -> B::Gradients {
        let grad_tensor = self.w.grad(&gradients).unwrap();
        let device = grad_tensor.device();
        let grad_len = grad_tensor.dims()[0];
        let mut data = vec![0.0f32; grad_len];
        for (dst, src) in data.iter_mut().zip(manual_grad.iter()) {
            *dst = *src;
        }
        let manual_tensor = Tensor::from_floats(data.as_slice(), &device);
        let updated_grad = grad_tensor + manual_tensor;
        self.w.grad_remove(&mut gradients);
        self.w.grad_replace(&mut gradients, updated_grad);
        gradients
    }

    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4], &device));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }

    fn freeze_short_term_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = if grad_tensor.dims()[0] >= 35 {
            grad_tensor.slice_assign([16..25], Tensor::zeros([9], &device))
        } else {
            grad_tensor.slice_assign([17..20], Tensor::zeros([3], &device))
        };

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }
}

#[derive(Debug, Default, Clone)]
pub struct ProgressState {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
}

#[derive(Debug, Default)]
pub struct CombinedProgressState {
    pub want_abort: bool,
    pub splits: Vec<ProgressState>,
    finished: bool,
}

impl CombinedProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub fn current(&self) -> usize {
        self.splits.iter().map(|s| s.current()).sum()
    }

    pub fn total(&self) -> usize {
        self.splits.iter().map(|s| s.total()).sum()
    }

    pub const fn finished(&self) -> bool {
        self.finished
    }
}

#[derive(Clone)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<CombinedProgressState>>,
    pub interrupter: TrainingInterrupter,
    /// The index of the split we should update.
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self {
            state,
            interrupter: Default::default(),
            index,
        }
    }
}

impl ProgressState {
    pub const fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub const fn total(&self) -> usize {
        self.epoch_total * self.items_total
    }
}

impl MetricsRenderer for ProgressCollector {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = item.epoch;
        split.epoch_total = item.epoch_total;
        split.items_processed = item.progress.items_processed;
        split.items_total = item.progress.items_total;
        if info.want_abort {
            self.interrupter.stop();
        }
    }

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 8)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 2023)]
    pub seed: u64,
    #[config(default = 2e-2)]
    pub learning_rate: f64,
    #[config(default = 1024)]
    pub max_seq_len: usize,
}

pub(crate) fn calculate_average_recall(items: &[FSRSItem]) -> f32 {
    let (total_recall, total_reviews) = items
        .iter()
        .map(|item| item.current())
        .fold((0u32, 0u32), |(sum, count), review| {
            (sum + (review.rating > 1) as u32, count + 1)
        });

    if total_reviews == 0 {
        return 0.0;
    }
    total_recall as f32 / total_reviews as f32
}

/// Input parameters for computing FSRS parameters
#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    /// The training set containing review history
    pub train_set: Vec<FSRSItem>,
    /// Optional progress tracking
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    /// Whether to enable short-term memory parameters
    pub enable_short_term: bool,
    /// Number of relearning steps
    pub num_relearning_steps: Option<usize>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        }
    }
}
/// Computes optimized parameters for the FSRS model based on training data.
///
/// This function trains the model on the provided dataset and returns optimized parameters.
///
/// # Arguments
/// * `input` - Input parameters including the training dataset and configuration
///
/// # Returns
/// A `Result<Vec<f32>>` containing the optimized parameters
pub fn compute_parameters(
    ComputeParametersInput {
        train_set,
        progress,
        enable_short_term,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Result<Vec<f32>> {
    let finish_progress = || {
        if let Some(progress) = &progress {
            // The progress state at completion time may not indicate completion, because:
            // - If there were fewer than 512 entries, render_train() will have never been called
            // - One or more of the splits may have ignored later epochs, if accuracy went backwards
            // Because of this, we need a separate finished flag.
            progress.lock().unwrap().finished = true;
        }
    };

    let (dataset_for_initialization, train_set) = prepare_training_data(train_set);
    let average_recall = calculate_average_recall(&train_set);
    if train_set.len() < 8 {
        finish_progress();
        return Ok(DEFAULT_PARAMETERS.to_vec());
    }

    let (initial_stability, initial_forgetting_curve, initial_rating_count) =
        initialize_parameters(dataset_for_initialization.clone(), average_recall).inspect_err(
            |_e| {
                finish_progress();
            },
        )?;
    let mut initialized_parameters = DEFAULT_PARAMETERS.to_vec();
    initialized_parameters[0..4].copy_from_slice(&initial_stability);
    initialized_parameters[27..35].copy_from_slice(&initial_forgetting_curve);
    if train_set.len() == dataset_for_initialization.len() || train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: Some(initial_stability),
            initial_forgetting_curve: Some(initial_forgetting_curve),
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.85)
            .with_epsilon(1e-8),
    );
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);

    if let Some(progress) = &progress {
        let progress_state = ProgressState {
            epoch_total: config.num_epochs,
            items_total: weighted_train_set.len(),
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().splits = vec![progress_state];
    }
    let model = train::<Autodiff<B>>(
        weighted_train_set.clone(),
        weighted_train_set,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    );

    let optimized_parameters = model
        .inspect_err(|_e| {
            finish_progress();
        })?
        .w
        .val()
        .to_data()
        .to_vec()
        .unwrap();

    finish_progress();

    if optimized_parameters
        .iter()
        .any(|parameter: &f32| parameter.is_infinite())
    {
        return Err(FSRSError::InvalidInput);
    }

    let mut optimized_initial_stability = optimized_parameters[0..4]
        .iter()
        .enumerate()
        .map(|(i, &val)| (i as u32 + 1, val))
        .collect();
    let clamped_stability =
        smooth_and_fill(&mut optimized_initial_stability, &initial_rating_count).unwrap();
    let optimized_parameters = clamped_stability
        .into_iter()
        .chain(optimized_parameters[4..].iter().copied())
        .collect();

    Ok(optimized_parameters)
}

pub fn benchmark(
    ComputeParametersInput {
        train_set,
        enable_short_term,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let average_recall = calculate_average_recall(&train_set);
    let (dataset_for_initialization, _next_train_set) = train_set
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    let (initial_stability, initial_forgetting_curve, _rating_count) =
        initialize_parameters(dataset_for_initialization, average_recall).unwrap();
    let mut config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: Some(initial_stability),
            initial_forgetting_curve: Some(initial_forgetting_curve),
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.85)
            .with_epsilon(1e-8),
    );
    // save RAM and speed up training
    config.max_seq_len = 64;
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    let model = train::<Autodiff<B>>(
        weighted_train_set.clone(),
        weighted_train_set,
        &config,
        None,
    );
    let parameters: Vec<f32> = model.unwrap().w.val().to_data().to_vec::<f32>().unwrap();
    parameters
}

fn train<B: AutodiffBackend>(
    train_set: Vec<WeightedFSRSItem>,
    test_set: Vec<WeightedFSRSItem>,
    config: &TrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    // Training data
    let total_size = train_set.len();
    let iterations = (total_size / config.batch_size + 1) * config.num_epochs;
    let batch_dataset =
        BatchTensorDataset::<B>::new(FSRSDataset::from(train_set), config.batch_size);
    let dataloader_train = ShuffleDataLoader::new(batch_dataset, config.seed);

    let batch_dataset = BatchTensorDataset::<B::InnerBackend>::new(
        FSRSDataset::from(test_set.clone()),
        config.batch_size,
    );
    let dataloader_valid = ShuffleDataLoader::new(batch_dataset, config.seed);

    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let interrupter = TrainingInterrupter::new();
    let mut renderer: Box<dyn MetricsRenderer> = match progress {
        Some(mut progress) => {
            progress.interrupter = interrupter.clone();
            Box::new(progress)
        }
        None => Box::new(NoProgress {}),
    };

    let mut model: Model<B> = config.model.init();
    let init_w = model.w.val();
    let init_w_vec = init_w.to_data().to_vec::<f32>().unwrap();
    let mut optim = config.optimizer.init::<B, Model<B>>();

    let mut best_loss = f64::INFINITY;
    let mut best_model = model.clone();
    for epoch in 1..=config.num_epochs {
        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;
        while let Some(item) = iterator.next() {
            iteration += 1;
            let real_batch_size = item.delta_ts.shape().dims[0];
            let lr = LrScheduler::step(&mut lr_scheduler);
            let progress = iterator.progress();
            let l2_weight = FSRS7_PENALTY_W_L2;
            let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
            let (_l2_penalty_value, mut manual_grad) = l2_penalty_value_and_grad(
                &w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
            );
            let (_schedule_value, schedule_grad) =
                fsrs7_schedule_penalty_value_and_grad(&w_vec, real_batch_size);
            let inv_total = 1.0 / total_size as f64;
            for i in 0..manual_grad.len().min(schedule_grad.len()) {
                manual_grad[i] += (schedule_grad[i] * inv_total) as f32;
            }
            let loss = model.forward_classification(
                item.t_historys,
                item.r_historys,
                item.delta_ts,
                item.labels,
                item.weights,
                Reduction::Sum,
            );
            let mut gradients = loss.backward();
            gradients = model.add_manual_weight_gradient(gradients, &manual_grad);
            if config.model.freeze_initial_stability {
                gradients = model.freeze_initial_stability(gradients);
            }
            if config.model.freeze_short_term_stability {
                gradients = model.freeze_short_term_stability(gradients);
            }
            let grads = GradientsParams::from_grads(gradients, &model);
            model = optim.step(lr, model, grads);
            model.w = parameter_clipper(
                model.w,
                config.model.num_relearning_steps,
                !config.model.freeze_short_term_stability,
            );
            // info!("epoch: {:?} iteration: {:?} lr: {:?}", epoch, iteration, lr);
            renderer.render_train(TrainingProgress {
                progress,
                epoch,
                epoch_total: config.num_epochs,
                iteration,
            });

            if interrupter.should_stop() {
                break;
            }
        }

        if interrupter.should_stop() {
            break;
        }

        let model_valid = model.valid();
        let mut loss_valid = 0.0;
        for batch in dataloader_valid.iter() {
            let real_batch_size = batch.delta_ts.shape().dims[0];
            let l2_weight = FSRS7_PENALTY_W_L2;
            let w_vec = model_valid.w.val().to_data().to_vec::<f32>().unwrap();
            let (l2_penalty_value, _) = l2_penalty_value_and_grad(
                &w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
            );
            let (schedule_value, _) =
                fsrs7_schedule_penalty_value_and_grad(&w_vec, real_batch_size);
            let schedule_penalty = schedule_value / total_size as f64;
            let loss = model_valid.forward_classification(
                batch.t_historys,
                batch.r_historys,
                batch.delta_ts,
                batch.labels,
                batch.weights,
                Reduction::Sum,
            );
            let loss = loss.into_scalar().to_f64();
            loss_valid += loss + l2_penalty_value + schedule_penalty;

            if interrupter.should_stop() {
                break;
            }
        }
        loss_valid /= test_set.len() as f64;
        info!("epoch: {:?} loss: {:?}", epoch, loss_valid);
        if loss_valid < best_loss {
            best_loss = loss_valid;
            best_model = model.clone();
        }
    }

    info!("best_loss: {:?}", best_loss);

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    Ok(best_model)
}

struct NoProgress {}

impl MetricsRenderer for NoProgress {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, _item: TrainingProgress) {}

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use crate::convertor_tests::data_from_csv;
    use crate::dataset::FSRSBatch;
    use crate::model::FSRS;
    use crate::test_helpers::TestHelper;
    use burn::backend::NdArray;
    use log::LevelFilter;

    #[test]
    fn test_calculate_average_recall() {
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        assert_eq!(average_recall, 0.9435269);
    }

    #[test]
    fn test_loss_and_grad() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::TensorData;

        let config = ModelConfig::default();
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let mut model: Model<B> = config.init();
        let init_w = model.w.val();
        let params_stddev = Tensor::from_floats(PARAMS_STDDEV, &device);

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
                TensorData::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ),
            r_historys: Tensor::from_floats(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );

        assert_eq!(loss.clone().into_scalar().to_f32(), 4.0466027);
        let gradients = loss.backward();

        let w_grad = model.w.grad(&gradients).unwrap();

        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            -0.095688485,
            -0.0051607806,
            -0.0012249565,
            0.007462064,
            0.03650761,
            -0.082112335,
            0.0593964,
            -2.1474836,
            0.57626534,
            -2.8751316,
            0.7154875,
            -0.028993709,
            0.0099172965,
            -0.2189217,
            -0.0017800558,
            -0.089381434,
            0.299141,
            0.068104014,
            -0.011605468,
            -0.25398168,
            0.27700496,
        ]);

        let config =
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new().with_epsilon(1e-8));
        let mut optim = config.optimizer.init::<B, Model<B>>();
        let lr = 0.04;
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(
            model.w,
            config.model.num_relearning_steps,
            !config.model.freeze_short_term_stability,
        );
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.252,
                1.3331,
                2.3464994,
                8.2556,
                6.3733,
                0.87340003,
                2.9794,
                0.040999997,
                1.8322,
                0.20660001,
                0.756,
                1.5235,
                0.021400042,
                0.3029,
                1.6882998,
                0.64140004,
                1.8329,
                0.5025,
                0.13119997,
                0.1058,
                0.1142,
            ]);

        let penalty =
            model.l2_regularization(init_w.clone(), params_stddev.clone(), 512, 1000, 2.0);
        assert_eq!(penalty.clone().into_scalar().to_f32(), 0.67711145);

        let gradients = penalty.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            0.0019813816,
            0.00087788026,
            0.00026506148,
            -0.000105618295,
            -0.25213888,
            1.0448985,
            -0.22755535,
            5.688889,
            -0.5385926,
            2.5283954,
            -0.75225013,
            0.9102214,
            -10.113569,
            3.1999993,
            0.2521374,
            1.3107208,
            -0.07721739,
            -0.85244584,
            0.79999936,
            4.1795917,
            -1.1237311,
        ]);

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
                TensorData::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ),
            r_historys: Tensor::from_floats(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );
        assert_eq!(loss.clone().into_scalar().to_f32(), 3.767796);
        let gradients = loss.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                -0.040530164,
                -0.0041278866,
                -0.0010157757,
                0.007239434,
                0.009321215,
                -0.120117955,
                0.039143264,
                -0.8628009,
                0.5794302,
                -2.5713828,
                0.7669307,
                -0.024242667,
                0.0,
                -0.16912507,
                -0.0017008218,
                -0.061857328,
                0.28093633,
                0.064058185,
                0.0063592787,
                -0.1903223,
                0.6257775,
            ]);
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(
            model.w,
            config.model.num_relearning_steps,
            !config.model.freeze_short_term_stability,
        );
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.2882918, 1.3726242, 2.3861322, 8.215636, 6.339965, 0.9130969, 2.940639,
                0.07696985, 1.7921946, 0.2464217, 0.71595186, 1.5631561, 0.001, 0.34230903,
                1.7282416, 0.68038, 1.7929853, 0.46258268, 0.14039303, 0.14509967, 0.1,
            ]);
    }

    #[test]
    fn test_training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }

        let artifact_dir = std::env::var("BURN_LOG");

        if let Ok(artifact_dir) = artifact_dir {
            let _ = create_dir_all(&artifact_dir);
            let log_file = Path::new(&artifact_dir).join("training.log");
            fern::Dispatch::new()
                .format(|out, message, record| {
                    out.finish(format_args!(
                        "[{}][{}] {}",
                        record.target(),
                        record.level(),
                        message
                    ))
                })
                .level(LevelFilter::Info)
                .chain(fern::log_file(log_file).unwrap())
                .apply()
                .unwrap();
        }
        for items in [anki21_sample_file_converted_to_fsrs(), data_from_csv()] {
            for enable_short_term in [true, false] {
                let progress = CombinedProgressState::new_shared();
                let progress2 = Some(progress.clone());
                thread::spawn(move || {
                    let mut finished = false;
                    while !finished {
                        thread::sleep(Duration::from_millis(500));
                        let guard = progress.lock().unwrap();
                        finished = guard.finished();
                        println!("progress: {}/{}", guard.current(), guard.total());
                    }
                });

                let parameters = compute_parameters(ComputeParametersInput {
                    train_set: items.clone(),
                    progress: progress2,
                    enable_short_term,
                    num_relearning_steps: None,
                })
                .unwrap();
                dbg!(&parameters);

                // evaluate
                let model = FSRS::new(&parameters).unwrap();
                let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                dbg!(&metrics);
            }
        }
    }

    #[test]
    fn test_fsrs7_schedule_penalty_is_finite() {
        use burn::backend::ndarray::NdArrayDevice;

        let config = ModelConfig::default();
        let device = NdArrayDevice::Cpu;
        let model: Model<NdArray<f32>> = config.init();
        let penalty = model
            .fsrs7_schedule_penalty(512)
            .to_device(&device)
            .into_scalar()
            .to_f32();
        assert!(penalty.is_finite());
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_fsrs7_schedule_penalty_has_finite_gradients() {
        type B = Autodiff<NdArray<f32>>;
        let config = ModelConfig::default();
        let model: Model<B> = config.init();
        let penalty = model.fsrs7_schedule_penalty(512);
        let value = penalty.clone().into_scalar().to_f32();
        assert!(value.is_finite());

        let gradients = penalty.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        let grads = w_grad.to_data().to_vec::<f32>().unwrap();
        assert!(grads.iter().all(|v| v.is_finite()));
        assert!(grads.iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn test_manual_l2_penalty_matches_autodiff_gradient() {
        type B = Autodiff<NdArray<f32>>;
        let config = ModelConfig::default();
        let model: Model<B> = config.init();
        let device = model.w.device();
        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
        let mut init_w_vec = w_vec.clone();
        for (i, init) in init_w_vec.iter_mut().enumerate() {
            *init -= 0.05 * ((i + 1) as f32) / (FSRS7_GRAD_LEN as f32);
        }

        let init_w = Tensor::from_floats(init_w_vec.as_slice(), &device);
        let params_stddev = Tensor::from_floats(PARAMS_STDDEV, &device);
        let penalty = model.l2_regularization(init_w, params_stddev, 512, 1000, FSRS7_PENALTY_W_L2);
        let expected_value = penalty.clone().into_scalar().to_f64();
        let gradients = penalty.backward();
        let expected_grad = model
            .w
            .grad(&gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let (actual_value, actual_grad) =
            l2_penalty_value_and_grad(&w_vec, &init_w_vec, 512, 1000, FSRS7_PENALTY_W_L2);
        assert!(
            (actual_value - expected_value).abs() < 1e-6,
            "l2 value mismatch actual={} expected={}",
            actual_value,
            expected_value
        );
        for (expected, actual) in expected_grad.iter().zip(actual_grad.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_manual_schedule_penalty_matches_autodiff_gradient() {
        type B = Autodiff<NdArray<f32>>;
        let config = ModelConfig::default();
        let model: Model<B> = config.init();
        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();

        let (actual_value, actual_grad) = fsrs7_schedule_penalty_value_and_grad(&w_vec, 512);
        let penalty = model.fsrs7_schedule_penalty(512);
        let expected_value = penalty.clone().into_scalar().to_f64();
        assert!(
            (actual_value - expected_value).abs() < 5e-3,
            "schedule value mismatch actual={} expected={}",
            actual_value,
            expected_value
        );

        let gradients = penalty.backward();
        let expected_grad = model
            .w
            .grad(&gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let mut max_abs_diff = 0.0f32;
        let mut max_index = 0usize;
        let mut expected_norm2 = 0.0f64;
        let mut diff_norm2 = 0.0f64;
        for (idx, (actual, expected)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
            let diff = ((*actual as f32) - *expected).abs();
            expected_norm2 += (*expected as f64) * (*expected as f64);
            diff_norm2 += (diff as f64) * (diff as f64);
            if diff > max_abs_diff {
                max_abs_diff = diff;
                max_index = idx;
            }
        }
        let relative_l2 = (diff_norm2.sqrt() / expected_norm2.sqrt().max(1e-12)) as f32;
        assert!(
            max_abs_diff < 0.2 && relative_l2 < 2e-3,
            "schedule grad mismatch max_abs_diff={} at index={} actual={} expected={} relative_l2={}",
            max_abs_diff,
            max_index,
            actual_grad[max_index],
            expected_grad[max_index],
            relative_l2
        );
    }

    #[test]
    fn test_manual_penalty_gradient_matches_autodiff_combined_objective() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::TensorData;

        type B = Autodiff<NdArray<f32>>;
        let config = ModelConfig::default();
        let model: Model<B> = config.init();
        let device = NdArrayDevice::Cpu;
        let total_size = 1000usize;
        let batch_size = 512usize;

        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
        let mut init_w_vec = w_vec.clone();
        for (i, init) in init_w_vec.iter_mut().enumerate() {
            *init -= 0.05 * ((i + 1) as f32) / (FSRS7_GRAD_LEN as f32);
        }

        let build_batch = || FSRSBatch {
            t_historys: Tensor::from_floats(
                TensorData::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ),
            r_historys: Tensor::from_floats(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        // Reference: full autograd path (BCE + L2 + schedule penalty).
        let batch = build_batch();
        let loss_ref = model.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
            batch.weights,
            Reduction::Sum,
        );
        let init_w_tensor = Tensor::from_floats(init_w_vec.as_slice(), &device);
        let params_stddev = Tensor::from_floats(PARAMS_STDDEV, &device);
        let l2_ref = model.l2_regularization(
            init_w_tensor,
            params_stddev,
            batch_size,
            total_size,
            FSRS7_PENALTY_W_L2,
        );
        let schedule_ref = model
            .fsrs7_schedule_penalty(batch_size)
            .div_scalar(total_size as f64);
        let grad_ref = (loss_ref + l2_ref + schedule_ref).backward();
        let expected_grad = model
            .w
            .grad(&grad_ref)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        // New path: BCE autograd + manual penalty gradient injection.
        let batch = build_batch();
        let loss_new = model.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
            batch.weights,
            Reduction::Sum,
        );
        let mut grad_new = loss_new.backward();
        let mut manual_grad = vec![0.0f32; w_vec.len()];
        let (_l2_value, l2_grad) = l2_penalty_value_and_grad(
            &w_vec,
            &init_w_vec,
            batch_size,
            total_size,
            FSRS7_PENALTY_W_L2,
        );
        for (g, l2) in manual_grad.iter_mut().zip(l2_grad.iter()) {
            *g += *l2;
        }
        let (_schedule_value, schedule_grad) =
            fsrs7_schedule_penalty_value_and_grad(&w_vec, batch_size);
        let inv_total = 1.0 / total_size as f64;
        for i in 0..manual_grad.len().min(schedule_grad.len()) {
            manual_grad[i] += (schedule_grad[i] * inv_total) as f32;
        }
        grad_new = model.add_manual_weight_gradient(grad_new, &manual_grad);
        let actual_grad = model
            .w
            .grad(&grad_new)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let mut expected_norm2 = 0.0f64;
        let mut diff_norm2 = 0.0f64;
        let mut max_abs_diff = 0.0f32;
        let mut max_index = 0usize;
        for (idx, (expected, actual)) in expected_grad.iter().zip(actual_grad.iter()).enumerate() {
            let diff = (actual - expected).abs();
            expected_norm2 += (*expected as f64) * (*expected as f64);
            diff_norm2 += (diff as f64) * (diff as f64);
            if diff > max_abs_diff {
                max_abs_diff = diff;
                max_index = idx;
            }
        }
        let relative_l2 = (diff_norm2.sqrt() / expected_norm2.sqrt().max(1e-12)) as f32;
        assert!(
            max_abs_diff < 0.2 && relative_l2 < 2e-3,
            "combined grad mismatch max_abs_diff={} at index={} expected={} actual={} relative_l2={}",
            max_abs_diff,
            max_index,
            expected_grad[max_index],
            actual_grad[max_index],
            relative_l2
        );
    }

    fn fsrs7_interval_growth_penalty_reference(
        model: &Model<NdArray<f32>>,
        n_reviews: usize,
        target_dr: f32,
        n_newton: usize,
        w_vec: &[f32],
    ) -> Tensor<NdArray<f32>, 1> {
        let mut s = model.w.val().get(2).clamp(S_MIN, S_MAX);
        let init_d4 = model.fsrs7_init_d(4.0);
        let mut d = model.fsrs7_init_d(3.0);
        let mut intervals: Vec<Tensor<NdArray<f32>, 1>> = Vec::with_capacity(n_reviews);
        for _ in 0..n_reviews {
            let t = model.fsrs7_interval_differentiable(s.clone(), target_dr, n_newton, w_vec);
            intervals.push(t.clone());
            s = model.fsrs7_next_s_good(s, d.clone(), t);
            d = model.fsrs7_next_d_good(d, init_d4.clone());
        }

        let device = model.w.device();
        let mut best_ratio: Option<Tensor<NdArray<f32>, 1>> = None;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..intervals.len().saturating_sub(1) {
            let prev = intervals[i].clone().detach().into_scalar().to_f32();
            if prev < FSRS7_ONE_DAY {
                continue;
            }
            let ratio = intervals[i + 1].clone() / intervals[i].clone();
            let ratio_val = ratio.clone().detach().into_scalar().to_f32();
            if ratio_val > best_val {
                best_val = ratio_val;
                best_ratio = Some(ratio);
            }
        }
        if let Some(ratio) = best_ratio {
            ratio.powi_scalar(2)
        } else {
            Tensor::zeros([1], &device)
        }
    }

    fn fsrs7_short_interval_penalty_reference(
        model: &Model<NdArray<f32>>,
        n_reviews: usize,
        n_newton: usize,
        target_drs: &[f32],
        w_vec: &[f32],
    ) -> Tensor<NdArray<f32>, 1> {
        let device = model.w.val().device();
        let mut penalties: Vec<Tensor<NdArray<f32>, 1>> = Vec::with_capacity(target_drs.len());

        for &target_dr in target_drs {
            let mut s = model.w.val().get(2).clamp(S_MIN, S_MAX);
            let init_d4 = model.fsrs7_init_d(4.0);
            let mut d = model.fsrs7_init_d(3.0);
            let mut intervals: Vec<Tensor<NdArray<f32>, 1>> = Vec::with_capacity(n_reviews);

            for _ in 0..n_reviews {
                let t = model.fsrs7_interval_differentiable(s.clone(), target_dr, n_newton, w_vec);
                intervals.push(t.clone());
                s = model.fsrs7_next_s_good(s, d.clone(), t);
                d = model.fsrs7_next_d_good(d, init_d4.clone());
            }

            let mut sum: Option<Tensor<NdArray<f32>, 1>> = None;
            let mut count = 0usize;
            for interval in intervals {
                let value = interval.clone().detach().into_scalar().to_f32();
                if value >= FSRS7_ONE_DAY {
                    continue;
                }
                sum = Some(match sum {
                    Some(current) => current + interval,
                    None => interval,
                });
                count += 1;
            }

            if count == 0 {
                continue;
            }

            let avg_t = sum.unwrap().div_scalar(count as f64).clamp_min(FSRS7_MIN_T);
            let inv_x = avg_t.powi_scalar(-1);
            penalties.push(inv_x.clamp_min(FSRS7_INV_C) - FSRS7_INV_C);
        }

        if penalties.is_empty() {
            Tensor::zeros([1], &device)
        } else {
            Tensor::cat(penalties, 0).mean()
        }
    }

    #[test]
    fn test_fsrs7_schedule_penalty_matches_reference_rollout() {
        let config = ModelConfig::default();
        let model: Model<NdArray<f32>> = config.init();
        let batch_size = 512usize;
        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();

        let p1 = fsrs7_interval_growth_penalty_reference(
            &model,
            FSRS7_PENALTY_N_REVIEWS,
            FSRS7_PENALTY_TARGET_DR,
            FSRS7_PENALTY_N_NEWTON,
            &w_vec,
        );
        let p2 = fsrs7_short_interval_penalty_reference(
            &model,
            FSRS7_PENALTY_N_REVIEWS,
            FSRS7_PENALTY_N_NEWTON,
            &FSRS7_PENALTY_TARGET_DRS,
            &w_vec,
        );
        let expected = (p1.mul_scalar(FSRS7_PENALTY_W_1) + p2.mul_scalar(FSRS7_PENALTY_W_2))
            .mul_scalar(batch_size as f64)
            .into_scalar()
            .to_f32();

        let actual = model
            .fsrs7_schedule_penalty(batch_size)
            .into_scalar()
            .to_f32();

        assert!((actual - expected).abs() < 1e-6);
    }
}
