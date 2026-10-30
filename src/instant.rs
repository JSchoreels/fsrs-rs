//! FSRS-Instant: FSRS-7 whose parameters are re-fitted online after every review.
//!
//! Start from parameters fitted the usual way (`compute_parameters`), then after each
//! review take one Adam step on that review's log loss. With a constant step size this
//! tracks the minimiser of an exponentially recency-weighted loss over the reviews seen so
//! far, so each prediction uses parameters fitted on the user's history up to that point,
//! recent reviews weighted most, and never on future reviews.
//!
//! The gradient comes from the forward-mode dual numbers of `analytic_v7`: one pass over
//! the reviewed card's history gives the prediction and its gradient together, so an
//! update costs one forward pass of that card (microseconds), with no batch re-fit.

use crate::analytic_v7::{
    Dual, MemoryStateDual, MemoryStateScalar, PARAM_LEN, bce_loss, dual_params,
    forgetting_curve, forgetting_curve_scalar, step, step_scalar,
};
use crate::parameter_clipper::clip_parameters_in_place;

/// Online optimiser settings.
#[derive(Clone, Copy, Debug)]
pub struct InstantConfig {
    /// Adam step size per review. 0 disables the online updates (plain FSRS-7).
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Pull towards the starting parameters, as `l2 * sum(((w - w0) / sigma)^2)` per review.
    pub l2: f64,
}

impl Default for InstantConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            l2: 0.0,
        }
    }
}

/// Prior standard deviations of the FSRS-7 parameters (the L2 prior used in training).
const PARAMS_STDDEV: [f64; PARAM_LEN] = [
    9999.0, 9999.0, 9999.0, 9999.0, 0.523, 0.2528, 0.4329, 0.2966, 0.2139, 0.2889, 0.1862, 0.175,
    0.3812, 0.3013, 0.9104, 0.3234, 0.2448, 0.3273, 0.1842, 0.1735, 0.4608, 0.311, 0.864, 0.0418,
    0.2596, 0.0798, 0.0682, 0.1282, 0.1397, 0.1407, 0.1489, 0.2, 0.15, 0.15,
];

/// One review of a card's history: elapsed days since the previous review (fractional),
/// and the rating (1..=4).
#[derive(Clone, Copy, Debug)]
pub struct InstantReview {
    pub delta_t: f32,
    pub rating: u32,
}

/// Online-updated FSRS-7 parameters for one user.
#[derive(Clone, Debug)]
pub struct InstantFsrs7 {
    config: InstantConfig,
    w: [f32; PARAM_LEN],
    anchor: [f32; PARAM_LEN],
    m: [f64; PARAM_LEN],
    v: [f64; PARAM_LEN],
    t: i32,
}

impl InstantFsrs7 {
    pub fn new(parameters: &[f32], config: InstantConfig) -> Self {
        assert_eq!(
            parameters.len(),
            PARAM_LEN,
            "FSRS-Instant needs 34 FSRS-7 parameters"
        );
        let w: [f32; PARAM_LEN] = core::array::from_fn(|i| parameters[i]);
        Self {
            config,
            w,
            anchor: w,
            m: [0.0; PARAM_LEN],
            v: [0.0; PARAM_LEN],
            t: 0,
        }
    }

    pub fn parameters(&self) -> &[f32] {
        &self.w
    }

    fn forward(&self, history: &[InstantReview], delta_t: f32) -> Dual {
        let w = dual_params(&self.w);
        let zero = Dual {
            value: 0.0,
            grad: [0.0; PARAM_LEN],
        };
        let mut state = MemoryStateDual {
            stability: zero,
            stability_fast: zero,
            difficulty: zero,
        };
        for (nth, review) in history.iter().enumerate() {
            state = step(
                &w,
                review.delta_t as f64,
                review.rating as usize,
                state,
                nth,
            );
        }
        forgetting_curve(
            &w,
            delta_t as f64,
            state.stability,
            state.stability_fast,
            state.difficulty,
        )
    }

    /// Probability of recall after `delta_t` days for a card with this history.
    pub fn predict(&self, history: &[InstantReview], delta_t: f32) -> f64 {
        self.forward(history, delta_t).value
    }

    /// [`Self::predict`] with other parameters, for example a snapshot of
    /// [`Self::parameters`] taken when the card was scheduled. Plain f64 pass, no gradient.
    pub fn predict_with(parameters: &[f32], history: &[InstantReview], delta_t: f32) -> f64 {
        let mut state = MemoryStateScalar {
            stability: 0.0,
            stability_fast: 0.0,
            difficulty: 0.0,
        };
        for (nth, review) in history.iter().enumerate() {
            state = step_scalar(
                parameters,
                review.delta_t as f64,
                review.rating as usize,
                state,
                nth,
            );
        }
        forgetting_curve_scalar(
            parameters,
            delta_t as f64,
            state.stability,
            state.stability_fast,
            state.difficulty,
        )
    }

    /// Predict the review, then learn from its outcome. Returns the prediction made
    /// before the update.
    pub fn observe(&mut self, history: &[InstantReview], delta_t: f32, recalled: bool) -> f64 {
        let r = self.forward(history, delta_t);
        if self.config.lr == 0.0 || history.is_empty() {
            return r.value;
        }
        let loss = bce_loss(r, if recalled { 1.0 } else { 0.0 }, 1.0);
        self.adam_step(&loss.grad);
        r.value
    }

    /// Like [`Self::observe`], for the prediction `sigmoid(logit(R) + offset)`, where
    /// `offset` is an extra logit term the caller learns (for example a per-card bias).
    /// Returns that prediction, made before the update.
    pub fn observe_with_offset(
        &mut self,
        history: &[InstantReview],
        delta_t: f32,
        recalled: bool,
        offset: f64,
    ) -> f64 {
        let r = self.forward(history, delta_t);
        let rv = r.value.clamp(1e-5, 1.0 - 1e-5);
        let p = 1.0 / (1.0 + (-((rv / (1.0 - rv)).ln() + offset)).exp());
        if self.config.lr == 0.0 || history.is_empty() {
            return p;
        }
        // d(log loss)/d(logit) = p - y, and d(logit R)/dR = 1 / (R (1 - R)).
        let scale = (p - if recalled { 1.0 } else { 0.0 }) / (rv * (1.0 - rv));
        let grad: [f64; PARAM_LEN] = core::array::from_fn(|i| r.grad[i] * scale);
        self.adam_step(&grad);
        p
    }

    fn adam_step(&mut self, grad: &[f64; PARAM_LEN]) {
        let c = self.config;
        self.t += 1;
        let bias1 = 1.0 - c.beta1.powi(self.t);
        let bias2 = 1.0 - c.beta2.powi(self.t);
        for i in 0..PARAM_LEN {
            let mut g = grad[i];
            if c.l2 > 0.0 {
                let sigma = PARAMS_STDDEV[i];
                g += 2.0 * c.l2 * (self.w[i] - self.anchor[i]) as f64 / (sigma * sigma);
            }
            if !g.is_finite() {
                continue;
            }
            self.m[i] = c.beta1 * self.m[i] + (1.0 - c.beta1) * g;
            self.v[i] = c.beta2 * self.v[i] + (1.0 - c.beta2) * g * g;
            let step = c.lr * (self.m[i] / bias1) / ((self.v[i] / bias2).sqrt() + c.eps);
            self.w[i] = (self.w[i] as f64 - step) as f32;
        }
        clip_parameters_in_place(&mut self.w, 0, true);
    }
}
