use super::training_v7;

pub(crate) fn l2_penalty_value_and_grad(
    w: &[f32],
    _init_w: &[f32],
    _batch_size: usize,
    _total_size: usize,
    _l2_weight: f64,
    _params_stddev: &[f32],
) -> (f64, Vec<f32>) {
    (0.0, vec![0.0; w.len()])
}

pub(crate) fn maybe_schedule_penalty_value_and_grad(
    _w: &[f32],
    _batch_size: usize,
    _enable_sched_penalties: bool,
) -> (f64, [f64; training_v7::GRAD_LEN]) {
    (0.0, [0.0; training_v7::GRAD_LEN])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsrs6_schedule_penalty_is_noop() {
        let w = crate::inference::FSRS6_DEFAULT_PARAMETERS.to_vec();
        let (value, grad) = maybe_schedule_penalty_value_and_grad(&w, 512, true);
        assert_eq!(value, 0.0);
        assert!(grad.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn test_fsrs6_l2_penalty_is_noop() {
        let w = crate::inference::FSRS6_DEFAULT_PARAMETERS.to_vec();
        let init_w = w.clone();
        let (value, grad) = l2_penalty_value_and_grad(
            &w,
            &init_w,
            512,
            1000,
            training_v7::PENALTY_W_L2,
            &training_v7::PARAMS_STDDEV,
        );
        assert_eq!(value, 0.0);
        assert_eq!(grad.len(), w.len());
        assert!(grad.iter().all(|g| *g == 0.0));
    }
}
