use super::training_v7;

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
}
