use fsrs::{DEFAULT_PARAMETERS, InstantConfig, InstantFsrs7, InstantReview};

fn history() -> Vec<InstantReview> {
    [
        (0.0, 3),
        (0.007, 3),
        (1.0, 3),
        (3.2, 1),
        (0.01, 3),
        (2.0, 3),
    ]
    .iter()
    .map(|&(delta_t, rating)| InstantReview { delta_t, rating })
    .collect()
}

#[test]
fn zero_lr_keeps_parameters() {
    let mut model = InstantFsrs7::new(
        &DEFAULT_PARAMETERS,
        InstantConfig {
            lr: 0.0,
            ..Default::default()
        },
    );
    let h = history();
    let p = model.predict(&h, 5.0);
    assert_eq!(model.observe(&h, 5.0, true), p);
    assert_eq!(model.parameters(), &DEFAULT_PARAMETERS[..]);
}

#[test]
fn update_moves_prediction_towards_outcome() {
    let h = history();
    for recalled in [true, false] {
        let mut model = InstantFsrs7::new(&DEFAULT_PARAMETERS, InstantConfig::default());
        let before = model.observe(&h, 5.0, recalled);
        let after = model.predict(&h, 5.0);
        assert!(before.is_finite() && after.is_finite());
        assert_eq!(after > before, recalled, "before {before}, after {after}");
    }
}
