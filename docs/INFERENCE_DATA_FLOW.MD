# Inference Data Flow

## Model and inputs

`FSRS<B>` owns the versioned Burn model and backend device. Construction
normalizes and clips the input parameters. Inference reads chronological
`FSRSItem.reviews` and creates transient `MemoryStateTensors` through
`Model::forward` and `Model::step`. Public memory states contain `stability`,
`difficulty`, and `stability_fast`.

FSRS-6 rounds elapsed days internally; FSRS-7 retains fractional elapsed days.
Scheduling reads the complete current memory state, desired retention, and
elapsed days. No caller-owned reviews or memory states are mutated.

## Evaluation

`evaluate()` batches prefix histories and aggregates recency-weighted binary
cross-entropy and R-matrix RMSE. `evaluate_with_card_ids()` accepts an aligned
card-ID vector, groups items in first-appearance order, and reuses the versioned
model's state along each expanding-window history. Non-prefix groups replay
items independently. Metrics are aggregated in original item order; tensor
batch shapes and unordered RMSE summation can cause last-bit differences.
Empty input, misaligned IDs, non-finite metrics, and cancellation are reported
through the existing error API. No grouping state is persisted.

Time-series evaluation slices card IDs alongside each training prefix and
forwards optional training hyperparameters. Progress callbacks receive transient
`ItemProgress` values; returning false stops evaluation before the next split.

## Upstream integration

This branch retains its generic backend API and tensor inference paths. Upstream
main's single-model scalar implementation is not substituted for the FSRS-6/7
model dispatcher. Historical scalar-port descriptions in the speed plan are
not a description of the current inference implementation.
