# FSRS7 Penalty Mode

## Purpose

Provide an optional FSRS-7 training objective that adds scheduling penalties to
the existing optimization loss, while keeping inference behavior unchanged.

## API Surface

- `ComputeParametersInput.fsrs7_penalty` (default: `false`)
  - `false`: existing objective (`BCE + L2`)
  - `true`: `BCE + weighted_L2 + FSRS7_schedule_penalties`

## Objective Components

When `fsrs7_penalty = true`, training uses:

- `L2` with weight `0.5` (applied via `FSRS7_PENALTY_W_L2`)
- schedule penalty 1 with weight `0.5`
- schedule penalty 2 with weight `0.0015`

Penalty simulation settings:

- `n_reviews = 10`
- target DR for penalty 1: `0.90`
- target DR set for penalty 2: `[0.99]`
- Newton iterations: `4`

## Penalty Backpropagation

Penalty terms are now differentiated without burn autograd:

1. `BCE` still uses burn autograd (`loss.backward()`).
2. `L2` and FSRS7 schedule penalty are computed in scalar code.
3. Manual penalty gradients are added to `w` gradients before optimizer step.

The schedule interval inversion still uses the same two-phase math:

1. Phase 1: Newton solve in `u = log(t)` scalar space.
2. Phase 2: one implicit-differentiation lift step around `t*`.

The derivative is propagated by an internal dual-number engine for 35 params.

## Data Flow

- Data created:
  - temporary scalar penalty values and dual gradients during each train/valid batch.
- Data read:
  - current model weights (`w[0..34]`), plus batch size for scaling.
- Data updated:
  - model gradient for `w` is updated by adding manual penalty gradients before optimizer step.
- Data storage:
  - no new persistent data fields; only in-memory training-time values.

## Scope

- Affects optimization only (`compute_parameters` path).
- Does not change runtime scheduling/inference formulas used by `FSRS::next_interval`,
  `FSRS::interval_at_retrievability`, or Anki runtime calls.
