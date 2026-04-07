# FSRS-7 Plot Notes

This folder contains exploratory FSRS-7 behavior plots generated from the benchmark implementation:

- Source model: `/Users/jschoreels/workspace/srs-benchmark/models/fsrs_v7.py`
- Plot script: `/Users/jschoreels/workspace/fsrs-rs/docs/fsrs7-plots/generate_fsrs7_plots.py`
- Output directory: `/Users/jschoreels/workspace/fsrs-rs/docs/fsrs7-plots`

## Assumptions Used

- Parameters: default `FSRS7.init_w` from code.
- Time unit: days.
- The forgetting curve is evaluated with the full FSRS-7 8-parameter curve block (`w[27..34]`).
- Grade-conditioned plot base state: `S=3.0`, `D=5.5`.
- Grade-conditioned plot uses one review update via `FSRS7.step()` then plots future forgetting curves from updated stability.

## Generated Figures

1. `01_forgetting_curve_by_stability.png`
- Purpose: show how recall decays for different fixed stability values.

2. `02_grade_conditioned_curves.png`
- Purpose: show how rating (`Again/Hard/Good/Easy`) changes post-review stability and therefore future forgetting curves.
- Includes two review gaps: `Δt=0.2d` and `Δt=3.0d`.

3. `03_transition_coefficient.png`
- Purpose: show FSRS-7 long/short blend coefficient behavior as a function of elapsed time.

## Regeneration

```bash
python3 /Users/jschoreels/workspace/fsrs-rs/docs/fsrs7-plots/generate_fsrs7_plots.py
```
