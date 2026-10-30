# FSRS-Instant

FSRS-Instant is FSRS-7 with its parameters re-fitted online after every review. It
starts from the usual fitted FSRS-7 parameters. After each review it takes one Adam step
on that review's log loss, so every prediction uses parameters fitted on the user's
history up to that point. With a constant step size the parameters follow an
exponentially recency-weighted fit, and future reviews never enter it. Two small logit
offsets, also learned online, sit on top of the curve: a user bias, and a weight on
ln(1 + number of Again answers in the card's history).

Code:
- `src/instant.rs` (branch `fsrs-instant`): `InstantFsrs7::observe` / `observe_with_offset`.
- `tools/instant-replay`: replays one user's review stream for srs-benchmark.
- srs-benchmark branch `fsrs-instant` (JSchoreels/srs-benchmark): `fsrs_instant.py` and
  `fsrs_instant_compare.py`; result files are in `runs/`.

## Protocol

This is the published FSRS-7 row: `script.py --algo FSRS-7 --short --secs
--equalize_test_with_non_secs`, with the same 5 time-series folds, test reviews and
metrics code (`utils.evaluate`).

- For each fold, the driver trains FSRS-7 exactly as `script.py` does. Those parameters
  give the FSRS-7 baseline, which is held fixed over the fold.
- FSRS-Instant starts the fold from the *same* parameters. It then walks every review in
  the fold's time span in revlog order (`review_th`, strictly increasing `review_time`):
  1. it predicts the review if it is a test review;
  2. it updates on the review's outcome, same-day reviews included.
- The comparison is therefore paired: FSRS-Instant equals the baseline plus online updates.
- Our baseline reproduces the published per-user results. On users 1-300 the per-user
  LogLoss is 0.3519 vs 0.3519 published. With updates off, the Rust replay matches the
  Python predictions to within 1.5e-4 (2e-5 on most users).
- The offsets persist across folds and start at 0 at the first test fold.
- The sibling (note) bias was tested and dropped. Cards missing from the cards table
  (deleted before the export) get no note bias at all, so deletion status never changes
  the model's behaviour (deleted-card leak).

Settings were chosen on 100 dev users: random ids above 5000, seed 0, in
`runs/dev-users.txt`. The quick check below uses users 1-300, which are disjoint from the
dev set. CIs are paired user-bootstrap 95% intervals (10,000 resamples) on the difference
vs FSRS-7. "Per user" is the README's averaging; "review-weighted" weights each user by
their number of test reviews.

## Results: users 1-300 (9.76M test reviews)

FSRS-Instant uses Adam lr 1e-3 (betas 0.9/0.999), user bias SGD rate 0.1 and Again-weight
rate 0.01, all fixed on the dev users before this run.

| Per user (README) | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
| --- | --- | --- | --- |
| FSRS-7 (published, same users) | 0.3519 | 0.0649 | 0.7145 |
| FSRS-7 (this run) | 0.3519 | 0.0648 | 0.7146 |
| **FSRS-Instant** | **0.3163** (−10.1%, Δ −0.0355 [−0.0420, −0.0297]) | **0.0376** (−42.1%, Δ −0.0273 [−0.0310, −0.0239]) | **0.7764** (+8.7%, Δ +0.0618 [+0.0536, +0.0705]) |

| Review-weighted | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
| --- | --- | --- | --- |
| FSRS-7 (this run) | 0.3245 | 0.0494 | 0.7278 |
| **FSRS-Instant** | **0.2967** (−8.6%, [−0.0341, −0.0225]) | **0.0248** (−49.8%, [−0.0286, −0.0212]) | **0.7851** (+7.9%, [+0.0477, +0.0679]) |

For scale, the README's 9,999-user means are FSRS-7 0.3437, FSRS-7 recency 0.3414, GRU
0.3328, RWKV-Curve 0.3193 and RWKV-Instant 0.2773. A −10% change on FSRS-7 would put
FSRS-Instant near RWKV-Curve. That is an extrapolation from 300 users: the full
9,999-user run has not been done.

## What helped

Ablations below are per-user LogLoss vs FSRS-7. The dev numbers are on 100 users; the last
column is users 1-300.

| Variant | Dev | Users 1-300 |
| --- | --- | --- |
| Parameters only, lr 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 | −3.0 / −4.9 / −6.7 / −7.6 / −6.4% | lr 3e-3: −8.3% |
| User bias only (rate 0.1, parameters fixed) | − | −8.0% |
| lr 1e-3 + user bias 0.03; lr 2e-3 + user bias 0.1 | −8.1%; −9.0% | − |
| + Again weight 0.01 (= FSRS-Instant) | −9.2% | −10.1% |
| user bias 0.3 instead of 0.1 | − | −10.8% (not tuned on dev) |

- **Online parameter steps** give most of the gain. The best step size is 1e-3 to 3e-3
  per review.
- **A fast user bias** (one logit offset, SGD after every review) gives about as much on
  its own, and combines with the parameter steps.
- **Again count** (a weight on ln(1 + Again answers), the RWKV hard-card signal) adds a
  small, consistent improvement of 0.2-0.4 points.

## What didn't

- **Per-card bias** and **per-note (sibling) bias**: no gain on top of the parameter steps
  (within 0.1 points on dev, at rates 0.02-0.1).
- **L2 pull towards the fold's fit**: 0.01 is neutral and 0.1 halves the gain. **Adam
  betas 0.7/0.98** (the batch trainer's) are slightly worse than 0.9/0.999.
- **Scheduling-time predictions.** The benchmark scores the probability of recall just
  before each review, and FSRS-Instant updates on every other card's reviews in the
  meantime, so its gain is largely an *instant* effect: it follows the user's recent form.
  A scheduler, by contrast, picks the interval at the card's previous review. To measure
  what a scheduler would get, the "curve" replay mode predicts each review with the
  parameters and offsets from right after that card's previous review:

  | Curve mode, users 1-300 | LogLoss per user | LogLoss review-weighted | RMSE(bins) per user |
  | --- | --- | --- | --- |
  | FSRS-Instant settings | +3.1% [worse] | +1.7% [worse] | −5.6% |
  | parameters only, lr 3e-4 | −0.3% [CI includes 0] | −1.6% [−0.0080, −0.0025] | −4.1% |
  | parameters only, lr 1e-4 | −0.2% [CI includes 0] | −1.6% [−0.0076, −0.0027] | −1.4% |

  The fast settings that win the benchmark make scheduling-time predictions *worse*. A
  slow online step (lr 1e-4 to 3e-4) gives a small curve-level gain, mostly for users
  with many reviews: on the per-user mean it is within noise, about the size of the
  README's FSRS-7 recency gain. This matches what the RWKV models showed: recency-heavy
  adaptation mostly helps the instant target, not the curve.

## Cost

An update is one forward pass over the reviewed card's history with forward-mode dual
numbers (34 partials), and that same pass also gives the prediction. Single-threaded, it
takes **about 8 µs per review** (5 users, mean history 4-7 reviews). Of that, 7.6 µs is the
dual forward pass and 0.3 µs is the Adam step and offsets. The state per user is 34
parameters plus Adam moments, about 1 KB. No batch re-fit is needed. For comparison, the
batch FSRS-7 fit in the benchmark took about 74 s per user (5 folds, Python).

## Caveats and next steps

- Only 300 users so far. The 9,999-user run costs about 200 CPU-hours (the per-fold FSRS-7
  re-fits dominate) and has not been run.
- The headline gain is an instant-prediction gain. For Anki it suits displayed
  retrievability and "which card is most at risk now", not interval choice. For
  scheduling, the slow-step curve variant is the candidate, and it needs its own dev
  tuning.
- In curve mode the snapshots are kept per card. A real deployment would store the
  parameters used when the interval was set, or re-derive them.
- This machine's Windows GNU toolchain has no C compiler or `dlltool`, so the crate's own
  test suite was not run here. The replay crate's tests pass.
