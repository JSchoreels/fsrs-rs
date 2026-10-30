# FSRS7-Instant

FSRS7-Instant is FSRS-7 with its parameters re-fitted online after every review. It
starts from the usual fitted FSRS-7 parameters. After each review it takes one Adam step
on that review's log loss, so every prediction uses parameters fitted on the user's
history up to that point. With a constant step size the parameters follow an
implicitly recency-weighted fit (see "Recency weighting" below), and future reviews never
enter it. Two small logit
offsets, also learned online, sit on top of the curve: a user bias, and a weight on
ln(1 + number of Again answers in the card's history).

Code:
- `src/instant.rs` (branch `fsrs-instant`): `InstantFsrs7::observe` / `observe_with_offset`.
- `tools/instant-replay`: replays one user's review stream for srs-benchmark.
- srs-benchmark branch `fsrs-instant` (JSchoreels/srs-benchmark): `fsrs_instant.py` and
  `fsrs_instant_compare.py`; result files are in `runs/`.

Names: the model is called FSRS7-Instant everywhere people read it: docs, tables, result
file names, and the srs-benchmark display name. Code identifiers keep their names
(`InstantFsrs7`, `src/instant.rs`, `tools/instant-replay`, `fsrs_instant*.py`, branch
`fsrs-instant`). Upstream's result file name comes from its flag (`--instant` →
`FSRS-7-instant-short-secs-...jsonl`, like `-recency`), so that one keeps the flag form.

## Recency weighting, compared with FSRS-7

The FSRS-7 formula is unchanged: forgetting curve, stability and difficulty updates, all
34 parameters and their bounds. What changes is how the parameters are fitted over time,
plus two new logit offsets.

| Part | FSRS-7 | FSRS7-Instant | Status |
| --- | --- | --- | --- |
| Batch fit weighting | see (1) | same as the published FSRS-7 row: uniform | same (as the benchmark's FSRS-7) |
| Per-review parameter step | none (fitted once) | (2) | new |
| User bias | none | (3) | new |
| Again-count weight | none | (4) | new |

**(1) FSRS-7's recency weight in training.** Each training item i (a card's review prefix,
sorted by review time, i = 0 … N−1) gets a weight on its log loss:

- Anki's optimizer (fsrs-rs `recency_weighted_fsrs_items`, `src/dataset.rs`):
  `weight_i = 0.25 + 0.75 · (i / (N−1))^3`. The oldest item gets 0.25 and the newest 1.0.
  The weight falls below half of the newest at i/(N−1) = (1/3)^(1/3) ≈ 0.69, so the
  oldest 69% of the history counts at less than half weight.
- srs-benchmark's "FSRS-7 recency" row (`script.py _apply_recency_weighting`):
  `weight_i = 0.0667 + 0.9333 · (i / N)^11.25`. Only the newest ~7% of the history is above
  half weight, with a floor of 0.0667.
- The published "FSRS-7" row, which is the baseline here, uses no recency weight:
  `weight_i = 1`.

Both are weightings over the whole history: there is no fixed half-life in reviews or
days, since the weight depends on the position i/N.

**FSRS7-Instant's batch fit** is FSRS-7's trainer with `weight_i = 1`, the same as the
published FSRS-7 row:

- protocol A: per-user, per-fold fits;
- protocol B: one shared fit on 100 training users.

It is not recency-weighted. All of FSRS7-Instant's recency comes from (2)-(4), so it is
**implicit**: there is no weight function, only constant-step online updates that keep
moving the fit towards recent reviews.

**(2) Per-review parameter step (new).** After each review, with p the prediction and y
the outcome (1 = recalled):

- `ℓ = −[y ln p + (1−y) ln(1−p)]` and `g = ∂ℓ/∂w` over all 34 FSRS-7 parameters, through
  the card's full history.
- One Adam step with lr 1e-3, β1 0.9, β2 0.999, eps 1e-8, with bias correction:
  `m ← β1·m + (1−β1)·g`, `v ← β2·v + (1−β2)·g²`, `w ← w − lr · m̂ / (√v̂ + eps)`.
- Then FSRS-7's own parameter bounds and monotonicity constraints are applied, and w is
  stored as float32.
- It runs on every review, same-day reviews included, with no L2 pull towards the start.
  In protocol A, w and the Adam state restart from each fold's fit.
- Effective recency: Adam moves each parameter by at most about lr = 1e-3 (raw units) per
  review, whatever the gradient size. The gradient direction is averaged over about 10
  reviews (β1) and normalised over about 1,000 reviews (β2).
- How fast old reviews are forgotten depends on the curvature and noise of each
  parameter's loss, so there is no closed-form half-life. I have not measured one (an
  impulse test would give it).

**(3) User bias (new).** One logit offset b per user, starting at 0, with plain SGD at
rate 0.1 and no explicit decay:

`p = σ(logit R + b + w_a · ln(1+a))`, then `b ← b − 0.1 · (p − y)`.

- Linearised around equilibrium, a deviation of b shrinks by a factor `1 − 0.1·p(1−p)` per
  review.
- The implicit half-life is therefore `ln 2 / (0.1 · p(1−p))` reviews, counting all of the
  user's reviews: about 65 reviews at p ≈ 0.87, 45 at p = 0.8, and 150 at p = 0.95.
- This is the fast part: b follows the user's form over the last few dozen reviews.

**(4) Again-count weight (new).** a is the number of Again answers in the card's earlier
reviews, learning steps included, so it counts more than lapses. The offset is
`w_a · ln(1+a)`, so the odds of recall are multiplied by `(1+a)^(w_a)`. When the learned
w_a is negative (more Agains, lower recall), this is `(1+a)^(−w)` with w = −w_a.

- w_a starts at 0 and is learned per user by SGD at rate 0.01: `w_a ← w_a − 0.01 · (p − y)
  · ln(1+a)`. There is no decay.
- Only cards with a ≥ 1 move it. Linearised, its half-life is
  `ln 2 / (0.01 · ln(1+a)² · p(1−p))`, about 1,400 such reviews at a = 1. In practice it
  is a slowly learned per-user constant, not a recency term.
- b and w_a carry over across folds in protocol A. In protocol B they start at the
  user's first review.

## Protocol

This is the published FSRS-7 row: `script.py --algo FSRS-7 --short --secs
--equalize_test_with_non_secs`, with the same 5 time-series folds, test reviews and
metrics code (`utils.evaluate`).

- For each fold, the driver trains FSRS-7 exactly as `script.py` does. Those parameters
  give the FSRS-7 baseline, which is held fixed over the fold.
- FSRS7-Instant starts the fold from the *same* parameters. It then walks every review in
  the fold's time span in revlog order (`review_th`, strictly increasing `review_time`):
  1. it predicts the review if it is a test review;
  2. it updates on the review's outcome, same-day reviews included.
- The comparison is therefore paired: FSRS7-Instant equals the baseline plus online updates.
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

FSRS7-Instant uses Adam lr 1e-3 (betas 0.9/0.999), user bias SGD rate 0.1 and Again-weight
rate 0.01, all fixed on the dev users before this run.

| Per user (README) | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
| --- | --- | --- | --- |
| FSRS-7 (published, same users) | 0.3519 | 0.0649 | 0.7145 |
| FSRS-7 (this run) | 0.3519 | 0.0648 | 0.7146 |
| **FSRS7-Instant** | **0.3163** (−10.1%, Δ −0.0355 [−0.0420, −0.0297]) | **0.0376** (−42.1%, Δ −0.0273 [−0.0310, −0.0239]) | **0.7764** (+8.7%, Δ +0.0618 [+0.0536, +0.0705]) |

| Review-weighted | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
| --- | --- | --- | --- |
| FSRS-7 (this run) | 0.3245 | 0.0494 | 0.7278 |
| **FSRS7-Instant** | **0.2967** (−8.6%, [−0.0341, −0.0225]) | **0.0248** (−49.8%, [−0.0286, −0.0212]) | **0.7851** (+7.9%, [+0.0477, +0.0679]) |

For scale, the README's 9,999-user means are FSRS-7 0.3401, FSRS-7 recency 0.3370, GRU
0.3328, RWKV-Curve 0.3193 and RWKV-Instant 0.2773. A −10% change on FSRS-7 would put
FSRS7-Instant near RWKV-Curve. That is an extrapolation from 300 users; see the next
section for a run on RWKV's 4,999 test users.

## Results: RWKV's test users 1-4999, one shared fit (170.8M test reviews)

This run uses the same users as srs-benchmark's RWKV rows. RWKV is trained on users
5000-10000 and tested on users 1-4999, with the same test reviews as FSRS-7's folds.

- **Fit.** One FSRS-7 parameter set is fitted on 100 of RWKV's training users: a random
  seed-0 sample from 5000-10000 (`runs/shared-train-100.txt`), 5.8M reviews pooled, using
  the benchmark's own trainer. The fit took 25 minutes. No test user is ever re-fitted.
- **Replay.** Each test user's whole review stream is replayed once, from their first
  review, starting from the shared parameters and using the online settings fixed on the
  dev users. Nothing was re-tuned. The test reviews and the metrics code are the
  benchmark's.
- **Three prediction modes**, from one pass:
  - *instant*: the parameters and offsets just before the review, as above;
  - *scheduling time*: the snapshot taken right after the card's previous review, when its
    interval was set;
  - *start of day*: the snapshot before the user's first review of that day (the revlog's
    day), so same-day reviews of other cards are not used.
- **Baseline.** The published rows (per-user FSRS-7 refits on 5 folds, FSRS-7 recency, GRU,
  RWKV-Curve = `RWKV.jsonl`, RWKV-Instant = `RWKV-P.jsonl`) are shown on the same users.
  CIs are paired user-bootstrap 95% intervals on the difference vs published FSRS-7.
- **Data note.** 284 of the 4,998 users have a slightly different test-review count here
  than in the published files (local dataset version). The pairing is per user, so this
  adds a little noise to the comparisons with published rows.
- "Parameters only, lr 3e-4" turns the offsets off and uses the slow step. It was picked
  as a curve candidate on users 1-300 (which are inside this test set), not on dev users,
  so read it as indicative.

Commands: `runs/shared-5k.sh`, then `python fsrs_instant_shared_compare.py runs/shared-5k`
(srs-benchmark branch `fsrs-instant`).

| Per user (README) | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
| --- | --- | --- | --- |
| FSRS-7 (published, per-user refit) | 0.3418 | 0.0628 | 0.7164 |
| FSRS-7 recency (published) | 0.3388 (−0.9%, [−0.0034, −0.0028]) | 0.0588 (−6.4%) | 0.7218 (+0.7%) |
| GRU (published) | 0.3343 (−2.2%, [−0.0081, −0.0070]) | 0.0544 (−13.4%) | 0.7321 (+2.2%) |
| RWKV-Curve (published) | 0.3212 (−6.0%, [−0.0215, −0.0198]) | 0.0547 (−12.9%) | 0.7680 (+7.2%) |
| RWKV-Instant (published) | 0.2791 (−18.3%, [−0.0647, −0.0608]) | 0.0252 (−59.9%) | 0.8320 (+16.1%) |
| FSRS-7 shared fit, no updates | 0.3623 (+6.0%, [+0.0193, +0.0217]) | 0.0873 (+38.9%) | 0.7039 (−1.8%) |
| **FSRS7-Instant, instant** | **0.3093** (−9.5%, [−0.0340, −0.0312]) | **0.0363** (−42.3%, [−0.0274, −0.0258]) | **0.7780** (+8.6%, [+0.0595, +0.0637]) |
| FSRS7-Instant, scheduling time | 0.3549 (+3.8%, [+0.0121, +0.0141]) | 0.0621 (−1.1%, [−0.0012, −0.0002]) | 0.6998 (−2.3%, [−0.0183, −0.0150]) |
| FSRS7-Instant, start of day | 0.3353 (−1.9%, [−0.0074, −0.0056]) | 0.0480 (−23.6%, [−0.0155, −0.0142]) | 0.7390 (+3.2%, [+0.0209, +0.0243]) |
| parameters only lr 3e-4, instant | 0.3264 (−4.5%, [−0.0162, −0.0147]) | 0.0507 (−19.2%) | 0.7495 (+4.6%) |
| parameters only lr 3e-4, scheduling time | 0.3446 (+0.8%, [+0.0022, +0.0034]) | 0.0636 (+1.2%, [+0.0003, +0.0012]) | 0.7092 (−1.0%) |
| parameters only lr 3e-4, start of day | 0.3315 (−3.0%, [−0.0110, −0.0097]) | 0.0531 (−15.5%) | 0.7378 (+3.0%) |

| Review-weighted | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
| --- | --- | --- | --- |
| FSRS-7 (published, per-user refit) | 0.3203 | 0.0452 | 0.7185 |
| RWKV-Curve (published) | 0.3017 (−5.8%, [−0.0196, −0.0178]) | 0.0349 (−22.8%) | 0.7717 (+7.4%) |
| RWKV-Instant (published) | 0.2735 (−14.6%, [−0.0491, −0.0448]) | 0.0149 (−67.0%) | 0.8228 (+14.5%) |
| FSRS-7 shared fit, no updates | 0.3389 (+5.8%, [+0.0172, +0.0199]) | 0.0701 (+55.1%) | 0.6988 (−2.7%) |
| **FSRS7-Instant, instant** | **0.2960** (−7.6%, [−0.0259, −0.0229]) | **0.0231** (−48.9%, [−0.0230, −0.0212]) | **0.7763** (+8.1%, [+0.0537, +0.0623]) |
| FSRS7-Instant, scheduling time | 0.3308 (+3.3%, [+0.0091, +0.0120]) | 0.0414 (−8.4%, [−0.0045, −0.0031]) | 0.7068 (−1.6%, [−0.0145, −0.0089]) |
| FSRS7-Instant, start of day | 0.3172 (−1.0%, [−0.0045, −0.0019]) | 0.0320 (−29.1%, [−0.0140, −0.0122]) | 0.7429 (+3.4%, [+0.0209, +0.0283]) |
| parameters only lr 3e-4, scheduling time | 0.3207 (+0.1%, [−0.0003, +0.0011]) | 0.0422 (−6.6%, [−0.0036, −0.0023]) | 0.7148 (−0.5%, [−0.0069, −0.0006]) |
| parameters only lr 3e-4, start of day | 0.3091 (−3.5%, [−0.0122, −0.0103]) | 0.0336 (−25.7%, [−0.0123, −0.0109]) | 0.7480 (+4.1%, [+0.0260, +0.0332]) |

What this shows:

- **Instant.** Starting from a shared fit with no per-user batch fit at all, FSRS7-Instant
  reaches −9.5% LogLoss vs per-user FSRS-7. It recovers the shared fit's +6.0% handicap
  and more. It sits between RWKV-Curve (−6.0%) and RWKV-Instant (−18.3%), and its
  RMSE(bins) is close to RWKV-Instant's. This matches the users 1-300 result (−10.1% with
  per-user fold fits).
- **Scheduling time.** It is 3.8% worse than per-user FSRS-7 (3.3% review-weighted), the
  same picture as on users 1-300. The slow parameters-only step is about neutral (+0.8%
  per user, +0.1% review-weighted, CI includes 0). Neither is a scheduling gain.
- **Start of day.** Only same-day reviews are excluded, and it is still −1.9% LogLoss and
  −24% RMSE(bins). With the slow step it is −3.0%. So part of the gain comes from
  day-to-day drift, not only from within-session form.
- **Cost.** The Rust replay ran at 44 µs per review for all three modes together (8
  processes on a loaded machine, GPU jobs running). Data loading took 97 µs per review.
  With the two other configs, the replay of the 306M stream reviews came to about 17
  CPU-hours, plus 0.5 hours for the shared fit. The benchmark's per-user refit protocol
  costs about 200 CPU-hours.

## What helped

Ablations below are per-user LogLoss vs FSRS-7. The dev numbers are on 100 users; the last
column is users 1-300.

| Variant | Dev | Users 1-300 |
| --- | --- | --- |
| Parameters only, lr 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 | −3.0 / −4.9 / −6.7 / −7.6 / −6.4% | lr 3e-3: −8.3% |
| User bias only (rate 0.1, parameters fixed) | − | −8.0% |
| lr 1e-3 + user bias 0.03; lr 2e-3 + user bias 0.1 | −8.1%; −9.0% | − |
| + Again weight 0.01 (= FSRS7-Instant) | −9.2% | −10.1% |
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
  before each review, and FSRS7-Instant updates on every other card's reviews in the
  meantime, so its gain is largely an *instant* effect: it follows the user's recent form.
  A scheduler, by contrast, picks the interval at the card's previous review. To measure
  what a scheduler would get, the "curve" replay mode predicts each review with the
  parameters and offsets from right after that card's previous review:

  | Curve mode, users 1-300 | LogLoss per user | LogLoss review-weighted | RMSE(bins) per user |
  | --- | --- | --- | --- |
  | FSRS7-Instant settings | +3.1% [worse] | +1.7% [worse] | −5.6% |
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
