//! Replays one user's review stream through FSRS-Instant for srs-benchmark.
//!
//! `instant_replay <input.bin> <output.bin>`. All values are little-endian.
//!
//! Input:
//!   i32 n_folds, n_cards, n_rows, n_configs, n_hist
//!   f32 [n_folds * 34]        fold parameters (the benchmark's own per-fold fit)
//!   f64 [n_configs * 10]      lr, beta1, beta2, eps, l2, user_bias_lr, card_bias_lr,
//!                             note_bias_lr, again_lr, curve
//!   i64 [n_cards + 1]         offsets of each card's full history in hist_*
//!   f32 [n_hist], i32 [n_hist] hist delta_t (days), rating
//!   i32 [n_rows] card, i32 [n_rows] prefix_len, f32 [n_rows] delta_t, i32 [n_rows] y,
//!   i32 [n_rows] fold (rows sorted by time; a new fold restarts from its parameters),
//!   i32 [n_rows] note index (-1: no note bias)
//!
//! The optional logit offsets (user, card and note biases, and a weight on
//! ln(1 + Again count) of the card's history) are learned by SGD after every review and
//! persist across folds. With `curve` != 0, each review is predicted with the
//! parameters and offsets from right after the card's previous review, i.e. what a
//! scheduler had when it picked the interval (the updates themselves are unchanged).
//! Output: f64 [n_configs * n_rows] predictions, then f64 [n_configs] seconds spent.
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use fsrs::{InstantConfig, InstantFsrs7, InstantReview};

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let out = self.bytes[self.pos..self.pos + N].try_into().unwrap();
        self.pos += N;
        out
    }
    fn i32s(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| i32::from_le_bytes(self.take())).collect()
    }
    fn i64s(&mut self, n: usize) -> Vec<i64> {
        (0..n).map(|_| i64::from_le_bytes(self.take())).collect()
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32::from_le_bytes(self.take())).collect()
    }
    fn f64s(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| f64::from_le_bytes(self.take())).collect()
    }
}

struct Snapshot {
    parameters: Vec<f32>,
    user_bias: f64,
    card_bias: f64,
    note_bias: f64,
    again_weight: f64,
}

fn with_logit_offset(r: f64, offset: f64) -> f64 {
    let r = r.clamp(1e-5, 1.0 - 1e-5);
    1.0 / (1.0 + (-((r / (1.0 - r)).ln() + offset)).exp())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).expect("read input");
    let mut rd = Reader {
        bytes: &bytes,
        pos: 0,
    };
    let head = rd.i32s(5);
    let (n_folds, n_cards, n_rows, n_configs, n_hist) = (
        head[0] as usize,
        head[1] as usize,
        head[2] as usize,
        head[3] as usize,
        head[4] as usize,
    );
    let fold_params = rd.f32s(n_folds * 34);
    let configs = rd.f64s(n_configs * 10);
    let offsets = rd.i64s(n_cards + 1);
    let hist_t = rd.f32s(n_hist);
    let hist_r = rd.i32s(n_hist);
    let card = rd.i32s(n_rows);
    let prefix = rd.i32s(n_rows);
    let delta_t = rd.f32s(n_rows);
    let y = rd.i32s(n_rows);
    let fold = rd.i32s(n_rows);
    let note = rd.i32s(n_rows);
    assert_eq!(rd.pos, bytes.len(), "trailing input");

    let hist: Vec<InstantReview> = hist_t
        .iter()
        .zip(&hist_r)
        .map(|(&delta_t, &rating)| InstantReview {
            delta_t,
            rating: rating as u32,
        })
        .collect();

    let mut preds = Vec::with_capacity(n_configs * n_rows);
    let mut secs = Vec::with_capacity(n_configs);
    for c in configs.chunks_exact(10) {
        let config = InstantConfig {
            lr: c[0],
            beta1: c[1],
            beta2: c[2],
            eps: c[3],
            l2: c[4],
        };
        let (user_lr, card_lr, note_lr, again_lr) = (c[5], c[6], c[7], c[8]);
        let curve = c[9] != 0.0;
        let with_offset = user_lr > 0.0 || card_lr > 0.0 || note_lr > 0.0 || again_lr > 0.0;
        let start = Instant::now();
        let mut model: Option<(usize, InstantFsrs7)> = None;
        let mut user_bias = 0.0;
        let mut again_weight = 0.0;
        let mut card_bias: HashMap<i32, f64> = HashMap::new();
        let mut note_bias: HashMap<i32, f64> = HashMap::new();
        let mut snapshots: HashMap<i32, Snapshot> = HashMap::new();
        for i in 0..n_rows {
            let k = fold[i] as usize;
            let fold_start = &fold_params[k * 34..(k + 1) * 34];
            if model.as_ref().map(|(f, _)| *f) != Some(k) {
                model = Some((k, InstantFsrs7::new(fold_start, config)));
            }
            let begin = offsets[card[i] as usize] as usize;
            let history = &hist[begin..begin + prefix[i] as usize];
            let recalled = y[i] == 1;
            let (_, m) = model.as_mut().unwrap();
            if !with_offset && !curve {
                preds.push(m.observe(history, delta_t[i], recalled));
                continue;
            }
            let again = (1.0 + history.iter().filter(|r| r.rating == 1).count() as f64).ln();
            let cb = card_bias.get(&card[i]).copied().unwrap_or(0.0);
            let nb = if note[i] >= 0 {
                note_bias.get(&note[i]).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            // Scheduling-time prediction: parameters and offsets as they were right after
            // this card's previous review (the fold's starting fit if none in the stream).
            let scheduled = curve.then(|| match snapshots.get(&card[i]) {
                Some(s) => with_logit_offset(
                    InstantFsrs7::predict_with(&s.parameters, history, delta_t[i]),
                    s.user_bias + s.card_bias + s.note_bias + s.again_weight * again,
                ),
                None => InstantFsrs7::predict_with(fold_start, history, delta_t[i]),
            });
            let offset = user_bias + cb + nb + again_weight * again;
            let p = m.observe_with_offset(history, delta_t[i], recalled, offset);
            let err = p - y[i] as f64;
            user_bias -= user_lr * err;
            card_bias.insert(card[i], cb - card_lr * err);
            if note[i] >= 0 {
                note_bias.insert(note[i], nb - note_lr * err);
            }
            again_weight -= again_lr * err * again;
            if curve {
                snapshots.insert(
                    card[i],
                    Snapshot {
                        parameters: m.parameters().to_vec(),
                        user_bias,
                        card_bias: cb - card_lr * err,
                        note_bias: if note[i] >= 0 {
                            nb - note_lr * err
                        } else {
                            0.0
                        },
                        again_weight,
                    },
                );
            }
            preds.push(scheduled.unwrap_or(p));
        }
        secs.push(start.elapsed().as_secs_f64());
    }

    let mut out = Vec::with_capacity((preds.len() + secs.len()) * 8);
    for v in preds.iter().chain(&secs) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&args[2], out).expect("write output");
}
