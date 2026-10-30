//! Replays one user's review stream through FSRS-Instant for srs-benchmark.
//!
//! `instant_replay <input.bin> <output.bin>`. All values are little-endian.
//!
//! Input:
//!   i32 n_folds, n_cards, n_rows, n_configs, n_hist
//!   f32 [n_folds * 34]        fold parameters (the benchmark's own per-fold fit)
//!   f64 [n_configs * 5]       lr, beta1, beta2, eps, l2
//!   i64 [n_cards + 1]         offsets of each card's full history in hist_*
//!   f32 [n_hist], i32 [n_hist] hist delta_t (days), rating
//!   i32 [n_rows] card, i32 [n_rows] prefix_len, f32 [n_rows] delta_t, i32 [n_rows] y,
//!   i32 [n_rows] fold (rows sorted by time; a new fold restarts from its parameters)
//! Output: f64 [n_configs * n_rows] predictions, then f64 [n_configs] seconds spent.
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
    let configs = rd.f64s(n_configs * 5);
    let offsets = rd.i64s(n_cards + 1);
    let hist_t = rd.f32s(n_hist);
    let hist_r = rd.i32s(n_hist);
    let card = rd.i32s(n_rows);
    let prefix = rd.i32s(n_rows);
    let delta_t = rd.f32s(n_rows);
    let y = rd.i32s(n_rows);
    let fold = rd.i32s(n_rows);
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
    for c in configs.chunks_exact(5) {
        let config = InstantConfig {
            lr: c[0],
            beta1: c[1],
            beta2: c[2],
            eps: c[3],
            l2: c[4],
        };
        let start = Instant::now();
        let mut model: Option<(i32, InstantFsrs7)> = None;
        for i in 0..n_rows {
            let k = fold[i];
            if model.as_ref().map(|(f, _)| *f) != Some(k) {
                let p = &fold_params[k as usize * 34..(k as usize + 1) * 34];
                model = Some((k, InstantFsrs7::new(p, config)));
            }
            let begin = offsets[card[i] as usize] as usize;
            let history = &hist[begin..begin + prefix[i] as usize];
            let (_, m) = model.as_mut().unwrap();
            preds.push(m.observe(history, delta_t[i], y[i] == 1));
        }
        secs.push(start.elapsed().as_secs_f64());
    }

    let mut out = Vec::with_capacity((preds.len() + secs.len()) * 8);
    for v in preds.iter().chain(&secs) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&args[2], out).expect("write output");
}
