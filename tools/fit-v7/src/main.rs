//! Fits FSRS-7 with fsrs-rs's own trainer (`fsrs::benchmark`, the path the srs-benchmark
//! "FSRS-rs" row uses) on one training set, for timing and accuracy checks from
//! srs-benchmark.
//!
//! usage: fit-v7 <in.bin> <out.bin>
//!
//! in.bin (little-endian): i32 n_items, i32 n_reviews, i64 offsets[n_items + 1],
//! f32 delta_t[n_reviews], i32 rating[n_reviews], i64 card_id[n_items]. Item k is
//! reviews offsets[k]..offsets[k + 1], in training order (review_th), its last review the
//! one being predicted.
//!
//! out.bin: f32 parameters[34], f64 fit seconds.

use std::fs;
use std::time::Instant;

use fsrs::{ComputeParametersInput, ComputeParametersVersion, FSRSItem, FSRSReview, benchmark};

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> &'a [u8] {
    let s = &bytes[*pos..*pos + n];
    *pos += n;
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).expect("read input");
    let mut pos = 0;
    let i32s = |pos: &mut usize, n: usize| -> Vec<i32> {
        take(&bytes, pos, 4 * n)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let head = i32s(&mut pos, 2);
    let (n_items, n_reviews) = (head[0] as usize, head[1] as usize);
    let i64s = |pos: &mut usize, n: usize| -> Vec<i64> {
        take(&bytes, pos, 8 * n)
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let offsets = i64s(&mut pos, n_items + 1);
    let delta_t: Vec<f32> = take(&bytes, &mut pos, 4 * n_reviews)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let rating = i32s(&mut pos, n_reviews);
    let card_ids = i64s(&mut pos, n_items);
    assert_eq!(pos, bytes.len(), "trailing input");

    let train_set: Vec<FSRSItem> = (0..n_items)
        .map(|k| FSRSItem {
            reviews: (offsets[k] as usize..offsets[k + 1] as usize)
                .map(|j| FSRSReview {
                    rating: rating[j] as u32,
                    delta_t: delta_t[j],
                })
                .collect(),
        })
        .collect();
    let t0 = Instant::now();
    let params = benchmark(ComputeParametersInput {
        train_set,
        card_ids: Some(card_ids),
        enable_short_term: true,
        model_version: ComputeParametersVersion::Fsrs7,
        ..Default::default()
    });
    let secs = t0.elapsed().as_secs_f64();
    let mut out: Vec<u8> = params.iter().flat_map(|x| x.to_le_bytes()).collect();
    out.extend(secs.to_le_bytes());
    fs::write(&args[2], out).expect("write output");
}
