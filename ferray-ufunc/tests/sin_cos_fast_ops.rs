//! Accuracy + dispatch tests for the public `sin_fast` / `cos_fast` ops.
use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_ufunc::{cos, cos_fast, sin, sin_fast};

#[test]
fn sin_fast_f64_matches_sin_over_array() {
    let n = 10_000;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 5.0).collect();
    let a = Array::from_vec(Ix1::new([n]), data).unwrap();
    let fast = sin_fast(&a).unwrap();
    let reference = sin(&a).unwrap();
    for (f, r) in fast.iter().zip(reference.iter()) {
        assert!((f - r).abs() < 1e-12, "sin_fast {f} vs sin {r}");
    }
}

#[test]
fn cos_fast_f64_matches_cos_over_array() {
    let n = 10_000;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 5.0).collect();
    let a = Array::from_vec(Ix1::new([n]), data).unwrap();
    let fast = cos_fast(&a).unwrap();
    let reference = cos(&a).unwrap();
    for (f, r) in fast.iter().zip(reference.iter()) {
        assert!((f - r).abs() < 1e-12, "cos_fast {f} vs cos {r}");
    }
}

#[test]
fn sin_cos_fast_f32_matches_over_array() {
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 5.0).collect();
    let a = Array::from_vec(Ix1::new([n]), data).unwrap();
    let sf = sin_fast(&a).unwrap();
    let sr = sin(&a).unwrap();
    let cf = cos_fast(&a).unwrap();
    let cr = cos(&a).unwrap();
    for (f, r) in sf.iter().zip(sr.iter()) {
        assert!((f - r).abs() < 1e-6, "sin_fast f32 {f} vs sin {r}");
    }
    for (f, r) in cf.iter().zip(cr.iter()) {
        assert!((f - r).abs() < 1e-6, "cos_fast f32 {f} vs cos {r}");
    }
}

#[test]
fn sin_fast_noncontiguous_correct() {
    // Transposed 2D array is non-contiguous → exercises the iter() fallback.
    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.3).collect();
    let a = Array::from_vec(Ix2::new([3, 4]), data).unwrap();
    let at = a.t().to_owned();
    let fast = sin_fast(&at).unwrap();
    let reference = sin(&at).unwrap();
    for (f, r) in fast.iter().zip(reference.iter()) {
        assert!((f - r).abs() < 1e-12, "sin_fast 2d {f} vs {r}");
    }
}
