//! Integration tests for the `ferray` umbrella crate.
//!
//! These live under `tests/` rather than `src/*/mod.rs` so they
//! exercise the crate's public API the same way a downstream user
//! would — they get no access to crate-private items. See issue #330
//! for the rationale: the umbrella previously had only in-line unit
//! tests, which masked missing re-exports because the unit tests
//! could still reach them via `super`.

// These integration tests assert exact float equality on hand-picked
// inputs (constants, identities, fixed FFT outputs) by design.
#![allow(clippy::float_cmp)]

use ferray::prelude::*;

#[test]
fn prelude_exposes_core_types_and_creation() {
    let a = zeros::<f64, Ix2>(Ix2::new([2, 3])).unwrap();
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.size(), 6);
}

#[test]
fn prelude_ufuncs_are_callable() {
    let a = ones::<f64, Ix1>(Ix1::new([4])).unwrap();
    let s = sin(&a).unwrap();
    assert_eq!(s.shape(), &[4]);
}

#[test]
fn prelude_stats_reductions_work() {
    let a: ferray::Array<f64, Ix1> =
        ferray::Array::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m = mean(&a, None).unwrap();
    let m_val = m.iter().next().copied().unwrap();
    assert!((m_val - 2.5).abs() < 1e-12);
}

#[test]
fn complex_is_re_exported_via_prelude() {
    // Issue #333: Complex<T> should be usable from the prelude
    // without a direct num-complex dependency.
    let c = Complex::<f64>::new(1.0, 2.0);
    assert_eq!(c.re, 1.0);
    assert_eq!(c.im, 2.0);
}

#[test]
fn bitwise_count_is_re_exported() {
    // Issue #396 ufunc — verify it's visible from the umbrella.
    let a = ferray::Array::<u32, Ix1>::from_vec(Ix1::new([3]), vec![0u32, 7, 255]).unwrap();
    let r = bitwise_count(&a).unwrap();
    assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![0u32, 3, 8]);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn threshold_constants_are_accessible() {
    assert!(ferray::threshold::PARALLEL_THRESHOLD_ELEMENTWISE > 0);
    assert!(ferray::threshold::PARALLEL_THRESHOLD_COMPUTE > 0);
    assert!(ferray::threshold::PARALLEL_THRESHOLD_REDUCTION > 0);
    assert!(ferray::threshold::PARALLEL_THRESHOLD_SORT > 0);
}

#[test]
fn with_num_threads_returns_ferray_result() {
    // Issue #218: set_num_threads / with_num_threads used to return
    // `Result<_, String>`; they now return `FerrayResult<_>` so
    // callers can use `?` with the rest of the ferray API.
    let r: FerrayResult<i32> = ferray::config::with_num_threads(2, || 42);
    assert_eq!(r.unwrap(), 42);
}

#[cfg(feature = "fft")]
#[test]
fn fft_submodule_exposed() {
    // Issue #330: pick a representative optional module and verify
    // the re-export works under the feature flag.
    use ferray::fft::{FftNorm, fft};
    let data: Vec<num_complex::Complex<f64>> = (0..8)
        .map(|i| num_complex::Complex::new(f64::from(i), 0.0))
        .collect();
    let a = ferray::Array::<num_complex::Complex<f64>, Ix1>::from_vec(Ix1::new([8]), data).unwrap();
    let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
    assert_eq!(spectrum.shape(), &[8]);
}
