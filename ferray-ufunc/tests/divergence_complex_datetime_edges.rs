//! Divergence pins: `ferray_ufunc::abs` (complex magnitude) computes the
//! magnitude as `(re*re + im*im).sqrt()` (ferray-ufunc/src/ops/complex.rs:79-83)
//! instead of `hypot(re, im)`. NumPy's `np.abs` on a complex dtype uses the C99
//! `npy_cabs`/`hypot` path, which is overflow-/underflow-safe AND honors the
//! C99 hypot special-value rule `hypot(±inf, anything) == +inf` (even when the
//! other component is NaN). The naive sum-of-squares diverges from numpy 2.4.4
//! in three observable ways:
//!
//!   * OVERFLOW: a finite magnitude whose squared components overflow the
//!     exponent range -> ferray returns `inf`, numpy returns the true value.
//!   * UNDERFLOW: tiny components whose squares flush to 0 -> ferray returns
//!     `0.0`, numpy returns the true (tiny but non-zero) magnitude.
//!   * SPECIAL VALUES: `abs(inf + NaN*i)` -> ferray's `(inf + NaN).sqrt()` is
//!     `NaN`, numpy (C99 hypot) returns `+inf`.
//!
//! Upstream cite: numpy registers `absolute` for the complex loops via
//! `generate_umath.py` (`absolute` `TD(cmplx, ...)` -> real-magnitude loop),
//! whose C kernel `CDOUBLE_absolute` calls `npy_cabs` == `npy_hypot`
//! (`numpy/_core/src/npymath/npy_math_complex.c.src`, `npy_cabs`), giving the
//! C99 overflow-safe magnitude. The naive `sqrt(re*re+im*im)` is exactly the
//! formula C99 hypot is specified to avoid.
//!
//! All expected values below are LIVE numpy 2.4.4 oracle results (R-CHAR-3),
//! gathered via `np.abs(np.array([...]))`:
//!   np.abs(1e200+1e200j)   == 1.414213562373095e+200
//!   np.abs(1e-200+1e-200j) == 1.414213562373095e-200
//!   np.abs(inf + nan*1j)   == inf
//!   np.abs(nan + inf*1j)   == inf
//!   np.abs(complex64(1e30+1e30j)) == 1.4142135130433894e+30
//! and the preserved-invariant baseline:
//!   np.abs(3+4j) == 5.0
//!
//! These are numerical-contract (special-value) divergences, NOT
//! within-tolerance float noise (S8) — they are infinite/zero/NaN vs finite,
//! so they are left un-`#[ignore]`d as release-blockers.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use num_complex::{Complex32, Complex64};

fn c64(data: Vec<Complex64>) -> Array<Complex64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

fn c32(data: Vec<Complex32>) -> Array<Complex32, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

/// Preserved invariant (NOT a divergence): `abs(3+4j) == 5.0` exactly. The
/// hypot fix must keep the ordinary case byte-identical. Expected = live
/// `np.abs(3+4j) == 5.0`.
#[test]
fn baseline_abs_complex_matches_numpy() {
    let out = ferray_ufunc::abs(&c64(vec![Complex64::new(3.0, 4.0)])).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[5.0_f64]);
}

/// Divergence: `abs(1e200 + 1e200j)`.
/// numpy (hypot) -> 1.414213562373095e+200; ferray (`(re*re+im*im).sqrt()`)
/// -> `inf` because `1e200 * 1e200 == 1e400` overflows f64 to `+inf`.
/// Upstream: `npy_cabs == npy_hypot` (npy_math_complex.c.src), overflow-safe.
#[test]
fn divergence_abs_complex_overflow_f64() {
    // numpy 2.4.4: np.abs(np.array([1e200+1e200j]))[0]
    const NUMPY_ABS_1E200: f64 = 1.414_213_562_373_095e200;
    let out = ferray_ufunc::abs(&c64(vec![Complex64::new(1e200, 1e200)])).unwrap();
    let v = out.as_slice().unwrap()[0];
    assert!(
        v.is_finite(),
        "ferray abs overflowed to {v:?}; numpy yields the finite {NUMPY_ABS_1E200:e}"
    );
    assert_eq!(v, NUMPY_ABS_1E200);
}

/// Divergence: `abs(1e-200 + 1e-200j)`.
/// numpy (hypot) -> 1.414213562373095e-200; ferray -> `0.0` because
/// `1e-200 * 1e-200 == 1e-400` underflows f64 to `0.0`.
#[test]
fn divergence_abs_complex_underflow_f64() {
    // numpy 2.4.4: np.abs(np.array([1e-200+1e-200j]))[0]
    const NUMPY_ABS_1EM200: f64 = 1.414_213_562_373_095e-200;
    let out = ferray_ufunc::abs(&c64(vec![Complex64::new(1e-200, 1e-200)])).unwrap();
    let v = out.as_slice().unwrap()[0];
    assert!(
        v != 0.0,
        "ferray abs underflowed to {v:?}; numpy yields the non-zero {NUMPY_ABS_1EM200:e}"
    );
    assert_eq!(v, NUMPY_ABS_1EM200);
}

/// Divergence: `abs(inf + NaN*i)`.
/// numpy (C99 hypot: `hypot(±inf, x) == +inf` for any x, incl. NaN) -> `inf`;
/// ferray -> `(inf*inf + NaN*NaN).sqrt()` == `(inf + NaN).sqrt()` ==
/// `NaN.sqrt()` == `NaN`.
#[test]
fn divergence_abs_complex_inf_nan_is_inf() {
    let out = ferray_ufunc::abs(&c64(vec![Complex64::new(f64::INFINITY, f64::NAN)])).unwrap();
    let v = out.as_slice().unwrap()[0];
    // numpy 2.4.4: np.abs(np.array([complex(inf, nan)]))[0] == inf
    assert!(
        v.is_infinite() && v > 0.0,
        "ferray abs(inf+NaNj) = {v:?}; numpy (C99 hypot) yields +inf"
    );
}

/// Divergence: `abs(NaN + inf*i)`. Symmetric to the above:
/// numpy (C99 hypot) -> `+inf`; ferray -> `NaN`.
#[test]
fn divergence_abs_complex_nan_inf_is_inf() {
    let out = ferray_ufunc::abs(&c64(vec![Complex64::new(f64::NAN, f64::INFINITY)])).unwrap();
    let v = out.as_slice().unwrap()[0];
    // numpy 2.4.4: np.abs(np.array([complex(nan, inf)]))[0] == inf
    assert!(
        v.is_infinite() && v > 0.0,
        "ferray abs(NaN+infj) = {v:?}; numpy (C99 hypot) yields +inf"
    );
}

/// Divergence (f32 / complex64 dtype): `abs(1e30 + 1e30j)` in single
/// precision. `1e30 * 1e30 == 1e60` overflows f32 (max ~3.4e38) to `inf`;
/// numpy's hypot path keeps it finite.
/// numpy 2.4.4: np.abs(np.array([np.complex64(1e30+1e30j)]))[0]
/// == 1.4142135130433894e+30.
#[test]
fn divergence_abs_complex_overflow_f32() {
    const NUMPY_ABS_1E30_F32: f32 = 1.414_213_5e30;
    let out = ferray_ufunc::abs(&c32(vec![Complex32::new(1e30, 1e30)])).unwrap();
    let v = out.as_slice().unwrap()[0];
    assert!(
        v.is_finite(),
        "ferray abs(f32 1e30+1e30j) overflowed to {v:?}; numpy keeps it finite"
    );
    assert_eq!(v, NUMPY_ABS_1E30_F32);
}
