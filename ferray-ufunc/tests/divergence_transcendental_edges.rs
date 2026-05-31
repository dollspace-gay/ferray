//! ACToR critic divergence pins for the ferray-ufunc transcendental /
//! float-intrinsic kernels (`src/ops/trig.rs`, `src/ops/explog.rs`,
//! `src/ops/rounding.rs`, `src/ops/floatintrinsic.rs`).
//!
//! These pins cover EDGE cases the existing conformance suites
//! (`conformance_trig.rs`, `conformance_explog.rs`,
//! `conformance_floatintrinsic.rs`) do not exercise. All expected values are
//! live numpy 2.4.4 oracle results (R-CHAR-3); never copied from the ferray
//! side.
//!
//! ## Confirmed divergence
//!
//! `spacing(x)` for NEGATIVE finite `x` must carry the SIGN of `x`
//! (`np.spacing` returns `nextafter(x, +inf) - x`, which is negative when
//! `x < 0`). ferray's `spacing_f64`/`spacing_f32`
//! (`src/ops/floatintrinsic.rs:248`, `:261`) compute
//! `nextafter(x.abs(), +inf) - x.abs()`, discarding the sign, so they return a
//! POSITIVE result for every negative input. numpy's `spacing` ufunc
//! (`numpy/_core/code_generators/generate_umath.py` `spacing`, libm
//! `npy_spacing`) keeps the sign.
//!
//! ## No-divergence guards
//!
//! The remaining tests assert edge behavior that ferray ALREADY matches
//! (arcsin/arccos/arccosh/arctanh domain edges -> nan/inf, log(0)/log1p(-1)
//! specials, rint banker's signed-zero, copysign(nan), frexp(inf/nan)). They
//! are kept as regression guards — they PASS today and pin the contract.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_ufunc::{
    arccos, arccosh, arcsin, arctanh, copysign, frexp, log, log1p, rint, spacing,
};

fn a1(v: Vec<f64>) -> Array<f64, Ix1> {
    Array::from_vec(Ix1::new([v.len()]), v).unwrap()
}

// ---------------------------------------------------------------------------
// DIVERGENCE: spacing sign for negative inputs (release blocker).
// ---------------------------------------------------------------------------

/// Divergence: `ferray_ufunc::spacing` diverges from numpy `spacing`
/// (`numpy/_core/code_generators/generate_umath.py` `spacing`; libm
/// `npy_spacing`) for negative finite inputs.
///
/// ferray `src/ops/floatintrinsic.rs:248` computes
/// `nextafter(x.abs(), +inf) - x.abs()` (always >= 0). numpy keeps the sign of
/// `x`:
///   np.spacing(-1.0)    == -2.220446049250313e-16
///   np.spacing(-5.0)    == -8.881784197001252e-16
///   np.spacing(-1024.0) == -2.2737367544323206e-13
/// ferray returns +2.22e-16 / +8.88e-16 / +2.27e-13 (positive). Live numpy
/// 2.4.4 oracle.
#[test]
fn divergence_spacing_negative_sign_f64() {
    let r = spacing(&a1(vec![-1.0, -5.0, -1024.0])).unwrap();
    let s = r.as_slice().unwrap();
    assert_eq!(
        s[0].to_bits(),
        (-2.220446049250313e-16_f64).to_bits(),
        "spacing(-1.0) should equal numpy -2.220446049250313e-16, got {}",
        s[0]
    );
    assert_eq!(
        s[1].to_bits(),
        (-8.881784197001252e-16_f64).to_bits(),
        "spacing(-5.0) should equal numpy -8.881784197001252e-16, got {}",
        s[1]
    );
    assert_eq!(
        s[2].to_bits(),
        (-2.2737367544323206e-13_f64).to_bits(),
        "spacing(-1024.0) should equal numpy -2.2737367544323206e-13, got {}",
        s[2]
    );
    // All negative-input spacings must themselves be negative.
    assert!(s[0] < 0.0 && s[1] < 0.0 && s[2] < 0.0, "spacing of negatives must be negative");
}

/// Divergence: `ferray_ufunc::spacing` f32 path
/// (`src/ops/floatintrinsic.rs:261`) drops the sign of negative inputs.
///   np.spacing(np.float32(-1.0)) == -1.1920929e-07
///   np.spacing(np.float32(-5.0)) == -4.7683716e-07
/// Live numpy 2.4.4 oracle.
#[test]
fn divergence_spacing_negative_sign_f32() {
    let input: Array<f32, Ix1> =
        Array::from_vec(Ix1::new([2]), vec![-1.0_f32, -5.0]).unwrap();
    let r = spacing(&input).unwrap();
    let s = r.as_slice().unwrap();
    assert_eq!(
        s[0].to_bits(),
        (-1.1920929e-07_f32).to_bits(),
        "f32 spacing(-1.0) should equal numpy -1.1920929e-07, got {}",
        s[0]
    );
    assert_eq!(
        s[1].to_bits(),
        (-4.76837158203125e-07_f32).to_bits(),
        "f32 spacing(-5.0) should equal numpy -4.7683716e-07, got {}",
        s[1]
    );
}

// ---------------------------------------------------------------------------
// Regression guards — ferray ALREADY matches these (pass today).
// ---------------------------------------------------------------------------

/// arcsin/arccos of |x|>1 -> nan (RuntimeWarning, NOT error). Live numpy 2.4.4.
#[test]
fn guard_inverse_trig_domain_nan() {
    assert!(arcsin(&a1(vec![2.0])).unwrap().as_slice().unwrap()[0].is_nan());
    assert!(arccos(&a1(vec![2.0])).unwrap().as_slice().unwrap()[0].is_nan());
    assert!(arccosh(&a1(vec![0.5])).unwrap().as_slice().unwrap()[0].is_nan());
}

/// arctanh(±1)=±inf, arctanh(|x|>1)=nan. Live numpy 2.4.4.
#[test]
fn guard_arctanh_domain() {
    let r = arctanh(&a1(vec![1.0, -1.0, 2.0])).unwrap();
    let s = r.as_slice().unwrap();
    assert_eq!(s[0], f64::INFINITY);
    assert_eq!(s[1], f64::NEG_INFINITY);
    assert!(s[2].is_nan());
}

/// log(0)=-inf, log(-1)=nan, log1p(-1)=-inf, log1p(-2)=nan. Live numpy 2.4.4.
#[test]
fn guard_log_specials() {
    let lg = log(&a1(vec![0.0, -1.0])).unwrap();
    assert_eq!(lg.as_slice().unwrap()[0], f64::NEG_INFINITY);
    assert!(lg.as_slice().unwrap()[1].is_nan());
    let l1 = log1p(&a1(vec![-1.0, -2.0])).unwrap();
    assert_eq!(l1.as_slice().unwrap()[0], f64::NEG_INFINITY);
    assert!(l1.as_slice().unwrap()[1].is_nan());
}

/// rint(-0.5) -> -0.0 (signed zero, banker's). np.rint(-0.5) signbit True.
#[test]
fn guard_rint_signed_zero() {
    let r = rint(&a1(vec![-0.5])).unwrap();
    let v = r.as_slice().unwrap()[0];
    assert_eq!(v, 0.0);
    assert!(v.is_sign_negative(), "rint(-0.5) must be -0.0");
}

/// copysign(nan, -1.0) -> nan with sign bit set. np.signbit == True.
#[test]
fn guard_copysign_nan_sign() {
    let r = copysign(&a1(vec![f64::NAN]), &a1(vec![-1.0])).unwrap();
    let v = r.as_slice().unwrap()[0];
    assert!(v.is_nan() && v.is_sign_negative());
}

/// frexp(inf)=(inf,0), frexp(-inf)=(-inf,0), frexp(nan)=(nan,0). numpy 2.4.4.
#[test]
fn guard_frexp_nonfinite() {
    let (m, e) = frexp(&a1(vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN])).unwrap();
    let ms = m.as_slice().unwrap();
    let es = e.as_slice().unwrap();
    assert_eq!(ms[0], f64::INFINITY);
    assert_eq!(es[0], 0);
    assert_eq!(ms[1], f64::NEG_INFINITY);
    assert_eq!(es[1], 0);
    assert!(ms[2].is_nan());
    assert_eq!(es[2], 0);
}
