//! ACToR critic divergence pins for `ferray_polynomial::roots`
//! (`find_roots_from_power_coeffs`, routed to numpy
//! `numpy/polynomial/polynomial.py::polyroots`, ascending-coefficient input).
//!
//! Oracle: installed numpy 2.4.x (`ferray-python/.venv/bin/python`), R-CHAR-3.
//! Expected values below are the live `numpy.polynomial.polynomial.polyroots`
//! outputs for the same ascending-order coefficient vectors.
//!
//! polyroots takes ascending coefficients (`c[i]` is the coefficient of
//! `x**i`), exactly like `find_roots_from_power_coeffs`, so the two are
//! directly comparable on the same input vector.
//!
//! Root SETS are compared after lexicographic sort (eigenvalue order is not
//! guaranteed identical across implementations); a tolerance band absorbs
//! ULP-level numerical noise.

use ferray_polynomial::roots::find_roots_from_power_coeffs;
use num_complex::Complex;

/// Sort a complex root vector lexicographically by (re, im) for set comparison.
fn sorted(mut roots: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    roots
}

/// Divergence: `ferray_polynomial::roots::find_roots_from_power_coeffs`
/// diverges from `numpy/polynomial/polynomial.py:1598-1607` (`polyroots`)
/// for the degree-2 polynomial `x**2` (ascending coeffs `[0, 0, 1]`, a
/// double root at the origin).
///
/// numpy oracle: `polyroots([0., 0., 1.]) -> array([0., 0.])` (dtype float64).
/// ferray returns `[(-0.0, 0.0), (NaN, 0.0)]`.
///
/// Root cause: the degree-2 branch (`roots.rs:60-80`) uses the
/// citardauq form `r2 = c / q`. For `c[0] == 0` the stabilizing factor
/// `q = -0.5 * (b + sign(b) * sqrt(disc))` collapses to `0`, so
/// `r2 = 0.0 / 0.0 = NaN` instead of the true second root `0`.
///
/// Tracking: #1085
#[test]
fn divergence_quadratic_double_root_at_origin() {
    // numpy: polyroots([0., 0., 1.]) == [0., 0.]
    let roots = find_roots_from_power_coeffs(&[0.0, 0.0, 1.0]).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots, got {roots:?}");
    let s = sorted(roots);
    for (i, r) in s.iter().enumerate() {
        assert!(
            r.re.is_finite() && r.im.is_finite(),
            "root {i} = {r:?} is non-finite; numpy returns finite 0",
        );
        assert!(
            r.re.abs() < 1e-12 && r.im.abs() < 1e-12,
            "root {i} = {r:?}; numpy polyroots([0,0,1]) returns 0",
        );
    }
}

/// Divergence: same NaN failure with a non-unit leading coefficient — the
/// quadratic `2*x**2` (ascending coeffs `[0, 0, 2]`).
///
/// numpy oracle: `polyroots([0., 0., 2.]) -> array([0., 0.])`.
/// ferray returns `[(-0.0, 0.0), (NaN, 0.0)]`.
///
/// Tracking: #1085
#[test]
fn divergence_quadratic_double_root_at_origin_scaled() {
    // numpy: polyroots([0., 0., 2.]) == [0., 0.]
    let roots = find_roots_from_power_coeffs(&[0.0, 0.0, 2.0]).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots, got {roots:?}");
    for (i, r) in roots.iter().enumerate() {
        assert!(
            r.re.is_finite() && r.im.is_finite(),
            "root {i} = {r:?} is non-finite; numpy returns finite 0",
        );
        assert!(
            r.norm() < 1e-12,
            "root {i} = {r:?}; numpy polyroots([0,0,2]) returns 0",
        );
    }
}

// ----- Coverage where ferray MATCHES the numpy oracle (S8) ----------------
// These are GREEN; they document the edge cases the critic verified against
// the live oracle and found NO divergence, so the pins above stand alone.

/// numpy: `polyroots([0., 1., -3., 2.]) -> array([0., 0.5, 1.])`.
/// A zero constant term (root at the origin) is preserved by the companion
/// path. Matches.
#[test]
fn coverage_zero_constant_term_root_at_origin() {
    let roots = find_roots_from_power_coeffs(&[0.0, 1.0, -3.0, 2.0]).unwrap();
    let s = sorted(roots);
    let expected = [0.0_f64, 0.5, 1.0];
    assert_eq!(s.len(), 3, "got {s:?}");
    for (i, (&e, r)) in expected.iter().zip(s.iter()).enumerate() {
        assert!(r.im.abs() < 1e-10, "root {i} unexpected imag {r:?}");
        assert!((r.re - e).abs() < 1e-9, "root {i} = {r:?}, expected {e}");
    }
}

/// numpy: `polyroots([0., -2., 1.]) -> array([0., 2.])` (x^2 - 2x). The
/// quadratic with a zero constant term but a NON-zero linear term avoids the
/// `c/q` NaN because `q != 0`. Matches.
#[test]
fn coverage_quadratic_zero_constant_nonzero_linear() {
    let roots = find_roots_from_power_coeffs(&[0.0, -2.0, 1.0]).unwrap();
    let s = sorted(roots);
    let expected = [0.0_f64, 2.0];
    assert_eq!(s.len(), 2, "got {s:?}");
    for (i, (&e, r)) in expected.iter().zip(s.iter()).enumerate() {
        assert!(r.im.abs() < 1e-12, "root {i} unexpected imag {r:?}");
        assert!((r.re - e).abs() < 1e-12, "root {i} = {r:?}, expected {e}");
    }
}

/// numpy: `polyroots([0., 0., 0.]) -> array([], dtype=float64)` (all-zero).
/// ferray returns an empty vector. Matches.
#[test]
fn coverage_all_zero_is_empty() {
    let roots = find_roots_from_power_coeffs(&[0.0, 0.0, 0.0]).unwrap();
    assert!(roots.is_empty(), "all-zero polynomial: got {roots:?}");
}

/// numpy: `polyroots([2., -3., 1., 0.]) -> array([1., 2.])` — a trailing
/// (highest-degree) zero coefficient is trimmed, dropping the spurious root.
/// Matches.
#[test]
fn coverage_trailing_high_degree_zero_trimmed() {
    let roots = find_roots_from_power_coeffs(&[2.0, -3.0, 1.0, 0.0]).unwrap();
    let s = sorted(roots);
    let expected = [1.0_f64, 2.0];
    assert_eq!(s.len(), 2, "got {s:?}");
    for (i, (&e, r)) in expected.iter().zip(s.iter()).enumerate() {
        assert!((r.re - e).abs() < 1e-10, "root {i} = {r:?}, expected {e}");
    }
}
