//! Cross-basis divmod accuracy tests (#254).
//!
//! For non-power bases (Chebyshev, Legendre, Hermite, HermiteE, Laguerre)
//! the divmod implementation routes through a power-basis pivot:
//!   a_basis -> a_power -> divmod_power -> q_power, r_power -> q_basis, r_basis
//! Round-trip conversion errors could compound silently. These tests pin
//! the algebraic invariant `a == q * b + r` and the eval consistency
//! across the whole pipeline at multiple input sizes.

use ferray_polynomial::traits::Poly;
use ferray_polynomial::{Chebyshev, Hermite, HermiteE, Laguerre, Legendre};

const EVAL_TOL: f64 = 1e-8;

/// Verify the divmod invariant via evaluation: a(x) == q(x) * b(x) + r(x)
/// at several test points.
fn check_divmod_invariant<P: Poly + Clone>(
    a: &P,
    b: &P,
    q: &P,
    r: &P,
    test_points: &[f64],
    label: &str,
) {
    for &x in test_points {
        let av = a.eval(x).unwrap();
        let bv = b.eval(x).unwrap();
        let qv = q.eval(x).unwrap();
        let rv = r.eval(x).unwrap();
        let recon = qv.mul_add(bv, rv);
        let scale = av.abs().max(1.0);
        assert!(
            (recon - av).abs() / scale < EVAL_TOL,
            "{label} at x={x}: a={av}, q*b+r={recon}, diff={}",
            recon - av
        );
    }
}

fn test_points() -> Vec<f64> {
    vec![-0.7, -0.3, 0.0, 0.25, 0.5, 0.9]
}

#[test]
fn chebyshev_divmod_invariant() {
    let a = Chebyshev::new(&[1.0, 2.0, 3.0, 4.0, 5.0]); // degree 4
    let b = Chebyshev::new(&[1.0, 0.0, 1.0]); // degree 2
    let (q, r) = a.divmod(&b).unwrap();
    check_divmod_invariant(&a, &b, &q, &r, &test_points(), "chebyshev");
}

#[test]
fn legendre_divmod_invariant() {
    let a = Legendre::new(&[1.0, 2.0, 3.0, 4.0]);
    let b = Legendre::new(&[1.0, 1.0]);
    let (q, r) = a.divmod(&b).unwrap();
    check_divmod_invariant(&a, &b, &q, &r, &test_points(), "legendre");
}

#[test]
fn hermite_divmod_invariant() {
    let a = Hermite::new(&[1.0, 2.0, 3.0, 4.0]);
    let b = Hermite::new(&[1.0, 1.0]);
    let (q, r) = a.divmod(&b).unwrap();
    check_divmod_invariant(&a, &b, &q, &r, &test_points(), "hermite");
}

#[test]
fn hermite_e_divmod_invariant() {
    let a = HermiteE::new(&[1.0, 2.0, 3.0, 4.0]);
    let b = HermiteE::new(&[1.0, 1.0]);
    let (q, r) = a.divmod(&b).unwrap();
    check_divmod_invariant(&a, &b, &q, &r, &test_points(), "hermite_e");
}

#[test]
fn laguerre_divmod_invariant() {
    let a = Laguerre::new(&[1.0, 2.0, 3.0]);
    let b = Laguerre::new(&[1.0, 1.0]);
    let (q, r) = a.divmod(&b).unwrap();
    // Laguerre default domain is [0, 1]; eval inside the canonical
    // window where the conversion stays well-conditioned.
    let pts = vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
    check_divmod_invariant(&a, &b, &q, &r, &pts, "laguerre");
}

// Higher-degree divisor: quotient must be zero, remainder must equal a.
#[test]
fn chebyshev_divmod_higher_degree_divisor() {
    let a = Chebyshev::new(&[1.0, 2.0]); // degree 1
    let b = Chebyshev::new(&[1.0, 1.0, 1.0]); // degree 2
    let (q, r) = a.divmod(&b).unwrap();
    // Quotient should be zero (or near-zero); remainder is a.
    for &x in test_points().iter() {
        let qv = q.eval(x).unwrap();
        assert!(qv.abs() < 1e-12, "q at x={x} = {qv}, expected 0");
        let rv = r.eval(x).unwrap();
        let av = a.eval(x).unwrap();
        assert!((rv - av).abs() < 1e-10, "r at x={x} = {rv}, expected a={av}");
    }
}

#[test]
fn legendre_divmod_higher_degree_divisor() {
    let a = Legendre::new(&[1.0, 2.0]);
    let b = Legendre::new(&[1.0, 0.0, 1.0]);
    let (q, r) = a.divmod(&b).unwrap();
    for &x in test_points().iter() {
        assert!(q.eval(x).unwrap().abs() < 1e-12);
        assert!((r.eval(x).unwrap() - a.eval(x).unwrap()).abs() < 1e-10);
    }
}

// Exact division: a = (x + 1) * (x - 1) = x^2 - 1, b = x - 1, q should
// be x + 1, r should be 0. Use power-basis-equivalent expressions.
#[test]
fn chebyshev_divmod_exact_division_zero_remainder() {
    // p(x) = x^2 - 1 in Chebyshev basis: x^2 = (T_0 + T_2)/2,
    // so x^2 - 1 = -T_0/2 + T_2/2 = [-0.5, 0, 0.5].
    let a = Chebyshev::new(&[-0.5, 0.0, 0.5]);
    // b(x) = x - 1 = T_1 - T_0 = [-1, 1].
    let b = Chebyshev::new(&[-1.0, 1.0]);
    let (_q, r) = a.divmod(&b).unwrap();
    // r should be ~0 at all points.
    for &x in test_points().iter() {
        assert!(
            r.eval(x).unwrap().abs() < 1e-10,
            "remainder at x={x} = {}, expected 0",
            r.eval(x).unwrap()
        );
    }
}

// Division by the zero polynomial must error.
#[test]
fn chebyshev_divmod_by_zero_errs() {
    let a = Chebyshev::new(&[1.0, 2.0, 3.0]);
    let zero = Chebyshev::new(&[0.0]);
    assert!(a.divmod(&zero).is_err());
}

#[test]
fn legendre_divmod_by_zero_errs() {
    let a = Legendre::new(&[1.0, 2.0, 3.0]);
    let zero = Legendre::new(&[0.0]);
    assert!(a.divmod(&zero).is_err());
}

#[test]
fn hermite_divmod_by_zero_errs() {
    let a = Hermite::new(&[1.0, 2.0]);
    let zero = Hermite::new(&[0.0]);
    assert!(a.divmod(&zero).is_err());
}

#[test]
fn hermite_e_divmod_by_zero_errs() {
    let a = HermiteE::new(&[1.0, 2.0]);
    let zero = HermiteE::new(&[0.0]);
    assert!(a.divmod(&zero).is_err());
}

#[test]
fn laguerre_divmod_by_zero_errs() {
    let a = Laguerre::new(&[1.0, 2.0]);
    let zero = Laguerre::new(&[0.0]);
    assert!(a.divmod(&zero).is_err());
}
