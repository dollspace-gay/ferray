//! Divergence pins: `ferray_polynomial` LIBRARY (Rust API) vs `numpy.polynomial`.
//!
//! Audit target: the `Poly::roots()` contract on `ferray_polynomial::Polynomial`
//! against `numpy.polynomial.polynomial.polyroots`.
//!
//! ## Upstream contract (numpy 2.4.5)
//!
//! `numpy/polynomial/polynomial.py:1601-1607` (`polyroots`):
//! ```text
//!     m = polycompanion(c)
//!     r = np.linalg.eigvals(m)
//!     r.sort()                          # <-- line 1603: roots SORTED ascending
//!     from numpy.linalg._linalg import _to_real_if_imag_zero
//!     r = _to_real_if_imag_zero(r, m)   # <-- lines 1606-1607: REAL dtype if all imag==0
//! ```
//! `np.ndarray.sort()` on a complex array sorts lexicographically by
//! (real, then imag) — confirmed by live oracle (see per-test expected values,
//! all generated from `numpy.polynomial.polynomial.polyroots`).
//!
//! ferray's `Polynomial::roots()` -> `find_roots_from_power_coeffs` (in
//! `src/roots.rs`) returns the faer companion-eigenvalue order with Newton
//! polishing applied IN PLACE and performs **no final sort**. The order is
//! therefore the raw eigensolver order, which diverges from numpy's sorted
//! order for several real and complex inputs below.
//!
//! NOTE on the REAL-vs-COMPLEX dtype distinction: numpy returns a real
//! `float64` array when every imaginary part is exactly zero. ferray's Rust
//! API signature is fixed as `Result<Vec<Complex<f64>>, FerrayError>`, so the
//! "return a real-dtype array" half of `polyroots` is a *binding-layer*
//! concern (the ferray-python wrapper must down-convert `Vec<Complex<f64>>`
//! to a real numpy array when all imag parts vanish). The **ordering** half is
//! a pure LIBRARY divergence and is what these tests pin.

use ferray_polynomial::{Poly, Polynomial};
use num_complex::Complex;

/// numpy-equivalent root ordering: sort by (re, then im), matching
/// `np.ndarray.sort()` on a complex array as used at `polynomial.py:1603`.
fn numpy_sort(mut v: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    v.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    v
}

fn approx_eq(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol)
}

/// Divergence: `Polynomial::roots()` does not sort the returned roots, so for
/// three negative real roots it returns descending order while numpy returns
/// ascending order.
///
/// Upstream `numpy/polynomial/polynomial.py:1603` (`r.sort()`):
/// `polyroots([10, 17, 8, 1])` (== `(x+1)(x+2)(x+5)`) returns
/// `array([-5., -2., -1.])` (ascending, real float64).
/// ferray returns `[-1, -2, -5]` (descending companion order).
/// Tracking: divergence_roots_ordering (library).
#[test]
fn divergence_roots_real_ascending_order() {
    // (x+1)(x+2)(x+5) = x^3 + 8x^2 + 17x + 10
    let p = Polynomial::new(&[10.0, 17.0, 8.0, 1.0]);
    let got = p.roots().unwrap();

    // Expected order from live numpy: polyroots([10,17,8,1]) -> [-5,-2,-1].
    let expected = vec![
        Complex::new(-5.0, 0.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-1.0, 0.0),
    ];

    assert!(
        approx_eq(&got, &expected, 1e-9),
        "roots() returned {got:?}, numpy.polyroots returns sorted-ascending {expected:?}"
    );
}

/// Divergence: complex-conjugate roots are returned in eigensolver order
/// (`+i` before `-i`), whereas numpy sorts so `-i` precedes `+i`.
///
/// Upstream `numpy/polynomial/polynomial.py:1603`:
/// `polyroots([-5, 1, -5, 1])` (== `(x^2+1)(x-5)`) returns
/// `array([0.-1.j, 0.+1.j, 5.+0.j])` (sorted: -i before +i).
/// ferray returns `[0+1j, 0-1j, 5+0j]`.
/// Tracking: divergence_roots_ordering (library).
#[test]
fn divergence_roots_complex_conjugate_order() {
    // (x^2 + 1)(x - 5) = x^3 - 5x^2 + x - 5
    let p = Polynomial::new(&[-5.0, 1.0, -5.0, 1.0]);
    let got = p.roots().unwrap();

    // Expected from live numpy: polyroots([-5,1,-5,1]) -> [0-1j, 0+1j, 5+0j].
    let expected = vec![
        Complex::new(0.0, -1.0),
        Complex::new(0.0, 1.0),
        Complex::new(5.0, 0.0),
    ];

    assert!(
        approx_eq(&got, &expected, 1e-9),
        "roots() returned {got:?}, numpy.polyroots returns sorted {expected:?}"
    );
}

/// Divergence: with a complex-conjugate pair and a separated real root, the
/// real root is placed LAST by ferray but FIRST by numpy's sort.
///
/// Upstream `numpy/polynomial/polynomial.py:1603`:
/// `polyroots([15, -1, 1, 1])` (== `(x^2-2x+5)(x+3)`) returns
/// `array([-3.+0.j, 1.-2.j, 1.+2.j])` (sorted by re then im).
/// ferray returns `[1+2j, 1-2j, -3+0j]`.
/// Tracking: divergence_roots_ordering (library).
#[test]
fn divergence_roots_mixed_real_complex_order() {
    // (x^2 - 2x + 5)(x + 3) = x^3 + x^2 - x + 15
    let p = Polynomial::new(&[15.0, -1.0, 1.0, 1.0]);
    let got = p.roots().unwrap();

    // Expected from live numpy: polyroots([15,-1,1,1]) -> [-3+0j, 1-2j, 1+2j].
    let expected = vec![
        Complex::new(-3.0, 0.0),
        Complex::new(1.0, -2.0),
        Complex::new(1.0, 2.0),
    ];

    assert!(
        approx_eq(&got, &expected, 1e-9),
        "roots() returned {got:?}, numpy.polyroots returns sorted {expected:?}"
    );
}

// ---------------------------------------------------------------------------
// Non-divergence guards: value-correctness checks that PASS against numpy.
// These are NOT divergences (they pass today); they document the audited
// surface so a future regression in the value math would be caught here.
// Expected values are from live numpy 2.4.5 oracle.
// ---------------------------------------------------------------------------

/// numpy `polyroots` is *order-insensitive as a set* for the all-real cubic
/// (x-1)(x-2)(x-3); ferray already matches the multiset of roots. (Audited:
/// no divergence in root VALUES, only in ORDER — pinned above.)
#[test]
fn roots_value_set_matches_numpy() {
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6  -> coeffs ascending [-6,11,-6,1]
    let p = Polynomial::new(&[-6.0, 11.0, -6.0, 1.0]);
    let got = numpy_sort(p.roots().unwrap());
    // numpy polyroots([-6,11,-6,1]) -> [1, 2, 3] (real, sorted).
    let expected = vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
    ];
    assert!(
        approx_eq(&got, &expected, 1e-9),
        "root multiset {got:?} != numpy {expected:?}"
    );
}

/// `Poly::eval` matches numpy `polyval`. (Audited: no divergence.)
#[test]
fn eval_matches_numpy() {
    // p = 2 - 3x + x^2 ; polyval(5, [2,-3,1]) == 12.0
    let p = Polynomial::new(&[2.0, -3.0, 1.0]);
    assert!((p.eval(5.0).unwrap() - 12.0).abs() < 1e-12);
}

/// `Poly::deriv` matches numpy `polyder`. (Audited: no divergence.)
#[test]
fn deriv_matches_numpy() {
    // polyder([1,2,3,4]) == [2, 6, 12]
    let p = Polynomial::new(&[1.0, 2.0, 3.0, 4.0]);
    let d = p.deriv(1).unwrap();
    let expected = [2.0, 6.0, 12.0];
    assert_eq!(d.coeffs().len(), expected.len());
    for (g, e) in d.coeffs().iter().zip(expected) {
        assert!((g - e).abs() < 1e-12, "deriv {g} != {e}");
    }
}

/// `Poly::integ` matches numpy `polyint`. (Audited: no divergence.)
#[test]
fn integ_matches_numpy() {
    // polyint([1,2,3]) == [0, 1, 1, 1]
    let p = Polynomial::new(&[1.0, 2.0, 3.0]);
    let i = p.integ(1, &[0.0]).unwrap();
    let expected = [0.0, 1.0, 1.0, 1.0];
    assert_eq!(i.coeffs().len(), expected.len());
    for (g, e) in i.coeffs().iter().zip(expected) {
        assert!((g - e).abs() < 1e-12, "integ {g} != {e}");
    }
}

/// `Poly::mul` matches numpy `polymul`. (Audited: no divergence.)
#[test]
fn mul_matches_numpy() {
    // polymul([1,2,3],[3,2,1]) == [3, 8, 14, 8, 3]
    let a = Polynomial::new(&[1.0, 2.0, 3.0]);
    let b = Polynomial::new(&[3.0, 2.0, 1.0]);
    let m = a.mul(&b).unwrap();
    let expected = [3.0, 8.0, 14.0, 8.0, 3.0];
    assert_eq!(m.coeffs().len(), expected.len());
    for (g, e) in m.coeffs().iter().zip(expected) {
        assert!((g - e).abs() < 1e-12, "mul {g} != {e}");
    }
}
