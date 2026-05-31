//! ACToR critic divergence pins for ferray-linalg COMPLEX kernels
//! (`ferray-linalg/src/complex.rs`) vs numpy 2.4.4
//! (`/home/doll/ferray/ferray-python/.venv`).
//!
//! Audited kernels: `matmul_complex`, `inv_complex`, `solve_complex`,
//! `solve_complex_vec`, `det_complex`, `eig_complex_f64`,
//! `eigvals_complex_f64`, `svd_complex_f64`, `qr_complex_f64`.
//!
//! NO DIVERGENCE found in: matmul (2x2 + non-trivial, conjugation handled by
//! faer), det (diag + general complex value matches numpy),
//! eig/eigvals (Av==λv, sum==trace, prod==det, sorted values match numpy),
//! svd (real nonneg descending s, U/Vh unitary, U diag(s) Vh == A,
//! s matches numpy), qr (Q^H Q == I, Q R == A) — all covered by
//! `complex.rs`'s `#[cfg(test)] mod tests`.
//!
//! DIVERGENCE found: `inv_complex` and `solve_complex` / `solve_complex_vec`
//! have NO singularity detection. The source explicitly admits this
//! (`complex.rs:249-253`: "Singularity detection would require calling
//! .is_nan() ... but we don't have a generic bound on T here"). For an
//! exactly-singular complex matrix, faer's partial-pivot LU produces a
//! finite (huge) result, so ferray returns `Ok(<finite garbage>)` where
//! numpy raises `LinAlgError("Singular matrix")`.
//!
//! Oracle (numpy 2.4.4) for the rank-1 complex matrix
//!   A = [[1+1j, 2+2j], [2+2j, 4+4j]]   (row2 == 2*row1, det == 0j):
//!   np.linalg.inv(A)            -> raises LinAlgError("Singular matrix")
//!   np.linalg.solve(A, [1, 1]) -> raises LinAlgError("Singular matrix")

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_linalg::complex::{inv_complex, solve_complex, solve_complex_vec};
use num_complex::Complex;

fn c64(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// A rank-1 (exactly singular) 2x2 complex matrix: row 2 == 2 * row 1.
/// det(A) == 0j (numpy oracle: `np.linalg.det(A) == 0j`).
fn singular_complex_2x2() -> Array<Complex<f64>, Ix2> {
    Array::<Complex<f64>, Ix2>::from_vec(
        Ix2::new([2, 2]),
        vec![c64(1.0, 1.0), c64(2.0, 2.0), c64(2.0, 2.0), c64(4.0, 4.0)],
    )
    .unwrap()
}

/// Divergence: `ferray_linalg::complex::inv_complex` diverges from
/// `numpy.linalg.inv` for an exactly-singular complex matrix.
///
/// Upstream: `numpy/linalg/_linalg.py` `inv` -> LAPACK `getrf`/`getri`
/// singular-pivot check raises `LinAlgError("Singular matrix")` for
/// A = [[1+1j, 2+2j], [2+2j, 4+4j]] (det == 0j).
/// Target: `complex.rs:244-254` runs faer `partial_piv_lu().inverse()` and
/// performs NO singularity check (admitted at `complex.rs:249-253`), so it
/// returns `Ok(<finite garbage>)`.
///
/// Tracking: #1083 (blocker).
#[test]
fn divergence_inv_complex_singular_returns_ok_not_error() {
    let a = singular_complex_2x2();
    let result = inv_complex(&a);
    assert!(
        result.is_err(),
        "inv_complex on an exactly-singular complex matrix must return an \
         error (numpy raises LinAlgError(\"Singular matrix\")); ferray \
         returned Ok({:?})",
        result.map(|m| m.iter().copied().collect::<Vec<_>>())
    );
}

/// Divergence: `ferray_linalg::complex::solve_complex` diverges from
/// `numpy.linalg.solve` for an exactly-singular complex coefficient matrix.
///
/// Upstream: `numpy/linalg/_linalg.py` `solve` -> LAPACK `gesv` raises
/// `LinAlgError("Singular matrix")` for A = [[1+1j, 2+2j], [2+2j, 4+4j]].
/// Target: `complex.rs:287-299` runs faer LU `solve` with no singularity
/// check and returns `Ok(<finite garbage>)`.
///
/// Tracking: #1083 (blocker).
#[test]
fn divergence_solve_complex_singular_returns_ok_not_error() {
    let a = singular_complex_2x2();
    let b =
        Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([2, 1]), vec![c64(1.0, 0.0), c64(1.0, 0.0)])
            .unwrap();
    let result = solve_complex(&a, &b);
    assert!(
        result.is_err(),
        "solve_complex on an exactly-singular complex A must return an error \
         (numpy raises LinAlgError(\"Singular matrix\")); ferray returned \
         Ok({:?})",
        result.map(|m| m.iter().copied().collect::<Vec<_>>())
    );
}

/// Divergence: `ferray_linalg::complex::solve_complex_vec` diverges from
/// `numpy.linalg.solve` (1-D rhs) for an exactly-singular complex matrix.
///
/// Upstream: `numpy.linalg.solve(A, [1, 1])` raises
/// `LinAlgError("Singular matrix")` for A = [[1+1j, 2+2j], [2+2j, 4+4j]].
/// Target: `complex.rs:328-341` runs faer LU `solve` with no singularity
/// check and returns `Ok(<finite garbage>)`.
///
/// Tracking: #1083 (blocker).
#[test]
fn divergence_solve_complex_vec_singular_returns_ok_not_error() {
    let a = singular_complex_2x2();
    let b = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([2]), vec![c64(1.0, 0.0), c64(1.0, 0.0)])
        .unwrap();
    let result = solve_complex_vec(&a, &b);
    assert!(
        result.is_err(),
        "solve_complex_vec on an exactly-singular complex A must return an \
         error (numpy raises LinAlgError(\"Singular matrix\")); ferray \
         returned Ok({:?})",
        result.map(|m| m.iter().copied().collect::<Vec<_>>())
    );
}
