//! ACToR critic divergence pins for ferray-linalg solve/inv singular-matrix
//! detection vs numpy 2.4.4 (`/home/doll/ferray/ferray-python/.venv`).
//!
//! Audited kernels: `decomp/qr.rs`, `decomp/svd.rs`, `solve.rs`, `norms.rs`.
//!
//! NO DIVERGENCE found in: qr (reduced/complete, tall, wide m<n; Q@R recon),
//! svd (thin/full, wide, zero matrix — all shapes + values match numpy),
//! svdvals, all vector norms (ord 0/1/2/inf/-inf), all matrix norms
//! (fro/nuc/1/-1/2/-2/inf/-inf — values within ULP of numpy).
//!
//! DIVERGENCE found: `solve` and `inv` detect singularity ONLY by scanning
//! the output for NaN/Inf (`solve.rs:84`, `solve.rs:320`). For an EXACTLY
//! singular matrix whose faer partial-pivot LU yields a tiny-but-finite
//! pivot (rounding noise), the scan misses it and ferray returns finite
//! garbage where numpy raises `LinAlgError("Singular matrix")`.
//!
//! Oracle (numpy 2.4.4) for `A = [[1,2,3],[4,5,6],[7,8,9]]` (det == 0.0):
//!   np.linalg.solve(A, [1,1,1]) -> raises LinAlgError("Singular matrix")
//!   np.linalg.inv(A)           -> raises LinAlgError("Singular matrix")

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_linalg::solve::{inv, solve};

/// Divergence: `ferray_linalg::solve::solve` diverges from
/// `numpy.linalg.solve` (`numpy/linalg/_linalg.py` `_assert_*` / LAPACK
/// `gesv` singular check) for the exactly-singular 3x3 matrix
/// `A = [[1,2,3],[4,5,6],[7,8,9]]` (det == 0.0).
///
/// Upstream: numpy raises `LinAlgError("Singular matrix")`.
/// Target: faer's partial_piv_lu yields a finite pivot, so the NaN/Inf
/// scan at `solve.rs:84` misses the singularity and `solve` returns
/// `Ok([0.05, -1.1, 1.05])` (finite garbage) instead of an error.
///
/// Tracking: #1077 (blocker).
#[test]
fn divergence_solve_exact_singular_3x3_returns_ok_not_error() {
    // det(A) == 0 exactly (rows are arithmetic progressions: r2 = 2*r1 - r0... rank 2).
    let a = Array::<f64, Ix2>::from_vec(
        Ix2::new([3, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 1.0, 1.0]).unwrap();

    let result = solve(&a, &b);

    // numpy raises LinAlgError("Singular matrix") — the contract is an error,
    // not a finite-but-meaningless solution vector.
    assert!(
        result.is_err(),
        "solve on exactly-singular A must return an error (numpy raises \
         LinAlgError); ferray returned Ok({:?})",
        result.map(|x| x.iter().copied().collect::<Vec<_>>())
    );
}

/// Divergence: `ferray_linalg::solve::inv` diverges from `numpy.linalg.inv`
/// for the exactly-singular 3x3 matrix `A = [[1,2,3],[4,5,6],[7,8,9]]`.
///
/// Upstream: numpy raises `LinAlgError("Singular matrix")`.
/// Target: faer's LU inverse produces a finite (huge) matrix, so the
/// NaN/Inf scan at `solve.rs:320` misses the singularity and `inv` returns
/// `Ok(<finite garbage>)` instead of `SingularMatrix`.
///
/// Tracking: #1078 (blocker).
#[test]
fn divergence_inv_exact_singular_3x3_returns_ok_not_error() {
    let a = Array::<f64, Ix2>::from_vec(
        Ix2::new([3, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let result = inv(&a);

    assert!(
        result.is_err(),
        "inv on exactly-singular A must return SingularMatrix (numpy raises \
         LinAlgError); ferray returned Ok(<finite>)"
    );
}
