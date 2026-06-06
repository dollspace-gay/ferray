// ferray-linalg: Decomposition module
//
// Matrix decompositions: Cholesky, QR, SVD, LU, Eigen.

/// Cholesky decomposition.
pub mod cholesky;
/// Eigendecomposition for general and symmetric matrices.
pub mod eigen;
/// LAPACK `getrf`+`getri`-backed matrix inverse, bit-identical to
/// scipy/numpy (#2117). Only present with the `openblas` feature (the
/// LAPACK symbols ship inside the linked OpenBLAS shared object).
#[cfg(feature = "openblas")]
pub mod inv_lapack;
/// LU decomposition with partial pivoting.
pub mod lu;
/// QR decomposition with reduced and complete modes.
pub mod qr;
/// Singular Value Decomposition.
pub mod svd;
/// LAPACK `gesdd`-backed SVD, bit-identical to scipy/numpy (#2116).
/// Only present with the `openblas` feature (the LAPACK symbols ship
/// inside the linked OpenBLAS shared object).
#[cfg(feature = "openblas")]
pub mod svd_lapack;

pub use cholesky::{cholesky, cholesky_batched, cholesky_upper, cholesky_upper_batched};
pub use eigen::{
    UPLO, eig, eigh, eigh_batched, eigh_batched_uplo, eigh_uplo, eigvals, eigvalsh,
    eigvalsh_batched, eigvalsh_batched_uplo, eigvalsh_uplo,
};
#[cfg(feature = "openblas")]
pub use inv_lapack::{LapackGetri, inv_lapack};
pub use lu::lu;
pub use qr::{QrMode, qr, qr_batched};
pub use svd::{svd, svd_batched, svd_hermitian, svdvals};
#[cfg(feature = "openblas")]
pub use svd_lapack::{LapackGesdd, svd_lapack};
