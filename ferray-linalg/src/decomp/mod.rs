// ferray-linalg: Decomposition module
//
// Matrix decompositions: Cholesky, QR, SVD, LU, Eigen.

/// Cholesky decomposition.
pub mod cholesky;
/// Eigendecomposition for general and symmetric matrices.
pub mod eigen;
/// LU decomposition with partial pivoting.
pub mod lu;
/// QR decomposition with reduced and complete modes.
pub mod qr;
/// Singular Value Decomposition.
pub mod svd;

pub use cholesky::{cholesky, cholesky_batched, cholesky_upper};
pub use eigen::{
    UPLO, eig, eigh, eigh_batched, eigh_uplo, eigvals, eigvalsh, eigvalsh_batched, eigvalsh_uplo,
};
pub use lu::lu;
pub use qr::{QrMode, qr, qr_batched};
pub use svd::{svd, svd_batched, svdvals};
