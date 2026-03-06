//! ferrum-linalg: Linear algebra operations for ferrum.
//!
//! Complete numpy.linalg parity powered by faer. Includes matrix products
//! (dot, matmul, einsum, tensordot, kron), decompositions (cholesky, QR, SVD,
//! LU, eigen), solvers (solve, lstsq, inv, pinv), norms and measures
//! (norm, cond, det, trace, matrix\_rank). All functions support batched
//! (stacked 3D+) arrays with automatic parallelization along batch dimensions.

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_return)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unused_enumerate_index)]

/// Batched dispatch for stacked (3D+) arrays with Rayon parallelism.
pub mod batch;
/// Matrix decompositions: Cholesky, QR, SVD, LU, eigendecomposition.
pub mod decomp;
/// Conversion bridge between ferrum arrays and faer matrices.
pub mod faer_bridge;
/// Norms, condition numbers, determinants, and related measures.
pub mod norms;
/// Matrix products: dot, matmul, einsum, tensordot, kron, multi_dot.
pub mod products;
/// Linear solvers: solve, lstsq, inv, pinv, matrix_power, tensorsolve, tensorinv.
pub mod solve;

/// f16 (half-precision) linalg operations with f64 promotion.
#[cfg(feature = "f16")]
pub mod f16_support;

// Re-export key types and functions at the crate root for ergonomic access

// Matrix products (Section 8.1)
pub use products::{
    TensordotAxes, dot, einsum, inner, kron, matmul, multi_dot, outer, tensordot, vdot, vecdot,
};

// Decompositions (Section 8.2)
pub use decomp::{QrMode, cholesky, eig, eigh, eigvals, eigvalsh, lu, qr, svd};

// Solving and inversion (Section 8.3)
pub use solve::{inv, lstsq, matrix_power, pinv, solve, tensorinv, tensorsolve};

// Norms and measures (Section 8.4)
pub use norms::{NormOrder, cond, det, det_batched, matrix_rank, norm, slogdet, trace};
