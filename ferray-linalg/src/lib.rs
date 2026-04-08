//! ferray-linalg: Linear algebra operations for ferray.
//!
//! Complete numpy.linalg parity powered by faer. Includes matrix products
//! (dot, matmul, einsum, tensordot, kron), decompositions (cholesky, QR, SVD,
//! LU, eigen), solvers (solve, lstsq, inv, pinv), norms and measures
//! (norm, cond, det, trace, matrix\_rank). All functions support batched
//! (stacked 3D+) arrays with automatic parallelization along batch dimensions.

// Kept intentionally narrow. `type_complexity` is accepted for
// decomposition return types (e.g. SVD returning `(U, S, V)` of nested
// generic arrays). `needless_range_loop` stays because many of the
// explicit-index loops here are index-arithmetic-heavy and the
// "idiomatic" zip-based rewrites are less readable for that style of
// code. Prior #193 removed broader `missing_docs`, `needless_return`,
// `if_same_then_else`, `unused_enumerate_index`, and `assign_op_pattern`
// allow-lists that masked real issues.
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

/// Batched dispatch for stacked (3D+) arrays with Rayon parallelism.
pub mod batch;
/// Complex matrix operations (matmul/inv/solve/det for `Array<Complex<T>, Ix2>`).
pub mod complex;
/// Matrix decompositions: Cholesky, QR, SVD, LU, eigendecomposition.
pub mod decomp;
/// Conversion bridge between ferray arrays and faer matrices.
pub mod faer_bridge;
/// Norms, condition numbers, determinants, and related measures.
pub mod norms;
/// Matrix products: dot, matmul, einsum, tensordot, kron, multi_dot.
pub mod products;
/// Sealed trait bounding the float types (f32, f64) supported by linalg.
pub mod scalar;
/// Linear solvers: solve, lstsq, inv, pinv, matrix_power, tensorsolve, tensorinv.
pub mod solve;

/// f16 (half-precision) linalg operations with f64 promotion.
#[cfg(feature = "f16")]
pub mod f16_support;

// Re-export key types and functions at the crate root for ergonomic access
pub use scalar::LinalgFloat;

// Matrix products (Section 8.1)
pub use products::{
    TensordotAxes, cross, dot, einsum, inner, kron, matmul, multi_dot, outer, tensordot, vdot,
    vecdot,
};

// Decompositions (Section 8.2)
pub use decomp::{
    QrMode, cholesky, cholesky_batched, eig, eigh, eigh_batched, eigvals, eigvalsh,
    eigvalsh_batched, lu, qr, qr_batched, svd, svd_batched, svdvals,
};

// Solving and inversion (Section 8.3)
pub use solve::{
    inv, inv_batched, lstsq, matrix_power, matrix_power_batched, pinv, pinv_batched, solve,
    solve_batched, tensorinv, tensorsolve,
};

// Norms and measures (Section 8.4)
pub use norms::{
    NormOrder, cond, det, det_batched, matrix_rank, matrix_rank_batched, norm, norm_axis,
    slogdet, slogdet_batched, trace,
};

// Complex matrix operations (#404) — take `Array<Complex<T>, Ix2>` directly
// since `LinalgFloat` is sealed to real floats.
pub use complex::{
    det_complex, inv_complex, matmul_complex, solve_complex, solve_complex_vec,
};
