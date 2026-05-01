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
// Linear algebra kernels divide by `usize`-typed dimensions (`norm`, mean
// centering for cov/cond), promote `f16/bf16` through `f32` for BLAS
// dispatch, and round-trip pivots/ranks across `usize <-> isize`. Oracle
// and property tests rely on exact float equality against analytic
// reference values.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms,
    clippy::suboptimal_flops
)]

/// Batched dispatch for stacked (3D+) arrays with Rayon parallelism.
pub mod batch;
/// Complex matrix operations (matmul/inv/solve/det for `Array<Complex<T>, Ix2>`).
pub mod complex;
/// Matrix decompositions: Cholesky, QR, SVD, LU, eigendecomposition.
pub mod decomp;
/// Conversion bridge between ferray arrays and faer matrices.
pub mod faer_bridge;
/// Hand-tuned Rust DGEMM micro-kernel for AVX2+FMA. Used by `matmul_raw`
/// when the runtime CPU supports it; falls back to faer otherwise. Beats
/// `private-gemm-x86` (faer's default backend) at small/medium N where
/// faer's parallel-rayon path adds more overhead than it saves.
pub mod gemm;
/// Norms, condition numbers, determinants, and related measures.
pub mod norms;
/// Matrix products: dot, matmul, einsum, tensordot, kron, `multi_dot`.
pub mod products;
/// Sealed trait bounding the float types (f32, f64) supported by linalg.
pub mod scalar;
/// Linear solvers: solve, lstsq, inv, pinv, `matrix_power`, tensorsolve, tensorinv.
pub mod solve;
/// Native triangular solve (TRSM) primitives: lower/upper × non-unit/unit
/// diagonal, in-place row-major. Used by Cholesky/LU back-substitution
/// chains; reuses the GEMM scaffolding for the trailing block update.
pub mod trsm;

/// f16 (half-precision) linalg operations with f64 promotion.
#[cfg(feature = "f16")]
pub mod f16_support;

// Re-export key types and functions at the crate root for ergonomic access
pub use scalar::LinalgFloat;

// Matrix products (Section 8.1)
pub use products::{
    TensordotAxes, cross, dot, einsum, inner, kron, matmul, matvec, multi_dot, outer, tensordot,
    vdot, vecdot, vecmat,
};

// Decompositions (Section 8.2)
pub use decomp::{
    QrMode, UPLO, cholesky, cholesky_batched, cholesky_upper, cholesky_upper_batched, eig, eigh,
    eigh_batched, eigh_batched_uplo, eigh_uplo, eigvals, eigvalsh, eigvalsh_batched,
    eigvalsh_batched_uplo, eigvalsh_uplo, lu, qr, qr_batched, svd, svd_batched, svd_hermitian,
    svdvals,
};

// Solving and inversion (Section 8.3)
pub use solve::{
    inv, inv_batched, lstsq, matrix_power, matrix_power_batched, pinv, pinv_batched, solve,
    solve_batched, tensorinv, tensorsolve,
};

// Norms and measures (Section 8.4)
pub use norms::{
    NormOrder, cond, det, det_batched, diagonal, matrix_norm, matrix_rank, matrix_rank_batched,
    matrix_transpose, norm, norm_axis, slogdet, slogdet_batched, trace, trace_offset, vector_norm,
};

// Complex matrix operations (#404) — take `Array<Complex<T>, Ix2>` directly
// since `LinalgFloat` is sealed to real floats.
pub use complex::{det_complex, inv_complex, matmul_complex, solve_complex, solve_complex_vec};
