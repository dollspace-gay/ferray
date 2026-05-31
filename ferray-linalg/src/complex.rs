//! Complex matrix operations (#404).
//!
//! `LinalgFloat` is sealed to `f32` / `f64` and provides the generic
//! entry points (`matmul`, `solve`, `inv`, ...) for real matrices.
//! Making `LinalgFloat` generic over both real and complex scalars
//! would require a multi-file refactor of every function in the
//! crate, so this module takes the pragmatic path: it exposes
//! complex-specific functions that take `Array<Complex<T>, Ix2>`
//! directly and thin-wrap faer's complex `Mat` operations.
//!
//! faer 0.24's `ComplexField` impl covers `Complex<T>` transparently
//! via `impl<T: RealField<Unit: ComplexField>> ComplexField for
//! Complex<T>`, so the faer calls here compile unchanged — only the
//! element type at the ferray ↔ faer boundary is different.
//!
//! Functions provided in this first pass:
//!
//! - [`matmul_complex`]  — complex matrix × matrix
//! - [`inv_complex`]     — complex matrix inverse via LU
//! - [`solve_complex`]   — A x = b with complex A, b
//! - [`det_complex`]     — complex determinant
//! - `svd_complex_f64` / `svd_complex_f32` — complex SVD (U, real S, Vh)
//! - `qr_complex_f64` / `qr_complex_f32`   — complex QR (Q unitary, R)
//!
//! Batched variants and the rest can be layered on later as demand appears.
//!
//! ## REQ status
//!
//! Two states only (SHIPPED / NOT-STARTED), per goal.md. Tracks the complex
//! matrix-op surface that `numpy.linalg` exposes over a complex-valued matrix
//! (`numpy/linalg/_linalg.py` accepts "a complex- or real-valued matrix").
//! Verified by this module's `#[cfg(test)]` reconstruction tests plus
//! `ferray-python/tests/test_expansion_complex_decomp.py` against live numpy.
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | complex matmul/inv/solve/det (#931) | SHIPPED | Impl: `matmul_complex`/`inv_complex`/`solve_complex`/`det_complex`. Consumer: `det`/`inv`/`solve`/`matmul` complex arms in `ferray-python/src/linalg.rs`. |
//! | complex eig/eigvals (#937) | SHIPPED | Impl: `eig_complex_f64`/`eigvals_complex_f64` (+f32). Consumer: `eig`/`eigvals` complex arms in `ferray-python/src/linalg.rs`. |
//! | complex svd/qr (#939) | SHIPPED | Impl: `svd_complex_f64`/`svd_complex_f32` (faer complex `Svd`, U/Vh complex, S real), `qr_complex_f64`/`qr_complex_f32` (faer complex `Qr`, Q unitary, R). Consumer: `svd`/`qr`/`pinv`/`matrix_rank` complex arms in `ferray-python/src/linalg.rs`. |
//! | complex pinv/matrix_rank/slogdet/matrix_power (#939) | SHIPPED | Composed from the above: pinv = `Vh^H diag(1/s) U^H`, matrix_rank = `#{s > tol}`, slogdet = `det/|det|` + `ln|det|` (via `det_complex`), matrix_power = repeated `matmul_complex` (`inv_complex` for n<0). Consumer: `pinv`/`matrix_rank`/`slogdet`/`matrix_power` complex arms in `ferray-python/src/linalg.rs`. |

use num_complex::Complex;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use faer::linalg::solvers::{DenseSolveCore, PartialPivLu, Solve};

/// Decide whether a complex square matrix is singular from its partial-pivot
/// LU factorization, matching numpy/LAPACK rather than scanning the output for
/// NaN/Inf.
///
/// numpy's `linalg.inv`/`solve` delegate to LAPACK complex `getrf`/`gesv`,
/// which report singularity when a pivot on the U-diagonal collapses to the
/// machine floor relative to the largest pivot. faer's complex `partial_piv_lu`
/// does **not** produce an exact zero for an exactly-rank-deficient input (it
/// yields a tiny-but-finite pivot), so a literal `== 0` test would miss it.
///
/// Criterion (mirrors the real path `crate::solve::lu_is_singular`): singular
/// iff `min|U_ii| <= n * eps * max|U_ii|`, where `|U_ii|` is the complex
/// modulus of each U-diagonal entry. This relative-pivot-floor test flags
/// exactly-singular and numerically rank-deficient inputs while leaving
/// genuinely invertible (even ill-conditioned) complex matrices solvable.
fn complex_lu_is_singular<T>(lu: &PartialPivLu<Complex<T>>, n: usize) -> bool
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    if n == 0 {
        return false;
    }
    let u = lu.U();
    // `|U_ii|` lives in faer's real associated type for the complex field.
    type Real<C> = <C as faer_traits::ComplexField>::Real;
    let mut min_abs: Real<Complex<T>> = faer_traits::math_utils::zero::<Real<Complex<T>>>();
    let mut max_abs: Real<Complex<T>> = faer_traits::math_utils::zero::<Real<Complex<T>>>();
    let mut first = true;
    for i in 0..n {
        let d = faer_traits::math_utils::abs::<Complex<T>>(&u[(i, i)]);
        if first || d < min_abs {
            min_abs = d.clone();
        }
        if first || d > max_abs {
            max_abs = d;
        }
        first = false;
    }
    let n_real = faer_traits::math_utils::from_f64::<Real<Complex<T>>>(n as f64);
    let eps = faer_traits::math_utils::eps::<Real<Complex<T>>>();
    let threshold = n_real * eps * max_abs;
    min_abs <= threshold
}

/// Convert a 2-D ferray `Array<Complex<T>, Ix2>` into a faer `Mat<Complex<T>>`.
fn array2c_to_faer<T>(a: &Array<Complex<T>, Ix2>) -> faer::Mat<Complex<T>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    if let Some(slice) = a.as_slice() {
        faer::Mat::from_fn(m, n, |i, j| slice[i * n + j])
    } else {
        let data: Vec<Complex<T>> = a.iter().copied().collect();
        faer::Mat::from_fn(m, n, |i, j| data[i * n + j])
    }
}

/// Convert a faer `Mat<Complex<T>>` back to a ferray `Array<Complex<T>, Ix2>`.
fn faer_to_array2c<T>(mat: &faer::Mat<Complex<T>>) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy,
{
    let (m, n) = mat.shape();
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat[(i, j)]);
        }
    }
    Array::from_vec(Ix2::new([m, n]), data)
}

/// Complex matrix multiplication: `C = A @ B`.
///
/// Both inputs must be 2-D with compatible inner dimensions; the
/// result is always 2-D. Delegates to `faer::linalg::matmul::matmul`
/// through the faer complex `Mat` type.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if inner dimensions don't match.
pub fn matmul_complex<T>(
    a: &Array<Complex<T>, Ix2>,
    b: &Array<Complex<T>, Ix2>,
) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k1 != k2 {
        return Err(FerrayError::shape_mismatch(format!(
            "matmul_complex: inner dimensions don't match ({m}x{k1} @ {k2}x{n})"
        )));
    }

    // Hand-tuned AVX2 CGEMM (Complex<f32>) path. Translated from OpenBLAS
    // cgemm_kernel_8x2_haswell.c. Mirrors the ZGEMM Complex<f64> path
    // below but with f32 lanes.
    #[cfg(target_arch = "x86_64")]
    {
        use std::any::TypeId;
        const HAND_TUNED_MIN_DIM_C32: usize = 16;
        if m.max(n).max(k1) >= HAND_TUNED_MIN_DIM_C32
            && TypeId::of::<Complex<T>>() == TypeId::of::<Complex<f32>>()
            && crate::gemm::cpu_supports_avx2_fma()
        {
            let a_complex: Vec<Complex<f32>> = a
                .iter()
                .map(|x| *unsafe { &*(x as *const Complex<T> as *const Complex<f32>) })
                .collect();
            let b_complex: Vec<Complex<f32>> = b
                .iter()
                .map(|x| *unsafe { &*(x as *const Complex<T> as *const Complex<f32>) })
                .collect();
            let mut c_complex: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; m * n];

            let ok = unsafe {
                crate::gemm::gemm_c32(
                    m,
                    n,
                    k1,
                    1.0,
                    0.0,
                    a_complex.as_ptr() as *const f32,
                    b_complex.as_ptr() as *const f32,
                    0.0,
                    0.0,
                    c_complex.as_mut_ptr() as *mut f32,
                )
            };
            if ok {
                let len = c_complex.len();
                let cap = c_complex.capacity();
                let ptr = c_complex.as_mut_ptr();
                std::mem::forget(c_complex);
                let data: Vec<Complex<T>> =
                    unsafe { Vec::from_raw_parts(ptr.cast::<Complex<T>>(), len, cap) };
                return Array::from_vec(Ix2::new([m, n]), data);
            }
        }
    }

    // Hand-tuned AVX2 4x2 ZGEMM path for Complex<f64> when AVX2+FMA is
    // available. Translated from OpenBLAS zgemm_kernel_4x2_haswell.c.
    // Routes via TypeId so f32 complex still goes through faer below.
    #[cfg(target_arch = "x86_64")]
    {
        use std::any::TypeId;
        const HAND_TUNED_MIN_DIM: usize = 16;
        if m.max(n).max(k1) >= HAND_TUNED_MIN_DIM
            && TypeId::of::<Complex<T>>() == TypeId::of::<Complex<f64>>()
            && crate::gemm::cpu_supports_avx2_fma()
        {
            // Materialise A and B as flat row-major Complex<f64> buffers.
            let a_complex: Vec<Complex<f64>> = a
                .iter()
                .map(|x| {
                    // SAFETY: TypeId guard verified Complex<T> == Complex<f64>.
                    *unsafe { &*(x as *const Complex<T> as *const Complex<f64>) }
                })
                .collect();
            let b_complex: Vec<Complex<f64>> = b
                .iter()
                .map(|x| *unsafe { &*(x as *const Complex<T> as *const Complex<f64>) })
                .collect();
            let mut c_complex: Vec<Complex<f64>> = vec![Complex { re: 0.0, im: 0.0 }; m * n];

            // SAFETY: Complex<f64> is repr(C) with 2 contiguous f64s, so
            // we can reinterpret as *const/mut f64. Buffer lengths match
            // the (m, n, k) dims; cpu_supports_avx2_fma verified at runtime.
            let ok = unsafe {
                crate::gemm::gemm_c64(
                    m,
                    n,
                    k1,
                    1.0,
                    0.0,
                    a_complex.as_ptr() as *const f64,
                    b_complex.as_ptr() as *const f64,
                    0.0,
                    0.0,
                    c_complex.as_mut_ptr() as *mut f64,
                )
            };
            if ok {
                // Convert Vec<Complex<f64>> -> Vec<Complex<T>> via reinterpret.
                let len = c_complex.len();
                let cap = c_complex.capacity();
                let ptr = c_complex.as_mut_ptr();
                std::mem::forget(c_complex);
                // SAFETY: TypeId guard verified Complex<T> == Complex<f64>.
                let data: Vec<Complex<T>> =
                    unsafe { Vec::from_raw_parts(ptr.cast::<Complex<T>>(), len, cap) };
                return Array::from_vec(Ix2::new([m, n]), data);
            }
        }
    }

    let a_faer = array2c_to_faer(a);
    let b_faer = array2c_to_faer(b);

    // faer's `*` operator on Mat dispatches through its matmul kernel
    // and handles the complex conjugate / transpose metadata for us.
    let c_faer = &a_faer * &b_faer;
    faer_to_array2c(&c_faer)
}

/// Complex matrix inverse via LU with partial pivoting.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if the matrix is not square.
/// - [`FerrayError::SingularMatrix`] if the LU U-diagonal pivots indicate a
///   rank-deficient matrix (relative-pivot-floor check, matching LAPACK).
pub fn inv_complex<T>(a: &Array<Complex<T>, Ix2>) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "inv_complex: requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    if n == 0 {
        return Array::from_vec(Ix2::new([0, 0]), Vec::new());
    }

    let mat = array2c_to_faer(a);
    let lu = mat.as_ref().partial_piv_lu();

    // numpy/LAPACK complex `getrf`/`getri` raise LinAlgError("Singular matrix")
    // on a rank-deficient input. faer's complex LU yields a tiny-but-finite
    // pivot (so the result is NaN/Inf garbage), so detect singularity from the
    // U-diagonal pivot magnitudes — the same relative-floor criterion as the
    // real path (`crate::solve::lu_is_singular`).
    if complex_lu_is_singular(&lu, n) {
        return Err(FerrayError::SingularMatrix {
            message: "matrix is singular; cannot compute inverse".to_string(),
        });
    }

    let inv_mat = lu.inverse();
    faer_to_array2c(&inv_mat)
}

/// Solve the complex linear system `A x = b`. `b` may be a vector
/// (1-D, length `n`) or a matrix of right-hand sides (2-D, `n × k`).
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] on incompatible shapes.
/// - [`FerrayError::SingularMatrix`] if A is singular (LU pivot-floor check).
pub fn solve_complex<T>(
    a: &Array<Complex<T>, Ix2>,
    b: &Array<Complex<T>, Ix2>,
) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex: A must be square, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape[0] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex: A is {}x{} but b has {} rows",
            n, n, b_shape[0]
        )));
    }
    let nrhs = b_shape[1];

    let a_faer = array2c_to_faer(a);
    let lu = a_faer.as_ref().partial_piv_lu();
    // numpy/LAPACK complex `gesv` raises LinAlgError("Singular matrix") on a
    // singular coefficient matrix; mirror the real-path pivot-floor check.
    if complex_lu_is_singular(&lu, n) {
        return Err(FerrayError::SingularMatrix {
            message: "matrix is singular; solve has no unique solution".to_string(),
        });
    }
    let b_faer = array2c_to_faer(b);
    let x = lu.solve(&b_faer);

    // Extract back into a ferray Array.
    let mut data = Vec::with_capacity(n * nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            data.push(x[(i, j)]);
        }
    }
    Array::from_vec(Ix2::new([n, nrhs]), data)
}

/// Solve a complex vector system: same as [`solve_complex`] but with
/// a 1-D right-hand side vector.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] on incompatible shapes.
/// - [`FerrayError::SingularMatrix`] if A is singular (LU pivot-floor check).
pub fn solve_complex_vec<T>(
    a: &Array<Complex<T>, Ix2>,
    b: &Array<Complex<T>, Ix1>,
) -> FerrayResult<Array<Complex<T>, Ix1>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex_vec: A must be square, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape[0] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex_vec: A is {}x{} but b has length {}",
            n, n, b_shape[0]
        )));
    }

    let a_faer = array2c_to_faer(a);
    let lu = a_faer.as_ref().partial_piv_lu();
    // numpy/LAPACK complex `gesv` raises LinAlgError("Singular matrix") on a
    // singular coefficient matrix; mirror the real-path pivot-floor check.
    if complex_lu_is_singular(&lu, n) {
        return Err(FerrayError::SingularMatrix {
            message: "matrix is singular; solve has no unique solution".to_string(),
        });
    }

    // Build b as an n×1 faer Mat.
    let b_data: Vec<Complex<T>> = b.iter().copied().collect();
    let b_mat = faer::Mat::from_fn(n, 1, |i, _| b_data[i]);

    let x = lu.solve(&b_mat);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push(x[(i, 0)]);
    }
    Array::from_vec(Ix1::new([n]), result)
}

/// Complex matrix determinant. Delegates to `faer::Mat::determinant`.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if the matrix is not square.
pub fn det_complex<T>(a: &Array<Complex<T>, Ix2>) -> FerrayResult<Complex<T>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "det_complex: requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let mat = array2c_to_faer(a);
    Ok(mat.as_ref().determinant())
}

// Complex eigendecomposition (#937). faer's general `eigen()`/`eigenvalues()`
// over a complex `Mat<Complex<T>>` return eigenvalues of faer's complexified
// scalar type `<Complex<T> as ComplexField>::Real` wrapped in `Complex`, which
// for a `Complex<T>` field is `Complex<T>` itself — but the generic
// associated-type identity `<Complex<T> as ComplexField>::Real == T` is opaque
// to the type checker, so these are emitted as concrete `f32`/`f64`
// monomorphisations (the only two widths ferray marshals). numpy.linalg.eig /
// eigvals accept "a complex- or real-valued matrix"
// (numpy/linalg/_linalg.py:1372 / :1193) and return complex eigenpairs whose
// order is not canonical (:1199 "not necessarily ordered" — callers sort both
// sides to compare against numpy).
macro_rules! impl_complex_eig {
    ($eig_fn:ident, $eigvals_fn:ident, $T:ty) => {
        /// Eigenvalues and right eigenvectors of a general complex square
        #[doc = concat!("matrix over `Complex<", stringify!($T), ">`. See [`crate::eig`].")]
        ///
        /// # Errors
        /// - [`FerrayError::ShapeMismatch`] if the matrix is not square.
        /// - [`FerrayError::InvalidValue`] if the eigensolver fails to converge.
        pub fn $eig_fn(
            a: &Array<Complex<$T>, Ix2>,
        ) -> FerrayResult<(Array<Complex<$T>, Ix1>, Array<Complex<$T>, Ix2>)> {
            let shape = a.shape();
            if shape[0] != shape[1] {
                return Err(FerrayError::shape_mismatch(format!(
                    "eig_complex: requires a square matrix, got {}x{}",
                    shape[0], shape[1]
                )));
            }
            let n = shape[0];
            let mat = array2c_to_faer(a);
            let decomp = mat.as_ref().eigen().map_err(|e| FerrayError::InvalidValue {
                message: format!("complex eigendecomposition failed: {e:?}"),
            })?;

            let s = decomp.S();
            let mut eigenvalues: Vec<Complex<$T>> = Vec::with_capacity(n);
            for i in 0..n {
                eigenvalues.push(s.column_vector()[i]);
            }

            let u = decomp.U();
            let mut eigvecs: Vec<Complex<$T>> = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    eigvecs.push(u[(i, j)]);
                }
            }

            let vals = Array::from_vec(Ix1::new([n]), eigenvalues)?;
            let vecs = Array::from_vec(Ix2::new([n, n]), eigvecs)?;
            Ok((vals, vecs))
        }

        /// Eigenvalues of a general complex square matrix over
        #[doc = concat!("`Complex<", stringify!($T), ">` (no eigenvectors). See [`crate::eigvals`].")]
        ///
        /// # Errors
        /// - [`FerrayError::ShapeMismatch`] if the matrix is not square.
        /// - [`FerrayError::InvalidValue`] if the eigensolver fails.
        pub fn $eigvals_fn(
            a: &Array<Complex<$T>, Ix2>,
        ) -> FerrayResult<Array<Complex<$T>, Ix1>> {
            let shape = a.shape();
            if shape[0] != shape[1] {
                return Err(FerrayError::shape_mismatch(format!(
                    "eigvals_complex: requires a square matrix, got {}x{}",
                    shape[0], shape[1]
                )));
            }
            let mat = array2c_to_faer(a);
            let vals = mat
                .as_ref()
                .eigenvalues()
                .map_err(|e| FerrayError::InvalidValue {
                    message: format!("complex eigenvalue computation failed: {e:?}"),
                })?;
            Array::from_vec(Ix1::new([vals.len()]), vals)
        }
    };
}

impl_complex_eig!(eig_complex_f64, eigvals_complex_f64, f64);
impl_complex_eig!(eig_complex_f32, eigvals_complex_f32, f32);

// Complex SVD / QR (#939). faer's `Mat::svd()` / `Mat::qr()` work over a
// complex `Mat<Complex<T>>` exactly as for the real case — the singular
// values come back in faer's REAL associated type (`<Complex<T> as
// ComplexField>::Real == T`), which is opaque to the type checker, so (like
// `impl_complex_eig!`) these are emitted as concrete `f32`/`f64`
// monomorphisations. numpy.linalg.svd / qr accept "a complex- or real-valued
// matrix" and return complex U/Vh (resp. Q/R) with REAL singular values
// (numpy/linalg/_linalg.py:1681 svd, :1083 qr). SVD/QR factors are NOT unique
// (U/Vh column phase, Q/R sign), so callers verify by reconstruction
// (`U diag(s) Vh == A`, `Q R == A`) and compare `s` directly.
macro_rules! impl_complex_svd_qr {
    ($svd_fn:ident, $qr_fn:ident, $T:ty) => {
        /// Full Singular Value Decomposition of a complex matrix over
        #[doc = concat!("`Complex<", stringify!($T), ">`. Returns `(U, s, Vh)` with")]
        /// `U` (m×m, complex, unitary), `s` (the REAL singular values, length
        /// `min(m,n)`, descending), and `Vh = V^H` (n×n, complex). Satisfies
        /// `U[:, :k] diag(s) Vh[:k, :] == A`. See [`crate::svd`].
        ///
        /// # Errors
        /// - [`FerrayError::InvalidValue`] if the SVD fails to converge.
        pub fn $svd_fn(
            a: &Array<Complex<$T>, Ix2>,
        ) -> FerrayResult<(
            Array<Complex<$T>, Ix2>,
            Array<$T, Ix1>,
            Array<Complex<$T>, Ix2>,
        )> {
            let shape = a.shape();
            let (m, n) = (shape[0], shape[1]);
            let mat = array2c_to_faer(a);
            let decomp = mat.as_ref().svd().map_err(|e| FerrayError::InvalidValue {
                message: format!("complex SVD failed to converge: {e:?}"),
            })?;

            let u = decomp.U();
            let v = decomp.V();
            let s_diag = decomp.S();

            // faer returns the full U (m×m) and V (n×n). numpy's full
            // `svd(full_matrices=True)` returns U (m×m), Vh (n×n).
            let (um, un) = u.shape();
            let mut u_data: Vec<Complex<$T>> = Vec::with_capacity(um * un);
            for i in 0..um {
                for j in 0..un {
                    u_data.push(u[(i, j)]);
                }
            }

            let min_dim = m.min(n);
            let mut s_vals: Vec<$T> = Vec::with_capacity(min_dim);
            for i in 0..min_dim {
                // faer stores the singular values of a complex SVD in the
                // complex field with a zero imaginary part; the mathematical
                // singular values are the real magnitudes. numpy returns the
                // REAL `s` (numpy/linalg/_linalg.py:1810 `s` is float).
                s_vals.push(s_diag.column_vector()[i].re);
            }

            // numpy returns Vh = V^H (conjugate transpose of V).
            let (vn, vk) = v.shape();
            let mut vh_data: Vec<Complex<$T>> = Vec::with_capacity(vk * vn);
            for i in 0..vk {
                for j in 0..vn {
                    let z = v[(j, i)];
                    vh_data.push(Complex::new(z.re, -z.im));
                }
            }

            let u_arr = Array::from_vec(Ix2::new([um, un]), u_data)?;
            let s_arr = Array::from_vec(Ix1::new([min_dim]), s_vals)?;
            let vh_arr = Array::from_vec(Ix2::new([vk, vn]), vh_data)?;
            Ok((u_arr, s_arr, vh_arr))
        }

        /// Reduced QR decomposition of a complex matrix over
        #[doc = concat!("`Complex<", stringify!($T), ">`. Returns `(Q, R)` with")]
        /// `Q` (m×k, complex, columns orthonormal — `Q^H Q == I`) and `R`
        /// (k×n, complex, upper-triangular), `k = min(m, n)`. Satisfies
        /// `Q R == A`. See [`crate::qr`].
        ///
        /// # Errors
        /// - [`FerrayError::InvalidValue`] on allocation failure inside faer.
        pub fn $qr_fn(
            a: &Array<Complex<$T>, Ix2>,
        ) -> FerrayResult<(Array<Complex<$T>, Ix2>, Array<Complex<$T>, Ix2>)> {
            let shape = a.shape();
            let (m, n) = (shape[0], shape[1]);
            let mat = array2c_to_faer(a);
            let decomp = mat.as_ref().qr();

            let q = decomp.compute_thin_Q();
            let r_mat = decomp.thin_R();

            let (qm, qk) = q.shape();
            let mut q_data: Vec<Complex<$T>> = Vec::with_capacity(qm * qk);
            for i in 0..qm {
                for j in 0..qk {
                    q_data.push(q[(i, j)]);
                }
            }

            let (rk, rn) = r_mat.shape();
            let mut r_data: Vec<Complex<$T>> = Vec::with_capacity(rk * rn);
            for i in 0..rk {
                for j in 0..rn {
                    r_data.push(r_mat[(i, j)]);
                }
            }
            // Silence unused-variable lints on m/n for the documented shapes.
            debug_assert!(qm == m && rn == n);

            let q_arr = Array::from_vec(Ix2::new([qm, qk]), q_data)?;
            let r_arr = Array::from_vec(Ix2::new([rk, rn]), r_data)?;
            Ok((q_arr, r_arr))
        }
    };
}

impl_complex_svd_qr!(svd_complex_f64, qr_complex_f64, f64);
impl_complex_svd_qr!(svd_complex_f32, qr_complex_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;
    use num_complex::Complex;

    fn c64(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    fn arr2(rows: usize, cols: usize, data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix2> {
        Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
    }

    #[test]
    fn eigvals_complex_matches_trace_and_det() {
        // M = [[1+1j, 2-1j], [3+0j, 4+2j]]. Eigenvalue invariants are exact:
        // sum(λ) == trace(M) = (5+3j), prod(λ) == det(M) = (-4+9j). Checks the
        // complex eigensolver without copying ferray's own output.
        let m = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)],
        );
        let res = eigvals_complex_f64(&m);
        assert!(res.is_ok(), "eigvals_complex_f64 failed: {res:?}");
        let Ok(w) = res else { return };
        let vals: Vec<Complex<f64>> = w.iter().copied().collect();
        assert_eq!(vals.len(), 2);
        let sum = vals[0] + vals[1];
        let trace = c64(5.0, 3.0);
        assert!(
            (sum - trace).norm() < 1e-10,
            "sum(λ)={sum:?} != trace={trace:?}"
        );
        let prod = vals[0] * vals[1];
        let det = c64(-4.0, 9.0);
        assert!(
            (prod - det).norm() < 1e-9,
            "prod(λ)={prod:?} != det={det:?}"
        );
        // numpy live oracle np.sort_complex(np.linalg.eigvals(M)) (numpy 2.4.4):
        //   [-0.34072265+1.76401733j, 5.34072265+1.23598267j]
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| {
            let r = a.re.total_cmp(&b.re);
            if r == core::cmp::Ordering::Equal {
                a.im.total_cmp(&b.im)
            } else {
                r
            }
        });
        assert!(
            (sorted[0] - c64(-0.34072265, 1.76401733)).norm() < 1e-6,
            "λ0={:?}",
            sorted[0]
        );
        assert!(
            (sorted[1] - c64(5.34072265, 1.23598267)).norm() < 1e-6,
            "λ1={:?}",
            sorted[1]
        );
    }

    #[test]
    fn eig_complex_av_equals_lambda_v() {
        // A v == λ v for every returned eigenpair (column j of V).
        let m = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)],
        );
        let res = eig_complex_f64(&m);
        assert!(res.is_ok(), "eig_complex_f64 failed: {res:?}");
        let Ok((w, v)) = res else { return };
        let vals: Vec<Complex<f64>> = w.iter().copied().collect();
        let vd: Vec<Complex<f64>> = v.iter().copied().collect(); // row-major 2x2
        let mdata = [c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)];
        for j in 0..2 {
            // eigenvector j is column j: [vd[0*2+j], vd[1*2+j]].
            let vj = [vd[j], vd[2 + j]];
            for i in 0..2 {
                let av = mdata[i * 2] * vj[0] + mdata[i * 2 + 1] * vj[1];
                let lv = vals[j] * vj[i];
                assert!((av - lv).norm() < 1e-9, "Av != λv at i={i}, j={j}");
            }
        }
    }

    #[test]
    fn matmul_complex_2x2() {
        // A = [[1+i, 2], [3, 4-i]]
        // B = [[1, 0], [0, 1]]
        // A @ B = A
        let a = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, -1.0)],
        );
        let b = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let c = matmul_complex(&a, &b).unwrap();
        let data: Vec<Complex<f64>> = c.iter().copied().collect();
        assert_eq!(
            data,
            vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, -1.0)]
        );
    }

    #[test]
    fn matmul_complex_nontrivial() {
        // A = [[1+i, 2-i], [3, 1]]
        // B = [[1, 0-i], [i, 1]]
        // AB[0][0] = (1+i)*1 + (2-i)*i = (1+i) + (2i + 1) = 2 + 3i
        // AB[0][1] = (1+i)*(-i) + (2-i)*1 = (1 - i) + (2 - i) = 3 - 2i
        // AB[1][0] = 3*1 + 1*i = 3 + i
        // AB[1][1] = 3*(-i) + 1*1 = 1 - 3i
        let a = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(1.0, 0.0)],
        );
        let b = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, -1.0), c64(0.0, 1.0), c64(1.0, 0.0)],
        );
        let c = matmul_complex(&a, &b).unwrap();
        let d: Vec<Complex<f64>> = c.iter().copied().collect();
        let expected = [c64(2.0, 3.0), c64(3.0, -2.0), c64(3.0, 1.0), c64(1.0, -3.0)];
        for (got, want) in d.iter().zip(expected.iter()) {
            assert!(
                (got.re - want.re).abs() < 1e-12,
                "re: got {got}, want {want}"
            );
            assert!(
                (got.im - want.im).abs() < 1e-12,
                "im: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn matmul_complex_shape_mismatch() {
        let a = arr2(2, 3, vec![c64(0.0, 0.0); 6]);
        let b = arr2(2, 2, vec![c64(0.0, 0.0); 4]);
        assert!(matmul_complex(&a, &b).is_err());
    }

    #[test]
    fn inv_complex_identity() {
        let i = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let inv = inv_complex(&i).unwrap();
        let d: Vec<Complex<f64>> = inv.iter().copied().collect();
        assert_eq!(
            d,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)]
        );
    }

    #[test]
    fn inv_complex_diag_i() {
        // diag(i, 2i) — inverse is diag(-i, -i/2)
        let a = arr2(
            2,
            2,
            vec![c64(0.0, 1.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 2.0)],
        );
        let inv = inv_complex(&a).unwrap();
        let d: Vec<Complex<f64>> = inv.iter().copied().collect();
        assert!((d[0] - c64(0.0, -1.0)).norm() < 1e-12);
        assert_eq!(d[1], c64(0.0, 0.0));
        assert_eq!(d[2], c64(0.0, 0.0));
        assert!((d[3] - c64(0.0, -0.5)).norm() < 1e-12);
    }

    #[test]
    fn solve_complex_identity_rhs() {
        // I * x = b  =>  x = b
        let i = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let b = arr2(2, 1, vec![c64(3.0, 4.0), c64(5.0, -6.0)]);
        let x = solve_complex(&i, &b).unwrap();
        let d: Vec<Complex<f64>> = x.iter().copied().collect();
        assert_eq!(d, vec![c64(3.0, 4.0), c64(5.0, -6.0)]);
    }

    #[test]
    fn solve_complex_vec_identity() {
        use ferray_core::dimension::Ix1;
        let i = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let b =
            Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([2]), vec![c64(1.0, 2.0), c64(3.0, 4.0)])
                .unwrap();
        let x = solve_complex_vec(&i, &b).unwrap();
        let d: Vec<Complex<f64>> = x.iter().copied().collect();
        assert_eq!(d, vec![c64(1.0, 2.0), c64(3.0, 4.0)]);
    }

    #[test]
    fn det_complex_diag() {
        // det(diag(2+i, 3-i)) = (2+i)*(3-i) = 6 - 2i + 3i - i^2 = 7 + i
        let a = arr2(
            2,
            2,
            vec![c64(2.0, 1.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(3.0, -1.0)],
        );
        let d = det_complex(&a).unwrap();
        assert!((d - c64(7.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn matmul_inv_is_identity() {
        // A = [[2+i, 1], [1, 2-i]], compute A * A^-1 and check it's I.
        let a = arr2(
            2,
            2,
            vec![c64(2.0, 1.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(2.0, -1.0)],
        );
        let inv = inv_complex(&a).unwrap();
        let prod = matmul_complex(&a, &inv).unwrap();
        let d: Vec<Complex<f64>> = prod.iter().copied().collect();
        assert!((d[0] - c64(1.0, 0.0)).norm() < 1e-12);
        assert!(d[1].norm() < 1e-12);
        assert!(d[2].norm() < 1e-12);
        assert!((d[3] - c64(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn det_non_square_errors() {
        let a = arr2(2, 3, vec![c64(0.0, 0.0); 6]);
        assert!(det_complex(&a).is_err());
    }

    #[test]
    fn inv_non_square_errors() {
        let a = arr2(2, 3, vec![c64(0.0, 0.0); 6]);
        assert!(inv_complex(&a).is_err());
    }

    // ----- complex SVD (#939) -------------------------------------------

    #[test]
    fn svd_complex_reconstructs() {
        // M = [[1+1j, 2-1j], [3+0j, 4+2j]]. Verify A == U diag(s) Vh and that
        // s are real and nonnegative (singular values are canonical even
        // though U/Vh phases are not).
        let m = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)],
        );
        let res = svd_complex_f64(&m);
        assert!(res.is_ok(), "svd_complex_f64 failed: {res:?}");
        let Ok((u, s, vh)) = res else { return };
        let ud: Vec<Complex<f64>> = u.iter().copied().collect();
        let sd: Vec<f64> = s.iter().copied().collect();
        let vhd: Vec<Complex<f64>> = vh.iter().copied().collect();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(vh.shape(), &[2, 2]);
        assert_eq!(sd.len(), 2);
        for &sv in &sd {
            assert!(sv >= 0.0, "singular value {sv} must be >= 0");
        }
        assert!(sd[0] >= sd[1], "singular values must be descending");
        let mdata = [c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = c64(0.0, 0.0);
                for p in 0..2 {
                    let up = ud[i * 2 + p];
                    let vp = vhd[p * 2 + j];
                    acc += up * Complex::new(sd[p], 0.0) * vp;
                }
                assert!(
                    (acc - mdata[i * 2 + j]).norm() < 1e-10,
                    "U diag(s) Vh[{i},{j}] = {acc} != {}",
                    mdata[i * 2 + j]
                );
            }
        }
        // numpy live oracle: np.linalg.svd(M).S (numpy 2.4) descending:
        //   [5.75034948, 1.71274074]
        assert!((sd[0] - 5.75034948).abs() < 1e-6, "s0={}", sd[0]);
        assert!((sd[1] - 1.71274074).abs() < 1e-6, "s1={}", sd[1]);
    }

    #[test]
    fn svd_complex_u_unitary() {
        // U columns are orthonormal: U^H U == I.
        let m = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)],
        );
        let res = svd_complex_f64(&m);
        assert!(res.is_ok(), "svd_complex_f64 failed: {res:?}");
        let Ok((u, _s, _vh)) = res else { return };
        let ud: Vec<Complex<f64>> = u.iter().copied().collect();
        for a in 0..2 {
            for b in 0..2 {
                let mut acc = c64(0.0, 0.0);
                for i in 0..2 {
                    let ua = ud[i * 2 + a]; // U[i,a]
                    let ub = ud[i * 2 + b]; // U[i,b]
                    acc += Complex::new(ua.re, -ua.im) * ub; // conj(U[i,a]) U[i,b]
                }
                let want = if a == b { c64(1.0, 0.0) } else { c64(0.0, 0.0) };
                assert!((acc - want).norm() < 1e-10, "U^H U[{a},{b}] = {acc}");
            }
        }
    }

    #[test]
    fn svd_complex_f32_reconstructs() {
        let m_res = Array::<Complex<f32>, Ix2>::from_vec(
            Ix2::new([2, 2]),
            vec![
                Complex::new(1.0f32, 1.0),
                Complex::new(2.0, -1.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 2.0),
            ],
        );
        assert!(m_res.is_ok());
        let Ok(m) = m_res else { return };
        let res = svd_complex_f32(&m);
        assert!(res.is_ok(), "svd_complex_f32 failed: {res:?}");
        let Ok((u, s, vh)) = res else { return };
        let ud: Vec<Complex<f32>> = u.iter().copied().collect();
        let sd: Vec<f32> = s.iter().copied().collect();
        let vhd: Vec<Complex<f32>> = vh.iter().copied().collect();
        let mdata = [
            Complex::new(1.0f32, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 2.0),
        ];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = Complex::new(0.0f32, 0.0);
                for p in 0..2 {
                    acc += ud[i * 2 + p] * Complex::new(sd[p], 0.0) * vhd[p * 2 + j];
                }
                assert!(
                    (acc - mdata[i * 2 + j]).norm() < 1e-4,
                    "f32 reconstruct[{i},{j}]"
                );
            }
        }
    }

    // ----- complex QR (#939) --------------------------------------------

    #[test]
    fn qr_complex_reconstructs_and_unitary() {
        // Q R == A and Q^H Q == I.
        let m = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)],
        );
        let res = qr_complex_f64(&m);
        assert!(res.is_ok(), "qr_complex_f64 failed: {res:?}");
        let Ok((q, r)) = res else { return };
        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        let qd: Vec<Complex<f64>> = q.iter().copied().collect();
        let rd: Vec<Complex<f64>> = r.iter().copied().collect();
        let mdata = [c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(4.0, 2.0)];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = c64(0.0, 0.0);
                for p in 0..2 {
                    acc += qd[i * 2 + p] * rd[p * 2 + j];
                }
                assert!(
                    (acc - mdata[i * 2 + j]).norm() < 1e-10,
                    "Q R[{i},{j}] = {acc} != {}",
                    mdata[i * 2 + j]
                );
            }
        }
        for a in 0..2 {
            for b in 0..2 {
                let mut acc = c64(0.0, 0.0);
                for i in 0..2 {
                    let qa = qd[i * 2 + a];
                    let qb = qd[i * 2 + b];
                    acc += Complex::new(qa.re, -qa.im) * qb;
                }
                let want = if a == b { c64(1.0, 0.0) } else { c64(0.0, 0.0) };
                assert!((acc - want).norm() < 1e-10, "Q^H Q[{a},{b}] = {acc}");
            }
        }
        // R is upper-triangular: R[1,0] == 0.
        assert!(rd[2].norm() < 1e-10, "R[1,0] = {} (must be 0)", rd[2]);
    }

    #[test]
    fn qr_complex_tall_thin() {
        // 3x2 complex matrix: reduced QR has Q (3x2), R (2x2).
        let m = arr2(
            3,
            2,
            vec![
                c64(1.0, 1.0),
                c64(2.0, 0.0),
                c64(0.0, 1.0),
                c64(3.0, -1.0),
                c64(1.0, 0.0),
                c64(0.0, 2.0),
            ],
        );
        let res = qr_complex_f64(&m);
        assert!(res.is_ok(), "qr_complex_f64 failed: {res:?}");
        let Ok((q, r)) = res else { return };
        assert_eq!(q.shape(), &[3, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        let qd: Vec<Complex<f64>> = q.iter().copied().collect();
        let rd: Vec<Complex<f64>> = r.iter().copied().collect();
        let mdata: Vec<Complex<f64>> = m.iter().copied().collect();
        for i in 0..3 {
            for j in 0..2 {
                let mut acc = c64(0.0, 0.0);
                for p in 0..2 {
                    acc += qd[i * 2 + p] * rd[p * 2 + j];
                }
                assert!((acc - mdata[i * 2 + j]).norm() < 1e-10, "tall Q R[{i},{j}]");
            }
        }
    }
}
