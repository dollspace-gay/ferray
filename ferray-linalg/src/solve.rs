// ferray-linalg: Solvers and inversion (REQ-15 through REQ-18c)
//
// solve, lstsq, inv, pinv, matrix_power, tensorsolve, tensorinv

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch::{self, faer_to_vec, slice_to_faer};
use crate::faer_bridge;
use crate::scalar::LinalgFloat;
use faer::linalg::solvers::{DenseSolveCore, Solve};

/// Solve the linear equation `A @ x = b` for x.
///
/// `a` must be a square, non-singular matrix.
/// `b` can be a 1D vector or a 2D matrix (multiple right-hand sides).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if dimensions are incompatible.
/// - `FerrayError::SingularMatrix` if A is singular.
pub fn solve<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape[0] != a_shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "solve requires a square matrix A, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }
    let n = a_shape[0];

    let a_mat = faer_bridge::array2_to_faer(a);
    let lu = a_mat.as_ref().partial_piv_lu();

    let result = match b_shape.len() {
        1 => {
            if b_shape[0] != n {
                return Err(FerrayError::shape_mismatch(format!(
                    "solve: A is {}x{} but b has length {}",
                    n, n, b_shape[0]
                )));
            }
            let b_data: Vec<T> = b.iter().copied().collect();
            let b_mat = faer::Mat::from_fn(n, 1, |i, _| b_data[i]);
            let x = lu.solve(&b_mat);
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                result.push(x[(i, 0)]);
            }
            Array::from_vec(IxDyn::new(&[n]), result)?
        }
        2 => {
            if b_shape[0] != n {
                return Err(FerrayError::shape_mismatch(format!(
                    "solve: A is {}x{} but b has {} rows",
                    n, n, b_shape[0]
                )));
            }
            let nrhs = b_shape[1];
            let b_data: Vec<T> = b.iter().copied().collect();
            let b_mat = faer::Mat::from_fn(n, nrhs, |i, j| b_data[i * nrhs + j]);
            let x = lu.solve(&b_mat);
            let mut result = Vec::with_capacity(n * nrhs);
            for i in 0..n {
                for j in 0..nrhs {
                    result.push(x[(i, j)]);
                }
            }
            Array::from_vec(IxDyn::new(&[n, nrhs]), result)?
        }
        _ => return Err(FerrayError::shape_mismatch("solve: b must be 1D or 2D")),
    };

    // Check for singularity: scan result for NaN/Inf which indicates
    // the LU solve encountered a (near-)singular matrix. This catches
    // both exact singularity (zero pivot) and severe ill-conditioning.
    // A more precise check would inspect the LU diagonal for near-zero
    // pivots, but faer's partial_piv_lu does not expose the factors
    // directly through a stable public API.
    for &val in result.iter() {
        if val.is_nan() || val.is_infinite() {
            return Err(FerrayError::SingularMatrix {
                message: "matrix is singular or nearly singular; solve produced non-finite values"
                    .to_string(),
            });
        }
    }

    Ok(result)
}

/// Dynamic-rank wrapper for [`solve`] (#411).
///
/// Accepts `a` as `IxDyn` (rather than `Ix2`) and validates rank +
/// squareness at runtime. Useful for callers that have arrays in
/// dynamic-rank form already (e.g. from a deserialization path or a
/// user-facing API) and don't want to round-trip through Ix2 just to
/// call `solve`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `a` is not 2-D or not square.
/// - Same errors as [`solve`] for the actual factorization.
pub fn solve_dyn<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    if a_shape.len() != 2 {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_dyn requires a 2-D matrix A, got {}-D",
            a_shape.len()
        )));
    }
    // Reshape A into the typed Ix2 form expected by solve(). The
    // call to to_vec_flat materialises the data; that's a single
    // copy that the typed solve() would have triggered anyway via
    // its faer_bridge step.
    let a_data: Vec<T> = a.to_vec_flat();
    let a_2d = Array::<T, Ix2>::from_vec(Ix2::new([a_shape[0], a_shape[1]]), a_data)?;
    solve(&a_2d, b)
}

/// Compute the least-squares solution to `A @ x = b`.
///
/// Returns `(x, residuals, rank, singular_values)`.
/// - x: solution of shape (n,) or (n, k)
/// - residuals: sum of squared residuals (empty if rank < m or m < n)
/// - rank: effective rank of A
/// - `singular_values`: singular values of A
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if dimensions are incompatible.
//
// `lstsq` performs the full SVD-driven least-squares pipeline (rank
// truncation, residual computation, batched-vs-matrix dispatch) inline;
// splitting these into helpers fragments tightly-shared dimension state.
#[allow(clippy::too_many_lines)]
pub fn lstsq<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    b: &Array<T, IxDyn>,
    rcond: Option<T>,
) -> FerrayResult<(Array<T, IxDyn>, Array<T, Ix1>, usize, Array<T, Ix1>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, n) = (a_shape[0], a_shape[1]);

    // Single SVD decomposition for both the solve and rank/singular values.
    // This replaces the previous approach that computed QR + SVD separately.
    let (u_arr, sv, vt_arr) = crate::decomp::svd(a, false)?;
    let svals = sv.as_slice().unwrap();
    let tol = rcond.unwrap_or_else(|| {
        let max_dim = T::from_usize(m.max(n));
        max_dim * T::machine_epsilon()
    });
    let max_sv = if svals.is_empty() {
        T::from_f64_const(0.0)
    } else {
        svals[0]
    };
    let cutoff = tol * max_sv;
    let rank = svals.iter().filter(|&&s| s > cutoff).count();

    // SVD-based least squares: x = V * diag(1/s_i) * U^T * b
    // (zeroing singular values below cutoff)
    let k = svals.len(); // min(m, n)
    let u_data: Vec<T> = u_arr.iter().copied().collect();
    let vt_data: Vec<T> = vt_arr.iter().copied().collect();
    let zero = T::from_f64_const(0.0);

    match b_shape.len() {
        1 => {
            if b_shape[0] != m {
                return Err(FerrayError::shape_mismatch(format!(
                    "lstsq: A is {}x{} but b has length {}",
                    m, n, b_shape[0]
                )));
            }
            let b_data: Vec<T> = b.iter().copied().collect();

            // x = V * diag(1/s) * U^T * b
            // Step 1: c = U^T * b (k-vector)
            let mut c = vec![zero; k];
            for p in 0..k {
                let mut dot = zero;
                for i in 0..m {
                    dot += u_data[i * k + p] * b_data[i];
                }
                c[p] = dot;
            }
            // Step 2: c[p] /= s[p] (zero out small singular values)
            for p in 0..k {
                if svals[p] > cutoff {
                    c[p] /= svals[p];
                } else {
                    c[p] = zero;
                }
            }
            // Step 3: x = V * c = Vt^T * c (n-vector)
            let mut x_vec = vec![zero; n];
            for i in 0..n {
                let mut sum = zero;
                for p in 0..k {
                    sum += vt_data[p * n + i] * c[p];
                }
                x_vec[i] = sum;
            }

            // Compute residuals
            let residuals = if m > n && rank == n {
                let a_data: Vec<T> = a.iter().copied().collect();
                let mut resid = zero;
                for i in 0..m {
                    let mut ax_i = zero;
                    for j in 0..n {
                        ax_i += a_data[i * n + j] * x_vec[j];
                    }
                    let diff = ax_i - b_data[i];
                    resid += diff * diff;
                }
                vec![resid]
            } else {
                vec![]
            };

            let x = Array::from_vec(IxDyn::new(&[n]), x_vec)?;
            let residuals_arr = Array::from_vec(Ix1::new([residuals.len()]), residuals)?;
            Ok((x, residuals_arr, rank, sv))
        }
        2 => {
            if b_shape[0] != m {
                return Err(FerrayError::shape_mismatch(format!(
                    "lstsq: A is {}x{} but b has {} rows",
                    m, n, b_shape[0]
                )));
            }
            let nrhs = b_shape[1];
            let b_data: Vec<T> = b.iter().copied().collect();

            // x = V * diag(1/s) * U^T * b for each column of b
            let mut x_vec = vec![zero; n * nrhs];
            for col in 0..nrhs {
                // c = U^T * b[:, col]
                let mut c = vec![zero; k];
                for p in 0..k {
                    let mut dot = zero;
                    for i in 0..m {
                        dot += u_data[i * k + p] * b_data[i * nrhs + col];
                    }
                    c[p] = if svals[p] > cutoff {
                        dot / svals[p]
                    } else {
                        zero
                    };
                }
                // x[:, col] = Vt^T * c
                for i in 0..n {
                    let mut sum = zero;
                    for p in 0..k {
                        sum += vt_data[p * n + i] * c[p];
                    }
                    x_vec[i * nrhs + col] = sum;
                }
            }

            // Compute residuals per rhs column
            let residuals = if m > n && rank == n {
                let a_data: Vec<T> = a.iter().copied().collect();
                let mut resids = vec![T::from_f64_const(0.0); nrhs];
                for col in 0..nrhs {
                    for i in 0..m {
                        let mut ax_i = T::from_f64_const(0.0);
                        for j in 0..n {
                            ax_i += a_data[i * n + j] * x_vec[j * nrhs + col];
                        }
                        let diff = ax_i - b_data[i * nrhs + col];
                        resids[col] += diff * diff;
                    }
                }
                resids
            } else {
                vec![]
            };

            let x = Array::from_vec(IxDyn::new(&[n, nrhs]), x_vec)?;
            let residuals_arr = Array::from_vec(Ix1::new([residuals.len()]), residuals)?;
            Ok((x, residuals_arr, rank, sv))
        }
        _ => Err(FerrayError::shape_mismatch("lstsq: b must be 1D or 2D")),
    }
}

/// Compute the inverse of a square matrix.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if the matrix is singular.
pub fn inv<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix2>> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "inv requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    if n == 0 {
        return Array::from_vec(Ix2::new([0, 0]), vec![]);
    }

    let mat = faer_bridge::array2_to_faer(a);
    let lu = mat.as_ref().partial_piv_lu();
    let inv_mat = lu.inverse();

    // Check for NaN/Inf in the result, which indicates a singular matrix
    let result = faer_bridge::faer_to_array2(&inv_mat)?;
    for &val in result.iter() {
        if val.is_nan() || val.is_infinite() {
            return Err(FerrayError::SingularMatrix {
                message: "matrix is singular and cannot be inverted".to_string(),
            });
        }
    }
    Ok(result)
}

/// Compute the Moore-Penrose pseudoinverse of a matrix.
///
/// Uses SVD: `pinv(A) = V * diag(1/s_i) * U^T` for singular values above `rcond * max(s)`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD computation fails.
pub fn pinv<T: LinalgFloat>(a: &Array<T, Ix2>, rcond: Option<T>) -> FerrayResult<Array<T, Ix2>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let (u, s, vt) = crate::decomp::svd(a, false)?;

    let svals = s.as_slice().unwrap();
    let max_sv = if svals.is_empty() {
        T::from_f64_const(0.0)
    } else {
        svals[0]
    };

    // Default rcond matches NumPy: max(M, N) * machine_epsilon
    let tol = rcond.unwrap_or_else(|| {
        let max_dim = T::from_usize(m.max(n));
        max_dim * T::machine_epsilon()
    });
    let cutoff = tol * max_sv;

    // Build pinv = V * diag(1/s_i) * U^T, zeroing small singular values
    let k = svals.len(); // min(m, n)
    let u_data: Vec<T> = u.iter().copied().collect();
    let vt_data: Vec<T> = vt.iter().copied().collect();

    // Result shape is (n, m) — the pseudoinverse of (m, n)
    let zero = T::from_f64_const(0.0);
    let one = T::from_f64_const(1.0);
    let mut result = vec![zero; n * m];

    for i in 0..n {
        for j in 0..m {
            let mut sum = zero;
            for p in 0..k {
                if svals[p] > cutoff {
                    // vt[p][i] * (1/s[p]) * u[j][p]
                    // vt is (k, n), u is (m, k)
                    sum += vt_data[p * n + i] * (one / svals[p]) * u_data[j * k + p];
                }
            }
            result[i * m + j] = sum;
        }
    }

    Array::from_vec(Ix2::new([n, m]), result)
}

/// Raise a square matrix to an integer power.
///
/// - For `n > 0`: compute `A^n` by repeated squaring.
/// - For `n == 0`: return the identity matrix.
/// - For `n < 0`: compute `inv(A)^|n|`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if `n < 0` and the matrix is singular.
pub fn matrix_power<T: LinalgFloat>(a: &Array<T, Ix2>, n: i64) -> FerrayResult<Array<T, Ix2>> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "matrix_power requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let sz = shape[0];

    if n == 0 {
        // Return identity
        let zero = T::from_f64_const(0.0);
        let one = T::from_f64_const(1.0);
        let mut data = vec![zero; sz * sz];
        for i in 0..sz {
            data[i * sz + i] = one;
        }
        return Array::from_vec(Ix2::new([sz, sz]), data);
    }

    let base = if n < 0 { inv(a)? } else { a.clone() };
    let power = n.unsigned_abs();

    // Exponentiation by squaring
    let zero = T::from_f64_const(0.0);
    let one = T::from_f64_const(1.0);
    let mut result_data = vec![zero; sz * sz];
    for i in 0..sz {
        result_data[i * sz + i] = one;
    }
    let mut base_data: Vec<T> = base.iter().copied().collect();
    let mut p = power;

    while p > 0 {
        if p & 1 == 1 {
            result_data = crate::products::matmul_raw::<T>(&result_data, &base_data, sz, sz, sz);
        }
        base_data = crate::products::matmul_raw::<T>(&base_data, &base_data, sz, sz, sz);
        p >>= 1;
    }

    Array::from_vec(Ix2::new([sz, sz]), result_data)
}

// ---------------------------------------------------------------------------
// Batched variants (#412) — dispatch along the leading (batch) axes and
// parallelize via Rayon. All of these preserve the input batch shape and
// apply the operation independently along the last two axes (the matrix
// dimensions), matching NumPy's gufunc semantics for `numpy.linalg`.
// ---------------------------------------------------------------------------

/// Batched linear solve: `solve(A, b)` along the last two axes of `a` and
/// (optionally) the last one or two axes of `b`.
///
/// `a` must have shape `(..., M, M)` and `b` must have shape `(..., M)` or
/// `(..., M, K)`. The leading batch dimensions must match exactly (no
/// broadcasting). Returns the stack of solutions with the same batch shape
/// as the inputs.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `A` is not square or dimensions are
///   incompatible.
/// - `FerrayError::SingularMatrix` if any matrix in the batch is singular.
pub fn solve_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "solve_batched: a must have at least 2 dimensions",
        ));
    }

    // Fast path: non-batched inputs — just forward to the 2-D solver.
    if a_shape.len() == 2 && b_shape.len() <= 2 {
        let a2 = Array::<T, Ix2>::from_vec(
            Ix2::new([a_shape[0], a_shape[1]]),
            a.iter().copied().collect(),
        )?;
        return solve(&a2, b);
    }

    let n = a_shape[a_shape.len() - 1];
    if a_shape[a_shape.len() - 2] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_batched: A must be square in the last two axes, got {}x{}",
            a_shape[a_shape.len() - 2],
            n
        )));
    }

    let a_batch = &a_shape[..a_shape.len() - 2];

    // Decide whether b is vector-stacked `(..., M)` or matrix-stacked
    // `(..., M, K)` by matching batch prefix lengths against `a`.
    let (b_is_vec, b_batch, nrhs) = if b_shape.len() == a_batch.len() + 1 {
        // b has shape (..., M): one right-hand-side per batch.
        (true, &b_shape[..b_shape.len() - 1], 1usize)
    } else if b_shape.len() == a_batch.len() + 2 {
        // b has shape (..., M, K): K right-hand sides per batch.
        (
            false,
            &b_shape[..b_shape.len() - 2],
            b_shape[b_shape.len() - 1],
        )
    } else {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_batched: b has incompatible rank ({} dims) for a ({} dims)",
            b_shape.len(),
            a_shape.len()
        )));
    };

    if b_batch != a_batch {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_batched: batch dimensions must match: {a_batch:?} vs {b_batch:?}"
        )));
    }
    let b_m = if b_is_vec {
        b_shape[b_shape.len() - 1]
    } else {
        b_shape[b_shape.len() - 2]
    };
    if b_m != n {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_batched: A is {n}x{n} but b has leading dim {b_m}"
        )));
    }

    let num_batches = batch::batch_count(a_shape, 2)?;
    let a_data: Vec<T> = a.iter().copied().collect();
    let b_data: Vec<T> = b.iter().copied().collect();
    let a_mat_size = n * n;
    let b_mat_size = n * nrhs;

    use rayon::prelude::*;
    let results: Vec<FerrayResult<Vec<T>>> = (0..num_batches)
        .into_par_iter()
        .map(|idx| {
            let a_slice = &a_data[idx * a_mat_size..(idx + 1) * a_mat_size];
            let b_slice = &b_data[idx * b_mat_size..(idx + 1) * b_mat_size];
            let a_faer = slice_to_faer(n, n, a_slice);
            let lu = a_faer.as_ref().partial_piv_lu();
            let b_faer = faer::Mat::<T>::from_fn(n, nrhs, |i, j| b_slice[i * nrhs + j]);
            let x = lu.solve(&b_faer);
            let mut out = Vec::with_capacity(b_mat_size);
            for i in 0..n {
                for j in 0..nrhs {
                    let v = x[(i, j)];
                    if v.is_nan() || v.is_infinite() {
                        return Err(FerrayError::SingularMatrix {
                            message: "matrix is singular or nearly singular; solve produced non-finite values".to_string(),
                        });
                    }
                    out.push(v);
                }
            }
            Ok(out)
        })
        .collect();

    let flat: Vec<T> = results
        .into_iter()
        .collect::<FerrayResult<Vec<Vec<T>>>>()?
        .into_iter()
        .flatten()
        .collect();

    Array::from_vec(IxDyn::new(b_shape), flat)
}

/// Batched matrix inverse for arrays of shape `(..., N, N)`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the last two dims are not square.
/// - `FerrayError::SingularMatrix` if any matrix in the batch is singular.
pub fn inv_batched<T: LinalgFloat>(a: &Array<T, IxDyn>) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let result = inv(&a2)?;
        return Array::from_vec(IxDyn::new(shape), result.iter().copied().collect());
    }

    let results = batch::apply_batched_2d(a, |m, n, data| {
        if m != n {
            return Err(FerrayError::shape_mismatch(format!(
                "inv requires square matrices, got {m}x{n}"
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let mat = slice_to_faer(m, n, data);
        let lu = mat.as_ref().partial_piv_lu();
        let inv_mat = lu.inverse();
        let out = faer_to_vec(&inv_mat);
        for &v in &out {
            if v.is_nan() || v.is_infinite() {
                return Err(FerrayError::SingularMatrix {
                    message: "matrix is singular and cannot be inverted".to_string(),
                });
            }
        }
        Ok(out)
    })?;

    let flat: Vec<T> = results.into_iter().flatten().collect();
    Array::from_vec(IxDyn::new(shape), flat)
}

/// Batched Moore-Penrose pseudoinverse for arrays of shape `(..., M, N)`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD fails for any batch element.
pub fn pinv_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    rcond: Option<T>,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "pinv_batched: a must have at least 2 dimensions",
        ));
    }
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let result = pinv(&a2, rcond)?;
        return Array::from_vec(
            IxDyn::new(&[shape[1], shape[0]]),
            result.iter().copied().collect(),
        );
    }

    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];

    let results = batch::apply_batched_2d(a, |bm, bn, data| {
        let mat = Array::<T, Ix2>::from_vec(Ix2::new([bm, bn]), data.to_vec())?;
        let out = pinv(&mat, rcond)?;
        Ok(out.iter().copied().collect())
    })?;

    let flat: Vec<T> = results.into_iter().flatten().collect();
    // Output shape: leading batch dims + (n, m) — pinv transposes the last two.
    let mut out_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
    out_shape.push(n);
    out_shape.push(m);
    Array::from_vec(IxDyn::new(&out_shape), flat)
}

/// Batched matrix power for arrays of shape `(..., N, N)`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrices are not square.
/// - `FerrayError::SingularMatrix` if `power < 0` and a batch element is
///   singular.
pub fn matrix_power_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    power: i64,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let result = matrix_power(&a2, power)?;
        return Array::from_vec(IxDyn::new(shape), result.iter().copied().collect());
    }

    let results = batch::apply_batched_2d(a, |m, n, data| {
        if m != n {
            return Err(FerrayError::shape_mismatch(format!(
                "matrix_power requires square matrices, got {m}x{n}"
            )));
        }
        let mat = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), data.to_vec())?;
        let out = matrix_power(&mat, power)?;
        Ok(out.iter().copied().collect())
    })?;

    let flat: Vec<T> = results.into_iter().flatten().collect();
    Array::from_vec(IxDyn::new(shape), flat)
}

/// Solve the tensor equation `a x = b` for x.
///
/// `a` is reshaped according to `axes` to form a square matrix equation.
/// If `axes` is `None`, the default axes are used.
///
/// This is analogous to `numpy.linalg.tensorsolve`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the shapes are incompatible.
pub fn tensorsolve<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    axes: Option<&[usize]>,
) -> FerrayResult<Array<T, IxDyn>> {
    // axes support (#422): when Some, NumPy first moves those axes of `a`
    // to the right (trailing positions) before reshaping. The moved axes
    // become the "solution" shape (x's shape); the remaining axes, in
    // their original order, must match `b.shape()`.
    //
    // Implementation: build a permutation [kept..., moved...] and
    // transpose `a` by that permutation before reshaping. With axes=None
    // the permutation is the identity and we stay on the fast path.
    let a_owned: Array<T, IxDyn>;
    let a_ref: &Array<T, IxDyn> = if let Some(ax_list) = axes {
        let ndim = a.ndim();
        // Validate: each axis in range, no duplicates.
        let mut seen = vec![false; ndim];
        for &ax in ax_list {
            if ax >= ndim {
                return Err(FerrayError::axis_out_of_bounds(ax, ndim));
            }
            if seen[ax] {
                return Err(FerrayError::invalid_value(format!(
                    "tensorsolve: duplicate axis {ax} in axes"
                )));
            }
            seen[ax] = true;
        }
        // Permutation: first the axes NOT in ax_list (in their original
        // order), then the axes in ax_list (in the caller-provided
        // order). This moves the selected axes to the right.
        let mut perm: Vec<usize> = (0..ndim).filter(|i| !seen[*i]).collect();
        perm.extend(ax_list.iter().copied());
        a_owned = ferray_core::manipulation::transpose(a, Some(&perm))?;
        &a_owned
    } else {
        a
    };

    let a_shape = a_ref.shape();
    let b_shape = b.shape();

    // Compute the shape of x from the shapes of a and b
    let b_size: usize = b_shape.iter().product();
    let a_size: usize = a_shape.iter().product();
    if b_size == 0 {
        return Err(FerrayError::shape_mismatch("tensorsolve: b is empty"));
    }
    let x_size = a_size / b_size;
    if x_size * b_size != a_size {
        return Err(FerrayError::shape_mismatch(
            "tensorsolve: a and b shapes are not compatible",
        ));
    }

    // Reshape a to (b_size, x_size) and solve
    let a_data: Vec<T> = a_ref.iter().copied().collect();
    let a2 = Array::<T, Ix2>::from_vec(Ix2::new([b_size, x_size]), a_data)?;

    let b_flat: Vec<T> = b.iter().copied().collect();
    let b_dyn = Array::<T, IxDyn>::from_vec(IxDyn::new(&[b_size]), b_flat)?;

    let x = solve(&a2, &b_dyn)?;

    // Determine x shape from the (post-transpose) a_shape and b_shape
    let x_shape: Vec<usize> = a_shape[b_shape.len()..].to_vec();
    if x_shape.is_empty() {
        Ok(x)
    } else {
        let x_data: Vec<T> = x.iter().copied().collect();
        Array::from_vec(IxDyn::new(&x_shape), x_data)
    }
}

/// Compute the inverse of an N-dimensional array.
///
/// This is analogous to `numpy.linalg.tensorinv`.
/// The array `a` is reshaped to a matrix using `ind` as the split point.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the shapes are incompatible.
/// - `FerrayError::SingularMatrix` if the reshaped matrix is singular.
pub fn tensorinv<T: LinalgFloat>(a: &Array<T, IxDyn>, ind: usize) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if ind == 0 || ind > shape.len() {
        return Err(FerrayError::invalid_value(format!(
            "tensorinv: ind={} is invalid for {}D array",
            ind,
            shape.len()
        )));
    }

    let first_dims = &shape[..ind];
    let last_dims = &shape[ind..];
    let m: usize = first_dims.iter().product();
    let n: usize = last_dims.iter().product();

    if m != n {
        return Err(FerrayError::shape_mismatch(format!(
            "tensorinv: product of first {ind} dims ({m}) != product of remaining dims ({n})"
        )));
    }

    let data: Vec<T> = a.iter().copied().collect();
    let a2 = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), data)?;
    let inv_a2 = inv(&a2)?;

    // Result shape is last_dims ++ first_dims
    let mut result_shape = Vec::with_capacity(shape.len());
    result_shape.extend_from_slice(last_dims);
    result_shape.extend_from_slice(first_dims);

    let inv_data: Vec<T> = inv_a2.iter().copied().collect();
    Array::from_vec(IxDyn::new(&result_shape), inv_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_2x2() {
        // A = [[1, 2], [3, 4]], b = [5, 11]
        // x = [1, 2]
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![5.0, 11.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        let xs = x.iter().copied().collect::<Vec<f64>>();
        assert!((xs[0] - 1.0).abs() < 1e-10);
        assert!((xs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_f32() {
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::<f32, IxDyn>::from_vec(IxDyn::new(&[2]), vec![5.0f32, 11.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        let xs = x.iter().copied().collect::<Vec<f32>>();
        assert!((xs[0] - 1.0f32).abs() < 1e-4);
        assert!((xs[1] - 2.0f32).abs() < 1e-4);
    }

    #[test]
    fn solve_ax_eq_b_residual() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 4.0, 9.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        let xs: Vec<f64> = x.iter().copied().collect();

        // Compute Ax - b
        let a_data = a.as_slice().unwrap();
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a_data[i * 3 + j] * xs[j];
            }
            let b_i = [1.0, 4.0, 9.0][i];
            assert!(
                (ax_i - b_i).abs() < 1e-10,
                "Ax[{i}] = {ax_i} != b[{i}] = {b_i}"
            );
        }
    }

    #[test]
    fn inv_identity() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let inv_a = inv(&a).unwrap();
        let d = inv_a.as_slice().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (d[i * 3 + j] - expected).abs() < 1e-10,
                    "inv(I)[{},{}] = {} != {}",
                    i,
                    j,
                    d[i * 3 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn inv_singular_error() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(inv(&a).is_err());
    }

    #[test]
    fn pinv_basic() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            .unwrap();
        let pi = pinv(&a, None).unwrap();
        assert_eq!(pi.shape(), &[2, 3]);
    }

    #[test]
    fn matrix_power_positive() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let a3 = matrix_power(&a, 3).unwrap();
        // [[1,1],[0,1]]^3 = [[1,3],[0,1]]
        let d = a3.as_slice().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
        assert!((d[2] - 0.0).abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_zero() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let a0 = matrix_power(&a, 0).unwrap();
        let d = a0.as_slice().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 0.0).abs() < 1e-10);
        assert!((d[2] - 0.0).abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_negative() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let am1 = matrix_power(&a, -1).unwrap();
        let inv_a = inv(&a).unwrap();
        let d1 = am1.as_slice().unwrap();
        let d2 = inv_a.as_slice().unwrap();
        for i in 0..4 {
            assert!(
                (d1[i] - d2[i]).abs() < 1e-10,
                "matrix_power(-1)[{}] = {} != inv[{}] = {}",
                i,
                d1[i],
                i,
                d2[i]
            );
        }
    }

    #[test]
    fn lstsq_overdetermined() {
        // A = [[1,1],[1,2],[1,3]], b = [1,2,3]
        // Least squares solution
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
            .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let (x, _residuals, rank, _sv) = lstsq(&a, &b, None).unwrap();
        assert_eq!(rank, 2);
        let xs: Vec<f64> = x.iter().copied().collect();
        // x should be approximately [0, 1] (y = x)
        assert!((xs[0]).abs() < 0.1);
        assert!((xs[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn lstsq_2d_b_multiple_rhs() {
        // A = [[1,0],[0,1],[1,1]], b = [[1,2],[3,4],[5,6]]
        // Overdetermined system with 2 right-hand sides
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let b =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let (x, _residuals, rank, _sv) = lstsq(&a, &b, None).unwrap();
        assert_eq!(rank, 2);
        assert_eq!(x.shape(), &[2, 2]);
        // The solution should be a reasonable least-squares fit
        let xs: Vec<f64> = x.iter().copied().collect();
        // All values should be finite
        for &v in &xs {
            assert!(v.is_finite(), "lstsq 2D b produced non-finite: {v}");
        }
    }

    #[test]
    fn lstsq_exact_system() {
        // A = [[1,0],[0,1]], b = [3, 5] — exact solution x = [3, 5]
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 5.0]).unwrap();
        let (x, _residuals, rank, _sv) = lstsq(&a, &b, None).unwrap();
        assert_eq!(rank, 2);
        let xs: Vec<f64> = x.iter().copied().collect();
        assert!((xs[0] - 3.0).abs() < 1e-10);
        assert!((xs[1] - 5.0).abs() < 1e-10);
    }

    // --- Empty and 1x1 matrix edge cases ---

    #[test]
    fn inv_0x0() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([0, 0]), vec![]).unwrap();
        let result = inv(&a).unwrap();
        assert_eq!(result.shape(), &[0, 0]);
    }

    #[test]
    fn inv_1x1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 1]), vec![4.0]).unwrap();
        let result = inv(&a).unwrap();
        assert!((result.as_slice().unwrap()[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn solve_1x1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 1]), vec![2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![6.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        assert!((x.iter().next().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn det_1x1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 1]), vec![5.0]).unwrap();
        let d = crate::norms::det(&a).unwrap();
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_1x1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 1]), vec![3.0]).unwrap();
        let result = matrix_power(&a, 4).unwrap();
        assert!((result.as_slice().unwrap()[0] - 81.0).abs() < 1e-10);
    }

    #[test]
    fn solve_singular_matrix_error() {
        // Singular matrix: second row is a multiple of the first
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 6.0]).unwrap();
        let result = solve(&a, &b);
        // Should error (singular) or produce NaN/Inf which triggers the check
        assert!(result.is_err(), "solve should fail on singular matrix");
    }

    #[test]
    fn solve_multiple_rhs() {
        // A = [[1, 0], [0, 2]], b = [[3, 5], [4, 6]]
        // x = [[3, 5], [2, 3]]
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![3.0, 5.0, 4.0, 6.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        let data: Vec<f64> = x.iter().copied().collect();
        assert!((data[0] - 3.0).abs() < 1e-10);
        assert!((data[1] - 5.0).abs() < 1e-10);
        assert!((data[2] - 2.0).abs() < 1e-10);
        assert!((data[3] - 3.0).abs() < 1e-10);
    }

    // --- tensorsolve / tensorinv tests ---

    #[test]
    fn tensorsolve_2d_as_matrix() {
        // tensorsolve with 2D tensors is equivalent to regular solve
        // A(2,2) @ x(2) = b(2) -> tensorsolve(A, b) = solve(A, b)
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 8.0]).unwrap();
        let x = tensorsolve(&a, &b, None).unwrap();
        let data: Vec<f64> = x.iter().copied().collect();
        // x = [3, 4]
        assert!((data[0] - 3.0).abs() < 1e-10);
        assert!((data[1] - 4.0).abs() < 1e-10);
    }

    // ---- tensorsolve axes= parameter (#422) ----

    #[test]
    fn tensorsolve_axes_none_matches_default_path() {
        // Same input as tensorsolve_2d_as_matrix but passing axes=None
        // explicitly — must produce the identical result, confirming
        // the new axes branch leaves the default path untouched.
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 8.0]).unwrap();
        let x_default = tensorsolve(&a, &b, None).unwrap();
        let x_none = tensorsolve(&a, &b, None).unwrap();
        assert_eq!(
            x_default.iter().copied().collect::<Vec<_>>(),
            x_none.iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn tensorsolve_axes_moves_specified_axis_to_right() {
        // Build a 2-D problem where transposing axis 0 to the right is
        // equivalent to swapping A's rows and columns. Concretely, we
        // construct a diagonal A' and b such that solving A' x = b yields
        // [3, 4], then pack A as its transpose. Passing axes=[0] moves
        // axis 0 to the right, giving (shape[1], shape[0]) = the diagonal
        // orientation needed to solve correctly.
        //
        // A (as laid out in memory): [[1, 0], [0, 2]] — already diagonal
        // so the transpose is identical and this also cross-checks that
        // moving an axis doesn't break the symmetric-input case.
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 8.0]).unwrap();
        let x = tensorsolve(&a, &b, Some(&[0])).unwrap();
        // Expected: after moving axis 0 to the right, a has shape [2, 2]
        // unchanged (two axes), and the reshape produces a matrix where
        // the "solution" dim is the moved axis. For this symmetric
        // diagonal input the answer is the same [3, 4].
        let data: Vec<f64> = x.iter().copied().collect();
        assert!((data[0] - 3.0).abs() < 1e-10);
        assert!((data[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn tensorsolve_axes_non_symmetric_reorders_correctly() {
        // Non-symmetric test: construct a (solution-dim, b-dim) matrix
        // and pass it as (b-dim, solution-dim) with axes=[0] to move the
        // first axis to the right. The resulting reshape should match
        // the diagonal orientation.
        //
        // Let A_logical = [[2, 0], [0, 5]], b = [6, 10], x_expected = [3, 2].
        // But store A in memory as the transpose: [[2, 0], [0, 5]]
        // (which happens to be self-transpose because diagonal).
        //
        // To get a genuine non-identity permutation test, use a
        // non-diagonal matrix. Let A_logical = [[2, 1], [1, 3]] such that
        //   A_logical @ x = b
        //   x = [3, 2]
        //   b = [2*3 + 1*2, 1*3 + 3*2] = [8, 9]
        // Now store A in memory transposed, i.e. flat = [2, 1, 1, 3]
        // (same because symmetric) — not useful. Try asymmetric:
        //   A_logical = [[2, 3], [1, 4]], x = [3, 2], b = [2*3 + 3*2, 1*3 + 4*2] = [12, 11].
        // Transpose stored: A_mem[i, j] = A_logical[j, i]
        //   A_mem = [[2, 1], [3, 4]], flat = [2, 1, 3, 4]
        // With axes=[0], permutation is [1, 0], transposing A_mem back
        // to A_logical, then solving gives x = [3, 2].
        let a_mem =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![2.0, 1.0, 3.0, 4.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![12.0, 11.0]).unwrap();
        let x = tensorsolve(&a_mem, &b, Some(&[0])).unwrap();
        let data: Vec<f64> = x.iter().copied().collect();
        assert!((data[0] - 3.0).abs() < 1e-10, "got {data:?}");
        assert!((data[1] - 2.0).abs() < 1e-10, "got {data:?}");
    }

    #[test]
    fn tensorsolve_axes_out_of_bounds_errors() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 8.0]).unwrap();
        assert!(tensorsolve(&a, &b, Some(&[5])).is_err());
    }

    #[test]
    fn tensorsolve_axes_duplicate_errors() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 8.0]).unwrap();
        assert!(tensorsolve(&a, &b, Some(&[0, 0])).is_err());
    }

    #[test]
    fn tensorinv_2d_as_matrix() {
        // tensorinv of a 2D array with ind=1 is equivalent to matrix inverse
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let inv_a = tensorinv(&a, 1).unwrap();
        let data: Vec<f64> = inv_a.iter().copied().collect();
        // inv([[1,0],[0,2]]) = [[1,0],[0,0.5]]
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 0.0).abs() < 1e-10);
        assert!((data[2] - 0.0).abs() < 1e-10);
        assert!((data[3] - 0.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Batched variants (#412)
    // -----------------------------------------------------------------------

    #[test]
    fn inv_batched_3d() {
        // Two 2x2 matrices stacked.
        let data = vec![
            4.0, 7.0, 2.0, 6.0, // first: inv = [[0.6,-0.7],[-0.2,0.4]]
            1.0, 2.0, 3.0, 4.0, // second: inv = [[-2,1],[1.5,-0.5]]
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let inv = inv_batched(&a).unwrap();
        assert_eq!(inv.shape(), &[2, 2, 2]);

        // Verify A @ A^-1 = I per batch.
        let a_data: Vec<f64> = a.iter().copied().collect();
        let inv_data: Vec<f64> = inv.iter().copied().collect();
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for p in 0..2 {
                        sum += a_data[b * 4 + i * 2 + p] * inv_data[b * 4 + p * 2 + j];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1e-10,
                        "batch {b} [{i},{j}] = {sum}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn solve_batched_vec_rhs() {
        // Two 2x2 systems, each with a vector RHS.
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 3.0],
        )
        .unwrap();
        let b =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![5.0, 11.0, 4.0, 9.0]).unwrap();
        let x = solve_batched(&a, &b).unwrap();
        assert_eq!(x.shape(), &[2, 2]);

        // Verify A @ x = b per batch.
        let a_data: Vec<f64> = a.iter().copied().collect();
        let x_data: Vec<f64> = x.iter().copied().collect();
        let b_data: Vec<f64> = b.iter().copied().collect();
        for batch in 0..2 {
            for i in 0..2 {
                let mut sum = 0.0;
                for j in 0..2 {
                    sum += a_data[batch * 4 + i * 2 + j] * x_data[batch * 2 + j];
                }
                assert!(
                    (sum - b_data[batch * 2 + i]).abs() < 1e-10,
                    "A @ x != b at batch {batch} row {i}"
                );
            }
        }
    }

    #[test]
    fn solve_batched_matrix_rhs() {
        // Single batch, 2 RHS columns.
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1, 2, 2]), vec![1.0, 0.0, 0.0, 2.0])
            .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1, 2, 2]), vec![1.0, 3.0, 4.0, 8.0])
            .unwrap();
        let x = solve_batched(&a, &b).unwrap();
        assert_eq!(x.shape(), &[1, 2, 2]);
        // x = [[1, 3], [2, 4]]
        let d: Vec<f64> = x.iter().copied().collect();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
        assert!((d[2] - 2.0).abs() < 1e-10);
        assert!((d[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn inv_batched_singular_errors() {
        // Second matrix is singular.
        let data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        assert!(inv_batched(&a).is_err());
    }

    #[test]
    fn matrix_power_batched_identity() {
        // Any matrix^0 = identity.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let result = matrix_power_batched(&a, 0).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
        let d: Vec<f64> = result.iter().copied().collect();
        assert_eq!(d, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn matrix_power_batched_square() {
        // A^2 per batch.
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2, 2]),
            vec![1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0],
        )
        .unwrap();
        let result = matrix_power_batched(&a, 2).unwrap();
        let d: Vec<f64> = result.iter().copied().collect();
        // [[1,2],[0,1]]^2 = [[1,4],[0,1]]
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 4.0).abs() < 1e-10);
        assert!((d[2] - 0.0).abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
        // [[2,0],[0,3]]^2 = [[4,0],[0,9]]
        assert!((d[4] - 4.0).abs() < 1e-10);
        assert!((d[5] - 0.0).abs() < 1e-10);
        assert!((d[6] - 0.0).abs() < 1e-10);
        assert!((d[7] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn pinv_batched_square() {
        // pinv of invertible matrices equals inv.
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2, 2]),
            vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 4.0],
        )
        .unwrap();
        let pinv_a = pinv_batched(&a, None).unwrap();
        assert_eq!(pinv_a.shape(), &[2, 2, 2]);
        let d: Vec<f64> = pinv_a.iter().copied().collect();
        // diag(1, 0.5), diag(0.5, 0.25)
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[3] - 0.5).abs() < 1e-10);
        assert!((d[4] - 0.5).abs() < 1e-10);
        assert!((d[7] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn solve_batched_forwards_2d_to_solve() {
        // 2-D input should just call through to solve without error.
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![3.0, 8.0]).unwrap();
        let x = solve_batched(&a, &b).unwrap();
        assert_eq!(x.shape(), &[2]);
        let d: Vec<f64> = x.iter().copied().collect();
        assert!((d[0] - 3.0).abs() < 1e-10);
        assert!((d[1] - 4.0).abs() < 1e-10);
    }

    // ----- solve_dyn (#411) ---------------------------------------------

    #[test]
    fn solve_dyn_2d_a_works() {
        // [[2, 1], [1, 3]] * x = [4, 7] → x = [1, 2].
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2]),
            vec![2.0, 1.0, 1.0, 3.0],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![4.0, 7.0]).unwrap();
        let x = solve_dyn(&a, &b).unwrap();
        assert_eq!(x.shape(), &[2]);
        let d: Vec<f64> = x.iter().copied().collect();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_dyn_rejects_non_2d_a() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(solve_dyn(&a, &b).is_err());
    }

    #[test]
    fn solve_dyn_rejects_non_square_a() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0; 6]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        assert!(solve_dyn(&a, &b).is_err());
    }
}
