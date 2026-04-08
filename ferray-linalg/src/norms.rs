// ferray-linalg: Norms and measures (REQ-19 through REQ-22)
//
// norm, cond, det, slogdet, matrix_rank, trace

use ferray_core::array::owned::Array;
use ferray_core::array::view::ArrayView;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch;
use crate::faer_bridge;
use crate::scalar::LinalgFloat;

/// Specifies the type of matrix or vector norm to compute.
#[derive(Debug, Clone, Copy)]
pub enum NormOrder {
    /// Frobenius norm (default for matrices): sqrt(sum of squares).
    Fro,
    /// Nuclear norm: sum of singular values.
    Nuc,
    /// Infinity norm: max row sum (for matrices), max abs (for vectors).
    Inf,
    /// Negative infinity norm: min row sum (for matrices), min abs (for vectors).
    NegInf,
    /// L1 norm: max column sum (for matrices), sum of abs (for vectors).
    L1,
    /// L2 norm / spectral norm: largest singular value (for matrices),
    /// Euclidean norm (for vectors).
    L2,
    /// General p-norm (for vectors): (sum |x_i|^p)^(1/p).
    P(f64),
}

/// Compute the norm of a vector or matrix.
///
/// For vectors (1D), this computes the vector norm.
/// For matrices (2D), this computes the matrix norm.
///
/// Reduces the full array to a single scalar. For axis-wise reduction
/// (the `axis` / `keepdims` parameters in `numpy.linalg.norm`), use
/// [`norm_axis`].
///
/// # Errors
/// - `FerrayError::InvalidValue` for invalid norm specifications.
pub fn norm<T: LinalgFloat>(a: &Array<T, IxDyn>, ord: NormOrder) -> FerrayResult<T> {
    let shape = a.shape();
    match shape.len() {
        1 => vector_norm(a, ord),
        2 => matrix_norm(a, ord),
        _ => {
            // For ND arrays, flatten and compute vector norm
            vector_norm(a, ord)
        }
    }
}

/// Compute the vector norm along a single axis.
///
/// Analogous to `numpy.linalg.norm(a, ord, axis=k, keepdims=...)` for the
/// 1-D axis case. For each 1-D lane along `axis`, applies the vector
/// norm `ord` and returns an array with shape matching the input except
/// that `axis` is removed (or set to 1 when `keepdims=true`).
///
/// Matrix-style norm axes (a `(int, int)` tuple in NumPy) are not yet
/// supported; pass the matrix to [`matrix_norm`] via `norm` instead.
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
/// - `FerrayError::InvalidValue` for norm orders that are not defined
///   for vectors (currently `NormOrder::Nuc`).
pub fn norm_axis<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    ord: NormOrder,
    axis: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }

    // Output shape: drop the reduced axis (or keep it as size 1).
    let mut out_shape = shape.clone();
    if keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    let axis_len = shape[axis];

    // Row-major strides for the input so we can walk each lane
    // independently.
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Enumerate all "outer" positions (the multi-index excluding
    // `axis`) and pull the corresponding lane of length `axis_len`.
    let outer_total: usize = out_shape.iter().product::<usize>().max(1);
    let data: Vec<T> = a.iter().copied().collect();
    let mut out: Vec<T> = Vec::with_capacity(outer_total);

    // Each outer index corresponds to a flat base offset in `data`;
    // we iterate through `outer_total` output positions and for each
    // one reconstruct the flat base offset into `data`.
    let mut outer_shape = shape.clone();
    outer_shape.remove(axis);
    let mut outer_strides = vec![1usize; outer_shape.len()];
    for i in (0..outer_shape.len().saturating_sub(1)).rev() {
        outer_strides[i] = outer_strides[i + 1] * outer_shape[i + 1];
    }

    for outer_flat in 0..outer_total {
        // Decompose outer_flat into per-axis multi-index (outer shape).
        let mut base = 0usize;
        let mut rem = outer_flat;
        let mut dim = 0usize;
        for d in 0..ndim {
            if d == axis {
                continue;
            }
            let idx = if outer_strides.is_empty() {
                0
            } else {
                rem / outer_strides[dim]
            };
            if !outer_strides.is_empty() {
                rem %= outer_strides[dim];
            }
            base += idx * strides[d];
            dim += 1;
        }
        // Extract the lane along `axis`.
        let mut lane = Vec::with_capacity(axis_len);
        let step = strides[axis];
        for k in 0..axis_len {
            lane.push(data[base + k * step]);
        }
        out.push(vector_norm_from_slice(&lane, ord)?);
    }

    Array::from_vec(IxDyn::new(&out_shape), out)
}

/// Helper that runs the same vector-norm math as `vector_norm` but on a
/// plain slice, avoiding the `&Array<T, IxDyn>` allocation that would
/// otherwise be required per-lane in [`norm_axis`].
fn vector_norm_from_slice<T: LinalgFloat>(
    data: &[T],
    ord: NormOrder,
) -> FerrayResult<T> {
    use num_traits::Float;
    let zero = T::from_f64_const(0.0);
    match ord {
        NormOrder::L2 | NormOrder::Fro => {
            let sum: T = data.iter().map(|&x| x * x).fold(zero, |a, b| a + b);
            Ok(sum.sqrt())
        }
        NormOrder::L1 => {
            let sum: T = data.iter().map(|x| x.abs()).fold(zero, |a, b| a + b);
            Ok(sum)
        }
        NormOrder::Inf => {
            let max = data
                .iter()
                .map(|x| x.abs())
                .fold(zero, |a, b| if a > b { a } else { b });
            Ok(max)
        }
        NormOrder::NegInf => {
            let min = data
                .iter()
                .map(|x| x.abs())
                .fold(<T as Float>::infinity(), |a, b| if a < b { a } else { b });
            Ok(min)
        }
        NormOrder::Nuc => Err(FerrayError::invalid_value(
            "nuclear norm is not defined for 1-D axis reductions",
        )),
        NormOrder::P(p) => {
            if p == 0.0 {
                let count = data.iter().filter(|&&x| x != zero).count();
                Ok(T::from_usize(count))
            } else if p == f64::INFINITY {
                let max = data
                    .iter()
                    .map(|x| x.abs())
                    .fold(zero, |a, b| if a > b { a } else { b });
                Ok(max)
            } else if p == f64::NEG_INFINITY {
                let min = data
                    .iter()
                    .map(|x| x.abs())
                    .fold(<T as Float>::infinity(), |a, b| if a < b { a } else { b });
                Ok(min)
            } else {
                let p_t = T::from_f64_const(p);
                let sum: T = data
                    .iter()
                    .map(|x| x.abs().powf(p_t))
                    .fold(zero, |a, b| a + b);
                Ok(sum.powf(T::from_f64_const(1.0) / p_t))
            }
        }
    }
}

fn vector_norm<T: LinalgFloat>(a: &Array<T, IxDyn>, ord: NormOrder) -> FerrayResult<T> {
    let data: Vec<T> = a.iter().copied().collect();
    match ord {
        NormOrder::L2 | NormOrder::Fro => {
            let sum: T = data
                .iter()
                .map(|&x| x * x)
                .fold(T::from_f64_const(0.0), |a, b| a + b);
            Ok(sum.sqrt())
        }
        NormOrder::L1 => {
            let sum: T = data
                .iter()
                .map(|x| x.abs())
                .fold(T::from_f64_const(0.0), |a, b| a + b);
            Ok(sum)
        }
        NormOrder::Inf => {
            let max = data
                .iter()
                .map(|x| x.abs())
                .fold(T::from_f64_const(0.0), |a, b| if a > b { a } else { b });
            Ok(max)
        }
        NormOrder::NegInf => {
            let min = data
                .iter()
                .map(|x| x.abs())
                .fold(<T as num_traits::Float>::infinity(), |a, b| {
                    if a < b { a } else { b }
                });
            Ok(min)
        }
        NormOrder::Nuc => {
            // NumPy raises ValueError when the nuclear norm is requested
            // for a vector; match that behaviour rather than silently
            // returning the L1 norm (issue #197).
            Err(FerrayError::invalid_value(
                "nuclear norm is not defined for 1-D arrays; use L1/L2/Inf/P for vectors",
            ))
        }
        NormOrder::P(p) => {
            if p == 0.0 {
                // Number of nonzero elements
                let zero = T::from_f64_const(0.0);
                let count = data.iter().filter(|&&x| x != zero).count();
                Ok(T::from_usize(count))
            } else if p == f64::INFINITY {
                let max = data
                    .iter()
                    .map(|x| x.abs())
                    .fold(T::from_f64_const(0.0), |a, b| if a > b { a } else { b });
                Ok(max)
            } else if p == f64::NEG_INFINITY {
                let min = data
                    .iter()
                    .map(|x| x.abs())
                    .fold(<T as num_traits::Float>::infinity(), |a, b| {
                        if a < b { a } else { b }
                    });
                Ok(min)
            } else {
                let p_t = T::from_f64_const(p);
                let sum: T = data
                    .iter()
                    .map(|x| x.abs().powf(p_t))
                    .fold(T::from_f64_const(0.0), |a, b| a + b);
                Ok(sum.powf(T::from_f64_const(1.0) / p_t))
            }
        }
    }
}

fn matrix_norm<T: LinalgFloat>(a: &Array<T, IxDyn>, ord: NormOrder) -> FerrayResult<T> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    let data: Vec<T> = a.iter().copied().collect();

    match ord {
        NormOrder::Fro => {
            let sum: T = data
                .iter()
                .map(|&x| x * x)
                .fold(T::from_f64_const(0.0), |a, b| a + b);
            Ok(sum.sqrt())
        }
        NormOrder::Nuc => {
            // Sum of singular values
            let a2 = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), data)?;
            let (_u, s, _vt) = crate::decomp::svd(&a2, false)?;
            let sum: T = s.iter().copied().fold(T::from_f64_const(0.0), |a, b| a + b);
            Ok(sum)
        }
        NormOrder::L2 => {
            // Largest singular value
            let a2 = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), data)?;
            let (_u, s, _vt) = crate::decomp::svd(&a2, false)?;
            let svals = s.as_slice().unwrap();
            Ok(if svals.is_empty() {
                T::from_f64_const(0.0)
            } else {
                svals[0]
            })
        }
        NormOrder::Inf => {
            // Max absolute row sum
            let mut max_sum = T::from_f64_const(0.0);
            for i in 0..m {
                let mut row_sum = T::from_f64_const(0.0);
                for j in 0..n {
                    row_sum += data[i * n + j].abs();
                }
                if row_sum > max_sum {
                    max_sum = row_sum;
                }
            }
            Ok(max_sum)
        }
        NormOrder::NegInf => {
            // Min absolute row sum
            let mut min_sum = <T as num_traits::Float>::infinity();
            for i in 0..m {
                let mut row_sum = T::from_f64_const(0.0);
                for j in 0..n {
                    row_sum += data[i * n + j].abs();
                }
                if row_sum < min_sum {
                    min_sum = row_sum;
                }
            }
            Ok(min_sum)
        }
        NormOrder::L1 => {
            // Max absolute column sum
            let mut max_sum = T::from_f64_const(0.0);
            for j in 0..n {
                let mut col_sum = T::from_f64_const(0.0);
                for i in 0..m {
                    col_sum += data[i * n + j].abs();
                }
                if col_sum > max_sum {
                    max_sum = col_sum;
                }
            }
            Ok(max_sum)
        }
        NormOrder::P(p) => {
            // Only specific p-norms are defined for matrices.
            // NumPy raises ValueError for unsupported orders.
            Err(FerrayError::invalid_value(format!(
                "invalid norm order P({p}) for matrices; \
                 use Fro, Nuc, L1, L2, Inf, or NegInf"
            )))
        }
    }
}

/// Compute the condition number of a matrix.
///
/// For L2 and Frobenius norms, non-square matrices are accepted (uses SVD).
/// For other norms, a square matrix is required (uses `inv`).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if a square matrix is required but not provided.
pub fn cond<T: LinalgFloat>(a: &Array<T, Ix2>, p: NormOrder) -> FerrayResult<T> {
    let shape = a.shape();

    match p {
        NormOrder::L2 | NormOrder::Fro => {
            // For square 2x2, use the closed-form singular value formula
            if shape[0] == 2 && shape[1] == 2 {
                let s = a.as_slice().unwrap();
                let (a11, a12, a21, a22) = (s[0], s[1], s[2], s[3]);
                let det_a = a11 * a22 - a12 * a21;
                let eps_threshold = T::machine_epsilon() * T::from_f64_const(1e-100);
                if det_a.abs() < eps_threshold {
                    return Ok(<T as num_traits::Float>::infinity());
                }
                let sum_sq = a11 * a11 + a12 * a12 + a21 * a21 + a22 * a22;
                let four_det_sq = T::from_f64_const(4.0) * det_a * det_a;
                let disc_inner = sum_sq * sum_sq - four_det_sq;
                let disc = if disc_inner < T::from_f64_const(0.0) {
                    T::from_f64_const(0.0)
                } else {
                    disc_inner
                }
                .sqrt();
                let s_max = ((sum_sq + disc) / T::from_f64_const(2.0)).sqrt();
                let s_min = det_a.abs() / s_max;
                return Ok(s_max / s_min);
            }
            // General case: cond = largest_sv / smallest_sv
            let (_u, s, _vt) = crate::decomp::svd(a, false)?;
            let svals = s.as_slice().unwrap();
            if svals.is_empty() {
                return Ok(T::from_f64_const(0.0));
            }
            let max_s = svals[0];
            let min_s = svals[svals.len() - 1];
            let zero = T::from_f64_const(0.0);
            if min_s == zero {
                Ok(<T as num_traits::Float>::infinity())
            } else {
                Ok(max_s / min_s)
            }
        }
        _ => {
            if shape[0] != shape[1] {
                return Err(FerrayError::shape_mismatch(format!(
                    "cond with {:?} norm requires a square matrix, got {}x{}",
                    p, shape[0], shape[1]
                )));
            }
            let a_dyn =
                Array::<T, IxDyn>::from_vec(IxDyn::new(shape), a.iter().copied().collect())?;
            let norm_a = norm(&a_dyn, p)?;
            let inv_a = crate::solve::inv(a)?;
            let inv_dyn = Array::<T, IxDyn>::from_vec(
                IxDyn::new(inv_a.shape()),
                inv_a.iter().copied().collect(),
            )?;
            let norm_inv = norm(&inv_dyn, p)?;
            Ok(norm_a * norm_inv)
        }
    }
}

/// Compute the determinant of a square matrix.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
pub fn det<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<T> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "det requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let mat = faer_bridge::array2_to_faer(a);
    Ok(mat.as_ref().determinant())
}

/// Batched determinant for 3D+ arrays.
///
/// Returns a 1D array of determinants, one per batch.
pub fn det_batched<T: LinalgFloat>(a: &Array<T, IxDyn>) -> FerrayResult<Array<T, Ix1>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let d = det(&a2)?;
        return Array::from_vec(Ix1::new([1]), vec![d]);
    }

    batch::apply_batched_scalar(a, |m, n, data| {
        if m != n {
            return Err(FerrayError::shape_mismatch(format!(
                "det requires square matrices, got {}x{}",
                m, n
            )));
        }
        let mat = batch::slice_to_faer(m, n, data);
        Ok(mat.as_ref().determinant())
    })
}

/// Compute the sign and natural logarithm of the determinant.
///
/// Returns `(sign, log_abs_det)` where `det = sign * exp(log_abs_det)`.
/// sign is -1.0, 0.0, or 1.0.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
pub fn slogdet<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<(T, T)> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "slogdet requires square matrices, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let zero = T::from_f64_const(0.0);
    let one = T::from_f64_const(1.0);
    let neg_one = T::from_f64_const(-1.0);

    if n == 0 {
        return Ok((one, zero));
    }

    // Use LU decomposition to compute log|det| without overflow/underflow.
    // det(A) = det(P)^-1 * det(L) * det(U) = (-1)^swaps * product(U_ii)
    let mat = faer_bridge::array2_to_faer(a);
    let lu = mat.as_ref().partial_piv_lu();
    let u_mat = lu.U().to_owned();

    // Accumulate log|det| = sum(log|U_ii|) and sign = product(sign(U_ii))
    let mut log_abs_det = zero;
    let mut sign = one;
    for i in 0..n {
        let diag = u_mat[(i, i)];
        if diag == zero {
            return Ok((zero, <T as num_traits::Float>::neg_infinity()));
        }
        log_abs_det += diag.abs().ln();
        if diag < zero {
            sign *= neg_one;
        }
    }

    // Account for permutation sign by counting transpositions via cycle decomposition.
    // Each cycle of length k contributes (k-1) transpositions.
    let perm_fwd = lu.P().arrays().0;
    let mut visited = vec![false; n];
    let mut num_swaps = 0usize;
    for i in 0..n {
        if !visited[i] {
            visited[i] = true;
            let mut j = perm_fwd[i];
            while j != i {
                visited[j] = true;
                num_swaps += 1;
                j = perm_fwd[j];
            }
        }
    }
    if num_swaps % 2 == 1 {
        sign *= neg_one;
    }

    Ok((sign, log_abs_det))
}

/// Batched sign-and-log-determinant for arrays of shape `(..., N, N)`.
///
/// Returns `(signs, log_abs_dets)` as two 1-D arrays flattened over the
/// leading batch dimensions (matching `det_batched`'s convention).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrices are not square.
pub fn slogdet_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
) -> FerrayResult<(Array<T, Ix1>, Array<T, Ix1>)> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 = Array::<T, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            a.iter().copied().collect(),
        )?;
        let (sign, logdet) = slogdet(&a2)?;
        return Ok((
            Array::from_vec(Ix1::new([1]), vec![sign])?,
            Array::from_vec(Ix1::new([1]), vec![logdet])?,
        ));
    }

    let num_batches = batch::batch_count(shape, 2)?;
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    if m != n {
        return Err(FerrayError::shape_mismatch(format!(
            "slogdet requires square matrices, got {}x{}",
            m, n
        )));
    }

    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = m * n;

    use rayon::prelude::*;
    let pairs: Vec<FerrayResult<(T, T)>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let slice = &data[b * mat_size..(b + 1) * mat_size];
            let mat = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), slice.to_vec())?;
            slogdet(&mat)
        })
        .collect();

    let mut signs = Vec::with_capacity(num_batches);
    let mut logdets = Vec::with_capacity(num_batches);
    for p in pairs {
        let (s, l) = p?;
        signs.push(s);
        logdets.push(l);
    }
    Ok((
        Array::from_vec(Ix1::new([num_batches]), signs)?,
        Array::from_vec(Ix1::new([num_batches]), logdets)?,
    ))
}

/// Compute the rank of a matrix using SVD.
///
/// Elements of the singular value array that are below `tol` are considered zero.
/// If `tol` is `None`, a default tolerance is used.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD fails.
pub fn matrix_rank<T: LinalgFloat>(a: &Array<T, Ix2>, tol: Option<T>) -> FerrayResult<usize> {
    let (_u, s, _vt) = crate::decomp::svd(a, false)?;
    let svals = s.as_slice().unwrap();

    let threshold = tol.unwrap_or_else(|| {
        let m = a.shape()[0];
        let n = a.shape()[1];
        let max_dim = T::from_usize(m.max(n));
        let max_s = if svals.is_empty() {
            T::from_f64_const(0.0)
        } else {
            svals[0]
        };
        max_dim * max_s * T::machine_epsilon()
    });

    Ok(svals.iter().filter(|&&s_val| s_val > threshold).count())
}

/// Batched matrix rank for arrays of shape `(..., M, N)`.
///
/// Returns a flat array of ranks (length = number of batches), matching
/// NumPy's `numpy.linalg.matrix_rank` which returns `int64` per batch.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD fails for any batch element.
pub fn matrix_rank_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    tol: Option<T>,
) -> FerrayResult<Array<i64, Ix1>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 = Array::<T, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            a.iter().copied().collect(),
        )?;
        let r = matrix_rank(&a2, tol)? as i64;
        return Array::from_vec(Ix1::new([1]), vec![r]);
    }

    let num_batches = batch::batch_count(shape, 2)?;
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = m * n;

    use rayon::prelude::*;
    let ranks: Vec<FerrayResult<i64>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let slice = &data[b * mat_size..(b + 1) * mat_size];
            let mat = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), slice.to_vec())?;
            matrix_rank(&mat, tol).map(|r| r as i64)
        })
        .collect();

    let ranks: Vec<i64> = ranks.into_iter().collect::<FerrayResult<Vec<i64>>>()?;
    Array::from_vec(Ix1::new([num_batches]), ranks)
}

/// Compute the trace of a matrix (sum of diagonal elements).
///
/// For non-square matrices, sums the diagonal up to min(m, n).
///
/// # Errors
/// Returns an error only if the array is malformed (never for valid input).
pub fn trace<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<T> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    let data: Vec<T> = a.iter().copied().collect();
    let min_dim = m.min(n);
    let mut sum = T::from_f64_const(0.0);
    for i in 0..min_dim {
        sum += data[i * n + i];
    }
    Ok(sum)
}

/// Extract the diagonal of a matrix as a 1-D array.
///
/// `offset` selects which diagonal is returned:
/// - `offset = 0` → main diagonal (default)
/// - `offset > 0` → super-diagonal (above the main, shifted right by `offset`)
/// - `offset < 0` → sub-diagonal (below the main, shifted down by `|offset|`)
///
/// Equivalent to `numpy.linalg.diagonal(x, offset=offset)` / `np.diag` on a
/// 2-D input. Unlike [`trace`], `diagonal` is a pure structural op so it
/// works for any [`Element`] type, not just `LinalgFloat`.
///
/// An `offset` that falls entirely outside the matrix produces an empty
/// result rather than an error — this matches NumPy (`np.diagonal(a, 999)`
/// yields an empty array).
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `offset` would overflow `isize`
/// during normalization (effectively never for real-world matrices).
pub fn diagonal<T>(a: &Array<T, Ix2>, offset: isize) -> FerrayResult<Array<T, Ix1>>
where
    T: Element,
{
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);

    // Work out the (row, col) starting position and the diagonal length.
    // offset >= 0 → start at (0, offset)        length = min(m, n - offset)
    // offset <  0 → start at (|offset|, 0)      length = min(m - |offset|, n)
    let (start_i, start_j, length) = if offset >= 0 {
        let off = offset as usize;
        if off >= n {
            (0usize, 0usize, 0usize)
        } else {
            (0, off, m.min(n - off))
        }
    } else {
        // offset < 0 — guard against isize::MIN overflow before negating.
        let off_neg = offset.checked_neg().ok_or_else(|| {
            FerrayError::invalid_value("diagonal: offset magnitude overflows isize")
        })?;
        let off = off_neg as usize;
        if off >= m {
            (0, 0, 0)
        } else {
            (off, 0, (m - off).min(n))
        }
    };

    // Materialize once — matches the pattern used by `trace` above.
    let data: Vec<T> = a.iter().cloned().collect();
    let mut result = Vec::with_capacity(length);
    for k in 0..length {
        let i = start_i + k;
        let j = start_j + k;
        result.push(data[i * n + j].clone());
    }
    Array::from_vec(Ix1::new([length]), result)
}

/// Transpose a 2-D matrix by swapping its axes, returning a zero-copy view.
///
/// This is the `numpy.linalg.matrix_transpose(x)` entry point from the
/// Array API standard. For a 2-D matrix it is equivalent to `a.t()` /
/// `x.T` — no data is copied, only the stride/shape metadata. Batched
/// matrix transpose (swap the last two axes of an N-D array) is covered
/// by the general batched matmul / batched ops tracked in #412 and will
/// land alongside that work.
pub fn matrix_transpose<T: Element>(a: &Array<T, Ix2>) -> ArrayView<'_, T, Ix2> {
    a.t()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_identity() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let d = det(&a).unwrap();
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn det_2x2() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let d = det(&a).unwrap();
        assert!((d - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn det_f32() {
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let d = det(&a).unwrap();
        assert!((d - (-2.0f32)).abs() < 1e-5);
    }

    #[test]
    fn det_non_square_error() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(det(&a).is_err());
    }

    #[test]
    fn slogdet_positive() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let (sign, logdet) = slogdet(&a).unwrap();
        assert!((sign - 1.0).abs() < 1e-10);
        assert!((logdet - (6.0f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn slogdet_negative() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (sign, logdet) = slogdet(&a).unwrap();
        assert!((sign - (-1.0)).abs() < 1e-10);
        assert!((logdet - (2.0f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn trace_3x3() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let t = trace(&a).unwrap();
        assert!((t - 15.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_rank_full() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(matrix_rank(&a, None).unwrap(), 3);
    }

    #[test]
    fn matrix_rank_deficient() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
        )
        .unwrap();
        assert_eq!(matrix_rank(&a, None).unwrap(), 1);
    }

    #[test]
    fn norm_vector_l2() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![3.0, 4.0, 0.0]).unwrap();
        let n = norm(&a, NormOrder::L2).unwrap();
        assert!((n - 5.0).abs() < 1e-10);
    }

    #[test]
    fn norm_matrix_fro() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let n = norm(&a, NormOrder::Fro).unwrap();
        assert!((n - 30.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn det_batched_test() {
        let data: Vec<f64> = vec![
            1.0, 0.0, 0.0, 1.0, // identity
            1.0, 2.0, 3.0, 4.0, // det = -2
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let dets = det_batched(&a).unwrap();
        let ds = dets.as_slice().unwrap();
        assert!((ds[0] - 1.0).abs() < 1e-10);
        assert!((ds[1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn cond_identity() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let c = cond(&a, NormOrder::L2).unwrap();
        assert!((c - 1.0).abs() < 1e-10);
    }

    // ---- Batched variants (#412) ----

    #[test]
    fn slogdet_batched_test() {
        // Batch 0: identity -> sign=1, log|det|=0
        // Batch 1: [[2,0],[0,3]] -> det=6, sign=1, log|det|=ln(6)
        // Batch 2: [[1,2],[3,4]] -> det=-2, sign=-1, log|det|=ln(2)
        let data: Vec<f64> = vec![
            1.0, 0.0, 0.0, 1.0, //
            2.0, 0.0, 0.0, 3.0, //
            1.0, 2.0, 3.0, 4.0,
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 2, 2]), data).unwrap();
        let (signs, logdets) = slogdet_batched(&a).unwrap();
        let s = signs.as_slice().unwrap();
        let l = logdets.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!(l[0].abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);
        assert!((l[1] - 6.0_f64.ln()).abs() < 1e-10);
        assert!((s[2] - (-1.0)).abs() < 1e-10);
        assert!((l[2] - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn matrix_rank_batched_test() {
        // Batch 0: full rank 2
        // Batch 1: rank 1 (two identical rows)
        let data: Vec<f64> = vec![
            1.0, 0.0, 0.0, 1.0, //
            1.0, 2.0, 2.0, 4.0, //
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let ranks = matrix_rank_batched(&a, None).unwrap();
        let r = ranks.as_slice().unwrap();
        assert_eq!(r[0], 2);
        assert_eq!(r[1], 1);
    }

    // ---- diagonal / matrix_transpose (#416) ----

    #[test]
    fn diagonal_main_square() {
        // [[1,2,3],[4,5,6],[7,8,9]] → main diagonal [1,5,9]
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.as_slice().unwrap(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diagonal_super_diagonal() {
        // offset=1 on 3x3 → elements (0,1), (1,2) → [2, 6]
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let d = diagonal(&a, 1).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.as_slice().unwrap(), &[2.0, 6.0]);
    }

    #[test]
    fn diagonal_sub_diagonal() {
        // offset=-1 on 3x3 → elements (1,0), (2,1) → [4, 8]
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let d = diagonal(&a, -1).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.as_slice().unwrap(), &[4.0, 8.0]);
    }

    #[test]
    fn diagonal_wide_matrix() {
        // 2x3 main diagonal is (0,0),(1,1) → length 2
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.as_slice().unwrap(), &[1.0, 5.0]);
    }

    #[test]
    fn diagonal_tall_matrix() {
        // 3x2 main diagonal is (0,0),(1,1) → length 2
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.as_slice().unwrap(), &[1.0, 4.0]);
    }

    #[test]
    fn diagonal_offset_out_of_range_is_empty() {
        // Matches NumPy: offsets past the matrix yield an empty result, not
        // an error.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 2]),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let d = diagonal(&a, 5).unwrap();
        assert_eq!(d.shape(), &[0]);
        let d2 = diagonal(&a, -5).unwrap();
        assert_eq!(d2.shape(), &[0]);
    }

    #[test]
    fn diagonal_on_non_float_element_type() {
        // Non-LinalgFloat element type: i64 works because diagonal only
        // requires Element, not LinalgFloat.
        let a = Array::<i64, Ix2>::from_vec(Ix2::new([3, 3]), vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.as_slice().unwrap(), &[1, 5, 9]);
    }

    #[test]
    fn diagonal_agrees_with_trace_via_sum() {
        // sum(diagonal(a)) == trace(a) for square inputs.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([4, 4]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let diag_sum: f64 = diagonal(&a, 0).unwrap().iter().copied().sum();
        let tr = trace(&a).unwrap();
        assert!((diag_sum - tr).abs() < 1e-10);
    }

    #[test]
    fn matrix_transpose_swaps_axes() {
        // [[1,2,3],[4,5,6]] transposes to a view with shape [3,2]:
        // [[1,4],[2,5],[3,6]]
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let t = matrix_transpose(&a);
        assert_eq!(t.shape(), &[3, 2]);
        // Walk the transposed view in logical order.
        let seen: Vec<f64> = t.iter().copied().collect();
        assert_eq!(seen, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn matrix_transpose_identity_roundtrip() {
        // Transposing twice yields a view over the original.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let t = matrix_transpose(&a);
        // Collect logical values by iterating the double-transposed view via
        // ndarray's reversed_axes twice: done through the owned array.
        let tt: Vec<f64> = t.t().iter().copied().collect();
        assert_eq!(tt, a.iter().copied().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_transpose_non_copy_element_type() {
        // Works for any Element — no LinalgFloat bound.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1, 2, 3, 4]).unwrap();
        let t = matrix_transpose(&a);
        let seen: Vec<i32> = t.iter().copied().collect();
        assert_eq!(seen, vec![1, 3, 2, 4]);
    }
}
