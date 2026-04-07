// ferray-linalg: Norms and measures (REQ-19 through REQ-22)
//
// norm, cond, det, slogdet, matrix_rank, trace

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
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
            // Nuclear norm is not typically defined for vectors; use L1
            let sum: T = data
                .iter()
                .map(|x| x.abs())
                .fold(T::from_f64_const(0.0), |a, b| a + b);
            Ok(sum)
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
                    row_sum = row_sum + data[i * n + j].abs();
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
                    row_sum = row_sum + data[i * n + j].abs();
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
                    col_sum = col_sum + data[i * n + j].abs();
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
        log_abs_det = log_abs_det + diag.abs().ln();
        if diag < zero {
            sign = sign * neg_one;
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
        sign = sign * neg_one;
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
        sum = sum + data[i * n + i];
    }
    Ok(sum)
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
}
