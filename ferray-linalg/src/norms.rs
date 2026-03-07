// ferray-linalg: Norms and measures (REQ-19 through REQ-22)
//
// norm, cond, det, slogdet, matrix_rank, trace

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch;
use crate::faer_bridge;

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
pub fn norm(a: &Array<f64, IxDyn>, ord: NormOrder) -> FerrayResult<f64> {
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

fn vector_norm(a: &Array<f64, IxDyn>, ord: NormOrder) -> FerrayResult<f64> {
    let data: Vec<f64> = a.iter().copied().collect();
    match ord {
        NormOrder::L2 | NormOrder::Fro => {
            let sum: f64 = data.iter().map(|x| x * x).sum();
            Ok(sum.sqrt())
        }
        NormOrder::L1 => {
            let sum: f64 = data.iter().map(|x| x.abs()).sum();
            Ok(sum)
        }
        NormOrder::Inf => {
            let max = data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            Ok(max)
        }
        NormOrder::NegInf => {
            let min = data.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min);
            Ok(min)
        }
        NormOrder::Nuc => {
            // Nuclear norm is not typically defined for vectors; use L1
            let sum: f64 = data.iter().map(|x| x.abs()).sum();
            Ok(sum)
        }
        NormOrder::P(p) => {
            if p == 0.0 {
                // Number of nonzero elements
                let count = data.iter().filter(|&&x| x != 0.0).count() as f64;
                Ok(count)
            } else if p == f64::INFINITY {
                let max = data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
                Ok(max)
            } else if p == f64::NEG_INFINITY {
                let min = data.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min);
                Ok(min)
            } else {
                let sum: f64 = data.iter().map(|x| x.abs().powf(p)).sum();
                Ok(sum.powf(1.0 / p))
            }
        }
    }
}

fn matrix_norm(a: &Array<f64, IxDyn>, ord: NormOrder) -> FerrayResult<f64> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    let data: Vec<f64> = a.iter().copied().collect();

    match ord {
        NormOrder::Fro => {
            let sum: f64 = data.iter().map(|x| x * x).sum();
            Ok(sum.sqrt())
        }
        NormOrder::Nuc => {
            // Sum of singular values
            let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([m, n]), data)?;
            let (_u, s, _vt) = crate::decomp::svd(&a2, false)?;
            let sum: f64 = s.iter().copied().sum();
            Ok(sum)
        }
        NormOrder::L2 => {
            // Largest singular value
            let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([m, n]), data)?;
            let (_u, s, _vt) = crate::decomp::svd(&a2, false)?;
            let svals = s.as_slice().unwrap();
            Ok(if svals.is_empty() { 0.0 } else { svals[0] })
        }
        NormOrder::Inf => {
            // Max absolute row sum
            let mut max_sum = 0.0f64;
            for i in 0..m {
                let row_sum: f64 = (0..n).map(|j| data[i * n + j].abs()).sum();
                max_sum = max_sum.max(row_sum);
            }
            Ok(max_sum)
        }
        NormOrder::NegInf => {
            // Min absolute row sum
            let mut min_sum = f64::INFINITY;
            for i in 0..m {
                let row_sum: f64 = (0..n).map(|j| data[i * n + j].abs()).sum();
                min_sum = min_sum.min(row_sum);
            }
            Ok(min_sum)
        }
        NormOrder::L1 => {
            // Max absolute column sum
            let mut max_sum = 0.0f64;
            for j in 0..n {
                let col_sum: f64 = (0..m).map(|i| data[i * n + j].abs()).sum();
                max_sum = max_sum.max(col_sum);
            }
            Ok(max_sum)
        }
        NormOrder::P(_) => {
            // General p-norm not standard for matrices; use Frobenius
            let sum: f64 = data.iter().map(|x| x * x).sum();
            Ok(sum.sqrt())
        }
    }
}

/// Compute the condition number of a matrix.
///
/// Uses the ratio of the largest to smallest singular value (for L2 norm).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if not a square matrix.
pub fn cond(a: &Array<f64, Ix2>, p: NormOrder) -> FerrayResult<f64> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "cond requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }

    match p {
        NormOrder::L2 | NormOrder::Fro => {
            // cond = largest_sv / smallest_sv
            let (_u, s, _vt) = crate::decomp::svd(a, false)?;
            let svals = s.as_slice().unwrap();
            if svals.is_empty() {
                return Ok(0.0);
            }
            let max_s = svals[0];
            let min_s = svals[svals.len() - 1];
            if min_s == 0.0 {
                Ok(f64::INFINITY)
            } else {
                Ok(max_s / min_s)
            }
        }
        _ => {
            let a_dyn =
                Array::<f64, IxDyn>::from_vec(IxDyn::new(shape), a.iter().copied().collect())?;
            let norm_a = norm(&a_dyn, p)?;
            let inv_a = crate::solve::inv(a)?;
            let inv_dyn = Array::<f64, IxDyn>::from_vec(
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
pub fn det(a: &Array<f64, Ix2>) -> FerrayResult<f64> {
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
pub fn det_batched(a: &Array<f64, IxDyn>) -> FerrayResult<Array<f64, Ix1>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 = Array::<f64, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            a.iter().copied().collect(),
        )?;
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
pub fn slogdet(a: &Array<f64, Ix2>) -> FerrayResult<(f64, f64)> {
    let d = det(a)?;
    if d == 0.0 {
        Ok((0.0, f64::NEG_INFINITY))
    } else if d > 0.0 {
        Ok((1.0, d.ln()))
    } else {
        Ok((-1.0, (-d).ln()))
    }
}

/// Compute the rank of a matrix using SVD.
///
/// Elements of the singular value array that are below `tol` are considered zero.
/// If `tol` is `None`, a default tolerance is used.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD fails.
pub fn matrix_rank(a: &Array<f64, Ix2>, tol: Option<f64>) -> FerrayResult<usize> {
    let (_u, s, _vt) = crate::decomp::svd(a, false)?;
    let svals = s.as_slice().unwrap();

    let threshold = tol.unwrap_or_else(|| {
        let m = a.shape()[0];
        let n = a.shape()[1];
        let max_dim = m.max(n) as f64;
        let max_s = if svals.is_empty() { 0.0 } else { svals[0] };
        max_dim * max_s * f64::EPSILON
    });

    Ok(svals.iter().filter(|&&s_val| s_val > threshold).count())
}

/// Compute the trace of a matrix (sum of diagonal elements).
///
/// For non-square matrices, sums the diagonal up to min(m, n).
///
/// # Errors
/// Returns an error only if the array is malformed (never for valid input).
pub fn trace(a: &Array<f64, Ix2>) -> FerrayResult<f64> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    let data: Vec<f64> = a.iter().copied().collect();
    let min_dim = m.min(n);
    let mut sum = 0.0;
    for i in 0..min_dim {
        sum += data[i * n + i];
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
}
