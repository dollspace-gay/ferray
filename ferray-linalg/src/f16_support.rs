//! f16 (half-precision) support for ferray-linalg.
//!
//! All f16 linalg operations promote inputs to f64 for numerical stability,
//! perform the operation, then convert results back to f16.

use ferray_core::array::owned::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::error::FerrayResult;
use half::f16;

/// Convert an f16 array to f64 for linalg computation.
fn promote_to_f64(arr: &Array<f16, IxDyn>) -> FerrayResult<Array<f64, IxDyn>> {
    let shape = arr.shape().to_vec();
    let data: Vec<f64> = arr.iter().map(|v| v.to_f64()).collect();
    Array::from_vec(IxDyn::new(&shape), data)
}

/// Convert an f64 result array back to f16.
fn demote_to_f16(arr: &Array<f64, IxDyn>) -> FerrayResult<Array<f16, IxDyn>> {
    let shape = arr.shape().to_vec();
    let data: Vec<f16> = arr.iter().map(|v| f16::from_f64(*v)).collect();
    Array::from_vec(IxDyn::new(&shape), data)
}

/// Matrix multiplication for f16 arrays.
///
/// Promotes inputs to f64 internally for numerical stability,
/// then converts the result back to f16.
///
/// Supports the same shapes as [`crate::matmul`]:
/// - 2D x 2D: standard matrix multiplication
/// - 1D x 2D: vector-matrix
/// - 2D x 1D: matrix-vector
/// - ND x ND: batched matmul
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if inner dimensions don't match.
pub fn matmul_f16(a: &Array<f16, IxDyn>, b: &Array<f16, IxDyn>) -> FerrayResult<Array<f16, IxDyn>> {
    let a_f64 = promote_to_f64(a)?;
    let b_f64 = promote_to_f64(b)?;
    let result = crate::matmul(&a_f64, &b_f64)?;
    demote_to_f16(&result)
}

/// Dot product for f16 arrays.
///
/// Promotes to f64 internally, delegates to [`crate::dot`].
pub fn dot_f16(a: &Array<f16, IxDyn>, b: &Array<f16, IxDyn>) -> FerrayResult<Array<f16, IxDyn>> {
    let a_f64 = promote_to_f64(a)?;
    let b_f64 = promote_to_f64(b)?;
    let result = crate::dot(&a_f64, &b_f64)?;
    demote_to_f16(&result)
}

/// Vector dot product for f16 arrays, returning f16.
///
/// Promotes to f64 internally, delegates to [`crate::vdot`].
pub fn vdot_f16(a: &Array<f16, IxDyn>, b: &Array<f16, IxDyn>) -> FerrayResult<f16> {
    let a_f64 = promote_to_f64(a)?;
    let b_f64 = promote_to_f64(b)?;
    let result = crate::vdot(&a_f64, &b_f64)?;
    Ok(f16::from_f64(result))
}

/// Outer product for f16 arrays.
///
/// Promotes to f64 internally, delegates to [`crate::outer`].
pub fn outer_f16(a: &Array<f16, IxDyn>, b: &Array<f16, IxDyn>) -> FerrayResult<Array<f16, IxDyn>> {
    let a_f64 = promote_to_f64(a)?;
    let b_f64 = promote_to_f64(b)?;
    let result = crate::outer(&a_f64, &b_f64)?;
    demote_to_f16(&result)
}

/// Matrix norm for f16 arrays, returning f16.
///
/// Promotes to f64 internally, delegates to [`crate::norm`].
pub fn norm_f16(a: &Array<f16, IxDyn>, order: crate::NormOrder) -> FerrayResult<f64> {
    let a_f64 = promote_to_f64(a)?;
    crate::norm(&a_f64, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_arr(shape: &[usize], data: &[f64]) -> Array<f16, IxDyn> {
        let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f64(v)).collect();
        Array::from_vec(IxDyn::new(shape), f16_data).unwrap()
    }

    #[test]
    fn test_matmul_f16_2x2() {
        let a = f16_arr(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = f16_arr(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let c = matmul_f16(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let vals: Vec<f16> = c.iter().copied().collect();
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        assert_eq!(vals[0].to_f64(), 19.0);
        assert_eq!(vals[1].to_f64(), 22.0);
        assert_eq!(vals[2].to_f64(), 43.0);
        assert_eq!(vals[3].to_f64(), 50.0);
    }

    #[test]
    fn test_matmul_f16_vec_mat() {
        let a = f16_arr(&[3], &[1.0, 2.0, 3.0]);
        let b = f16_arr(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = matmul_f16(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2]);
        let vals: Vec<f16> = c.iter().copied().collect();
        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        assert_eq!(vals[0].to_f64(), 22.0);
        assert_eq!(vals[1].to_f64(), 28.0);
    }

    #[test]
    fn test_dot_f16() {
        let a = f16_arr(&[3], &[1.0, 2.0, 3.0]);
        let b = f16_arr(&[3], &[4.0, 5.0, 6.0]);
        let c = dot_f16(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        let vals: Vec<f16> = c.iter().copied().collect();
        assert_eq!(vals[0].to_f64(), 32.0);
    }

    #[test]
    fn test_vdot_f16() {
        let a = f16_arr(&[3], &[1.0, 2.0, 3.0]);
        let b = f16_arr(&[3], &[4.0, 5.0, 6.0]);
        let result = vdot_f16(&a, &b).unwrap();
        assert_eq!(result.to_f64(), 32.0);
    }

    #[test]
    fn test_outer_f16() {
        let a = f16_arr(&[2], &[1.0, 2.0]);
        let b = f16_arr(&[3], &[3.0, 4.0, 5.0]);
        let c = outer_f16(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let vals: Vec<f16> = c.iter().copied().collect();
        assert_eq!(vals[0].to_f64(), 3.0);
        assert_eq!(vals[1].to_f64(), 4.0);
        assert_eq!(vals[2].to_f64(), 5.0);
        assert_eq!(vals[3].to_f64(), 6.0);
        assert_eq!(vals[4].to_f64(), 8.0);
        assert_eq!(vals[5].to_f64(), 10.0);
    }

    #[test]
    fn test_norm_f16() {
        let a = f16_arr(&[3], &[3.0, 4.0, 0.0]);
        let n = norm_f16(&a, crate::NormOrder::Fro).unwrap();
        assert!((n - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_f16_shape_mismatch() {
        let a = f16_arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f16_arr(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(matmul_f16(&a, &b).is_err());
    }
}
