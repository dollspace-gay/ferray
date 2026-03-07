// ferray-linalg: tensordot (REQ-5)
//
// Contraction over specified axis pairs.

use ferray_core::array::owned::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::error::{FerrayError, FerrayResult};

/// Specifies axes for tensordot contraction.
#[derive(Debug, Clone)]
pub enum TensordotAxes {
    /// Contract over N axes: the last N of `a` with the first N of `b`.
    Scalar(usize),
    /// Explicit pairs: `(axes_a, axes_b)` each of the same length.
    Pairs(Vec<usize>, Vec<usize>),
}

/// Compute the tensor dot product of two arrays over specified axes.
///
/// Analogous to `numpy.tensordot`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if axes are incompatible.
pub fn tensordot(
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
    axes: TensordotAxes,
) -> FerrayResult<Array<f64, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let (axes_a, axes_b) = match axes {
        TensordotAxes::Scalar(n) => {
            if n > a_shape.len() || n > b_shape.len() {
                return Err(FerrayError::shape_mismatch(format!(
                    "tensordot: cannot contract {} axes from shapes {:?} and {:?}",
                    n, a_shape, b_shape
                )));
            }
            let axes_a: Vec<usize> = (a_shape.len() - n..a_shape.len()).collect();
            let axes_b: Vec<usize> = (0..n).collect();
            (axes_a, axes_b)
        }
        TensordotAxes::Pairs(axes_a, axes_b) => {
            if axes_a.len() != axes_b.len() {
                return Err(FerrayError::shape_mismatch(
                    "tensordot: axes_a and axes_b must have the same length",
                ));
            }
            (axes_a, axes_b)
        }
    };

    // Verify contracted dimensions match
    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        if ax_a >= a_shape.len() || ax_b >= b_shape.len() {
            return Err(FerrayError::shape_mismatch(format!(
                "tensordot: axis out of bounds (a axis {} for {}D, b axis {} for {}D)",
                ax_a,
                a_shape.len(),
                ax_b,
                b_shape.len()
            )));
        }
        if a_shape[ax_a] != b_shape[ax_b] {
            return Err(FerrayError::shape_mismatch(format!(
                "tensordot: contracted dimensions must match (a[{}]={} != b[{}]={})",
                ax_a, a_shape[ax_a], ax_b, b_shape[ax_b]
            )));
        }
    }

    // Compute free axes (non-contracted)
    let a_free: Vec<usize> = (0..a_shape.len()).filter(|i| !axes_a.contains(i)).collect();
    let b_free: Vec<usize> = (0..b_shape.len()).filter(|i| !axes_b.contains(i)).collect();

    // Output shape
    let mut out_shape: Vec<usize> = Vec::with_capacity(a_free.len() + b_free.len());
    for &ax in &a_free {
        out_shape.push(a_shape[ax]);
    }
    for &ax in &b_free {
        out_shape.push(b_shape[ax]);
    }

    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Contracted dimension sizes
    let contract_size: usize = axes_a.iter().map(|&ax| a_shape[ax]).product();

    // Reshape a into (free_a_size, contract_size) and b into (contract_size, free_b_size)
    let free_a_size: usize = a_free
        .iter()
        .map(|&ax| a_shape[ax])
        .product::<usize>()
        .max(1);
    let free_b_size: usize = b_free
        .iter()
        .map(|&ax| b_shape[ax])
        .product::<usize>()
        .max(1);

    // Build permutation for a: free axes first, then contracted
    let mut a_perm: Vec<usize> = Vec::with_capacity(a_shape.len());
    a_perm.extend_from_slice(&a_free);
    a_perm.extend_from_slice(&axes_a);

    // Build permutation for b: contracted first, then free
    let mut b_perm: Vec<usize> = Vec::with_capacity(b_shape.len());
    b_perm.extend_from_slice(&axes_b);
    b_perm.extend_from_slice(&b_free);

    // Collect data in permuted order
    let a_data: Vec<f64> = a.iter().copied().collect();
    let b_data: Vec<f64> = b.iter().copied().collect();

    let a_perm_data = permute_data(&a_data, a_shape, &a_perm);
    let b_perm_data = permute_data(&b_data, b_shape, &b_perm);

    // Matrix multiply: (free_a_size x contract_size) @ (contract_size x free_b_size)
    let mut result = vec![0.0; out_size];
    for i in 0..free_a_size {
        for j in 0..free_b_size {
            let mut sum = 0.0;
            for k in 0..contract_size {
                sum += a_perm_data[i * contract_size + k] * b_perm_data[k * free_b_size + j];
            }
            result[i * free_b_size + j] = sum;
        }
    }

    if out_shape.is_empty() {
        out_shape.push(1);
        // Scalar result stored in 1-element array
    }
    Array::from_vec(IxDyn::new(&out_shape), result)
}

/// Permute array data according to a given axis permutation.
fn permute_data(data: &[f64], shape: &[usize], perm: &[usize]) -> Vec<f64> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();
    if total == 0 {
        return vec![];
    }

    // Compute strides for original array (C-order)
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute shape and strides in permuted order
    let perm_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let perm_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();

    let mut result = vec![0.0; total];
    let mut idx = vec![0usize; ndim];

    for out_flat in 0..total {
        // Convert out_flat to multi-index in permuted shape
        let mut rem = out_flat;
        for d in (0..ndim).rev() {
            if perm_shape[d] > 0 {
                idx[d] = rem % perm_shape[d];
                rem /= perm_shape[d];
            }
        }
        // Convert to flat index in original array
        let mut orig_flat = 0;
        for d in 0..ndim {
            orig_flat += idx[d] * perm_strides[d];
        }
        result[out_flat] = data[orig_flat];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensordot_scalar_2() {
        // Equivalent to matmul for 2D arrays with axes=1
        // a: (2,3), b: (3,2) -> (2,2)
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c = tensordot(&a, &b, TensordotAxes::Scalar(1)).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let cd: Vec<f64> = c.iter().copied().collect();
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        assert!((cd[0] - 58.0).abs() < 1e-10);
        assert!((cd[1] - 64.0).abs() < 1e-10);
        assert!((cd[2] - 139.0).abs() < 1e-10);
        assert!((cd[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn tensordot_explicit_axes() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c = tensordot(&a, &b, TensordotAxes::Pairs(vec![1], vec![0])).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }
}
