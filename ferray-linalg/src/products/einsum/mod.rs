// ferray-linalg: Einstein summation (REQ-6, REQ-25, REQ-26)
//
// Public einsum() function that parses subscripts, optimizes, and executes.

/// Generic contraction loop for arbitrary einsum patterns.
pub mod contraction;
/// Optimizer that detects matmul/tensordot shortcuts.
pub mod optimizer;
/// Subscript string parser.
pub mod parser;

use ferray_core::array::owned::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::error::FerrayResult;

use self::contraction::generic_contraction;
use self::optimizer::{EinsumStrategy, optimize};
use self::parser::parse_subscripts;
use crate::scalar::LinalgFloat;

/// Compute Einstein summation notation.
///
/// This is the equivalent of `numpy.einsum`. It supports:
/// - Implicit output mode: `"ij,jk"` (output labels are alphabetically sorted
///   unique labels appearing exactly once)
/// - Explicit output mode: `"ij,jk->ik"`
/// - Trace: `"ii->i"` or `"ii"`
/// - Batch dimensions: `"bij,bjk->bik"`
/// - Ellipsis broadcasting: `"...ij,...jk->...ik"`
///
/// For 2-operand cases, matmul and tensordot shortcuts are detected and used
/// when possible for better performance.
///
/// # Arguments
/// - `subscripts`: The einsum subscript string.
/// - `operands`: Slice of references to input arrays.
///
/// # Errors
/// - `FerrayError::InvalidValue` for malformed subscripts.
/// - `FerrayError::ShapeMismatch` for incompatible operand shapes.
pub fn einsum<T: LinalgFloat>(
    subscripts: &str,
    operands: &[&Array<T, IxDyn>],
) -> FerrayResult<Array<T, IxDyn>> {
    let shapes: Vec<&[usize]> = operands.iter().map(|o| o.shape()).collect();
    let expr = parse_subscripts(subscripts, &shapes)?;

    // Try optimization for 2-operand cases
    if operands.len() == 2 {
        let strategy = optimize(&expr);
        match strategy {
            EinsumStrategy::Matmul => {
                return execute_matmul(operands[0], operands[1], &expr);
            }
            EinsumStrategy::Tensordot { axes_a, axes_b } => {
                return execute_tensordot(operands[0], operands[1], axes_a, axes_b);
            }
            EinsumStrategy::Generic => {}
        }
    }

    generic_contraction(&expr, operands)
}

fn execute_matmul<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    expr: &parser::EinsumExpr,
) -> FerrayResult<Array<T, IxDyn>> {
    // For simple 2D matmul, delegate to our matmul
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let a_data: Vec<T> = a.iter().copied().collect();
        let b_data: Vec<T> = b.iter().copied().collect();
        let (m, k) = (a_shape[0], a_shape[1]);
        let n = b_shape[1];

        let zero = T::from_f64_const(0.0);
        let mut result = vec![zero; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a_data[i * k + p];
                for j in 0..n {
                    result[i * n + j] = result[i * n + j] + a_ip * b_data[p * n + j];
                }
            }
        }
        return Array::from_vec(IxDyn::new(&[m, n]), result);
    }

    // Fall back to generic for batched matmul
    generic_contraction(expr, &[a, b])
}

fn execute_tensordot<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    axes_a: Vec<usize>,
    axes_b: Vec<usize>,
) -> FerrayResult<Array<T, IxDyn>> {
    use crate::products::tensordot::{TensordotAxes, tensordot};
    tensordot(a, b, TensordotAxes::Pairs(axes_a, axes_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn einsum_matmul_explicit() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 58.0).abs() < 1e-10);
        assert!((data[1] - 64.0).abs() < 1e-10);
        assert!((data[2] - 139.0).abs() < 1e-10);
        assert!((data[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_matmul_implicit() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c = einsum("ij,jk", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn einsum_trace_diagonal() {
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let c = einsum("ii->i", &[&a]).unwrap();
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn einsum_trace_scalar() {
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let c = einsum("ii", &[&a]).unwrap();
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_outer_product() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![3.0, 4.0, 5.0]).unwrap();
        let c = einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 3.0).abs() < 1e-10);
        assert!((data[1] - 4.0).abs() < 1e-10);
        assert!((data[5] - 10.0).abs() < 1e-10);
    }
}
