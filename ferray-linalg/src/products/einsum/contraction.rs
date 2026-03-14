// ferray-linalg: Generic einsum contraction loop
//
// Handles arbitrary subscript patterns including trace, diagonal, etc.

use ferray_core::array::owned::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::error::{FerrayError, FerrayResult};
use std::collections::HashMap;

use super::parser::{EinsumExpr, Label};
use crate::scalar::LinalgFloat;

/// Execute the generic contraction for an einsum expression.
///
/// This handles all patterns including trace, diagonal extraction,
/// multi-operand contractions, etc.
pub fn generic_contraction<T: LinalgFloat>(
    expr: &EinsumExpr,
    operands: &[&Array<T, IxDyn>],
) -> FerrayResult<Array<T, IxDyn>> {
    // Build a mapping from each label to its size
    let mut label_sizes: HashMap<Label, usize> = HashMap::new();
    for (op_idx, labels) in expr.inputs.iter().enumerate() {
        let shape = operands[op_idx].shape();
        for (dim_idx, &label) in labels.iter().enumerate() {
            let size = shape[dim_idx];
            if let Some(&existing) = label_sizes.get(&label) {
                if existing != size {
                    return Err(FerrayError::shape_mismatch(format!(
                        "einsum: label {:?} has inconsistent sizes {} and {}",
                        label, existing, size
                    )));
                }
            } else {
                label_sizes.insert(label, size);
            }
        }
    }

    // Determine output shape
    let out_shape: Vec<usize> = expr
        .output
        .iter()
        .map(|l| {
            label_sizes.get(l).copied().ok_or_else(|| {
                FerrayError::invalid_value(format!(
                    "einsum: output label {:?} not found in any input",
                    l
                ))
            })
        })
        .collect::<FerrayResult<Vec<_>>>()?;

    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Determine summation labels: labels in inputs but not in output
    let output_set: std::collections::HashSet<Label> = expr.output.iter().copied().collect();
    let mut sum_labels: Vec<Label> = Vec::new();
    for labels in &expr.inputs {
        for &l in labels {
            if !output_set.contains(&l) && !sum_labels.contains(&l) {
                sum_labels.push(l);
            }
        }
    }

    let sum_shape: Vec<usize> = sum_labels.iter().map(|l| label_sizes[l]).collect();
    let _sum_size: usize = if sum_shape.is_empty() {
        1
    } else {
        sum_shape.iter().product()
    };

    // Collect operand data and strides
    let operand_data: Vec<Vec<T>> = operands
        .iter()
        .map(|o| o.iter().copied().collect())
        .collect();

    // Compute strides for each operand
    let operand_strides: Vec<Vec<usize>> = operands
        .iter()
        .map(|o| {
            let shape = o.shape();
            let mut strides = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            strides
        })
        .collect();

    // For each output index and summation index, compute the contribution
    let zero = T::from_f64_const(0.0);
    let one = T::from_f64_const(1.0);
    let mut result = vec![zero; out_size];

    // Build all labels = output_labels ++ sum_labels
    let all_labels: Vec<Label> = expr
        .output
        .iter()
        .chain(sum_labels.iter())
        .copied()
        .collect();
    let all_shape: Vec<usize> = out_shape.iter().chain(sum_shape.iter()).copied().collect();
    let total_iters: usize = if all_shape.is_empty() {
        1
    } else {
        all_shape.iter().product()
    };

    // Precompute label-to-dim mapping for each operand
    let op_label_dims: Vec<Vec<(usize, usize)>> = expr
        .inputs
        .iter()
        .enumerate()
        .map(|(_op_idx, labels)| {
            labels
                .iter()
                .enumerate()
                .map(|(dim_idx, label)| {
                    let all_idx = all_labels.iter().position(|l| l == label).unwrap();
                    (dim_idx, all_idx)
                })
                .collect()
        })
        .collect();

    let n_out_dims = out_shape.len();

    // Iterate over all combinations
    let mut multi_idx = vec![0usize; all_labels.len()];
    for _iter in 0..total_iters {
        // Compute output flat index
        let mut out_flat = 0;
        {
            let mut s = 1;
            for d in (0..n_out_dims).rev() {
                out_flat += multi_idx[d] * s;
                s *= out_shape[d];
            }
        }

        // Compute product of all operands at current index
        let mut product = one;
        for (op_idx, label_dims) in op_label_dims.iter().enumerate() {
            let mut op_flat = 0;
            for &(dim_idx, all_idx) in label_dims {
                op_flat += multi_idx[all_idx] * operand_strides[op_idx][dim_idx];
            }
            product = product * operand_data[op_idx][op_flat];
        }

        if out_size > 0 {
            result[out_flat] = result[out_flat] + product;
        }

        // Increment multi_idx (like an odometer)
        let mut carry = true;
        for d in (0..all_labels.len()).rev() {
            if carry {
                multi_idx[d] += 1;
                if multi_idx[d] >= all_shape[d] {
                    multi_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    if out_shape.is_empty() {
        Array::from_vec(IxDyn::new(&[]), result[..1].to_vec())
            .or_else(|_| Array::from_vec(IxDyn::new(&[1]), result[..1].to_vec()))
    } else {
        Array::from_vec(IxDyn::new(&out_shape), result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::products::einsum::parser::parse_subscripts;

    #[test]
    fn trace_contraction() {
        // einsum("ii->i", a) where a is 3x3
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let expr = parse_subscripts("ii->i", &[&[3, 3]]).unwrap();
        let result = generic_contraction(&expr, &[&a]).unwrap();
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn trace_scalar() {
        // einsum("ii", a) = trace
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let expr = parse_subscripts("ii", &[&[3, 3]]).unwrap();
        let result = generic_contraction(&expr, &[&a]).unwrap();
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data.len(), 1);
        assert!((data[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_via_contraction() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let expr = parse_subscripts("ij,jk->ik", &[&[2, 3], &[3, 2]]).unwrap();
        let result = generic_contraction(&expr, &[&a, &b]).unwrap();
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(result.shape(), &[2, 2]);
        assert!((data[0] - 58.0).abs() < 1e-10);
        assert!((data[1] - 64.0).abs() < 1e-10);
        assert!((data[2] - 139.0).abs() < 1e-10);
        assert!((data[3] - 154.0).abs() < 1e-10);
    }
}
