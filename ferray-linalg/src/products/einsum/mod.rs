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
use self::parser::{Label, parse_subscripts};
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

    // For 3+ operand cases, contract pairwise left-to-right (#102).
    // The previous behaviour fell straight through to generic_contraction
    // which iterates `prod(all_dims)` cells across all operands at once —
    // O(n^k) for k operands. Pairwise contraction reduces it to a sum of
    // O(n^3) matmul-shaped costs for typical chain expressions like
    // 'ij,jk,kl->il'. The order matters for cost (full optimal path
    // planning matches numpy's optimize='greedy' / 'optimal'); a simple
    // left-to-right reduction handles most real einsum usage and matches
    // numpy's optimize=False path within a constant factor.
    if operands.len() >= 3 {
        return einsum_pairwise(&expr, operands);
    }

    generic_contraction(&expr, operands)
}

/// Reduce an N-operand einsum (N >= 3) to a sequence of 2-operand
/// contractions, left-to-right.
///
/// At each step the first two operands are contracted into one. The
/// intermediate output's labels are: every label that appears in the
/// remaining operands or the final output. Labels that only appear in
/// the two we're contracting and aren't needed downstream get summed
/// out at the contraction step.
fn einsum_pairwise<T: LinalgFloat>(
    expr: &parser::EinsumExpr,
    operands: &[&Array<T, IxDyn>],
) -> FerrayResult<Array<T, IxDyn>> {
    debug_assert!(operands.len() >= 3, "pairwise path needs >=3 operands");

    // Working copies of the operand list and their input labels. After
    // each contraction step we replace [0..2] with the result.
    let mut current_ops: Vec<Array<T, IxDyn>> =
        operands.iter().map(|o| (*o).clone()).collect();
    let mut current_inputs: Vec<Vec<Label>> = expr.inputs.clone();
    let final_output = expr.output.clone();

    // Build a label-size map from the original operands. We need this
    // for the greedy contraction-order heuristic. (#418)
    let mut label_size: std::collections::HashMap<Label, usize> =
        std::collections::HashMap::new();
    for (op_idx, op) in operands.iter().enumerate() {
        let labels = &expr.inputs[op_idx];
        let shape = op.shape();
        for (i, &lab) in labels.iter().enumerate() {
            label_size.entry(lab).or_insert(shape[i]);
        }
    }

    while current_ops.len() >= 2 {
        // #418: greedy contraction-order heuristic — pick the pair
        // (i, j) that produces the smallest intermediate.
        // Reorders current_ops/current_inputs so the chosen pair is at
        // positions [0..2], then proceeds as the original L-to-R path.
        if current_ops.len() > 2 {
            let (best_i, best_j) =
                pick_greedy_pair(&current_inputs, &final_output, &label_size);
            // Move the chosen pair to positions 0 and 1.
            // Swap operates in-place to avoid extra allocation.
            // Use 0 first to avoid index invalidation.
            if best_j == 0 {
                current_ops.swap(0, best_i);
                current_inputs.swap(0, best_i);
            } else if best_i == 0 {
                current_ops.swap(1, best_j);
                current_inputs.swap(1, best_j);
            } else {
                current_ops.swap(0, best_i);
                current_inputs.swap(0, best_i);
                current_ops.swap(1, best_j);
                current_inputs.swap(1, best_j);
            }
        }

        let a_labels = &current_inputs[0];
        let b_labels = &current_inputs[1];

        let intermediate_labels: Vec<Label> = if current_ops.len() == 2 {
            // Last pair: target the final einsum output directly.
            final_output.clone()
        } else {
            // Keep any label that's needed downstream (in operand 2..N)
            // or in the final output, and that appears in operand 0 or 1.
            let mut kept: Vec<Label> = Vec::new();
            for &lab in a_labels.iter().chain(b_labels.iter()) {
                if kept.contains(&lab) {
                    continue;
                }
                let needed_downstream = current_inputs[2..]
                    .iter()
                    .any(|inp| inp.contains(&lab));
                let needed_in_output = final_output.contains(&lab);
                if needed_downstream || needed_in_output {
                    kept.push(lab);
                }
            }
            kept
        };

        // Build the subscript string for this 2-operand contraction.
        let subs = format!(
            "{}{}{}{}{}",
            labels_to_string(a_labels),
            ",",
            labels_to_string(b_labels),
            "->",
            labels_to_string(&intermediate_labels),
        );

        // Recursive einsum call on the 2-operand subexpression. The
        // 2-operand path will pick matmul/tensordot/generic as
        // appropriate.
        let pair_result = einsum::<T>(&subs, &[&current_ops[0], &current_ops[1]])?;

        current_ops.drain(0..2);
        current_inputs.drain(0..2);
        current_ops.insert(0, pair_result);
        current_inputs.insert(0, intermediate_labels);
    }

    Ok(current_ops.into_iter().next().unwrap())
}

/// Greedy pair selection for multi-operand einsum (#418).
///
/// At each pairwise step we pick the (i, j) whose contraction produces
/// the smallest intermediate. The intermediate's labels are exactly
/// those labels in operand i or j that are also needed downstream
/// (in another remaining operand or in the final output). Smaller
/// intermediates dominate cost in chain expressions like
/// 'ab,bc,cd,de->ae' — picking adjacent pairs that share only one
/// axis with the rest produces a row-vector intermediate of size
/// O(n^2) instead of O(n^3).
///
/// Returns `(i, j)` with `i < j`. Always finds a valid pair when
/// there are at least 2 operands.
fn pick_greedy_pair(
    inputs: &[Vec<Label>],
    output: &[Label],
    label_size: &std::collections::HashMap<Label, usize>,
) -> (usize, usize) {
    debug_assert!(inputs.len() >= 2);
    let mut best: Option<(usize, usize, usize)> = None;
    for i in 0..inputs.len() {
        for j in (i + 1)..inputs.len() {
            // Compute the intermediate-label set: every label in
            // inputs[i] ∪ inputs[j] that appears either in another
            // operand or in the output.
            let mut intermediate_size: usize = 1;
            let mut seen: Vec<Label> = Vec::new();
            for &lab in inputs[i].iter().chain(inputs[j].iter()) {
                if seen.contains(&lab) {
                    continue;
                }
                seen.push(lab);
                let needed_downstream = inputs
                    .iter()
                    .enumerate()
                    .any(|(k, inp)| k != i && k != j && inp.contains(&lab));
                let needed_in_output = output.contains(&lab);
                if needed_downstream || needed_in_output {
                    intermediate_size = intermediate_size.saturating_mul(
                        *label_size.get(&lab).unwrap_or(&1),
                    );
                }
            }
            match best {
                None => best = Some((i, j, intermediate_size)),
                Some((_, _, prev_size)) if intermediate_size < prev_size => {
                    best = Some((i, j, intermediate_size));
                }
                _ => {}
            }
        }
    }
    let (i, j, _) = best.expect("at least 2 operands");
    (i, j)
}

/// Render a `Vec<Label>` as a subscript string for the parser.
fn labels_to_string(labels: &[Label]) -> String {
    labels
        .iter()
        .map(|l| match l {
            Label::Char(c) => *c,
            // Ellipsis-bearing intermediate contractions aren't generated
            // by einsum_pairwise (we reduce ellipsis-aware expressions
            // via the existing generic path), so we don't need to emit
            // "..." in this serialiser. Mark unreachable to surface a
            // bug if we ever do.
            Label::Ellipsis(_) => '?',
        })
        .collect()
}

fn execute_matmul<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    expr: &parser::EinsumExpr,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Simple 2-D matmul: use the shared faer-backed helper so we get the
    // same small/large dispatch and BLAS-quality performance as the
    // top-level `matmul` entry point.
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let (m, k) = (a_shape[0], a_shape[1]);
        let n = b_shape[1];
        let a_data: Vec<T> = if let Some(s) = a.as_slice() {
            s.to_vec()
        } else {
            a.iter().copied().collect()
        };
        let b_data: Vec<T> = if let Some(s) = b.as_slice() {
            s.to_vec()
        } else {
            b.iter().copied().collect()
        };
        let result = crate::products::matmul_raw::<T>(&a_data, &b_data, m, k, n);
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

    #[test]
    fn einsum_three_operand_chain_matches_generic() {
        // 'ij,jk,kl->il' — chain matmul. Pairwise contraction should
        // produce the same result as generic_contraction. (#102)
        // Concrete test: A (2x3) @ B (3x4) @ C (4x2) = result (2x2).
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 4]),
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let c = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[4, 2]),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        // Pairwise path (this is what einsum() uses by default for 3+ operands).
        let pairwise = einsum::<f64>("ij,jk,kl->il", &[&a, &b, &c]).unwrap();
        // Force the generic path for comparison.
        let expr = parse_subscripts("ij,jk,kl->il", &[a.shape(), b.shape(), c.shape()]).unwrap();
        let generic = generic_contraction::<f64>(&expr, &[&a, &b, &c]).unwrap();

        assert_eq!(pairwise.shape(), generic.shape());
        for (p, g) in pairwise.iter().zip(generic.iter()) {
            assert!(
                (p - g).abs() < 1e-10,
                "pairwise vs generic mismatch: {p} vs {g}",
            );
        }
    }

    #[test]
    fn einsum_three_operand_inner_chain() {
        // 'i,ij,j->' — full reduction through three operands.
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let m = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 1.0, 0.0]).unwrap();
        // a^T M b = [1,2] · [[1,0,0],[0,1,0]] · [1,1,0] = [1,2] · [1,1] = 1+2 = 3
        let r = einsum::<f64>("i,ij,j->", &[&a, &m, &b]).unwrap();
        let val: f64 = r.iter().copied().next().unwrap();
        assert!((val - 3.0).abs() < 1e-10, "got {val}, expected 3");
    }

    #[test]
    fn einsum_four_operand_chain_greedy_order_correct() {
        // #418: 'ab,bc,cd,de->ae' — chain expression where the greedy
        // picker may choose a different contraction order than
        // left-to-right. Verify the result matches the analytical
        // matrix-product expectation regardless of order picked.
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2]),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2]),
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let c = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2]),
            vec![2.0, 0.0, 0.0, 2.0],
        )
        .unwrap();
        let d = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2]),
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        // a * b = a (b is identity), * c = 2a, * d:
        //   [[1,2],[3,4]] * 2 = [[2,4],[6,8]]
        //   [[2,4],[6,8]] * [[1,1],[1,1]] = [[6,6],[14,14]]
        let r = einsum::<f64>("ab,bc,cd,de->ae", &[&a, &b, &c, &d]).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        let v: Vec<f64> = r.iter().copied().collect();
        assert!((v[0] - 6.0).abs() < 1e-10);
        assert!((v[1] - 6.0).abs() < 1e-10);
        assert!((v[2] - 14.0).abs() < 1e-10);
        assert!((v[3] - 14.0).abs() < 1e-10);
    }
}
