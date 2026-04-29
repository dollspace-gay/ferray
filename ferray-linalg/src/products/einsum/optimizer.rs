// ferray-linalg: Einsum optimizer (REQ-26)
//
// For 2-operand cases, detect and dispatch to matmul/tensordot when possible.

use super::parser::{EinsumExpr, Label};

/// Strategy for executing an einsum expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EinsumStrategy {
    /// Dispatch to matmul: standard matrix multiplication.
    Matmul,
    /// Dispatch to tensordot with specific axes.
    Tensordot {
        /// Axes of the first operand to contract.
        axes_a: Vec<usize>,
        /// Axes of the second operand to contract.
        axes_b: Vec<usize>,
    },
    /// Use the generic contraction loop.
    Generic,
}

/// Analyze a 2-operand einsum expression and determine the optimal strategy.
#[must_use]
pub fn optimize(expr: &EinsumExpr) -> EinsumStrategy {
    if expr.inputs.len() != 2 {
        return EinsumStrategy::Generic;
    }

    let input_a = &expr.inputs[0];
    let input_b = &expr.inputs[1];
    let output = &expr.output;

    // Check if this is a matmul: "ij,jk->ik" or "...ij,...jk->...ik"
    if is_matmul_pattern(input_a, input_b, output) {
        return EinsumStrategy::Matmul;
    }

    // Check if this can be expressed as tensordot
    if let Some(strategy) = try_tensordot(input_a, input_b, output) {
        return strategy;
    }

    EinsumStrategy::Generic
}

fn is_matmul_pattern(a: &[Label], b: &[Label], out: &[Label]) -> bool {
    // Matmul pattern: batch dims match, a has ...ij, b has ...jk, output has ...ik
    if a.len() < 2 || b.len() < 2 {
        return false;
    }

    let a_batch = &a[..a.len() - 2];
    let b_batch = &b[..b.len() - 2];
    let out_batch = if out.len() >= 2 {
        &out[..out.len() - 2]
    } else {
        return false;
    };

    // Check batch dims match
    if a_batch != b_batch || a_batch != out_batch {
        return false;
    }

    let a_i = a[a.len() - 2];
    let a_j = a[a.len() - 1];
    let b_j = b[b.len() - 2];
    let b_k = b[b.len() - 1];
    let out_i = out[out.len() - 2];
    let out_k = out[out.len() - 1];

    // Pattern: a[.., i, j] @ b[.., j, k] -> out[.., i, k]
    a_j == b_j && a_i == out_i && b_k == out_k && a_i != a_j && b_j != b_k
}

fn try_tensordot(a: &[Label], b: &[Label], out: &[Label]) -> Option<EinsumStrategy> {
    // Find contracted axes: labels that appear in both a and b but not in output
    let out_set: std::collections::HashSet<Label> = out.iter().copied().collect();

    let mut axes_a = Vec::new();
    let mut axes_b = Vec::new();

    for (ai, &la) in a.iter().enumerate() {
        if !out_set.contains(&la) {
            // This label is contracted
            for (bi, &lb) in b.iter().enumerate() {
                if la == lb && !axes_b.contains(&bi) {
                    axes_a.push(ai);
                    axes_b.push(bi);
                    break;
                }
            }
        }
    }

    if axes_a.is_empty() {
        return None;
    }

    // Verify that the output order matches tensordot's natural order:
    // free_a dims then free_b dims
    let a_free: Vec<Label> = a
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes_a.contains(i))
        .map(|(_, &l)| l)
        .collect();
    let b_free: Vec<Label> = b
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes_b.contains(i))
        .map(|(_, &l)| l)
        .collect();

    let natural_out: Vec<Label> = a_free.iter().chain(b_free.iter()).copied().collect();
    if natural_out == *out {
        Some(EinsumStrategy::Tensordot { axes_a, axes_b })
    } else {
        // Output order doesn't match; fall back to generic
        Some(EinsumStrategy::Generic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::products::einsum::parser::parse_subscripts;

    #[test]
    fn detect_matmul() {
        let expr = parse_subscripts("ij,jk->ik", &[&[2, 3], &[3, 4]]).unwrap();
        assert_eq!(optimize(&expr), EinsumStrategy::Matmul);
    }

    #[test]
    fn detect_batch_matmul() {
        let expr = parse_subscripts("bij,bjk->bik", &[&[2, 3, 4], &[2, 4, 5]]).unwrap();
        assert_eq!(optimize(&expr), EinsumStrategy::Matmul);
    }

    #[test]
    fn detect_tensordot() {
        // ij,jk->ik could also be tensordot with axes=([1],[0])
        // but should be detected as matmul first
        let expr = parse_subscripts("ij,jk->ik", &[&[2, 3], &[3, 4]]).unwrap();
        assert_eq!(optimize(&expr), EinsumStrategy::Matmul);
    }

    #[test]
    fn generic_trace() {
        let expr = parse_subscripts("ii->i", &[&[3, 3]]).unwrap();
        assert_eq!(optimize(&expr), EinsumStrategy::Generic);
    }
}
