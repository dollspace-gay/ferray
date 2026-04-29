// ferray-linalg: Einsum subscript parser (REQ-25)
//
// Parses subscript strings like "ij,jk->ik", "ij,jk", "ii->i", "...ij,...jk->...ik"

// `parse_subscripts` and `generic_contraction` are parser/dispatcher
// functions that genuinely fan out across the einsum spec — splitting
// them would just produce arbitrary helper boundaries. The wildcard
// match arms over `Label` are deliberate fall-throughs (`Label::Char` is
// the only "other" today; future variants should expand explicitly here
// not silently slip through).
#![allow(clippy::too_many_lines, clippy::match_wildcard_for_single_variants)]

use ferray_core::error::{FerrayError, FerrayResult};
use std::collections::HashMap;

/// Parsed einsum expression.
#[derive(Debug, Clone)]
pub struct EinsumExpr {
    /// Input subscript labels for each operand.
    pub inputs: Vec<Vec<Label>>,
    /// Output subscript labels. If `None`, implicit mode is used.
    pub output: Vec<Label>,
    /// Whether the expression uses ellipsis broadcasting.
    pub has_ellipsis: bool,
}

/// A single label in a subscript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    /// A named axis (a-z).
    Char(char),
    /// An ellipsis placeholder dimension (index within the broadcast dims).
    Ellipsis(usize),
}

/// Parse an einsum subscript string.
///
/// Supports:
/// - Explicit output: `"ij,jk->ik"`
/// - Implicit output: `"ij,jk"` (output is alphabetically sorted unique labels that appear once)
/// - Trace: `"ii->i"` or `"ii"`
/// - Batch: `"bij,bjk->bik"`
/// - Ellipsis: `"...ij,...jk->...ik"`
///
/// # Errors
/// - `FerrayError::InvalidValue` for malformed subscripts.
pub fn parse_subscripts(subscripts: &str, operand_shapes: &[&[usize]]) -> FerrayResult<EinsumExpr> {
    let subscripts = subscripts.replace(' ', "");
    let has_ellipsis = subscripts.contains("...");

    let (inputs_str, output_str) = if let Some(arrow_pos) = subscripts.find("->") {
        let inputs = &subscripts[..arrow_pos];
        let output = &subscripts[arrow_pos + 2..];
        (inputs.to_string(), Some(output.to_string()))
    } else {
        (subscripts, None)
    };

    let input_strs: Vec<&str> = inputs_str.split(',').collect();
    if input_strs.len() != operand_shapes.len() {
        return Err(FerrayError::invalid_value(format!(
            "einsum: subscript has {} operands but {} were provided",
            input_strs.len(),
            operand_shapes.len()
        )));
    }

    // Parse each input
    let mut inputs: Vec<Vec<Label>> = Vec::with_capacity(input_strs.len());
    let mut label_counts: HashMap<char, usize> = HashMap::new();

    // Determine ellipsis dimensions
    let mut ellipsis_ndims: Option<usize> = None;

    for (idx, &input_str) in input_strs.iter().enumerate() {
        let shape = operand_shapes[idx];
        let (labels, n_explicit) = parse_input_labels(input_str, has_ellipsis)?;

        if has_ellipsis && labels.iter().any(|l| matches!(l, Label::Ellipsis(_))) {
            let n_ellipsis = shape.len().saturating_sub(n_explicit);
            match ellipsis_ndims {
                None => ellipsis_ndims = Some(n_ellipsis),
                Some(existing) => {
                    let max_e = existing.max(n_ellipsis);
                    ellipsis_ndims = Some(max_e);
                }
            }
        }

        inputs.push(labels);
    }

    // Expand ellipsis placeholders
    let ellipsis_ndims = ellipsis_ndims.unwrap_or(0);
    let mut expanded_inputs: Vec<Vec<Label>> = Vec::with_capacity(inputs.len());
    for (idx, labels) in inputs.iter().enumerate() {
        let shape = operand_shapes[idx];
        let mut expanded = Vec::new();
        let n_explicit = labels
            .iter()
            .filter(|l| !matches!(l, Label::Ellipsis(_)))
            .count();
        let n_ellipsis_here = shape.len().saturating_sub(n_explicit);

        for label in labels {
            match label {
                Label::Ellipsis(_) => {
                    // Expand to n_ellipsis_here dims, padded on the left
                    let start = ellipsis_ndims - n_ellipsis_here;
                    for i in start..ellipsis_ndims {
                        expanded.push(Label::Ellipsis(i));
                    }
                }
                other => expanded.push(*other),
            }
        }

        // Verify expanded label count matches shape
        if expanded.len() != shape.len() {
            return Err(FerrayError::invalid_value(format!(
                "einsum: operand {} has {} dimensions but subscript implies {}",
                idx,
                shape.len(),
                expanded.len()
            )));
        }

        for l in &expanded {
            if let Label::Char(c) = l {
                *label_counts.entry(*c).or_insert(0) += 1;
            }
        }

        expanded_inputs.push(expanded);
    }

    // Parse or compute output
    let output = if let Some(ref out_str) = output_str {
        let (out_labels, _) = parse_input_labels(out_str, has_ellipsis)?;
        // Expand ellipsis in output
        let mut expanded_out = Vec::new();
        for label in &out_labels {
            match label {
                Label::Ellipsis(_) => {
                    for i in 0..ellipsis_ndims {
                        expanded_out.push(Label::Ellipsis(i));
                    }
                }
                other => expanded_out.push(*other),
            }
        }
        expanded_out
    } else {
        // Implicit mode: output labels are those that appear exactly once,
        // sorted alphabetically
        let mut output_chars: Vec<char> = label_counts
            .iter()
            .filter(|&(_, &count)| count == 1)
            .map(|(&c, _)| c)
            .collect();
        output_chars.sort_unstable();

        let mut out = Vec::new();
        // Add ellipsis dims first if present
        if has_ellipsis {
            for i in 0..ellipsis_ndims {
                out.push(Label::Ellipsis(i));
            }
        }
        for c in output_chars {
            out.push(Label::Char(c));
        }
        out
    };

    Ok(EinsumExpr {
        inputs: expanded_inputs,
        output,
        has_ellipsis,
    })
}

fn parse_input_labels(s: &str, has_ellipsis: bool) -> FerrayResult<(Vec<Label>, usize)> {
    let mut labels = Vec::new();
    let mut n_explicit = 0;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '.'
            && has_ellipsis
            && i + 2 < chars.len()
            && chars[i + 1] == '.'
            && chars[i + 2] == '.'
        {
            labels.push(Label::Ellipsis(0)); // placeholder, will be expanded
            i += 3;
        } else if chars[i].is_ascii_lowercase() {
            labels.push(Label::Char(chars[i]));
            n_explicit += 1;
            i += 1;
        } else {
            return Err(FerrayError::invalid_value(format!(
                "einsum: invalid character '{}' in subscript",
                chars[i]
            )));
        }
    }
    Ok((labels, n_explicit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_matmul_explicit() {
        let expr = parse_subscripts("ij,jk->ik", &[&[2, 3], &[3, 4]]).unwrap();
        assert_eq!(expr.inputs.len(), 2);
        assert_eq!(expr.inputs[0].len(), 2);
        assert_eq!(expr.output.len(), 2);
        assert!(!expr.has_ellipsis);
    }

    #[test]
    fn parse_matmul_implicit() {
        let expr = parse_subscripts("ij,jk", &[&[2, 3], &[3, 4]]).unwrap();
        // j appears twice, so not in output. i and k appear once.
        assert_eq!(expr.output.len(), 2);
        assert_eq!(expr.output[0], Label::Char('i'));
        assert_eq!(expr.output[1], Label::Char('k'));
    }

    #[test]
    fn parse_trace() {
        let expr = parse_subscripts("ii->i", &[&[3, 3]]).unwrap();
        assert_eq!(expr.inputs[0], vec![Label::Char('i'), Label::Char('i')]);
        assert_eq!(expr.output, vec![Label::Char('i')]);
    }

    #[test]
    fn parse_trace_scalar() {
        let expr = parse_subscripts("ii", &[&[3, 3]]).unwrap();
        // i appears twice, so empty output (summed out)
        assert!(expr.output.is_empty());
    }

    #[test]
    fn parse_batch() {
        let expr = parse_subscripts("bij,bjk->bik", &[&[2, 3, 4], &[2, 4, 5]]).unwrap();
        assert_eq!(expr.inputs[0].len(), 3);
        assert_eq!(expr.output.len(), 3);
    }

    #[test]
    fn parse_ellipsis() {
        let expr = parse_subscripts("...ij,...jk->...ik", &[&[2, 3, 4], &[2, 4, 5]]).unwrap();
        assert!(expr.has_ellipsis);
    }

    #[test]
    fn wrong_operand_count() {
        let result = parse_subscripts("ij,jk->ik", &[&[2, 3]]);
        assert!(result.is_err());
    }
}
