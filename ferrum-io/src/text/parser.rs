// ferrum-io: Delimited text parser with missing value handling

use ferrum_core::error::{FerrumError, FerrumResult};

/// Options for parsing delimited text files.
#[derive(Debug, Clone)]
pub struct TextParseOptions {
    /// Column delimiter character (default: ',').
    pub delimiter: char,
    /// Number of header rows to skip (default: 0).
    pub skiprows: usize,
    /// Comment character: lines starting with this (after trimming) are skipped.
    pub comments: Option<char>,
    /// Maximum number of rows to read (None = all).
    pub max_rows: Option<usize>,
}

impl Default for TextParseOptions {
    fn default() -> Self {
        Self {
            delimiter: ',',
            skiprows: 0,
            comments: Some('#'),
            max_rows: None,
        }
    }
}

/// Parse a delimited text into a 2D grid of string cells.
///
/// Returns `(rows, ncols)` where `rows` is a flat vector of cells in row-major order.
pub fn parse_text_grid(
    content: &str,
    opts: &TextParseOptions,
) -> FerrumResult<(Vec<String>, usize, usize)> {
    let all_lines: Vec<&str> = content.lines().collect();

    // Filter comments and empty lines first, then skip rows.
    // This matches NumPy behavior: comments are stripped, then skiprows
    // counts over the remaining data lines.
    let non_comment_lines: Vec<&str> = all_lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return false;
            }
            if let Some(comment_char) = opts.comments {
                if trimmed.starts_with(comment_char) {
                    return false;
                }
            }
            true
        })
        .copied()
        .collect();

    // Skip initial rows
    let data_lines: &[&str] = if opts.skiprows < non_comment_lines.len() {
        &non_comment_lines[opts.skiprows..]
    } else {
        &[]
    };

    // Apply max_rows
    let data_lines = if let Some(max) = opts.max_rows {
        &data_lines[..data_lines.len().min(max)]
    } else {
        data_lines
    };

    if data_lines.is_empty() {
        return Ok((vec![], 0, 0));
    }

    let delim = opts.delimiter;
    let nrows = data_lines.len();

    // Parse first line to determine number of columns
    let first_fields: Vec<&str> = data_lines[0].split(delim).collect();
    let ncols = first_fields.len();

    let mut cells = Vec::with_capacity(nrows * ncols);

    for (row_idx, line) in data_lines.iter().enumerate() {
        let fields: Vec<&str> = line.split(delim).collect();
        if fields.len() != ncols {
            return Err(FerrumError::io_error(format!(
                "row {} has {} columns, expected {} (line: '{}')",
                row_idx + opts.skiprows,
                fields.len(),
                ncols,
                line,
            )));
        }
        for field in fields {
            cells.push(field.trim().to_string());
        }
    }

    Ok((cells, nrows, ncols))
}

/// Parse a delimited text into a 2D grid, allowing missing values.
///
/// Missing values (empty cells or cells matching `missing_values`) are returned
/// as `None`. Present values are returned as `Some(String)`.
pub fn parse_text_grid_with_missing(
    content: &str,
    opts: &TextParseOptions,
    missing_values: &[&str],
) -> FerrumResult<(Vec<Option<String>>, usize, usize)> {
    let all_lines: Vec<&str> = content.lines().collect();

    // Filter comments and empty lines first, then skip rows.
    let non_comment_lines: Vec<&str> = all_lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return false;
            }
            if let Some(comment_char) = opts.comments {
                if trimmed.starts_with(comment_char) {
                    return false;
                }
            }
            true
        })
        .copied()
        .collect();

    let data_lines: &[&str] = if opts.skiprows < non_comment_lines.len() {
        &non_comment_lines[opts.skiprows..]
    } else {
        &[]
    };

    let data_lines = if let Some(max) = opts.max_rows {
        &data_lines[..data_lines.len().min(max)]
    } else {
        data_lines
    };

    if data_lines.is_empty() {
        return Ok((vec![], 0, 0));
    }

    let delim = opts.delimiter;
    let nrows = data_lines.len();

    let first_fields: Vec<&str> = data_lines[0].split(delim).collect();
    let ncols = first_fields.len();

    let mut cells = Vec::with_capacity(nrows * ncols);

    for (row_idx, line) in data_lines.iter().enumerate() {
        let fields: Vec<&str> = line.split(delim).collect();
        // Allow rows with fewer columns (missing trailing values)
        for col_idx in 0..ncols {
            if col_idx >= fields.len() {
                cells.push(None);
            } else {
                let field = fields[col_idx].trim();
                if field.is_empty() || missing_values.contains(&field) {
                    cells.push(None);
                } else {
                    cells.push(Some(field.to_string()));
                }
            }
        }
        // Extra columns beyond ncols generate an error
        if fields.len() > ncols {
            return Err(FerrumError::io_error(format!(
                "row {} has {} columns, expected {} (line: '{}')",
                row_idx + opts.skiprows,
                fields.len(),
                ncols,
                line,
            )));
        }
    }

    Ok((cells, nrows, ncols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_csv() {
        let content = "1,2,3\n4,5,6\n";
        let opts = TextParseOptions {
            delimiter: ',',
            ..Default::default()
        };
        let (cells, nrows, ncols) = parse_text_grid(content, &opts).unwrap();
        assert_eq!(nrows, 2);
        assert_eq!(ncols, 3);
        assert_eq!(cells, vec!["1", "2", "3", "4", "5", "6"]);
    }

    #[test]
    fn parse_with_skiprows() {
        let content = "# header\nname,value\n1,10\n2,20\n";
        let opts = TextParseOptions {
            delimiter: ',',
            skiprows: 1,
            comments: Some('#'),
            ..Default::default()
        };
        let (cells, nrows, ncols) = parse_text_grid(content, &opts).unwrap();
        assert_eq!(nrows, 2);
        assert_eq!(ncols, 2);
        assert_eq!(cells[0], "1");
    }

    #[test]
    fn parse_with_comments() {
        let content = "1,2\n# comment\n3,4\n";
        let opts = TextParseOptions {
            delimiter: ',',
            comments: Some('#'),
            ..Default::default()
        };
        let (cells, nrows, _) = parse_text_grid(content, &opts).unwrap();
        assert_eq!(nrows, 2);
        assert_eq!(cells, vec!["1", "2", "3", "4"]);
    }

    #[test]
    fn parse_tab_delimited() {
        let content = "1\t2\t3\n4\t5\t6\n";
        let opts = TextParseOptions {
            delimiter: '\t',
            ..Default::default()
        };
        let (cells, nrows, ncols) = parse_text_grid(content, &opts).unwrap();
        assert_eq!(nrows, 2);
        assert_eq!(ncols, 3);
        assert_eq!(cells[0], "1");
    }

    #[test]
    fn parse_inconsistent_columns_error() {
        let content = "1,2,3\n4,5\n";
        let opts = TextParseOptions::default();
        assert!(parse_text_grid(content, &opts).is_err());
    }

    #[test]
    fn parse_missing_values() {
        let content = "1,2,3\n4,,6\n7,8,\n";
        let opts = TextParseOptions::default();
        let (cells, nrows, ncols) =
            parse_text_grid_with_missing(content, &opts, &[]).unwrap();
        assert_eq!(nrows, 3);
        assert_eq!(ncols, 3);
        assert_eq!(cells[0], Some("1".to_string()));
        assert_eq!(cells[4], None); // empty field
        assert_eq!(cells[8], None); // trailing empty
    }

    #[test]
    fn parse_custom_missing_marker() {
        let content = "1,NA,3\n4,5,NA\n";
        let opts = TextParseOptions::default();
        let (cells, _, _) =
            parse_text_grid_with_missing(content, &opts, &["NA"]).unwrap();
        assert_eq!(cells[1], None);
        assert_eq!(cells[5], None);
        assert_eq!(cells[0], Some("1".to_string()));
    }

    #[test]
    fn parse_empty_content() {
        let content = "";
        let opts = TextParseOptions::default();
        let (cells, nrows, ncols) = parse_text_grid(content, &opts).unwrap();
        assert_eq!(nrows, 0);
        assert_eq!(ncols, 0);
        assert!(cells.is_empty());
    }
}
