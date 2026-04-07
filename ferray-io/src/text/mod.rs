// ferray-io: Text I/O
//
// REQ-7: savetxt(path, &array, delimiter, fmt) writes 2D array as delimited text
// REQ-8: loadtxt::<T>(path, delimiter, skiprows) reads delimited text into 2D array
// REQ-9: genfromtxt(path, delimiter, filling_values) reads text with missing value handling

pub mod parser;

use std::fmt::Display;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

use ferray_core::Array;
use ferray_core::dimension::Ix2;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use self::parser::{TextParseOptions, parse_text_grid, parse_text_grid_with_missing};

/// Options for saving text files.
#[derive(Debug, Clone)]
pub struct SaveTxtOptions {
    /// Column delimiter (default: ',').
    pub delimiter: char,
    /// Format string for each element. Uses Rust format syntax.
    /// If `None`, the default `Display` formatting is used.
    pub fmt: Option<String>,
    /// Optional header line written before data.
    pub header: Option<String>,
    /// Optional footer line written after data.
    pub footer: Option<String>,
    /// Line ending (default: "\n").
    pub newline: String,
}

impl Default for SaveTxtOptions {
    fn default() -> Self {
        Self {
            delimiter: ',',
            fmt: None,
            header: None,
            footer: None,
            newline: "\n".to_string(),
        }
    }
}

/// Format a single value using a format string.
///
/// Supports:
/// - NumPy printf-style: `"%.6e"`, `"%.18e"`, `"%10.5f"`, `"%.4f"`, `"%d"`
/// - Rust-style with braces: `"{:.6}"`, `"{:.6e}"`, `"{:>10.5}"`
/// - Plain `"{}"` — default Display
///
/// Unrecognized patterns fall back to default Display formatting.
fn format_value<T: Display>(val: &T, fmt_str: &str) -> String {
    // Rust-style: contains "{" — parse precision patterns.
    if fmt_str.contains('{') {
        if let Some(spec) = fmt_str.strip_prefix("{:").and_then(|s| s.strip_suffix('}')) {
            if let Some(prec_str) = spec.strip_prefix('.') {
                let is_sci = prec_str.ends_with('e') || prec_str.ends_with('E');
                let digits_str = if is_sci {
                    &prec_str[..prec_str.len() - 1]
                } else {
                    prec_str
                };
                if let Ok(prec) = digits_str.parse::<usize>() {
                    // Parse value as f64 for numeric formatting
                    if let Ok(v) = val.to_string().parse::<f64>() {
                        return if is_sci {
                            format!("{v:.prec$e}")
                        } else {
                            format!("{v:.prec$}")
                        };
                    }
                }
            }
        }
        // Fallback: simple substitution
        return fmt_str.replace("{}", &val.to_string());
    }

    // NumPy printf-style: starts with "%"
    if let Some(spec) = fmt_str.strip_prefix('%') {
        let (body, mode) = if let Some(rest) = spec.strip_suffix('e') {
            (rest, 'e')
        } else if let Some(rest) = spec.strip_suffix('E') {
            (rest, 'E')
        } else if let Some(rest) = spec.strip_suffix('f') {
            (rest, 'f')
        } else if let Some(rest) = spec.strip_suffix('g') {
            (rest, 'g')
        } else {
            // %d, %i, or unrecognized — use default Display
            return format!("{val}");
        };

        // Parse value as f64 for numeric formatting
        if let Ok(v) = val.to_string().parse::<f64>() {
            if let Some(dot_pos) = body.find('.') {
                let prec_str = &body[dot_pos + 1..];
                if let Ok(prec) = prec_str.parse::<usize>() {
                    return match mode {
                        'e' => format!("{v:.prec$e}"),
                        'E' => format!("{v:.prec$E}"),
                        _ => format!("{v:.prec$}"),
                    };
                }
            } else if body.is_empty() {
                return match mode {
                    'e' => format!("{v:e}"),
                    'E' => format!("{v:E}"),
                    _ => format!("{v}"),
                };
            }
        }
    }

    // Unrecognized format — default Display
    format!("{val}")
}

/// Save a 2D array as delimited text.
///
/// The `fmt` field in [`SaveTxtOptions`] supports:
/// - NumPy printf-style: `"%.6e"`, `"%.18e"`, `"%10.5f"`, `"%d"`
/// - Rust-style: `"{:.6}"`, `"{:.6e}"`
/// - Default: `None` uses standard `Display` formatting.
///
/// # Errors
/// Returns `FerrayError::IoError` on file write failures.
/// Returns `FerrayError::IoError` if the array is not contiguous.
pub fn savetxt<T: Element + Display, P: AsRef<Path>>(
    path: P,
    array: &Array<T, Ix2>,
    opts: &SaveTxtOptions,
) -> FerrayResult<()> {
    let mut file = std::fs::File::create(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to create file '{}': {e}",
            path.as_ref().display()
        ))
    })?;

    savetxt_to_writer(&mut file, array, opts)
}

/// Save a 2D array as delimited text to a writer.
pub fn savetxt_to_writer<T: Element + Display, W: Write>(
    writer: &mut W,
    array: &Array<T, Ix2>,
    opts: &SaveTxtOptions,
) -> FerrayResult<()> {
    let shape = array.shape();
    let nrows = shape[0];
    let ncols = shape[1];

    if let Some(ref header) = opts.header {
        write!(writer, "{header}").map_err(|e| FerrayError::io_error(e.to_string()))?;
        writer
            .write_all(opts.newline.as_bytes())
            .map_err(|e| FerrayError::io_error(e.to_string()))?;
    }

    let slice = array
        .as_slice()
        .ok_or_else(|| FerrayError::io_error("cannot save non-contiguous array as text"))?;

    for row in 0..nrows {
        for col in 0..ncols {
            if col > 0 {
                write!(writer, "{}", opts.delimiter)
                    .map_err(|e| FerrayError::io_error(e.to_string()))?;
            }
            let val = &slice[row * ncols + col];
            if let Some(ref fmt_str) = opts.fmt {
                // Format string support:
                // - NumPy printf-style: "%.6e", "%.18e", "%10.5f", "%d"
                // - Rust-style: "{:.6}", "{:.6e}", "{:>10.5}"
                // We convert common printf patterns to Rust format, then
                // fall back to string substitution.
                let formatted = format_value(val, fmt_str);
                write!(writer, "{formatted}").map_err(|e| FerrayError::io_error(e.to_string()))?;
            } else {
                write!(writer, "{val}").map_err(|e| FerrayError::io_error(e.to_string()))?;
            }
        }
        writer
            .write_all(opts.newline.as_bytes())
            .map_err(|e| FerrayError::io_error(e.to_string()))?;
    }

    if let Some(ref footer) = opts.footer {
        write!(writer, "{footer}").map_err(|e| FerrayError::io_error(e.to_string()))?;
        writer
            .write_all(opts.newline.as_bytes())
            .map_err(|e| FerrayError::io_error(e.to_string()))?;
    }

    writer
        .flush()
        .map_err(|e| FerrayError::io_error(e.to_string()))?;
    Ok(())
}

/// Load a delimited text file into a 2D array.
///
/// Each row of the text file becomes a row in the array. All rows must
/// have the same number of columns.
///
/// # Type Parameters
/// - `T`: Element type to parse each cell into. Must implement `FromStr`.
///
/// # Errors
/// - Returns `FerrayError::IoError` on file read or parse failures.
pub fn loadtxt<T, P>(path: P, delimiter: char, skiprows: usize) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + FromStr,
    T::Err: Display,
    P: AsRef<Path>,
{
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to read file '{}': {e}",
            path.as_ref().display()
        ))
    })?;

    loadtxt_from_str(&content, delimiter, skiprows)
}

/// Load delimited text from a string into a 2D array.
pub fn loadtxt_from_str<T>(
    content: &str,
    delimiter: char,
    skiprows: usize,
) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + FromStr,
    T::Err: Display,
{
    let opts = TextParseOptions {
        delimiter,
        skiprows,
        ..Default::default()
    };

    let (cells, nrows, ncols) = parse_text_grid(content, &opts)?;

    if nrows == 0 {
        return Array::from_vec(Ix2::new([0, 0]), vec![]);
    }

    let data: FerrayResult<Vec<T>> = cells
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            cell.parse::<T>().map_err(|e| {
                let row = i / ncols;
                let col = i % ncols;
                FerrayError::io_error(format!(
                    "failed to parse value '{cell}' at row {row}, col {col}: {e}"
                ))
            })
        })
        .collect();

    let data = data?;
    Array::from_vec(Ix2::new([nrows, ncols]), data)
}

/// Load a delimited text file with missing value handling.
///
/// Missing values (empty cells or cells matching common missing indicators)
/// are replaced with `filling_values`. This is analogous to NumPy's `genfromtxt`.
///
/// Returns a 2D `f64` array where missing values are replaced with `filling_value`
/// (typically `f64::NAN`).
///
/// # Errors
/// Returns `FerrayError::IoError` on file read or parse failures.
pub fn genfromtxt<P: AsRef<Path>>(
    path: P,
    delimiter: char,
    filling_value: f64,
    skiprows: usize,
    missing_values: &[&str],
) -> FerrayResult<Array<f64, Ix2>> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to read file '{}': {e}",
            path.as_ref().display()
        ))
    })?;

    genfromtxt_from_str(&content, delimiter, filling_value, skiprows, missing_values)
}

/// Load delimited text from a string with missing value handling.
pub fn genfromtxt_from_str(
    content: &str,
    delimiter: char,
    filling_value: f64,
    skiprows: usize,
    missing_values: &[&str],
) -> FerrayResult<Array<f64, Ix2>> {
    let opts = TextParseOptions {
        delimiter,
        skiprows,
        ..Default::default()
    };

    // Default missing markers
    let mut all_missing: Vec<&str> = vec!["", "NA", "N/A", "nan", "NaN", "NAN", "--", "null"];
    for mv in missing_values {
        if !all_missing.contains(mv) {
            all_missing.push(mv);
        }
    }

    let (cells, nrows, ncols) = parse_text_grid_with_missing(content, &opts, &all_missing)?;

    if nrows == 0 {
        return Array::from_vec(Ix2::new([0, 0]), vec![]);
    }

    let data: FerrayResult<Vec<f64>> = cells
        .iter()
        .enumerate()
        .map(|(i, cell)| match cell {
            None => Ok(filling_value),
            Some(s) => s.parse::<f64>().map_err(|e| {
                let row = i / ncols;
                let col = i % ncols;
                FerrayError::io_error(format!(
                    "failed to parse value '{s}' at row {row}, col {col}: {e}"
                ))
            }),
        })
        .collect();

    let data = data?;
    Array::from_vec(Ix2::new([nrows, ncols]), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loadtxt_simple_csv() {
        let content = "1.0,2.0,3.0\n4.0,5.0,6.0\n";
        let arr: Array<f64, Ix2> = loadtxt_from_str(content, ',', 0).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn loadtxt_with_skiprows() {
        let content = "# header\nname,value\n1.0,10.0\n2.0,20.0\n";
        let arr: Array<f64, Ix2> = loadtxt_from_str(content, ',', 1).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.as_slice().unwrap()[0], 1.0);
    }

    #[test]
    fn loadtxt_tab_delimited() {
        let content = "1\t2\t3\n4\t5\t6\n";
        let arr: Array<i32, Ix2> = loadtxt_from_str(content, '\t', 0).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn loadtxt_integers() {
        let content = "10,20\n30,40\n";
        let arr: Array<i64, Ix2> = loadtxt_from_str(content, ',', 0).unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[10i64, 20, 30, 40]);
    }

    #[test]
    fn loadtxt_file_roundtrip() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();

        let dir = std::env::temp_dir().join(format!("ferray_io_text_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.csv");

        savetxt(&path, &arr, &SaveTxtOptions::default()).unwrap();
        let loaded: Array<f64, Ix2> = loadtxt(&path, ',', 0).unwrap();

        assert_eq!(loaded.shape(), &[2, 3]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn savetxt_custom_delimiter() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), data).unwrap();

        let mut buf = Vec::new();
        let opts = SaveTxtOptions {
            delimiter: '\t',
            ..Default::default()
        };
        savetxt_to_writer(&mut buf, &arr, &opts).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains('\t'));
        assert!(!output.contains(','));
    }

    #[test]
    fn savetxt_with_header_footer() {
        let data = vec![1.0f64, 2.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([1, 2]), data).unwrap();

        let mut buf = Vec::new();
        let opts = SaveTxtOptions {
            header: Some("# my header".to_string()),
            footer: Some("# end".to_string()),
            ..Default::default()
        };
        savetxt_to_writer(&mut buf, &arr, &opts).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.starts_with("# my header\n"));
        assert!(output.ends_with("# end\n"));
    }

    #[test]
    fn genfromtxt_missing_nan() {
        let content = "1.0,2.0,3.0\n4.0,,6.0\n7.0,8.0,\n";
        let arr = genfromtxt_from_str(content, ',', f64::NAN, 0, &[]).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        let slice = arr.as_slice().unwrap();
        assert_eq!(slice[0], 1.0);
        assert!(slice[4].is_nan()); // missing value replaced with NaN
        assert!(slice[8].is_nan()); // trailing empty
    }

    #[test]
    fn genfromtxt_na_marker() {
        let content = "1.0,NA,3.0\n4.0,5.0,NA\n";
        let arr = genfromtxt_from_str(content, ',', -999.0, 0, &["NA"]).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        let slice = arr.as_slice().unwrap();
        assert_eq!(slice[1], -999.0);
        assert_eq!(slice[5], -999.0);
    }

    #[test]
    fn genfromtxt_with_skiprows() {
        let content = "col1,col2\n1.0,2.0\n3.0,4.0\n";
        let arr = genfromtxt_from_str(content, ',', f64::NAN, 1, &[]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.as_slice().unwrap()[0], 1.0);
    }

    #[test]
    fn genfromtxt_file() {
        let content = "1.0,2.0\n,4.0\n";
        let dir = std::env::temp_dir().join(format!("ferray_io_text_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("genfromtxt_test.csv");
        std::fs::write(&path, content).unwrap();

        let arr = genfromtxt(&path, ',', f64::NAN, 0, &[]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert!(arr.as_slice().unwrap()[2].is_nan());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loadtxt_empty() {
        let content = "";
        let arr: Array<f64, Ix2> = loadtxt_from_str(content, ',', 0).unwrap();
        assert_eq!(arr.shape(), &[0, 0]);
    }
}
