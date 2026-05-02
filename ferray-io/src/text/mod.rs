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
use ferray_core::dimension::{Ix1, Ix2};
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
/// - `NumPy` printf-style: `"%.6e"`, `"%.18e"`, `"%10.5f"`, `"%.4f"`, `"%d"`
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
/// - `NumPy` printf-style: `"%.6e"`, `"%.18e"`, `"%10.5f"`, `"%d"`
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

/// 1-D variant of [`savetxt`] (#494).
///
/// Equivalent to `numpy.savetxt(path, array)` for a 1-D input —
/// each element is written on its own line. Mirrors numpy's
/// behavior of treating a 1-D array as a single-column 2-D array.
///
/// # Errors
/// Same as [`savetxt`]; additionally if the array is non-contiguous.
pub fn savetxt_1d<T: Element + Display, P: AsRef<Path>>(
    path: P,
    array: &Array<T, Ix1>,
    opts: &SaveTxtOptions,
) -> FerrayResult<()> {
    let mut file = std::fs::File::create(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to create file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    savetxt_1d_to_writer(&mut file, array, opts)
}

/// 1-D variant of [`savetxt_to_writer`].
pub fn savetxt_1d_to_writer<T: Element + Display, W: Write>(
    writer: &mut W,
    array: &Array<T, Ix1>,
    opts: &SaveTxtOptions,
) -> FerrayResult<()> {
    if let Some(ref header) = opts.header {
        write!(writer, "{header}").map_err(|e| FerrayError::io_error(e.to_string()))?;
        writer
            .write_all(opts.newline.as_bytes())
            .map_err(|e| FerrayError::io_error(e.to_string()))?;
    }
    let slice = array
        .as_slice()
        .ok_or_else(|| FerrayError::io_error("cannot save non-contiguous array as text"))?;
    for val in slice {
        let formatted = if let Some(ref fmt_str) = opts.fmt {
            format_value(val, fmt_str)
        } else {
            format!("{val}")
        };
        writer
            .write_all(formatted.as_bytes())
            .map_err(|e| FerrayError::io_error(e.to_string()))?;
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

/// 1-D counterpart of [`loadtxt`] (#494).
///
/// Reads a single-column delimited text file (one value per line)
/// into an `Array<T, Ix1>`. Equivalent to NumPy's
/// `numpy.loadtxt(path, ndmin=1)` when the file has one column.
///
/// # Errors
/// Same as [`loadtxt`].
pub fn loadtxt_1d<T, P>(
    path: P,
    delimiter: char,
    skiprows: usize,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + FromStr,
    T::Err: Display,
    P: AsRef<Path>,
{
    let arr2 = loadtxt::<T, _>(path, delimiter, skiprows)?;
    let shape = arr2.shape();
    let n = shape[0] * shape[1];
    let data: Vec<T> = arr2.iter().cloned().collect();
    Array::<T, Ix1>::from_vec(Ix1::new([n]), data)
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
/// are replaced with `filling_values`. This is analogous to `NumPy`'s `genfromtxt`.
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

// ---------------------------------------------------------------------------
// fromregex
// ---------------------------------------------------------------------------

/// Read text using a regular expression to extract structured groups.
///
/// `regex` must contain at least one capturing group. For every line in
/// `content`, the regex is matched against the full line; matches where every
/// capture is parsed successfully via `T::from_str` produce one row of the
/// output. Lines that do not match (or that contain unparseable captures)
/// are skipped.
///
/// The result is a 2-D `Array<T, Ix2>` of shape `(rows, captures)`.
///
/// Analogous to `numpy.fromregex`. NumPy's structured-dtype support is not
/// modeled here — every capture group must parse to the same `T`; for mixed
/// dtypes use one call per column or the structured-record API in
/// `ferray-core::record`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if the regex cannot be compiled or contains
///   no capture groups.
pub fn fromregex<T>(content: &str, regex: &str) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + FromStr,
    T::Err: Display,
{
    let re = regex::Regex::new(regex)
        .map_err(|e| FerrayError::invalid_value(format!("fromregex: invalid regex: {e}")))?;
    let n_groups = re.captures_len().saturating_sub(1);
    if n_groups == 0 {
        return Err(FerrayError::invalid_value(
            "fromregex: regex must contain at least one capture group",
        ));
    }
    let mut data: Vec<T> = Vec::new();
    let mut nrows = 0usize;
    'lines: for line in content.lines() {
        if let Some(caps) = re.captures(line) {
            // Try to parse every capture group into T. If any fails, skip this row.
            let start = data.len();
            for g in 1..=n_groups {
                let m = caps.get(g).map_or("", |m| m.as_str());
                match m.parse::<T>() {
                    Ok(v) => data.push(v),
                    Err(_) => {
                        // Roll back this row's pushes, then continue with next line.
                        data.truncate(start);
                        continue 'lines;
                    }
                }
            }
            nrows += 1;
        }
    }
    Array::from_vec(Ix2::new([nrows, n_groups]), data)
}

/// Read regex-extracted rows from a file, parsing every capture group as `T`.
///
/// Convenience wrapper that reads `path` to a string and calls [`fromregex`].
///
/// # Errors
/// - `FerrayError::IoError` if the file cannot be read.
/// - Errors from [`fromregex`] (regex compile / no groups).
pub fn fromregex_from_file<T, P>(path: P, regex: &str) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + FromStr,
    T::Err: Display,
    P: AsRef<Path>,
{
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "fromregex: failed to read file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    fromregex::<T>(&content, regex)
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Roundtrip tests assert exact equality on hand-picked text values.
mod tests {
    use super::*;

    // ---- 1-D variants (#494) -------------------------------------------

    #[test]
    fn savetxt_1d_writes_one_value_per_line() {
        let arr =
            Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.5, 2.5, 3.0, 4.0]).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        let opts = SaveTxtOptions::default();
        savetxt_1d_to_writer(&mut buf, &arr, &opts).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "1.5\n2.5\n3\n4\n");
    }

    #[test]
    fn savetxt_1d_then_loadtxt_1d_roundtrip() {
        let arr =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, -2.5, 3.5, 0.0, 7.25]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("vec.txt");
        let opts = SaveTxtOptions::default();
        savetxt_1d(&p, &arr, &opts).unwrap();
        let back: Array<f64, Ix1> = loadtxt_1d(&p, ',', 0).unwrap();
        assert_eq!(back.shape(), &[5]);
        assert_eq!(back.as_slice().unwrap(), arr.as_slice().unwrap());
    }

    #[test]
    fn loadtxt_1d_flattens_multicolumn_input() {
        // Two-column file: loadtxt_1d should flatten in row-major order.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("rect.txt");
        std::fs::write(&p, "1,2\n3,4\n5,6\n").unwrap();
        let v: Array<i64, Ix1> = loadtxt_1d(&p, ',', 0).unwrap();
        assert_eq!(v.as_slice().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

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

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.csv");

        savetxt(&path, &arr, &SaveTxtOptions::default()).unwrap();
        let loaded: Array<f64, Ix2> = loadtxt(&path, ',', 0).unwrap();

        assert_eq!(loaded.shape(), &[2, 3]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
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
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("genfromtxt_test.csv");
        std::fs::write(&path, content).unwrap();

        let arr = genfromtxt(&path, ',', f64::NAN, 0, &[]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert!(arr.as_slice().unwrap()[2].is_nan());
    }

    #[test]
    fn loadtxt_empty() {
        let content = "";
        let arr: Array<f64, Ix2> = loadtxt_from_str(content, ',', 0).unwrap();
        assert_eq!(arr.shape(), &[0, 0]);
    }

    // -- fromregex --

    #[test]
    fn fromregex_basic_one_group() {
        // Pull integers out of "value=NN" lines, ignore other lines.
        let s = "value=10\nvalue=20\nirrelevant\nvalue=30\n";
        let arr: Array<i32, Ix2> = fromregex(s, r"^value=(\d+)$").unwrap();
        assert_eq!(arr.shape(), &[3, 1]);
        assert_eq!(arr.as_slice().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn fromregex_multiple_groups() {
        // Two captures per row → shape (n, 2).
        let s = "1,2\n3,4\n5,6\n";
        let arr: Array<f64, Ix2> = fromregex(s, r"^([\d.]+),([\d.]+)$").unwrap();
        assert_eq!(arr.shape(), &[3, 2]);
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn fromregex_no_groups_errs() {
        let r: FerrayResult<Array<i32, Ix2>> = fromregex("a\nb\n", r"^[ab]$");
        assert!(r.is_err());
    }

    #[test]
    fn fromregex_invalid_regex_errs() {
        let r: FerrayResult<Array<i32, Ix2>> = fromregex("", r"(unclosed");
        assert!(r.is_err());
    }

    #[test]
    fn fromregex_skips_unparseable_rows() {
        // Second row has a non-numeric capture; it should be skipped silently.
        let s = "v=10\nv=foo\nv=20\n";
        let arr: Array<i32, Ix2> = fromregex(s, r"^v=(\S+)$").unwrap();
        assert_eq!(arr.shape(), &[2, 1]);
        assert_eq!(arr.as_slice().unwrap(), &[10, 20]);
    }

    #[test]
    fn fromregex_from_file_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("regex_test.txt");
        std::fs::write(&path, "x=1\nx=2\nx=3\n").unwrap();
        let arr: Array<i32, Ix2> = fromregex_from_file(&path, r"^x=(\d+)$").unwrap();
        assert_eq!(arr.shape(), &[3, 1]);
        assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3]);
    }
}
