// ferray-strings: numpy.strings extras
//
// Functions added to round out NumPy 2.x parity that don't fit neatly
// into the existing search/case/classify/strip/etc. modules:
//   - encode/decode (UTF-8 codec ufuncs)
//   - expandtabs
//   - mod_ (printf-style % formatting)
//   - partition / rpartition
//   - slice (string-slicing ufunc)
//   - translate (char-translate ufunc)

use std::collections::HashMap;

use ferray_core::dimension::Dimension;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::string_array::StringArray;

// ===========================================================================
// encode / decode (UTF-8 codec ufuncs)
// ===========================================================================

/// Encode each string as UTF-8 bytes, returning a `Vec<Vec<u8>>` per element.
///
/// Equivalent to `numpy.strings.encode(arr, encoding="utf-8")` for the
/// common UTF-8 case. Other encodings are intentionally not supported —
/// the workspace standardizes on UTF-8 strings, and supporting Latin-1 /
/// CP1252 / Shift-JIS would pull in `encoding_rs`.
///
/// Errors if `encoding` is not `"utf-8"` (case-insensitive) and not empty.
///
/// # Errors
/// - `FerrayError::InvalidValue` if an unsupported encoding is requested.
pub fn encode<D: Dimension>(a: &StringArray<D>, encoding: &str) -> FerrayResult<Vec<Vec<u8>>> {
    let enc = encoding.trim().to_ascii_lowercase();
    if !enc.is_empty() && enc != "utf-8" && enc != "utf8" {
        return Err(FerrayError::invalid_value(format!(
            "encode: only UTF-8 is supported, got {encoding:?}"
        )));
    }
    Ok(a.iter().map(|s| s.as_bytes().to_vec()).collect())
}

/// Decode UTF-8 byte buffers back into a [`StringArray`] preserving the
/// input shape.
///
/// Equivalent to `numpy.strings.decode(arr, encoding="utf-8")` for the
/// UTF-8 case. The number of byte-buffers must equal the array's element
/// count. Invalid UTF-8 produces `FerrayError::InvalidValue`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `byte_arrays.len()` != `shape_len()`.
/// - `FerrayError::InvalidValue` for unsupported encoding or invalid UTF-8.
pub fn decode<D: Dimension>(
    byte_arrays: &[Vec<u8>],
    shape: D,
    encoding: &str,
) -> FerrayResult<StringArray<D>> {
    let enc = encoding.trim().to_ascii_lowercase();
    if !enc.is_empty() && enc != "utf-8" && enc != "utf8" {
        return Err(FerrayError::invalid_value(format!(
            "decode: only UTF-8 is supported, got {encoding:?}"
        )));
    }
    let expected: usize = shape.as_slice().iter().product();
    if byte_arrays.len() != expected {
        return Err(FerrayError::shape_mismatch(format!(
            "decode: got {} byte buffers, but shape {:?} requires {}",
            byte_arrays.len(),
            shape.as_slice(),
            expected,
        )));
    }
    let mut data = Vec::with_capacity(byte_arrays.len());
    for (i, bytes) in byte_arrays.iter().enumerate() {
        match std::str::from_utf8(bytes) {
            Ok(s) => data.push(s.to_owned()),
            Err(e) => {
                return Err(FerrayError::invalid_value(format!(
                    "decode: element {i} is not valid UTF-8: {e}"
                )));
            }
        }
    }
    StringArray::from_vec(shape, data)
}

// ===========================================================================
// expandtabs
// ===========================================================================

/// Replace tab characters in each element with spaces, expanding to the
/// given tab size on column boundaries.
///
/// Equivalent to `numpy.strings.expandtabs(arr, tabsize)`.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn expandtabs<D: Dimension>(
    a: &StringArray<D>,
    tabsize: usize,
) -> FerrayResult<StringArray<D>> {
    a.map(|s| {
        let mut out = String::with_capacity(s.len());
        let mut col = 0usize;
        for c in s.chars() {
            match c {
                '\t' => {
                    let pad = if tabsize == 0 {
                        0
                    } else {
                        tabsize - (col % tabsize)
                    };
                    for _ in 0..pad {
                        out.push(' ');
                    }
                    col += pad;
                }
                '\n' | '\r' => {
                    out.push(c);
                    col = 0;
                }
                other => {
                    out.push(other);
                    col += 1;
                }
            }
        }
        out
    })
}

// ===========================================================================
// mod_ (printf-style % formatting)
// ===========================================================================

/// Format each element as a printf-style template, substituting `args` in.
///
/// Equivalent to `numpy.strings.mod(arr, args)` (the `%`-style binding).
/// Supports the most common conversions:
///   - `%s` — verbatim insertion of the next argument
///   - `%%` — literal percent sign
///   - `%d` — integer (parses argument as i64)
///   - `%f` — float (parses argument as f64, default 6-decimal precision)
///   - `%.Nf` — float with explicit precision N
///   - `%e` / `%g` — scientific / shortest-of-fe representation
///
/// `args` is a slice indexed in conversion-order; each `%X` consumes the
/// next argument in `args`.
///
/// Width/flag specifiers beyond `%.Nf` are not supported — parsing the
/// full printf grammar inside a ufunc isn't worth the complexity given
/// most NumPy users reach for f-strings instead.
///
/// # Errors
/// - `FerrayError::InvalidValue` if `args.len()` is fewer than the number
///   of conversions, an unknown specifier appears, or a `%d`/`%f`/etc.
///   argument fails to parse.
pub fn mod_<D: Dimension>(a: &StringArray<D>, args: &[&str]) -> FerrayResult<StringArray<D>> {
    let strings: Vec<String> = a
        .iter()
        .map(|s| format_template(s, args))
        .collect::<FerrayResult<Vec<_>>>()?;
    StringArray::from_vec(a.dim().clone(), strings)
}

fn format_template(template: &str, args: &[&str]) -> FerrayResult<String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    let mut argi = 0usize;
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        // Parse conversion spec.
        let next = chars
            .next()
            .ok_or_else(|| FerrayError::invalid_value("mod: trailing '%' with no spec"))?;
        if next == '%' {
            out.push('%');
            continue;
        }
        // Optional `.N` precision.
        let mut precision: Option<usize> = None;
        let mut spec = next;
        if next == '.' {
            let mut digits = String::new();
            while let Some(&p) = chars.peek() {
                if p.is_ascii_digit() {
                    digits.push(p);
                    chars.next();
                } else {
                    break;
                }
            }
            precision = Some(
                digits
                    .parse::<usize>()
                    .map_err(|_| FerrayError::invalid_value("mod: bad precision after '%.'"))?,
            );
            spec = chars
                .next()
                .ok_or_else(|| FerrayError::invalid_value("mod: missing spec char after '%.N'"))?;
        }
        let arg = args.get(argi).ok_or_else(|| {
            FerrayError::invalid_value(format!("mod: not enough args at conversion {argi}"))
        })?;
        argi += 1;
        match spec {
            's' => out.push_str(arg),
            'd' | 'i' => {
                let v: i64 = arg.parse().map_err(|_| {
                    FerrayError::invalid_value(format!("mod: %d arg {arg:?} is not an integer"))
                })?;
                out.push_str(&v.to_string());
            }
            'f' => {
                let v: f64 = arg.parse().map_err(|_| {
                    FerrayError::invalid_value(format!("mod: %f arg {arg:?} is not a float"))
                })?;
                let p = precision.unwrap_or(6);
                out.push_str(&format!("{v:.p$}"));
            }
            'e' => {
                let v: f64 = arg.parse().map_err(|_| {
                    FerrayError::invalid_value(format!("mod: %e arg {arg:?} is not a float"))
                })?;
                let p = precision.unwrap_or(6);
                out.push_str(&format!("{v:.p$e}"));
            }
            'g' => {
                let v: f64 = arg.parse().map_err(|_| {
                    FerrayError::invalid_value(format!("mod: %g arg {arg:?} is not a float"))
                })?;
                // Pick the shorter of `{:.6}` and `{:.6e}`, mirroring printf %g's intent.
                let fixed = format!("{v}");
                let sci = format!("{v:e}");
                out.push_str(if fixed.len() <= sci.len() {
                    &fixed
                } else {
                    &sci
                });
            }
            other => {
                return Err(FerrayError::invalid_value(format!(
                    "mod: unsupported conversion '%{other}'"
                )));
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// partition / rpartition
// ===========================================================================

/// Split each element on the first occurrence of `sep`, returning a
/// `(before, sep, after)` triple per element.
///
/// If `sep` is not found in an element, the result is `(element, "", "")`.
/// Mirrors `numpy.strings.partition`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if `sep` is empty.
pub fn partition<D: Dimension>(
    a: &StringArray<D>,
    sep: &str,
) -> FerrayResult<Vec<(String, String, String)>> {
    if sep.is_empty() {
        return Err(FerrayError::invalid_value(
            "partition: separator must not be empty",
        ));
    }
    Ok(a.iter()
        .map(|s| match s.find(sep) {
            Some(i) => (
                s[..i].to_owned(),
                sep.to_owned(),
                s[i + sep.len()..].to_owned(),
            ),
            None => (s.clone(), String::new(), String::new()),
        })
        .collect())
}

/// Split each element on the last occurrence of `sep`, returning a
/// `(before, sep, after)` triple per element.
///
/// If `sep` is not found, the result is `("", "", element)` (matching
/// Python / NumPy `rpartition`).
///
/// # Errors
/// - `FerrayError::InvalidValue` if `sep` is empty.
pub fn rpartition<D: Dimension>(
    a: &StringArray<D>,
    sep: &str,
) -> FerrayResult<Vec<(String, String, String)>> {
    if sep.is_empty() {
        return Err(FerrayError::invalid_value(
            "rpartition: separator must not be empty",
        ));
    }
    Ok(a.iter()
        .map(|s| match s.rfind(sep) {
            Some(i) => (
                s[..i].to_owned(),
                sep.to_owned(),
                s[i + sep.len()..].to_owned(),
            ),
            None => (String::new(), String::new(), s.clone()),
        })
        .collect())
}

// ===========================================================================
// slice (string-slicing ufunc)
// ===========================================================================

/// Slice each element by character index — `s[start..stop]` with negative
/// indices counting from the end (Python-style). `None` for either bound
/// keeps the corresponding edge.
///
/// `step` is intentionally not supported here (numpy.strings.slice's
/// `step` parameter requires character-level reverse iteration that is
/// non-trivial under multi-byte UTF-8). Pass two slice calls if you need
/// `[start:stop:step]` behavior.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn slice<D: Dimension>(
    a: &StringArray<D>,
    start: Option<isize>,
    stop: Option<isize>,
) -> FerrayResult<StringArray<D>> {
    a.map(|s| slice_str(s, start, stop))
}

fn slice_str(s: &str, start: Option<isize>, stop: Option<isize>) -> String {
    let n = s.chars().count() as isize;
    let resolve = |i: isize| -> usize {
        let r = if i < 0 { n + i } else { i };
        r.clamp(0, n) as usize
    };
    let lo = resolve(start.unwrap_or(0));
    let hi = resolve(stop.unwrap_or(n));
    if hi <= lo {
        return String::new();
    }
    s.chars().skip(lo).take(hi - lo).collect()
}

// ===========================================================================
// translate
// ===========================================================================

/// Apply a per-character translation table to each element.
///
/// `table` maps each input char to:
///   - `Some(c)` — replacement character (single char)
///   - `None` — drop the character (delete from output)
///
/// Characters not present in `table` are passed through unchanged.
/// Mirrors `numpy.strings.translate(arr, table)`.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn translate<D: Dimension>(
    a: &StringArray<D>,
    table: &HashMap<char, Option<char>>,
) -> FerrayResult<StringArray<D>> {
    a.map(|s| {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match table.get(&c) {
                Some(Some(replacement)) => out.push(*replacement),
                Some(None) => {} // drop
                None => out.push(c),
            }
        }
        out
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn encode_decode_roundtrip() {
        let a = array(&["hello", "world", "café"]).unwrap();
        let bytes = encode(&a, "utf-8").unwrap();
        assert_eq!(bytes.len(), 3);
        let back = decode(&bytes, a.dim().clone(), "utf-8").unwrap();
        assert_eq!(back.as_slice(), a.as_slice());
    }

    #[test]
    fn encode_default_encoding_empty_string_is_utf8() {
        let a = array(&["x"]).unwrap();
        assert!(encode(&a, "").is_ok());
    }

    #[test]
    fn encode_unsupported_errs() {
        let a = array(&["x"]).unwrap();
        assert!(encode(&a, "latin-1").is_err());
    }

    #[test]
    fn decode_invalid_utf8_errs() {
        use ferray_core::dimension::Ix1;
        let bad = vec![vec![0xff_u8, 0xfe, 0xfd]];
        let r = decode(&bad, Ix1::new([1]), "utf-8");
        assert!(r.is_err());
    }

    #[test]
    fn expandtabs_basic() {
        let a = array(&["a\tb\tc"]).unwrap();
        let r = expandtabs(&a, 4).unwrap();
        // 'a' at col 0, tab pads to col 4 (3 spaces); 'b' at col 4, tab pads
        // to col 8 (3 spaces); 'c' at col 8.
        assert_eq!(r.as_slice(), &["a   b   c"]);
    }

    #[test]
    fn expandtabs_resets_on_newline() {
        let a = array(&["x\ty\n\tz"]).unwrap();
        let r = expandtabs(&a, 4).unwrap();
        // First line: x___y. Newline resets col=0; second tab pads to col 4
        // → 4 spaces, then z.
        assert_eq!(r.as_slice(), &["x   y\n    z"]);
    }

    #[test]
    fn mod_simple_s() {
        let a = array(&["hello %s"]).unwrap();
        let r = mod_(&a, &["world"]).unwrap();
        assert_eq!(r.as_slice(), &["hello world"]);
    }

    #[test]
    fn mod_d_and_f() {
        let a = array(&["%d items @ %.2f each"]).unwrap();
        let r = mod_(&a, &["7", "2.5"]).unwrap();
        assert_eq!(r.as_slice(), &["7 items @ 2.50 each"]);
    }

    #[test]
    fn mod_double_percent_literal() {
        let a = array(&["100%% pure %s"]).unwrap();
        let r = mod_(&a, &["rust"]).unwrap();
        assert_eq!(r.as_slice(), &["100% pure rust"]);
    }

    #[test]
    fn mod_too_few_args_errs() {
        let a = array(&["%s and %s"]).unwrap();
        assert!(mod_(&a, &["only one"]).is_err());
    }

    #[test]
    fn partition_found() {
        let a = array(&["a-b-c", "no-sep-here"]).unwrap();
        let r = partition(&a, "-").unwrap();
        assert_eq!(
            r,
            vec![
                ("a".to_owned(), "-".to_owned(), "b-c".to_owned()),
                ("no".to_owned(), "-".to_owned(), "sep-here".to_owned()),
            ]
        );
    }

    #[test]
    fn partition_not_found() {
        let a = array(&["abc"]).unwrap();
        let r = partition(&a, "-").unwrap();
        assert_eq!(r, vec![("abc".to_owned(), String::new(), String::new())]);
    }

    #[test]
    fn rpartition_found_picks_last() {
        let a = array(&["a-b-c"]).unwrap();
        let r = rpartition(&a, "-").unwrap();
        assert_eq!(r, vec![("a-b".to_owned(), "-".to_owned(), "c".to_owned())]);
    }

    #[test]
    fn rpartition_not_found() {
        let a = array(&["abc"]).unwrap();
        let r = rpartition(&a, "-").unwrap();
        assert_eq!(r, vec![(String::new(), String::new(), "abc".to_owned())]);
    }

    #[test]
    fn partition_empty_sep_errs() {
        let a = array(&["x"]).unwrap();
        assert!(partition(&a, "").is_err());
        assert!(rpartition(&a, "").is_err());
    }

    #[test]
    fn slice_basic() {
        let a = array(&["hello", "world"]).unwrap();
        let r = slice(&a, Some(1), Some(4)).unwrap();
        assert_eq!(r.as_slice(), &["ell", "orl"]);
    }

    #[test]
    fn slice_negative_indices() {
        let a = array(&["abcdef"]).unwrap();
        let r = slice(&a, Some(-3), None).unwrap();
        assert_eq!(r.as_slice(), &["def"]);
        let r2 = slice(&a, None, Some(-2)).unwrap();
        assert_eq!(r2.as_slice(), &["abcd"]);
    }

    #[test]
    fn slice_unicode_char_aware() {
        let a = array(&["café"]).unwrap();
        let r = slice(&a, Some(2), Some(4)).unwrap();
        // 4 chars: c, a, f, é → [2..4] = "fé"
        assert_eq!(r.as_slice(), &["fé"]);
    }

    #[test]
    fn slice_empty_when_stop_le_start() {
        let a = array(&["hello"]).unwrap();
        let r = slice(&a, Some(3), Some(2)).unwrap();
        assert_eq!(r.as_slice(), &[""]);
    }

    #[test]
    fn translate_replace_and_drop() {
        let a = array(&["hello"]).unwrap();
        let mut table: HashMap<char, Option<char>> = HashMap::new();
        table.insert('l', Some('L'));
        table.insert('o', None); // drop
        let r = translate(&a, &table).unwrap();
        assert_eq!(r.as_slice(), &["heLL"]);
    }

    #[test]
    fn translate_passthrough_unmapped() {
        let a = array(&["abc"]).unwrap();
        let table: HashMap<char, Option<char>> = HashMap::new();
        let r = translate(&a, &table).unwrap();
        assert_eq!(r.as_slice(), &["abc"]);
    }
}
