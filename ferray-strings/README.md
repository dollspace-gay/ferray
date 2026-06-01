# ferray-strings

Vectorized string operations on arrays of strings — a Rust port of `numpy.strings` / `numpy.char`.

Part of the [ferray](../README.md) workspace, a Rust-native, drop-in NumPy replacement.

## Overview

The primary type is `StringArray<D>`, a specialized N-dimensional string array
backed by `Vec<String>` (with `StringArray1` / `StringArray2` aliases).
Because `String` does not implement `ferray_core::Element`, it is a separate
type from `NdArray<T, D>`. Every operation returns `FerrayResult<T>` and
threads the input dimension `D` through to the output.

Operations are grouped into families, re-exported under a flat namespace
mirroring `numpy.strings.upper`, `numpy.strings.find`, …:

- **Construction**: `array`, plus `StringArray::from_vec` / `from_rows`.
- **Case** (`case`): `upper`, `lower`, `title`, `capitalize`; `swapcase` (in `str_ops`).
- **Alignment** (`align`): `center`, `ljust`, `ljust_with`, `rjust`, `rjust_with`, `zfill`.
- **Stripping** (`strip`): `strip`, `lstrip`, `rstrip`.
- **Search** (`search`): `find`, `rfind`, `index`, `rindex`, `count`, `startswith`, `endswith`, `replace`.
- **Split / join** (`split_join`): `split`, `rsplit`, `splitlines`, `split_ragged`, `join`, `join_array`.
- **Concat / repeat** (`concat`): `add`, `add_same`, `multiply`.
- **Regex** (`regex_ops`): `match_`, `match_compiled`, `extract`, `extract_compiled` (with `regex::Regex` re-exported).
- **Predicates** (`classify`): `isalpha`, `isdigit`, `isdecimal`, `isnumeric`, `isalnum`, `isspace`, `isupper`, `islower`, `istitle`.
- **Comparison** (`str_ops`): `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `str_len`.
- **Extras** (`extras`): `partition`, `rpartition`, `slice`, `translate`, `expandtabs`, `mod_`, `encode`, `decode`.

**Unicode correctness**: the `classify` predicates do not delegate to Rust's
`std` Unicode tables. Instead they binary-search sorted codepoint-range tables
derived live from the oracle CPython 3.13 / Unicode 15.1.0 build that backs the
verified `numpy` 2.4 reference — so `isdigit`, `isnumeric`, `isalpha`, … match
the NumPy oracle exactly rather than a newer Unicode revision (which would
classify ~9000 post-15.1.0 codepoints differently).

## NumPy correspondence

| ferray-strings | numpy.strings / numpy.char |
|----------------|----------------------------|
| `upper`, `lower`, `title`, `capitalize`, `swapcase` | `upper`, `lower`, `title`, `capitalize`, `swapcase` |
| `strip`, `lstrip`, `rstrip` | `strip`, `lstrip`, `rstrip` |
| `find`, `rfind`, `index`, `rindex`, `count` | `find`, `rfind`, `index`, `rindex`, `count` |
| `startswith`, `endswith`, `replace` | `startswith`, `endswith`, `replace` |
| `split`, `rsplit`, `splitlines`, `join`, `partition`, `rpartition` | `split`, `rsplit`, `splitlines`, `join`, `partition`, `rpartition` |
| `center`, `ljust`, `rjust`, `zfill` | `center`, `ljust`, `rjust`, `zfill` |
| `add`, `multiply` | `add`, `multiply` |
| `isalpha`, `isdigit`, `isdecimal`, `isnumeric`, `isalnum`, `isspace`, `isupper`, `islower`, `istitle` | `isalpha`, `isdigit`, `isdecimal`, `isnumeric`, `isalnum`, `isspace`, `isupper`, `islower`, `istitle` |
| `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` | `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` |
| `str_len`, `encode`, `decode`, `expandtabs`, `translate`, `mod_`, `slice` | `str_len`, `encode`, `decode`, `expandtabs`, `translate`, `mod`, `slice` |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `compact-storage` | off | Enables the `compact` module: an Arrow-style offsets+values backend (`CompactStringArray`, `CompactStringIter`, `estimated_string_array_bytes`) for memory-dense, cache-friendly string arrays. |

## Example

```rust
use ferray_strings::*;

// Build a 1-D StringArray and run a small pipeline.
let raw = array(&["  Hello  ", " World "])?;
let cleaned = strip(&raw, None)?;          // ["Hello", "World"]
let shouted = upper(&cleaned)?;            // ["HELLO", "WORLD"]
let suffix = array(&["!"])?;
let out = add(&shouted, &suffix)?;         // broadcasts: ["HELLO!", "WORLD!"]
assert_eq!(out.as_slice(), &["HELLO!", "WORLD!"]);

// Search returns NumPy-style i64 indices (-1 when absent).
let idx = find(&array(&["hello", "world"])?, "ll")?;
assert_eq!(idx.as_slice().unwrap(), &[2_i64, -1]);

// Regex match with capture-group extraction.
let a = array(&["abc123", "def", "ghi456"])?;
assert_eq!(match_(&a, r"\d+")?.as_slice().unwrap(), &[true, false, true]);
assert_eq!(extract(&a, r"(\d+)")?.as_slice(), &["123", "", "456"]);
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Edition 2024, MSRV 1.88.
- Licensed under MIT OR Apache-2.0.
