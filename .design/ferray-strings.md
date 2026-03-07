# Feature: ferray-strings — Vectorized string operations on arrays

## Summary
Implements `numpy.strings` (NumPy 2.0+): vectorized elementwise string operations on arrays of strings with broadcasting. Covers case manipulation, alignment/padding, stripping, find/replace, splitting/joining, and regex support. Operates on `StringArray` — a separate array type backed by `Vec<String>`.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Array1, broadcasting, FerrayError)
- **Downstream**: ferray (re-export)
- **External crates**: `regex` (regex operations)
- **Phase**: 3 — Completeness

## Requirements

### StringArray Type
- REQ-1: Define `StringArray<D>` as a specialized array type backed by `Vec<String>` (not generic `NdArray<String, D>` — strings are not `Element`). Provide `StringArray1`, `StringArray2` aliases.
- REQ-2: `strings::array(["hello", "world"])` constructor from string slices

### String Operations (Section 12)
- REQ-3: Concatenation: `strings::add(&a, &b)` — elementwise concatenation with broadcasting
- REQ-4: Repetition: `strings::multiply(&a, n)` — repeat each string n times
- REQ-5: Case: `upper`, `lower`, `capitalize`, `title`
- REQ-6: Alignment: `center(width, fillchar)`, `ljust(width)`, `rjust(width)`, `zfill(width)`
- REQ-7: Stripping: `strip(chars)`, `lstrip(chars)`, `rstrip(chars)`
- REQ-8: Replace: `replace(old, new, count)`
- REQ-9: Search predicates: `startswith(prefix)` → `Array<bool>`, `endswith(suffix)` → `Array<bool>`
- REQ-10: Search indices: `find(sub)` → `Array<i64>` (-1 if not found), `count(sub)` → `Array<usize>`
- REQ-11: Split/join: `split(sep)` → `Array1<Vec<String>>`, `join(sep, &a)` → `Array1<String>`

### Regex Support
- REQ-12: `strings::match_(&a, pattern)` → `Array<bool>` (whether each element matches)
- REQ-13: `strings::extract(&a, pattern)` → `StringArray` (first capture group from each element)

## Acceptance Criteria
- [ ] AC-1: `strings::upper(&["hello", "world"])` produces `["HELLO", "WORLD"]`
- [ ] AC-2: `strings::add` broadcasts a scalar string against an array correctly
- [ ] AC-3: `strings::find(&a, "ll")` returns correct indices (2 for "hello", -1 for "world")
- [ ] AC-4: `strings::split(&["a-b", "c-d"], "-")` returns `[vec!["a","b"], vec!["c","d"]]`
- [ ] AC-5: Regex `match_` and `extract` work correctly with capture groups
- [ ] AC-6: `cargo test -p ferray-strings` passes. `cargo clippy` clean.

## Architecture

### Crate Layout
```
ferray-strings/
  Cargo.toml
  src/
    lib.rs
    string_array.rs           # StringArray<D> type definition
    case.rs                   # upper, lower, capitalize, title
    align.rs                  # center, ljust, rjust, zfill
    strip.rs                  # strip, lstrip, rstrip
    search.rs                 # find, count, startswith, endswith, replace
    split_join.rs             # split, join
    concat.rs                 # add, multiply
    regex_ops.rs              # match_, extract
```

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Unicode normalization (use the `unicode-normalization` crate directly)
- Full pandas-style string accessor
