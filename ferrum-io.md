---
title: ferrum-io — Array serialization, text I/O, and memory mapping
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Implements NumPy-compatible binary (.npy/.npz) and text (CSV/delimited) I/O for ferrum arrays. Supports typed loading (`io::load::<f64>()`), dynamic loading (`io::load_dynamic()` returning `DynArray`), compressed archives, and memory-mapped files for large arrays. This is essential infrastructure — without it, ferrum arrays cannot persist or interoperate with Python workflows.

### Requirements

- REQ-1: `io::save(path, &array)` writes a single array in NumPy .npy format (magic bytes, version, header with dtype/shape/fortran_order, raw data)
- REQ-2: `io::load::<T, D>(path)` reads a .npy file and returns `Result<NdArray<T, D>, FerrumError>`. Returns `Err(InvalidDtype)` if the file's dtype does not match `T`.
- REQ-3: `io::load_dynamic(path)` reads a .npy file and returns `Result<DynArray, FerrumError>` using runtime dtype dispatch
- REQ-4: `io::savez(path, &[("name", &array), ...])` writes multiple arrays to a .npz (zip) archive
- REQ-5: `io::savez_compressed(path, ...)` writes a gzip-compressed .npz archive
- REQ-6: Support NumPy format versions 1.0, 2.0, and 3.0. Parse headers with structured dtype descriptions. ### Text I/O
- REQ-7: `io::savetxt(path, &array, delimiter, fmt)` writes a 2D array as delimited text
- REQ-8: `io::loadtxt::<T>(path, delimiter, skiprows)` reads delimited text into a 2D array
- REQ-9: `io::genfromtxt(path, delimiter, filling_values)` reads text with missing value handling (fills missing with the specified value, defaulting to NaN) ### Memory Mapping
- REQ-10: `io::memmap::<T>(path, mode)` returns a memory-mapped array view with `MemmapMode::ReadOnly`, `ReadWrite`, or `CopyOnWrite`
- REQ-11: Memory-mapped arrays must implement `ArrayView` semantics — they are views into the file's memory, not owned copies ### Endianness
- REQ-12: Support reading and writing both little-endian and big-endian .npy files. Default to system-native endianness for writes.

### Acceptance Criteria

- [ ] AC-1: Round-trip: `save` then `load` produces a bit-identical array for all supported dtypes (f32, f64, i32, i64, u8, bool, Complex64, Complex128)
- [ ] AC-2: Files written by ferrum can be read by `np.load()` in Python, and vice versa
- [ ] AC-3: `savez` / `savez_compressed` produce valid .npz archives readable by NumPy
- [ ] AC-4: `load_dynamic` correctly dispatches on dtype string in .npy header
- [ ] AC-5: `loadtxt` correctly parses CSV with header skiprows, custom delimiter, and numeric data
- [ ] AC-6: `genfromtxt` fills missing values with NaN (or specified fill value)
- [ ] AC-7: Memory-mapped array modifications in `ReadWrite` mode persist to disk
- [ ] AC-8: `cargo test -p ferrum-io` passes. `cargo clippy -p ferrum-io -- -D warnings` clean.

### Architecture

### Crate Layout
```
ferrum-io/
  Cargo.toml
  src/
    lib.rs
    npy/
      mod.rs                  # save, load, load_dynamic
      header.rs               # .npy header parsing (magic, version, dtype, shape, order)
      dtype_parse.rs          # NumPy dtype string parsing ("<f8", "|b1", etc.)
    npz/
      mod.rs                  # savez, savez_compressed, load from .npz
    text/
      mod.rs                  # savetxt, loadtxt, genfromtxt
      parser.rs               # Delimited text parser with missing value handling
    memmap.rs                 # Memory-mapped array via memmap2
    format.rs                 # MemmapMode enum, format version constants
```

### NumPy Format Compatibility
The .npy format header is a Python dict literal (e.g., `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4)}`). The parser must handle this format exactly, including structured dtypes. The format spec is documented at numpy.org/neps/nep-0001-npy-format.html.

### Out of Scope

- Arrow/Polars interop (Phase 3, implemented in ferrum-core or a separate interop layer)
- PyO3/NumPy interop (ferrum-numpy-interop)
- Serde serialization of arrays (feature-gated in ferrum-core)

### dependencies

- **Upstream**: `ferrum-core` (NdArray, DynArray, DType, Element, FerrumError)
- **Downstream**: ferrum (re-export)
- **External crates**: `byteorder` (endian-aware binary I/O), `flate2` (gzip for .npz compressed), `memmap2` (memory-mapped I/O), `zip` (.npz archive handling)
- **Phase**: 1 — Core Array and Ufuncs

