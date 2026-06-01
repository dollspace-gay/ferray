# ferray-io

NumPy-compatible array I/O — `.npy`, `.npz`, and delimited text — for the ferray workspace.

Part of the [ferray](../README.md) workspace.

## Overview

`ferray-io` reads and writes N-dimensional `ferray` arrays in NumPy's on-disk
formats. It does not expose any third-party array types: everything is built on
`ferray_core::Array` / `DynArray`.

- **`.npy`** — single-array binary format. `npy::save` / `npy::load` round-trip a
  typed `Array<T, D>`; `npy::load_dynamic` returns a `DynArray` when the dtype is
  not known at compile time; `npy::save_record` / `npy::load_record` handle
  structured (compound) dtypes via the `FerrayRecord` trait. The reader accepts
  NPY format versions **1.0, 2.0, and 3.0** (`format::VERSION_1_0` /
  `VERSION_2_0` / `VERSION_3_0`), covering both C- and Fortran-order layouts.
- **`.npz`** — zip archives of `.npy` members, backed by the `zip` crate.
  `npz::savez` writes a stored (uncompressed) archive and
  `npz::savez_compressed` writes a deflated one, each from a slice of
  `(name, &DynArray)` pairs. `npz::NpzFile::open` lazily reads an archive;
  `names`, `get`, `len`, and `is_empty` enumerate and extract its members.
- **Text I/O** — `text::savetxt` / `text::loadtxt` (2-D) and `savetxt_1d` /
  `loadtxt_1d` (1-D) handle delimited numeric text with configurable delimiter
  and `skiprows`; `text::genfromtxt` parses heterogeneous tables, and
  `text::fromregex` extracts fields by regular expression.
- **Memory mapping** — `memmap::memmap_readonly` / `memmap::memmap_mut` provide
  zero-copy, file-backed access to `.npy` data.

## NumPy correspondence

| NumPy                    | ferray-io                       |
|--------------------------|---------------------------------|
| `np.save`                | `npy::save`                     |
| `np.load` (.npy)         | `npy::load` / `npy::load_dynamic` |
| `np.savez`               | `npz::savez`                    |
| `np.savez_compressed`    | `npz::savez_compressed`         |
| `np.load` (.npz)         | `npz::NpzFile::open` + `get`    |
| `np.savetxt`             | `text::savetxt` / `savetxt_1d`  |
| `np.loadtxt`             | `text::loadtxt` / `loadtxt_1d`  |
| `np.genfromtxt`          | `text::genfromtxt`              |
| `np.fromregex`           | `text::fromregex`               |

## Feature flags

| Feature   | Effect                                                              |
|-----------|--------------------------------------------------------------------|
| (default) | No optional features enabled.                                      |
| `f16`     | IEEE binary16 dtype support (pulls in `half`, enables `ferray-core/f16`). |
| `bf16`    | bfloat16 dtype support (pulls in `half`, enables `ferray-core/bf16`).     |

## Example

```rust
use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_io::npy::{save, load};

// Save a 1-D f64 array to .npy ...
let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0])?;
save("data.npy", &a)?;

// ... and load it straight back.
let b: Array<f64, Ix1> = load("data.npy")?;
assert_eq!(a, b);
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Rust edition 2024, MSRV 1.88.
- Licensed under MIT OR Apache-2.0.
