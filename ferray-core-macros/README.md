# ferray-core-macros

Procedural-macro support crate for [`ferray-core`](../ferray-core).

Part of the [ferray](../README.md) workspace.

## Overview

This is a `proc-macro` crate (`proc-macro = true`). It exports the compile-time
machinery that `ferray-core` relies on for record types, slicing syntax, and
type promotion:

- `#[derive(FerrayRecord)]` — derives `unsafe impl ferray_core::record::FerrayRecord`
  for a `#[repr(C)]` struct with named fields. The generated `field_descriptors()`
  builds a static slice of `FieldDescriptor` (name, dtype, offset, size per field,
  via `Element::dtype()` / `offset_of!` / `size_of`), and `record_size()` returns
  `size_of::<Self>()`. Emits a compile error if the struct is not `#[repr(C)]`,
  is not a struct, or lacks named fields.
- `s![...]` — NumPy-style slice macro. Expands a comma-separated list of axis
  specifiers into a `Vec<ferray_core::dtype::SliceInfoElem>`. Supports integer
  indices (`s![3]`), full/partial ranges (`s![..]`, `s![2..]`, `s![..5]`,
  `s![1..5]`), and a `;step` suffix (`s![1..5;2]`).
- `promoted_type!(T1, T2)` — compile-time type promotion. Resolves two numeric
  types to the smallest common type following NumPy's promotion rules (e.g.
  `promoted_type!(i32, f32)` → `f64`, `promoted_type!(u8, i8)` → `i16`). Covers
  bool, the signed/unsigned integers, `f16`/`bf16`, `f32`/`f64`, and
  `Complex<f32>`/`Complex<f64>`. Emits a compile error when no lossless common
  type exists (e.g. `u128` + a signed integer).

## Usage

This is internal infrastructure, normally pulled in transitively by `ferray-core`;
you should not depend on it directly — use [`ferray`](../README.md) or
[`ferray-core`](../ferray-core) instead, which re-export the macros at their
public paths. A minimal `#[derive(FerrayRecord)]` example:

```rust
use ferray_core::record::FerrayRecord;

#[repr(C)]
#[derive(FerrayRecord)]
struct Point {
    x: f64,
    y: f64,
    id: i32,
}

let descriptors = Point::field_descriptors();
assert_eq!(descriptors.len(), 3);
assert_eq!(Point::record_size(), std::mem::size_of::<Point>());
```

## Feature flags

None. The crate has no `[features]` table.

## MSRV & edition

- Edition: 2024
- MSRV: 1.88
- License: MIT OR Apache-2.0
