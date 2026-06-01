# ferray-stride-tricks

Zero-copy stride-manipulation views for the ferray scientific computing workspace — a Rust-native, drop-in NumPy replacement.

Part of the [ferray](../README.md) workspace.

## Overview

This crate implements `numpy.lib.stride_tricks`: it constructs new array
views over existing buffers by synthesizing shapes and strides, without
allocating or copying element data. Views share the base pointer of their
source; stride-0 axes broadcast a single element across a whole dimension.

Public functions (re-exported at the crate root):

- `sliding_window_view` — overlapping window views over all axes.
- `sliding_window_view_axis` — windowing along a single axis.
- `broadcast_to` — broadcast one array to a target shape (NumPy rules).
- `broadcast_arrays` — mutually broadcast a set of arrays to a common shape.
- `broadcast_shapes` — compute the broadcast shape without touching data.
- `as_strided` — arbitrary shape/strides, with a non-overlap safety check.
- `as_strided_signed` — `as_strided` accepting signed (negative) strides.
- `as_strided_unchecked` / `as_strided_signed_unchecked` — `unsafe` variants
  that skip the overlap check.

The `StridedSource` trait abstracts over the array inputs `as_strided` accepts.

## NumPy correspondence

| ferray | NumPy |
|--------|-------|
| `sliding_window_view` | `np.lib.stride_tricks.sliding_window_view` |
| `as_strided` / `as_strided_unchecked` | `np.lib.stride_tricks.as_strided` |
| `broadcast_to` | `np.broadcast_to` |
| `broadcast_arrays` | `np.broadcast_arrays` |
| `broadcast_shapes` | `np.broadcast_shapes` |

## Safety

`as_strided` validates that the requested shape/strides produce no
overlapping or aliasing element accesses before returning a view; an
overlapping request (e.g. strides `[1, 1]` over a 1-D buffer) returns
`Err`. The `as_strided_unchecked` and `as_strided_signed_unchecked`
variants are `unsafe fn`: they skip the overlap check, so the caller must
guarantee every computed offset stays in bounds of the source buffer.
Stride-0 (broadcasting) views alias the same element many times by design,
so they are only reachable through the unchecked path. As with NumPy's
`as_strided`, a malformed call can fabricate out-of-bounds reads.

## Feature flags

This crate defines no Cargo features. It depends only on `ferray-core` and
builds on workspace defaults (no extra runtime dependencies).

## Example

```rust
use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_stride_tricks::sliding_window_view;

let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5])?;

// Window length 3 -> a (3, 3) view: [[1,2,3], [2,3,4], [3,4,5]]
let windows = sliding_window_view(&a, &[3])?;
assert_eq!(windows.shape(), &[3, 3]);

// Zero-copy: the view shares the source's base pointer.
assert_eq!(windows.as_ptr(), a.as_ptr());
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Edition: 2024
- MSRV: Rust 1.88
- License: MIT OR Apache-2.0
