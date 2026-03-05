---
title: ferrum-stride-tricks — as_strided, sliding windows, and broadcast views
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Implements `numpy.lib.stride_tricks`: low-level view construction via custom strides for operations like windowed convolution, Toeplitz matrices, and overlapping tiles. Provides both safe (non-overlapping) and unsafe (overlapping) variants, plus safe convenience functions (`sliding_window_view`, `broadcast_to`, `broadcast_arrays`, `broadcast_shapes`).

### Requirements

- REQ-1: `stride_tricks::sliding_window_view(&a, window_shape)` — returns read-only views of sliding windows. Always safe (views are immutable, no aliasing concern).
- REQ-2: `stride_tricks::broadcast_to(&a, shape)` — zero-copy broadcast via stride manipulation. Always read-only.
- REQ-3: `ferrum::broadcast_arrays(&[&a, &b, ...])` — broadcast all arrays to a common shape
- REQ-4: `ferrum::broadcast_shapes(&[shape1, shape2, ...])` — compute the broadcast result shape without allocating arrays ### Unsafe Stride Tricks
- REQ-5: `stride_tricks::as_strided(&a, shape, strides)` — SAFE variant that validates non-overlapping strides at runtime, returning `Result<ArrayView, FerrumError>`
- REQ-6: `unsafe stride_tricks::as_strided_unchecked(&a, shape, strides)` — UNSAFE variant for overlapping strides. The safety contract: caller must ensure no concurrent mutation through any alias of the overlapping memory region. This is documented at the call site. ### Safety Documentation
- REQ-7: `as_strided_unchecked` must have a `# Safety` doc section explaining exactly what invariants the caller must uphold, with examples of both correct and incorrect usage

### Acceptance Criteria

- [ ] AC-1: `sliding_window_view(&[1,2,3,4,5], (3,))` returns views `[[1,2,3], [2,3,4], [3,4,5]]`
- [ ] AC-2: `broadcast_to(&ones(3), (4, 3))` returns a (4,3) view without allocation
- [ ] AC-3: `broadcast_shapes(&[(3,1), (1,4)])` returns `(3,4)`
- [ ] AC-4: `as_strided` with non-overlapping strides succeeds; overlapping strides returns `Err`
- [ ] AC-5: `as_strided_unchecked` compiles only in unsafe block
- [ ] AC-6: `cargo test -p ferrum-stride-tricks` passes. `cargo clippy` clean.

### Architecture

### Crate Layout
```
ferrum-stride-tricks/
  Cargo.toml
  src/
    lib.rs
    as_strided.rs             # Safe and unsafe as_strided variants
    sliding_window.rs         # sliding_window_view
    broadcast.rs              # broadcast_to, broadcast_arrays, broadcast_shapes
    overlap_check.rs          # Runtime overlap detection for safe as_strided
```

### Out of Scope

- Mutable sliding window views (fundamentally unsafe with overlapping windows)

### dependencies

- **Upstream**: `ferrum-core` (NdArray, ArrayView, Dimension, FerrumError)
- **Downstream**: ferrum (re-export)
- **Phase**: 3 — Completeness

