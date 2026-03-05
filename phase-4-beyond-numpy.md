---
title: Phase 4 — Beyond NumPy
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Extensions that go beyond NumPy's current capabilities, leveraging Rust's type system and performance characteristics: half-precision float (f16) support throughout, `no_std` core for embedded targets, compile-time dimensionality checking via const generics, and foundations for automatic differentiation. These are stretch goals that enhance ferrum's value proposition beyond "NumPy in Rust."

### Requirements

- REQ-1: Enable the `f16` feature flag to make `f16` (from the `half` crate) a first-class element type throughout all ferrum crates
- REQ-2: All ufunc operations must support `f16` inputs/outputs with appropriate SIMD paths (where hardware supports f16 natively) or fallback through f32 promotion
- REQ-3: f16 I/O: .npy files with `<f2` dtype must round-trip correctly
- REQ-4: f16 linalg operations: promote to f32 internally for numerical stability, return f16 results ### no_std Core
- REQ-5: `ferrum-core` and `ferrum-ufunc` must compile with `#![no_std]` when the `no_std` feature is enabled (requires `alloc` but not `std`)
- REQ-6: Features requiring std (I/O, threading, memmap) are gated behind `std` feature flag
- REQ-7: Core numeric operations (array creation, indexing, ufuncs, basic reductions) must work in no_std environments ### Const Generic Shapes
- REQ-8: Opt-in static shape checking via per-rank const generic types: `Shape1<const N: usize>`, `Shape2<const M: usize, const N: usize>`, through `Shape6`. All implement `Dimension`. This approach works on stable Rust (no `generic_const_exprs` required) and covers 99%+ of real NumPy usage (1D through 6D).
- REQ-9: Static shapes implement `From`/`Into` with dynamic shapes (`IxDyn`) for seamless interop — static-shaped arrays can be passed to any function expecting dynamic shapes
- REQ-10: Matrix multiplication with static shapes: `Array<T, Shape2<M, K>> * Array<T, Shape2<K, N>> -> Array<T, Shape2<M, N>>` — K mismatch is a compile error
- REQ-10a: Broadcasting with static shapes: `Shape2<M, 1> + Shape2<1, N>` resolves to `Shape2<M, N>` at compile time
- REQ-10b: Reshape with static shapes: `Array<T, Shape1<12>>.reshape::<Shape2<3, 4>>()` — compiler verifies 3*4 == 12 via const evaluation ### Automatic Differentiation Foundation
- REQ-11: Define a `DualNumber<T>` type supporting forward-mode autodiff via operator overloading
- REQ-12: All differentiable ufuncs (sin, cos, exp, log, etc.) must support `DualNumber<f64>` as an element type
- REQ-13: This is a foundation layer — a full autograd tape/reverse-mode system is post-1.0

### Acceptance Criteria

- [ ] AC-1: `Array1<f16>` can be created, indexed, and have ufuncs applied. Results match f32 computation downcast to f16.
- [ ] AC-2: `cargo build -p ferrum-core --no-default-features --features no_std` compiles for a bare-metal target (e.g., thumbv7em-none-eabihf)
- [ ] AC-3: `Array<f64, Shape2<3, 4>> * Array<f64, Shape2<4, 5>>` compiles and produces `Array<f64, Shape2<3, 5>>`; `Shape2<3, 4> * Shape2<5, 5>` does not compile (K=4 vs K=5 mismatch)
- [ ] AC-3a: `Array<f64, Shape2<3, 4>>` converts to `ArrayD<f64>` via `Into<IxDyn>` for interop with dynamic code
- [ ] AC-3b: `Array<f64, Shape1<12>>.reshape::<Shape2<3, 4>>()` compiles; `.reshape::<Shape2<3, 5>>()` does not (3*5 != 12)
- [ ] AC-4: Forward-mode autodiff: `d/dx(sin(x))` at x=0 returns `cos(0) = 1.0` via DualNumber
- [ ] AC-5: `cargo test --workspace --all-features` passes

### Architecture

These features are implemented as additions to existing crates, not new crates:
- f16: additions to `ferrum-core` (Element impl), `ferrum-ufunc` (kernels), `ferrum-io` (dtype)
- no_std: conditional compilation (`#[cfg(feature = "no_std")]`) throughout ferrum-core and ferrum-ufunc
- Const generics: new `shape` module in `ferrum-core` with `Shape1<const N>`, `Shape2<const M, const N>`, through `Shape6`, all implementing `Dimension`
- DualNumber: new `ferrum-autodiff` crate (not in the main workspace initially, added as an optional dependency)

### Out of Scope

- Full reverse-mode autograd with tape (separate project)
- GPU compute (separate crate, not part of ferrum core)
- Distributed/multi-node arrays

### dependencies

- **Upstream**: All Phase 1-3 crates must be complete and integrated
- **Phase**: 4 — Beyond NumPy (Month 13+)

