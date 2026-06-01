# ferray-core

N-dimensional array type and foundational primitives for ferray.

Part of the [ferray](../README.md) workspace — a Rust-native, drop-in NumPy replacement powered by SIMD.

## Overview

`ferray-core` is the foundation crate every other ferray crate builds on. It defines
the public array type and the type system around it; `ndarray` is used internally for
storage but is never exposed in the public API.

- **`Array<T, D>`** — the owned, heap-allocated N-dimensional array (`NdArray<T, D>`,
  analogous to `numpy.ndarray`), plus the borrowing variants `ArrayView`,
  `ArrayViewMut`, `ArcArray`, and `CowArray`.
- **Type aliases** — `Array1`, `Array2`, `Array3`, … and `ArrayD` (dynamic rank),
  with matching `ArrayView*` / `ArrayViewMut*` / `ArcArray*` / `CowArray*` families.
- **Dimensions** — the `Dimension` trait with `Ix0`–`Ix6`, `IxDyn`, and `Axis`.
- **`Element`** — the trait implemented by every supported scalar (f16/bf16, f32, f64,
  `Complex`, signed/unsigned integers, bool), gated by `DType`.
- **`DType`** — the runtime dtype enum plus NEP-50 promotion (`result_type`,
  `promote_types`) and casting (`can_cast`, `CastKind`), `finfo`/`iinfo` limits.
- **`FerrayError` / `FerrayResult`** — the workspace error hierarchy; library code
  never panics, every public fallible function returns `FerrayResult<T>`.
- **Array creation, manipulation, indexing, broadcasting, reductions, and `NdIter`.**

## NumPy correspondence

| ferray-core | NumPy |
|---|---|
| `Array<T, D>`, `Array1`/`Array2`/`ArrayD`, `ArrayView` | `numpy.ndarray` |
| `DType`, `result_type`, `promote_types`, `can_cast` | dtype, NEP-50 promotion, casting rules |
| `zeros`, `ones`, `arange`, `linspace`, `eye`, `full`, `meshgrid` | `np.zeros`, `np.arange`, `np.linspace`, `np.eye`, … |
| `reshape`, `concatenate`, `stack`, `transpose`, `split`, `flip` | `np.reshape`, `np.concatenate`, `np.stack`, `np.transpose`, … |
| `broadcast_to`, `expand_dims`, `squeeze` | broadcasting, `np.broadcast_to`, `np.expand_dims` |
| `.sum()`, `.mean()`, `.var()`, `.argmax()`, `.sum_axis(axis)` | `arr.sum()`, `arr.mean()`, `arr.var()`, axis reductions |
| `NdIter`, `s![]` | `np.nditer`, basic/advanced slicing |

## Feature flags

| Feature | Default | Description |
|---|---|---|
| `std` | yes | Enables the `ndarray`-backed `Array` type and all std-only modules (creation, manipulation, indexing, reductions, `NdIter`, …). Disable for `no_std`: only `DType`, `Element`, `FerrayError`, dimension types, constants, and layout enums remain. |
| `f16` | no | IEEE binary16 (`half::f16`) as a first-class `Element`. |
| `bf16` | no | bfloat16 (`half::bf16`) as a first-class `Element`. |
| `serde` | no | `serde` derive support for arrays and dtypes (pulls in `ndarray/serde`). |
| `const_shapes` | no | Compile-time const-generic shapes (`Shape1<N>`–`Shape6`, `StaticBroadcast`, `StaticMatMul`). |
| `chrono` | no | `DateTime64 ↔ chrono::DateTime<Utc>` bridge methods in `dtype::datetime`. |

## Example

```rust
use ferray_core::{Array1, Array2, Ix2, FerrayResult};

fn demo() -> FerrayResult<()> {
    // Creation: a 3×4 matrix of zeros and a 100-point linspace.
    let m = Array2::<f64>::zeros(Ix2::new([3, 4]))?;
    let xs = ferray_core::creation::linspace::<f64>(0.0, 1.0, 100, true)?;

    // Reductions are methods on the array.
    let total = m.sum();
    let mean = xs.mean();
    let var = xs.var(0);

    // Manipulation returns new arrays.
    let flat: Array1<f64> = ferray_core::manipulation::flatten(&m)?;

    println!("sum={total}, mean={mean:?}, var={var:?}, size={}", flat.size());
    Ok(())
}
```

Most users should depend on the umbrella [`ferray`](../README.md) crate, which
re-exports these types through its prelude.

## MSRV & edition

- Rust edition **2024**, MSRV **1.88**.
- Licensed under **MIT OR Apache-2.0**.
