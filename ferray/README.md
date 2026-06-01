# ferray

The umbrella crate — import the whole NumPy-equivalent surface from one dependency.

Part of the [ferray](../README.md) workspace — a Rust-native, drop-in NumPy replacement powered by SIMD.

## Overview

`ferray` is the top-level crate that re-exports every ferray sub-crate under a single
unified namespace, analogous to `import numpy as np` in Python. Most users should depend
on this crate rather than on the individual `ferray-core`, `ferray-ufunc`, `ferray-stats`,
etc. crates directly. The umbrella itself contains only re-exports plus a thin thread-pool
configuration layer, and forbids unsafe code.

At the top level it surfaces:

- **Core types** — `Array`, `ArrayView`, `ArrayViewMut`, `ArcArray`, `CowArray`,
  `DynArray`, dimension types (`Ix0`–`Ix6`, `IxDyn`, `Axis`), `DType` / `Element`,
  `FerrayError` / `FerrayResult`, the `s!` / `promoted_type!` macros, and `Complex`.
- **Constants** — `PI`, `E`, `INF`, `NAN`, `NEWAXIS`, `EULER_GAMMA`, …
- **Creation** — `zeros`, `ones`, `arange`, `linspace`, `eye`, `full`, `meshgrid`, …
- **Manipulation & indexing** — `reshape`, `concatenate`, `stack`, `transpose`, `take`,
  `nonzero`, `where_`, …
- **Ufuncs** (from `ferray-ufunc`) — `sin`, `cos`, `exp`, `sqrt`, `add`, `clip`, bitwise
  ops, comparisons, `convolve`, `interp`, …
- **Statistics** (from `ferray-stats`) — `sum`, `mean`, `var`, `std_`, `median`,
  `quantile`, `histogram`, `argsort`, `unique`, set operations, NaN-aware reducers, …
- **Thread-pool config** — `set_num_threads`, `with_num_threads`, and the
  `threshold::PARALLEL_THRESHOLD_*` constants.

Optional domain crates are feature-gated and exposed as named modules: `ferray::linalg`,
`ferray::fft`, `ferray::random`, `ferray::io`, `ferray::polynomial`, `ferray::window`,
`ferray::strings`, `ferray::ma`, `ferray::stride_tricks`, `ferray::autodiff`, and
`ferray::numpy_interop`. A few of their functions are also lifted to the top level to
match NumPy (`matmul`, `vecdot`, `matvec`, `vecmat`, `broadcast_arrays`, `broadcast_shapes`).

The [`prelude`](src/prelude.rs) re-exports the ~95%-case subset — array types and aliases
(`Array1`/`Array2`/`ArrayD`/…), dimensions, `DType`/`Element`, errors, the `s!` macro, the
common creation functions, the common math ufuncs, the core reductions, the bitwise/logical
traits, and `Complex` — so that `use ferray::prelude::*;` covers most code.

## Feature flags

| Feature | Pulls in / enables |
|---|---|
| `default` | `strings`, `ma`, `io`, `window` |
| `full` | `default` + `linalg`, `fft`, `random`, `polynomial`, `stride-tricks`, `autodiff`, `f16`, `serde` |
| `linalg` | `ferray-linalg` — decompositions, solvers, norms, `matmul`/`vecdot`/`matvec`/`vecmat` |
| `fft` | `ferray-fft` — FFT/IFFT, real FFTs (rustfft + realfft) |
| `random` | `ferray-random` — generators (PCG64DXSM/MT19937/SFC64) and distributions |
| `polynomial` | `ferray-polynomial` (power + 5 bases); implies `linalg` |
| `window` | `ferray-window` — window functions (Kaiser, Bartlett, …) |
| `strings` | `ferray-strings` — vectorized string-array operations |
| `ma` | `ferray-ma` — masked arrays |
| `io` | `ferray-io` — `.npy`/`.npz`/text/memmap I/O |
| `stride-tricks` | `ferray-stride-tricks` — `as_strided`, `sliding_window_view`, broadcast helpers |
| `autodiff` | `ferray-autodiff` — forward-mode automatic differentiation (`DualNumber<T>`) |
| `numpy` | `ferray-numpy-interop` — PyO3 bridge (exposed as `ferray::numpy_interop`) |
| `f16` | IEEE binary16 element type (forwards `f16` to `ferray-core`/`ferray-ufunc`/`ferray-io`) |
| `bf16` | bfloat16 element type (forwards `bf16` to `ferray-core`/`ferray-io`) |
| `serde` | Array/dtype serde impls via `ferray-core/serde` |
| `chrono` | `DateTime64 ↔ chrono::DateTime<Utc>` conversions via `ferray-core/chrono` |
| `const_shapes` | Compile-time const-generic shapes via `ferray-core/const_shapes` |

`no_std` is **not** supported by the umbrella crate (`ferray-ufunc` and `ferray-stats`
require `std`). For `no_std` environments, depend on `ferray-core` directly with its
`no_std` feature.

## Example

```rust
use ferray::prelude::*;

fn demo() -> FerrayResult<()> {
    // Create arrays through the unified surface.
    let a = zeros::<f64, Ix2>(Ix2::new([2, 3]))?;
    let xs = linspace::<f64>(0.0, 1.0, 100, /* include_endpoint = */ true)?;

    // Elementwise ufuncs and reductions, all at the top level.
    let s = sin(&xs)?;
    let m = mean(&s, /* axis = */ None)?;

    println!("shape={:?}, mean={m:?}", a.shape());
    Ok(())
}
```

With the `linalg`/`fft` features enabled, the domain modules are reachable as
`ferray::linalg::det(&m)?`, `ferray::fft::rfft(...)`, and so on.

## Relationship to the workspace

`ferray` is the recommended entry point for Rust users — one dependency, one prelude, the
full NumPy-equivalent surface. The individual `ferray-*` sub-crates can still be depended
on directly when you only need a slice of the API (for example `ferray-core` alone for the
array type, or `ferray-core` in `no_std` mode). The `ferray-python` crate builds on this
umbrella to expose ferray to CPython, where it is used as `import ferray as np`.

## MSRV & edition

- Rust edition **2024**, MSRV **1.88**.
- Licensed under **MIT OR Apache-2.0**.
