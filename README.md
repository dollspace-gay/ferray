# ferray

A NumPy-equivalent scientific computing library for Rust. Correctly-rounded math, SIMD-accelerated operations, and zero panics.

## Why ferray?

- **More accurate than NumPy** on every transcendental function (CORE-MATH, < 0.5 ULP)
- **Faster than NumPy** on 23 of 55 benchmarks — all FFT sizes, all variance/std, small reductions
- **Memory safe** without garbage collection (17 Kani formal verification proof harnesses)
- **Zero panics** in library code — all public functions return `Result<T, FerrayError>`
- **Full NumPy API surface** — linalg, fft, random, polynomial, masked arrays, string arrays

## Quick Start

```toml
[dependencies]
ferray = "0.2"
```

```rust
use ferray::prelude::*;

// Create arrays
let a = ferray::linspace::<f64>(0.0, 1.0, 100, /* include_endpoint= */ true)?;
let b = ferray::sin(&a)?;

// Linear algebra
let m = ferray::eye::<f64>(3, 3, /* k (diagonal offset)= */ 0)?;
let det = ferray::linalg::det(&m)?;

// FFT
let spectrum = ferray::fft::rfft(
    &b,
    /* n= */ None,
    /* axis= */ None,
    /* norm= */ ferray::fft::FftNorm::Backward,
)?;

// Statistics
let mean = ferray::mean(&a, /* axis= */ None)?;
let std = ferray::std_(&a, /* axis= */ None, /* ddof= */ 0)?;
```

## Performance

Benchmarked against NumPy 2.3.5 on Linux (Rust 1.85, LTO, target-cpu=native).

### Where ferray dominates

| Operation | Speedup vs NumPy |
|-----------|-----------------|
| fft/64 | **17.0x faster** |
| var/1K | **20.8x faster** |
| std/1K | **15.8x faster** |
| mean/1K | **8.7x faster** |
| fft/1024 | **2.9x faster** |
| var/1M | **2.5x faster** |
| sum/1K | **1.9x faster** |
| fft/16384 | **1.8x faster** |
| fft/65536 | **1.6x faster** |
| arctan/100K | **1.5x faster** |

### Where NumPy wins

| Operation | Ratio | Reason |
|-----------|-------|--------|
| sin/cos/exp/log at scale | 1.4-2.1x | CORE-MATH correctly-rounded algorithms (deliberate accuracy tradeoff) |
| matmul 50x50-100x100 | 4.0-4.6x | OpenBLAS/MKL hand-tuned assembly vs faer pure Rust |
| sqrt 1M | 3.7x | Memory bandwidth bound at 8MB |

**Scorecard: ferray 23, NumPy 32.** All NumPy wins are transcendentals (accuracy tradeoff) or matmul (BLAS gap). GPU acceleration via CUDA is planned for Phase 6.

### Fast mode: `exp_fast`

For throughput-sensitive workloads, ferray offers `exp_fast()` — an Even/Odd Remez decomposition that is **~30% faster than CORE-MATH** while maintaining ≤1 ULP accuracy (faithfully rounded). It auto-vectorizes for SSE/AVX2/AVX-512/NEON with no lookup tables.

```rust
// Default: correctly rounded (≤0.5 ULP, CORE-MATH)
let result = ferray::exp(&array)?;

// Fast mode: faithfully rounded (≤1 ULP, ~30% faster)
let result = ferray::exp_fast(&array)?;
```

Both are more accurate than NumPy's libm-based `exp()` (which can be up to 8 ULP).

### Accuracy

ferray uses [CORE-MATH](https://core-math.gitlabpages.inria.fr/) — the only correctly-rounded math library in production. Every transcendental returns the closest representable floating-point value to the mathematical truth.

| | ferray | NumPy (glibc) |
|---|---|---|
| sin accuracy | < 0.5 ULP | up to 8,192 ULP at edge cases |
| exp accuracy | < 0.5 ULP | up to 8 ULP |
| Summation | Pairwise (O(epsilon log N)) | Pairwise |

## Crate Structure

ferray is a workspace of 15 focused crates:

| Crate | Description |
|-------|-------------|
| `ferray-core` | `NdArray<T, D>`, broadcasting, indexing, shape manipulation |
| `ferray-ufunc` | SIMD-accelerated universal functions (sin, cos, exp, sqrt, ...) |
| `ferray-stats` | Reductions, sorting, histograms, set operations |
| `ferray-linalg` | Matrix products, decompositions, solvers, einsum |
| `ferray-fft` | FFT/IFFT with plan caching, real FFTs |
| `ferray-random` | Generator API, 30+ distributions, permutations |
| `ferray-io` | NumPy .npy/.npz file I/O with memory mapping |
| `ferray-polynomial` | 6 basis classes, fitting, root-finding |
| `ferray-window` | Window functions, vectorize, piecewise |
| `ferray-strings` | StringArray with vectorized operations |
| `ferray-ma` | MaskedArray with mask propagation |
| `ferray-stride-tricks` | sliding_window_view, as_strided |
| `ferray-numpy-interop` | PyO3 zero-copy, Arrow/Polars conversion |
| `ferray-autodiff` | Forward-mode automatic differentiation |
| `ferray` | Re-export crate with prelude |

## Key Design Decisions

- **ndarray 0.17** for internal storage — NOT exposed in public API
- **pulp 0.22** for portable SIMD (SSE2/AVX2/AVX-512/NEON) on stable Rust
- **faer 0.24** for linear algebra, **rustfft 6.4** for FFT
- **CORE-MATH 1.0** for correctly-rounded transcendentals
- **Edition 2024**, MSRV 1.85
- All contiguous inner loops have SIMD paths for f32, f64, i32, i64

## Beyond NumPy

Features that go beyond NumPy's capabilities:

- **f16 support** — half-precision floats as first-class citizens across all crates
- **no_std core** — `ferray-core` and `ferray-ufunc` compile without `std` (requires `alloc`)
- **Const generic shapes** — `Shape1<N>` through `Shape6` for compile-time dimension checking
- **Automatic differentiation** — forward-mode autodiff via `DualNumber<T>`
- **Memory safety** — guaranteed by Rust's type system + Kani formal verification

## GPU Acceleration (Planned)

Phase 6 design complete (`.design/ferray-gpu.md`). Architecture:

- **CubeCL** for cross-platform GPU kernels (write once, compile to CUDA/Vulkan/Metal/WebGPU)
- **cudarc** for NVIDIA vendor libraries (cuBLAS 100x matmul, cuFFT, cuSOLVER)
- `GpuArray<T, D>` with explicit host-device transfers and async stream execution
- Expected 10-100x speedups for large arrays on GPU

## Building

```bash
cargo build --release
cargo test --workspace          # 1479 tests
cargo clippy --workspace -- -D warnings
```

## License

MIT OR Apache-2.0
