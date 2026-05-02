# ferray

A NumPy-equivalent scientific computing library for Rust. Hand-tuned GEMM kernels, SIMD-accelerated everything, zero panics.

## Why ferray?

- **Faster than NumPy** on 40 of 55 benchmarks (73 %) — all FFT sizes, all variance/std sizes, all sin/cos/tan large sizes, all sum/mean medium sizes, plus matmul 50×50
- **Hand-tuned BLAS suite** — DGEMM/SGEMM/ZGEMM/CGEMM ports of OpenBLAS Haswell + Skylake-X kernels, plus i8/i16 quantized GEMM with AVX-VNNI / AVX-512 VNNI fast paths (1.5 TOPS at N=1024)
- **Memory safe** without garbage collection (17 Kani formal verification proof harnesses)
- **Zero panics** in library code — all public functions return `Result<T, FerrayError>`
- **Full NumPy API surface** — linalg, fft, random, polynomial, masked arrays, string arrays, complex transcendentals
- **More accurate than NumPy** for arctan/asin/sinh/cosh and family (CORE-MATH, < 0.5 ULP); sin/cos/tan match NumPy via libm at performance parity

## Quick Start

```toml
[dependencies]
ferray = "0.3"
```

```rust
use ferray::prelude::*;

// Create arrays
let a = ferray::linspace::<f64>(0.0, 1.0, 100, /* include_endpoint= */ true)?;
let b = ferray::sin(&a)?;

// Linear algebra (matmul now beats NumPy at 50x50!)
let m = ferray::eye::<f64>(3, 3, /* k (diagonal offset)= */ 0)?;
let det = ferray::linalg::det(&m)?;

// FFT (17.2x faster than NumPy at size 64)
let spectrum = ferray::fft::rfft(
    &b,
    /* n= */ None,
    /* axis= */ None,
    /* norm= */ ferray::fft::FftNorm::Backward,
)?;

// Statistics (28.8x faster than NumPy on var at N=1K)
let mean = ferray::mean(&a, /* axis= */ None)?;
let std = ferray::std_(&a, /* axis= */ None, /* ddof= */ 0)?;
```

## Performance

Benchmarked against NumPy 2.4.4 on Linux (Rust 1.95, LTO, target-cpu=native, Raptor Lake 14700K with 8-thread P-core pool). Full report in [BENCHMARKS.md](BENCHMARKS.md).

### Where ferray dominates

| Operation | Speedup vs NumPy |
|-----------|-----------------:|
| var/1K | **28.8× faster** |
| std/1K | **17.8× faster** |
| fft/64 | **17.2× faster** |
| mean/1K | **13.4× faster** |
| matmul/10×10 | **4.9× faster** |
| tan/1M | **4.8× faster** |
| arctan/1M | **4.3× faster** |
| fft/1024 | **4.3× faster** |
| sin/1M | **4.2× faster** |
| cos/1M | **4.2× faster** |
| std/1M | **3.1× faster** |
| var/1M | **3.4× faster** |
| tanh/1M | **2.8× faster** |
| fft/65536 | **2.2× faster** |
| sum/1K | **1.9× faster** |
| sum/1M | **1.1× faster** (DRAM-bandwidth ceiling) |
| matmul/50×50 | **1.05× faster** (was 1.33× slower in 0.2.x!) |

### Where NumPy still wins

| Operation | Ratio | Reason |
|-----------|------:|--------|
| sqrt 10K-1M | 1.2-1.9× | NumPy uses streaming-store hints |
| matmul 100×100 | 1.2× | faer-vs-MKL at the small-N parallel band |
| exp/log 1K-100K | 1.0-1.3× | Near parity; ferray pulls ahead at 1M (1.7-1.8× faster) |
| mean 1M | 1.3× | DRAM-bandwidth ceiling; sum at 1M is now 1.1× faster |

**Scorecard: ferray 40, NumPy 15** (was 23/32 in 0.2.x). All NumPy wins now sit within 1.0-1.9× — no major structural regressions remain.

### Hand-tuned GEMM throughput peaks (this host)

| Kernel | Peak GFLOPS | At N |
|---|---:|---:|
| DGEMM (f64) | 260 | 1024 |
| SGEMM (f32) | 987 | 1536 |
| ZGEMM (c64) | 456 | 1024 |
| CGEMM (c32) | 852 | 1024 |
| i8 quantized | 1488 GOPS | 1024 (AVX-VNNI) |

### Fast mode: `exp_fast`

For throughput-sensitive workloads, ferray offers `exp_fast()` — an Even/Odd Remez decomposition that is **~30% faster than CORE-MATH** while maintaining ≤1 ULP accuracy (faithfully rounded). It auto-vectorizes for SSE/AVX2/AVX-512/NEON with no lookup tables.

```rust
// Default: libm-equivalent (matches NumPy, ~1-2 ULP)
let result = ferray::exp(&array)?;

// Fast mode: faithfully rounded (≤1 ULP, ~30% faster)
let result = ferray::exp_fast(&array)?;
```

### Accuracy

ferray uses [CORE-MATH](https://core-math.gitlabpages.inria.fr/) for the family of transcendentals where it offers a clear speed-and-accuracy win (arctan, asin, sinh, cosh, asinh, atanh, expm1, log1p, …) — every result is the closest representable floating-point value to mathematical truth.

For sin/cos/tan, ferray routes through libm (Float::sin) for performance parity with NumPy; the correctly-rounded variants stay reachable via `cr_math::CrMath::cr_sin`/`cr_cos`/`cr_tan` for callers that prefer accuracy over speed.

| | ferray | NumPy (glibc) |
|---|---|---|
| arctan / asin / sinh accuracy | < 0.5 ULP (CORE-MATH) | up to thousands of ULP at edge cases |
| sin / cos / tan accuracy | ~1-2 ULP (libm, matches NumPy) | ~1-2 ULP (libm) |
| Summation | Pairwise (O(ε log N), base 4096) | Pairwise |

## Crate Structure

ferray is a workspace of 18 focused crates:

| Crate | Description |
|-------|-------------|
| `ferray-core` | `NdArray<T, D>`, broadcasting, indexing, shape manipulation |
| `ferray-core-macros` | proc-macro support (`#[derive(FerrayRecord)]`, `promoted_type!`) |
| `ferray-ufunc` | SIMD-accelerated universal functions (sin, cos, exp, sqrt, …) plus 20 complex transcendentals |
| `ferray-stats` | Reductions (pairwise SIMD sum at base 4096), sorting, histograms, set operations |
| `ferray-linalg` | Hand-tuned GEMM (DGEMM/SGEMM/ZGEMM/CGEMM/i8/i16/bf16/f16), TRSM, decompositions, solvers, einsum |
| `ferray-fft` | FFT/IFFT with plan caching, real FFTs |
| `ferray-random` | Generator API (PCG64DXSM/MT19937/SFC64), 30+ distributions, permutations |
| `ferray-io` | NumPy `.npy`/`.npz` file I/O with memory mapping, DataSource for URL/compressed inputs |
| `ferray-polynomial` | 6 basis classes, fitting, root-finding, Gauss quadrature |
| `ferray-window` | Window functions, vectorize, piecewise |
| `ferray-strings` | StringArray with vectorized operations |
| `ferray-ma` | MaskedArray with mask propagation, harden/soften, full reductions |
| `ferray-stride-tricks` | sliding_window_view, as_strided |
| `ferray-numpy-interop` | PyO3 zero-copy, Arrow/Polars conversion, complex/null/f16/bf16 support |
| `ferray-autodiff` | Forward-mode automatic differentiation |
| `ferray-test-oracle` | Test oracle harness for cross-validating against NumPy fixtures |
| `ferray` | Re-export crate with prelude |

## Cargo Features

| Feature | Crate | Description |
|---------|-------|-------------|
| `avx512` | ferray-linalg | AVX-512 GEMM kernels (DGEMM/SGEMM/ZGEMM/CGEMM/i16/VNNI). Bumps MSRV to 1.89. |
| `avxvnni` | ferray-linalg | AVX-VNNI i8 GEMM via `vpdpbusd`. Bumps MSRV to 1.87. |
| `bf16` | ferray-linalg | bf16 inputs → f32 accumulator GEMM (gemm_bf16_f32). |
| `f16` | ferray-linalg, ferray | f16 (IEEE binary16) inputs → f32 accumulator GEMM (gemm_f16_f32). |
| `openblas` | ferray-linalg | Optional system-OpenBLAS backend for f32/f64/c32/c64 GEMM. Provides a perf-floor guarantee. |

Default builds stay on workspace MSRV 1.85 with no extra runtime deps.

## Key Design Decisions

- **ndarray 0.17** for internal storage — NOT exposed in public API
- **pulp 0.22** for portable SIMD (SSE2/AVX2/AVX-512/NEON) on stable Rust
- **faer 0.24** for medium-N matmul (N < 256) and decompositions, hand-tuned kernels for N ≥ 256
- **rustfft 6.4** for FFT
- **CORE-MATH 1.0** for correctly-rounded arctan/asin/sinh family
- **Edition 2024**, MSRV 1.85 (default features)
- All contiguous inner loops have SIMD paths for f32, f64, i32, i64

## What's new in 0.3.5

The `[0.3.5]` release sweeps the open numpy-parity backlog to zero. Curated
highlights — see `CHANGELOG.md` for the full per-crate breakdown:

- **Statistics**: `unique_axis`, broadcast-aware `where`, scipy descriptive
  helpers (skew/kurtosis/zscore/mode/iqr/sem/gmean/hmean), seven rule-based
  histogram bin variants, five hypothesis tests (`pearsonr`, `spearmanr`,
  `ttest_1samp`, `ttest_ind`, `ks_2samp`, `chi2_contingency`).
- **Polynomial**: `from_roots` / `linspace` / `identity` / `basis` / `compose`
  / `Display` across all six bases; `ComplexPolynomial`; F32 sibling types
  for the five non-power bases.
- **FFT**: pre-allocated `_into` variants and `FftPlanND` for fixed-shape
  multi-dim transforms.
- **Random**: broadcast-aware distributions, full state serialization for
  every BitGenerator, typed integer generators (u8/i8/…/u64), N-D
  shuffle/permuted/choice, `multivariate_hypergeometric` and
  `multivariate_normal_array`.
- **Window**: `chebwin` (byte-exact scipy parity), `dpss` Slepian taper,
  closed-form `boxcar`/`triang`/`bohman`/`flattop`/`lanczos`, plus f32
  siblings of the numpy 5.
- **Strings**: `rsplit`, `splitlines`, N-D StringArray indexing,
  `CompactStringArray` Arrow-style backend behind `compact-storage`.
- **Stride tricks**: `sliding_window_view_axis`, `as_strided_signed`.
- **I/O**: 1-D `savetxt` / `loadtxt`, `Complex` round-trip in `.npy`,
  structured-dtype load/save into `FerrayRecord` arrays.
- **Core**: `DType::FixedAscii` / `FixedUnicode` / `RawBytes` for numpy's
  `S{n}` / `U{n}` / `V{n}` families.

Bug fixes: `correlate` was computing convolution; `corrcoef` now returns
NaN for zero-variance rows (matches numpy).

## Beyond NumPy

Features that go beyond NumPy's capabilities:

- **f16 / bf16 support** — half-precision floats as first-class citizens; mixed-precision GEMM
- **Hand-tuned i8 / i16 quantized GEMM** — ONNX `QLinearMatMul` formula end-to-end
- **Complex transcendentals** — `np.sin(complex_array)` style ufuncs (sin_complex, cos_complex, …, expm1_complex, log1p_complex with cancellation-avoiding paths near z=0)
- **Native TRSM** — block-recursive triangular solve for f32/f64 × lower/upper × non-unit/unit-diag
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
cargo test --workspace          # 1500+ tests
cargo clippy --workspace -- -D warnings

# Optional features:
cargo build --release --features ferray-linalg/avx512    # AVX-512 GEMM (Rust 1.89+)
cargo build --release --features ferray-linalg/openblas  # System OpenBLAS backend
cargo build --release --features ferray-linalg/bf16,ferray-linalg/f16  # Mixed-precision GEMM
```

## License

MIT OR Apache-2.0
