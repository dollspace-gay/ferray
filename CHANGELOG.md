# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-03-07

### Added

#### Phase 1: Core Array and Ufuncs
- **ferray-core**: `NdArray<T, D>` with full ownership model (owned, view, mutable view, ArcArray, CowArray)
- **ferray-core**: Broadcasting (NumPy rules), basic/advanced/extended indexing, `s![]` macro
- **ferray-core**: Array creation (`zeros`, `ones`, `arange`, `linspace`, `eye`, `meshgrid`, etc.)
- **ferray-core**: Shape manipulation (`reshape`, `transpose`, `concatenate`, `stack`, `split`, `pad`, `tile`, etc.)
- **ferray-core**: `Element` trait for 17 dtypes (f16, f32, f64, Complex, i8-i128, u8-u128, bool)
- **ferray-core**: `DType` runtime enum, `finfo`/`iinfo`, type promotion rules
- **ferray-core**: `FerrayError` hierarchy with diagnostic context, zero panics
- **ferray-core-macros**: `#[derive(FerrayRecord)]` proc macro, `promoted_type!` macro
- **ferray-ufunc**: 40+ SIMD-accelerated universal functions via `pulp` (sin, cos, exp, log, sqrt, etc.)
- **ferray-ufunc**: CORE-MATH correctly-rounded transcendentals (< 0.5 ULP from mathematical truth)
- **ferray-ufunc**: Binary ops (add, sub, mul, div, pow) with broadcasting
- **ferray-stats**: Reductions (sum, mean, var, std, min, max, argmin, argmax) with SIMD pairwise summation
- **ferray-stats**: Sorting (sort, argsort, partition), histograms, set operations (unique, intersect, union)
- **ferray-io**: NumPy `.npy`/`.npz` file I/O with memory mapping, structured dtype support
- **ferray**: Re-export crate with prelude module

#### Phase 2: Submodules
- **ferray-linalg**: Matrix products (dot, matmul, einsum, tensordot, kron, multi_dot)
- **ferray-linalg**: Decompositions (Cholesky, QR, SVD, LU, eig, eigh) via `faer`
- **ferray-linalg**: Solvers (solve, lstsq, inv, pinv, matrix_power, tensorsolve)
- **ferray-linalg**: Norms and measures (norm, cond, det, slogdet, matrix_rank, trace)
- **ferray-linalg**: Batched operations for 3D+ stacked arrays with Rayon parallelism
- **ferray-fft**: 1D/2D/ND FFT and IFFT via `rustfft` with plan caching
- **ferray-fft**: Real FFTs (rfft, irfft, rfft2, rfftn), Hermitian FFTs
- **ferray-fft**: Frequency utilities (fftfreq, rfftfreq, fftshift, ifftshift)
- **ferray-fft**: `FftNorm` (Backward, Forward, Ortho) matching NumPy
- **ferray-random**: Generator API with PCG64, Philox, SFC64, MT19937 BitGenerators
- **ferray-random**: 30+ distributions (Normal, Uniform, Poisson, Binomial, etc.)
- **ferray-random**: Permutations (shuffle, permutation, choice)
- **ferray-polynomial**: 6 basis classes (Power, Chebyshev, Legendre, Laguerre, Hermite, HermiteE)
- **ferray-polynomial**: Poly trait with fitting, root-finding, companion matrix eigenvalues
- **ferray-window**: Window functions (hann, hamming, blackman, kaiser, etc.)

#### Phase 3: Interop and Specialized
- **ferray-strings**: `StringArray` with vectorized operations (case, strip, search, split, regex)
- **ferray-ma**: `MaskedArray` with mask propagation, masked reductions and sorting
- **ferray-stride-tricks**: `sliding_window_view`, `as_strided`, overlap checking
- **ferray-numpy-interop**: PyO3 zero-copy NumPy conversion, Arrow and Polars interop

#### Phase 4: Beyond NumPy
- **f16 support**: Half-precision floats as first-class element type across all crates
- **no_std core**: `ferray-core` and `ferray-ufunc` compile with `#![no_std]` (requires `alloc`)
- **Const generic shapes**: `Shape1<N>` through `Shape6` with compile-time dimension checking
- **ferray-autodiff**: Forward-mode automatic differentiation via `DualNumber<T>`

#### Phase 5: Verification Infrastructure
- NumPy oracle fixture generation for cross-validation
- Property-based tests with `proptest` (256 cases per property)
- Fuzz targets for public function families
- SIMD vs scalar verification (all tests pass with `FERRAY_FORCE_SCALAR=1`)
- Kani formal verification harnesses for ferray-core
- Statistical equivalence benchmarks (47/47 accuracy tests pass)

#### Design
- **ferray-gpu design doc**: Phase 6 GPU acceleration architecture
  - CubeCL for cross-platform GPU kernels (CUDA/ROCm/Vulkan/Metal/WebGPU)
  - cudarc for NVIDIA vendor libraries (cuBLAS, cuFFT, cuSOLVER)
  - `GpuArray<T, D>` with stream-ordered async execution
  - Pinned memory transfers, 6 fused kernels, auto-dispatch

### Changed

#### Performance Optimizations
- **SIMD pairwise summation**: 4 accumulators to saturate FPU throughput, base case 256 elements
- **Fused SIMD variance**: `simd_sum_sq_diff_f64()` with FMA, no intermediate allocation (5x speedup)
- **FFT 1D fast path**: Skip lane extraction for 1D arrays (fft/64: 17x faster than NumPy)
- **FFT thread-local scratch**: `par_iter().map_init()` reuses buffers per Rayon thread
- **4-wide SIMD sqrt unroll**: Hides 12-cycle sqrt latency with instruction-level parallelism
- **Uninit output buffers**: `Vec::with_capacity` + `set_len` skips zeroing for ufunc outputs
- **SIMD square/reciprocal**: Real pulp SIMD kernels for common operations
- **matmul 3-tier dispatch**: Naive ikj (<=64), faer::Seq (65-255), faer::Rayon (>=256)
- **Rayon threshold**: 1M elements minimum to avoid thread pool overhead on small arrays
- **Batch benchmark mode**: Single-process benchmarking eliminates subprocess cold-cache bias
- **LTO + codegen-units=1**: 10-20% speedup across the board in release builds

#### Benchmark Results (vs NumPy 2.3.5)
- **ferray wins 23/55 benchmarks**, NumPy wins 32/55
- Dominates: FFT all sizes (1.6-17x), var/std all sizes (2.1-20.8x)
- Beats: mean/sum small (1.1-8.7x), arctan 10K+ (1.1-1.5x), sqrt/1K (1.1x), tanh/1K (1.3x)
- Slower: transcendentals at scale 1.4-2.1x (CORE-MATH accuracy tradeoff), matmul medium 4x (faer vs BLAS)

### Fixed
- Patched all design docs for adversarial review findings
- Fixed .gitignore: added target/, __pycache__/, *.pyc, .worktrees/, fuzz artifacts
- Purged 3,591 accidentally committed build artifacts from git history (163MB -> 1.3MB)
