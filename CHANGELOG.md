# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-03-07

### Added
- ferray-ufunc: no reduce/accumulate/outer/at/reduceat as generic ufunc methods (#381)
- ferray-ufunc: every operation allocates a new Vec — no out= parameter (#378)
- ferray-linalg: no batched solve/inv/det — only cholesky has batched variant (#412)
- ferray-random: standard_normal uses Box-Muller instead of Ziggurat — slower (#442)
- ferray-random: only f64 output — no f32 random generation (#441)
- ferray-fft: complex FFTs require Complex<f64> input — no auto-promotion from real arrays (#427)
- ferray-fft: complex FFTs only support Complex<f64> — no f32 FFT (#426)
- ferray-fft: rfft doesn't use rustfft's real-to-complex FFT — promotes to complex first (#432)
- ferray-ma: MaskedArray is not interoperable with ferray Array functions (#505)
- ferray-ma: no masked array broadcasting in arithmetic ops (#504)
- ferray-ma: no fill_value concept — masked elements get T::zero() in operations (#501)
- ferray-ma: reductions are whole-array only — no axis parameter (#500)
- ferray-polynomial: no domain/window mapping — polynomials operate on raw x values (#474)
- Add oracle test suite that validates all crates against NumPy fixture outputs (#45)
- ferray-random: all distributions return 1D arrays only — no shape parameter for ND output (#440)
- ferray-stats: no multi-axis reductions — axis only accepts single usize (#458)
- ferray-stats: no keepdims= parameter on any reduction (#457)
- ferray-core: no concatenate/stack/split in core (#362)
- ferray-core: no safe casting functions (can_cast, astype equivalent) (#361)
- ferray-core: no reduction methods (sum, prod, min, max, mean) on Array (#368)
- ferray-core: ops.rs operators don't broadcast — diverges from NumPy behavior (#346)
- Add operator overloading (+, -, *, /, %, unary -) for Array types (#43)
- Yank all ferrum crates from crates.io for rename to ferray (#38)
- Add exp_fast() with Even/Odd Remez v2 algorithm (#37)
- ferray-core: no nditer equivalent — missing general-purpose strided iterator (#340)
- ferray-linalg: faer_bridge always copies — no zero-copy path for C-contiguous arrays (#405)
- ferray-core: no np.where equivalent (ternary selection) (#372)
- ferray-core: no scalar-array arithmetic (array + scalar) (#347)
- ferray-ma: no std::ops overloads — must use masked_add/masked_sub functions (#502)
- ferray-stats: no dtype= parameter on sum/prod — integer overflow risk (#460)
- ferray-autodiff: no jacobian() function for vector-valued functions (#543)
- Add Index/IndexMut trait impls for Array with fixed-size array indices (#339)
- Add Serialize/Deserialize impls for Array<T, D> behind serde feature (#338)

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
- ferray-fft: plan cache is f64-only; no f32 FFT support (#109)
- ferray-fft: rfft computes full FFT then discards half (#108)
- ferray-ufunc oracle.rs: replace closure-suppression with macro type pinning (#561)
- ferray-ufunc: binary ops require same shape — broadcast variants are separate functions (#379)
- Prepare all crates for crates.io publishing with GitHub URLs (#35)
- ferray-core: NdIter — restore and complete BinaryBroadcastIter Iterator impl and binary_map_to (#560)

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
- ferray-test-oracle: no self-tests for the oracle framework (#204)
- ferray-test-oracle: cross-sign ULP distance calculation incorrect (#199)
- ferray-numpy-interop: zero-copy claims are false — every conversion copies (#195)
- ferray-numpy-interop: NpElement not implemented for bool (#196)
- ferray-numpy-interop: no NaN roundtrip test for Polars (#198)
- ferray-test-oracle: parse_complex_data panics on NaN/Inf string values (#202)
- ferray-ufunc: rayon parallelism is defined but never used in ops (#382)
- ferray-linalg: matmul/dot use naive O(n^3) loops — not using faer/BLAS for matrix multiply (#413)
- ferray-io: no f16/bf16 support in npy I/O even with feature enabled (#118)
- ferray-fft: nd.rs lane extraction copies each lane individually — O(n) allocations for ND FFT (#433)
- Fix bugs found by oracle test suite (#46)
- Fix exp_fast SIMD dispatch so it auto-vectorizes without target-cpu=native (#42)
- Fix Windows compilation: add signgam compat header for core-math lgamma.c (#41)
- ferray-linalg: tensorsolve and tensorinv have zero tests (#105)
- ferray-linalg: matmul_batched (ND x ND) has zero tests (#106)
- ferray-linalg: lstsq 2D b code path untested (#104)
- ferray-core-macros: no tests for negative indices in s! macro (#212)
- ferray-fft: all FFT functions flatten input via iter().copied().collect() before computing (#428)
- ferray-polynomial: oracle/property tests only cover power basis (#132)
- ferray-fft: no property tests for rfft/irfft family (#114)
- ferray-random: README references nonexistent types (SFC64, MT19937, wrong API) (#147)
- ferray-polynomial: no high-degree root finding tests (degree >= 10) (#130)
- ferray-random: no reproducibility tests against known reference values (#145)
- ferray-random: no statistical goodness-of-fit tests (KS, chi-squared) (#134)
- ferray-io: no tests for loading big-endian .npy files (#115)
- ferray-fft: no tests for fft2/fftn s (shape/padding) parameter (#113)
- ferray-fft: no tests for rfft/irfft along non-last axes (#112)
- ferray-fft: no tests for normalization modes on N-D FFTs (#111)
- ferray-strings: ljust and rjust lack fillchar parameter (#165)
- ferray-stride-tricks: no tests for non-contiguous (Fortran-ordered) source arrays (#176)
- ferray-io: no tests for malformed .npy files (#117)
- ferray-strings: no Unicode tests for multi-byte characters (#158)
- ferray-ma: no tests for 2D or higher-dimensional masked arrays (#151)
- ferray-linalg: no tests for solve with singular matrices (#103)
- ferray-linalg: no tests for empty (0x0) or 1x1 matrices (#107)
- ferray-io: no tests for Fortran-order arrays (#116)
- ferray-random: standard_normal_parallel is sequential — parallelism is not implemented (#443)
- ferray-random: binomial uses normal approximation for large n*p instead of BTPE algorithm (#450)
- ferray-fft: irfft output is Complex<f64> instead of f64 — should return real array (#437)
- ferray-core: broadcast_to rejects negative strides — NumPy handles them (#359)
- ferray-autodiff: no numerical gradient comparison tests (finite-difference) (#188)
- ferray-window: apply_along_axis returns wrong shape for 1-D input (#180)
- ferray-autodiff: powf with both variable base and exponent untested (#190)
- ferray-linalg: dot/matmul/outer all flatten via iter().copied().collect() before computing (#425)
- ferray-autodiff: zero tests for Rem derivative (#192)
- ferray-linalg: solve detects singularity via NaN check instead of pivot analysis (#414)
- ferray-autodiff: higher-order derivatives (nested DualNumber) untested (#544)
- ferray-core: ArcArray only supports C-contiguous layout (#360)
- ferray-linalg: lstsq computes full SVD just for rank/singular values (#419)
- ferray-stats: mean requires Float — can't compute mean of integer arrays (#461)
- ferray-stats: sort/argsort axis defaults to None (flatten) — NumPy defaults to -1 (last axis) (#465)
- ferray-stride-tricks: sliding_window_view rejects negative strides — can't window reversed arrays (#527)
- ferray-io: savetxt fmt is a simple string replace — not NumPy's printf-style format (#495)
- ferray-io: savez/savez_compressed buffer each .npy in memory before writing to zip (#493)
- ferray-stats: nan_aware allocates per-lane filter Vec — could use in-place NaN compaction (#471)
- ferray-linalg: norm P(p) for matrices silently falls back to Frobenius (#420)
- ferray-window: vectorize only works on Ix1 — not generic over dimension (#533)
- ferray-io: load reads element-by-element — no bulk read for native endianness (#488)
- ferray-io: save writes element-by-element — no bulk write_all on contiguous data (#487)
- ferray-ufunc: add_reduce collects all data into flat Vec before reducing — O(2n) memory (#391)
- ferray-linalg: pinv ignores rcond parameter (#415)
- ferray-io: save fails on non-contiguous arrays — should auto-contiguify (#489)
- ferray-ufunc: no clip for integer types — requires Float bound (#402)
- ferray-ufunc: gcd/lcm use Float trait — should work on actual integers (#390)
- ferray-ufunc: multiply_outer copies data twice (into Ix2, then into IxDyn) (#392)
- ferray-ufunc: divmod computes floor_divide and remainder separately — double work (#399)
- ferray-ma: set_mask_flat uses iter_mut().nth(flat_idx) — O(n) per call (#511)
- ferray-core: flat_index uses .iter().nth() — O(n) for non-contiguous arrays (#350)
- ferray-autodiff: no tests for mathematical singularities (sqrt(0), ln(0), tan(pi/2), asin(1)) (#540)
- ferray-autodiff: gradient() allocates O(n^2) — should reuse buffer (#539)
- ferray-autodiff: no tests for ReLU pattern — max(x, 0) derivative is critical for deep learning (#538)
- ferray-autodiff: powf has division by zero and undefined ln at x=0 and x<0 (#537)
- ferray-autodiff: Rem<T> has dead computation — T::zero() * q is always 0 (#536)
- ferray-core-macros: promoted_type! does not support f16/bf16 (#208)
- ferray-core-macros: s! macro rfind(';') splits incorrectly for complex step expressions (#207)
- ferray-random: default_rng() entropy source is weak (#143)
- ferray-io: TOCTOU race in memmap read_npy_header_with_offset (#120)
- ferray-ufunc: force_scalar() LazyLock caching makes SIMD/scalar tests ineffective (#86)
- ferray-ufunc: frexp uses iterative loops instead of bit extraction — O(1074) for subnormals (#83)
- ferray-core: display formatting uses Relaxed atomic ordering (#79)
- ferray-core: frombuffer creates arbitrary T from bytes without validity check (#78)
- ferray-core: to_bytes() is unsound for types with padding bytes (#77)
- ferray-core: 13+ unwrap/expect in non-test code violates zero-panic guarantee (#67)
- ferray-core: concatenate uses iter().nth(src_flat) — O(n^2) and can panic (#57)
- ferray-random: binomial powi truncates n to i32 for large n (#140)
- ferray-random: Poisson large-lambda squeeze test is dead code (#138)
- ferray-random: Laplace sampling produces Inf when u.abs() == 0.5 (#136)
- ferray-ma: masked_binary_op missing shape validation — silently truncates (#70)
- ferray-io: unbounded header allocation from untrusted header_len (DoS) (#69)
- ferray-io: no bounds check on shape product overflow from untrusted .npy files (DoS) (#68)
- ferray-stats: sort(axis=None) returns wrong shape (#91)
- ferray-stats: histogram density parameter is silently ignored (#63)
- ferray-ufunc: unwrap, gcd, lcm can infinite-loop on NaN or large inputs (#62)
- ferray-ufunc: logaddexp/logaddexp2 silently drop NaN inputs (#61)
- ferray-ufunc: spacing does not return true ULP (#60)
- ferray-ufunc: nextafter uses float approximation instead of bit manipulation (#59)
- ferray-ma: min/max silently ignore NaN in unmasked elements (#149)
- ferray-window: kaiser rejects negative beta, but NumPy allows it (#179)
- ferray-stride-tricks: sliding_window_view has no bounds check on resulting view (#172)
- ferray-stride-tricks: sliding_window_view casts isize strides to usize without negative check (#169)
- ferray-polynomial: degree() uses different tolerance than trim() (#124)
- ferray-strings: search/boolean ops flatten output to Ix1 for multidim input (#71)
- ferray-stats: unsafe pointer casts in SIMD type-dispatch (#92)
- ferray-linalg: pinv ignores rcond parameter (#98)
- ferray-linalg: cond unnecessarily requires square matrices (#97)
- ferray-stats: min/max inconsistent NaN propagation (#90)
- ferray-fft: unnecessary unsafe impl Send/Sync for FftPlan (#110)
- ferray-linalg: solve never returns SingularMatrix error despite documenting it (#99)
- ferray-ufunc: cumsum/cumprod with axis=None returns original shape, not flattened (#84)
- ferray-linalg: slogdet computes raw determinant first, defeating numerical purpose (#64)
- ferray-linalg: inv uses unreliable determinant-based singularity check (#65)
- ferray-polynomial: quadratic formula numerically unstable (catastrophic cancellation) (#66)
- ferray-core: view_cast does not check alignment in Vec::from_raw_parts (#58)
- ferray umbrella: indexing module (~18 functions) not re-exported (#75)
- ferray umbrella: config.rs .expect() can panic in library code (#74)
- ferray umbrella: bf16 feature not exposed (#73)
- ferray umbrella: f16 feature not forwarded to ferray-ufunc (#72)
- ferray-core: iinfo::<u128>().max overflows i128, wraps to -1 (#56)
- ferray umbrella: no_std feature violates Cargo additivity, --all-features fails (#52)
- ferray-io: Fortran-order silently ignored for complex32/complex64 dynamic loading (#51)
- ferray-linalg: array2_to_faer O((m*n)^2) allocation for non-contiguous arrays (#50)
- ferray-core: broadcast_to casts negative isize strides to usize — UB for transposed views (#49)
- Patched all design docs for adversarial review findings
- Fixed .gitignore: added target/, __pycache__/, *.pyc, .worktrees/, fuzz artifacts
- Purged 3,591 accidentally committed build artifacts from git history (163MB -> 1.3MB)
