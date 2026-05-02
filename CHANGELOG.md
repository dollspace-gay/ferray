# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

> **Note**: Entries in the historical `[0.1.0 .. 0.2.10]` block below were auto-appended
> from closed-issue titles by the crosslink hook, so wording reads like the original gap
> rather than the resolution (e.g. "no f32 random generation" closed when f32 random was
> added). They are kept verbatim for traceability; future versions will use curated
> entries.

## [Unreleased]

## [0.3.5] - 2026-05-02

This release sweeps the open numpy-parity backlog to zero. Roughly 80 issues
resolved across the workspace; the curated highlights below cover the
material new code. Per-issue closing comments carry the full detail trail.

### Added

#### ferray-stats
- `unique_axis(a, axis)` returns sorted unique hyperslices along an axis (#464).
- `where_broadcast(c, x, y)` is the 3-arg `where` with NumPy broadcasting (#468).
- Eight scipy.stats descriptive helpers: `skew`, `kurtosis`, `zscore`, `mode`,
  `iqr`, `sem`, `gmean`, `hmean` (#470).
- Seven rule-based histogram bin variants on `Bins<T>` (`Sturges` / `Sqrt` /
  `Rice` / `Scott` / `Fd` / `Doane` / `Auto`); enum is now
  `#[non_exhaustive]` (#472).
- Five hypothesis tests: `pearsonr`, `spearmanr`, `ttest_1samp`,
  `ttest_ind` (Welch), `ks_2samp`. p-values via Lentz-CF regularised
  incomplete beta (#724).
- `chi2_contingency` returning statistic + p-value + dof + expected. p-value
  via regularised upper incomplete gamma (#743).

#### ferray-polynomial
- `from_roots` constructor across all six bases via a default on
  `FromPowerBasis` (#476).
- `linspace`, `Poly::basis(deg)`, `FromPowerBasis::identity()` defaults (#477, #478).
- `integ_with_bounds(m, k, lbnd, scl)` on `Polynomial` and as a default for
  the other five bases (#482, #732).
- `convert_with_mapping` carrying domain/window across basis conversion (#483).
- `Polynomial::compose` plus a `Poly::compose` default for non-power bases (#485, #733).
- `Display` impls across all six basis types (#486).
- `ComplexPolynomial` for complex-coefficient power-basis arithmetic (#730).
- `ChebyshevF32` / `HermiteF32` / `HermiteEF32` / `LegendreF32` / `LaguerreF32`
  via a single `impl_f32_wrapper!` macro (#725-#729).

#### ferray-fft
- `fft_into` / `ifft_into` / `rfft_into` / `irfft_into` for pre-allocated
  output buffers (#429).
- `FftPlanND` for repeated multi-dim FFTs of fixed shape (#436).

#### ferray-random
- `multivariate_hypergeometric` distribution (#445).
- `shuffle_dyn` / `permuted_dyn` / `choice_dyn` for N-D arrays along an axis (#447, #448).
- Broadcast-aware distribution params: `normal_array`, `uniform_array`,
  `exponential_array`, `poisson_array` (#449).
- `multivariate_normal_array` taking `Array<f64, Ix1/Ix2>` and using
  `ferray_linalg::cholesky` (faer-backed) (#451, #452).
- `state_bytes` / `set_state_bytes` round-trip for all six BitGenerators
  with structural-invariant validation on restore (#453).
- `random_into` / `standard_normal_into` / `standard_exponential_into` for
  hot-loop buffer reuse (#454).
- Seven typed integer generators: `integers_{u8,i8,u16,i16,u32,i32,u64}` (#456).

#### ferray-window
- `f32` siblings for the numpy 5 (#529).
- Closed-form scipy windows: `boxcar`, `triang`, `bohman`, `flattop`,
  `lanczos` (#738).
- `chebwin` Dolph–Chebyshev via direct DFT, byte-exact scipy parity (#740).
- `dpss` Slepian taper via power iteration on the symmetric tridiagonal
  concentration matrix (#745).

#### ferray-strings
- `rsplit` and `splitlines` (#515).
- `at(i)` / `get_row(i)` / `slice_axis(axis, range)` for N-D StringArray
  indexing (#519).
- `CompactStringArray` Arrow-style offsets+values backend behind the
  `compact-storage` feature flag — roughly 3× memory reduction for
  short-string workloads (#736).

#### ferray-stride-tricks
- `sliding_window_view_axis` for axis-restricted sliding windows (#523).
- `as_strided_signed` with `isize` strides + base offset (#737).

#### ferray-io
- `savetxt_1d` / `loadtxt_1d` for 1-D text I/O (#494).
- `NpyElement` for `Complex<f32>` and `Complex<f64>` so typed save/load works
  end-to-end (#498).
- `load_record` / `save_record` for structured-dtype `.npy` files into
  `FerrayRecord`-typed arrays (#742).

#### ferray-numpy-interop
- `NpElement` for `half::f16` behind the `f16` feature (#739, partial).

#### ferray-core
- `DType::FixedAscii(n)` / `FixedUnicode(n)` / `RawBytes(n)` for numpy's
  `S{n}` / `U{n}` / `V{n}` dtype families (#741).
- Blanket `impl<T: FerrayRecord> Element for T` so structured-array load
  paths work without per-type Element wiring (#742).

### Changed
- ferray-polynomial `roots` delegates to `ferray_linalg::eigvals` (faer Schur),
  removing ~275 lines of in-tree QR scaffolding (#479).
- ferray-polynomial Hermite / HermiteE / Laguerre `deriv` / `integ` use direct
  in-basis recurrences (O(n) per step) instead of round-tripping through the
  power basis (#720).
- ferray-fft `fft_along_axes` skips the redundant upfront `data.to_vec()`
  copy (#439).

### Fixed
- ferray-stats `correlate` was computing convolution; now matches numpy's
  cross-correlation `c[k] = Σ_n a[n+k] * v[n]`. Pre-existing `test_correlate_valid`
  was pinning the buggy output; corrected (#722).
- ferray-stats `corrcoef` returns NaN for zero-deviation rows (matches numpy);
  previously returned 0 / 1 (#723).
- ferray-linalg AVX2 i8 GEMM kernel test harness used debug-overflowing `i8 * i8`
  arithmetic on its synthetic inputs; switched to `wrapping_mul` / `wrapping_sub`.

## [0.3.1] - 2026-05-01

### Added
- ferray-numpy-interop: no Arrow RecordBatch interop (#559)
- ferray-numpy-interop: no Polars DataFrame <-> ferray 2D array conversion (#558)
- ferray-numpy-interop: no null handling strategy — nulls always rejected (#556)
- ferray-numpy-interop: no DynArray interop — runtime-typed arrays can't be converted (#554)
- ferray-numpy-interop: no complex type support in any backend (#552)
- ferray-numpy-interop: no f16/bf16 support in any backend (#551)
- ferray-numpy-interop: no bool support for NumPy interop (#553)
- ferray-window: f64-only — no f32 window generation (#529)
- ferray-window: no scipy.signal window functions (tukey, flattop, cosine, nuttall, etc.) (#531)
- ferray-stride-tricks: as_strided only accepts usize strides — no negative strides (#524)
- ferray-stride-tricks: sliding_window_view doesn't support axis parameter (#523)
- ferray-strings: no indexing/slicing on StringArray (#519)
- ferray-strings: missing functions — swapcase, rfind, index, rindex, partition, rpartition, rsplit, splitlines, translate, expandtabs, mod, encode, decode, slice (#515)
- ferray-io: no structured/string dtype support for .npy files (#490)
- ferray-io: no allow_pickle parameter — NumPy's load has security considerations (#499)
- ferray-io: no fromregex equivalent (#497)
- ferray-io: NpyElement not implemented for Complex — typed save/load for complex arrays requires DynArray path (#498)
- ferray-io: no f16/bf16 support in .npy I/O (#491)
- ferray-io: loadtxt/savetxt only work with Ix2 — no 1D text I/O (#494)
- ferray-polynomial: no Display/Debug formatting matching NumPy's polynomial printing (#486)
- ferray-polynomial: no polynomial composition (p(q(x))) (#485)
- ferray-polynomial: no convert/cast methods for cross-basis conversion with domain (#483)
- ferray-polynomial: fit doesn't support weights parameter (#484)
- ferray-polynomial: integ missing lbnd (lower bound) and scl (scale) parameters (#482)
- ferray-polynomial: no std::ops trait implementations — must use .add()/.mul() methods (#481)
- ferray-polynomial: no identity or basis class methods (#478)
- ferray-polynomial: no linspace method (#477)
- ferray-polynomial: no fromroots constructor (#476)
- ferray-polynomial: f64-only — no generic float or complex support (#475)
- ferray-stats: histogram bins='auto'/'sturges'/'fd' etc. not supported (#472)
- ferray-stats: no scipy.stats equivalents — no t-test, ks-test, mode, skew, kurtosis, zscore (#470)
- ferray-stats: unique only works on flattened arrays — no axis parameter (#464)
- ferray-random: integers() only returns i64 — no dtype parameter (#456)
- ferray-random: no out= parameter — every generation allocates (#454)
- ferray-random: no state serialization — can't save/restore Generator state (#453)
- ferray-random: distributions don't accept array parameters — no broadcasting (#449)
- ferray-random: choice only works on 1D arrays — no axis parameter (#448)
- ferray-random: shuffle/permuted only work on 1D arrays (#447)
- ferray-random: missing distributions — zipf, noncentral_chisquare, noncentral_f, multivariate_hypergeometric (#445)
- ferray-random: no SFC64 (Small Fast Chaotic) BitGenerator (#444)
- ferray-fft: ND FFT applies transforms sequentially per axis — no simultaneous multi-axis transform (#439)
- ferray-fft: FftPlan only supports 1D — no plan for 2D/ND transforms (#436)
- ferray-fft: no out= parameter on any FFT function (#429)
- ferray-linalg: no LinAlgError type — uses generic FerrayError variants (#423)
- ferray-linalg: einsum generic contraction is O(product of all dims) — no opt_einsum-style optimization (#418)
- ferray-linalg: svd hermitian= parameter missing — no fast path for symmetric matrices (#407)
- ferray-ufunc: no floating-point error state handling (errstate/seterr) (#386)
- ferray-ufunc: no integer SIMD paths — all integer ops are scalar (#384)
- ferray-ufunc: SIMD only for f64, no f32 intrinsic kernels (#383)
- ferray-core: no unique/searchsorted/set operations (#374)
- ferray-core: no structured/record dtype in DType enum (#342)
- ferray-core: no WRITEBACKIFCOPY pattern for safe out-parameter mutation (#370)
- ferray-core: no WRITEBACKIFCOPY pattern for safe out-parameter mutation (#370)
- ferray-core: no buffer protocol / FFI-safe array descriptor (#358)
- ferray-core: no memory-mapped array support (#356)
- ferray-core: no einsum in the workspace (#354)
- ferray-core: no alignment flag or ALIGNED check (#345)
- ferray-core: no byte-order / endianness handling in DType (#344)
- ferray-core: no structured/record dtype in DType enum (#342)
- Bump oracle fixture tolerance floor to 10 ULP for pure-Rust precision envelope (#47)
- Run cargo tarpaulin code coverage analysis (#44)
- ferray-python: NumPy drop-in replacement crate (PyO3 bindings, phased) (#684)
- ferray-python phase 4 follow-up: as_strided binding (NumPy byte strides vs ferray element strides — needs careful semantic translation) (#714)
- ferray-python phase 2 follow-up: datetime/timedelta arithmetic and predicates (isnat, add/sub_datetime/timedelta) (#706)
- ferray-python phase 3 follow-up: random Generator class (PyO3 #[pyclass]) for default_rng() modern API (#712)
- ferray-python phase 3 follow-up: random additional distributions (binomial, poisson, geometric, beta, gamma, exponential, lognormal, chisquare, multinomial, multivariate_normal) (#713)
- ferray-python phase 3 follow-up: batched linalg (cholesky_batched, eigh_batched, svd_batched, qr_batched, solve_batched, inv_batched, pinv_batched) (#709)
- ferray-python phase 3 follow-up: complex linalg (matmul_complex, solve_complex, det_complex, inv_complex, eig) (#708)
- ferray-python phase 3 follow-up: hfft/ihfft (Hermitian FFT) (#707)
- ferray-python phase 3 follow-up: lstsq (4-tuple return), tensorinv, tensorsolve, matrix_transpose, diagonal (#711)
- ferray-python phase 3 follow-up: norm_axis, cond, einsum, multi_dot, matmul/vecdot/matvec/vecmat at top level (numpy 2.0 vector products) (#710)
- ferray-python phase 2 follow-up: histogram/histogram2d/histogramdd/bincount/digitize (#704)
- ferray-python phase 2 follow-up: percentile/quantile/median + cov/corrcoef/correlate (#703)
- ferray-python phase 2 follow-up: gradient, trapezoid, ediff1d, sinc, i0, convolve, correlate, interp (#702)
- ferray-python phase 2 follow-up: gcd/lcm, frexp/ldexp/modf/spacing/nextafter, exp_fast, divmod, fmod (#701)
- ferray-python phase 2 follow-up: complex ops (real/imag/conj/conjugate/angle, iscomplex/isreal) (#700)
- ferray-python phase 1 follow-up: bind frombuffer/fromfile/fromstring/fromiter/fromfunction (#692)
- ferray-python phase 2 follow-up: bitwise ops (bitwise_and/or/xor/not/invert, left_shift, right_shift, bitwise_count) (#699)
- ferray-python phase 1 follow-up: bind take/take_along_axis/choose/compress/select/place/putmask/extract (#697)
- ferray-python phase 1 follow-up: bind split/array_split/vsplit/hsplit/dsplit (#696)
- ferray-python phase 1 follow-up: bind block/r_/c_/column_stack/row_stack (#695)
- ferray-python phase 1 follow-up: bind meshgrid/mgrid/ogrid/vander (#693)
- ferray-python phase 1 follow-up: bind pad/tile/repeat/delete/insert/append/resize/trim_zeros (#694)
- ferray-python phase 2 follow-up: lexsort/partition/argpartition, sort_complex, unique_all/counts/inverse, allclose/isclose/array_equal/array_equiv (#705)
- ferray-python phase 4 follow-up: ferray.autodiff DualNumber bindings (ferray-specific, not NumPy parity but useful for Python users) (#718)
- ferray-python phase 4 follow-up: numpy.lib top-level (vectorize, apply_along_axis, apply_over_axes, piecewise from ferray-window functional utilities) (#719)
- ferray-python phase 4 follow-up: numpy.ma masked arrays — masked_array class + masked_where / mask-aware reductions (#716)
- ferray-python phase 4 follow-up: numpy.char namespace (lower/upper/strip/replace/find/count/etc) — needs StringArray pyclass (#715)
- ferray-python phase 4 follow-up: numpy.polynomial class API — Polynomial/Chebyshev/Hermite/HermiteE/Laguerre/Legendre #[pyclass] wrappers (#717)
- ferray-python phase 4: window + polynomial + autodiff + strings + ma + stride_tricks (#689)
- ferray-python phase 3: linalg + fft + random submodule bindings (#688)
- ferray-python phase 2: ufunc + stats bindings (#687)
- ferray-python phase 1: core creation + dtype + manipulation + indexing bindings (#686)
- ferray-python phase 0: cdylib + pyo3 scaffold + maturin build (#685)
- Add 9 missing windows: cosine, exponential, gaussian, general_cosine, general_hamming, nuttall, parzen, taylor, tukey (#683)

- ferray-window: 9 new windows extending the existing bartlett/blackman/hamming/hanning/kaiser
  set with the SciPy / `torch.signal.windows` extras: `cosine`, `exponential` (with optional
  centre + tau), `gaussian` (configurable std), `general_cosine` (arbitrary cosine-sum
  coefficients — recovers Hann, Blackman, etc.), `general_hamming` (alpha-parameterised),
  `nuttall` (4-term minimum-derivative), `parzen` (cubic B-spline), `taylor` (Carrara &
  Goodman with `nbar`/`sll`/`norm` knobs), `tukey` (cosine-tapered, `alpha` ∈ [0, 1]).
  All return `Array<f64, Ix1>` matching the existing convention; 22 new tests cover
  endpoints, symmetry, equivalence with the canonical-cosine forms (Hann ≡ Tukey(1) ≡
  general_hamming(0.5), classical Blackman ≡ general_cosine([0.42, 0.5, 0.08])), and
  parameter-validation error paths. (#683)

### Workspace

- All workspace crates bumped from 0.3.0 → 0.3.1 to ship the new windows.

## [0.3.0] - 2026-04-30

### Performance

- ferray-ufunc: route default sin/cos/tan through libm (Float::sin) — ferray now wins or ties NumPy at every size for trig (#679)
- ferray-stats: pairwise-sum base 256 → 4096, cuts horizontal-sum overhead 16× — sum/mean 100K-1M now beats NumPy or ties at DRAM ceiling (#678)
- ferray-linalg: eliminate Vec round-trip in matmul/dot 2D dispatch — matmul 50×50 went from 1.33× **slower** than MKL to 1.04× **faster** (#677)

### Added (GEMM)

- ferray-linalg: optional OpenBLAS backend (cargo feature openblas) for perf-floor guarantee with system libopenblas (#675)
- ferray-linalg: native TRSM primitives (block-recursive triangular solve, lower/upper × non-unit/unit-diag for f32/f64) (#674)
- ferray-linalg: mixed-precision GEMM (gemm_bf16_f32 / gemm_f16_f32, behind `bf16`/`f16` features) (#673)
- ferray-linalg: AVX-512 4×16 i16 GEMM kernel (32 i16 per VPMADDWD, 2× peak vs AVX2) (#672)
- ferray-linalg: dedicated AVX2 i8 signed-signed kernel (pre-widen-A + i8-B + kernel-side VPMOVSXBW, 1.62× vs old wrapper at N=1024) (#671)
- ferray-linalg: AVX-512 VNNI i8 GEMM kernel (zmm-wide vpdpbusd, 1.5 TOPS at N=1024) (#667)
- ferray-linalg: i8 quantized GEMM (u8 × i8 → i32) for ML inference, with AVX2 baseline + AVX-VNNI fast path (#666)
- ferray-linalg: AVX-512 ZGEMM + CGEMM kernel ports from OpenBLAS Skylake-X (#665)
- ferray-linalg: CGEMM (Complex f32) kernel port from OpenBLAS Haswell (#664)
- ferray-linalg: ZGEMM (Complex f64) kernel port from OpenBLAS Haswell (#663)
- ferray-linalg: AVX-512 SGEMM kernel port from OpenBLAS Skylake-X (#662)
- ferray-linalg: AVX-512 DGEMM kernel port from OpenBLAS Skylake-X (#661)
- ferray-linalg: SGEMM Haswell kernel port from OpenBLAS (4×16 tile, 987 GFLOPS at N=1536) (#660)
- ferray-linalg: hybrid 2D (M × N) parallelism grid for small-N regime (#659)
- ferray-linalg: per-thread B-prepack + larger KC pushed N=512 past 200 GFLOPS (#657)
- ferray-linalg: per-thread A-pack reuse (#656)
- ferray-linalg: M1/M2 pipelining + better threading at mid-N for the translated 4×8 kernel (#655)
- ferray-linalg: hand-tuned Rust BLAS micro-kernels (OpenBLAS Haswell DGEMM port, 4×8 tile, base of all later GEMM work) (#654)
- ferray-linalg: matmul 50x50 was 10× slower than NumPy — sweep + regression analysis (#653)
- ferray-linalg: quantized_matmul with zero-points + scales (ONNX QLinearMatMul / oneDNN formula) (#670)
- ferray-linalg: i8 × i8 signed-signed GEMM via VPMOVSXBW + VPMADDWD (#669)
- ferray-linalg: i16 × i16 → i32 GEMM via VPMADDWD (#668)

### Added (other)

- ferray-ufunc: f32 SIMD kernels for abs / neg / square / reciprocal (parity with f64; 8 lanes per AVX2 ymm) (#676)
- ferray-ufunc: 20 complex transcendentals (sin/cos/tan, sinh/cosh/tanh, asin/acos/atan, asinh/acosh/atanh, exp/ln/log2/log10/sqrt/expm1/log1p/power) (#398)
- ferray-ufunc: matmul / vecdot / matvec / vecmat already provide gufunc-semantic dispatch via IxDyn shape inspection (research-resolution closed) (#397)

### Polish

- Workspace polish: re-ran benchmarks, refreshed BENCHMARKS.md, bumped to 0.3.0, cargo fmt --all, cargo clippy --workspace -D warnings clean across default + bf16 + f16 + openblas feature combos, removed two truly-dead pub(crate) legacy aliases in ferray-fft/src/plan.rs (#680)

### Notes

- ferray-bench is held at version 0.1.0 (benchmark binary, not a published crate).
- `bf16` cargo feature added on ferray-linalg (mirrors the existing `f16` feature).
- The `openblas` feature defaults to `["system", "cblas"]` of openblas-src; users who want vendored static linking should add `openblas-src/static` at the workspace level.

## [0.2.11] - 2026-04-28

### Changed
- ferray-window: dpss (discrete prolate spheroidal sequences) (#745)
- ferray-io: load structured dtype .npy into FerrayRecord arrays (#742)
- ferray-core: backing DType variants for fixed-width strings (FixedAscii, FixedUnicode, RawBytes) (#741)
- ferray-window: dpss and chebwin (need numerical solvers) (#740)
- ferray-strings: alternative compact storage backend (fixed-width or Arrow-style) (#736)
- ferray-numpy-interop: expansion epic — f16/bf16, complex, DynArray, nulls, DataFrame, RecordBatch (#739)
- ferray-polynomial: genericize Poly trait over coefficient type T: Float (#731)
- ferray-polynomial: complex-coefficient polynomials (#730)
- ferray-stats: chi2_contingency (needs incomplete gamma) (#743)
- ferray-core: expose RawArrayView for negative-stride view construction (#744)
- ferray-stride-tricks: as_strided_signed with isize strides + base offset (#737)
- ferray-stats: add scipy.stats statistical-test functions (ttest, ks_2samp, chi2, pearsonr) (#724)
- ferray-io: parse numpy structured dtype descriptors and load into FerrayRecord arrays (#735)
- ferray-io: parse numpy fixed-width string dtype descriptors (S20, U10, V8) (#734)
- ferray-polynomial: compose for non-power bases (#733)
- ferray-polynomial: integ_with_bounds for non-power bases (#732)
- ferray-window: remaining scipy.signal windows (flattop, bohman, boxcar, triang, lanczos, dpss, chebwin) (#738)
- ferray-polynomial: LaguerreF32 type (#729)
- ferray-polynomial: LegendreF32 type (#728)
- ferray-polynomial: HermiteEF32 type (#727)
- ferray-polynomial: HermiteF32 type (#726)
- ferray-polynomial: ChebyshevF32 type (f32 sibling of Chebyshev) (#725)
- ferray-ufunc: expand f32 test coverage in arithmetic.rs (follow-up to #152) (#721)
- ferray-polynomial: migrate Hermite/HermiteE/Laguerre deriv/integ to direct recurrences (follow-up to #131) (#720)
- ferray-polynomial: lift from f64-only to T: Element + Float (and complex coeffs) (#612)
- ferray-numpy-interop: bool/non-bool split requires separate trait names (ToArrow vs ToArrowBool) (#557)
- ferray-numpy-interop: Arrow FromArrow consumes the PrimitiveArray — should accept reference (#555)
- ferray-window: functional utilities (vectorize, piecewise, apply_along_axis) are misplaced in window crate (#532)
- ferray-strings: uses owned String per element — no fixed-width or interned storage (#522)
- ferray-polynomial: fitting uses internal normal equations instead of ferray-linalg lstsq (#480)
- ferray-polynomial: roots uses internal QR iteration instead of ferray-linalg eigvals (#479)
- ferray-stats: where_ always copies and flattens — should use nonzero for index arrays (#468)
- ferray-random: multivariate_normal has its own Cholesky — should use ferray-linalg (#452)
- ferray-random: multivariate_normal takes flat cov slice instead of Array<f64, Ix2> (#451)
- ferray-linalg: solve only accepts Ix2 for A and IxDyn for b — inconsistent typing (#411)
- ferray-core: indexing always returns IxDyn — loses compile-time rank (#349)
- ferray-core: Element trait is sealed — no user-defined element types (#365)
- ferray-core: no __array_function__-style dispatch protocol for duck-typing (#357)
- ferray-core: indexing always returns IxDyn — loses compile-time rank (#349)
- ferray-core: strides are element-counted, not byte-counted — limits dtype-erased operations (#341)
- ferray-core-macros: all macro errors point to call_site() instead of actual token span (#324)
- ferray-core-macros: s! macro string-based parsing is fragile (#322)
- ferray-numpy-interop: misleading SAFETY comment on safe API call (#314)
- ferray-numpy-interop: duplicate Bool traits (ToArrowBool/FromArrowBool/ToPolarsBool/FromPolarsBool) (#311)
- ferray-numpy-interop: IntoNumPy for Array2 uses flatten-then-reshape (#310)
- ferray-numpy-interop: FromArrow consumes self but copies anyway (#309)
- ferray-numpy-interop: missing f16 <-> Float16 mapping in dtype_map (#308)
- ferray-numpy-interop: Arrow/Polars conversions are 1-D only (#307)
- ferray-strings: broadcast_binary materializes all index pairs into Vec (#281)
- ferray-strings: no Serialize/Deserialize support for StringArray (#279)
- ferray-strings: split returns Vec<Vec<String>> instead of structured array type (#277)
- ferray-ma: var() iterates data twice (#274)
- ferray-ma: sort/compressed always flatten to 1D (#271)
- ferray-ma: no ddof parameter on var/std (#270)
- ferray-ma: argsort returns Vec<usize> instead of Array (#269)
- ferray-ma: count_masked ignores axis parameter (#268)
- ferray-random: weighted_sample_with_replacement uses O(log n) binary search, not O(1) alias method (#265)
- ferray-polynomial: redundant From impls alongside ConvertBasis trait (#249)
- ferray-polynomial: no operator overloading (Add/Sub/Mul/Neg traits) (#248)
- ferray-polynomial: pow() uses naive repeated multiplication instead of binary exponentiation (#247)
- ferray-polynomial: companion_size is dead code (#246)
- ferray-polynomial: x_pow_prev computed but never used in power_to_chebyshev/legendre (#245)
- ferray-io: build_dim_from_shape uses Box<dyn Any> downcast (#236)
- ferray-io: element-at-a-time I/O for npy read/write (#234)
- ferray-autodiff: no ferray-core integration — DualNumber standalone (#186)
- ferray-stats: SIMD acceleration only for f64, not f32 (#173)
- ferray-stats: mean/var/std_ don't support integer input with float output (#170)
- ferray-ma: ufuncs don't delegate to SIMD-accelerated ferray-ufunc (#157)
- ferray-ma: set_mask_flat uses iter_mut().nth(n) — O(n) per call (#154)
- ferray-polynomial: non-power basis ops do unnecessary roundtrip conversions (#131)
- ferray-polynomial: massive code duplication across 5 non-power bases (#128)
- ferray-polynomial: Poly trait hardcoded to f64 (#126)
- ferray-polynomial: missing domain/window support for Chebyshev/Legendre fitting (#122)
- ferray-linalg: einsum generic_contraction uses scalar loops for all cases (#102)
- ferray-ufunc: convolve is O(n*m) with no FFT fallback for large inputs (#89)
- ferray-ufunc: basic arithmetic (add/sub/mul/div) don't use SIMD dispatch (#88)
- Audit ferray crates against NumPy modules to surface API gaps (#578)
- chore: bump workspace to 0.2.10 (#567)
- ferray-bench: add single mode that returns result arrays for accuracy testing (#652)
- speed_benchmark.py: matmul size_str missing second matrix shape (#651)
- Benchmark all ferray APIs against real NumPy with raw output (#650)
- Add Claude skills: ferray-python binding patterns + per-submodule API context (#691)
- ferray-python: CI wheel matrix (Linux/macOS/Windows × py3.10-3.13) + PyPI publish (#690)
- ferray-ufunc: add isnat (datetime NaT — gated on datetime dtype scope) (#594)
- ferray-numpy-interop: complex types (links #552) (#642)
- ferray-numpy-interop: null/missing handling (links #556) (#646)
- ferray-numpy-interop: Arrow RecordBatch (links #559) (#648)
- ferray-numpy-interop: Polars DataFrame round-trip (links #558) (#647)
- ferray-numpy-interop: null/missing handling (links #556) (#646)
- ferray-numpy-interop: Arrow FromArrow impls (links #555) (#645)
- ferray-numpy-interop: DynArray integration (links #554) (#644)
- ferray-numpy-interop: bool support (links #553/#557) (#643)
- ferray-numpy-interop: complex types (links #552) (#642)
- ferray-numpy-interop: f16/bf16 support (links #551) (#641)
- ferray-ma: class helpers (mvoid, nomask, masked_singleton, masked_print_option, mr_, isMaskedArray/isMA/isarray, getmaskarray, ids) (#640)
- ferray-ma: bitwise/comparison ufunc support (and/or/xor/shift, equal/less/greater family, logical_*, floor_divide/fmod/power/remainder/true_divide/hypot/negative/sqrt/square) (#639)
- ferray-ma: fill-value protocol (default_fill_value, maximum_fill_value, minimum_fill_value, set_fill_value, common_fill_value) (#638)
- ferray-ma: apply_along_axis/apply_over_axes/vander (#637)
- ferray-ma: linalg & set & corr ops (dot/inner/outer/innerproduct/outerproduct/trace, correlate/convolve/corrcoef/cov, unique/in1d/isin/intersect1d/setdiff1d/setxor1d/union1d, polyfit) (#636)
- ferray-ma: mask manipulation (harden/soften, mask_or, make_mask*, compress_rows/cols/rowcols/nd, mask_rows/cols/rowcols, clump_*masked, *notmasked_contiguous/edges, flatten_mask, flatten_structured_array) (#635)
- ferray-ma: manipulation (concatenate/stack family/reshape/ravel/squeeze/swapaxes/transpose/expand_dims/atleast_*/take/put/putmask/repeat/resize/append/diag/diagflat/diagonal/choose/clip) (#634)
- ferray-ma: array creation (arange, zeros, ones, empty, identity, masked_all, masked_array, masked_values, masked_object, masked_outside, masked_less/_equal, masked_not_equal, frombuffer, fromfunction, fromflex) (#633)
- ferray-ma: full reductions (sum/prod/mean/std/var/cumsum/cumprod/min/max/argmin/argmax/median/ptp/average/anom) (#632)
- ferray-strings: add translate (char-translate ufunc) (#631)
- ferray-strings: add slice (string-slicing ufunc) (#630)
- ferray-strings: add rfind (#629)
- ferray-strings: add partition/rpartition (#628)
- ferray-strings: add mod (printf-style % formatting) (#627)
- ferray-strings: add isdecimal (#626)
- ferray-strings: add index/rindex (find variants that raise on miss) (#625)
- ferray-strings: add expandtabs (#624)
- ferray-strings: add decode/encode (UTF-8 codec ufuncs) (#623)
- ferray-polynomial: add mulx (multiply-by-x) for all 6 bases (#622)
- ferray-polynomial: expose explicit basis-conversion functions (cheb2poly, poly2cheb, etc.) (#621)
- ferray-polynomial: add Chebyshev-only extras (chebpts1, chebpts2, chebweight, chebinterpolate) (#620)
- ferray-polynomial: add Gauss quadrature (chebgauss/leggauss/hermgauss/lagguass/hermegauss) (#619)
- ferray-polynomial: add multivariate Vandermonde (vander2d/vander3d) for all 6 bases (#618)
- ferray-polynomial: add multivariate grid (grid2d/grid3d) for all 6 bases (#617)
- ferray-polynomial: add multivariate eval (val2d/val3d) for all 6 bases (#616)
- ferray-polynomial: add polyvalfromroots (#615)
- ferray-polynomial: add polyline/chebline/legline/lagline/hermline/hermeline (#614)
- ferray-polynomial: add polyfromroots/chebfromroots/legfromroots/lagfromroots/hermfromroots/hermefromroots (#613)
- ferray-polynomial: lift from f64-only to T: Element + Float (and complex coeffs) (#612)
- ferray-random: add standard_t alias for student_t (numpy spelling) (#611)
- ferray-random: expose explicit SeedSequence type (#610)
- ferray-random: add SFC64 BitGenerator (#609)
- ferray-random: add PCG64DXSM BitGenerator (#608)
- ferray-random: add MT19937 BitGenerator (#607)
- ferray-random: add zipf distribution (#606)
- ferray-random: add noncentral_f distribution (#605)
- ferray-random: add noncentral_chisquare distribution (#604)
- ferray-io: add lib.format helpers (descr_to_dtype, header_data_from_array_1_0, read_array, write_array) (#603)
- ferray-io: add DataSource (URL/compressed file abstraction) (#602)
- ferray-io: add fromregex (#601)
- ferray-stats: add unique_values/unique_counts/unique_inverse/unique_all (Array API) (#600)
- ferray-stats: add sort_complex (#599)
- ferray-stats: add histogram_bin_edges (#598)
- ferray-stats: add ptp (peak-to-peak) (#597)
- ferray-stats: add average (weighted mean) (#596)
- ferray-stats: add nanargmin/nanargmax (#595)
- ferray-ufunc: add isnat (datetime NaT — gated on datetime dtype scope) (#594)
- ferray-ufunc: re-export matmul/vecdot/vecmat/matvec at top level for ufunc-call parity (#593)
- ferray-ufunc: add cumulative_sum/cumulative_prod aliases (Array API names) (#592)
- ferray-ufunc: add iscomplex/isreal/iscomplexobj/isrealobj/isscalar predicates (#591)
- ferray-ufunc: add modf (split into integer/fractional parts) (#590)
- ferray-core: add top-level copy function (#589)
- ferray-core: add c_/r_/index_exp slice builders (#588)
- ferray-core: add fromfile/fromfunction/fromstring (#587)
- ferray-core: add mask_indices (#586)
- ferray-core: add place/putmask/extract (#585)
- ferray-core: top-level alias for where (currently where_select) (#584)
- ferray-core: top-level alias for unique (currently in ferray-stats) (#583)
- ferray-core: add vander (Vandermonde matrix) (#582)
- ferray-core: add ascontiguousarray/asfortranarray/asanyarray/asarray_chkfinite/require (#581)
- ferray-core: add broadcast/broadcast_arrays/broadcast_shapes top-level (#580)
- ferray-core: add atleast_1d/atleast_2d/atleast_3d (#579)
- Fix crosslink hub sync divergence (225 commits ahead of remote) (#649)
- release: commit 0.2.11 maintenance + dep upgrades (#577)
- chore: periodic codebase maintenance pass (deps/lint/tests/docs/issues) (#569)
- chore: bump workspace to 0.2.11 (#576)
- Bump workspace to 0.2.11
- Major dep upgrades to highest semver: pyo3 0.24 → 0.28, numpy 0.24 → 0.28, arrow 54 → 58,
  polars 0.46 → 0.53, zip 2.2 → 8, getrandom 0.2 → 0.4, rand_core 0.6 → 0.10, proptest 1.6 → 1.11,
  half 2.4 → 2.7, realfft 3.4 → 3.5, rayon 1.11 → 1.12, regex 1.11 → 1.12, flate2 1.0 → 1.1
- PyO3 0.28 API migration: `prepare_freethreaded_python` + `Python::with_gil` → `Python::initialize`
  + `Python::attach`
- getrandom 0.4 API: `getrandom::getrandom` → `getrandom::fill`
- Workspace clippy clean at `-D warnings -W clippy::pedantic -W clippy::nursery` (started from
  3529 warnings under that lint set)

### Security
- Resolved RUSTSEC-2026-0097 (rand 0.9.2 unsound) via dep upgrades. Paste 1.0.15 unmaintained
  advisory remains, transitive via pulp/gemm with no fix path until upstream migrates.

## [0.2.10] - 2026-04-28

### Added
- ferray-ufunc: no matmul/vecdot/matvec ufuncs (gufuncs) (#397)
- ferray-ufunc: complex transcendentals missing (complex exp, log, sin, cos, sqrt, power) (#398)
- ferray-core: no datetime/timedelta dtype support (#343)
- `MaskAware` trait + `ma_apply_unary` helper in ferray-ma for generic mask-preserving unary ops (#505)

### Changed
- chore: major dep upgrades + finish maintenance issues #570/#571/#572/#573 + strict clippy (#574)
- tracking: continue issue #574 work (deps + lints) (#575)
- docs: README.md test count (1479) and crate count (15) stale — actual 2716 tests / 17 crates (#572)
- docs: CHANGELOG.md lists open-issue limitations under [0.1.0] ### Added (#571)
- ferray-core: invert `no_std` feature semantics (now opt-in `std`); fix `const_shapes` bitrot; cargo fmt
- Bump all crates to 0.2.10
- Cargo deps updated via dependabot group bump
- ferray-ufunc: fix Rust 1.95 clippy lints in test code — `arr1(&[...])` → `arr1([...])` (×31), inconsistent hex grouping (#570)
- ferray-ma: fix Rust 1.95 clippy lints — needless `Ok(...)?` and consolidated generic bounds (test code)
- ferray umbrella: silence `clippy::assertions_on_constants` on intentional re-export-accessibility tests

### Fixed
- ferray-stats: corrcoef returns 0.0 for constant-variable rows (numpy returns NaN) (#723)
- ferray-stats::correlate computes convolution instead of cross-correlation (numpy mismatch) (#722)
- ferray-ufunc: no subnormal handling in fast_exp — flushes to zero below -708.4 (#401)
- ferray-core-macros: no test for FerrayRecord with generics (#326)
- ferray-numpy-interop: no tests for large arrays (#313)
- ferray-numpy-interop: no property/fuzz tests (#312)
- ferray-autodiff: no second-order derivative (nested dual) tests (#305)
- ferray-autodiff: no f32 property tests (#304)
- ferray-autodiff: no edge-case tests at domain boundaries (#303)
- ferray-strings: oracle tests don't cover alignment/concat/regex ops (#284)
- ferray-strings: no tests for split with empty separator (#283)
- ferray-strings: no tests for empty arrays (zero elements) (#282)
- ferray-ma: data_mut() allows breaking shape invariant (#273)
- ferray-ma: mean() unwrap_or(one) could silently return sum instead of mean (#267)
- ferray-ma: masked_inside/masked_outside don't swap v1>v2 (#266)
- ferray-polynomial: no tests for fit_weighted with non-uniform weights (#255)
- ferray-polynomial: no tests for divmod of non-power basis (#254)
- ferray-polynomial: no tests for Chebyshev integ() accuracy (#253)
- ferray-polynomial: no NaN/Inf input validation (#252)
- ferray-polynomial: no tests for large/small coefficient magnitudes (#251)
- ferray-polynomial: no tests for polynomials with repeated roots (#250)
- ferray-io: test cleanup relies on remove_file, not RAII (#244)
- ferray-io: no memmap tests with multidimensional arrays (#243)
- ferray-io: no property tests for u16/u32/u64/i16/u128/i128 roundtrips (#242)
- ferray-io: no complex type roundtrip through save/load_dynamic (#241)
- ferray-io: no test for .npy v2.0 headers (#240)
- ferray-io: MemmapArray Send/Sync impl is fragile (#238)
- ferray-linalg: no tests for eig (non-symmetric) correctness (#221)
- ferray-linalg: no property tests for Cholesky or LU (#215)
- ferray-linalg: no f32 oracle tests (#213)
- ferray-stats: no tests for digitize with decreasing bins (#187)
- ferray-stats: no f32 tests anywhere (#185)
- ferray-stats: no tests for cov/corrcoef with single observation (#183)
- ferray-stats: no edge-case tests for correlate (#181)
- ferray-stats: no tests for single-element var/std with different ddof (#178)
- ferray-stats: no tests for infinity values in any statistical function (#177)
- ferray-ufunc: no f32 tests for most ufuncs (#152)
- ferray-core: Shape6 missing Debug derive breaks --all-features (#568)
- ferray-numpy-interop: preserve F-contiguous layout through IntoNumPy (asfortranarray gap) (#698)
- (none worth a separate entry — see the auto-generated history below for prior fixes)

## [0.1.0 .. 0.2.10] - 2026-03-07 .. 2026-04-28

> Auto-generated rolling block from closed-issue titles. Phase callouts and
> performance notes were curated by hand at the original 0.1.0 release.

### Added
- ferray-strings: StringArray is completely separate from Array — no trait interop (#514)
- ferray-linalg: no complex matrix support — LinalgFloat is sealed to f32/f64 (#404)
- ferray-ufunc: no vectorized math library for transcendentals (sleef/svml equivalent) (#393)
- ferray-ufunc: no ufunc object — functions are standalone, not composable abstractions (#376)
- ferray-numpy-interop: only supports 1D arrays for Arrow and Polars (#549)
- ferray-ufunc: no mixed-type operations (type promotion at ufunc level) (#380)
- ferray-autodiff: no integration with ferray-core — DualNumber can't be stored in Array (#534)
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
- ferray-ufunc: clippy errors (needless_borrows_for_generic_args, inconsistent_digit_grouping) on Rust 1.95 (#570)
- ferray-autodiff: god file — 1866 lines in single lib.rs (#535)
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
- ferray-ufunc: SIMD dispatch via pulp doesn't actually use SIMD for transcendentals (#377)
- ferray-strings: operations return flat StringArray — don't preserve higher-D shape correctly (#520)
- ferray-core-macros: no compile-fail tests for macro error paths (#210)
- ferray-numpy-interop: IntoNumPy creates flat PyArray1 then reshapes — extra allocation (#550)
- ferray-numpy-interop: AsFerray always copies — doc claims zero-copy borrow but implementation copies (#548)
- ferray-numpy-interop: to_arrow always copies — claims zero-copy but never achieves it (#547)
- ferray-test-oracle: fixtures exist without corresponding oracle tests (#205)
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
