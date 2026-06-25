# NumPy Parity Audit

Date: 2026-06-25

This audit records the current areas where Ferray does not have full NumPy
parity, plus the areas where parity is intentionally different or not directly
proved by a NumPy fixture yet.

## Reference Snapshot

- Local NumPy oracle: `numpy 2.4.5`
- Local Ferray Python package: `ferray 0.4.1`
- Workspace version metadata: `0.5.0`
- Python boundary: `ferray-python/python/ferray/__init__.py`
- Rust conformance inventories: `ferray-*/tests/conformance/_surface.json`
- Rust conformance exclusions: `ferray-*/tests/conformance/_surface_exclusions.toml`
- Rust accepted divergences: `ferray-*/tests/conformance/_divergences.toml`

Verification run during this audit:

```bash
PYTHONPATH=ferray-python/python python3 - <<'PY'
import ferray as fr, numpy as np
print(fr.__version__)
print(np.__version__)
PY

cd ferray-python
PYTHONPATH=python python3 -m pytest tests/divergence_*.py tests/test_divergence_*.py -q --tb=short --disable-warnings
PYTHONPATH=python python3 -m pytest tests/ -q --tb=short --disable-warnings

cargo test --all
cargo test --all --test conformance_surface_coverage

python3 -c "import tomllib; print(len(tomllib.load(open('tooling/translate-routes.toml','rb'))['route']))"
grep -l "## REQ status" $(python3 -c "import tomllib; [print(r['crate_pattern']) for r in tomllib.load(open('tooling/translate-routes.toml','rb'))['route']]") 2>/dev/null | wc -l
```

Observed results:

- Python divergence suite: `909 passed, 2 warnings`.
- Full Python suite: `5277 passed, 12 warnings`.
- Full Rust workspace suite: `cargo test --all` passed.
- Rust conformance surface coverage: all 14 coverage test binaries passed.
- Focused FFT package suite after adding direct wrapper and real/Hermitian
  coverage:
  `cargo test -p ferray-fft` passed.
- Focused ufunc package suite after adding direct `_into`, operator-wrapper,
  gap-function, first-class ufunc object/method, integer `gcd`/`lcm`, special
  `i0`, cumulative/difference/integration, interpolation, convolution, `unwrap`,
  `exp_fast`, feature-gated f16 numeric entry points, datetime/timedelta
  arithmetic and `isnat` coverage, SciPy-backed `fftconvolve`, NumPy
  `isscalar` array-input semantics, NaN extrema reduction, arithmetic
  alias/reduction/promoted paths, logical, bitwise, float-intrinsic, complex
  accessor/predicate, complex transcendental, and errstate state-machine
  coverage:
  `cargo test -p ferray-ufunc` passed.
- Focused masked-array conformance suite after adding direct NumPy fixture
  coverage for axis reductions, variance/stddev reductions, and
  count/count_masked helpers, masked sorting, masking constructors, and
  masked arithmetic plus shape/index manipulation, masked ufunc-support
  wrappers, extras helper/fill-mask protocol functions, and extras
  numeric/shape methods plus set/membership/comparison/logical and functional
  helpers, plus MaskedArray core accessors/state, interop adapters, and
  feature-gated masked-array I/O:
  `cargo test -p ferray-ma --test conformance_ma` passed.
- Focused masked comparison-result fill coverage after adding typed fill-value
  metadata for Rust boolean comparison results and NumPy-delegated Python
  `ma.<comparison>` helpers:
  `cargo test -p ferray-ma --test conformance_ma masked_extras_set_logic_match_numpy`
  and
  `PYTHONPATH=python python3 -m pytest tests/test_expansion_ma_repr_fill.py::test_module_comparison_preserves_materialized_numeric_fill -q`
  passed.
- Focused random package suite after adding direct distribution/permutation
  conformance evidence and fixing the logarithmic-series sampler:
  `cargo test -p ferray-random` passed.
- Focused NumPy-interop conformance after clearing bridge-trait exclusions:
  `cargo test -p ferray-numpy-interop`,
  `cargo test -p ferray-numpy-interop --features python --test conformance_interop numpy_traits_roundtrip_path`,
  `cargo test -p ferray-numpy-interop --features arrow --test conformance_interop`,
  and
  `cargo test -p ferray-numpy-interop --features polars --test conformance_interop`
  passed.
- Focused IO conformance after clearing datasource/memmap method exclusions:
  `cargo test -p ferray-io --test conformance_surface_coverage` and
  `cargo test -p ferray-io` passed.
- Focused window conformance after clearing inner-path window/functional
  exclusions: `cargo test -p ferray-window --test conformance_surface_coverage`
  and `cargo test -p ferray-window` passed.
- Focused FFT conformance after clearing plan/helper/type-system/extension
  exclusions: `cargo test -p ferray-fft --test conformance_surface_coverage`
  and `cargo test -p ferray-fft` passed.
- Focused autodiff conformance after clearing dual-number method/re-export
  exclusions: `cargo test -p ferray-autodiff --test conformance_surface_coverage`
  and `cargo test -p ferray-autodiff` passed.
- Focused stats conformance after clearing crate-root re-export exclusions:
  `cargo test -p ferray-stats --test conformance_surface_coverage` and
  `cargo test -p ferray-stats` passed.
- Focused ufunc conformance after clearing internal dispatch/helper/kernel
  exclusions: `cargo test -p ferray-ufunc --test conformance_surface_coverage`
  and `cargo test -p ferray-ufunc` passed.
- Focused ufunc errstate integration after wiring numeric kernels into the
  floating-point event state machine:
  `cargo test -p ferray-ufunc --test conformance_errstate` and
  `cargo test -p ferray-ufunc` passed.
- Focused polynomial conformance after clearing crate-root re-export,
  canonical constructor/method, f32, and complex polynomial exclusions:
  `cargo test -p ferray-polynomial --test conformance_surface_coverage` and
  `cargo test -p ferray-polynomial` passed.
- Focused linalg conformance after clearing root re-export, batch/faer bridge,
  decomposition, solve/norm/product, complex, TRSM, and internal GEMM surface
  exclusions: `cargo test -p ferray-linalg --test conformance_surface_coverage`
  and `cargo test -p ferray-linalg` passed.
- Focused core conformance after clearing array ownership/view, dtype/time,
  dynamic-array, indexing, iterator, broadcast, op, writeback, record, and
  feature-gated anchor exclusions: `cargo test -p ferray-core --test
  conformance_surface_coverage` and `cargo test -p ferray-core --test
  conformance_internal_surface` passed.
- Focused stride-tricks conformance after wiring signed raw-view construction
  through ferray-core: `cargo test -p ferray-stride-tricks as_strided_signed`,
  `cargo test -p ferray-stride-tricks --test conformance_stride_tricks`, and
  `cargo test -p ferray-core --test conformance_internal_surface` passed.
- Route/REQ-status mechanical count: `75` routed units and `75` files with
  `## REQ status`.

## Current Public-Name Gaps

These are names NumPy advertises in `__all__` where Ferray currently has no
corresponding attribute. The refreshed machine-readable copy is
`tooling/numpy-gap.json`.

Current status: no remaining public-attribute gaps in the tracked namespaces.
The namespaces checked are `ferray`, `ferray.linalg`, `ferray.fft`,
`ferray.random`, `ferray.ma`, `ferray.char`, `ferray.strings`,
`ferray.polynomial`, `ferray.lib`, `ferray.emath`, and `ferray.rec`.

## Export-List Gaps

These attributes exist but are omitted from Ferray's `__all__`, so star import
behavior still differs from NumPy:

Current status: no remaining `__all__` gaps in the tracked namespaces.

## Behavioral Gaps Not Caught By Name Diff

Current status: the behavior gaps identified by the namespace audit are closed.
`ferray.test` now mirrors `numpy.test`, and the formerly missing constants,
index helpers, and shared modules are exposed.

## Accepted Non-NumPy Differences

These are not current bugs; they are explicitly documented divergences where
Ferray claims the reference result is less mathematically accurate.

| Crate | Ferray item | Reference | Difference |
|---|---|---|---|
| `ferray-polynomial` | `ferray_polynomial::Polynomial::fit` | `numpy.polyfit` | For a small degree-1 least-squares input, Ferray's QR path returns an intercept closer to the exact analytic value than NumPy's SVD/lstsq path. Documented in `ferray-polynomial/tests/conformance/_divergences.toml`. |
| `ferray-window` | `ferray_window::taylor` | `scipy.signal.windows.taylor` | For even `M`, Ferray normalizes by the analytic Taylor-window peak instead of SciPy's midpoint/secant approximation. Documented in `ferray-window/tests/conformance/_divergences.toml`. |

No other Rust crate currently has a non-empty `_divergences.toml`.

## Conformance-Evidence Gaps

The Rust conformance surface gates pass, meaning each inventoried public Rust
item has a conformance test, an exclusion, or a documented divergence. Some
exclusions are still fixture/evidence debt rather than proof of direct NumPy
parity.

Current Rust direct-or-accepted surface evidence: 2,951 of 2,951 inventoried
items, or 100.0%. No inventoried Rust public items are still represented by
exclusion entries rather than direct conformance tests or accepted divergences.

| Area | Evidence debt |
|---|---|
| `ferray-core` | No exclusion entries remain. Existing fixture-backed and oracle conformance covers core creation, manipulation, indexing, dtype, reduction, casting, and property surfaces. Additional internal-surface conformance now directly exercises array ownership/view aliases and methods, raw buffer descriptors, signed raw-view construction, print options, dtype predicates, datetime/timedelta unit semantics, finfo/iinfo, promoted and explicit casts, `I256`, `DynArray`, uninitialized array construction, error constructors, record descriptors, basic and advanced indexing methods, iterator/reduction/sort methods, broadcasting and in-place/copy operations, `NdIter`, and writeback guard semantics while explicitly anchoring feature-gated const-shape and chrono bridge paths. |
| `ferray-ufunc` | No exclusion entries remain. Direct NumPy/SciPy-fixture-backed tests cover the `_into` arithmetic/trig/exp-log wrappers, operator-style arithmetic and bitwise wrappers (`array_add`, `array_sub`, `array_mul`, `array_div`, `array_rem`, `array_neg`, `array_bitand`, `array_bitor`, `array_bitxor`, `array_bitnot`, `array_shl`, `array_shr`), first-class ufunc objects and generic ufunc methods (`Ufunc`, `add_ufunc`, `subtract_ufunc`, `multiply_ufunc`, `divide_ufunc`, `reduce_all`, `reduce_axis`, `reduce_axis_keepdims`, `reduce_axes`, `accumulate_axis`, `ufunc_outer`, `at`), arithmetic broadcast aliases, promoted arithmetic paths, add/multiply reduction and NaN-reduction paths, `gcd`/`lcm` float-domain rejection plus `gcd_int`/`lcm_int` integer loops, special `i0` plus its scalar kernel, `around`, `fabs`, `floor_divide`, `fmod`, `hypot`, degree/radian conversions, `logaddexp`, `logaddexp2`, `positive`, `sign`, `true_divide`, `divmod`, interpolation (`interp`, `interp_one`), direct convolution (`ConvolveMode`, `convolve`), SciPy-backed FFT convolution (`fftconvolve`), `unwrap`, `exp_fast`, feature-gated direct f16 coverage for all crate-root f16 arithmetic, exp/log, float-intrinsic, rounding, special, and trig entry points plus canonical `ops::floatintrinsic::*_f16` paths, datetime/timedelta arithmetic and predicates (`isnat_datetime`, `isnat_timedelta`, `sub_datetime`, `add_datetime_timedelta`, `sub_datetime_timedelta`, `add_timedelta`, `sub_timedelta`, `sub_datetime_promoted`, `add_datetime_timedelta_promoted`, `add_timedelta_promoted`), NumPy `isscalar` array-input semantics, NaN extrema reductions (`nan_max_reduce`, `nan_max_reduce_all`, `nan_max_reduce_axes`, `nan_min_reduce`, `nan_min_reduce_all`, `nan_min_reduce_axes`), cumulative/difference/integration helpers (`cumsum`, `cumprod`, `cumulative_sum`, `cumulative_prod`, `nancumsum`, `nancumprod`, `diff`, `ediff1d`, `gradient`, `cross`, `trapezoid`), float manipulation/intrinsic helpers (`clip_ord`, `copysign`, `float_power`, `frexp`, `ldexp`, `modf`, `nextafter`, `spacing`, `isneginf`, `isposinf`, `signbit`), six comparison predicates plus their broadcast variants, `isclose`, `isclose_broadcast`, `allclose`, `array_equal`, `array_equiv`, logical operators, logical scalar and axis reductions, bitwise operators, shifts, `invert`, `bitwise_count`, complex accessors/predicates (`real`, `imag`, `angle`, `abs`, `conj`, `conjugate`, `iscomplex`, `isreal`, `iscomplexobj`, `isrealobj`, and real-input variants), complex transcendental/power functions, errstate state-management APIs (`geterr`, `seterr`, `with_errstate`, `ErrstateGuard`, `FpErrorClass`, `FpErrorState`, `record_fp_event`, `take_fp_events`, and `check_fp_errors`), and errstate event emission from arithmetic/promoted arithmetic, exp/log, trig domain/range, and float-intrinsic kernels. Additional conformance now directly exercises the public internal dispatch/helper/fast-math/kernel surface, with the cfg-test-only `test_util::arr1` explicitly anchored as non-runtime helper surface. |
| `ferray-polynomial` | No exclusion entries remain. Existing fixture-backed and inline conformance tests cover power-basis evaluation, fitting, roots, arithmetic, calculus, mapping, conversion, companion matrices, Vandermonde/least-squares helpers, polynomial extras, f32 power and wrapper types, and complex polynomial operations. Additional root-surface conformance now directly exercises crate-root type/trait/function re-exports, canonical per-basis constructor/domain/window/fit methods, power-basis inherent helpers, f32 method paths, complex method paths, and `traits::ToPowerBasis`/`traits::FromPowerBasis`; the accepted `Polynomial::fit` QR-vs-SVD precision divergence remains documented. |
| `ferray-linalg` | No exclusion entries remain. Existing fixture-backed and oracle conformance covers matrix products, norms/measures, decompositions, solvers, inverse/pseudoinverse, rank/condition/determinant paths, f32 variants, and documented divergence pins. Additional internal-surface conformance now directly exercises root aliases, batch dispatch helpers, faer bridge conversions, product/norm/solve/decomposition aliases, complex matrix operations, einsum parser/optimizer types, portable GEMM dispatch checks, and TRSM primitives while explicitly anchoring architecture-, feature-, and packing-specific GEMM/f16/openblas implementation paths as the matmul/outer implementation surface they support. |
| `ferray-ma` | No `#754` exclusion entries remain in the current manifest. Direct fixture-backed tests now cover flat masked reductions (`sum`, `mean`, `min`, `max`), axis masked reductions (`sum_axis`, `mean_axis`, `min_axis`, `max_axis`) including output-mask comparison for all-masked lanes, masked variance/stddev (`var`, `var_ddof`, `var_axis`, `var_axis_ddof`, `std`, `std_ddof`, `std_axis`, `std_axis_ddof`), count/count_axis, count_masked/count_masked_axis, masked arithmetic (`masked_add`, `masked_sub`, `masked_mul`, `masked_div` and regular-array variants) including NumPy division-by-zero masking and masked payload behavior, masked manipulation/shape-indexing (`reshape`, `ravel`, `flatten`, `transpose`, `t`, `squeeze`, `boolean_index`, `take`, `get_flat`) including NumPy fill-value behavior, masked ufunc-support wrappers (`masked_unary`, `masked_binary`, domain helpers, unary trig/exp/log/rounding/arithmetic wrappers, and binary `add`/`subtract`/`multiply`/`divide`/`power`) including `numpy.ma` payload/mask behavior and NumPy ufunc-only fallback payload behavior, extras helper constants and mask helpers (`NOMASK`, `default_fill_value_*`, `maximum_fill_value`, `minimum_fill_value`, `mask_or`, `make_mask`, `make_mask_none`, `masked_all`, `masked_all_like`, `masked_values`, `getmaskarray`, `is_masked`, `is_ma`, `is_masked_array`, and `filled_default`) including NumPy fill-value behavior, extras numeric/shape methods (`prod`, `cumsum_flat`, `cumprod_flat`, `argmin`, `argmax`, `ptp`, `median`, `average`, `anom`, `clip`, `repeat`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `expand_dims`, `diagonal`, `ma_dot_flat`, and `trace`) including NumPy masked payload and fill-value behavior, extras set/membership/comparison/logical helpers (`ma_unique`, `ma_vander`, `ma_in1d`, `ma_isin`, `ma_equal`, `ma_not_equal`, `ma_less`, `ma_greater`, `ma_less_equal`, `ma_greater_equal`, `ma_logical_and`, `ma_logical_or`, `ma_logical_xor`, and `ma_logical_not`) including masked-slot payload behavior, typed comparison-result fill metadata, and NumPy's unique/stable-sort membership behavior, extras functional helpers (`ma_concatenate`, `ma_apply_along_axis`, and `ma_apply_over_axes`) including NumPy axis/output-shape behavior, flat masked sorting (`sort`, `argsort`), axis masked sorting (`sort_axis`), masking constructors (`fix_invalid`, `masked_where`, comparison constructors, `masked_inside`, `masked_outside`) including NumPy fill-value behavior, MaskedArray core accessors/state, mask hardening/softening, interop adapters (`MaskAware`, `apply_unary`, `apply_unary_to`, `apply_binary`, `ma_apply_unary`, `into_data`), feature-gated `.npy` pair I/O (`save_masked`, `load_masked`), mask/data accessors, masked-invalid construction, `compressed`, and `filled`. |
| `ferray-fft` | No exclusion entries remain. Direct NumPy-fixture-backed tests cover `fftn` / `ifftn`, real-input full-spectrum wrappers, 1-D preallocated `_into` wrappers, `rfft2` / `rfftn` / `irfft2` / `irfftn`, and NumPy-backed 1-D Hermitian `hfft` / `ihfft`; additional conformance tests now cover the non-NumPy plan API, normalization/type-system surface, private axis/N-D helper paths through public negative-axis and N-D calls, and Ferray/Torch-style N-D Hermitian helpers (`hfft2` / `ihfft2` / `hfftn` / `ihfftn`). |
| `ferray-io` | No exclusion entries remain. Direct conformance now covers the `DataSource` path-resolution/open helpers and read-only/read-write `MemmapArray`/`MemmapArrayMut` shape, slice, view, copy, mutation, and flush paths alongside the existing `.npy`, `.npz`, text I/O, dtype, header, and format helper coverage. |
| `ferray-numpy-interop` | No exclusion entries remain. Direct conformance now covers the Python `AsFerray`/`IntoNumPy`/`NpElement` path under the `python` feature, Arrow marker and conversion traits under the `arrow` feature, and Polars marker and conversion traits under the `polars` feature. |
| `ferray-random` | No exclusion entries remain, down from 106 before the random-core and distribution/permutation evidence slices. Direct conformance tests now read `fixtures/random/distribution_moments.json` for the covered moment checks and directly cover the core `Generator`/`BitGenerator`/`SeedSequence` state surface, top-level and module re-exports, deterministic seeded construction, non-deterministic constructor sanity, state byte round-trips, byte draws, bounded integer draws, child spawning, uniform `random`/`random_f32`/`random_into`/`uniform_array`/`uniform_f32`, normal `normal`/`normal_f32`/`normal_array`/`lognormal`/`lognormal_f32`/`standard_normal_f32`/`standard_normal_into`/`standard_normal_parallel`, exponential `standard_exponential`/`standard_exponential_f32`/`standard_exponential_into`/`exponential_array`/`exponential_f32`, Poisson/binomial distribution moments, remaining discrete distributions (`geometric`, `hypergeometric`, `logseries`, `negative_binomial`, `poisson_array`, `zipf`), gamma-family helpers, miscellaneous continuous distributions, multivariate distribution shape/sum contracts, dynamic/range permutation helpers, concrete bit-generator state contracts, `SeedSequence`, and the internal ziggurat kernels through the public standard-normal paths. The added `logseries` evidence found and fixed an all-ones sampler bug. Random conformance remains distributional, not sample-exact, because Ferray's modern default bit-generator stream is not byte-identical to NumPy's modern `Generator` stream. |
| `ferray-stride-tricks` | No exclusion entries remain. Direct fixture-backed tests still cover NumPy's `as_strided`, `broadcast_to`, `broadcast_arrays`, `broadcast_shapes`, and `sliding_window_view`, and direct conformance smoke tests now cover the `StridedSource` trait plus signed positive/negative stride APIs and overlap-skipping variants. |
| `ferray-window` | No exclusion entries remain. Direct fixture-backed tests now cover crate-root and canonical `windows::*` paths for NumPy/SciPy window functions, direct functional utility paths for `vectorize`, `piecewise`, `apply_along_axis`, `apply_over_axes`, and `sum_axis_keepdims`, and the accepted `taylor` analytic-peak divergence remains documented. |
| `ferray-autodiff` | No exclusion entries remain. Existing analytic fixture tests cover `derivative`, `gradient`, `jacobian`, array-aware autodiff helpers, and free `atan2`; additional conformance checks now cover the `DualNumber` canonical path, constructors, crate-root scalar-function re-exports, and every differentiable `DualNumber` method against its analytic forward-mode value/dual rule. |
| `ferray-stats` | No exclusion entries remain. Existing fixture-backed and inline conformance tests cover reductions, NaN-aware reductions, quantiles, sorting, searching, set operations, histograms, correlation/covariance, descriptive statistics, and hypothesis tests through canonical inner paths; additional root-surface conformance now compile-checks every crate-root re-export alias and compares representative root aliases to their canonical implementations. |

The Rust surface manifests no longer carry exclusion-based evidence debt. This
audit no longer records an explicit behavior gap in the tracked surface;
accepted divergences remain listed above where Ferray intentionally differs.

## Documentation And Tooling Inaccuracies

- `README.md` says "Full NumPy API surface". The public Python namespace and
  `__all__` gaps from this audit are now closed, and the Rust manifests no
  longer carry exclusion-based evidence debt. The README should still link to
  current audit evidence rather than relying on the slogan alone.
- `README.md` also says the `[0.3.5]` release swept the open NumPy-parity
  backlog to zero. That is historical release text and should not be read as a
  claim that every Rust public item has a direct NumPy fixture today.
- `docs/conformance-suites.md` describes the committed JSON fixture reference
  as `numpy 2.4.5`; current fixture metadata confirms that pin across all 382
  JSON fixtures. The ufunc `fftconvolve` fixture additionally records
  `scipy 1.17.1` as its reference library. The live Python divergence tests in
  this workspace currently run against installed `numpy 2.4.5`, so audit notes
  should distinguish fixture-pin evidence from live-oracle evidence.
- `tooling/numpy-gap.json` was stale before this audit. It has been refreshed
  to the current `numpy.__all__` advertised-name check and currently records
  no tracked namespace gaps.

## Current Bottom Line

The current checked-in divergence tests do not expose open behavioral
inaccuracies: all 909 historical divergence pins now pass. The tracked Python
namespace/export gaps are closed. Conformance surface debt is closed for
autodiff, core, FFT, linalg, polynomial, stats, stride-tricks, IO, ufunc, and window. Core array
ownership/view aliases, dtype/time surfaces, dynamic arrays, indexing,
iterators, broadcast operations, writeback guards, record descriptors, and
feature-gated anchors have reduced `ferray-core` exclusion debt to zero. Direct ufunc wrapper,
operator-wrapper,
first-class ufunc object/method,
gap-function, arithmetic alias/reduction/promoted paths, integer `gcd`/`lcm`,
special `i0`, interpolation, direct convolution, `unwrap`, `exp_fast`,
feature-gated f16 numeric entry points, NaN extrema reduction,
cumulative/difference/integration, datetime/timedelta arithmetic and predicate
coverage, SciPy-backed `fftconvolve`, NumPy `isscalar` array-input semantics,
float-intrinsic, logical, bitwise, complex accessor/predicate, complex
transcendental/power, errstate state-machine and kernel-event coverage, and internal
dispatch/helper/kernel surface coverage have reduced `ferray-ufunc` exclusion
debt to zero; polynomial root aliases, canonical method paths, f32 power
methods, and complex polynomial methods have reduced `ferray-polynomial`
exclusion debt to zero; linalg root aliases, batch/faer helpers, decompositions,
solve/norm/product aliases, complex paths, TRSM primitives, and internal GEMM
implementation anchors have reduced `ferray-linalg` exclusion debt to zero;
random core generator,
bit-generator, seed-sequence, uniform/normal/exponential variants,
Poisson/binomial moments, remaining discrete/gamma-family/miscellaneous
continuous and multivariate distributions, and dynamic/range permutation
helpers have reduced `ferray-random` exclusion debt to zero; masked-array axis
reductions, variance/stddev reductions, count/count_masked helpers, sorting,
masking constructors, shape/index manipulation, masked ufunc-support wrappers,
extras helper/fill-mask protocol functions, extras numeric/shape methods, and
extras set/membership/comparison/logical and functional helpers now have direct
NumPy fixture coverage, and MaskedArray core/interop/I/O surface evidence is
closed;
masked arithmetic now follows NumPy's division-by-zero masking and masked
payload behavior, masked `take` now follows NumPy's fill-value reset behavior,
`masked_values` now preserves NumPy's compared-value fill value, masked extras
fill helpers now follow NumPy's bool/default and min/max fill-value semantics,
`anom` now preserves masked payloads and source fill values, `clip`, `repeat`,
`expand_dims`, and `diagonal` now preserve source fill values,
`ma_unique` now returns NumPy's masked unique result directly, logical helpers
now preserve NumPy masked-slot payloads,
membership helpers now follow NumPy's unique/stable-sort masked behavior,
comparison helpers now preserve NumPy's left-truthiness masked payloads and
typed fill-value metadata,
and `ma_apply_over_axes` now preserves rank the way NumPy does,
and masked ufunc-support wrappers now follow `numpy.ma` domain/payload semantics
plus NumPy ufunc-only fallback payload semantics;
nanmin/nanmax all-NaN slices now return NaN like NumPy, signed
`bitwise_count` follows NumPy's absolute-value semantics, signed stride-tricks
views now support negative strides without materializing copies, and IO
datasource/memmap method coverage has reduced `ferray-io` exclusion debt to
zero; window inner-path fixture and functional utility coverage has reduced
`ferray-window` exclusion debt to zero. The audit now records no remaining
tracked namespace/export, Rust conformance-surface, or explicit ufunc errstate
gap.
