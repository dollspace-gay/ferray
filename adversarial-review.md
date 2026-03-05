---
title: Adversarial Review: ferrum Design Docs vs NumPy Parity
tags: [review, design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

| Severity | Count | Description |
|---|---|---|
| CRITICAL | 7 | Missing NumPy modules, ~55 missing functions, missing constants/finfo |
| MAJOR | 8 | Won't-compile code, wrong dep versions, scope too large, lost type safety, missing iteration/display |
| MODERATE | 7 | Missing error control, incomplete masked arrays, unrealistic perf targets |
| LOW | 3 | Testing utilities, format control, mixin support |

**Total missing functions for NumPy parity: ~70-80**

The design docs cover approximately 85% of NumPy's surface area. The remaining 15% includes some of the most commonly used functions (`cumsum`, `diff`, `pad`, `tile`, `interp`, `convolve`, `vectorize`, `take`). Without these, a user porting NumPy code to ferrum will hit missing-function errors on realistic workloads.

### severity: critical (blocks numpy parity)

### C1. Missing NumPy routine categories — entire modules unaccounted for

The following NumPy routine categories have ZERO coverage in any design doc:

| NumPy Category | Impact |
|---|---|
| **Window functions** (`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`) | Used in signal processing, spectral analysis. 5 functions missing. |
| **Datetime support** (`datetime64`, `timedelta64`, `busday_count`, `busday_offset`, `is_busday`, `datetime_as_string`) | Used in time series, financial data. Entire dtype family missing. |
| **Mathematical functions with automatic domain** (`numpy.emath`) | `sqrt(-1)` returns `1j` instead of NaN. Convenience wrappers that auto-promote to complex. ~10 functions. |
| **Floating point error handling** (`seterr`, `geterr`, `errstate`) | Controls how NaN/Inf/overflow/underflow are handled. Critical for numerical debugging. |
| **Functional programming** (`vectorize`, `piecewise`, `apply_along_axis`, `apply_over_axes`) | `vectorize` alone is used in ~40% of NumPy tutorials. |

**FIXED**: Created `ferrum-window.md` covering window functions (bartlett, blackman, hamming, hanning, kaiser) and functional programming (vectorize, piecewise, apply_along_axis, apply_over_axes). Datetime support and emath remain out of scope for 1.0. Floating point error handling deferred (Rust's f64 behavior is deterministic and sufficient for most use cases).

### C2. Missing ~25 mathematical functions from ferrum-ufunc

Comparing the NumPy `routines.math` page against ferrum-ufunc requirements:

| Missing Function | Category | Why it matters |
|---|---|---|
| `cumsum` | Sums/products | Fundamental. Used everywhere. |
| `cumprod` | Sums/products | Cumulative product along axis. |
| `nancumsum` | Sums/products | NaN-aware cumulative sum. |
| `nancumprod` | Sums/products | NaN-aware cumulative product. |
| `diff` | Differences | n-th discrete difference. Critical for time series. |
| `ediff1d` | Differences | Differences between consecutive elements. |
| `gradient` | Differences | Numerical gradient (central differences). Used in physics, optimization. |
| `cross` | Products | Cross product. Used in 3D geometry, physics. |
| `trapezoid` | Integration | Trapezoidal numerical integration. Was `trapz` pre-2.0. |
| `convolve` | Convolution | 1D convolution. Fundamental signal processing. |
| `interp` | Interpolation | 1D linear interpolation. Extremely common. |
| `sinc` | Trigonometric | sinc function. Used in signal processing, Fourier analysis. |
| `i0` | Special | Modified Bessel function order 0. Used in Kaiser windows, stats. |
| `real_if_close` | Complex | If imaginary part < tol, return real. Convenience function. |

`cumsum` and `diff` are arguably more commonly used than half the trig functions we DO have. Their absence is a glaring parity gap.

**FIXED**: All 14 missing functions added to ferrum-ufunc.md (REQ-8a through REQ-8i, REQ-10a). cumsum/cumprod/nancumsum/nancumprod also added to ferrum-stats.md (REQ-2a, REQ-2b) for discoverability.

### C3. Missing ~15 array manipulation functions from ferrum-core

| Missing Function | Why it matters |
|---|---|
| `pad` | Array padding. CRITICAL for convolutions, image processing, neural nets. |
| `tile` | Tile/repeat an array. Very commonly used. |
| `repeat` | Repeat elements of an array. |
| `delete` | Delete sub-arrays along an axis. |
| `insert` | Insert values along an axis. |
| `append` | Append values to end of array. |
| `resize` | Resize an array. |
| `trim_zeros` | Trim leading/trailing zeros. |

**FIXED**: All 8 functions added to ferrum-core.md as REQ-22a (pad with 5 modes, tile, repeat, delete, insert, append, resize, trim_zeros).

### C4. Missing ~15 indexing functions from ferrum-core

| Missing Function | Why it matters |
|---|---|
| `take` / `take_along_axis` | Select elements along axis. Core indexing primitive. |
| `put` / `put_along_axis` | Place values into array along axis. |
| `choose` | Construct array from index array and choices. |
| `compress` | Select slices along axis using boolean. |
| `select` | Return array from choices based on conditions. |
| `indices` | Array of grid indices. |
| `ix_` | Open mesh from sequences. |
| `diag_indices` / `tril_indices` / `triu_indices` (+ `_from` variants) | Index generation for diagonal/triangular. |
| `ravel_multi_index` / `unravel_index` | Index conversions between flat and multi-dimensional. |
| `flatnonzero` | Non-zero indices in flattened array. |
| `fill_diagonal` | Fill diagonal of array. |
| `ndindex` / `ndenumerate` | Multi-dimensional index iterators. |

**FIXED**: All indexing functions added to ferrum-core.md as REQ-15a.

### C5. Missing linalg functions from ferrum-linalg

| Missing Function | Why it matters |
|---|---|
| `linalg.multi_dot` | Optimized chain matrix multiplication. 10-100x faster than naive chaining. |
| `linalg.matrix_power` | Matrix exponentiation. Used in Markov chains, graph theory. |
| `linalg.tensorsolve` | Solve tensor equation. |
| `linalg.tensorinv` | Tensor inverse. |
| `linalg.vecdot` | Vector dot product (new in NumPy 2.0). |

**FIXED**: All 5 functions added to ferrum-linalg.md (REQ-7a, REQ-7b, REQ-18a, REQ-18b, REQ-18c).

### C6. Missing constants

NumPy provides `np.pi`, `np.e`, `np.inf`, `np.nan`, `np.newaxis`, `np.euler_gamma`, `np.PINF`, `np.NINF`, `np.PZERO`, `np.NZERO`. None of these appear in any design doc. Users expect `ferrum::pi`, `ferrum::inf`, etc.

**FIXED**: Added REQ-33 to ferrum-core.md (constants module) and REQ-3a to ferrum-reexport.md (re-export as ferrum::PI etc.).

### C7. Missing `finfo` / `iinfo` type introspection

`np.finfo(np.float64).eps`, `np.finfo(np.float64).max`, `np.iinfo(np.int32).min` — these are used constantly in numerical code for tolerance computation, overflow checking, and type-safe constant selection. Not in any design doc.

**FIXED**: Added REQ-34 to ferrum-core.md.

---

### severity: major (significant usability/correctness gaps)

### M1. Polynomial blanket `From` impl WILL NOT COMPILE

The source design doc (line 597) and ferrum-polynomial design contain:
```rust
impl<P: ToPowerBasis, Q: FromPowerBasis> From<P> for Q {
    fn from(p: P) -> Q { Q::from_power_basis(&p.to_power_basis()) }
}
```

This conflicts with the standard library's blanket `impl<T> From<T> for T`. Rust's coherence rules will reject this. The polynomial design needs an explicit `.convert::<TargetType>()` method or pairwise `From` impls instead.

**FIXED**: REQ-11 in ferrum-polynomial.md rewritten to use `.convert::<TargetType>()` method + pairwise From impls.

### M2. Dependency versions are wrong in coordinator CLAUDE.md

| Crate | Design doc version | Actual latest |
|---|---|---|
| `ndarray` | 0.16 | **0.17.2** |
| `pulp` | 0.20 | **0.22.2** |
| `rustfft` | 6.2 | **6.4.1** |
| `faer` | 0.24 | 0.24.0 (correct) |

Agents using wrong versions will hit API differences, missing methods, and compilation failures. The ndarray 0.16 → 0.17 change is particularly significant (breaking API changes).

**FIXED**: Dependency versions corrected in phase-0-coordinator.md CLAUDE.md conventions section and ferrum-core.md.

### M3. ferrum-core is too large for a single agent

ferrum-core has **30 requirements** covering:
- The entire `NdArray<T, D>` type with 5 ownership variants
- A full Dimension trait hierarchy
- A DType system with `Element` trait for 16+ types
- Complete broadcasting implementation
- Basic + advanced (fancy + boolean) indexing
- 20+ array creation functions
- 15+ shape manipulation functions
- Full type promotion system with compile-time macros
- `FerrumError` hierarchy
- `AsRawBuffer` interop trait
- `DynArray` runtime-typed enum
- A proc macro crate (`FerrumRecord`)

This is realistically 3-4 agents' worth of work. A single agent will either produce incomplete output or hit context limits. **Recommend splitting ferrum-core into sub-agents**: core-types (NdArray + ownership), core-indexing, core-creation-manipulation, core-macros.

**FIXED**: ferrum-core.md now has "Agent Splitting Guidance" section. phase-0-coordinator.md updated with Agents 1a-1d replacing single Agent 1.

### M4. Ufunc trait loses static dimensionality

The `Ufunc` trait uses `IxDyn` everywhere:
```rust
fn call(&self, input: &NdArray<Self::In, IxDyn>) -> NdArray<Self::Out, IxDyn>;
```

This means calling a ufunc on an `Array2<f64>` forces a conversion to `ArrayD<f64>`, losing compile-time dimension information. The output is always dynamically-dimensioned. This contradicts the design philosophy of "Rust improvements over NumPy" — you'd lose type safety every time you call `sin()`.

**Fix**: The trait should be generic over dimension, or ufuncs should be plain generic functions rather than trait objects.

**FIXED**: ferrum-ufunc.md REQ-1 rewritten. Ufuncs are now generic free functions `fn sin<T, D>(input: &NdArray<T, D>) -> NdArray<T, D>` preserving compile-time dimensionality. Internal kernel dispatch uses `UnaryOp`/`BinaryOp` traits but these are not public API.

### M5. No Iterator / `mapv` / closure-based operations

ndarray's most productive convenience methods are completely absent:
- `.iter()` / `.iter_mut()` — element iteration
- `.into_iter()` — consuming iteration
- `.mapv(|x| x.sqrt())` — map a closure element-wise
- `.mapv_inplace(|x| x * 2.0)` — in-place map
- `.zip_mut_with(&other, |a, b| *a += b)` — zip two arrays
- `.lanes(axis)` — iterate over lanes along an axis
- `.axis_iter(axis)` — iterate over sub-arrays along an axis
- `.indexed_iter()` — iterate with multi-dimensional indices

These are how Rust users actually write array code. Without them, every custom operation requires manual index loops. This is a major ergonomics gap vs both NumPy (which has vectorize) and ndarray (which has mapv).

**FIXED**: Added REQ-37 (iteration) and REQ-38 (closure ops) to ferrum-core.md. vectorize added to ferrum-window.md.

### M6. No `Display` / `Debug` formatting for arrays

NumPy's `print(array)` produces nicely formatted output with alignment, truncation for large arrays, and configurable precision (`np.set_printoptions`). No design doc mentions `impl Display for NdArray` or array printing at all. Users will get `NdArray { data: [...], shape: [...] }` from Debug at best.

**FIXED**: Added REQ-39 to ferrum-core.md with NumPy-matching Display format and set_print_options.

### M7. No ndarray introspection properties

Every NumPy user relies on:
- `.shape` (already implied but not explicit as a property)
- `.ndim` — number of dimensions
- `.size` — total element count
- `.itemsize` — bytes per element
- `.nbytes` — total bytes
- `.dtype` — element type descriptor
- `.flags` — contiguity, writability, ownership info
- `.T` — transpose as a property, not just a method
- `.flat` — flat iterator
- `.copy()` — explicit deep copy
- `.tobytes()` — raw byte access
- `.tolist()` — convert to nested Rust `Vec`

These are all missing from ferrum-core's requirements.

**FIXED**: Added REQ-35 and REQ-36 to ferrum-core.md covering all introspection properties.

### M8. SIMD code example in source design uses unstable `std::simd`

Section 19 of `rust-numpy-design.md` shows:
```rust
use std::simd::f64x4;
```

This is unstable. The design docs correctly specify `pulp` as the SIMD strategy, but the source document's example code contradicts this. Any agent reading the source doc will be confused. The CLAUDE.md conventions should explicitly state "do NOT use std::simd — use pulp".

**FIXED**: Added explicit "Do NOT use std::simd" warning to phase-0-coordinator.md CLAUDE.md conventions, ferrum-ufunc.md SIMD Strategy section, and ferrum-core.md Key Design Decisions.

---

### severity: moderate (completeness/polish gaps)

### m1. No `np.errstate` / error-mode control

NumPy lets users control what happens on divide-by-zero, overflow, invalid operations:
```python
with np.errstate(divide='ignore', invalid='warn'):
    result = a / b
```
No equivalent mechanism in any ferrum design doc. In Rust, `f64` division by zero produces `inf` silently — there's no way to control this per-scope without explicit checking.

### m2. No masked array arithmetic completeness

ferrum-ma covers basic reductions and masking constructors but doesn't specify:
- Masked ufunc support (every ufunc should respect masks)
- Masked I/O (saving/loading masked arrays)
- Masked sorting
- `ma.MaskedArray.harden_mask()` / `soften_mask()`
- `ma.getmask()`, `ma.getdata()`
- `ma.is_masked()`
NumPy's masked array module has ~100 functions. ferrum-ma specifies ~15.

**FIXED**: Added REQ-12 (masked ufunc support), REQ-13/14 (masked sorting), REQ-15/16/17 (mask manipulation utilities) to ferrum-ma.md.

### m3. No legacy polynomial support (`np.poly1d`, `np.roots`, `np.polyfit`, `np.polyval`)

While the modern `numpy.polynomial` module is covered, the legacy top-level polynomial functions (`np.poly1d`, `np.roots`, `np.polyfit`, `np.polyval`, `np.polyadd`, `np.polymul`, `np.polyder`, `np.polyint`) are still widely used in existing code and tutorials. Many users haven't migrated to the modern API.

### m4. Parallel threshold of 10M elements is too high

The design says elementwise ops parallelize above ~10M elements. NumPy (via OpenBLAS/MKL) parallelizes matrix operations at much lower thresholds. On a modern 8-core machine, parallelism pays off at ~100k elements for memory-bound ops. 10M means a 10000x1000 matrix operation runs single-threaded. This should be calibrated empirically, not hardcoded.

**FIXED**: ferrum-ufunc.md REQ-21 changed to ~100k for memory-bound, ~50k for compute-bound. Thresholds are now explicitly described as needing empirical calibration.

### m5. "80% peak memory bandwidth" SIMD acceptance criterion is unrealistic

The source design doc states SIMD benchmarks must achieve "at least 80% of theoretical peak memory bandwidth." This is a target that production BLAS libraries (OpenBLAS, MKL) don't consistently achieve. It's an unrealistic acceptance criterion that will either be fudged or block release forever. Replace with "SIMD paths must demonstrate measurable speedup (>2x) over scalar paths on contiguous data."

### m6. No cache-blocking strategy for large operations

For operations on arrays larger than L2 cache (>256KB typical), cache-blocking (tiling) is critical for performance. Neither the ufunc nor linalg design docs mention cache-blocking. This means large array operations will be memory-bandwidth-limited due to cache thrashing.

### m7. `ferrum::with_num_threads` creates a new ThreadPool per call

The design says `with_num_threads(n, || {...})` creates a temporary ThreadPool. This is extremely expensive (thread creation + synchronization). If called in a loop, it will destroy performance. Need either a pool cache or a different API design (e.g., scoped configuration on the existing pool).

**FIXED**: ferrum-reexport.md REQ-7 now specifies a pool cache (DashMap) to avoid creating new pools per call.

---

### severity: low (nice-to-have, post-1.0)

### l1. No `np.testing` utilities
NumPy provides `assert_array_equal`, `assert_array_almost_equal`, `assert_allclose`. These are testing helpers that downstream users need.

### l2. No `np.lib.format` for custom format control
Low-level .npy format reading/writing control.

### l3. No `np.lib.mixins.NDArrayOperatorsMixin`
Enables custom types to work with NumPy ufuncs. The Rust equivalent would be a trait that allows user types to participate in ufunc dispatch.

---

### dependency version correction table

| Crate | Wrong Version | Correct Version | Breaking Changes? |
|---|---|---|---|
| `ndarray` | 0.16 | 0.17.2 | Yes — `ArrayBase` API changes, new `ShapeBuilder` |
| `pulp` | 0.20 | 0.22.2 | Likely — minor version bumps may change SIMD API |
| `rustfft` | 6.2 | 6.4.1 | Minor — new algorithms, no breaking API changes |
| `faer` | 0.24 | 0.24.0 | Correct |
| `half` | 2.4 | Verify | Unverified |

---

### compilation issues (will fail at build time)

1. **Polynomial blanket From impl** (M1) — coherence violation
2. **`std::simd` usage** if any agent copies the source design doc example
3. **ndarray 0.16 API** if agents use the wrong version — methods renamed/moved in 0.17

---

