# Feature: ferray-stats — Statistical functions, sorting, searching, set operations, and histograms

## Summary
Implements all NumPy statistical functions as array methods and top-level free functions: reductions (sum, prod, min, max, mean, var, std, median, percentile, quantile), cumulative operations, NaN-aware variants, correlations, covariance, histograms, sorting, searching, unique, and set operations. These are the bread-and-butter data analysis functions that most users reach for after array creation.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Dimension, Element, FerrayError), `ferray-ufunc` (comparison ufuncs for sorting/searching)
- **Downstream**: ferray-linalg (uses reductions), ferray-random (may use stats for validation), ferray (re-export)
- **External crates**: `rayon` (parallel reductions), `num-traits`
- **Phase**: 1 — Core Array and Ufuncs

## Requirements

### Reductions (Section 13)
- REQ-1: Array methods: `sum()`, `prod()`, `min()`, `max()`, `argmin()`, `argmax()`, `mean()`, `var(ddof)`, `std(ddof)`, `median()`, `percentile(q)`, `quantile(q)` — all accept optional `axis` parameter. `axis: None` reduces over entire array.
- REQ-2: All reductions available as both array methods and top-level free functions

### Cumulative Operations
- REQ-2a: `cumsum(&a, axis)` and `cumprod(&a, axis)` — cumulative sum/product along axis (also in ferray-ufunc, but exposed here as array methods and free functions for discoverability)
- REQ-2b: `nancumsum(&a, axis)` and `nancumprod(&a, axis)` — NaN-aware cumulative operations (NaN treated as 0 for sum, 1 for product)

### NaN-Aware Variants
- REQ-3: `nansum`, `nanprod`, `nanmin`, `nanmax`, `nanmean`, `nanvar(ddof)`, `nanstd(ddof)`, `nanmedian`, `nanpercentile(q)` — skip NaN values in computation
- REQ-4: NaN-aware reductions on all-NaN slices must return a well-defined result (NaN for nanmean, 0 for nansum, matching NumPy)

### Correlations and Covariance
- REQ-5: `correlate(&a, &v, mode)` with modes Full, Same, Valid — matching `np.correlate`
- REQ-6: `corrcoef(&x, rowvar)` — Pearson correlation coefficient matrix
- REQ-7: `cov(&m, rowvar, ddof)` — covariance matrix

### Histograms
- REQ-8: `histogram(&a, bins, range, density)` returning `(counts, bin_edges)` — matching `np.histogram`
- REQ-9: `histogram2d(&x, &y, bins)` and `histogramdd(&sample, bins)` for multi-dimensional histograms
- REQ-10: `bincount(&x, weights, minlength)` and `digitize(&x, &bins, right)`

### Sorting and Searching
- REQ-11: `sort(axis, kind)` with `SortKind::Stable` (merge sort) and `SortKind::Quick` — matching `np.sort`
- REQ-12: `argsort(axis)` returning index array
- REQ-13: `searchsorted(&a, &v, side)` with `Side::Left` and `Side::Right`
- REQ-14: `unique(&a, return_index, return_counts)` returning sorted unique elements
- REQ-15: `nonzero(&a)` returning tuple of index arrays
- REQ-16: `where_(&condition, &x, &y)` — conditional selection matching `np.where`
- REQ-17: `count_nonzero(&a, axis)`

### Set Operations (Section 14)
- REQ-18: `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `in1d`, `isin` — all with `assume_unique` optimization flag

### Parallelism
- REQ-19: Large reductions (>10k elements) use parallel tree-reduce via Rayon on ferray's owned pool
- REQ-20: Sorting of large arrays (>100k elements) uses parallel merge sort via Rayon

### Python marshalling boundary (ferray-python shim)
- REQ-21: **Full reductions must return a numpy SCALAR, not a 0-d `ndarray`.** Across the `ferray-python` boundary, any reduction whose result is 0-dimensional — a full reduction (`axis=None`, or an inherently scalar-producing reduction like `np.sum([1,2,3])`) — must return a numpy *scalar* (`numpy.int64` / `numpy.float64` / `numpy.bool_` / `numpy.complex128`), NOT a 0-d `numpy.ndarray`, mirroring numpy's `$OUT_SCALAR` contract (numpy/\_core/code_generators/ufunc_docstrings.py:38-39 — `'OUT_SCALAR_1': "This is a scalar if \`x\` is a scalar."`). An axis reduction that leaves ≥1 dimension stays an `ndarray`. numpy realises this collapse structurally: the `fromnumeric` reduction wrappers return the bare `ufunc.reduce(...)` result (numpy/\_core/fromnumeric.py:83 `return ufunc.reduce(obj, axis, dtype, out, **passkwargs)` for `sum`/`prod`/`min`/`max`/`ptp`; :2539 / :2634 `_wrapreduction_any_all(...)` for `any`/`all`; `_methods._mean`/`_var`/`_std` at numpy/\_core/\_methods.py:115/148/217 and `_amin`/`_amax` at :43/:39; `median`/`average`/`percentile`/`quantile` at numpy/lib/\_function_base_impl.py:3915/:451/:4065/:4268; the `nan*` family at numpy/lib/\_nanfunctions_impl.py:635 `nansum`/:953 `nanmean`/:383 `nanmax`/:253 `nanmin`/:511 `nanargmin`/:572 `nanargmax`), and a 0-d ufunc/method reduce result is auto-boxed into a numpy scalar.
  - **In-scope reduction `#[pyfunction]`s** (full-reduction egress must collapse to a scalar): `sum`, `prod`, `min`, `max`, `ptp`, `mean`, `var`, `std`, `argmin`, `argmax`, `all`, `any`, `average`, `median`, `percentile`, `quantile`, `count_nonzero`, and the `nan*` family (`nansum`, `nanprod`, `nanmin`, `nanmax`, `nanmean`, `nanvar`, `nanstd`, `nanmedian`, `nanpercentile`, `nanargmin`, `nanargmax`) — including those generated by the `bind_nan_reduction!` macro.
  - **OUT of scope:** CUMULATIVE ops (`cumsum`, `cumprod`, `nancumsum`, `nancumprod`) and TUPLE-returning ops (`unique(return_index/return_counts)`, `histogram`, `histogram2d`, `histogramdd`, `nonzero`, `slogdet`, …) — these never produce a bare 0-d scalar.

## Acceptance Criteria
- [ ] AC-1: `a.sum(axis=None)` and `a.sum(axis=0)` produce results matching NumPy for 1D, 2D, and 3D arrays of f64
- [ ] AC-2: `a.mean()`, `a.var(ddof=0)`, `a.std(ddof=1)` match NumPy to within 4 ULPs on fixture data
- [ ] AC-3: NaN-aware variants skip NaN values correctly: `nanmean([1.0, NaN, 3.0]) == 2.0`
- [ ] AC-4: `histogram()` produces bin counts and edges matching `np.histogram` exactly for integer inputs
- [ ] AC-5: `sort()` with `SortKind::Stable` preserves relative order of equal elements
- [ ] AC-6: `unique()` with `return_counts=true` matches `np.unique(return_counts=True)`
- [ ] AC-7: Set operations: `union1d([1,2,3], [2,3,4]) == [1,2,3,4]`, etc.
- [ ] AC-8: `where_(&mask, &x, &y)` selects from x where mask is true, y otherwise
- [ ] AC-9: `cargo test -p ferray-stats` passes. `cargo clippy -p ferray-stats -- -D warnings` clean.
- [ ] AC-10: `cumsum(&[1,2,3,4], None)` returns `[1,3,6,10]`. `nancumsum(&[1.0, NaN, 3.0], None)` returns `[1.0, 1.0, 4.0]`.
- [ ] AC-11: `cumprod(&[1,2,3,4], None)` returns `[1,2,6,24]`.
- [x] AC-12 (REQ-21): `type(fr.sum([1,2,3])) is numpy.int64` (NOT `numpy.ndarray`); `isinstance(fr.sum([1,2,3]), numpy.ndarray) is False`; `type(fr.mean([1.,2,3])) is numpy.float64`; `type(fr.all([True,True])) is numpy.bool_`; while `fr.sum([[1,2],[3,4]], axis=0)` stays a 1-D `numpy.ndarray` (`array([4,6])`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-21 (reduction full-reduction returns numpy scalar, not 0-d ndarray) | SHIPPED (#992) | Every in-scope full-reduction (0-d) egress in `ferray-python/src/stats.rs` now routes through `conv::scalarize` (impl `ferray-python/src/conv.rs:1424` — `result.ndim == 0 -> result[()]`, no-op otherwise). Non-test production consumers (the `#[pyfunction]`s registered in `_ferray`): `pub fn sum` `stats.rs:1342` `crate::conv::scalarize(match_dtype_numeric!(…))`; `pub fn all` `stats.rs:1049` `crate::conv::scalarize(match_dtype_all!(…))`; `pub fn mean` `stats.rs:1551`; `pub fn argmax` `stats.rs:1907` `crate::conv::scalarize(u64_arrd_to_i64_pyarray(py, r)?)`; `pub fn median` `stats.rs:4452`; `pub fn average` `stats.rs:5015`; `pub fn count_nonzero` `stats.rs:3527`; the `bind_nan_reduction!` macro tails (`stats.rs:2088`/`2135`/`2187` + the int-native early-return `stats.rs:2171`) cover `nansum`/`nanmean`/`nanmin`/`nanmax`/`nanprod`; `nanmedian` `stats.rs:2859`; `nanargmin`/`nanargmax` `stats.rs:2889`/`2914`; scalar-`q` `percentile`/`quantile`/`nanpercentile`/`nanquantile` via the `Err(scalar) => crate::conv::scalarize(single(scalar)?)` arm (`stats.rs:4322`/`4874`) in `quantile_dispatch`/`nan_quantile_dispatch`. Complex-dispatch helpers collapse at their `axis=None` egress too (`complex_fold_dispatch`, `complex_mean_dispatch`, `complex_var_std_dispatch`, `complex_minmax_reduce`, `complex_argextremum_reduce`, `complex_ptp_reduce`, `complex_weighted_average`, `complex_median_dispatch`, `complex_nan_fold_dispatch`, `complex_nanmean_dispatch`, `complex_nan_minmax_dispatch`, `complex_nan_var_std_dispatch`, `complex_nanmedian_dispatch`). Live-verified `$OUT_SCALAR` parity: `type(fr.sum([1,2,3])) is numpy.int64`, `not isinstance(_, numpy.ndarray)`, `type(fr.mean([1.,2,3])) is numpy.float64`, `type(fr.all([True,True])) is numpy.bool_`, `type(fr.sum([1+2j,3+4j])) is numpy.complex128`, while `fr.sum([[1,2],[3,4]], axis=0)` stays a 1-D `numpy.ndarray` (`scalarize`'s ndim≠0 no-op). Pinned by `ferray-python/tests/divergence_stats.py::test_req21_*` (23 cases). Mirrors numpy/\_core/fromnumeric.py:83 (`return ufunc.reduce(...)`); numpy/\_core/\_methods.py (`_mean`/`_var`/`_amin`/`_amax`); numpy/lib/\_function_base_impl.py (`median`/`average`/`percentile`). Axis reductions stay `ndarray`. AXIS results unaffected (scalarize is the no-op branch). |

## Architecture

### Crate Layout
```
ferray-stats/
  Cargo.toml
  src/
    lib.rs
    reductions/
      mod.rs                  # sum, prod, min, max, mean, var, std
      nan_aware.rs            # nansum, nanmean, etc.
      quantile.rs             # median, percentile, quantile
      cumulative.rs           # cumsum, cumprod, nancumsum, nancumprod
    correlation.rs            # correlate, corrcoef, cov
    histogram.rs              # histogram, histogram2d, histogramdd, bincount, digitize
    sorting.rs                # sort, argsort, searchsorted
    searching.rs              # unique, nonzero, where_, count_nonzero
    set_ops.rs                # union1d, intersect1d, setdiff1d, setxor1d, in1d, isin
    parallel.rs               # Rayon threshold dispatch for reductions and sorting
```

### Design Notes
- Reductions are implemented as iterator-based operations over the array's axis lanes, reusing ndarray's internal `Lanes` iterator via ferray-core's private API.
- NaN-aware variants filter NaN from lanes before reducing. The `nanmean` of an all-NaN slice returns NaN (matching NumPy).
- `median` requires a partial sort (O(n) via quickselect), not a full sort.
- Cumulative operations (`cumsum`, `cumprod`) are also exposed by ferray-ufunc. ferray-stats re-exports them and adds NaN-aware variants.

### Python marshalling boundary (REQ-21)
- The `ferray-python` reduction `#[pyfunction]`s (in `stats.rs`) are thin shims: extract a numpy array, dispatch on dtype, call the typed `ferray_stats::*` / `ferray_ufunc::*` library reduction, then push the owned `ArrayD<T>` result back into a `numpy.ndarray` via `into_pyarray`/`into_any` (or `u64_arrd_to_i64_pyarray` for index/count results).
- numpy collapses a 0-dimensional reduction result to a numpy *scalar* — its reduction wrappers return the bare `ufunc.reduce(...)`/method result (numpy/\_core/fromnumeric.py:83; :2539/:2634 for `any`/`all`), and a 0-d reduce auto-boxes to a `numpy.<dtype>` scalar. The ferray shim instead returns a 0-d `numpy.ndarray`, a marshalling-layer divergence.
- The collapse mechanism already exists on the ferray side: `pub fn scalarize` in `conv.rs` (`result.ndim == 0 -> result[()]`, else no-op) is numpy's own `res[()]` scalar-extraction, and is already consumed by the ufunc shim through `pub fn all_scalar_inputs` in `conv.rs`. REQ-21 generalises that collapse to the full reduction family; only the reduction call-sites in `stats.rs` do not yet route through it (tracked by blocker #992). Axis reductions that leave ≥1 dimension must remain `ndarray` — `scalarize`'s `ndim != 0` no-op branch guarantees this.

## Verification
- `cargo test -p ferray-python` — compiles and runs the binding unit/doctests.
- `python -c "import ferray as fr, numpy as np; assert type(fr.sum([1,2,3])) is np.int64; assert not isinstance(fr.sum([1,2,3]), np.ndarray); assert type(fr.mean([1.,2,3])) is np.float64; assert type(fr.all([True,True])) is np.bool_; assert isinstance(fr.sum([[1,2],[3,4]], axis=0), np.ndarray)"` — the live `$OUT_SCALAR` parity check for REQ-21. Currently FAILS (full reductions return a 0-d `numpy.ndarray`), so REQ-21 is NOT-STARTED pending blocker #992.
- `ferray-python/tests/test_stats.py` — the scalar-vs-0d assertions for the in-scope reduction set become the AC-12 gauntlet once the egresses route through `conv::scalarize`.

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Ufunc-based elementwise operations (ferray-ufunc)
- Statistical distributions / random sampling (ferray-random)
- Polynomial fitting (ferray-polynomial)
- CUMULATIVE reductions (`cumsum`/`cumprod`/`nancumsum`/`nancumprod`) and TUPLE-returning ops (`unique(return_*)`, `histogram*`, `nonzero`, `slogdet`, …) are OUT of scope for the REQ-21 scalar-collapse contract — they never produce a bare 0-d scalar.
