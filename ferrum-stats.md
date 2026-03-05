---
title: ferrum-stats — Statistical functions, sorting, searching, set operations, and histograms
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Implements all NumPy statistical functions as array methods and top-level free functions: reductions (sum, prod, min, max, mean, var, std, median, percentile, quantile), cumulative operations, NaN-aware variants, correlations, covariance, histograms, sorting, searching, unique, and set operations. These are the bread-and-butter data analysis functions that most users reach for after array creation.

### Requirements

- REQ-1: Array methods: `sum()`, `prod()`, `min()`, `max()`, `argmin()`, `argmax()`, `mean()`, `var(ddof)`, `std(ddof)`, `median()`, `percentile(q)`, `quantile(q)` — all accept optional `axis` parameter. `axis: None` reduces over entire array.
- REQ-2: All reductions available as both array methods and top-level free functions ### Cumulative Operations
- REQ-2a: `cumsum(&a, axis)` and `cumprod(&a, axis)` — cumulative sum/product along axis (also in ferrum-ufunc, but exposed here as array methods and free functions for discoverability)
- REQ-2b: `nancumsum(&a, axis)` and `nancumprod(&a, axis)` — NaN-aware cumulative operations (NaN treated as 0 for sum, 1 for product) ### NaN-Aware Variants
- REQ-3: `nansum`, `nanprod`, `nanmin`, `nanmax`, `nanmean`, `nanvar(ddof)`, `nanstd(ddof)`, `nanmedian`, `nanpercentile(q)` — skip NaN values in computation
- REQ-4: NaN-aware reductions on all-NaN slices must return a well-defined result (NaN for nanmean, 0 for nansum, matching NumPy) ### Correlations and Covariance
- REQ-5: `correlate(&a, &v, mode)` with modes Full, Same, Valid — matching `np.correlate`
- REQ-6: `corrcoef(&x, rowvar)` — Pearson correlation coefficient matrix
- REQ-7: `cov(&m, rowvar, ddof)` — covariance matrix ### Histograms
- REQ-8: `histogram(&a, bins, range, density)` returning `(counts, bin_edges)` — matching `np.histogram`
- REQ-9: `histogram2d(&x, &y, bins)` and `histogramdd(&sample, bins)` for multi-dimensional histograms
- REQ-10: `bincount(&x, weights, minlength)` and `digitize(&x, &bins, right)` ### Sorting and Searching
- REQ-11: `sort(axis, kind)` with `SortKind::Stable` (merge sort) and `SortKind::Quick` — matching `np.sort`
- REQ-12: `argsort(axis)` returning index array
- REQ-13: `searchsorted(&a, &v, side)` with `Side::Left` and `Side::Right`
- REQ-14: `unique(&a, return_index, return_counts)` returning sorted unique elements
- REQ-15: `nonzero(&a)` returning tuple of index arrays
- REQ-16: `where_(&condition, &x, &y)` — conditional selection matching `np.where`
- REQ-17: `count_nonzero(&a, axis)` ### Set Operations (Section 14)
- REQ-18: `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `in1d`, `isin` — all with `assume_unique` optimization flag ### Parallelism
- REQ-19: Large reductions (>10k elements) use parallel tree-reduce via Rayon on ferrum's owned pool
- REQ-20: Sorting of large arrays (>100k elements) uses parallel merge sort via Rayon

### Acceptance Criteria

- [ ] AC-1: `a.sum(axis=None)` and `a.sum(axis=0)` produce results matching NumPy for 1D, 2D, and 3D arrays of f64
- [ ] AC-2: `a.mean()`, `a.var(ddof=0)`, `a.std(ddof=1)` match NumPy to within 4 ULPs on fixture data
- [ ] AC-3: NaN-aware variants skip NaN values correctly: `nanmean([1.0, NaN, 3.0]) == 2.0`
- [ ] AC-4: `histogram()` produces bin counts and edges matching `np.histogram` exactly for integer inputs
- [ ] AC-5: `sort()` with `SortKind::Stable` preserves relative order of equal elements
- [ ] AC-6: `unique()` with `return_counts=true` matches `np.unique(return_counts=True)`
- [ ] AC-7: Set operations: `union1d([1,2,3], [2,3,4]) == [1,2,3,4]`, etc.
- [ ] AC-8: `where_(&mask, &x, &y)` selects from x where mask is true, y otherwise
- [ ] AC-9: `cargo test -p ferrum-stats` passes. `cargo clippy -p ferrum-stats -- -D warnings` clean.
- [ ] AC-10: `cumsum(&[1,2,3,4], None)` returns `[1,3,6,10]`. `nancumsum(&[1.0, NaN, 3.0], None)` returns `[1.0, 1.0, 4.0]`.
- [ ] AC-11: `cumprod(&[1,2,3,4], None)` returns `[1,2,6,24]`.

### Architecture

### Crate Layout
```
ferrum-stats/
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
- Reductions are implemented as iterator-based operations over the array's axis lanes, reusing ndarray's internal `Lanes` iterator via ferrum-core's private API.
- NaN-aware variants filter NaN from lanes before reducing. The `nanmean` of an all-NaN slice returns NaN (matching NumPy).
- `median` requires a partial sort (O(n) via quickselect), not a full sort.
- Cumulative operations (`cumsum`, `cumprod`) are also exposed by ferrum-ufunc. ferrum-stats re-exports them and adds NaN-aware variants.

### Out of Scope

- Ufunc-based elementwise operations (ferrum-ufunc)
- Statistical distributions / random sampling (ferrum-random)
- Polynomial fitting (ferrum-polynomial)

### dependencies

- **Upstream**: `ferrum-core` (NdArray, Dimension, Element, FerrumError), `ferrum-ufunc` (comparison ufuncs for sorting/searching)
- **Downstream**: ferrum-linalg (uses reductions), ferrum-random (may use stats for validation), ferrum (re-export)
- **External crates**: `rayon` (parallel reductions), `num-traits`
- **Phase**: 1 — Core Array and Ufuncs

