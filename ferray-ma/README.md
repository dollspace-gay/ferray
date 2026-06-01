# ferray-ma

NumPy `numpy.ma`-style masked arrays with mask propagation for the ferray workspace.

Part of the [ferray](../README.md) workspace.

## Overview

`ferray-ma` implements masked arrays for ferray's Rust-native, drop-in NumPy
replacement. The core type is `MaskedArray<T, D>`, which pairs a data array
with a boolean mask (`true` = masked/invalid). Every operation respects the
mask by skipping masked elements.

- **Core type** — `MaskedArray<T, D>` built via `MaskedArray::new(data, mask)`
  or `MaskedArray::from_data(data)` (no mask). Accessors: `data()`, `mask()`,
  `mask_opt()`, `shape()`, `ndim()`, `size()`, `with_fill_value()`,
  `set_fill_value()`; mask control via `harden_mask()` / `soften_mask()`,
  `is_hard_mask()`, `set_mask()`, `set_mask_flat()`.
- **Constructors** — `masked_where`, `masked_invalid`, `masked_equal`,
  `masked_not_equal`, `masked_greater`, `masked_greater_equal`, `masked_less`,
  `masked_less_equal`, `masked_inside`, `masked_outside`, `fix_invalid`, plus
  `extras::{masked_all, masked_all_like, masked_values}`.
- **filled / compressed** — `filled(fill)`, `filled_default()`, and
  `compressed()` (unmasked elements as a 1-D array).
- **Mask-skipping reductions** (`reductions`) — `count`, `sum`, `mean`, `min`,
  `max`, `var`, `var_ddof`, `std`, `std_ddof`, plus axis variants
  (`sum_axis`, `mean_axis`, `min_axis`, `max_axis`, `var_axis`, `std_axis`,
  `count_axis`). All-masked reductions return NaN (`mean`) or error (`min`).
- **Arithmetic** (`arithmetic`) — `masked_add`/`masked_sub`/`masked_mul`/
  `masked_div` (and `_array` variants) with mask-union propagation; `&a + &b`
  operator syntax is supported.
- **Ufunc support** (`ufunc_support`) — named wrappers (`sin`, `exp`, `sqrt`,
  `power`, …), generic escape hatches `masked_unary` / `masked_binary`, and
  domain-masking wrappers `sqrt_domain`, `log_domain`, `arcsin_domain`,
  `arccos_domain`, `arctanh_domain`, `divide_domain`, … that auto-mask
  out-of-domain inputs.
- **ndarray interop** (`mask_ops`, `interop`) — `getdata`, `getmask`,
  `is_masked`, `count_masked`, `count_masked_axis`; `MaskAware` trait and
  `ma_apply_unary` for code polymorphic over `Array` and `MaskedArray`.
- **`extras` namespace** — `average` / `average_returned`, `ma_median_axis`,
  `ma_cov`, `ma_corrcoef`, `notmasked_contiguous_axis`, `notmasked_edges`,
  `clump_masked`, `clump_unmasked`, set ops (`ma_unique`, `ma_in1d`,
  `ma_isin`, `ma_intersect1d`, `ma_union1d`, …), fill-value protocol
  (`default_fill_value_f64`, `maximum_fill_value`, `minimum_fill_value`,
  `common_fill_value`), and mask builders (`make_mask`, `make_mask_none`,
  `mask_or`).

## NumPy correspondence

| NumPy (`numpy.ma`) | ferray-ma |
|--------------------|-----------|
| `np.ma.MaskedArray` | `MaskedArray<T, D>` |
| `np.ma.masked_where` | `masked_where` |
| `np.ma.masked_invalid` | `masked_invalid` |
| `np.ma.masked_equal` | `masked_equal` |
| `MaskedArray.filled` | `MaskedArray::filled` |
| `MaskedArray.compressed` | `MaskedArray::compressed` |
| `np.ma.getdata` / `np.ma.getmask` | `getdata` / `getmask` |
| `np.ma.average` | `MaskedArray::average` |
| `np.ma.median` | `ma_median_axis` |
| `np.ma.cov` / `np.ma.corrcoef` | `ma_cov` / `ma_corrcoef` |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `io` | off | Enables `io::save_masked` / `io::load_masked` for `MaskedArray` round-trip through paired `.npy` files (data + mask) via `ferray-io`. Optional so core functionality stays free of disk-I/O dependencies. |

## Example

```rust
use ferray_ma::MaskedArray;
use ferray_core::Array;
use ferray_core::dimension::Ix1;

// Build a masked array, masking out index 2.
let data = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
let mask = Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, false, true, false, false])?;
let ma = MaskedArray::new(data, mask)?;

// Mask-skipping mean: (1 + 2 + 4 + 5) / 4 == 3.0
let mean = ma.mean()?;
assert!((mean - 3.0).abs() < 1e-10);

// Fill masked slots, then keep only the unmasked elements.
let filled = ma.filled(0.0)?;          // [1.0, 2.0, 0.0, 4.0, 5.0]
let kept = ma.compressed()?;           // [1.0, 2.0, 4.0, 5.0]
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Edition 2024, MSRV 1.88.
- License: MIT OR Apache-2.0.
