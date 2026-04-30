# ferray-ma

Masked arrays with mask propagation for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- `MaskedArray<T, D>` — array with boolean mask for missing/invalid data
- **Mask propagation**: arithmetic and ufuncs automatically propagate masks
- **Full reductions**: `sum`, `prod`, `mean`, `std`, `var`, `cumsum`, `cumprod`, `min`, `max`, `argmin`, `argmax`, `median`, `ptp`, `average`, `anom`
- **Bitwise / comparison ufuncs**: `and`/`or`/`xor`/shift, equal/less/greater family, logical_*, floor_divide/fmod/power/remainder/true_divide/hypot/negative/sqrt/square
- **Array creation**: `arange`, `zeros`, `ones`, `empty`, `identity`, `masked_all`, `masked_array`, `masked_values`, `masked_object`, `masked_outside`, `masked_less/_equal`, `masked_not_equal`, `frombuffer`, `fromfunction`, `fromflex`
- **Manipulation**: `concatenate`/`stack` family, `reshape`, `ravel`, `squeeze`, `swapaxes`, `transpose`, `expand_dims`, `atleast_*`, `take`, `put`, `putmask`, `repeat`, `resize`, `append`, `diag`, `diagflat`, `diagonal`, `choose`, `clip`
- **Mask manipulation**: `harden_mask`/`soften_mask`, `mask_or`, `make_mask*`, `compress_rows/cols/rowcols/nd`, `mask_rows/cols/rowcols`, `clump_*masked`, `*notmasked_contiguous/edges`, `flatten_mask`, `flatten_structured_array`
- **Linalg / set / corr**: `dot`, `inner`, `outer`, `innerproduct`, `outerproduct`, `trace`, `correlate`, `convolve`, `corrcoef`, `cov`, `unique`, `in1d`, `isin`, `intersect1d`, `setdiff1d`, `setxor1d`, `union1d`, `polyfit`
- **`apply_along_axis` / `apply_over_axes` / `vander`** for axis-aware mapping
- **Fill-value protocol**: `default_fill_value`, `maximum_fill_value`, `minimum_fill_value`, `set_fill_value`, `common_fill_value`
- **Class helpers**: `mvoid`, `nomask`, `masked_singleton`, `masked_print_option`, `mr_`, `isMaskedArray`/`isMA`/`isarray`, `getmaskarray`, `ids`

Implements `numpy.ma` for Rust.

## Usage

```rust
use ferray_ma::MaskedArray;
use ferray_core::prelude::*;

let data = Array1::<f64>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0])?;
let mask = vec![false, false, true, false]; // mask out index 2
let ma = MaskedArray::new(data, mask)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `ma` feature (enabled by default).

## License

MIT OR Apache-2.0
