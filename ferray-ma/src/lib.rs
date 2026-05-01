// ferray-ma: Masked arrays with mask propagation
//
// This crate implements `numpy.ma`-style masked arrays for the ferray workspace.
// A `MaskedArray<T, D>` pairs a data array with a boolean mask array where
// `true` = masked/invalid. All operations (arithmetic, reductions, ufuncs)
// respect the mask by skipping masked elements.
//
// # Modules
// - `masked_array`: The core `MaskedArray<T, D>` type
// - `reductions`: Masked mean, sum, min, max, var, std, count
// - `constructors`: masked_where, masked_invalid, masked_equal, etc.
// - `arithmetic`: Masked binary ops with mask union
// - `ufunc_support`: Wrapper functions for ufunc operations on MaskedArrays
// - `sorting`: Masked sort, argsort
// - `mask_ops`: harden_mask, soften_mask, getmask, getdata, is_masked, count_masked
// - `filled`: filled, compressed

// Masked reductions divide running sums by valid-element counts and
// truncate `f64` results to integer index types in argmin/argmax. Float
// equality is also intrinsic to `masked_equal` and `getdata` semantics.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

pub mod arithmetic;
pub mod constructors;
pub mod extras;
pub mod filled;
pub mod interop;
/// Binary I/O (save/load) for `MaskedArray` via ferray-io (#509).
///
/// Gated behind the `io` cargo feature so callers who don't need
/// disk I/O don't have to pull in the zip + binary reader dependency
/// tree through ferray-io.
#[cfg(feature = "io")]
pub mod io;
pub mod manipulation;
pub mod mask_ops;
pub mod masked_array;
pub mod reductions;
pub mod sorting;
pub mod ufunc_support;

// Re-export the primary type at crate root
pub use masked_array::MaskedArray;

// Re-export masking constructors
pub use constructors::{
    fix_invalid, masked_equal, masked_greater, masked_greater_equal, masked_inside, masked_invalid,
    masked_less, masked_less_equal, masked_not_equal, masked_outside, masked_where,
};

// Re-export arithmetic operations
pub use arithmetic::{
    masked_add, masked_add_array, masked_div, masked_div_array, masked_mul, masked_mul_array,
    masked_sub, masked_sub_array,
};

// Re-export mask manipulation functions
pub use mask_ops::{count_masked, count_masked_axis, getdata, getmask, is_masked};

// Re-export MaskAware trait (#505) for downstream code that wants
// to write functions polymorphic over Array and MaskedArray.
pub use interop::{MaskAware, ma_apply_unary};

// Re-export generic ufunc helpers (#513) — the escape hatch for
// ufuncs that don't have a dedicated named wrapper. Users with an
// arbitrary `Fn(T) -> T` / `Fn(T, T) -> T` closure can call
// `ferray_ma::masked_unary(ma, my_fn)` directly.
pub use ufunc_support::{masked_binary, masked_unary};

// Domain-aware ufunc wrappers (#503) — auto-mask out-of-domain
// inputs so the result mask carries a "safe to use" contract.
pub use ufunc_support::{
    arccos_domain, arccosh_domain, arcsin_domain, arctanh_domain, divide_domain, log_domain,
    log2_domain, log10_domain, masked_binary_domain, masked_unary_domain, sqrt_domain,
};

// numpy.ma extras: full reductions, constructors, mask manipulation,
// linalg-lite, set ops, fill-value protocol, comparison/logical ufuncs,
// and class helpers. See extras.rs for the catalogue.
pub use extras::{
    NOMASK, common_fill_value, default_fill_value_bool, default_fill_value_f32,
    default_fill_value_f64, default_fill_value_i64, getmaskarray, ids, is_ma, is_masked_array,
    ma_apply_along_axis, ma_apply_over_axes, ma_concatenate, ma_equal, ma_greater,
    ma_greater_equal, ma_in1d, ma_isin, ma_less, ma_less_equal, ma_logical_and, ma_logical_not,
    ma_logical_or, ma_logical_xor, ma_not_equal, ma_unique, ma_vander, make_mask, make_mask_none,
    mask_or, masked_all, masked_all_like, masked_values, maximum_fill_value, minimum_fill_value,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Array;
    use ferray_core::dimension::Ix1;

    // -----------------------------------------------------------------------
    // AC-1: MaskedArray::new([1,2,3,4,5], [false,false,true,false,false]).mean() == 3.0
    // -----------------------------------------------------------------------
    #[test]
    fn ac1_masked_mean_skips_masked() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, false, true, false, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let mean = ma.mean().unwrap();
        // (1 + 2 + 4 + 5) / 4 = 3.0
        assert!((mean - 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // AC-2: filled(0.0) replaces masked elements with 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn ac2_filled_replaces_masked() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, true, false, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let filled = ma.filled(0.0).unwrap();
        assert_eq!(filled.as_slice().unwrap(), &[1.0, 0.0, 3.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // AC-3: compressed() returns only unmasked elements as 1D
    // -----------------------------------------------------------------------
    #[test]
    fn ac3_compressed_returns_unmasked() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, true, false, true, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let compressed = ma.compressed().unwrap();
        assert_eq!(compressed.as_slice().unwrap(), &[10.0, 30.0, 50.0]);
    }

    // -----------------------------------------------------------------------
    // AC-4: masked_invalid masks NaN and Inf
    // -----------------------------------------------------------------------
    #[test]
    fn ac4_masked_invalid_nan_inf() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, 3.0, f64::INFINITY])
                .unwrap();
        let ma = masked_invalid(&data).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false, true]);
    }

    // -----------------------------------------------------------------------
    // AC-5: ma1 + ma2 produces correct mask union and correct values
    // -----------------------------------------------------------------------
    #[test]
    fn ac5_add_mask_union() {
        let d1 = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m1 =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, true, false, false]).unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let m2 =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, false, true, false]).unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        let result = masked_add(&ma1, &ma2).unwrap();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        // Mask union: [false, true, true, false]
        assert_eq!(mask_vals, vec![false, true, true, false]);
        // Unmasked values: 1+10=11, 4+40=44; masked get 0.0
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 11.0).abs() < 1e-10);
        assert!((data_vals[3] - 44.0).abs() < 1e-10);
    }

    #[test]
    fn operator_add_matches_masked_add() {
        let d1 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let m2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, true]).unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        // Use operator syntax
        let result = (&ma1 + &ma2).unwrap();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, true]);
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 11.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // AC-7: sin(masked_array) returns same mask, correct values
    // -----------------------------------------------------------------------
    #[test]
    fn ac7_ufunc_sin_masked() {
        use std::f64::consts::FRAC_PI_2;
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.0, FRAC_PI_2, FRAC_PI_2]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let result = ufunc_support::sin(&ma).unwrap();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        // sin(0) = 0, masked = 0.0 (skipped), sin(pi/2) = 1.0
        assert!((data_vals[0] - 0.0).abs() < 1e-10);
        assert!((data_vals[2] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // AC-8: sort places masked at end; harden_mask prevents clearing
    // -----------------------------------------------------------------------
    #[test]
    fn ac8_sort_masked_at_end() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, false, true, false, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let sorted = ma.sort().unwrap();
        let data_vals: Vec<f64> = sorted.data().iter().copied().collect();
        let mask_vals: Vec<bool> = sorted.mask().iter().copied().collect();
        // Unmasked [5, 1, 2, 4] sorted = [1, 2, 4, 5], then masked [3]
        assert_eq!(data_vals, vec![1.0, 2.0, 4.0, 5.0, 3.0]);
        assert_eq!(mask_vals, vec![false, false, false, false, true]);
    }

    #[test]
    fn ac8_harden_mask_prevents_clearing() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let mut ma = MaskedArray::new(data, mask).unwrap();

        ma.harden_mask().unwrap();
        assert!(ma.is_hard_mask());

        // Try to clear the mask at index 1 — should be silently ignored
        ma.set_mask_flat(1, false).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);

        // Setting a mask bit to true should still work
        ma.set_mask_flat(0, true).unwrap();
        let mask_vals2: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals2, vec![true, true, false]);

        // Soften and then clearing should work
        ma.soften_mask().unwrap();
        assert!(!ma.is_hard_mask());
        ma.set_mask_flat(1, false).unwrap();
        let mask_vals3: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals3, vec![true, false, false]);
    }

    // -----------------------------------------------------------------------
    // AC-9: is_masked returns true/false correctly
    // -----------------------------------------------------------------------
    #[test]
    fn ac9_is_masked() {
        let data1 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma1 = MaskedArray::new(data1, mask1).unwrap();
        assert!(is_masked(&ma1).unwrap());

        let data2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let ma2 = MaskedArray::new(data2, mask2).unwrap();
        assert!(!is_masked(&ma2).unwrap());
    }

    // -----------------------------------------------------------------------
    // Additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn shape_mismatch_error() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([2]), vec![false, true]).unwrap();
        assert!(MaskedArray::new(data, mask).is_err());
    }

    #[test]
    fn from_data_no_mask() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let ma = MaskedArray::from_data(data).unwrap();
        assert!(!is_masked(&ma).unwrap());
        assert_eq!(ma.count().unwrap(), 3);
    }

    #[test]
    fn sum_skips_masked() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, true, false, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!((ma.sum().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn min_max_skip_masked() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, true, false, false, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!((ma.min().unwrap() - 2.0).abs() < 1e-10);
        assert!((ma.max().unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn var_std_skip_masked() {
        // values: [2, 4, 6] (mask out index 1 and 4)
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![2.0, 99.0, 4.0, 6.0, 99.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, true, false, false, true])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let mean = ma.mean().unwrap();
        assert!((mean - 4.0).abs() < 1e-10);
        // var = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3
        let v = ma.var().unwrap();
        assert!((v - 8.0 / 3.0).abs() < 1e-10);
        let s = ma.std().unwrap();
        assert!((s - (8.0_f64 / 3.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn count_elements() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0; 5]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, true, true, false, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert_eq!(ma.count().unwrap(), 3);
    }

    #[test]
    fn masked_equal_test() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 2.0, 1.0]).unwrap();
        let ma = masked_equal(&data, 2.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false, true, false]);
    }

    #[test]
    fn masked_greater_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ma = masked_greater(&data, 2.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, false, true, true]);
    }

    #[test]
    fn masked_less_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ma = masked_less(&data, 3.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![true, true, false, false]);
    }

    #[test]
    fn masked_not_equal_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let ma = masked_not_equal(&data, 2.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![true, false, true]);
    }

    #[test]
    fn masked_greater_equal_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ma = masked_greater_equal(&data, 3.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, false, true, true]);
    }

    #[test]
    fn masked_less_equal_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ma = masked_less_equal(&data, 2.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![true, true, false, false]);
    }

    #[test]
    fn masked_inside_test() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let ma = masked_inside(&data, 2.0, 4.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, true, true, false]);
    }

    #[test]
    fn masked_outside_test() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let ma = masked_outside(&data, 2.0, 4.0).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![true, false, false, false, true]);
    }

    #[test]
    fn masked_where_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cond =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        let ma = masked_where(&cond, &data).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![true, false, true, false]);
    }

    #[test]
    fn argsort_test() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, false, true, false, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let indices = ma.argsort().unwrap();
        // Unmasked: index 1 (1.0), 3 (2.0), 4 (4.0), 0 (5.0); masked: 2
        assert_eq!(indices.shape(), &[5]);
        assert_eq!(indices.as_slice().unwrap(), &[1u64, 3, 4, 0, 2]);
    }

    #[test]
    fn getmask_getdata_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma = MaskedArray::new(data.clone(), mask.clone()).unwrap();

        let got_mask = getmask(&ma).unwrap();
        let got_data = getdata(&ma).unwrap();

        assert_eq!(got_mask.as_slice().unwrap(), mask.as_slice().unwrap());
        assert_eq!(got_data.as_slice().unwrap(), data.as_slice().unwrap());
    }

    #[test]
    fn count_masked_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0; 5]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, true, false])
                .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert_eq!(count_masked(&ma).unwrap(), 3);
    }

    #[test]
    fn count_masked_axis_2d_along_rows() {
        // #268: per-row masked counts on a 2x3 array.
        // Mask:
        //   [[T, F, T],
        //    [F, F, T]]
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, false, true, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        // axis=1 reduces columns -> per-row counts: [2, 1].
        let counts = count_masked_axis(&ma, 1).unwrap();
        assert_eq!(counts.shape(), &[2]);
        assert_eq!(counts.as_slice().unwrap(), &[2u64, 1]);
    }

    #[test]
    fn count_masked_axis_2d_along_cols() {
        // axis=0 reduces rows -> per-column counts: [1, 0, 2].
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, false, true, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let counts = count_masked_axis(&ma, 0).unwrap();
        assert_eq!(counts.shape(), &[3]);
        assert_eq!(counts.as_slice().unwrap(), &[1u64, 0, 2]);
    }

    #[test]
    fn count_masked_axis_rejects_out_of_bounds() {
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![false; 6]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!(count_masked_axis(&ma, 2).is_err());
    }

    #[test]
    fn sort_axis_2d_per_row() {
        // #271: sort along axis=1 (columns) should sort each row
        // independently. Row 0: [3, 1, _] (mask 2) → unmasked sorted
        // ascending [1, 3] then masked → [1, 3, _].
        // Row 1: [_, 4, 2] (mask 0) → [2, 4, _].
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![3.0, 1.0, 99.0, 99.0, 4.0, 2.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, false, true, true, false, false],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let sorted = ma.sort_axis(1).unwrap();
        assert_eq!(sorted.shape(), &[2, 3]);
        let d: Vec<f64> = sorted.data().iter().copied().collect();
        let m: Vec<bool> = sorted.mask().iter().copied().collect();
        // Row 0
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!(m[2], "row 0 col 2 should be masked");
        // Row 1
        assert!((d[3] - 2.0).abs() < 1e-12);
        assert!((d[4] - 4.0).abs() < 1e-12);
        assert!(m[5], "row 1 col 2 should be masked");
    }

    #[test]
    fn sort_axis_2d_per_column() {
        // axis=0 sorts each column. Column 0: [3, 1] both unmasked →
        // [1, 3]. Column 1: [2, _] (row 1 masked) → [2, _].
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 2]),
            vec![3.0, 2.0, 1.0, 99.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 2]),
            vec![false, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let sorted = ma.sort_axis(0).unwrap();
        let d: Vec<f64> = sorted.data().iter().copied().collect();
        let m: Vec<bool> = sorted.mask().iter().copied().collect();
        // Column 0: index [0,0]=1, [1,0]=3.
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        // Column 1: index [0,1]=2 (unmasked), [1,1]=99 (masked).
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!(m[3]);
    }

    #[test]
    fn sort_axis_rejects_out_of_bounds() {
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![false; 6]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!(ma.sort_axis(2).is_err());
    }

    #[test]
    fn masked_add_array_test() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let result = masked_add_array(&ma, &arr).unwrap();
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);
        assert!((data_vals[0] - 11.0).abs() < 1e-10);
        assert!((data_vals[2] - 33.0).abs() < 1e-10);
    }

    #[test]
    fn all_masked_mean_is_nan() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!(ma.mean().unwrap().is_nan());
    }

    #[test]
    fn all_masked_min_errors() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!(ma.min().is_err());
    }

    #[test]
    fn ufunc_exp_masked() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.0, 1.0, 2.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let result = ufunc_support::exp(&ma).unwrap();
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);
        assert!((data_vals[0] - 1.0).abs() < 1e-10); // exp(0) = 1
        assert!((data_vals[2] - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn ufunc_sqrt_masked() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![4.0, 9.0, 16.0, 25.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, true, false, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let result = ufunc_support::sqrt(&ma).unwrap();
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 2.0).abs() < 1e-10);
        assert!((data_vals[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn set_mask_hardened() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let mut ma = MaskedArray::new(data, mask).unwrap();
        ma.harden_mask().unwrap();

        // set_mask with all-false should not clear the existing true
        let new_mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        ma.set_mask(new_mask).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        // Hard mask: union of old [false, true, false] and new [false, false, false] = [false, true, false]
        assert_eq!(mask_vals, vec![false, true, false]);
    }

    #[test]
    fn masked_sub_test() {
        let d1 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let m1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, true]).unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        let result = masked_sub(&ma1, &ma2).unwrap();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, true]);
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn masked_mul_test() {
        let d1 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 3.0, 4.0]).unwrap();
        let m1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![5.0, 6.0, 7.0]).unwrap();
        let m2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        let result = masked_mul(&ma1, &ma2).unwrap();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 10.0).abs() < 1e-10);
        assert!((data_vals[2] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn masked_div_test() {
        let d1 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let m1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, true]).unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 5.0, 6.0]).unwrap();
        let m2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        let result = masked_div(&ma1, &ma2).unwrap();
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 5.0).abs() < 1e-10);
        assert!((data_vals[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn masked_invalid_negative_inf() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NEG_INFINITY, 3.0]).unwrap();
        let ma = masked_invalid(&data).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);
    }

    #[test]
    fn empty_array_operations() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert_eq!(ma.count().unwrap(), 0);
        assert!(ma.mean().unwrap().is_nan());
        let compressed = ma.compressed().unwrap();
        assert_eq!(compressed.size(), 0);
    }

    #[test]
    fn ndim_shape_size() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0; 5]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false; 5]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert_eq!(ma.ndim(), 1);
        assert_eq!(ma.shape(), &[5]);
        assert_eq!(ma.size(), 5);
    }

    #[test]
    fn ufunc_binary_power() {
        let d1 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 3.0, 4.0]).unwrap();
        let m1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![3.0, 2.0, 2.0]).unwrap();
        let m2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        let result = ufunc_support::power(&ma1, &ma2).unwrap();
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, false]);
        assert!((data_vals[0] - 8.0).abs() < 1e-10); // 2^3 = 8
        assert!((data_vals[2] - 16.0).abs() < 1e-10); // 4^2 = 16
    }

    #[test]
    fn filled_with_custom_value() {
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let filled = ma.filled(-999.0).unwrap();
        assert_eq!(filled.as_slice().unwrap(), &[-999.0, 2.0, -999.0, 4.0]);
    }

    // --- 2D masked array tests ---

    #[test]
    fn masked_2d_construction() {
        use ferray_core::dimension::Ix2;
        let data =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, true, false, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert_eq!(ma.ndim(), 2);
        assert_eq!(ma.shape(), &[2, 3]);
        assert_eq!(ma.size(), 6);
        assert_eq!(ma.count().unwrap(), 4);
    }

    #[test]
    fn masked_2d_mean() {
        use ferray_core::dimension::Ix2;
        let data =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        // Mask out 2.0 and 6.0
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, true, false, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        // mean of [1, 3, 4, 5] = 13/4 = 3.25
        let m = ma.mean().unwrap();
        assert!((m - 3.25).abs() < 1e-10);
    }

    #[test]
    fn masked_2d_sum() {
        use ferray_core::dimension::Ix2;
        let data =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, true, false, false, false, true],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        // sum of [1, 3, 4, 5] = 13
        assert!((ma.sum().unwrap() - 13.0).abs() < 1e-10);
    }

    #[test]
    fn masked_2d_add_operator() {
        use ferray_core::dimension::Ix2;
        let d1 = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m1 = Array::<bool, Ix2>::from_vec(Ix2::new([2, 2]), vec![false, true, false, false])
            .unwrap();
        let ma1 = MaskedArray::new(d1, m1).unwrap();

        let d2 =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let m2 = Array::<bool, Ix2>::from_vec(Ix2::new([2, 2]), vec![false, false, true, false])
            .unwrap();
        let ma2 = MaskedArray::new(d2, m2).unwrap();

        let result = (&ma1 + &ma2).unwrap();
        let mask_vals: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(mask_vals, vec![false, true, true, false]);
        let data_vals: Vec<f64> = result.data().iter().copied().collect();
        assert!((data_vals[0] - 11.0).abs() < 1e-10);
        assert!((data_vals[3] - 44.0).abs() < 1e-10);
    }

    #[test]
    fn masked_2d_compressed() {
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mask =
            Array::<bool, Ix2>::from_vec(Ix2::new([2, 2]), vec![false, true, false, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let compressed = ma.compressed().unwrap();
        assert_eq!(compressed.as_slice().unwrap(), &[1.0, 3.0]);
    }
}
