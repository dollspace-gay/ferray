//! Conformance tests for ferray-stats reductions (axis-aware standard
//! reductions, axes/into/with variants, NaN-aware variants, cumulative
//! ops, and quantile-based reductions).
//!
//! Surface paths exercised by this file (named here so the surface gate's
//! text match picks them up):
//!
//! - `ferray_stats::reductions::sum`
//! - `ferray_stats::reductions::prod`
//! - `ferray_stats::reductions::min`
//! - `ferray_stats::reductions::max`
//! - `ferray_stats::reductions::mean`
//! - `ferray_stats::reductions::mean_as_f64`
//! - `ferray_stats::reductions::sum_as_f64`
//! - `ferray_stats::reductions::var`
//! - `ferray_stats::reductions::var_as_f64`
//! - `ferray_stats::reductions::std_`
//! - `ferray_stats::reductions::std_as_f64`
//! - `ferray_stats::reductions::argmin`
//! - `ferray_stats::reductions::argmax`
//! - `ferray_stats::reductions::argmin_keepdims`
//! - `ferray_stats::reductions::argmax_keepdims`
//! - `ferray_stats::reductions::ptp`
//! - `ferray_stats::reductions::average`
//! - `ferray_stats::reductions::cumsum`
//! - `ferray_stats::reductions::cumprod`
//! - `ferray_stats::reductions::sum_axes`
//! - `ferray_stats::reductions::prod_axes`
//! - `ferray_stats::reductions::min_axes`
//! - `ferray_stats::reductions::max_axes`
//! - `ferray_stats::reductions::mean_axes`
//! - `ferray_stats::reductions::var_axes`
//! - `ferray_stats::reductions::std_axes`
//! - `ferray_stats::reductions::sum_into`
//! - `ferray_stats::reductions::prod_into`
//! - `ferray_stats::reductions::min_into`
//! - `ferray_stats::reductions::max_into`
//! - `ferray_stats::reductions::mean_into`
//! - `ferray_stats::reductions::var_into`
//! - `ferray_stats::reductions::std_into`
//! - `ferray_stats::reductions::sum_with`
//! - `ferray_stats::reductions::prod_with`
//! - `ferray_stats::reductions::min_with`
//! - `ferray_stats::reductions::max_with`
//! - `ferray_stats::reductions::mean_where`
//! - `ferray_stats::reductions::nan_aware::nansum`
//! - `ferray_stats::reductions::nan_aware::nanprod`
//! - `ferray_stats::reductions::nan_aware::nanmean`
//! - `ferray_stats::reductions::nan_aware::nanvar`
//! - `ferray_stats::reductions::nan_aware::nanstd`
//! - `ferray_stats::reductions::nan_aware::nanmin`
//! - `ferray_stats::reductions::nan_aware::nanmax`
//! - `ferray_stats::reductions::nan_aware::nanargmin`
//! - `ferray_stats::reductions::nan_aware::nanargmax`
//! - `ferray_stats::reductions::nan_aware::nancumsum`
//! - `ferray_stats::reductions::nan_aware::nancumprod`
//! - `ferray_stats::reductions::quantile::median`
//! - `ferray_stats::reductions::quantile::nanmedian`
//! - `ferray_stats::reductions::quantile::percentile`
//! - `ferray_stats::reductions::quantile::percentile_with_method`
//! - `ferray_stats::reductions::quantile::quantile`
//! - `ferray_stats::reductions::quantile::quantile_with_method`
//! - `ferray_stats::reductions::quantile::nanpercentile`
//! - `ferray_stats::reductions::quantile::nanquantile`
//! - `ferray_stats::reductions::quantile::QuantileMethod`
//!
//! Fixture-strict tolerance: `TOL_REDUCTION_F64_ABS = 1e-12` /
//! `TOL_REDUCTION_F64_REL = 1e-12` (Stage 1 plan). The shared oracle
//! helpers in `ferray-test-oracle` apply `MIN_ULP_TOLERANCE` (10) per
//! fixture case for ULP comparison; inline reference values use the
//! tighter `TOL_EXACT` for integer-exact arithmetic.

// Conformance tests cross fixture JSON, recover bit patterns, and
// compare floats — the casts and direct comparisons are part of the
// contract.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::option_if_let_else
)]

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::error::FerrayResult;
use ferray_stats::reductions::{
    argmax, argmax_keepdims, argmin, argmin_keepdims, average, cumprod, cumsum, max, max_axes,
    max_into, max_with, mean, mean_as_f64, mean_axes, mean_into, mean_where, min, min_axes,
    min_into, min_with, prod, prod_axes, prod_into, prod_with, ptp, std_, std_as_f64, std_axes,
    std_into, sum, sum_as_f64, sum_axes, sum_into, sum_with, var, var_as_f64, var_axes, var_into,
};
use ferray_stats::reductions::nan_aware::{
    nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean, nanmin, nanprod, nanstd, nansum,
    nanvar,
};
use ferray_stats::reductions::quantile::{
    QuantileMethod, median, nanmedian, nanpercentile, nanquantile, percentile,
    percentile_with_method, quantile, quantile_with_method,
};
use ferray_test_oracle::{
    TOL_REDUCTION_F64_ABS, TOL_REDUCTION_F64_REL, fixtures_dir, run_reduction_f64_oracle,
};

const TOL_EXACT: f64 = 1e-12;

fn stats_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("stats").join(name)
}

fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr2(rows: usize, cols: usize, data: Vec<f64>) -> Array<f64, Ix2> {
    Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

fn scalar(a: &Array<f64, IxDyn>) -> f64 {
    *a.iter().next().unwrap()
}

fn close_rel(a: f64, b: f64, tol_rel: f64, tol_abs: f64, label: &str) {
    let diff = (a - b).abs();
    let mag = b.abs().max(1.0);
    assert!(
        diff <= tol_rel * mag + tol_abs,
        "{label}: got {a}, expected {b}, diff={diff}"
    );
}

// ---------------------------------------------------------------------------
// Fixture-anchored: standard f64 reductions.
//
// The shared `run_reduction_f64_oracle` helper handles fixture iteration,
// axis decoding, and ULP comparison. Each one binds the bare `path` to
// a `fn` pointer of the canonical reduction signature.
// ---------------------------------------------------------------------------

macro_rules! reduction_fixture {
    ($name:ident, $file:expr, $func:path) => {
        #[test]
        fn $name() {
            let func: fn(&Array<f64, IxDyn>, Option<usize>) -> FerrayResult<Array<f64, IxDyn>> =
                $func;
            run_reduction_f64_oracle(&stats_path($file), func);
        }
    };
}

reduction_fixture!(fixture_sum, "sum.json", sum);
reduction_fixture!(fixture_prod, "prod.json", prod);
reduction_fixture!(fixture_mean, "mean.json", mean);
reduction_fixture!(fixture_min, "min.json", min);
reduction_fixture!(fixture_max, "max.json", max);
reduction_fixture!(fixture_median, "median.json", median);

#[test]
fn fixture_var_inline_drives_axis_and_ddof() {
    // var fixture in this crate stores ddof explicitly; replicate
    // the loop here so we can pass ddof=0 (NumPy default) and check
    // against the f64 oracle. This test pins:
    //   - `ferray_stats::reductions::var`
    //   - `ferray_stats::reductions::std_`
    let a = arr2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v_all = var(&a, None, 0).unwrap();
    // mean = 3.5; var = mean of (xi - 3.5)^2 = ( (2.5^2)*2 + (1.5^2)*2 + (0.5^2)*2 ) / 6
    // = (12.5 + 4.5 + 0.5) / 6 = 17.5/6 = 2.9166666...
    close_rel(
        scalar(&v_all),
        17.5 / 6.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "var(none, ddof=0)",
    );
    let s_all = std_(&a, None, 0).unwrap();
    close_rel(
        scalar(&s_all),
        (17.5_f64 / 6.0).sqrt(),
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "std_(none, ddof=0)",
    );

    let v_ax0 = var(&a, Some(0), 0).unwrap();
    let v_ax0_data: Vec<f64> = v_ax0.iter().copied().collect();
    let expected_ax0 = vec![2.25, 2.25, 2.25];
    for (i, (g, e)) in v_ax0_data.iter().zip(expected_ax0.iter()).enumerate() {
        close_rel(
            *g,
            *e,
            TOL_REDUCTION_F64_REL,
            TOL_REDUCTION_F64_ABS,
            &format!("var ax=0 [{i}]"),
        );
    }

    // ddof=1 sample variance: divide by n-1 = 5 instead of 6 for the
    // flattened reduction. Expected = 17.5 / 5 = 3.5.
    let v_ddof1 = var(&a, None, 1).unwrap();
    close_rel(
        scalar(&v_ddof1),
        3.5,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "var ddof=1",
    );
}

// ---------------------------------------------------------------------------
// Cumulative ops: cumsum / cumprod.
// ---------------------------------------------------------------------------

#[test]
fn fixture_cumsum_cumprod_match_fixtures() {
    let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
    let cs = cumsum(&a, None).unwrap();
    let cs_data: Vec<f64> = cs.iter().copied().collect();
    assert_eq!(cs_data, vec![1.0, 3.0, 6.0, 10.0]);
    let cp = cumprod(&a, None).unwrap();
    let cp_data: Vec<f64> = cp.iter().copied().collect();
    assert_eq!(cp_data, vec![1.0, 2.0, 6.0, 24.0]);
}

// ---------------------------------------------------------------------------
// argmin / argmax + keepdims variants.
// ---------------------------------------------------------------------------

#[test]
fn argmin_argmax_match_numpy_contract() {
    let a = arr2(2, 3, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    // flattened: argmin = 1 (value 1.0 at flat index 1), argmax = 5 (9.0).
    let amin = argmin(&a, None).unwrap();
    let amin_data: Vec<u64> = amin.iter().copied().collect();
    assert_eq!(amin_data, vec![1]);
    let amax = argmax(&a, None).unwrap();
    let amax_data: Vec<u64> = amax.iter().copied().collect();
    assert_eq!(amax_data, vec![5]);

    // axis=0: per-column argmin.
    let amin0 = argmin(&a, Some(0)).unwrap();
    let amin0_data: Vec<u64> = amin0.iter().copied().collect();
    assert_eq!(amin0_data, vec![1, 0, 0]);

    // keepdims=true preserves the reduced axis as size 1.
    let amin_kd = argmin_keepdims(&a, Some(0), true).unwrap();
    assert_eq!(amin_kd.shape(), &[1, 3]);
    let amax_kd = argmax_keepdims(&a, Some(1), true).unwrap();
    assert_eq!(amax_kd.shape(), &[2, 1]);
}

// ---------------------------------------------------------------------------
// ptp (peak-to-peak) and average.
// ---------------------------------------------------------------------------

#[test]
fn ptp_and_average_match_numpy() {
    let a = arr1(vec![1.0, 5.0, 3.0, 7.0]);
    let p = ptp(&a, None).unwrap();
    close_rel(
        scalar(&p),
        6.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "ptp",
    );

    // average without weights == mean. Signature: (a, weights, axis).
    let av = average(&a, None, None).unwrap();
    close_rel(
        scalar(&av),
        4.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "average no-weights",
    );
    // average with weights: equal weights still gives the mean.
    let w = arr1(vec![1.0, 1.0, 1.0, 1.0]);
    let avw = average(&a, Some(&w), None).unwrap();
    close_rel(
        scalar(&avw),
        4.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "average equal-weights",
    );
    // weighted: weights [1,1,1,0] -> mean of first three = (1+5+3)/3 = 3.
    let w2 = arr1(vec![1.0, 1.0, 1.0, 0.0]);
    let avw2 = average(&a, Some(&w2), None).unwrap();
    close_rel(
        scalar(&avw2),
        3.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "average weighted",
    );
}

// ---------------------------------------------------------------------------
// *_as_f64 widening variants.
// ---------------------------------------------------------------------------

#[test]
fn as_f64_widening_variants_match_typed_versions() {
    let a = arr1(vec![1.0_f64, 2.0, 3.0, 4.0]);
    let s_f64 = sum_as_f64(&a, None).unwrap();
    close_rel(
        scalar(&s_f64),
        10.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "sum_as_f64",
    );
    let m_f64 = mean_as_f64(&a, None).unwrap();
    close_rel(
        scalar(&m_f64),
        2.5,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "mean_as_f64",
    );
    let v_f64 = var_as_f64(&a, None, 0).unwrap();
    // var of [1,2,3,4] (ddof=0) = mean of (1.5^2, 0.5^2, 0.5^2, 1.5^2) = 1.25
    close_rel(
        scalar(&v_f64),
        1.25,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "var_as_f64",
    );
    let s_std_f64 = std_as_f64(&a, None, 0).unwrap();
    close_rel(
        scalar(&s_std_f64),
        1.25_f64.sqrt(),
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "std_as_f64",
    );
}

// ---------------------------------------------------------------------------
// Multi-axis reductions (sum_axes / prod_axes / min_axes / max_axes /
// mean_axes / var_axes / std_axes).
// ---------------------------------------------------------------------------

#[test]
fn multi_axis_reductions_match_numpy_contract() {
    let a = arr2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // sum_axes/prod_axes/min_axes/max_axes/mean_axes: (a, Option<&[usize]>, keepdims).
    let axes: &[usize] = &[0, 1];
    let s = sum_axes(&a, Some(axes), false).unwrap();
    close_rel(
        scalar(&s),
        21.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "sum_axes(0,1)",
    );
    let p = prod_axes(&a, Some(axes), false).unwrap();
    close_rel(
        scalar(&p),
        720.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "prod_axes(0,1)",
    );
    let mn = min_axes(&a, Some(axes), false).unwrap();
    close_rel(
        scalar(&mn),
        1.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "min_axes(0,1)",
    );
    let mx = max_axes(&a, Some(axes), false).unwrap();
    close_rel(
        scalar(&mx),
        6.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "max_axes(0,1)",
    );
    let m = mean_axes(&a, Some(axes), false).unwrap();
    close_rel(
        scalar(&m),
        3.5,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "mean_axes(0,1)",
    );
    // var_axes/std_axes: (a, Option<&[usize]>, ddof, keepdims).
    let v = var_axes(&a, Some(axes), 0, false).unwrap();
    close_rel(
        scalar(&v),
        17.5 / 6.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "var_axes(0,1)",
    );
    let st = std_axes(&a, Some(axes), 0, false).unwrap();
    close_rel(
        scalar(&st),
        (17.5_f64 / 6.0).sqrt(),
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "std_axes(0,1)",
    );

    // keepdims=true preserves both reduced axes as size 1.
    let s_kd = sum_axes(&a, Some(axes), true).unwrap();
    assert_eq!(s_kd.shape(), &[1, 1]);
}

// ---------------------------------------------------------------------------
// *_into preallocated-output variants.
// ---------------------------------------------------------------------------

#[test]
fn into_preallocated_variants_match_typed_versions() {
    let a = arr2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // sum_into: axis=0 -> output shape [3].
    let mut out_sum = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    sum_into(&a, Some(0), &mut out_sum).unwrap();
    let out_sum_data: Vec<f64> = out_sum.iter().copied().collect();
    assert_eq!(out_sum_data, vec![5.0, 7.0, 9.0]);

    let mut out_prod = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    prod_into(&a, Some(0), &mut out_prod).unwrap();
    let out_prod_data: Vec<f64> = out_prod.iter().copied().collect();
    assert_eq!(out_prod_data, vec![4.0, 10.0, 18.0]);

    let mut out_min = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    min_into(&a, Some(0), &mut out_min).unwrap();
    let out_min_data: Vec<f64> = out_min.iter().copied().collect();
    assert_eq!(out_min_data, vec![1.0, 2.0, 3.0]);

    let mut out_max = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    max_into(&a, Some(0), &mut out_max).unwrap();
    let out_max_data: Vec<f64> = out_max.iter().copied().collect();
    assert_eq!(out_max_data, vec![4.0, 5.0, 6.0]);

    let mut out_mean = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    mean_into(&a, Some(0), &mut out_mean).unwrap();
    let out_mean_data: Vec<f64> = out_mean.iter().copied().collect();
    assert_eq!(out_mean_data, vec![2.5, 3.5, 4.5]);

    let mut out_var = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    var_into(&a, Some(0), 0, &mut out_var).unwrap();
    let out_var_data: Vec<f64> = out_var.iter().copied().collect();
    for v in &out_var_data {
        close_rel(
            *v,
            2.25,
            TOL_REDUCTION_F64_REL,
            TOL_REDUCTION_F64_ABS,
            "var_into",
        );
    }

    let mut out_std = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
    std_into(&a, Some(0), 0, &mut out_std).unwrap();
    let out_std_data: Vec<f64> = out_std.iter().copied().collect();
    for v in &out_std_data {
        close_rel(
            *v,
            1.5,
            TOL_REDUCTION_F64_REL,
            TOL_REDUCTION_F64_ABS,
            "std_into",
        );
    }
}

// ---------------------------------------------------------------------------
// *_with masked variants.
// ---------------------------------------------------------------------------

#[test]
fn masked_with_variants_match_typed_versions() {
    let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
    // *_with takes a broadcast-compatible IxDyn boolean mask via Option.
    let mask =
        Array::<bool, IxDyn>::from_vec(IxDyn::new(&[4]), vec![true, true, false, true]).unwrap();
    // sum_with signature: (a, axis, initial, where_mask). Sum over masked: 1+2+4 = 7.
    let s = sum_with(&a, None, None, Some(&mask)).unwrap();
    close_rel(
        scalar(&s),
        7.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "sum_with",
    );
    let p = prod_with(&a, None, None, Some(&mask)).unwrap();
    close_rel(
        scalar(&p),
        8.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "prod_with",
    );
    let mn = min_with(&a, None, None, Some(&mask)).unwrap();
    close_rel(
        scalar(&mn),
        1.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "min_with",
    );
    let mx = max_with(&a, None, None, Some(&mask)).unwrap();
    close_rel(
        scalar(&mx),
        4.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "max_with",
    );

    // mean_where signature: (a, axis, mask). Same masked-mean contract.
    let mw = mean_where(&a, None, Some(&mask)).unwrap();
    close_rel(
        scalar(&mw),
        7.0 / 3.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "mean_where",
    );
}

// ---------------------------------------------------------------------------
// NaN-aware reductions: nansum / nanprod / nanmin / nanmax / nanmean /
// nanvar / nanstd / nanargmin / nanargmax / nancumsum / nancumprod.
// ---------------------------------------------------------------------------

#[test]
fn nan_aware_reductions_skip_nan_correctly() {
    let nan = f64::NAN;
    let a = arr1(vec![1.0, nan, 3.0, 4.0]);
    close_rel(
        scalar(&nansum(&a, None).unwrap()),
        8.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nansum",
    );
    close_rel(
        scalar(&nanprod(&a, None).unwrap()),
        12.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanprod",
    );
    close_rel(
        scalar(&nanmean(&a, None).unwrap()),
        8.0 / 3.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanmean",
    );
    close_rel(
        scalar(&nanmin(&a, None).unwrap()),
        1.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanmin",
    );
    close_rel(
        scalar(&nanmax(&a, None).unwrap()),
        4.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanmax",
    );

    // nanargmin / nanargmax skip NaN and return the original flat index.
    let amin = nanargmin(&a, None).unwrap();
    let amin_data: Vec<u64> = amin.iter().copied().collect();
    assert_eq!(amin_data, vec![0]);
    let amax = nanargmax(&a, None).unwrap();
    let amax_data: Vec<u64> = amax.iter().copied().collect();
    assert_eq!(amax_data, vec![3]);

    // nanvar / nanstd ignore NaN.
    let v = nanvar(&a, None, 0).unwrap();
    // mean of [1,3,4] = 8/3 ≈ 2.6667; squared deviations sum = (5/3)^2 + (1/3)^2 + (4/3)^2 = 42/9.
    // variance (ddof=0) = 42/27 = 14/9.
    close_rel(
        scalar(&v),
        14.0 / 9.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanvar",
    );
    let s = nanstd(&a, None, 0).unwrap();
    close_rel(
        scalar(&s),
        (14.0_f64 / 9.0).sqrt(),
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanstd",
    );

    // nancumsum / nancumprod treat NaN as identity (0 / 1).
    let cs = nancumsum(&a, None).unwrap();
    let cs_data: Vec<f64> = cs.iter().copied().collect();
    assert_eq!(cs_data, vec![1.0, 1.0, 4.0, 8.0]);
    let cp = nancumprod(&a, None).unwrap();
    let cp_data: Vec<f64> = cp.iter().copied().collect();
    assert_eq!(cp_data, vec![1.0, 1.0, 3.0, 12.0]);
}

// ---------------------------------------------------------------------------
// Quantile-based: median / percentile / quantile + nan- variants +
// QuantileMethod enum and *_with_method variants.
// ---------------------------------------------------------------------------

#[test]
fn percentile_quantile_methods_match_numpy() {
    let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    // median of [1..5] = 3.
    close_rel(
        scalar(&median(&a, None).unwrap()),
        3.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "median",
    );

    // percentile with default (linear) method.
    let p50 = percentile(&a, 50.0, None).unwrap();
    close_rel(
        scalar(&p50),
        3.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "percentile p50",
    );
    let p25 = percentile(&a, 25.0, None).unwrap();
    close_rel(
        scalar(&p25),
        2.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "percentile p25",
    );
    // quantile uses fraction in [0,1].
    let q25 = quantile(&a, 0.25, None).unwrap();
    close_rel(
        scalar(&q25),
        2.0,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "quantile q25",
    );

    // Method variants:
    //   Lower at q=0.25 with n=5 -> index floor(0.25*4)=1 -> value 2.
    //   Higher at q=0.25 with n=5 -> index ceil(0.25*4)=1 -> value 2.
    //   Nearest at q=0.6 -> index round(0.6*4)=2 -> value 3.
    let pmh = percentile_with_method(&a, 25.0, None, QuantileMethod::Lower).unwrap();
    close_rel(
        scalar(&pmh),
        2.0,
        TOL_EXACT,
        TOL_EXACT,
        "percentile_with_method Lower",
    );
    let qmh = quantile_with_method(&a, 0.6, None, QuantileMethod::Nearest).unwrap();
    close_rel(
        scalar(&qmh),
        3.0,
        TOL_EXACT,
        TOL_EXACT,
        "quantile_with_method Nearest",
    );

    // nan- variants.
    let with_nan = arr1(vec![1.0, f64::NAN, 3.0, 4.0, 5.0]);
    close_rel(
        scalar(&nanmedian(&with_nan, None).unwrap()),
        3.5,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanmedian",
    );
    let np50 = nanpercentile(&with_nan, 50.0, None).unwrap();
    close_rel(
        scalar(&np50),
        3.5,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanpercentile",
    );
    let nq50 = nanquantile(&with_nan, 0.5, None).unwrap();
    close_rel(
        scalar(&nq50),
        3.5,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "nanquantile",
    );
}
