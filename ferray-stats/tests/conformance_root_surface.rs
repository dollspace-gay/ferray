//! Conformance coverage for the crate-root `ferray_stats::*` re-export surface.
//!
//! The inner module paths are already covered by fixture-backed conformance
//! tests. This file pins the flat crate-root aliases as callable public API and
//! checks representative aliases against their canonical inner implementations.

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2};

fn arr1<T: ferray_core::Element>(data: Vec<T>) -> Array<T, Ix1> {
    let n = data.len();
    Array::<T, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr2<T: ferray_core::Element>(rows: usize, cols: usize, data: Vec<T>) -> Array<T, Ix2> {
    Array::<T, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

fn assert_f64_slices_close(actual: &[f64], expected: &[f64], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-12, "{name}[{i}]: got {a}, expected {e}");
    }
}

/// Path anchors:
/// `ferray_stats::Bins`, `ferray_stats::Chi2ContingencyResult`,
/// `ferray_stats::CorrelateMode`, `ferray_stats::ModeResult`,
/// `ferray_stats::QuantileMethod`, `ferray_stats::Side`,
/// `ferray_stats::SortKind`, `ferray_stats::UniqueResult`,
/// `ferray_stats::argpartition`, `ferray_stats::argsort`,
/// `ferray_stats::average`, `ferray_stats::bincount`,
/// `ferray_stats::bincount_u64`, `ferray_stats::bincount_weighted`,
/// `ferray_stats::chi2_contingency`, `ferray_stats::corrcoef`,
/// `ferray_stats::correlate`, `ferray_stats::count_nonzero`,
/// `ferray_stats::cov`, `ferray_stats::cumprod`, `ferray_stats::cumsum`,
/// `ferray_stats::digitize`, `ferray_stats::gmean`,
/// `ferray_stats::histogram2d`, `ferray_stats::histogram_bin_edges`,
/// `ferray_stats::histogramdd`, `ferray_stats::hmean`,
/// `ferray_stats::in1d`, `ferray_stats::intersect1d`,
/// `ferray_stats::iqr`, `ferray_stats::isin`, `ferray_stats::ks_2samp`,
/// `ferray_stats::kurtosis`, `ferray_stats::lexsort`,
/// `ferray_stats::max`, `ferray_stats::max_into`,
/// `ferray_stats::max_with`, `ferray_stats::mean`,
/// `ferray_stats::mean_as_f64`, `ferray_stats::mean_into`,
/// `ferray_stats::mean_where`, `ferray_stats::median`,
/// `ferray_stats::min`, `ferray_stats::min_into`,
/// `ferray_stats::min_with`, `ferray_stats::mode`,
/// `ferray_stats::nanargmax`, `ferray_stats::nanargmin`,
/// `ferray_stats::nancumprod`, `ferray_stats::nancumsum`,
/// `ferray_stats::nanmax`, `ferray_stats::nanmean`,
/// `ferray_stats::nanmedian`, `ferray_stats::nanmin`,
/// `ferray_stats::nanpercentile`, `ferray_stats::nanprod`,
/// `ferray_stats::nanquantile`, `ferray_stats::nanstd`,
/// `ferray_stats::nansum`, `ferray_stats::nanvar`,
/// `ferray_stats::nonzero`, `ferray_stats::partition`,
/// `ferray_stats::pearsonr`, `ferray_stats::percentile`,
/// `ferray_stats::percentile_with_method`, `ferray_stats::prod`,
/// `ferray_stats::prod_into`, `ferray_stats::prod_with`,
/// `ferray_stats::ptp`, `ferray_stats::quantile`,
/// `ferray_stats::quantile_with_method`, `ferray_stats::searchsorted`,
/// `ferray_stats::searchsorted_with_sorter`, `ferray_stats::sem`,
/// `ferray_stats::setdiff1d`, `ferray_stats::setxor1d`,
/// `ferray_stats::skew`, `ferray_stats::sort_complex`,
/// `ferray_stats::spearmanr`, `ferray_stats::std_`,
/// `ferray_stats::std_into`, `ferray_stats::sum`,
/// `ferray_stats::sum_as_f64`, `ferray_stats::sum_into`,
/// `ferray_stats::sum_with`, `ferray_stats::ttest_1samp`,
/// `ferray_stats::ttest_ind`, `ferray_stats::union1d`,
/// `ferray_stats::unique`, `ferray_stats::unique_all`,
/// `ferray_stats::unique_axis`, `ferray_stats::unique_counts`,
/// `ferray_stats::unique_inverse`, `ferray_stats::unique_values`,
/// `ferray_stats::var`, `ferray_stats::var_into`, `ferray_stats::where_`,
/// `ferray_stats::where_broadcast`, `ferray_stats::where_condition`,
/// `ferray_stats::zscore`.
#[test]
fn crate_root_aliases_resolve_as_public_surface() {
    fn touch<T>(_value: T) {}

    touch(ferray_stats::argpartition::<f64>);
    touch(ferray_stats::argsort::<f64, Ix1>);
    touch(ferray_stats::average::<f64, Ix1>);
    touch(ferray_stats::bincount);
    touch(ferray_stats::bincount_u64);
    touch(ferray_stats::bincount_weighted);
    touch(ferray_stats::chi2_contingency::<f64>);
    touch(ferray_stats::corrcoef::<f64, Ix2>);
    touch(ferray_stats::correlate::<f64>);
    touch(ferray_stats::count_nonzero::<f64, Ix1>);
    touch(ferray_stats::cov::<f64, Ix2>);
    touch(ferray_stats::cumprod::<f64, Ix1>);
    touch(ferray_stats::cumsum::<f64, Ix1>);
    touch(ferray_stats::digitize::<f64>);
    touch(ferray_stats::gmean::<f64, Ix1>);
    touch(ferray_stats::histogram::<f64>);
    touch(ferray_stats::histogram2d::<f64>);
    touch(ferray_stats::histogram_bin_edges::<f64>);
    touch(ferray_stats::histogramdd::<f64>);
    touch(ferray_stats::hmean::<f64, Ix1>);
    touch(ferray_stats::in1d::<f64>);
    touch(ferray_stats::intersect1d::<f64>);
    touch(ferray_stats::iqr::<f64, Ix1>);
    touch(ferray_stats::isin::<f64>);
    touch(ferray_stats::ks_2samp::<f64, Ix1>);
    touch(ferray_stats::kurtosis::<f64, Ix1>);
    touch(ferray_stats::lexsort::<f64>);
    touch(ferray_stats::max::<f64, Ix1>);
    touch(ferray_stats::max_into::<f64, Ix2>);
    touch(ferray_stats::max_with::<f64, Ix1>);
    touch(ferray_stats::mean::<f64, Ix1>);
    touch(ferray_stats::mean_as_f64::<f64, Ix1>);
    touch(ferray_stats::mean_into::<f64, Ix2>);
    touch(ferray_stats::mean_where::<f64, Ix1>);
    touch(ferray_stats::median::<f64, Ix1>);
    touch(ferray_stats::min::<f64, Ix1>);
    touch(ferray_stats::min_into::<f64, Ix2>);
    touch(ferray_stats::min_with::<f64, Ix1>);
    touch(ferray_stats::mode::<f64, Ix1>);
    touch(ferray_stats::nanargmax::<f64, Ix1>);
    touch(ferray_stats::nanargmin::<f64, Ix1>);
    touch(ferray_stats::nancumprod::<f64, Ix1>);
    touch(ferray_stats::nancumsum::<f64, Ix1>);
    touch(ferray_stats::nanmax::<f64, Ix1>);
    touch(ferray_stats::nanmean::<f64, Ix1>);
    touch(ferray_stats::nanmedian::<f64, Ix1>);
    touch(ferray_stats::nanmin::<f64, Ix1>);
    touch(ferray_stats::nanpercentile::<f64, Ix1>);
    touch(ferray_stats::nanprod::<f64, Ix1>);
    touch(ferray_stats::nanquantile::<f64, Ix1>);
    touch(ferray_stats::nanstd::<f64, Ix1>);
    touch(ferray_stats::nansum::<f64, Ix1>);
    touch(ferray_stats::nanvar::<f64, Ix1>);
    touch(ferray_stats::nonzero::<f64, Ix1>);
    touch(ferray_stats::partition::<f64>);
    touch(ferray_stats::pearsonr::<f64, Ix1>);
    touch(ferray_stats::percentile::<f64, Ix1>);
    touch(ferray_stats::percentile_with_method::<f64, Ix1>);
    touch(ferray_stats::prod::<f64, Ix1>);
    touch(ferray_stats::prod_into::<f64, Ix2>);
    touch(ferray_stats::prod_with::<f64, Ix1>);
    touch(ferray_stats::ptp::<f64, Ix1>);
    touch(ferray_stats::quantile::<f64, Ix1>);
    touch(ferray_stats::quantile_with_method::<f64, Ix1>);
    touch(ferray_stats::searchsorted::<f64>);
    touch(ferray_stats::searchsorted_with_sorter::<f64>);
    touch(ferray_stats::sem::<f64, Ix1>);
    touch(ferray_stats::setdiff1d::<f64>);
    touch(ferray_stats::setxor1d::<f64>);
    touch(ferray_stats::skew::<f64, Ix1>);
    touch(ferray_stats::sort::<f64, Ix1>);
    touch(ferray_stats::sort_complex::<f64>);
    touch(ferray_stats::spearmanr::<f64, Ix1>);
    touch(ferray_stats::std_::<f64, Ix1>);
    touch(ferray_stats::std_into::<f64, Ix2>);
    touch(ferray_stats::sum::<f64, Ix1>);
    touch(ferray_stats::sum_as_f64::<f64, Ix1>);
    touch(ferray_stats::sum_into::<f64, Ix2>);
    touch(ferray_stats::sum_with::<f64, Ix1>);
    touch(ferray_stats::ttest_1samp::<f64, Ix1>);
    touch(ferray_stats::ttest_ind::<f64, Ix1>);
    touch(ferray_stats::union1d::<f64>);
    touch(ferray_stats::unique::<f64, Ix1>);
    touch(ferray_stats::unique_all::<f64, Ix1>);
    touch(ferray_stats::unique_axis::<f64, Ix2>);
    touch(ferray_stats::unique_counts::<f64, Ix1>);
    touch(ferray_stats::unique_inverse::<f64, Ix1>);
    touch(ferray_stats::unique_values::<f64, Ix1>);
    touch(ferray_stats::var::<f64, Ix1>);
    touch(ferray_stats::var_into::<f64, Ix2>);
    touch(ferray_stats::where_::<f64, Ix1>);
    touch(ferray_stats::where_broadcast::<f64>);
    touch(ferray_stats::where_condition::<Ix1>);
    touch(ferray_stats::zscore::<f64, Ix1>);

    let _ = ferray_stats::Bins::<f64>::Count(3);
    let _ = ferray_stats::CorrelateMode::Valid;
    let _ = ferray_stats::QuantileMethod::Linear;
    let _ = ferray_stats::Side::Left;
    let _ = ferray_stats::SortKind::Quick;
    let _: Option<ferray_stats::Chi2ContingencyResult> = None;
    let _: Option<ferray_stats::ModeResult<f64>> = None;
    let _: Option<ferray_stats::UniqueResult<f64>> = None;
}

#[test]
fn root_reductions_match_canonical_inner_paths() {
    let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);

    let root_sum = ferray_stats::sum(&a, None).unwrap();
    let inner_sum = ferray_stats::reductions::sum(&a, None).unwrap();
    assert_f64_slices_close(
        root_sum.as_slice().unwrap(),
        inner_sum.as_slice().unwrap(),
        "sum",
    );

    let root_median = ferray_stats::median(&a, None).unwrap();
    let inner_median = ferray_stats::reductions::quantile::median(&a, None).unwrap();
    assert_f64_slices_close(
        root_median.as_slice().unwrap(),
        inner_median.as_slice().unwrap(),
        "median",
    );

    let with_nan = arr1(vec![1.0, f64::NAN, 3.0]);
    let root_nansum = ferray_stats::nansum(&with_nan, None).unwrap();
    let inner_nansum = ferray_stats::reductions::nan_aware::nansum(&with_nan, None).unwrap();
    assert_f64_slices_close(
        root_nansum.as_slice().unwrap(),
        inner_nansum.as_slice().unwrap(),
        "nansum",
    );
}

#[test]
fn root_search_sort_set_aliases_match_canonical_inner_paths() {
    let a = arr1(vec![3.0, 1.0, 2.0, 1.0]);

    let root_sort = ferray_stats::sort(&a, None, ferray_stats::SortKind::Quick).unwrap();
    let inner_sort =
        ferray_stats::sorting::sort(&a, None, ferray_stats::sorting::SortKind::Quick).unwrap();
    assert_f64_slices_close(
        root_sort.as_slice().unwrap(),
        inner_sort.as_slice().unwrap(),
        "sort",
    );

    let root_unique = ferray_stats::unique_values(&a).unwrap();
    let inner_unique = ferray_stats::searching::unique_values(&a).unwrap();
    assert_f64_slices_close(
        root_unique.as_slice().unwrap(),
        inner_unique.as_slice().unwrap(),
        "unique_values",
    );

    let b = arr1(vec![2.0, 4.0]);
    let root_union = ferray_stats::union1d(&a, &b, false).unwrap();
    let inner_union = ferray_stats::set_ops::union1d(&a, &b, false).unwrap();
    assert_f64_slices_close(
        root_union.as_slice().unwrap(),
        inner_union.as_slice().unwrap(),
        "union1d",
    );
}

#[test]
fn root_histogram_descriptive_and_hypothesis_aliases_match_inner_paths() {
    let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);

    let (root_counts, root_edges) =
        ferray_stats::histogram(&a, ferray_stats::Bins::Count(2), None, false).unwrap();
    let (inner_counts, inner_edges) = ferray_stats::histogram::histogram(
        &a,
        ferray_stats::histogram::Bins::Count(2),
        None,
        false,
    )
    .unwrap();
    assert_f64_slices_close(
        root_counts.as_slice().unwrap(),
        inner_counts.as_slice().unwrap(),
        "histogram counts",
    );
    assert_f64_slices_close(
        root_edges.as_slice().unwrap(),
        inner_edges.as_slice().unwrap(),
        "histogram edges",
    );

    let root_mode = ferray_stats::mode(&a).unwrap();
    let inner_mode = ferray_stats::descriptive::mode(&a).unwrap();
    assert_eq!(root_mode.value, inner_mode.value);
    assert_eq!(root_mode.count, inner_mode.count);

    let y = arr1(vec![2.0, 4.0, 6.0, 8.0]);
    let root_pearson = ferray_stats::pearsonr(&a, &y).unwrap();
    let inner_pearson = ferray_stats::hypothesis::pearsonr(&a, &y).unwrap();
    assert!((root_pearson.0 - inner_pearson.0).abs() < 1e-12);
    assert!((root_pearson.1 - inner_pearson.1).abs() < 1e-12);

    let obs = arr2(2, 2, vec![10.0, 10.0, 20.0, 20.0]);
    let root_chi = ferray_stats::chi2_contingency(&obs).unwrap();
    let inner_chi = ferray_stats::hypothesis::chi2_contingency(&obs).unwrap();
    assert!((root_chi.statistic - inner_chi.statistic).abs() < 1e-12);
    assert_eq!(root_chi.dof, inner_chi.dof);
}
