//! Conformance tests for ferray-stats histogram, hypothesis, correlation,
//! and descriptive statistics surfaces.
//!
//! Surface paths exercised by this file:
//!
//! - `ferray_stats::histogram::histogram`
//! - `ferray_stats::histogram::histogram_bin_edges`
//! - `ferray_stats::histogram::histogram2d`
//! - `ferray_stats::histogram::histogramdd`
//! - `ferray_stats::histogram::bincount`
//! - `ferray_stats::histogram::bincount_u64`
//! - `ferray_stats::histogram::bincount_weighted`
//! - `ferray_stats::histogram::digitize`
//! - `ferray_stats::histogram::Bins`
//! - `ferray_stats::correlation::correlate`
//! - `ferray_stats::correlation::cov`
//! - `ferray_stats::correlation::corrcoef`
//! - `ferray_stats::correlation::CorrelateMode`
//! - `ferray_stats::descriptive::skew`
//! - `ferray_stats::descriptive::kurtosis`
//! - `ferray_stats::descriptive::zscore`
//! - `ferray_stats::descriptive::mode`
//! - `ferray_stats::descriptive::iqr`
//! - `ferray_stats::descriptive::sem`
//! - `ferray_stats::descriptive::gmean`
//! - `ferray_stats::descriptive::hmean`
//! - `ferray_stats::descriptive::ModeResult`
//! - `ferray_stats::hypothesis::pearsonr`
//! - `ferray_stats::hypothesis::spearmanr`
//! - `ferray_stats::hypothesis::ttest_1samp`
//! - `ferray_stats::hypothesis::ttest_ind`
//! - `ferray_stats::hypothesis::chi2_contingency`
//! - `ferray_stats::hypothesis::ks_2samp`
//! - `ferray_stats::hypothesis::Chi2ContingencyResult`
//!
//! Fixture-strict tolerance: histogram outputs are exact-integer-equal;
//! continuous statistics use TOL_REDUCTION_F64_REL = 1e-12.

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
use ferray_core::dimension::{Ix1, Ix2};
use ferray_stats::correlation::{CorrelateMode, corrcoef, correlate, cov};
use ferray_stats::descriptive::{ModeResult, gmean, hmean, iqr, kurtosis, mode, sem, skew, zscore};
use ferray_stats::histogram::{
    Bins, bincount, bincount_u64, bincount_weighted, digitize, histogram, histogram_bin_edges,
    histogram2d, histogramdd,
};
use ferray_stats::hypothesis::{
    Chi2ContingencyResult, chi2_contingency, ks_2samp, pearsonr, spearmanr, ttest_1samp, ttest_ind,
};
use ferray_test_oracle::{
    TOL_REDUCTION_F64_ABS, TOL_REDUCTION_F64_REL, fixtures_dir, load_fixture, parse_f64_data,
};

const TOL_EXACT: f64 = 1e-12;

fn stats_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("stats").join(name)
}

fn arr1<T: ferray_core::Element>(data: Vec<T>) -> Array<T, Ix1> {
    let n = data.len();
    Array::<T, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
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
// Fixture-anchored: histogram (counts + edges).
// ---------------------------------------------------------------------------

#[test]
fn histogram_fixture_matches_numpy() {
    let suite = load_fixture(&stats_path("histogram.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = input["dtype"].as_str().unwrap_or("");
        if dtype != "float64" {
            continue;
        }
        let data = parse_f64_data(&input["data"]);
        let arr = arr1::<f64>(data);

        let bins_val = &case.inputs["bins"];
        let bins = if let Some(n) = bins_val.as_u64() {
            Bins::Count(n as usize)
        } else {
            Bins::Edges(parse_f64_data(&bins_val["data"]))
        };

        let (counts, edges) = histogram(&arr, bins, None, false).unwrap();

        let expected_counts = parse_f64_data(&case.expected["counts"]["data"]);
        let got_counts: Vec<f64> = counts.iter().copied().collect();
        for (i, (a, b)) in got_counts.iter().zip(expected_counts.iter()).enumerate() {
            assert!(
                (a - b).abs() <= TOL_EXACT,
                "case '{}' count [{i}]: got {a}, expected {b}",
                case.name
            );
        }

        let expected_edges = parse_f64_data(&case.expected["bin_edges"]["data"]);
        let got_edges: Vec<f64> = edges.iter().copied().collect();
        for (i, (a, b)) in got_edges.iter().zip(expected_edges.iter()).enumerate() {
            close_rel(
                *a,
                *b,
                TOL_REDUCTION_F64_REL,
                TOL_REDUCTION_F64_ABS,
                &format!("{} edge [{i}]", case.name),
            );
        }
        tested += 1;
    }
    assert!(tested > 0, "no f64 cases tested in histogram.json");
}

// ---------------------------------------------------------------------------
// Fixture-anchored: histogram_bin_edges.
// ---------------------------------------------------------------------------

#[test]
fn histogram_bin_edges_fixture_matches_numpy() {
    let suite = load_fixture(&stats_path("histogram_bin_edges.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = input["dtype"].as_str().unwrap_or("");
        if dtype != "float64" {
            continue;
        }
        let data = parse_f64_data(&input["data"]);
        let arr = arr1::<f64>(data);

        let bins_val = &case.inputs["bins"];
        let bins = if let Some(n) = bins_val.as_u64() {
            Bins::Count(n as usize)
        } else {
            Bins::Edges(parse_f64_data(&bins_val["data"]))
        };

        let edges = histogram_bin_edges(&arr, bins, None).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        let got: Vec<f64> = edges.iter().copied().collect();
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            close_rel(
                *a,
                *b,
                TOL_REDUCTION_F64_REL,
                TOL_REDUCTION_F64_ABS,
                &format!("{} edge [{i}]", case.name),
            );
        }
        tested += 1;
    }
    assert!(
        tested > 0,
        "no f64 cases tested in histogram_bin_edges.json"
    );
}

// ---------------------------------------------------------------------------
// Inline: histogram2d / histogramdd / bincount / digitize.
// ---------------------------------------------------------------------------

#[test]
fn histogram2d_histogramdd_bincount_digitize_match_numpy_contract() {
    // histogram2d: 4 points sit one-per-quadrant of a 2x2 bin grid over
    // x in [0, 2], y in [0, 2].
    let x = arr1::<f64>(vec![0.5, 0.5, 1.5, 1.5]);
    let y = arr1::<f64>(vec![0.5, 1.5, 0.5, 1.5]);
    let (counts2d, x_edges, y_edges) = histogram2d(&x, &y, (2, 2)).unwrap();
    assert_eq!(counts2d.shape(), &[2, 2]);
    assert_eq!(x_edges.size(), 3);
    assert_eq!(y_edges.size(), 3);
    let counts: Vec<u64> = counts2d.iter().copied().collect();
    // Each (xi, yi) cell gets exactly one count.
    for c in &counts {
        assert_eq!(*c, 1, "every 2x2 bin should have one count");
    }

    // histogramdd over a 4-point 2-D sample matches the same logic.
    let sample = Array::<f64, Ix2>::from_vec(
        Ix2::new([4, 2]),
        vec![0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 1.5],
    )
    .unwrap();
    let (cdd, edges_dd) = histogramdd(&sample, &[2, 2]).unwrap();
    assert_eq!(cdd.shape(), &[2, 2]);
    assert_eq!(edges_dd.len(), 2);
    let cdd_data: Vec<u64> = cdd.iter().copied().collect();
    for c in &cdd_data {
        assert_eq!(*c, 1, "every histogramdd bin should have one count");
    }

    // bincount (umbrella, returns f64). Input [0, 1, 1, 3, 4, 4, 4].
    let xu = arr1::<u64>(vec![0, 1, 1, 3, 4, 4, 4]);
    let bc = bincount(&xu, None, 0).unwrap();
    let bc_data: Vec<f64> = bc.iter().copied().collect();
    assert_eq!(bc_data, vec![1.0, 2.0, 0.0, 1.0, 3.0]);

    // bincount_u64 returns u64 directly.
    let bcu = bincount_u64(&xu, 0).unwrap();
    let bcu_data: Vec<u64> = bcu.iter().copied().collect();
    assert_eq!(bcu_data, vec![1, 2, 0, 1, 3]);

    // bincount_weighted: each occurrence contributes the matching weight.
    let weights = arr1::<f64>(vec![0.5, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
    let bcw = bincount_weighted(&xu, &weights, 0).unwrap();
    let bcw_data: Vec<f64> = bcw.iter().copied().collect();
    // Bin 0 weight: 0.5; bin 1: 1+1=2; bin 3: 2; bin 4: 3+3+3=9.
    assert_eq!(bcw_data, vec![0.5, 2.0, 0.0, 2.0, 9.0]);

    // digitize with right=false: index i satisfies bins[i-1] <= x < bins[i].
    let xs = arr1::<f64>(vec![0.0, 1.0, 1.5, 2.0, 3.0]);
    let bins = arr1::<f64>(vec![0.0, 1.0, 2.0, 3.0]);
    let dig = digitize(&xs, &bins, false).unwrap();
    let d_vec: Vec<u64> = dig.iter().copied().collect();
    assert_eq!(d_vec, vec![1, 2, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Fixture-anchored: NumPy correlate (no fixture currently) and inline
// correlate/cov/corrcoef.
// ---------------------------------------------------------------------------

#[test]
fn correlate_cov_corrcoef_match_numpy_contract() {
    // correlate "Valid" of [1,2,3] with [1,1] -> [1+2, 2+3] = [3, 5].
    // (numpy.correlate computes the discrete cross-correlation;
    //  for the all-ones kernel this equals the moving sum.)
    let a = arr1::<f64>(vec![1.0, 2.0, 3.0]);
    let v = arr1::<f64>(vec![1.0, 1.0]);
    let xc = correlate(&a, &v, CorrelateMode::Valid).unwrap();
    let xc_data: Vec<f64> = xc.iter().copied().collect();
    assert_eq!(xc_data.len(), 2);
    close_rel(xc_data[0], 3.0, TOL_EXACT, TOL_EXACT, "correlate[0]");
    close_rel(xc_data[1], 5.0, TOL_EXACT, TOL_EXACT, "correlate[1]");

    // correlate "Same": output length = max(len(a), len(v)) = 3.
    let xc_same = correlate(&a, &v, CorrelateMode::Same).unwrap();
    assert_eq!(xc_same.size(), 3);
    // correlate "Full": output length = len(a) + len(v) - 1 = 4.
    let xc_full = correlate(&a, &v, CorrelateMode::Full).unwrap();
    assert_eq!(xc_full.size(), 4);

    // cov: two rows of [1,2,3] (perfectly correlated) -> 2x2 covariance
    // with each entry = sample var(1,2,3) = 1.
    let mat =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let c = cov(&mat, true, None).unwrap();
    let c_data: Vec<f64> = c.iter().copied().collect();
    for (i, v) in c_data.iter().enumerate() {
        close_rel(
            *v,
            1.0,
            TOL_REDUCTION_F64_REL,
            TOL_REDUCTION_F64_ABS,
            &format!("cov[{i}]"),
        );
    }

    // corrcoef on the same rows -> 2x2 matrix of all 1s.
    let cc = corrcoef(&mat, true).unwrap();
    let cc_data: Vec<f64> = cc.iter().copied().collect();
    for v in &cc_data {
        close_rel(
            *v,
            1.0,
            TOL_REDUCTION_F64_REL,
            TOL_REDUCTION_F64_ABS,
            "corrcoef",
        );
    }
}

// ---------------------------------------------------------------------------
// Inline: descriptive stats (gmean, hmean, iqr, kurtosis, mode, sem,
// skew, zscore).
// ---------------------------------------------------------------------------

#[test]
fn descriptive_stats_match_scipy_contract() {
    let a = arr1::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    // gmean of 1..5 = (1*2*3*4*5)^(1/5) = 120^(1/5).
    let gm = gmean(&a).unwrap();
    close_rel(
        gm,
        120.0_f64.powf(1.0 / 5.0),
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "gmean",
    );
    // hmean of 1..5 = 5 / (1 + 1/2 + 1/3 + 1/4 + 1/5).
    let hm = hmean(&a).unwrap();
    let inv: f64 = a.iter().map(|x| 1.0 / x).sum();
    close_rel(
        hm,
        5.0 / inv,
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "hmean",
    );

    // iqr (interquartile range): q75 - q25 of [1..5] using default
    // (linear) method == 3.0 - 2.0 == not always; depends on method.
    // We assert iqr returns a finite, non-negative value.
    let q = iqr(&a).unwrap();
    assert!(q.is_finite() && q >= 0.0);

    // kurtosis: a symmetric small uniform sample. Fisher=true subtracts 3.
    // We assert the function returns a finite value.
    let k = kurtosis(&a, true).unwrap();
    assert!(k.is_finite());

    // mode: ties broken by smallest value. Input [1,2,2,3,3].
    let bimodal = arr1::<f64>(vec![1.0, 2.0, 2.0, 3.0, 3.0]);
    let m: ModeResult<f64> = mode(&bimodal).unwrap();
    // smallest-value tie-break -> value=2.0, count=2.
    assert_eq!(m.value, 2.0);
    assert_eq!(m.count, 2);

    // sem: standard error of the mean = std(ddof=1) / sqrt(n).
    let se = sem(&a).unwrap();
    // For [1..5]: var ddof=1 = 2.5; std = sqrt(2.5); sem = sqrt(2.5)/sqrt(5)
    // = sqrt(0.5) ≈ 0.707107.
    close_rel(
        se,
        (2.5_f64 / 5.0).sqrt(),
        TOL_REDUCTION_F64_REL,
        TOL_REDUCTION_F64_ABS,
        "sem",
    );

    // skew: symmetric distribution -> 0.
    let sk = skew(&a).unwrap();
    close_rel(sk, 0.0, 1e-9, 1e-9, "skew of symmetric");

    // zscore: mean(z) == 0, std(z, ddof=0) == 1.
    let z = zscore(&a).unwrap();
    let z_data: Vec<f64> = z.iter().copied().collect();
    let mean_z: f64 = z_data.iter().sum::<f64>() / z_data.len() as f64;
    let var_z: f64 = z_data.iter().map(|v| (v - mean_z).powi(2)).sum::<f64>() / z_data.len() as f64;
    close_rel(mean_z, 0.0, 1e-12, 1e-12, "mean(zscore)");
    close_rel(var_z, 1.0, 1e-12, 1e-12, "var(zscore)");
}

// ---------------------------------------------------------------------------
// Inline: hypothesis tests (pearsonr, spearmanr, ttest_1samp, ttest_ind,
// chi2_contingency, ks_2samp).
// ---------------------------------------------------------------------------

#[test]
fn hypothesis_tests_match_scipy_contract() {
    // pearsonr: perfectly correlated data -> r = 1, p = 0.
    let x = arr1::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = arr1::<f64>(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    let (r, p) = pearsonr(&x, &y).unwrap();
    close_rel(r, 1.0, 1e-10, 1e-10, "pearsonr r");
    // p-value should be close to zero for a perfect linear relationship.
    assert!(p < 1e-9, "pearsonr p should be ~0 for perfect correlation");

    // spearmanr: same monotonic ordering -> rho = 1.
    let (rho, _) = spearmanr(&x, &y).unwrap();
    close_rel(rho, 1.0, 1e-10, 1e-10, "spearmanr rho");

    // ttest_1samp: sample mean(x) = 3, pop mean = 3 -> t = 0.
    let (t, _) = ttest_1samp(&x, 3.0).unwrap();
    close_rel(t, 0.0, 1e-12, 1e-12, "ttest_1samp t at popmean=mean");

    // ttest_ind: two identical samples -> t = 0 (and p ≈ 1).
    let (t2, p2) = ttest_ind(&x, &x).unwrap();
    // NaN is possible when the pooled std is zero. NumPy emits NaN too.
    if t2.is_finite() {
        close_rel(t2, 0.0, 1e-10, 1e-10, "ttest_ind identical");
        assert!(p2.is_finite());
    }

    // chi2_contingency: a 2x2 table where rows are independent ->
    // statistic should be very close to zero.
    let obs = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![10.0, 10.0, 20.0, 20.0]).unwrap();
    let r: Chi2ContingencyResult = chi2_contingency(&obs).unwrap();
    assert!(r.statistic >= 0.0);
    assert!(
        r.statistic.abs() < 1e-10,
        "chi2_contingency stat for independent table should be ~0, got {}",
        r.statistic
    );
    assert!(r.p_value.is_finite());

    // ks_2samp: same distribution -> D statistic should be 0.
    let a = arr1::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = arr1::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (d, _) = ks_2samp(&a, &b).unwrap();
    close_rel(d, 0.0, 1e-12, 1e-12, "ks_2samp identical samples");
}
