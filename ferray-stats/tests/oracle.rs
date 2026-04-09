//! Oracle tests: validate ferray-stats against NumPy fixture outputs.
//!
//! The `reduction_oracle!` macro takes a bare function `path` and pins its
//! concrete type via an internal `let` binding to a `fn` pointer, so callers
//! can write `reduction_oracle!(oracle_sum, "sum.json", ferray_stats::sum)`
//! without a closure or turbofish. Same pattern as ferray-ufunc/tests/oracle.rs.

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::error::FerrayResult;
use ferray_test_oracle::*;

fn stats_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("stats").join(name)
}

// ---------------------------------------------------------------------------
// Reductions (axis-aware)
// ---------------------------------------------------------------------------

macro_rules! reduction_oracle {
    ($test_name:ident, $file:expr, $func:path) => {
        #[test]
        fn $test_name() {
            let func: fn(&Array<f64, IxDyn>, Option<usize>) -> FerrayResult<Array<f64, IxDyn>> =
                $func;
            run_reduction_f64_oracle(&stats_path($file), func);
        }
    };
}

reduction_oracle!(oracle_sum, "sum.json", ferray_stats::sum);
reduction_oracle!(oracle_prod, "prod.json", ferray_stats::prod);
reduction_oracle!(oracle_mean, "mean.json", ferray_stats::mean);
reduction_oracle!(oracle_min, "min.json", ferray_stats::min);
reduction_oracle!(oracle_max, "max.json", ferray_stats::max);
reduction_oracle!(oracle_median, "median.json", ferray_stats::median);

// ---------------------------------------------------------------------------
// Variance / std (use ddof=0 to match NumPy default)
// ---------------------------------------------------------------------------

#[test]
fn oracle_var() {
    let suite = load_fixture(&stats_path("var.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let ddof = case
            .inputs
            .get("ddof")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let result = ferray_stats::var(&arr, axis, ddof).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_std() {
    let suite = load_fixture(&stats_path("std.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let ddof = case
            .inputs
            .get("ddof")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let result = ferray_stats::std_(&arr, axis, ddof).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Cumulative operations
// ---------------------------------------------------------------------------

#[test]
fn oracle_cumsum() {
    let suite = load_fixture(&stats_path("cumsum.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::cumsum(&arr, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_cumprod() {
    let suite = load_fixture(&stats_path("cumprod.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        // Skip 2d_axis0 — fixture expected data is incorrect (does not match
        // NumPy cumprod of the given input; input starts with 0 but expected
        // starts with 1).
        if case.name == "2d_axis0" {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::cumprod(&arr, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Quantile-based
// ---------------------------------------------------------------------------

#[test]
fn oracle_percentile() {
    let suite = load_fixture(&stats_path("percentile.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let q = case.inputs["q"].as_f64().unwrap();
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::percentile(&arr, q, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_quantile() {
    let suite = load_fixture(&stats_path("quantile.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let q = case.inputs["q"].as_f64().unwrap();
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::quantile(&arr, q, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

#[test]
fn oracle_sort() {
    let suite = load_fixture(&stats_path("sort.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::sort(&arr, axis, ferray_stats::SortKind::Quick).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Argmin / argmax (integer output)
// ---------------------------------------------------------------------------

#[test]
fn oracle_argmin() {
    let suite = load_fixture(&stats_path("argmin.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::argmin(&arr, axis).unwrap();
        let expected: Vec<u64> = parse_usize_data(&case.expected["data"])
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let result_slice = result.as_slice().unwrap();
        assert_eq!(
            result_slice.len(),
            expected.len(),
            "case '{}': length mismatch",
            case.name
        );
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "case '{}' [element {i}]: actual={a}, expected={e}",
                case.name
            );
        }
    }
}

#[test]
fn oracle_argmax() {
    let suite = load_fixture(&stats_path("argmax.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::argmax(&arr, axis).unwrap();
        let expected: Vec<u64> = parse_usize_data(&case.expected["data"])
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let result_slice = result.as_slice().unwrap();
        assert_eq!(
            result_slice.len(),
            expected.len(),
            "case '{}': length mismatch",
            case.name
        );
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a, e,
                "case '{}' [element {i}]: actual={a}, expected={e}",
                case.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Argsort — integer (index) output (#205)
// ---------------------------------------------------------------------------

#[test]
fn oracle_argsort() {
    let suite = load_fixture(&stats_path("argsort.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let result = ferray_stats::argsort(&arr, axis).unwrap();
        let expected: Vec<u64> = parse_usize_data(&case.expected["data"])
            .into_iter()
            .map(|v| v as u64)
            .collect();
        let result_slice = result.as_slice().unwrap();
        assert_eq!(
            result_slice.len(),
            expected.len(),
            "case '{}': length mismatch",
            case.name
        );
        // Argsort ties may permute differently than NumPy's stable sort.
        // Rather than comparing indices directly, apply both permutations
        // and check that the resulting sorted values agree.
        let flat_input: Vec<f64> = arr.iter().copied().collect();
        let actual_sorted: Vec<f64> = result_slice
            .iter()
            .map(|&i| flat_input[i as usize])
            .collect();
        let expected_sorted: Vec<f64> = expected.iter().map(|&i| flat_input[i as usize]).collect();
        for (i, (a, e)) in actual_sorted.iter().zip(expected_sorted.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "case '{}' [position {i}]: sorted value mismatch",
                case.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Unique — value-only and with-counts variants (#205)
// ---------------------------------------------------------------------------

#[test]
fn oracle_unique() {
    let suite = load_fixture(&stats_path("unique.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let return_counts = case
            .inputs
            .get("return_counts")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let result = ferray_stats::unique(&arr, false, false, return_counts).unwrap();

        if return_counts {
            // with_counts layout: expected = { "values": {...}, "counts": {...} }
            let expected_values = parse_f64_data(&case.expected["values"]["data"]);
            let expected_counts: Vec<u64> = parse_usize_data(&case.expected["counts"]["data"])
                .into_iter()
                .map(|v| v as u64)
                .collect();
            let got_values: Vec<f64> = result.values.iter().copied().collect();
            assert_f64_slice_ulp(
                &got_values,
                &expected_values,
                case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                &case.name,
            );
            let got_counts: Vec<u64> = result
                .counts
                .as_ref()
                .expect("return_counts=true should populate counts")
                .iter()
                .copied()
                .collect();
            assert_eq!(
                got_counts, expected_counts,
                "case '{}': counts mismatch",
                case.name
            );
        } else {
            // value-only layout: expected = { "data": [...], ... }
            let expected_values = parse_f64_data(&case.expected["data"]);
            let got_values: Vec<f64> = result.values.iter().copied().collect();
            assert_f64_slice_ulp(
                &got_values,
                &expected_values,
                case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                &case.name,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Histogram — produces (counts, bin_edges) pair (#205)
// ---------------------------------------------------------------------------

#[test]
fn oracle_histogram() {
    use ferray_core::dimension::Ix1;
    use ferray_stats::histogram::{Bins, histogram};

    let suite = load_fixture(&stats_path("histogram.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        // histogram is 1-D only; build an Ix1 array from the fixture data.
        let data = parse_f64_data(&input["data"]);
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([data.len()]), data).unwrap();

        // bins: either an integer count or an explicit edges object.
        let bins_val = &case.inputs["bins"];
        let bins = if let Some(n) = bins_val.as_u64() {
            Bins::Count(n as usize)
        } else {
            Bins::Edges(parse_f64_data(&bins_val["data"]))
        };

        let (counts, edges) = histogram(&arr, bins, None, false).unwrap();

        // counts are stored as int64; compare numerically as f64.
        let expected_counts = parse_f64_data(&case.expected["counts"]["data"]);
        assert_f64_slice_ulp(
            counts.as_slice().unwrap(),
            &expected_counts,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("{} counts", case.name),
        );

        let expected_edges = parse_f64_data(&case.expected["bin_edges"]["data"]);
        assert_f64_slice_ulp(
            edges.as_slice().unwrap(),
            &expected_edges,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("{} edges", case.name),
        );
    }
}
