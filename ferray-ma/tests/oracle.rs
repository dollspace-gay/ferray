//! Oracle tests: validate ferray-ma (masked arrays) against `NumPy` fixture outputs.

use ferray_core::dimension::IxDyn;
use ferray_ma::MaskedArray;
use ferray_test_oracle::*;

fn ma_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("ma").join(name)
}

/// Build a `MaskedArray` from fixture inputs that have "data" and "mask" fields.
fn make_masked_array(inputs: &ferray_test_oracle::serde_json::Value) -> MaskedArray<f64, IxDyn> {
    let data_arr = make_f64_array(&inputs["data"]);
    let mask_arr = make_bool_array(&inputs["mask"]);
    MaskedArray::new(data_arr, mask_arr).unwrap()
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

#[test]
fn oracle_masked_mean() {
    let suite = load_fixture(&ma_path("masked_mean.json"));
    for case in &suite.test_cases {
        // Skip axis-aware tests for now (masked mean() is scalar-only in ferray-ma)
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let result = ma.mean().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_masked_sum() {
    let suite = load_fixture(&ma_path("masked_sum.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let result = ma.sum().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_masked_min() {
    let suite = load_fixture(&ma_path("masked_min.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let result = ma.min().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_masked_max() {
    let suite = load_fixture(&ma_path("masked_max.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let result = ma.max().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
    }
}

// ---------------------------------------------------------------------------
// Mask operations
// ---------------------------------------------------------------------------

#[test]
fn oracle_getmask_getdata() {
    let suite = load_fixture(&ma_path("getmask_getdata.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);

        // Verify getdata returns the original data
        let data = ferray_ma::getdata(&ma).unwrap();
        let expected_data = parse_f64_data(&case.expected["data"]["data"]);
        assert_f64_slice_ulp(
            data.as_slice().unwrap(),
            &expected_data,
            0,
            &format!("case '{}' getdata", case.name),
        );

        // Verify getmask returns the original mask
        let mask = ferray_ma::getmask(&ma).unwrap();
        let expected_mask = parse_bool_data(&case.expected["mask"]["data"]);
        let mask_slice = mask.as_slice().unwrap();
        for (i, (&a, &e)) in mask_slice.iter().zip(expected_mask.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' getmask [element {i}]", case.name);
        }
    }
}

// ---------------------------------------------------------------------------
// masked_invalid
// ---------------------------------------------------------------------------

#[test]
fn oracle_masked_invalid() {
    let suite = load_fixture(&ma_path("masked_invalid.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["data"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_ma::masked_invalid(&arr).unwrap();
        let expected_mask = parse_bool_data(&case.expected["mask"]["data"]);
        let mask_slice = result.mask().as_slice().unwrap();
        for (i, (&a, &e)) in mask_slice.iter().zip(expected_mask.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' mask [element {i}]", case.name);
        }
    }
}

// ---------------------------------------------------------------------------
// compressed
// ---------------------------------------------------------------------------

#[test]
fn oracle_compressed() {
    let suite = load_fixture(&ma_path("compressed.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        let result = ma.compressed().unwrap();
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
