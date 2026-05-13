//! Conformance tests for ferray-ma: load each numpy.ma fixture and
//! assert ferray's output matches the NumPy oracle within the
//! fixture-declared ULP tolerance.
//!
//! Each test mentions BOTH the canonical inner path (e.g.
//! `ferray_ma::reductions::MaskedArray::sum`) and the user-facing
//! crate-root re-export path (e.g. `ferray_ma::sum`) in its doc
//! comment so the surface-coverage gate's text match picks both up.
//!
//! `numpy.ma` reductions (`sum`/`mean`/`min`/`max`) are exposed as
//! `MaskedArray` methods in ferray, so the "user-facing" path is the
//! method call `ma.sum()` rather than a free function. The surface
//! gate is text-based; mentioning the canonical method path in the
//! doc comment is sufficient.

use ferray_core::dimension::IxDyn;
use ferray_ma::MaskedArray;
use ferray_test_oracle::{
    assert_f64_slice_ulp, assert_f64_ulp, fixtures_dir, load_fixture, make_bool_array,
    make_f64_array, parse_bool_data, parse_f64_data, parse_f64_value, should_skip_f64,
    MIN_ULP_TOLERANCE,
};

fn ma_fx(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("ma").join(name)
}

/// Build a `MaskedArray<f64, IxDyn>` from a fixture case's `data` +
/// `mask` inputs. Used by every reduction / mask-op test below.
fn make_masked_array(
    inputs: &ferray_test_oracle::serde_json::Value,
) -> MaskedArray<f64, IxDyn> {
    let data = make_f64_array(&inputs["data"]);
    let mask = make_bool_array(&inputs["mask"]);
    MaskedArray::new(data, mask).unwrap()
}

// ---------------------------------------------------------------------------
// Reductions: sum, mean, min, max
//
// Axis-aware variants (`sum_axis`, `mean_axis`, `min_axis`, `max_axis`)
// are not yet exercised against fixtures — the existing fixtures
// include 2D `axis=0` cases that ferray-ma can compute, but they
// return arrays whose comparison is deferred to the umbrella issue
// (#754). Skipping `axis`-keyed cases matches the policy used by
// `tests/oracle.rs`.
// ---------------------------------------------------------------------------

/// Covers `ferray_ma::reductions::MaskedArray::sum` (canonical path)
/// and the user-facing `ferray_ma::MaskedArray::sum` method call.
/// Fixture: `fixtures/ma/masked_sum.json` (numpy.ma.sum).
#[test]
fn sum_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_sum.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let actual = ma.sum().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            actual,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("sum case '{}'", case.name),
        );
    }
}

/// Covers `ferray_ma::reductions::MaskedArray::mean` (canonical path)
/// and the user-facing `ferray_ma::MaskedArray::mean` method call.
/// Fixture: `fixtures/ma/masked_mean.json` (numpy.ma.mean).
#[test]
fn mean_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_mean.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let actual = ma.mean().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            actual,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("mean case '{}'", case.name),
        );
    }
}

/// Covers `ferray_ma::reductions::MaskedArray::min` (canonical path)
/// and the user-facing `ferray_ma::MaskedArray::min` method call.
/// Fixture: `fixtures/ma/masked_min.json` (numpy.ma.min).
#[test]
fn min_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_min.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let actual = ma.min().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            actual,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("min case '{}'", case.name),
        );
    }
}

/// Covers `ferray_ma::reductions::MaskedArray::max` (canonical path)
/// and the user-facing `ferray_ma::MaskedArray::max` method call.
/// Fixture: `fixtures/ma/masked_max.json` (numpy.ma.max).
#[test]
fn max_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_max.json"));
    for case in &suite.test_cases {
        if case.inputs.get("axis").is_some() {
            continue;
        }
        let ma = make_masked_array(&case.inputs);
        let actual = ma.max().unwrap();
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            actual,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("max case '{}'", case.name),
        );
    }
}

// ---------------------------------------------------------------------------
// Mask manipulation: getmask, getdata
// ---------------------------------------------------------------------------

/// Covers `ferray_ma::mask_ops::getmask` (canonical),
/// `ferray_ma::mask_ops::getdata` (canonical),
/// `ferray_ma::getmask` (re-export), and `ferray_ma::getdata` (re-export).
/// Fixture: `fixtures/ma/getmask_getdata.json` (numpy.ma.getmask/getdata).
#[test]
fn getmask_and_getdata_match_numpy() {
    let suite = load_fixture(&ma_fx("getmask_getdata.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);

        // getdata: returns the underlying data array unchanged.
        let data = ferray_ma::getdata(&ma).unwrap();
        let expected_data = parse_f64_data(&case.expected["data"]["data"]);
        assert_f64_slice_ulp(
            data.as_slice().unwrap(),
            &expected_data,
            0,
            &format!("getdata case '{}'", case.name),
        );

        // getmask: returns the mask array.
        let mask = ferray_ma::getmask(&ma).unwrap();
        let expected_mask = parse_bool_data(&case.expected["mask"]["data"]);
        for (i, (&actual, &expected)) in
            mask.as_slice().unwrap().iter().zip(expected_mask.iter()).enumerate()
        {
            assert_eq!(
                actual, expected,
                "getmask case '{}' element {i}",
                case.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors: masked_invalid
// ---------------------------------------------------------------------------

/// Covers `ferray_ma::constructors::masked_invalid` (canonical path)
/// and `ferray_ma::masked_invalid` (re-export). Fixture:
/// `fixtures/ma/masked_invalid.json` (numpy.ma.masked_invalid).
#[test]
fn masked_invalid_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_invalid.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["data"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_ma::masked_invalid(&arr).unwrap();
        let expected_mask = parse_bool_data(&case.expected["mask"]["data"]);
        let mask_slice = result.mask().as_slice().unwrap();
        for (i, (&actual, &expected)) in
            mask_slice.iter().zip(expected_mask.iter()).enumerate()
        {
            assert_eq!(
                actual, expected,
                "masked_invalid case '{}' element {i}",
                case.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// filled / compressed
// ---------------------------------------------------------------------------

/// Covers `ferray_ma::filled::MaskedArray::compressed` (canonical
/// method path) — the user-facing entry point is
/// `MaskedArray::compressed`. Fixture: `fixtures/ma/compressed.json`
/// (numpy.ma.compressed).
#[test]
fn compressed_matches_numpy() {
    let suite = load_fixture(&ma_fx("compressed.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        let result = ma.compressed().unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("compressed case '{}'", case.name),
        );
    }
}

/// Covers `ferray_ma::filled::MaskedArray::filled` (canonical method
/// path) — the user-facing entry point is `MaskedArray::filled`.
/// Fixture: `fixtures/ma/creation_filled.json` (numpy.ma.array + filled).
///
/// The fixture's two cases use different `expected` schemas:
///
/// - `basic_creation`: `expected.filled_0` is the array obtained by
///   `filled(0.0)`.
/// - `filled_custom`: `expected.filled` is the array obtained by
///   `filled(fill_value)` where `inputs.fill_value` is provided.
#[test]
fn filled_matches_numpy() {
    let suite = load_fixture(&ma_fx("creation_filled.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);

        // Determine the fill value and the corresponding expected key.
        let (fill_value, expected_key) =
            if let Some(fv) = case.inputs.get("fill_value") {
                (parse_f64_value(fv), "filled")
            } else {
                (0.0_f64, "filled_0")
            };

        let result = ma.filled(fill_value).unwrap();
        let expected = parse_f64_data(&case.expected[expected_key]["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("filled case '{}'", case.name),
        );
    }
}
