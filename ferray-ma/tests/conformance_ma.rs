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

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1, Ix2, IxDyn};
use ferray_ma::MaskedArray;
use ferray_test_oracle::{
    MIN_ULP_TOLERANCE, assert_f64_slice_ulp, assert_f64_ulp, fixtures_dir, load_fixture,
    make_bool_array, make_f64_array, parse_bool_data, parse_f64_data, parse_f64_value, parse_shape,
    parse_usize_data, should_skip_f64,
};

fn ma_fx(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("ma").join(name)
}

/// Build a `MaskedArray<f64, IxDyn>` from a fixture case's `data` +
/// `mask` inputs. Used by every reduction / mask-op test below.
fn make_masked_array(inputs: &ferray_test_oracle::serde_json::Value) -> MaskedArray<f64, IxDyn> {
    let data = make_f64_array(&inputs["data"]);
    let mask = make_bool_array(&inputs["mask"]);
    let mut ma = MaskedArray::new(data, mask).unwrap();
    if let Some(fill_value) = inputs.get("fill_value") {
        ma.set_fill_value(parse_f64_value(fill_value));
    }
    ma
}

fn make_masked_array_from_keys(
    inputs: &ferray_test_oracle::serde_json::Value,
    data_key: &str,
    mask_key: &str,
    fill_value_key: &str,
) -> MaskedArray<f64, IxDyn> {
    let data = make_f64_array(&inputs[data_key]);
    let mask = make_bool_array(&inputs[mask_key]);
    let mut ma = MaskedArray::new(data, mask).unwrap();
    if let Some(fill_value) = inputs.get(fill_value_key) {
        ma.set_fill_value(parse_f64_value(fill_value));
    }
    ma
}

fn make_masked_ix1(inputs: &ferray_test_oracle::serde_json::Value) -> MaskedArray<f64, Ix1> {
    let shape = parse_shape(&inputs["data"]["shape"]);
    assert_eq!(shape.len(), 1, "fixture input must be 1-D");
    let data = Array::from_vec(
        Ix1::new([shape[0]]),
        parse_f64_data(&inputs["data"]["data"]),
    )
    .unwrap();
    let mask = Array::from_vec(
        Ix1::new([shape[0]]),
        parse_bool_data(&inputs["mask"]["data"]),
    )
    .unwrap();
    let mut ma = MaskedArray::new(data, mask).unwrap();
    if let Some(fill_value) = inputs.get("fill_value") {
        ma.set_fill_value(parse_f64_value(fill_value));
    }
    ma
}

fn make_masked_ix2(inputs: &ferray_test_oracle::serde_json::Value) -> MaskedArray<f64, Ix2> {
    let shape = parse_shape(&inputs["data"]["shape"]);
    assert_eq!(shape.len(), 2, "fixture input must be 2-D");
    let data = Array::from_vec(
        Ix2::new([shape[0], shape[1]]),
        parse_f64_data(&inputs["data"]["data"]),
    )
    .unwrap();
    let mask = Array::from_vec(
        Ix2::new([shape[0], shape[1]]),
        parse_bool_data(&inputs["mask"]["data"]),
    )
    .unwrap();
    let mut ma = MaskedArray::new(data, mask).unwrap();
    if let Some(fill_value) = inputs.get("fill_value") {
        ma.set_fill_value(parse_f64_value(fill_value));
    }
    ma
}

fn make_bool_masked_array_from_keys(
    inputs: &ferray_test_oracle::serde_json::Value,
    data_key: &str,
    mask_key: &str,
    fill_value_key: &str,
) -> MaskedArray<bool, IxDyn> {
    let data = make_bool_array(&inputs[data_key]);
    let mask = make_bool_array(&inputs[mask_key]);
    let mut ma = MaskedArray::new(data, mask).unwrap();
    if let Some(fill_value) = inputs.get(fill_value_key) {
        ma.set_fill_value(fill_value.as_bool().unwrap());
    }
    ma
}

fn assert_masked_matches<D: Dimension>(
    actual: &MaskedArray<f64, D>,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
    case_name: &str,
) {
    let expected_shape = parse_shape(&expected["data"]["shape"]);
    assert_eq!(
        actual.shape(),
        expected_shape.as_slice(),
        "{case_name}: data shape"
    );
    assert_eq!(
        actual.mask().shape(),
        expected_shape.as_slice(),
        "{case_name}: mask shape"
    );

    let expected_data = parse_f64_data(&expected["data"]["data"]);
    let expected_mask = parse_bool_data(&expected["mask"]["data"]);
    let actual_data = actual.data().as_slice().unwrap();
    let actual_mask = actual.mask().as_slice().unwrap();

    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "{case_name}: data len"
    );
    assert_eq!(
        actual_mask.len(),
        expected_mask.len(),
        "{case_name}: mask len"
    );

    for (i, (&actual, (&expected, &masked))) in actual_data
        .iter()
        .zip(expected_data.iter().zip(expected_mask.iter()))
        .enumerate()
    {
        assert_eq!(actual_mask[i], masked, "{case_name}: mask element {i}");
        if !masked {
            assert_f64_ulp(
                actual,
                expected,
                tolerance_ulps.max(MIN_ULP_TOLERANCE),
                &format!("{case_name} element {i}"),
            );
        }
    }

    if let Some(expected_fill_value) = expected.get("fill_value") {
        assert_f64_ulp(
            actual.fill_value(),
            parse_f64_value(expected_fill_value),
            tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("{case_name} fill_value"),
        );
    }
}

fn assert_masked_all_data_matches<D: Dimension>(
    actual: &MaskedArray<f64, D>,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
    case_name: &str,
) {
    let expected_shape = parse_shape(&expected["data"]["shape"]);
    assert_eq!(
        actual.shape(),
        expected_shape.as_slice(),
        "{case_name}: data shape"
    );
    assert_eq!(
        actual.mask().shape(),
        expected_shape.as_slice(),
        "{case_name}: mask shape"
    );

    let expected_data = parse_f64_data(&expected["data"]["data"]);
    let expected_mask = parse_bool_data(&expected["mask"]["data"]);
    let actual_data = actual.data().as_slice().unwrap();
    let actual_mask = actual.mask().as_slice().unwrap();

    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "{case_name}: data len"
    );
    assert_eq!(
        actual_mask.len(),
        expected_mask.len(),
        "{case_name}: mask len"
    );

    for (i, (&actual, (&expected, &masked))) in actual_data
        .iter()
        .zip(expected_data.iter().zip(expected_mask.iter()))
        .enumerate()
    {
        assert_eq!(actual_mask[i], masked, "{case_name}: mask element {i}");
        assert_f64_ulp(
            actual,
            expected,
            tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("{case_name} data element {i}"),
        );
    }

    if let Some(expected_fill_value) = expected.get("fill_value") {
        assert_f64_ulp(
            actual.fill_value(),
            parse_f64_value(expected_fill_value),
            tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("{case_name} fill_value"),
        );
    }
}

fn assert_masked_metadata_matches<D: Dimension>(
    actual: &MaskedArray<f64, D>,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
    case_name: &str,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        actual.shape(),
        expected_shape.as_slice(),
        "{case_name}: data shape"
    );
    assert_eq!(
        actual.mask().shape(),
        expected_shape.as_slice(),
        "{case_name}: mask shape"
    );

    let expected_mask = parse_bool_data(&expected["mask"]["data"]);
    assert_eq!(
        actual.mask().as_slice().unwrap(),
        expected_mask.as_slice(),
        "{case_name}: mask"
    );
    assert_f64_ulp(
        actual.fill_value(),
        parse_f64_value(&expected["fill_value"]),
        tolerance_ulps.max(MIN_ULP_TOLERANCE),
        &format!("{case_name} fill_value"),
    );
}

fn assert_bool_array_matches<D: Dimension>(
    actual: &Array<bool, D>,
    expected: &ferray_test_oracle::serde_json::Value,
    case_name: &str,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        actual.shape(),
        expected_shape.as_slice(),
        "{case_name}: shape"
    );
    let expected_data = parse_bool_data(&expected["data"]);
    assert_eq!(
        actual.as_slice().unwrap(),
        expected_data.as_slice(),
        "{case_name}: data"
    );
}

fn assert_masked_bool_all_data_matches<D: Dimension>(
    actual: &MaskedArray<bool, D>,
    expected: &ferray_test_oracle::serde_json::Value,
    case_name: &str,
) {
    let expected_shape = parse_shape(&expected["data"]["shape"]);
    assert_eq!(
        actual.shape(),
        expected_shape.as_slice(),
        "{case_name}: data shape"
    );
    assert_eq!(
        actual.mask().shape(),
        expected_shape.as_slice(),
        "{case_name}: mask shape"
    );

    let expected_data = parse_bool_data(&expected["data"]["data"]);
    let expected_mask = parse_bool_data(&expected["mask"]["data"]);
    assert_eq!(
        actual.data().as_slice().unwrap(),
        expected_data.as_slice(),
        "{case_name}: data"
    );
    assert_eq!(
        actual.mask().as_slice().unwrap(),
        expected_mask.as_slice(),
        "{case_name}: mask"
    );
    if let Some(expected_fill_value) = expected.get("fill_value") {
        assert_eq!(
            actual.fill_value(),
            expected_fill_value.as_bool().unwrap(),
            "{case_name}: fill_value"
        );
    }
}

// ---------------------------------------------------------------------------
// Reductions: sum, mean, min, max
//
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

/// Covers `ferray_ma::reductions::MaskedArray::sum_axis`,
/// `ferray_ma::reductions::MaskedArray::mean_axis`,
/// `ferray_ma::reductions::MaskedArray::min_axis`, and
/// `ferray_ma::reductions::MaskedArray::max_axis`.
/// Fixture: `fixtures/ma/masked_reductions_axis.json`
/// (`numpy.ma` axis reductions).
#[test]
fn axis_reductions_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_reductions_axis.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        let axis = case.inputs["axis"].as_u64().unwrap() as usize;
        let result = match case.inputs["op"].as_str().unwrap() {
            "sum" => ma.sum_axis(axis).unwrap(),
            "mean" => ma.mean_axis(axis).unwrap(),
            "min" => ma.min_axis(axis).unwrap(),
            "max" => ma.max_axis(axis).unwrap(),
            op => panic!("unexpected axis reduction op {op}"),
        };
        assert_masked_matches(&result, &case.expected, case.tolerance_ulps, &case.name);
    }
}

/// Covers `ferray_ma::reductions::MaskedArray::count`,
/// `ferray_ma::reductions::MaskedArray::count_axis`,
/// `ferray_ma::mask_ops::count_masked`,
/// `ferray_ma::mask_ops::count_masked_axis`,
/// `ferray_ma::count_masked`, and `ferray_ma::count_masked_axis`.
/// Fixture: `fixtures/ma/masked_counts.json`
/// (`numpy.ma.count` / `numpy.ma.count_masked`).
#[test]
fn counts_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_counts.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        let expected_shape = parse_shape(&case.expected["shape"]);
        let expected = parse_usize_data(&case.expected["data"]);
        match case.inputs["op"].as_str().unwrap() {
            "count" => {
                assert!(
                    expected_shape.is_empty(),
                    "count case '{}' expected scalar",
                    case.name
                );
                assert_eq!(ma.count().unwrap(), expected[0], "{}", case.name);
            }
            "count_axis" => {
                let axis = case.inputs["axis"].as_u64().unwrap() as usize;
                let actual = ma.count_axis(axis).unwrap();
                assert_eq!(actual.shape(), expected_shape.as_slice(), "{}", case.name);
                assert_eq!(
                    actual.as_slice().unwrap(),
                    expected
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    "{}",
                    case.name
                );
            }
            "count_masked" => {
                assert!(
                    expected_shape.is_empty(),
                    "count_masked case '{}' expected scalar",
                    case.name
                );
                assert_eq!(
                    ferray_ma::count_masked(&ma).unwrap(),
                    expected[0],
                    "{}",
                    case.name
                );
            }
            "count_masked_axis" => {
                let axis = case.inputs["axis"].as_u64().unwrap() as usize;
                let actual = ferray_ma::count_masked_axis(&ma, axis).unwrap();
                assert_eq!(actual.shape(), expected_shape.as_slice(), "{}", case.name);
                assert_eq!(
                    actual.as_slice().unwrap(),
                    expected
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    "{}",
                    case.name
                );
            }
            op => panic!("unexpected count op {op}"),
        }
    }
}

/// Covers `ferray_ma::reductions::MaskedArray::var`,
/// `ferray_ma::reductions::MaskedArray::var_ddof`,
/// `ferray_ma::reductions::MaskedArray::var_axis`,
/// `ferray_ma::reductions::MaskedArray::var_axis_ddof`,
/// `ferray_ma::reductions::MaskedArray::std`,
/// `ferray_ma::reductions::MaskedArray::std_ddof`,
/// `ferray_ma::reductions::MaskedArray::std_axis`, and
/// `ferray_ma::reductions::MaskedArray::std_axis_ddof`.
/// Fixture: `fixtures/ma/masked_var_std.json`
/// (`numpy.ma.var` / `numpy.ma.std`, with `ddof` and axis cases).
#[test]
fn var_std_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_var_std.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        let op = case.inputs["op"].as_str().unwrap();
        let ddof = case.inputs["ddof"].as_u64().unwrap() as usize;

        if let Some(axis_value) = case.inputs.get("axis") {
            let axis = axis_value.as_u64().unwrap() as usize;
            let result = match (op, ddof) {
                ("var", 0) => ma.var_axis(axis).unwrap(),
                ("var", _) => ma.var_axis_ddof(axis, ddof).unwrap(),
                ("std", 0) => ma.std_axis(axis).unwrap(),
                ("std", _) => ma.std_axis_ddof(axis, ddof).unwrap(),
                _ => panic!("unexpected var/std op {op}"),
            };
            assert_masked_matches(&result, &case.expected, case.tolerance_ulps, &case.name);
        } else {
            let actual = match (op, ddof) {
                ("var", 0) => ma.var().unwrap(),
                ("var", _) => ma.var_ddof(ddof).unwrap(),
                ("std", 0) => ma.std().unwrap(),
                ("std", _) => ma.std_ddof(ddof).unwrap(),
                _ => panic!("unexpected var/std op {op}"),
            };
            let expected = parse_f64_value(&case.expected["data"]);
            assert_f64_ulp(
                actual,
                expected,
                case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                &case.name,
            );
        }
    }
}

/// Covers `ferray_ma::sorting::MaskedArray::sort`,
/// `ferray_ma::sorting::MaskedArray::argsort`, and
/// `ferray_ma::sorting::MaskedArray::sort_axis`.
/// Fixture: `fixtures/ma/masked_sorting.json`
/// (`numpy.ma.sort` / `numpy.ma.argsort`).
#[test]
fn sorting_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_sorting.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        match case.inputs["op"].as_str().unwrap() {
            "sort" => {
                let actual = ma.sort().unwrap();
                assert_masked_matches(&actual, &case.expected, case.tolerance_ulps, &case.name);
            }
            "sort_axis" => {
                let axis = case.inputs["axis"].as_u64().unwrap() as usize;
                let actual = ma.sort_axis(axis).unwrap();
                assert_masked_matches(&actual, &case.expected, case.tolerance_ulps, &case.name);
            }
            "argsort" => {
                let actual = ma.argsort().unwrap();
                let expected_shape = parse_shape(&case.expected["shape"]);
                let expected = parse_usize_data(&case.expected["data"]);
                assert_eq!(actual.shape(), expected_shape.as_slice(), "{}", case.name);
                assert_eq!(
                    actual.as_slice().unwrap(),
                    expected
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    "{}",
                    case.name
                );
            }
            op => panic!("unexpected sorting op {op}"),
        }
    }
}

/// Covers canonical constructor paths
/// `ferray_ma::constructors::fix_invalid`,
/// `ferray_ma::constructors::masked_where`,
/// `ferray_ma::constructors::masked_equal`,
/// `ferray_ma::constructors::masked_not_equal`,
/// `ferray_ma::constructors::masked_greater`,
/// `ferray_ma::constructors::masked_greater_equal`,
/// `ferray_ma::constructors::masked_less`,
/// `ferray_ma::constructors::masked_less_equal`,
/// `ferray_ma::constructors::masked_inside`, and
/// `ferray_ma::constructors::masked_outside`, plus the root re-exports
/// `ferray_ma::fix_invalid`, `ferray_ma::masked_where`,
/// `ferray_ma::masked_equal`, `ferray_ma::masked_not_equal`,
/// `ferray_ma::masked_greater`, `ferray_ma::masked_greater_equal`,
/// `ferray_ma::masked_less`, `ferray_ma::masked_less_equal`,
/// `ferray_ma::masked_inside`, and `ferray_ma::masked_outside`.
/// Fixture: `fixtures/ma/masked_constructors.json`
/// (`numpy.ma` masking constructors).
#[test]
fn masking_constructors_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_constructors.json"));
    for case in &suite.test_cases {
        let data = make_f64_array(&case.inputs["data"]);
        let result = match case.inputs["op"].as_str().unwrap() {
            "fix_invalid" => {
                let fill_value = parse_f64_value(&case.inputs["fill_value"]);
                ferray_ma::fix_invalid(&data, fill_value).unwrap()
            }
            "masked_where" => {
                let condition = make_bool_array(&case.inputs["condition"]);
                ferray_ma::masked_where(&condition, &data).unwrap()
            }
            "masked_equal" => {
                let value = parse_f64_value(&case.inputs["value"]);
                ferray_ma::masked_equal(&data, value).unwrap()
            }
            "masked_not_equal" => {
                let value = parse_f64_value(&case.inputs["value"]);
                ferray_ma::masked_not_equal(&data, value).unwrap()
            }
            "masked_greater" => {
                let value = parse_f64_value(&case.inputs["value"]);
                ferray_ma::masked_greater(&data, value).unwrap()
            }
            "masked_greater_equal" => {
                let value = parse_f64_value(&case.inputs["value"]);
                ferray_ma::masked_greater_equal(&data, value).unwrap()
            }
            "masked_less" => {
                let value = parse_f64_value(&case.inputs["value"]);
                ferray_ma::masked_less(&data, value).unwrap()
            }
            "masked_less_equal" => {
                let value = parse_f64_value(&case.inputs["value"]);
                ferray_ma::masked_less_equal(&data, value).unwrap()
            }
            "masked_inside" => {
                let v1 = parse_f64_value(&case.inputs["v1"]);
                let v2 = parse_f64_value(&case.inputs["v2"]);
                ferray_ma::masked_inside(&data, v1, v2).unwrap()
            }
            "masked_outside" => {
                let v1 = parse_f64_value(&case.inputs["v1"]);
                let v2 = parse_f64_value(&case.inputs["v2"]);
                ferray_ma::masked_outside(&data, v1, v2).unwrap()
            }
            op => panic!("unexpected constructor op {op}"),
        };
        assert_masked_matches(&result, &case.expected, case.tolerance_ulps, &case.name);
    }
}

/// Covers canonical arithmetic paths
/// `ferray_ma::arithmetic::masked_add`,
/// `ferray_ma::arithmetic::masked_sub`,
/// `ferray_ma::arithmetic::masked_mul`,
/// `ferray_ma::arithmetic::masked_div`,
/// `ferray_ma::arithmetic::masked_add_array`,
/// `ferray_ma::arithmetic::masked_sub_array`,
/// `ferray_ma::arithmetic::masked_mul_array`, and
/// `ferray_ma::arithmetic::masked_div_array`, plus root re-exports
/// `ferray_ma::masked_add`, `ferray_ma::masked_sub`,
/// `ferray_ma::masked_mul`, `ferray_ma::masked_div`,
/// `ferray_ma::masked_add_array`, `ferray_ma::masked_sub_array`,
/// `ferray_ma::masked_mul_array`, and `ferray_ma::masked_div_array`.
/// Fixture: `fixtures/ma/masked_arithmetic.json`
/// (`numpy.ma` masked binary arithmetic).
#[test]
fn masked_arithmetic_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_arithmetic.json"));
    for case in &suite.test_cases {
        let lhs_data = make_f64_array(&case.inputs["lhs_data"]);
        let lhs_mask = make_bool_array(&case.inputs["lhs_mask"]);
        let lhs = MaskedArray::new(lhs_data, lhs_mask).unwrap();
        let result = match case.inputs["op"].as_str().unwrap() {
            "masked_add" => {
                let rhs_data = make_f64_array(&case.inputs["rhs_data"]);
                let rhs_mask = make_bool_array(&case.inputs["rhs_mask"]);
                let rhs = MaskedArray::new(rhs_data, rhs_mask).unwrap();
                ferray_ma::masked_add(&lhs, &rhs).unwrap()
            }
            "masked_sub" => {
                let rhs_data = make_f64_array(&case.inputs["rhs_data"]);
                let rhs_mask = make_bool_array(&case.inputs["rhs_mask"]);
                let rhs = MaskedArray::new(rhs_data, rhs_mask).unwrap();
                ferray_ma::masked_sub(&lhs, &rhs).unwrap()
            }
            "masked_mul" => {
                let rhs_data = make_f64_array(&case.inputs["rhs_data"]);
                let rhs_mask = make_bool_array(&case.inputs["rhs_mask"]);
                let rhs = MaskedArray::new(rhs_data, rhs_mask).unwrap();
                ferray_ma::masked_mul(&lhs, &rhs).unwrap()
            }
            "masked_div" => {
                let rhs_data = make_f64_array(&case.inputs["rhs_data"]);
                let rhs_mask = make_bool_array(&case.inputs["rhs_mask"]);
                let rhs = MaskedArray::new(rhs_data, rhs_mask).unwrap();
                ferray_ma::masked_div(&lhs, &rhs).unwrap()
            }
            "masked_add_array" => {
                let rhs = make_f64_array(&case.inputs["rhs_data"]);
                ferray_ma::masked_add_array(&lhs, &rhs).unwrap()
            }
            "masked_sub_array" => {
                let rhs = make_f64_array(&case.inputs["rhs_data"]);
                ferray_ma::masked_sub_array(&lhs, &rhs).unwrap()
            }
            "masked_mul_array" => {
                let rhs = make_f64_array(&case.inputs["rhs_data"]);
                ferray_ma::masked_mul_array(&lhs, &rhs).unwrap()
            }
            "masked_div_array" => {
                let rhs = make_f64_array(&case.inputs["rhs_data"]);
                ferray_ma::masked_div_array(&lhs, &rhs).unwrap()
            }
            op => panic!("unexpected arithmetic op {op}"),
        };
        assert_masked_all_data_matches(&result, &case.expected, case.tolerance_ulps, &case.name);
    }
}

/// Covers `ferray_ma::manipulation::MaskedArray::reshape`,
/// `ferray_ma::manipulation::MaskedArray::ravel`,
/// `ferray_ma::manipulation::MaskedArray::flatten`,
/// `ferray_ma::manipulation::MaskedArray::transpose`,
/// `ferray_ma::manipulation::MaskedArray::t`,
/// `ferray_ma::manipulation::MaskedArray::squeeze`,
/// `ferray_ma::manipulation::MaskedArray::boolean_index`,
/// `ferray_ma::manipulation::MaskedArray::take`, and
/// `ferray_ma::manipulation::MaskedArray::get_flat`.
/// Fixture: `fixtures/ma/masked_manipulation.json`
/// (`numpy.ma` shape manipulation, flat indexing, and take).
#[test]
fn masked_manipulation_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_manipulation.json"));
    for case in &suite.test_cases {
        let ma = make_masked_array(&case.inputs);
        match case.inputs["op"].as_str().unwrap() {
            "reshape" => {
                let new_shape = parse_usize_data(&case.inputs["new_shape"]);
                let actual = ma.reshape(&new_shape).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ravel" => {
                let actual = ma.ravel().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "flatten" => {
                let actual = ma.flatten().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "transpose" => {
                let axes = case.inputs.get("axes").map(parse_usize_data);
                let actual = ma.transpose(axes.as_deref()).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "t" => {
                let actual = ma.t().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "squeeze" => {
                let axis = case
                    .inputs
                    .get("axis")
                    .map(|axis| axis.as_u64().unwrap() as usize);
                let actual = ma.squeeze(axis).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "boolean_index" => {
                let selector = make_bool_array(&case.inputs["selector"]);
                let actual = ma.boolean_index(&selector).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "take" => {
                let indices = parse_usize_data(&case.inputs["indices"]["data"]);
                let actual = ma.take(&indices).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "get_flat" => {
                let flat_index = case.inputs["flat_index"].as_u64().unwrap() as usize;
                let (actual_value, actual_masked) = ma.get_flat(flat_index).unwrap();
                let expected_value = parse_f64_value(&case.expected["value"]);
                let expected_masked = case.expected["masked"].as_bool().unwrap();
                assert_f64_ulp(
                    actual_value,
                    expected_value,
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} get_flat value", case.name),
                );
                assert_eq!(
                    actual_masked, expected_masked,
                    "{} get_flat mask",
                    case.name
                );
            }
            op => panic!("unexpected masked manipulation op {op}"),
        }
    }
}

/// Covers canonical masked ufunc-support paths
/// `ferray_ma::ufunc_support::masked_unary`,
/// `ferray_ma::ufunc_support::masked_binary`,
/// `ferray_ma::ufunc_support::masked_unary_domain`,
/// `ferray_ma::ufunc_support::masked_binary_domain`,
/// `ferray_ma::ufunc_support::absolute`,
/// `ferray_ma::ufunc_support::add`,
/// `ferray_ma::ufunc_support::arccos`,
/// `ferray_ma::ufunc_support::arccos_domain`,
/// `ferray_ma::ufunc_support::arccosh`,
/// `ferray_ma::ufunc_support::arccosh_domain`,
/// `ferray_ma::ufunc_support::arcsin`,
/// `ferray_ma::ufunc_support::arcsin_domain`,
/// `ferray_ma::ufunc_support::arcsinh`,
/// `ferray_ma::ufunc_support::arctan`,
/// `ferray_ma::ufunc_support::arctanh`,
/// `ferray_ma::ufunc_support::arctanh_domain`,
/// `ferray_ma::ufunc_support::ceil`,
/// `ferray_ma::ufunc_support::cos`,
/// `ferray_ma::ufunc_support::cosh`,
/// `ferray_ma::ufunc_support::divide`,
/// `ferray_ma::ufunc_support::divide_domain`,
/// `ferray_ma::ufunc_support::exp`,
/// `ferray_ma::ufunc_support::exp2`,
/// `ferray_ma::ufunc_support::expm1`,
/// `ferray_ma::ufunc_support::floor`,
/// `ferray_ma::ufunc_support::log`,
/// `ferray_ma::ufunc_support::log10`,
/// `ferray_ma::ufunc_support::log10_domain`,
/// `ferray_ma::ufunc_support::log1p`,
/// `ferray_ma::ufunc_support::log2`,
/// `ferray_ma::ufunc_support::log2_domain`,
/// `ferray_ma::ufunc_support::log_domain`,
/// `ferray_ma::ufunc_support::multiply`,
/// `ferray_ma::ufunc_support::negative`,
/// `ferray_ma::ufunc_support::power`,
/// `ferray_ma::ufunc_support::reciprocal`,
/// `ferray_ma::ufunc_support::round`,
/// `ferray_ma::ufunc_support::sign`,
/// `ferray_ma::ufunc_support::sin`,
/// `ferray_ma::ufunc_support::sinh`,
/// `ferray_ma::ufunc_support::sqrt`,
/// `ferray_ma::ufunc_support::sqrt_domain`,
/// `ferray_ma::ufunc_support::square`,
/// `ferray_ma::ufunc_support::subtract`,
/// `ferray_ma::ufunc_support::tan`,
/// `ferray_ma::ufunc_support::tanh`, and
/// `ferray_ma::ufunc_support::trunc`, plus root re-exports
/// `ferray_ma::masked_unary`, `ferray_ma::masked_binary`,
/// `ferray_ma::masked_unary_domain`, `ferray_ma::masked_binary_domain`,
/// `ferray_ma::arccos_domain`, `ferray_ma::arccosh_domain`,
/// `ferray_ma::arcsin_domain`, `ferray_ma::arctanh_domain`,
/// `ferray_ma::divide_domain`, `ferray_ma::log_domain`,
/// `ferray_ma::log2_domain`, `ferray_ma::log10_domain`, and
/// `ferray_ma::sqrt_domain`.
/// Fixture: `fixtures/ma/masked_ufunc_support.json`
/// (`numpy` masked ufunc protocol).
#[test]
fn masked_ufunc_support_matches_numpy() {
    let suite = load_fixture(&ma_fx("masked_ufunc_support.json"));
    for case in &suite.test_cases {
        let result = match case.inputs["op"].as_str().unwrap() {
            "absolute" => ferray_ma::ufunc_support::absolute(&make_masked_array(&case.inputs)),
            "arccos" => ferray_ma::ufunc_support::arccos(&make_masked_array(&case.inputs)),
            "arccos_domain" => ferray_ma::arccos_domain(&make_masked_array(&case.inputs)),
            "arccosh" => ferray_ma::ufunc_support::arccosh(&make_masked_array(&case.inputs)),
            "arccosh_domain" => ferray_ma::arccosh_domain(&make_masked_array(&case.inputs)),
            "arcsin" => ferray_ma::ufunc_support::arcsin(&make_masked_array(&case.inputs)),
            "arcsin_domain" => ferray_ma::arcsin_domain(&make_masked_array(&case.inputs)),
            "arcsinh" => ferray_ma::ufunc_support::arcsinh(&make_masked_array(&case.inputs)),
            "arctan" => ferray_ma::ufunc_support::arctan(&make_masked_array(&case.inputs)),
            "arctanh" => ferray_ma::ufunc_support::arctanh(&make_masked_array(&case.inputs)),
            "arctanh_domain" => ferray_ma::arctanh_domain(&make_masked_array(&case.inputs)),
            "ceil" => ferray_ma::ufunc_support::ceil(&make_masked_array(&case.inputs)),
            "cos" => ferray_ma::ufunc_support::cos(&make_masked_array(&case.inputs)),
            "cosh" => ferray_ma::ufunc_support::cosh(&make_masked_array(&case.inputs)),
            "exp" => ferray_ma::ufunc_support::exp(&make_masked_array(&case.inputs)),
            "exp2" => ferray_ma::ufunc_support::exp2(&make_masked_array(&case.inputs)),
            "expm1" => ferray_ma::ufunc_support::expm1(&make_masked_array(&case.inputs)),
            "floor" => ferray_ma::ufunc_support::floor(&make_masked_array(&case.inputs)),
            "log" => ferray_ma::ufunc_support::log(&make_masked_array(&case.inputs)),
            "log10" => ferray_ma::ufunc_support::log10(&make_masked_array(&case.inputs)),
            "log10_domain" => ferray_ma::log10_domain(&make_masked_array(&case.inputs)),
            "log1p" => ferray_ma::ufunc_support::log1p(&make_masked_array(&case.inputs)),
            "log2" => ferray_ma::ufunc_support::log2(&make_masked_array(&case.inputs)),
            "log2_domain" => ferray_ma::log2_domain(&make_masked_array(&case.inputs)),
            "log_domain" => ferray_ma::log_domain(&make_masked_array(&case.inputs)),
            "masked_unary" => {
                ferray_ma::masked_unary(&make_masked_array(&case.inputs), |x| x * 10.0 + 1.0)
            }
            "masked_unary_domain" => ferray_ma::masked_unary_domain(
                &make_masked_array(&case.inputs),
                |x| x + 100.0,
                |x| x >= 0.0,
            ),
            "negative" => ferray_ma::ufunc_support::negative(&make_masked_array(&case.inputs)),
            "reciprocal" => ferray_ma::ufunc_support::reciprocal(&make_masked_array(&case.inputs)),
            "round" => ferray_ma::ufunc_support::round(&make_masked_array(&case.inputs)),
            "sign" => ferray_ma::ufunc_support::sign(&make_masked_array(&case.inputs)),
            "sin" => ferray_ma::ufunc_support::sin(&make_masked_array(&case.inputs)),
            "sinh" => ferray_ma::ufunc_support::sinh(&make_masked_array(&case.inputs)),
            "sqrt" => ferray_ma::ufunc_support::sqrt(&make_masked_array(&case.inputs)),
            "sqrt_domain" => ferray_ma::sqrt_domain(&make_masked_array(&case.inputs)),
            "square" => ferray_ma::ufunc_support::square(&make_masked_array(&case.inputs)),
            "tan" => ferray_ma::ufunc_support::tan(&make_masked_array(&case.inputs)),
            "tanh" => ferray_ma::ufunc_support::tanh(&make_masked_array(&case.inputs)),
            "trunc" => ferray_ma::ufunc_support::trunc(&make_masked_array(&case.inputs)),
            "add" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::ufunc_support::add(&lhs, &rhs)
            }
            "subtract" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::ufunc_support::subtract(&lhs, &rhs)
            }
            "multiply" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::ufunc_support::multiply(&lhs, &rhs)
            }
            "divide" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::ufunc_support::divide(&lhs, &rhs)
            }
            "divide_domain" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::divide_domain(&lhs, &rhs)
            }
            "power" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::ufunc_support::power(&lhs, &rhs)
            }
            "masked_binary" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::masked_binary(&lhs, &rhs, |x, y| x * 2.0 + y, "masked_binary")
            }
            "masked_binary_domain" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                ferray_ma::masked_binary_domain(
                    &lhs,
                    &rhs,
                    |x, y| x - y,
                    |_x, y| y > 0.0,
                    "masked_binary_domain",
                )
            }
            op => panic!("unexpected masked ufunc-support op {op}"),
        }
        .unwrap();

        assert_masked_all_data_matches(&result, &case.expected, case.tolerance_ulps, &case.name);
    }
}

/// Covers helper constants and functions:
/// `ferray_ma::NOMASK`, `ferray_ma::extras::default_fill_value_bool`,
/// `ferray_ma::extras::default_fill_value_f32`,
/// `ferray_ma::extras::default_fill_value_f64`,
/// `ferray_ma::extras::default_fill_value_i64`,
/// `ferray_ma::extras::maximum_fill_value`,
/// `ferray_ma::extras::minimum_fill_value`, `ferray_ma::extras::mask_or`,
/// `ferray_ma::extras::make_mask`, `ferray_ma::extras::make_mask_none`,
/// `ferray_ma::extras::masked_all`, `ferray_ma::extras::masked_all_like`,
/// `ferray_ma::extras::masked_values`, `ferray_ma::extras::getmaskarray`,
/// `ferray_ma::extras::is_ma`, `ferray_ma::extras::is_masked_array`,
/// `ferray_ma::mask_ops::is_masked`,
/// `ferray_ma::filled::MaskedArray::filled_default`, plus root re-exports
/// `ferray_ma::default_fill_value_bool`,
/// `ferray_ma::default_fill_value_f32`,
/// `ferray_ma::default_fill_value_f64`,
/// `ferray_ma::default_fill_value_i64`, `ferray_ma::maximum_fill_value`,
/// `ferray_ma::minimum_fill_value`, `ferray_ma::mask_or`,
/// `ferray_ma::make_mask`, `ferray_ma::make_mask_none`,
/// `ferray_ma::masked_all`, `ferray_ma::masked_all_like`,
/// `ferray_ma::masked_values`, `ferray_ma::getmaskarray`,
/// `ferray_ma::is_masked`, `ferray_ma::is_ma`, and
/// `ferray_ma::is_masked_array`.
/// Fixture: `fixtures/ma/masked_extras_helpers.json`
/// (`numpy.ma` extras helper functions).
#[test]
fn masked_extras_helpers_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_extras_helpers.json"));
    for case in &suite.test_cases {
        match case.inputs["op"].as_str().unwrap() {
            "default_fill_values" => {
                assert_f64_ulp(
                    ferray_ma::default_fill_value_f64(),
                    parse_f64_value(&case.expected["f64"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} f64", case.name),
                );
                let expected_f32 = parse_f64_value(&case.expected["f32"]) as f32;
                assert_eq!(
                    ferray_ma::default_fill_value_f32().to_bits(),
                    expected_f32.to_bits(),
                    "{} f32",
                    case.name
                );
                assert_eq!(
                    ferray_ma::default_fill_value_bool(),
                    case.expected["bool"].as_bool().unwrap(),
                    "{} bool",
                    case.name
                );
                assert_eq!(
                    ferray_ma::default_fill_value_i64(),
                    case.expected["i64"].as_i64().unwrap(),
                    "{} i64",
                    case.name
                );
            }
            "extremum_fill_values" => {
                assert_eq!(
                    ferray_ma::maximum_fill_value::<f64>(),
                    parse_f64_value(&case.expected["maximum_f64"]),
                    "{} maximum_f64",
                    case.name
                );
                assert_eq!(
                    ferray_ma::minimum_fill_value::<f64>(),
                    parse_f64_value(&case.expected["minimum_f64"]),
                    "{} minimum_f64",
                    case.name
                );
            }
            "nomask" => {
                assert_eq!(
                    ferray_ma::NOMASK,
                    case.expected["value"].as_bool().unwrap(),
                    "{} NOMASK",
                    case.name
                );
            }
            "mask_or" => {
                let lhs = make_bool_array(&case.inputs["lhs_mask"]);
                let rhs = make_bool_array(&case.inputs["rhs_mask"]);
                let actual = ferray_ma::mask_or(&lhs, &rhs).unwrap();
                assert_bool_array_matches(&actual, &case.expected, &case.name);
            }
            "make_mask" => {
                let values = parse_bool_data(&case.inputs["values"]["data"]);
                let shape = parse_shape(&case.inputs["values"]["shape"]);
                let actual = ferray_ma::make_mask(&values, IxDyn::new(&shape)).unwrap();
                assert_bool_array_matches(&actual, &case.expected, &case.name);
            }
            "make_mask_none" => {
                let shape = parse_shape(&case.inputs["shape"]);
                let actual = ferray_ma::make_mask_none(IxDyn::new(&shape)).unwrap();
                assert_bool_array_matches(&actual, &case.expected, &case.name);
            }
            "masked_all" => {
                let shape = parse_shape(&case.inputs["shape"]);
                let actual = ferray_ma::masked_all::<f64, IxDyn>(IxDyn::new(&shape)).unwrap();
                assert_masked_metadata_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "masked_all_like" => {
                let reference = make_f64_array(&case.inputs["reference"]);
                let actual: MaskedArray<f64, IxDyn> =
                    ferray_ma::masked_all_like(&reference).unwrap();
                assert_masked_metadata_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "masked_values" => {
                let data = make_f64_array(&case.inputs["data"]);
                let actual = ferray_ma::masked_values(
                    &data,
                    parse_f64_value(&case.inputs["value"]),
                    parse_f64_value(&case.inputs["rtol"]),
                    parse_f64_value(&case.inputs["atol"]),
                )
                .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "getmaskarray" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ferray_ma::getmaskarray(&ma).unwrap();
                assert_bool_array_matches(&actual, &case.expected, &case.name);
            }
            "is_masked" => {
                let ma = make_masked_array(&case.inputs);
                let expected = case.expected["value"].as_bool().unwrap();
                assert_eq!(ferray_ma::mask_ops::is_masked(&ma).unwrap(), expected);
                assert_eq!(ferray_ma::is_masked(&ma).unwrap(), expected);
            }
            "is_ma" => {
                let ma = make_masked_array(&case.inputs);
                let expected = case.expected["value"].as_bool().unwrap();
                assert_eq!(ferray_ma::extras::is_ma(&ma), expected);
                assert_eq!(ferray_ma::is_ma(&ma), expected);
            }
            "is_masked_array" => {
                let ma = make_masked_array(&case.inputs);
                let expected = case.expected["value"].as_bool().unwrap();
                assert_eq!(ferray_ma::extras::is_masked_array(&ma), expected);
                assert_eq!(ferray_ma::is_masked_array(&ma), expected);
            }
            "filled_default" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.filled_default().unwrap();
                let expected_data = parse_f64_data(&case.expected["data"]);
                assert_f64_slice_ulp(
                    actual.as_slice().unwrap(),
                    &expected_data,
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} filled_default", case.name),
                );
            }
            op => panic!("unexpected masked extras helper op {op}"),
        }
    }
}

/// Covers `ferray_ma::extras::MaskedArray::anom`,
/// `ferray_ma::extras::MaskedArray::argmax`,
/// `ferray_ma::extras::MaskedArray::argmin`,
/// `ferray_ma::extras::MaskedArray::atleast_1d`,
/// `ferray_ma::extras::MaskedArray::atleast_2d`,
/// `ferray_ma::extras::MaskedArray::atleast_3d`,
/// `ferray_ma::extras::MaskedArray::average`,
/// `ferray_ma::extras::MaskedArray::clip`,
/// `ferray_ma::extras::MaskedArray::cumprod_flat`,
/// `ferray_ma::extras::MaskedArray::cumsum_flat`,
/// `ferray_ma::extras::MaskedArray::diagonal`,
/// `ferray_ma::extras::MaskedArray::expand_dims`,
/// `ferray_ma::extras::MaskedArray::ma_dot_flat`,
/// `ferray_ma::extras::MaskedArray::median`,
/// `ferray_ma::extras::MaskedArray::prod`,
/// `ferray_ma::extras::MaskedArray::ptp`,
/// `ferray_ma::extras::MaskedArray::repeat`, and
/// `ferray_ma::extras::MaskedArray::trace`.
/// Fixture: `fixtures/ma/masked_extras_methods.json`
/// (`numpy.ma` extras numeric and shape methods).
#[test]
fn masked_extras_methods_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_extras_methods.json"));
    for case in &suite.test_cases {
        match case.inputs["op"].as_str().unwrap() {
            "prod" => {
                let ma = make_masked_array(&case.inputs);
                assert_f64_ulp(
                    ma.prod().unwrap(),
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "ptp" => {
                let ma = make_masked_array(&case.inputs);
                assert_f64_ulp(
                    ma.ptp().unwrap(),
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "median" => {
                let ma = make_masked_array(&case.inputs);
                assert_f64_ulp(
                    ma.median().unwrap(),
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "average" => {
                let ma = make_masked_array(&case.inputs);
                assert_f64_ulp(
                    ma.average(None).unwrap(),
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "trace" => {
                let ma = make_masked_ix2(&case.inputs);
                assert_f64_ulp(
                    ma.trace(0).unwrap(),
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "argmin" => {
                let ma = make_masked_array(&case.inputs);
                assert_eq!(
                    ma.argmin().unwrap(),
                    case.expected["value"].as_u64().unwrap() as usize,
                    "{}",
                    case.name
                );
            }
            "argmax" => {
                let ma = make_masked_array(&case.inputs);
                assert_eq!(
                    ma.argmax().unwrap(),
                    case.expected["value"].as_u64().unwrap() as usize,
                    "{}",
                    case.name
                );
            }
            "average_returned" => {
                let ma = make_masked_array(&case.inputs);
                let (avg, sum_weights) = ma.average_returned(None).unwrap();
                assert_f64_ulp(
                    avg,
                    parse_f64_value(&case.expected["average"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} average", case.name),
                );
                assert_f64_ulp(
                    sum_weights,
                    parse_f64_value(&case.expected["sum_weights"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} sum_weights", case.name),
                );
            }
            "average_weighted_returned" => {
                let ma = make_masked_array(&case.inputs);
                let weights = make_f64_array(&case.inputs["weights"]);
                let (avg, sum_weights) = ma.average_returned(Some(&weights)).unwrap();
                assert_f64_ulp(
                    avg,
                    parse_f64_value(&case.expected["average"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} average", case.name),
                );
                assert_f64_ulp(
                    sum_weights,
                    parse_f64_value(&case.expected["sum_weights"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} sum_weights", case.name),
                );
            }
            "cumsum_flat" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.cumsum_flat().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "cumprod_flat" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.cumprod_flat().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "anom" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.anom().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "clip" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma
                    .clip(
                        parse_f64_value(&case.inputs["min"]),
                        parse_f64_value(&case.inputs["max"]),
                    )
                    .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "repeat" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma
                    .repeat(case.inputs["repeats"].as_u64().unwrap() as usize)
                    .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "atleast_1d" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.atleast_1d().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "atleast_2d" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.atleast_2d().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "atleast_3d" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.atleast_3d().unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "expand_dims" => {
                let ma = make_masked_array(&case.inputs);
                let axis = case.inputs["axis"].as_u64().unwrap() as usize;
                let actual = ma.expand_dims(axis).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "diagonal" | "diagonal_offset1" => {
                let ma = make_masked_ix2(&case.inputs);
                let offset = case
                    .inputs
                    .get("offset")
                    .and_then(ferray_test_oracle::serde_json::Value::as_i64)
                    .unwrap_or(0) as isize;
                let actual = ma.diagonal(offset).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_dot_flat" => {
                let lhs = make_masked_array(&case.inputs);
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                assert_f64_ulp(
                    lhs.ma_dot_flat(&rhs).unwrap(),
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            op => panic!("unexpected masked extras method op {op}"),
        }
    }
}

/// Covers `ferray_ma::MaskedArray`,
/// `ferray_ma::MaskAware`,
/// `ferray_ma::masked_array::MaskedArray`,
/// `ferray_ma::masked_array::MaskedArray::new`,
/// `ferray_ma::masked_array::MaskedArray::from_data`,
/// `ferray_ma::masked_array::MaskedArray::data`,
/// `ferray_ma::masked_array::MaskedArray::data_mut`,
/// `ferray_ma::masked_array::MaskedArray::mask`,
/// `ferray_ma::masked_array::MaskedArray::mask_opt`,
/// `ferray_ma::masked_array::MaskedArray::set_mask`,
/// `ferray_ma::masked_array::MaskedArray::set_mask_flat`,
/// `ferray_ma::masked_array::MaskedArray::shape`,
/// `ferray_ma::masked_array::MaskedArray::ndim`,
/// `ferray_ma::masked_array::MaskedArray::size`,
/// `ferray_ma::masked_array::MaskedArray::dim`,
/// `ferray_ma::masked_array::MaskedArray::fill_value`,
/// `ferray_ma::masked_array::MaskedArray::set_fill_value`,
/// `ferray_ma::masked_array::MaskedArray::with_fill_value`,
/// `ferray_ma::masked_array::MaskedArray::has_real_mask`,
/// `ferray_ma::masked_array::MaskedArray::is_hard_mask`,
/// `ferray_ma::masked_array::MaskedArray::shares_mask`,
/// `ferray_ma::mask_ops::MaskedArray::harden_mask`,
/// `ferray_ma::mask_ops::MaskedArray::soften_mask`,
/// `ferray_ma::extras::common_fill_value`, `ferray_ma::common_fill_value`,
/// `ferray_ma::extras::ids`, `ferray_ma::ids`,
/// `ferray_ma::interop::MaskAware`,
/// `ferray_ma::interop::MaskedArray::into_data`,
/// `ferray_ma::interop::MaskedArray::apply_unary`,
/// `ferray_ma::interop::MaskedArray::apply_unary_to`,
/// `ferray_ma::interop::MaskedArray::apply_binary`,
/// `ferray_ma::interop::ma_apply_unary`, and
/// `ferray_ma::ma_apply_unary`, plus optional I/O helpers
/// `ferray_ma::io::save_masked` and `ferray_ma::io::load_masked`.
/// Fixture: `fixtures/ma/masked_core_surface.json`
/// (`numpy.ma` MaskedArray core surface and mask state).
#[test]
fn masked_core_surface_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_core_surface.json"));
    for case in &suite.test_cases {
        match case.inputs["op"].as_str().unwrap() {
            "masked_array_accessors" => {
                let ma = make_masked_array(&case.inputs);
                let expected_data = parse_f64_data(&case.expected["data"]["data"]);
                let expected_mask = parse_bool_data(&case.expected["mask"]["data"]);
                let expected_shape = parse_shape(&case.expected["shape"]);

                assert_f64_slice_ulp(
                    ma.data().as_slice().unwrap(),
                    &expected_data,
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} data", case.name),
                );
                assert_eq!(ma.mask().as_slice().unwrap(), expected_mask.as_slice());
                assert_eq!(ma.shape(), expected_shape.as_slice());
                assert_eq!(ma.dim().as_slice(), expected_shape.as_slice());
                assert_eq!(ma.ndim(), case.expected["ndim"].as_u64().unwrap() as usize);
                assert_eq!(ma.size(), case.expected["size"].as_u64().unwrap() as usize);
                assert_f64_ulp(
                    ma.fill_value(),
                    parse_f64_value(&case.expected["fill_value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} fill_value", case.name),
                );
                assert_eq!(
                    ma.has_real_mask(),
                    case.expected["has_real_mask"].as_bool().unwrap()
                );
                assert_eq!(
                    ma.mask_opt().is_some(),
                    case.expected["mask_opt"].as_bool().unwrap()
                );
                assert_eq!(
                    ma.is_hard_mask(),
                    case.expected["is_hard_mask"].as_bool().unwrap()
                );
            }
            "masked_array_from_data" => {
                let data = make_f64_array(&case.inputs["data"]);
                let ma = MaskedArray::from_data(data).unwrap();
                let expected_data = parse_f64_data(&case.expected["data"]["data"]);
                let expected_mask = parse_bool_data(&case.expected["mask"]["data"]);
                let expected_shape = parse_shape(&case.expected["shape"]);

                assert_f64_slice_ulp(
                    ma.data().as_slice().unwrap(),
                    &expected_data,
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} data", case.name),
                );
                assert_eq!(ma.shape(), expected_shape.as_slice());
                assert_eq!(ma.dim().as_slice(), expected_shape.as_slice());
                assert_eq!(ma.ndim(), case.expected["ndim"].as_u64().unwrap() as usize);
                assert_eq!(ma.size(), case.expected["size"].as_u64().unwrap() as usize);
                assert_eq!(
                    ma.has_real_mask(),
                    case.expected["has_real_mask"].as_bool().unwrap()
                );
                assert_eq!(
                    ma.mask_opt().is_some(),
                    case.expected["mask_opt"].as_bool().unwrap()
                );
                assert_eq!(ma.mask().as_slice().unwrap(), expected_mask.as_slice());
                assert_f64_ulp(
                    ma.fill_value(),
                    parse_f64_value(&case.expected["fill_value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} fill_value", case.name),
                );
            }
            "masked_array_mutation" => {
                let replacement_fill_value =
                    parse_f64_value(&case.inputs["replacement_fill_value"]);
                let ma_with_fill =
                    make_masked_array(&case.inputs).with_fill_value(replacement_fill_value);
                assert_f64_ulp(
                    ma_with_fill.fill_value(),
                    parse_f64_value(&case.expected["fill_value_after_set"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} with_fill_value", case.name),
                );

                let mut ma = make_masked_array(&case.inputs);
                ma.set_fill_value(replacement_fill_value);
                assert_f64_ulp(
                    ma.fill_value(),
                    parse_f64_value(&case.expected["fill_value_after_set"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} set_fill_value", case.name),
                );
                ma.data_mut().unwrap()[1] = 20.0;
                let expected_data = parse_f64_data(&case.expected["data_after_data_mut"]["data"]);
                assert_f64_slice_ulp(
                    ma.data().as_slice().unwrap(),
                    &expected_data,
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &format!("{} data_mut", case.name),
                );

                let replacement_mask = make_bool_array(&case.inputs["replacement_mask"]);
                ma.set_mask(replacement_mask).unwrap();
                let expected_mask = parse_bool_data(&case.expected["mask_after_set_mask"]["data"]);
                assert_eq!(ma.mask().as_slice().unwrap(), expected_mask.as_slice());
            }
            "masked_array_hard_soft" => {
                let mut ma = make_masked_array(&case.inputs);
                ma.harden_mask().unwrap();
                assert!(ma.is_hard_mask(), "{} harden_mask flag", case.name);
                ma.set_mask_flat(1, false).unwrap();
                ma.set_mask_flat(0, true).unwrap();
                let expected_harden = parse_bool_data(&case.expected["mask_after_harden"]["data"]);
                assert_eq!(ma.mask().as_slice().unwrap(), expected_harden.as_slice());

                ma.soften_mask().unwrap();
                assert!(!ma.is_hard_mask(), "{} soften_mask flag", case.name);
                ma.set_mask_flat(1, false).unwrap();
                let expected_soften = parse_bool_data(&case.expected["mask_after_soften"]["data"]);
                assert_eq!(ma.mask().as_slice().unwrap(), expected_soften.as_slice());
            }
            "masked_array_shares_mask" => {
                let ma = make_masked_array(&case.inputs);
                let mut child = ma.clone();
                assert_eq!(
                    ma.shares_mask(),
                    case.expected["shares_after_clone"].as_bool().unwrap()
                );
                assert_eq!(
                    child.shares_mask(),
                    case.expected["shares_after_clone"].as_bool().unwrap()
                );
                child.set_mask_flat(0, true).unwrap();
                assert_eq!(
                    ma.shares_mask(),
                    case.expected["shares_after_child_mutation"]
                        .as_bool()
                        .unwrap()
                );
                assert_eq!(
                    child.shares_mask(),
                    case.expected["shares_after_child_mutation"]
                        .as_bool()
                        .unwrap()
                );

                let expected_parent = parse_bool_data(&case.expected["parent_mask"]["data"]);
                let expected_child = parse_bool_data(&case.expected["child_mask"]["data"]);
                assert_eq!(ma.mask().as_slice().unwrap(), expected_parent.as_slice());
                assert_eq!(child.mask().as_slice().unwrap(), expected_child.as_slice());
            }
            "common_fill_value_same" => {
                let lhs = make_masked_array(&case.inputs);
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::common_fill_value(&lhs, &rhs);
                assert_f64_ulp(
                    actual,
                    parse_f64_value(&case.expected["value"]),
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "ids" => {
                let ma = make_masked_array(&case.inputs);
                let (data_ptr, mask_ptr) = ferray_ma::ids(&ma);
                assert_eq!(case.expected["tuple_len"].as_u64().unwrap(), 2);
                assert!(!data_ptr.is_null(), "{} data ptr", case.name);
                assert!(!mask_ptr.is_null(), "{} mask ptr", case.name);
            }
            "interop_into_data" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma.into_data();
                let expected_data = parse_f64_data(&case.expected["data"]);
                assert_f64_slice_ulp(
                    actual.as_slice().unwrap(),
                    &expected_data,
                    case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
                    &case.name,
                );
            }
            "interop_apply_unary" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ma
                    .apply_unary(|arr| {
                        let data = arr.iter().map(|value| value.sqrt()).collect::<Vec<_>>();
                        Array::from_vec(arr.dim().clone(), data)
                    })
                    .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "interop_apply_unary_to" => {
                let ma = make_masked_array(&case.inputs);
                let default_for_masked = case.inputs["default_for_masked"].as_bool().unwrap();
                let actual = ma
                    .apply_unary_to(
                        |arr| {
                            let data = arr.iter().map(|value| *value > 0.0).collect::<Vec<_>>();
                            Array::from_vec(arr.dim().clone(), data)
                        },
                        default_for_masked,
                    )
                    .unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "interop_apply_binary" => {
                let lhs = make_masked_array(&case.inputs);
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = lhs
                    .apply_binary(&rhs, |left, right| {
                        let data = left
                            .iter()
                            .zip(right.iter())
                            .map(|(left, right)| *left + *right)
                            .collect::<Vec<_>>();
                        Array::from_vec(left.dim().clone(), data)
                    })
                    .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_apply_unary_masked" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ferray_ma::ma_apply_unary(&ma, |arr| {
                    let data = arr.iter().map(|value| *value * 2.0).collect::<Vec<_>>();
                    Array::from_vec(arr.dim().clone(), data)
                })
                .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_apply_unary_plain" => {
                let data = make_f64_array(&case.inputs["data"]);
                let actual = ferray_ma::ma_apply_unary(&data, |arr| {
                    let data = arr.iter().map(|value| *value * 2.0).collect::<Vec<_>>();
                    Array::from_vec(arr.dim().clone(), data)
                })
                .unwrap();
                assert_eq!(
                    actual.has_real_mask(),
                    case.expected["has_real_mask"].as_bool().unwrap()
                );
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "io_roundtrip" => {
                #[cfg(feature = "io")]
                {
                    let ma = make_masked_array(&case.inputs);
                    let dir = std::env::temp_dir().join(format!(
                        "ferray_ma_conformance_io_{}_{}",
                        std::process::id(),
                        case.name
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    let data_path = dir.join("data.npy");
                    let mask_path = dir.join("mask.npy");

                    ferray_ma::io::save_masked(&data_path, &mask_path, &ma).unwrap();
                    let actual: MaskedArray<f64, IxDyn> =
                        ferray_ma::io::load_masked(&data_path, &mask_path).unwrap();
                    assert_masked_all_data_matches(
                        &actual,
                        &case.expected,
                        case.tolerance_ulps,
                        &case.name,
                    );

                    let _ = std::fs::remove_file(&data_path);
                    let _ = std::fs::remove_file(&mask_path);
                    let _ = std::fs::remove_dir(&dir);
                }
                #[cfg(not(feature = "io"))]
                {
                    continue;
                }
            }
            op => panic!("unexpected masked core surface op {op}"),
        }
    }
}

/// Covers `ferray_ma::extras::ma_unique`,
/// `ferray_ma::extras::ma_vander`,
/// `ferray_ma::extras::ma_in1d`,
/// `ferray_ma::extras::ma_isin`,
/// `ferray_ma::extras::ma_equal`,
/// `ferray_ma::extras::ma_not_equal`,
/// `ferray_ma::extras::ma_less`,
/// `ferray_ma::extras::ma_greater`,
/// `ferray_ma::extras::ma_less_equal`,
/// `ferray_ma::extras::ma_greater_equal`,
/// `ferray_ma::extras::ma_logical_and`,
/// `ferray_ma::extras::ma_logical_or`,
/// `ferray_ma::extras::ma_logical_xor`,
/// `ferray_ma::extras::ma_logical_not`, plus root re-exports
/// `ferray_ma::ma_unique`, `ferray_ma::ma_vander`,
/// `ferray_ma::ma_in1d`, `ferray_ma::ma_isin`,
/// `ferray_ma::ma_equal`, `ferray_ma::ma_not_equal`,
/// `ferray_ma::ma_less`, `ferray_ma::ma_greater`,
/// `ferray_ma::ma_less_equal`, `ferray_ma::ma_greater_equal`,
/// `ferray_ma::ma_logical_and`, `ferray_ma::ma_logical_or`,
/// `ferray_ma::ma_logical_xor`, and `ferray_ma::ma_logical_not`.
/// Fixture: `fixtures/ma/masked_extras_set_logic.json`
/// (`numpy.ma` extras set, membership, comparison, and logical helpers).
#[test]
fn masked_extras_set_logic_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_extras_set_logic.json"));
    for case in &suite.test_cases {
        match case.inputs["op"].as_str().unwrap() {
            "ma_unique" => {
                let ma = make_masked_array(&case.inputs);
                let actual = ferray_ma::ma_unique(&ma).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_vander" => {
                let ma = make_masked_ix1(&case.inputs);
                let actual =
                    ferray_ma::ma_vander(&ma, Some(case.inputs["n"].as_u64().unwrap() as usize))
                        .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_in1d" => {
                let ma = make_masked_ix1(&case.inputs);
                let test_values = parse_f64_data(&case.inputs["test_values"]["data"]);
                let actual = ferray_ma::ma_in1d(&ma, &test_values).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_isin" => {
                let ma = make_masked_array(&case.inputs);
                let test_values = parse_f64_data(&case.inputs["test_values"]["data"]);
                let actual = ferray_ma::ma_isin(&ma, &test_values).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_equal" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_equal(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_not_equal" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_not_equal(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_less" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_less(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_greater" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_greater(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_less_equal" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_less_equal(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_greater_equal" => {
                let lhs = make_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_greater_equal(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_logical_and" => {
                let lhs = make_bool_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_bool_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_logical_and(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_logical_or" => {
                let lhs = make_bool_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_bool_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_logical_or(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_logical_xor" => {
                let lhs = make_bool_masked_array_from_keys(
                    &case.inputs,
                    "lhs_data",
                    "lhs_mask",
                    "lhs_fill_value",
                );
                let rhs = make_bool_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let actual = ferray_ma::ma_logical_xor(&lhs, &rhs).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            "ma_logical_not" => {
                let ma =
                    make_bool_masked_array_from_keys(&case.inputs, "data", "mask", "fill_value");
                let actual = ferray_ma::ma_logical_not(&ma).unwrap();
                assert_masked_bool_all_data_matches(&actual, &case.expected, &case.name);
            }
            op => panic!("unexpected masked extras set/logical op {op}"),
        }
    }
}

/// Covers `ferray_ma::extras::ma_concatenate`,
/// `ferray_ma::extras::ma_apply_along_axis`,
/// `ferray_ma::extras::ma_apply_over_axes`, plus root re-exports
/// `ferray_ma::ma_concatenate`, `ferray_ma::ma_apply_along_axis`,
/// and `ferray_ma::ma_apply_over_axes`.
/// Fixture: `fixtures/ma/masked_extras_functional.json`
/// (`numpy.ma` extras functional helpers).
#[test]
fn masked_extras_functional_match_numpy() {
    let suite = load_fixture(&ma_fx("masked_extras_functional.json"));
    for case in &suite.test_cases {
        match case.inputs["op"].as_str().unwrap() {
            "ma_concatenate" => {
                let lhs = make_masked_array(&case.inputs);
                let rhs = make_masked_array_from_keys(
                    &case.inputs,
                    "rhs_data",
                    "rhs_mask",
                    "rhs_fill_value",
                );
                let axis = case.inputs["axis"].as_u64().unwrap() as usize;
                let actual = ferray_ma::ma_concatenate(&lhs, &rhs, axis).unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_apply_along_axis" => {
                let ma = make_masked_array(&case.inputs);
                let axis = case.inputs["axis"].as_u64().unwrap() as usize;
                let actual = ferray_ma::ma_apply_along_axis(
                    &ma,
                    axis,
                    |lane: &MaskedArray<f64, Ix1>| -> ferray_core::FerrayResult<(f64, bool)> {
                        let all_masked = lane.mask().iter().all(|masked| *masked);
                        if all_masked {
                            Ok((0.0, true))
                        } else {
                            Ok((lane.sum()?, false))
                        }
                    },
                )
                .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            "ma_apply_over_axes" => {
                let ma = make_masked_array(&case.inputs);
                let axes = case.inputs["axes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|axis| axis.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                let actual = ferray_ma::ma_apply_over_axes(
                    &ma,
                    &axes,
                    |lane: &MaskedArray<f64, Ix1>| -> ferray_core::FerrayResult<(f64, bool)> {
                        let all_masked = lane.mask().iter().all(|masked| *masked);
                        if all_masked {
                            Ok((0.0, true))
                        } else {
                            Ok((lane.sum()?, false))
                        }
                    },
                )
                .unwrap();
                assert_masked_all_data_matches(
                    &actual,
                    &case.expected,
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            op => panic!("unexpected masked extras functional op {op}"),
        }
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
        for (i, (&actual, &expected)) in mask
            .as_slice()
            .unwrap()
            .iter()
            .zip(expected_mask.iter())
            .enumerate()
        {
            assert_eq!(actual, expected, "getmask case '{}' element {i}", case.name);
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
        for (i, (&actual, &expected)) in mask_slice.iter().zip(expected_mask.iter()).enumerate() {
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
        let (fill_value, expected_key) = if let Some(fv) = case.inputs.get("fill_value") {
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
