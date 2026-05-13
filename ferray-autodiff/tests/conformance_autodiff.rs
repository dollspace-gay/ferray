//! Conformance tests for ferray-autodiff against analytic forward-mode
//! derivatives. Each test loads a fixture under `fixtures/autodiff/`
//! and compares ferray's output to the analytic expected value.
//!
//! No autograd backward lane: ferray-autodiff is forward-mode dual
//! numbers; conformance covers `eval(f, x)` and `grad(f, x)` only.
//!
//! Each test calls the user-facing path and names the canonical inner
//! path in its doc comment so the surface-coverage gate's text-match
//! picks it up.

use ferray_autodiff::DualNumber;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_core::{Array, ArrayView};
use ferray_test_oracle::{
    assert_f64_slice_ulp, assert_f64_ulp, fixtures_dir, load_fixture, parse_f64_data,
    parse_f64_value, parse_shape, parse_usize_data,
};

fn ad_fixture(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("autodiff").join(name)
}

fn fn_id_to_str(value: &serde_json::Value) -> &str {
    value.as_str().expect("fn_id must be a string")
}

// ---------------------------------------------------------------------------
// api::derivative
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::api::derivative` (re-exported as
/// `ferray_autodiff::derivative`).
///
/// The fixture encodes `inputs.x` and `inputs.fn_id`; the test selects
/// the closure by id and compares the returned scalar derivative to
/// the analytic `expected.deriv`. Forward-mode AD seeds `x` as a
/// `DualNumber::variable`, so the dual part of `f(x)` equals `f'(x)`.
#[test]
fn derivative_matches_analytic() {
    let suite = load_fixture(&ad_fixture("derivative.json"));
    for case in &suite.test_cases {
        let x = parse_f64_value(&case.inputs["x"]);
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let deriv = match fn_id {
            "x_squared" => ferray_autodiff::derivative(|v| v * v, x),
            "sin" => ferray_autodiff::derivative(|v| v.sin(), x),
            "exp" => ferray_autodiff::derivative(|v| v.exp(), x),
            other => panic!("unknown fn_id '{other}' in derivative.json"),
        };
        let expected = parse_f64_value(&case.expected["deriv"]);
        assert_f64_ulp(deriv, expected, case.tolerance_ulps, &case.name);
    }
}

// ---------------------------------------------------------------------------
// api::gradient
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::api::gradient` (re-exported as
/// `ferray_autodiff::gradient`).
///
/// Forward-mode AD with one variable seeded per pass: for each
/// component the function is evaluated with that component as a
/// `DualNumber::variable` and the rest as `DualNumber::constant`.
#[test]
fn gradient_matches_analytic() {
    let suite = load_fixture(&ad_fixture("gradient.json"));
    for case in &suite.test_cases {
        let point = parse_f64_data(&case.inputs["point"]);
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let grad = match fn_id {
            "sum_of_squares" => ferray_autodiff::gradient(
                |v: &[DualNumber<f64>]| v[0] * v[0] + v[1] * v[1],
                &point,
            ),
            "product_of_three" => ferray_autodiff::gradient(
                |v: &[DualNumber<f64>]| v[0] * v[1] * v[2],
                &point,
            ),
            other => panic!("unknown fn_id '{other}' in gradient.json"),
        };
        let expected = parse_f64_data(&case.expected["gradient"]);
        assert_f64_slice_ulp(&grad, &expected, case.tolerance_ulps, &case.name);
    }
}

// ---------------------------------------------------------------------------
// api::jacobian
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::api::jacobian` (re-exported as
/// `ferray_autodiff::jacobian`).
///
/// Returns `(flat row-major Vec<T>, m)`. The fixture encodes the
/// expected flat Jacobian, output dimension `m`, and input dimension
/// `n`. We verify both `m` and the flat values.
#[test]
fn jacobian_matches_analytic() {
    let suite = load_fixture(&ad_fixture("jacobian.json"));
    for case in &suite.test_cases {
        let point = parse_f64_data(&case.inputs["point"]);
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let (jac, m) = match fn_id {
            "linear_pair" => ferray_autodiff::jacobian(
                |v: &[DualNumber<f64>]| vec![v[0] + v[1], v[0] - v[1]],
                &point,
            ),
            "product_and_sum" => ferray_autodiff::jacobian(
                |v: &[DualNumber<f64>]| vec![v[0] * v[1], v[0] + v[1]],
                &point,
            ),
            other => panic!("unknown fn_id '{other}' in jacobian.json"),
        };
        let expected_m = case.expected["m"].as_u64().expect("m must be u64") as usize;
        assert_eq!(m, expected_m, "case '{}': m mismatch", case.name);
        let expected = parse_f64_data(&case.expected["jacobian"]);
        assert_f64_slice_ulp(&jac, &expected, case.tolerance_ulps, &case.name);
    }
}

// ---------------------------------------------------------------------------
// array_ops::derivative_elementwise
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::array_ops::derivative_elementwise`
/// (re-exported as `ferray_autodiff::derivative_elementwise`).
#[test]
fn derivative_elementwise_matches_analytic() {
    let suite = load_fixture(&ad_fixture("derivative_elementwise.json"));
    for case in &suite.test_cases {
        let data = parse_f64_data(&case.inputs["x"]["data"]);
        let shape = parse_shape(&case.inputs["x"]["shape"]);
        let n = shape[0];
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let dx = match fn_id {
            "x_squared" => ferray_autodiff::array_ops::derivative_elementwise(|v| v * v, &x)
                .unwrap_or_else(|e| panic!("derivative_elementwise {}: {e}", case.name)),
            other => panic!("unknown fn_id '{other}' in derivative_elementwise.json"),
        };
        let expected = parse_f64_data(&case.expected["data"]);
        let actual: &[f64] = dx.as_slice().unwrap();
        assert_f64_slice_ulp(actual, &expected, case.tolerance_ulps, &case.name);
    }
}

// ---------------------------------------------------------------------------
// array_ops::value_and_derivative_elementwise
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::array_ops::value_and_derivative_elementwise`
/// (re-exported as `ferray_autodiff::value_and_derivative_elementwise`).
#[test]
fn value_and_derivative_elementwise_matches_analytic() {
    let suite = load_fixture(&ad_fixture("value_and_derivative_elementwise.json"));
    for case in &suite.test_cases {
        let data = parse_f64_data(&case.inputs["x"]["data"]);
        let shape = parse_shape(&case.inputs["x"]["shape"]);
        let n = shape[0];
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let (values, derivs) = match fn_id {
            "x_squared" => {
                ferray_autodiff::array_ops::value_and_derivative_elementwise(|v| v * v, &x)
                    .unwrap_or_else(|e| {
                        panic!("value_and_derivative_elementwise {}: {e}", case.name)
                    })
            }
            other => panic!(
                "unknown fn_id '{other}' in value_and_derivative_elementwise.json"
            ),
        };
        let expected_values = parse_f64_data(&case.expected["values"]["data"]);
        let expected_derivs = parse_f64_data(&case.expected["derivs"]["data"]);
        assert_f64_slice_ulp(
            values.as_slice().unwrap(),
            &expected_values,
            case.tolerance_ulps,
            &format!("{}:values", case.name),
        );
        assert_f64_slice_ulp(
            derivs.as_slice().unwrap(),
            &expected_derivs,
            case.tolerance_ulps,
            &format!("{}:derivs", case.name),
        );
    }
}

// ---------------------------------------------------------------------------
// array_ops::gradient_vector
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::array_ops::gradient_vector` (re-exported
/// as `ferray_autodiff::gradient_vector`).
#[test]
fn gradient_vector_matches_analytic() {
    let suite = load_fixture(&ad_fixture("gradient_vector.json"));
    for case in &suite.test_cases {
        let data = parse_f64_data(&case.inputs["point"]["data"]);
        let shape = parse_shape(&case.inputs["point"]["shape"]);
        let n = shape[0];
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let grad = match fn_id {
            "weighted_squares" => ferray_autodiff::array_ops::gradient_vector(
                |xs: &[DualNumber<f64>]| {
                    xs[0] * xs[0] + xs[1] * xs[1] * 2.0 + xs[2] * xs[2] * 3.0
                },
                &x,
            )
            .unwrap_or_else(|e| panic!("gradient_vector {}: {e}", case.name)),
            other => panic!("unknown fn_id '{other}' in gradient_vector.json"),
        };
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            grad.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// array_ops::value_and_gradient
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::array_ops::value_and_gradient` (re-exported
/// as `ferray_autodiff::value_and_gradient`).
#[test]
fn value_and_gradient_matches_analytic() {
    let suite = load_fixture(&ad_fixture("value_and_gradient.json"));
    for case in &suite.test_cases {
        let data = parse_f64_data(&case.inputs["point"]["data"]);
        let shape = parse_shape(&case.inputs["point"]["shape"]);
        let n = shape[0];
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let (value, grad) = match fn_id {
            "linear_plus_product" => ferray_autodiff::array_ops::value_and_gradient(
                |xs: &[DualNumber<f64>]| xs[0] + xs[1] * xs[2],
                &x,
            )
            .unwrap_or_else(|e| panic!("value_and_gradient {}: {e}", case.name)),
            other => panic!("unknown fn_id '{other}' in value_and_gradient.json"),
        };
        let expected_value = parse_f64_value(&case.expected["value"]);
        assert_f64_ulp(
            value,
            expected_value,
            case.tolerance_ulps,
            &format!("{}:value", case.name),
        );
        let expected_grad = parse_f64_data(&case.expected["gradient"]["data"]);
        assert_f64_slice_ulp(
            grad.as_slice().unwrap(),
            &expected_grad,
            case.tolerance_ulps,
            &format!("{}:gradient", case.name),
        );
    }
}

// ---------------------------------------------------------------------------
// array_ops::jacobian_array
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::array_ops::jacobian_array` (re-exported
/// as `ferray_autodiff::jacobian_array`).
#[test]
fn jacobian_array_matches_analytic() {
    let suite = load_fixture(&ad_fixture("jacobian_array.json"));
    for case in &suite.test_cases {
        let data = parse_f64_data(&case.inputs["point"]["data"]);
        let shape = parse_shape(&case.inputs["point"]["shape"]);
        let n = shape[0];
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let fn_id = fn_id_to_str(&case.inputs["fn_id"]);
        let jac = match fn_id {
            "linear_pair" => ferray_autodiff::array_ops::jacobian_array(
                |xs: &[DualNumber<f64>]| vec![xs[0] + xs[1], xs[0] - xs[1]],
                &x,
            )
            .unwrap_or_else(|e| panic!("jacobian_array {}: {e}", case.name)),
            other => panic!("unknown fn_id '{other}' in jacobian_array.json"),
        };
        let expected_shape = parse_usize_data(&case.expected["shape"]);
        assert_eq!(
            jac.shape(),
            expected_shape.as_slice(),
            "case '{}': jacobian shape mismatch",
            case.name,
        );
        let expected = parse_f64_data(&case.expected["data"]);
        let view: ArrayView<'_, f64, Ix2> = jac.view();
        let actual: Vec<f64> = view.iter().copied().collect();
        assert_f64_slice_ulp(&actual, &expected, case.tolerance_ulps, &case.name);
    }
}

// ---------------------------------------------------------------------------
// functions::atan2 (free function on DualNumber)
// ---------------------------------------------------------------------------

/// Covers: `ferray_autodiff::functions::atan2` — the free-function
/// form (re-exported as `ferray_autodiff::atan2`). The fixture seeds
/// the dual parts of `y` and `x` directly so a single evaluation
/// returns either ∂atan2/∂y or ∂atan2/∂x depending on the seeding.
#[test]
fn atan2_matches_analytic() {
    let suite = load_fixture(&ad_fixture("atan2.json"));
    for case in &suite.test_cases {
        let y_re = parse_f64_value(&case.inputs["y"]);
        let x_re = parse_f64_value(&case.inputs["x"]);
        let y_du = parse_f64_value(&case.inputs["y_dual"]);
        let x_du = parse_f64_value(&case.inputs["x_dual"]);
        let y = DualNumber::new(y_re, y_du);
        let x = DualNumber::new(x_re, x_du);
        let out = ferray_autodiff::functions::atan2(y, x);
        let expected_value = parse_f64_value(&case.expected["value"]);
        let expected_deriv = parse_f64_value(&case.expected["deriv"]);
        assert_f64_ulp(
            out.real,
            expected_value,
            case.tolerance_ulps,
            &format!("{}:value", case.name),
        );
        assert_f64_ulp(
            out.dual,
            expected_deriv,
            case.tolerance_ulps,
            &format!("{}:deriv", case.name),
        );
    }
}
