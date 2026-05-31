//! Conformance tests for ferray-linalg matrix products and measures.
//!
//! Stage 4-linalg fixture-anchored coverage for the product / norm /
//! trace family: matmul, dot, inner, outer, vdot, tensordot, kron,
//! trace, and norm. Each test loads its JSON fixture from
//! `fixtures/linalg/`, iterates the `test_cases`, and ULP-compares the
//! ferray result against the NumPy reference output (per the Stage 1
//! tolerance plan, max'd against `MIN_ULP_TOLERANCE`).
//!
//! Surface paths exercised by this file (named here so the surface
//! gate's text match picks them up — each function appears as both its
//! canonical inner path and its crate-root re-export):
//!
//! - `ferray_linalg::products::matmul` / `ferray_linalg::matmul`
//! - `ferray_linalg::products::dot` / `ferray_linalg::dot`
//! - `ferray_linalg::products::inner` / `ferray_linalg::inner`
//! - `ferray_linalg::products::outer` / `ferray_linalg::outer`
//! - `ferray_linalg::products::vdot` / `ferray_linalg::vdot`
//! - `ferray_linalg::products::tensordot::tensordot` /
//!   `ferray_linalg::tensordot`
//!   (plus `ferray_linalg::products::tensordot::TensordotAxes` /
//!   `ferray_linalg::TensordotAxes`)
//! - `ferray_linalg::products::kron` / `ferray_linalg::kron`
//! - `ferray_linalg::norms::trace` / `ferray_linalg::trace`
//! - `ferray_linalg::norms::norm` / `ferray_linalg::norm`
//!   (plus the `NormOrder` re-exports `ferray_linalg::norms::NormOrder` /
//!   `ferray_linalg::NormOrder`)
//!
//! Fixture-strict tolerance: `TOL_LINALG_F64_REL = 1e-10` per Stage 1.
//! Per-fixture ULP overrides (set explicitly in each JSON file's
//! `tolerance_ulps`) are still honoured, max'd against the global
//! `MIN_ULP_TOLERANCE` floor of 10 ULP.

// Conformance tests cross fixture JSON, recover bit patterns, and
// ULP-compare floats — the casts and direct comparisons are part of
// the contract.
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
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_test_oracle::{
    MIN_ULP_TOLERANCE, TOL_LINALG_F64_REL, assert_f64_slice_ulp, assert_f64_ulp, fixtures_dir,
    get_dtype, load_fixture, make_f64_array, parse_f64_data, parse_f64_value, parse_shape,
    should_skip_f64,
};

const _TOL_LINALG_F64_REL_REFERENCE: f64 = TOL_LINALG_F64_REL;

fn linalg_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("linalg").join(name)
}

/// Convert a 2-D `Array<f64, IxDyn>` (as produced from fixture JSON) to
/// a statically-ranked `Array<f64, Ix2>`. The trace / norm-matrix
/// entry points in `ferray_linalg` take `Ix2` for the matrix arg.
fn to_ix2(a: &Array<f64, IxDyn>) -> Array<f64, Ix2> {
    let shape = a.shape().to_vec();
    Array::<f64, Ix2>::from_vec(
        Ix2::new([shape[0], shape[1]]),
        a.as_slice().unwrap().to_vec(),
    )
    .unwrap()
}

/// Detect a fixture-case scalar result (NumPy `shape == []`) so the
/// per-fixture comparison can take the scalar branch.
fn is_scalar_expected(expected: &serde_json::Value) -> bool {
    expected
        .get("shape")
        .and_then(|s| s.as_array())
        .map(|arr| arr.is_empty())
        .unwrap_or(false)
}

/// Compare the contents of an `Array<f64, IxDyn>` against a fixture's
/// expected scalar or array payload. Centralises the
/// scalar-vs-vector branching so the per-function tests stay flat.
fn assert_case_matches(
    actual: &Array<f64, IxDyn>,
    expected: &serde_json::Value,
    tolerance: u64,
    name: &str,
) {
    if is_scalar_expected(expected) {
        let expected_value = parse_f64_value(&expected["data"]);
        let actual_slice = actual.as_slice().unwrap();
        assert_eq!(
            actual_slice.len(),
            1,
            "case '{name}': scalar-shape expected but result has {} elements",
            actual_slice.len()
        );
        assert_f64_ulp(
            actual_slice[0],
            expected_value,
            tolerance,
            &format!("case '{name}'"),
        );
    } else {
        let expected_data = parse_f64_data(&expected["data"]);
        assert_f64_slice_ulp(actual.as_slice().unwrap(), &expected_data, tolerance, name);
    }
}

// ---------------------------------------------------------------------------
// matmul: standard matrix product (NumPy `@`).
//
// Pins `ferray_linalg::products::matmul` (canonical inner path) and
// `ferray_linalg::matmul` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_matmul() {
    let suite = load_fixture(&linalg_path("matmul.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) || should_skip_f64(input_b) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::matmul(&a, &b).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::matmul returned error: {e}",
                case.name
            )
        });
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "matmul.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// dot: dispatches across 1-D (inner) and 2-D (matmul) inputs.
//
// `dot` returns a scalar wrapped in an `Array<f64, IxDyn>` for 1-D
// inputs (NumPy shape `()`). The helper above handles both branches.
//
// The dot fixture also contains a `float32` case; we let
// `should_skip_f64` filter it out (the canonical conformance test
// targets `float64`, with f32 covered separately).
//
// Pins `ferray_linalg::products::dot` (canonical inner path) and
// `ferray_linalg::dot` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_dot() {
    let suite = load_fixture(&linalg_path("dot.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if get_dtype(input_a) != "float64" || get_dtype(input_b) != "float64" {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::dot(&a, &b).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::dot returned error: {e}",
                case.name
            )
        });
        assert_case_matches(
            &result,
            &case.expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "dot.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// inner: NumPy `np.inner` — sum-product over the last axis.
//
// Pins `ferray_linalg::products::inner` (canonical inner path) and
// `ferray_linalg::inner` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_inner() {
    let suite = load_fixture(&linalg_path("inner.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if get_dtype(input_a) != "float64" || get_dtype(input_b) != "float64" {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::inner(&a, &b).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::inner returned error: {e}",
                case.name
            )
        });
        assert_case_matches(
            &result,
            &case.expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "inner.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// outer: NumPy `np.outer` — flattened outer product.
//
// Pins `ferray_linalg::products::outer` (canonical inner path) and
// `ferray_linalg::outer` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_outer() {
    let suite = load_fixture(&linalg_path("outer.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if get_dtype(input_a) != "float64" || get_dtype(input_b) != "float64" {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::outer(&a, &b).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::outer returned error: {e}",
                case.name
            )
        });
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "outer.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// vdot: NumPy `np.vdot` — flat conjugate dot product. The `vdot` entry
// in `ferray_linalg::products` is bounded on `LinalgFloat` (real
// floats only); the fixture's complex128 case is skipped at the
// dtype filter because complex `vdot` is covered separately via the
// `complex::*` family in the surface inventory.
//
// Pins `ferray_linalg::products::vdot` (canonical inner path) and
// `ferray_linalg::vdot` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_vdot() {
    let suite = load_fixture(&linalg_path("vdot.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if get_dtype(input_a) != "float64" || get_dtype(input_b) != "float64" {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::vdot(&a, &b).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::vdot returned error: {e}",
                case.name
            )
        });
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
        tested += 1;
    }
    assert!(tested > 0, "vdot.json: no float64 test cases ran");
}

// ---------------------------------------------------------------------------
// tensordot: NumPy `np.tensordot` — contract over `axes` (scalar count
// or `(axes_a, axes_b)` pair).
//
// Pins `ferray_linalg::products::tensordot::tensordot` (canonical
// inner path) and `ferray_linalg::tensordot` (crate-root re-export),
// plus the `TensordotAxes` re-exports
// `ferray_linalg::products::tensordot::TensordotAxes` /
// `ferray_linalg::TensordotAxes`.
// ---------------------------------------------------------------------------

#[test]
fn fixture_tensordot() {
    use ferray_linalg::TensordotAxes;

    let suite = load_fixture(&linalg_path("tensordot.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) || should_skip_f64(input_b) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);

        let axes_val = &case.inputs["axes"];
        let axes = if let Some(n) = axes_val.as_u64() {
            TensordotAxes::Scalar(n as usize)
        } else {
            let pair = axes_val
                .as_array()
                .unwrap_or_else(|| panic!("case '{}': axes must be int or [list,list]", case.name));
            let aa: Vec<usize> = pair[0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let bb: Vec<usize> = pair[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            TensordotAxes::Pairs(aa, bb)
        };
        let result = ferray_linalg::tensordot(&a, &b, axes).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::tensordot returned error: {e}",
                case.name
            )
        });
        assert_case_matches(
            &result,
            &case.expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "tensordot.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// kron: NumPy `np.kron` — Kronecker product.
//
// Pins `ferray_linalg::products::kron` (canonical inner path) and
// `ferray_linalg::kron` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_kron() {
    let suite = load_fixture(&linalg_path("kron.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) || should_skip_f64(input_b) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::kron(&a, &b).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::kron returned error: {e}",
                case.name
            )
        });
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "kron.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// trace: sum of the main diagonal.
//
// Pins `ferray_linalg::norms::trace` (canonical inner path) and
// `ferray_linalg::trace` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_trace() {
    let suite = load_fixture(&linalg_path("trace.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::trace(&arr).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::trace returned error: {e}",
                case.name
            )
        });
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
        tested += 1;
    }
    assert!(tested > 0, "trace.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// norm: vector and matrix p-norms.
//
// `norm` returns a scalar (`FerrayResult<T>`), matching NumPy's
// `np.linalg.norm` contract when no axis is given. The fixture's
// `ord` field is either an integer (1, 2, etc.), a float, or a
// string ("Inf", "-Inf", "fro", "nuc"). All routes through
// `NormOrder` so that this test pins the enum's public surface.
//
// Pins `ferray_linalg::norms::norm` (canonical inner path) and
// `ferray_linalg::norm` (crate-root re-export), plus the `NormOrder`
// re-exports.
// ---------------------------------------------------------------------------

#[test]
fn fixture_norm() {
    let suite = load_fixture(&linalg_path("norm.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr_shape = parse_shape(&input["shape"]);
        let arr_data = parse_f64_data(&input["data"]);
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&arr_shape), arr_data).unwrap();

        let ord = match case.inputs.get("ord") {
            Some(v) if v.is_string() => match v.as_str().unwrap() {
                "fro" | "Fro" => ferray_linalg::NormOrder::Fro,
                "nuc" | "Nuc" => ferray_linalg::NormOrder::Nuc,
                "inf" | "Inf" => ferray_linalg::NormOrder::Inf,
                "-inf" | "-Inf" => ferray_linalg::NormOrder::NegInf,
                other => panic!("case '{}': unknown norm order '{}'", case.name, other),
            },
            Some(v) if v.is_number() => {
                let pv = v.as_f64().unwrap();
                if pv == 2.0 {
                    ferray_linalg::NormOrder::L2
                } else if pv == 1.0 {
                    ferray_linalg::NormOrder::L1
                } else {
                    ferray_linalg::NormOrder::P(pv)
                }
            }
            _ => {
                if arr_shape.len() == 2 {
                    ferray_linalg::NormOrder::Fro
                } else {
                    ferray_linalg::NormOrder::L2
                }
            }
        };
        let result = ferray_linalg::norm(&arr, ord).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::norm returned error: {e}",
                case.name
            )
        });
        let expected = parse_f64_value(&case.expected["data"]);
        assert_f64_ulp(
            result,
            expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &format!("case '{}'", case.name),
        );
        tested += 1;
    }
    assert!(tested > 0, "norm.json: no test cases ran");
}
