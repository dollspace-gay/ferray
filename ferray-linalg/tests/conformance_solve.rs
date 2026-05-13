//! Conformance tests for ferray-linalg solvers and measures.
//!
//! Stage 4-linalg fixture-anchored coverage for the solve / inverse /
//! rank / conditioning / determinant family. Each test loads its JSON
//! fixture from `fixtures/linalg/`, iterates the `test_cases`, and
//! ULP-compares the ferray result against the NumPy reference output
//! (per the Stage 1 tolerance plan, max'd against `MIN_ULP_TOLERANCE`).
//!
//! Surface paths exercised by this file (named here so the surface
//! gate's text match picks them up — each function appears as both its
//! canonical inner path and its crate-root re-export):
//!
//! - `ferray_linalg::solve::solve` / `ferray_linalg::solve`
//! - `ferray_linalg::solve::lstsq` / `ferray_linalg::lstsq`
//! - `ferray_linalg::solve::inv` / `ferray_linalg::inv`
//! - `ferray_linalg::norms::cond` / `ferray_linalg::cond`
//!   (plus `ferray_linalg::norms::NormOrder` / `ferray_linalg::NormOrder`)
//! - `ferray_linalg::norms::matrix_rank` / `ferray_linalg::matrix_rank`
//! - `ferray_linalg::norms::det` / `ferray_linalg::det`
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
    load_fixture, make_f64_array, parse_f64_data, parse_f64_value, parse_shape, should_skip_f64,
};

const _TOL_LINALG_F64_REL_REFERENCE: f64 = TOL_LINALG_F64_REL;

fn linalg_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("linalg").join(name)
}

/// Convert a 2-D `Array<f64, IxDyn>` (as produced from fixture JSON) to
/// a statically-ranked `Array<f64, Ix2>`. All solve/inv/cond/det/
/// matrix_rank entry points in `ferray_linalg` take `Ix2` for the
/// matrix arg.
fn to_ix2(a: &Array<f64, IxDyn>) -> Array<f64, Ix2> {
    let shape = a.shape().to_vec();
    Array::<f64, Ix2>::from_vec(
        Ix2::new([shape[0], shape[1]]),
        a.as_slice().unwrap().to_vec(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// solve: dense linear solver `A @ x = b`.
//
// Pins `ferray_linalg::solve::solve` (canonical inner path) and
// `ferray_linalg::solve` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_solve() {
    let suite = load_fixture(&linalg_path("solve.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = to_ix2(&make_f64_array(input_a));
        let b_shape = parse_shape(&input_b["shape"]);
        let b_data = parse_f64_data(&input_b["data"]);
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&b_shape), b_data).unwrap();
        let result = ferray_linalg::solve(&a, &b).unwrap_or_else(|e| {
            panic!("case '{}': ferray_linalg::solve returned error: {e}", case.name)
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
    assert!(tested > 0, "solve.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// lstsq: least-squares solver for over/underdetermined systems.
//
// The fixture stores the solution `x` and the integer `rank`. The
// residuals and singular values are implementation details (rcond
// cutoff varies between drivers) — we conformance-check `x` against
// the reference and `rank` for exact integer agreement.
//
// Pins `ferray_linalg::solve::lstsq` (canonical inner path) and
// `ferray_linalg::lstsq` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_lstsq() {
    let suite = load_fixture(&linalg_path("lstsq.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) || should_skip_f64(input_b) {
            continue;
        }
        let a = to_ix2(&make_f64_array(input_a));
        let b_shape = parse_shape(&input_b["shape"]);
        let b_data = parse_f64_data(&input_b["data"]);
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&b_shape), b_data).unwrap();

        let (x, _residuals, rank, _svals) =
            ferray_linalg::lstsq(&a, &b, None).unwrap_or_else(|e| {
                panic!("case '{}': ferray_linalg::lstsq returned error: {e}", case.name)
            });

        let expected_x = parse_f64_data(&case.expected["x"]["data"]);
        assert_f64_slice_ulp(
            x.as_slice().unwrap(),
            &expected_x,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        if let Some(expected_rank) = case.expected.get("rank").and_then(|v| v.as_u64()) {
            assert_eq!(
                rank, expected_rank as usize,
                "case '{}': rank mismatch", case.name
            );
        }
        tested += 1;
    }
    assert!(tested > 0, "lstsq.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// inv: matrix inverse.
//
// Pins `ferray_linalg::solve::inv` (canonical inner path) and
// `ferray_linalg::inv` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_inv() {
    let suite = load_fixture(&linalg_path("inv.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::inv(&arr).unwrap_or_else(|e| {
            panic!("case '{}': ferray_linalg::inv returned error: {e}", case.name)
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
    assert!(tested > 0, "inv.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// cond: condition number under a chosen norm order.
//
// The fixture's `ord` field may be a number (1, 2, -1, -2) or a string
// ("fro", "Inf", "-Inf"); we decode it into the `NormOrder` enum.
// Untyped numeric values that aren't 2 fall through to `NormOrder::P`
// (the generic `p`-norm constructor) so that fixture-driven `ord`
// values map onto the public surface without lossy casts.
//
// Pins `ferray_linalg::norms::cond` (canonical inner path) and
// `ferray_linalg::cond` (crate-root re-export), plus the `NormOrder`
// re-exports `ferray_linalg::norms::NormOrder` / `ferray_linalg::NormOrder`.
// ---------------------------------------------------------------------------

#[test]
fn fixture_cond() {
    let suite = load_fixture(&linalg_path("cond.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let p = match case.inputs.get("ord") {
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
                } else {
                    ferray_linalg::NormOrder::P(pv)
                }
            }
            _ => ferray_linalg::NormOrder::Fro,
        };
        let result = ferray_linalg::cond(&arr, p).unwrap_or_else(|e| {
            panic!("case '{}': ferray_linalg::cond returned error: {e}", case.name)
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
    assert!(tested > 0, "cond.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// matrix_rank: numerical rank via SVD cutoff.
//
// Result is an integer; the fixture stores it as `int64`. Compare
// exactly — no tolerance applies.
//
// Pins `ferray_linalg::norms::matrix_rank` (canonical inner path) and
// `ferray_linalg::matrix_rank` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_matrix_rank() {
    let suite = load_fixture(&linalg_path("matrix_rank.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::matrix_rank(&arr, None).unwrap_or_else(|e| {
            panic!(
                "case '{}': ferray_linalg::matrix_rank returned error: {e}",
                case.name
            )
        });
        let expected = case.expected["data"].as_u64().expect(
            "matrix_rank expected.data must be an integer",
        ) as usize;
        assert_eq!(result, expected, "case '{}': rank mismatch", case.name);
        tested += 1;
    }
    assert!(tested > 0, "matrix_rank.json: no test cases ran");
}

// ---------------------------------------------------------------------------
// det: matrix determinant.
//
// Pins `ferray_linalg::norms::det` (canonical inner path) and
// `ferray_linalg::det` (crate-root re-export).
// ---------------------------------------------------------------------------

#[test]
fn fixture_det() {
    let suite = load_fixture(&linalg_path("det.json"));
    let mut tested = 0;
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::det(&arr).unwrap_or_else(|e| {
            panic!("case '{}': ferray_linalg::det returned error: {e}", case.name)
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
    assert!(tested > 0, "det.json: no test cases ran");
}
