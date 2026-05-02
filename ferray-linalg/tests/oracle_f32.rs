//! f32 oracle tests for ferray-linalg (#213).
//!
//! Mirrors `tests/oracle.rs` for f32. Fixtures are stored in float64;
//! we cast inputs and expected outputs down to f32 and compare with
//! a wider ULP tolerance to absorb the precision gap.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_test_oracle::*;

fn linalg_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("linalg").join(name)
}

fn to_ix2_f32(a: &Array<f32, IxDyn>) -> Array<f32, Ix2> {
    let shape = a.shape().to_vec();
    Array::<f32, Ix2>::from_vec(
        Ix2::new([shape[0], shape[1]]),
        a.as_slice().unwrap().to_vec(),
    )
    .unwrap()
}

fn cast_expected(values: &serde_json::Value) -> Vec<f32> {
    parse_f64_data(values)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// f32 ULP budget. f32 has ~7 decimal digits vs f64's ~15, so a fixture
/// case authored at the f64 ULP-2 baseline still needs a much wider
/// envelope when cast down. 4096 ULPs at f32 magnitude ~1e0 corresponds
/// to ~4.9e-4 absolute — comfortably above f32 round-trip rounding for
/// the kinds of small matrices stored in the linalg fixtures.
const F32_TOL_ULPS: u64 = 4096;

#[test]
fn oracle_det_f32() {
    let suite = load_fixture(&linalg_path("det.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let result = ferray_linalg::det(&arr).unwrap();
        let expected = parse_f64_value(&case.expected["data"]) as f32;
        assert_f32_ulp(
            result,
            expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_trace_f32() {
    let suite = load_fixture(&linalg_path("trace.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let result = ferray_linalg::trace(&arr).unwrap();
        let expected = parse_f64_value(&case.expected["data"]) as f32;
        assert_f32_ulp(
            result,
            expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_inv_f32() {
    let suite = load_fixture(&linalg_path("inv.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let result = ferray_linalg::inv(&arr).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_cholesky_f32() {
    let suite = load_fixture(&linalg_path("cholesky.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let result = ferray_linalg::cholesky(&arr).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_matmul_f32() {
    let suite = load_fixture(&linalg_path("matmul.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = make_f32_array_from_any(input_a);
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::matmul(&a, &b).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_dot_f32() {
    let suite = load_fixture(&linalg_path("dot.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = make_f32_array_from_any(input_a);
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::dot(&a, &b).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_inner_f32() {
    let suite = load_fixture(&linalg_path("inner.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = make_f32_array_from_any(input_a);
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::inner(&a, &b).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_outer_f32() {
    let suite = load_fixture(&linalg_path("outer.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = make_f32_array_from_any(input_a);
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::outer(&a, &b).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_kron_f32() {
    let suite = load_fixture(&linalg_path("kron.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = make_f32_array_from_any(input_a);
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::kron(&a, &b).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_vdot_f32() {
    let suite = load_fixture(&linalg_path("vdot.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = make_f32_array_from_any(input_a);
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::vdot(&a, &b).unwrap();
        let expected = parse_f64_value(&case.expected["data"]) as f32;
        assert_f32_ulp(
            result,
            expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_norm_f32() {
    let suite = load_fixture(&linalg_path("norm.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = make_f32_array_from_any(input);
        let ord = match case.inputs.get("ord") {
            Some(v) if v.is_string() => match v.as_str().unwrap() {
                "fro" => ferray_linalg::NormOrder::Fro,
                "nuc" => ferray_linalg::NormOrder::Nuc,
                "inf" | "Inf" => ferray_linalg::NormOrder::Inf,
                "-inf" | "-Inf" => ferray_linalg::NormOrder::NegInf,
                other => panic!("unknown norm order string: {other}"),
            },
            Some(v) if v.is_number() => {
                let p = v.as_f64().unwrap();
                if p == 1.0 {
                    ferray_linalg::NormOrder::L1
                } else if p == 2.0 {
                    ferray_linalg::NormOrder::L2
                } else if p == f64::INFINITY {
                    ferray_linalg::NormOrder::Inf
                } else if p == f64::NEG_INFINITY {
                    ferray_linalg::NormOrder::NegInf
                } else {
                    ferray_linalg::NormOrder::P(p)
                }
            }
            _ => ferray_linalg::NormOrder::Fro,
        };
        let result = ferray_linalg::norm(&arr, ord).unwrap();
        let expected = parse_f64_value(&case.expected["data"]) as f32;
        assert_f32_ulp(
            result,
            expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &format!("case '{}'", case.name),
        );
    }
}

#[test]
fn oracle_solve_f32() {
    let suite = load_fixture(&linalg_path("solve.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f32(input_a) || should_skip_f32(input_b) {
            continue;
        }
        let a = to_ix2_f32(&make_f32_array_from_any(input_a));
        let b = make_f32_array_from_any(input_b);
        let result = ferray_linalg::solve(&a, &b).unwrap();
        let expected = cast_expected(&case.expected["data"]);
        assert_f32_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_matrix_rank_f32() {
    let suite = load_fixture(&linalg_path("matrix_rank.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let result = ferray_linalg::matrix_rank(&arr, None).unwrap();
        let expected = case.expected["data"].as_u64().unwrap() as usize;
        assert_eq!(result, expected, "case '{}': rank mismatch", case.name);
    }
}

#[test]
fn oracle_eigh_f32() {
    let suite = load_fixture(&linalg_path("eigh.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let (eigenvalues, _eigenvectors) = ferray_linalg::eigh(&arr).unwrap();
        let expected_w = cast_expected(&case.expected["eigenvalues"]["data"]);
        assert_f32_slice_ulp(
            eigenvalues.as_slice().unwrap(),
            &expected_w,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_svd_f32() {
    let suite = load_fixture(&linalg_path("svd.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let (_u, s, _vt) = ferray_linalg::svd(&arr, true).unwrap();
        let expected_s = cast_expected(&case.expected["S"]["data"]);
        assert_f32_slice_ulp(
            s.as_slice().unwrap(),
            &expected_s,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}

#[test]
fn oracle_qr_f32() {
    let suite = load_fixture(&linalg_path("qr.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f32(input) {
            continue;
        }
        let arr = to_ix2_f32(&make_f32_array_from_any(input));
        let (q, r) = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced).unwrap();
        let q_dyn =
            Array::<f32, IxDyn>::from_vec(IxDyn::new(q.shape()), q.as_slice().unwrap().to_vec())
                .unwrap();
        let r_dyn =
            Array::<f32, IxDyn>::from_vec(IxDyn::new(r.shape()), r.as_slice().unwrap().to_vec())
                .unwrap();
        let qr_product = ferray_linalg::matmul(&q_dyn, &r_dyn).unwrap();
        let expected = cast_expected(&input["data"]);
        assert_f32_slice_ulp(
            qr_product.as_slice().unwrap(),
            &expected,
            F32_TOL_ULPS.max(case.tolerance_ulps),
            &case.name,
        );
    }
}
