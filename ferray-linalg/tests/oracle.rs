/// Oracle tests: validate ferray-linalg against NumPy fixture outputs.
use ferray_core::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_test_oracle::*;

fn linalg_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("linalg").join(name)
}

// ---------------------------------------------------------------------------
// Matrix -> scalar operations
// ---------------------------------------------------------------------------

#[test]
fn oracle_det() {
    run_matrix_scalar_oracle(&linalg_path("det.json"), |a| {
        let shape = a.shape().to_vec();
        let a2 = Array::<f64, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            a.as_slice().unwrap().to_vec(),
        )
        .unwrap();
        ferray_linalg::det(&a2)
    });
}

#[test]
fn oracle_trace() {
    run_matrix_scalar_oracle(&linalg_path("trace.json"), |a| {
        let shape = a.shape().to_vec();
        let a2 = Array::<f64, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            a.as_slice().unwrap().to_vec(),
        )
        .unwrap();
        ferray_linalg::trace(&a2)
    });
}

// ---------------------------------------------------------------------------
// Helper: convert IxDyn to Ix2 for linalg functions that require Ix2
// ---------------------------------------------------------------------------

fn to_ix2(a: &Array<f64, IxDyn>) -> Array<f64, Ix2> {
    let shape = a.shape().to_vec();
    Array::<f64, Ix2>::from_vec(
        Ix2::new([shape[0], shape[1]]),
        a.as_slice().unwrap().to_vec(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Matrix -> matrix operations
// ---------------------------------------------------------------------------

#[test]
fn oracle_inv() {
    let suite = load_fixture(&linalg_path("inv.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::inv(&arr).unwrap();
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
fn oracle_cholesky() {
    let suite = load_fixture(&linalg_path("cholesky.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::cholesky(&arr).unwrap();
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
// Matrix products (IxDyn)
// ---------------------------------------------------------------------------

#[test]
fn oracle_matmul() {
    let suite = load_fixture(&linalg_path("matmul.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::matmul(&a, &b).unwrap();
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
fn oracle_dot() {
    let suite = load_fixture(&linalg_path("dot.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::dot(&a, &b).unwrap();
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
fn oracle_inner() {
    let suite = load_fixture(&linalg_path("inner.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::inner(&a, &b).unwrap();
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
fn oracle_outer() {
    let suite = load_fixture(&linalg_path("outer.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::outer(&a, &b).unwrap();
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
fn oracle_kron() {
    let suite = load_fixture(&linalg_path("kron.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::kron(&a, &b).unwrap();
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
fn oracle_vdot() {
    let suite = load_fixture(&linalg_path("vdot.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = make_f64_array(input_a);
        let b = make_f64_array(input_b);
        let result = ferray_linalg::vdot(&a, &b).unwrap();
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
// Norms
// ---------------------------------------------------------------------------

#[test]
fn oracle_norm() {
    let suite = load_fixture(&linalg_path("norm.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
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
// Solve
// ---------------------------------------------------------------------------

#[test]
fn oracle_solve() {
    let suite = load_fixture(&linalg_path("solve.json"));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        if should_skip_f64(input_a) {
            continue;
        }
        let a = to_ix2(&make_f64_array(input_a));
        let b = make_f64_array(input_b);
        let result = ferray_linalg::solve(&a, &b).unwrap();
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
// Condition number
// ---------------------------------------------------------------------------

#[test]
fn oracle_cond() {
    let suite = load_fixture(&linalg_path("cond.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let p = match case.inputs.get("ord") {
            Some(v) if v.is_string() => match v.as_str().unwrap() {
                "fro" => ferray_linalg::NormOrder::Fro,
                "inf" | "Inf" => ferray_linalg::NormOrder::Inf,
                "-inf" | "-Inf" => ferray_linalg::NormOrder::NegInf,
                other => panic!("unknown norm order: {other}"),
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
        let result = ferray_linalg::cond(&arr, p).unwrap();
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
// Matrix rank
// ---------------------------------------------------------------------------

#[test]
fn oracle_matrix_rank() {
    let suite = load_fixture(&linalg_path("matrix_rank.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let result = ferray_linalg::matrix_rank(&arr, None).unwrap();
        let expected = case.expected["data"].as_u64().unwrap() as usize;
        assert_eq!(result, expected, "case '{}': rank mismatch", case.name);
    }
}

// ---------------------------------------------------------------------------
// Decompositions (compare eigenvalues sorted by magnitude)
// ---------------------------------------------------------------------------

#[test]
fn oracle_eigh() {
    let suite = load_fixture(&linalg_path("eigh.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let (eigenvalues, _eigenvectors) = ferray_linalg::eigh(&arr).unwrap();
        // Compare eigenvalues (sorted ascending for symmetric matrices)
        let expected_w = parse_f64_data(&case.expected["eigenvalues"]["data"]);
        assert_f64_slice_ulp(
            eigenvalues.as_slice().unwrap(),
            &expected_w,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// SVD (compare singular values only — vectors may differ by sign)
// ---------------------------------------------------------------------------

#[test]
fn oracle_svd() {
    let suite = load_fixture(&linalg_path("svd.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = to_ix2(&make_f64_array(input));
        let (_u, s, _vt) = ferray_linalg::svd(&arr, true).unwrap();
        // Only compare singular values (sign ambiguity in U and Vt)
        let expected_s = parse_f64_data(&case.expected["S"]["data"]);
        assert_f64_slice_ulp(
            s.as_slice().unwrap(),
            &expected_s,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// QR (compare R only — Q may differ by sign convention)
// ---------------------------------------------------------------------------

#[test]
fn oracle_qr() {
    let suite = load_fixture(&linalg_path("qr.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["a"];
        if should_skip_f64(input) {
            continue;
        }
        let _ = parse_shape(&input["shape"]);
        let arr = to_ix2(&make_f64_array(input));
        let (q, r) = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced).unwrap();
        // Verify Q*R = A within tolerance (decomposition correctness)
        let q_dyn =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(q.shape()), q.as_slice().unwrap().to_vec())
                .unwrap();
        let r_dyn =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(r.shape()), r.as_slice().unwrap().to_vec())
                .unwrap();
        let qr_product = ferray_linalg::matmul(&q_dyn, &r_dyn).unwrap();
        let expected = parse_f64_data(&input["data"]);
        assert_f64_slice_ulp(
            qr_product.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}
