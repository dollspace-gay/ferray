//! Oracle tests: validate ferray-polynomial against `NumPy` fixture outputs.

// Oracle tests cross JSON fixture indices through `usize` and ULP-compare
// against NumPy reference outputs — the same dtype roundtrip contract as
// the other oracle suites.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_polynomial::{Poly, Polynomial};
use ferray_test_oracle::*;

fn poly_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("polynomial").join(name)
}

#[test]
fn oracle_polyval() {
    let suite = load_fixture(&poly_path("polyval.json"));
    for case in &suite.test_cases {
        // Fixture provides coefficients in ascending order (ferray convention)
        let coeffs = parse_f64_data(&case.inputs["coefficients"]["data"]);
        let x_data = parse_f64_data(&case.inputs["x"]["data"]);
        let expected = parse_f64_data(&case.expected["data"]);

        let poly = Polynomial::new(&coeffs);
        let result = poly.eval_many(&x_data).unwrap();

        assert_eq!(
            result.len(),
            expected.len(),
            "case '{}': length mismatch",
            case.name
        );
        for (i, (&a, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert_f64_ulp(
                a,
                e,
                case.tolerance_ulps
                    .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
                &format!("case '{}' [element {i}]", case.name),
            );
        }
    }
}

#[test]
fn oracle_polyfit() {
    let suite = load_fixture(&poly_path("polyfit.json"));
    for case in &suite.test_cases {
        let x_data = parse_f64_data(&case.inputs["x"]["data"]);
        let y_data = parse_f64_data(&case.inputs["y"]["data"]);
        let deg = case.inputs["degree"].as_u64().unwrap() as usize;

        let poly = Polynomial::fit(&x_data, &y_data, deg).unwrap();

        // Fixture stores coefficients in NumPy (descending) order; ferray uses ascending.
        // Reverse the expected coefficients for comparison.
        let mut expected = parse_f64_data(&case.expected["coefficients_numpy_order"]["data"]);
        expected.reverse();
        let result = poly.coeffs();
        assert_eq!(
            result.len(),
            expected.len(),
            "case '{}': coefficient count mismatch",
            case.name
        );
        for (i, (&a, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert_f64_ulp(
                a,
                e,
                case.tolerance_ulps
                    .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
                &format!("case '{}' [coeff {i}]", case.name),
            );
        }
    }
}

#[test]
fn oracle_roots() {
    let suite = load_fixture(&poly_path("roots.json"));
    for case in &suite.test_cases {
        // Fixture stores coefficients in NumPy (descending) order; reverse for ferray.
        let mut coeffs = parse_f64_data(&case.inputs["coefficients_numpy_order"]["data"]);
        coeffs.reverse();
        let poly = Polynomial::new(&coeffs);
        let mut result = poly.roots().unwrap();

        // Two fixture formats: "roots_real_sorted" (real only) or "roots" (complex)
        if let Some(real_sorted) = case.expected.get("roots_real_sorted") {
            // Real roots — compare real parts, imaginary should be ~0
            let expected = parse_f64_data(&real_sorted["data"]);
            let mut real_parts: Vec<f64> = result.iter().map(|c| c.re).collect();
            real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(
                real_parts.len(),
                expected.len(),
                "case '{}': root count mismatch",
                case.name
            );
            for (i, (&a, &e)) in real_parts.iter().zip(expected.iter()).enumerate() {
                assert_f64_ulp(
                    a,
                    e,
                    case.tolerance_ulps
                        .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
                    &format!("case '{}' [root {i}]", case.name),
                );
            }
        } else {
            // Complex roots
            let mut expected = parse_complex_data(&case.expected["roots"]["data"]);

            // Sort both by real part, then imaginary for stable comparison
            result.sort_by(|a, b| {
                a.re.partial_cmp(&b.re)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
            });
            expected.sort_by(|a, b| {
                a.re.partial_cmp(&b.re)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
            });

            assert_eq!(
                result.len(),
                expected.len(),
                "case '{}': root count mismatch",
                case.name
            );
            for (i, (a, e)) in result.iter().zip(expected.iter()).enumerate() {
                assert_f64_ulp(
                    a.re,
                    e.re,
                    case.tolerance_ulps
                        .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
                    &format!("case '{}' [root {i}].re", case.name),
                );
                assert_f64_ulp(
                    a.im,
                    e.im,
                    case.tolerance_ulps
                        .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
                    &format!("case '{}' [root {i}].im", case.name),
                );
            }
        }
    }
}
