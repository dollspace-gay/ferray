// Property-based tests for ferray-linalg
//
// Tests mathematical invariants of linear algebra operations using proptest.

// Property tests sample integer sizes and assert exact float equality on
// matrix-identity / decomp invariants by design.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use ferray_core::Array;
use ferray_core::dimension::{Ix2, IxDyn};

use ferray_linalg::decomp::{QrMode, qr, svd};
use ferray_linalg::norms::{NormOrder, det, norm, trace};
use ferray_linalg::products::dot;
use ferray_linalg::solve::{inv, solve};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mat2(rows: usize, cols: usize, data: Vec<f64>) -> Array<f64, Ix2> {
    Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

fn matd(shape: &[usize], data: Vec<f64>) -> Array<f64, IxDyn> {
    Array::<f64, IxDyn>::from_vec(IxDyn::new(shape), data).unwrap()
}

/// Simple matrix multiplication for verification.
fn naive_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

/// Generate a well-conditioned square matrix by making it diagonally dominant.
fn diag_dominant_matrix(n: usize, data: Vec<f64>) -> Vec<f64> {
    let mut result = data;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j {
                row_sum += result[i * n + j].abs();
            }
        }
        result[i * n + i] = result[i * n + i].abs() + row_sum + 1.0;
    }
    result
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. QR reconstruction: A ~= Q * R
    // -----------------------------------------------------------------------
    #[test]
    fn prop_qr_reconstructs(
        data in proptest::collection::vec(-10.0f64..10.0, 9),
    ) {
        let n = 3;
        let a = mat2(n, n, data);
        let (q, r) = qr(&a, QrMode::Reduced).unwrap();

        let qs = q.as_slice().unwrap();
        let rs = r.as_slice().unwrap();
        let k = q.shape()[1];
        let reconstructed = naive_matmul(qs, rs, n, k, n);

        let orig = a.as_slice().unwrap();
        for i in 0..(n * n) {
            prop_assert!(
                (reconstructed[i] - orig[i]).abs() < 1e-8,
                "Q*R[{}] = {} != {}",
                i, reconstructed[i], orig[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. QR orthogonality: Qt * Q ~= I
    // -----------------------------------------------------------------------
    #[test]
    fn prop_qr_orthogonal(
        data in proptest::collection::vec(-10.0f64..10.0, 9),
    ) {
        let n = 3;
        let a = mat2(n, n, data);
        let (q, _r) = qr(&a, QrMode::Complete).unwrap();
        let qs = q.as_slice().unwrap();

        for i in 0..n {
            for j in 0..n {
                let mut dot_val = 0.0;
                for k in 0..n {
                    dot_val += qs[k * n + i] * qs[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!(
                    (dot_val - expected).abs() < 1e-8,
                    "Qt*Q[{},{}] = {} != {}",
                    i, j, dot_val, expected
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. SVD reconstruction: A ~= U * diag(S) * Vt
    // -----------------------------------------------------------------------
    #[test]
    fn prop_svd_reconstructs(
        data in proptest::collection::vec(-10.0f64..10.0, 6),
    ) {
        let a = mat2(3, 2, data);
        let (u, s, vt) = svd(&a, false).unwrap();

        let us = u.as_slice().unwrap();
        let ss = s.as_slice().unwrap();
        let vts = vt.as_slice().unwrap();
        let (m, n) = (3, 2);
        let k = ss.len();

        let orig = a.as_slice().unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for p in 0..k {
                    val += us[i * k + p] * ss[p] * vts[p * n + j];
                }
                prop_assert!(
                    (val - orig[i * n + j]).abs() < 1e-8,
                    "USVt[{},{}] = {} != {}",
                    i, j, val, orig[i * n + j]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4. SVD singular values are non-negative
    // -----------------------------------------------------------------------
    #[test]
    fn prop_svd_singular_values_nonneg(
        data in proptest::collection::vec(-10.0f64..10.0, 9),
    ) {
        let a = mat2(3, 3, data);
        let (_u, s, _vt) = svd(&a, false).unwrap();
        for &val in s.as_slice().unwrap() {
            prop_assert!(val >= -1e-14, "singular value {} should be >= 0", val);
        }
    }

    // -----------------------------------------------------------------------
    // 5. Inverse: A * inv(A) ~= I for invertible A
    // -----------------------------------------------------------------------
    #[test]
    fn prop_inverse_identity(
        data in proptest::collection::vec(-5.0f64..5.0, 9),
    ) {
        let n = 3;
        let dd = diag_dominant_matrix(n, data);
        let a = mat2(n, n, dd);
        let a_inv = inv(&a).unwrap();

        let a_data = a.as_slice().unwrap();
        let inv_data = a_inv.as_slice().unwrap();
        let product = naive_matmul(a_data, inv_data, n, n, n);

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!(
                    (product[i * n + j] - expected).abs() < 1e-6,
                    "A*inv(A)[{},{}] = {} != {}",
                    i, j, product[i * n + j], expected
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 6. Trace linearity: trace(A + B) == trace(A) + trace(B)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_trace_linear(
        a_data in proptest::collection::vec(-10.0f64..10.0, 9),
        b_data in proptest::collection::vec(-10.0f64..10.0, 9),
    ) {
        let n = 3;
        let a = mat2(n, n, a_data.clone());
        let b = mat2(n, n, b_data.clone());

        let sum_data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
        let ab = mat2(n, n, sum_data);

        let ta = trace(&a).unwrap();
        let tb = trace(&b).unwrap();
        let tab = trace(&ab).unwrap();

        prop_assert!(
            (tab - (ta + tb)).abs() < 1e-10,
            "trace(A+B)={} != trace(A)+trace(B)={}",
            tab, ta + tb
        );
    }

    // -----------------------------------------------------------------------
    // 7. Norm homogeneity: norm(c * A) == |c| * norm(A) for Frobenius
    // -----------------------------------------------------------------------
    #[test]
    fn prop_norm_homogeneous(
        data in proptest::collection::vec(-10.0f64..10.0, 9),
        c in -10.0f64..10.0,
    ) {
        let scaled: Vec<f64> = data.iter().map(|&x| x * c).collect();
        let a = matd(&[3, 3], data);
        let ca = matd(&[3, 3], scaled);

        let norm_a = norm(&a, NormOrder::Fro).unwrap();
        let norm_ca = norm(&ca, NormOrder::Fro).unwrap();

        let expected = c.abs() * norm_a;
        prop_assert!(
            (norm_ca - expected).abs() < 1e-8,
            "norm(c*A)={} != |c|*norm(A)={}",
            norm_ca, expected
        );
    }

    // -----------------------------------------------------------------------
    // 8. Solve: if x = solve(A, b), then A * x ~= b
    // -----------------------------------------------------------------------
    #[test]
    fn prop_solve_reconstructs(
        a_raw in proptest::collection::vec(-5.0f64..5.0, 9),
        b_data in proptest::collection::vec(-10.0f64..10.0, 3),
    ) {
        let n = 3;
        let dd = diag_dominant_matrix(n, a_raw);
        let a = mat2(n, n, dd.clone());
        let b = matd(&[n], b_data.clone());

        let x = solve(&a, &b).unwrap();
        let x_data: Vec<f64> = x.iter().copied().collect();

        for i in 0..n {
            let mut val = 0.0;
            for j in 0..n {
                val += dd[i * n + j] * x_data[j];
            }
            prop_assert!(
                (val - b_data[i]).abs() < 1e-6,
                "Ax[{}] = {} != b[{}] = {}",
                i, val, i, b_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. Determinant scaling: det(cA) = c^n * det(A) for nxn matrix
    // -----------------------------------------------------------------------
    #[test]
    fn prop_det_scaling(
        data in proptest::collection::vec(-5.0f64..5.0, 4),
        c in 0.1f64..5.0,
    ) {
        let n = 2;
        let a = mat2(n, n, data.clone());
        let scaled: Vec<f64> = data.iter().map(|&x| x * c).collect();
        let ca = mat2(n, n, scaled);

        let det_a = det(&a).unwrap();
        let det_ca = det(&ca).unwrap();

        let expected = c.powi(n as i32) * det_a;
        prop_assert!(
            (det_ca - expected).abs() < 1e-6 * expected.abs().max(1.0),
            "det(cA)={} != c^n*det(A)={}",
            det_ca, expected
        );
    }

    // -----------------------------------------------------------------------
    // 10. Norm non-negativity
    // -----------------------------------------------------------------------
    #[test]
    fn prop_norm_nonneg(
        data in proptest::collection::vec(-10.0f64..10.0, 9),
    ) {
        let a = matd(&[3, 3], data);
        let n = norm(&a, NormOrder::Fro).unwrap();
        prop_assert!(n >= 0.0, "norm should be non-negative, got {}", n);
    }

    // -----------------------------------------------------------------------
    // 11. Trace equals sum of diagonal elements
    // -----------------------------------------------------------------------
    #[test]
    fn prop_trace_is_diag_sum(
        data in proptest::collection::vec(-10.0f64..10.0, 9),
    ) {
        let n = 3;
        let a = mat2(n, n, data.clone());
        let t = trace(&a).unwrap();
        let diag_sum: f64 = (0..n).map(|i| data[i * n + i]).sum();
        prop_assert!(
            (t - diag_sum).abs() < 1e-10,
            "trace={} != diag_sum={}",
            t, diag_sum
        );
    }

    // -----------------------------------------------------------------------
    // 12. dot product commutativity for 1-D vectors
    // -----------------------------------------------------------------------
    #[test]
    fn prop_dot_commutative(
        a_data in proptest::collection::vec(-10.0f64..10.0, 5),
        b_data in proptest::collection::vec(-10.0f64..10.0, 5),
    ) {
        let a = matd(&[5], a_data);
        let b = matd(&[5], b_data);

        let ab = dot(&a, &b).unwrap();
        let ba = dot(&b, &a).unwrap();

        let ab_val: Vec<f64> = ab.iter().copied().collect();
        let ba_val: Vec<f64> = ba.iter().copied().collect();

        prop_assert!(
            (ab_val[0] - ba_val[0]).abs() < 1e-8,
            "dot(a,b)={} != dot(b,a)={}",
            ab_val[0], ba_val[0]
        );
    }
}
