// ferray-polynomial: Root finding via companion matrix eigenvalues (REQ-12)
//
// Computes polynomial roots by constructing a companion matrix and finding
// its eigenvalues. The high-degree path delegates to ferray-linalg::eigvals
// (faer-backed) for the eigendecomposition. Newton's method then polishes
// each eigenvalue to full f64 precision (#479).

use ferray_core::error::FerrayError;
use ferray_core::{Array, Ix2};
use num_complex::Complex;

use crate::companion::companion_matrix;

/// Find roots of a polynomial given its power basis coefficients.
///
/// Coefficients are in ascending order: `coeffs[i]` is the coefficient of x^i.
/// Returns all roots (including complex ones) as `Complex<f64>` values.
///
/// For degree-0 polynomials (constants), returns an empty vector.
/// For degree-1 (linear), solves directly.
/// For degree-2 (quadratic), uses the quadratic formula.
/// For higher degrees, uses companion matrix eigenvalues via QR iteration.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if coefficients are empty or
/// the leading coefficient is zero.
pub fn find_roots_from_power_coeffs(coeffs: &[f64]) -> Result<Vec<Complex<f64>>, FerrayError> {
    if coeffs.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot find roots of empty polynomial",
        ));
    }

    // NaN/Inf coefficients would silently corrupt the companion-matrix
    // eigendecomposition (and the quadratic formula path) into producing
    // NaN roots, #252.
    for (i, &c) in coeffs.iter().enumerate() {
        if !c.is_finite() {
            return Err(FerrayError::invalid_value(format!(
                "polynomial coefficient at index {i} is not finite: {c}"
            )));
        }
    }

    // Trim trailing near-zero coefficients
    let mut n = coeffs.len();
    while n > 1 && coeffs[n - 1].abs() < f64::EPSILON * 100.0 {
        n -= 1;
    }

    let deg = n - 1;

    match deg {
        0 => Ok(Vec::new()),
        1 => {
            // c[0] + c[1]*x = 0 => x = -c[0]/c[1]
            let root = -coeffs[0] / coeffs[1];
            Ok(vec![Complex::new(root, 0.0)])
        }
        2 => {
            // Numerically stable quadratic formula: c[0] + c[1]*x + c[2]*x^2 = 0
            // Uses q = -(b + sign(b)*sqrt(disc))/2 to avoid catastrophic cancellation.
            let a = coeffs[2];
            let b = coeffs[1];
            let c = coeffs[0];
            let disc = b.mul_add(b, -(4.0 * a * c));
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let sign_b = if b >= 0.0 { 1.0 } else { -1.0 };
                let q = -0.5 * (b + sign_b * sqrt_disc);
                let r1 = q / a;
                let r2 = c / q;
                Ok(vec![Complex::new(r1, 0.0), Complex::new(r2, 0.0)])
            } else {
                let sqrt_disc = (-disc).sqrt();
                let re = -b / (2.0 * a);
                let im = sqrt_disc / (2.0 * a);
                Ok(vec![Complex::new(re, im), Complex::new(re, -im)])
            }
        }
        _ => {
            // Companion matrix eigenvalues via ferray-linalg (faer LU
            // + Schur). Faer's eigvals is more numerically robust than
            // the in-tree QR for ill-conditioned companion matrices.
            // Newton polishing afterwards still drives roots to full
            // f64 precision.
            let mat_vec = companion_matrix(&coeffs[..n])?;
            let mat_arr = Array::<f64, Ix2>::from_vec(Ix2::new([deg, deg]), mat_vec)?;
            let eig_arr = ferray_linalg::eigvals(&mat_arr)?;
            let mut eigenvalues: Vec<Complex<f64>> = eig_arr.iter().copied().collect();
            for root in &mut eigenvalues {
                newton_polish(&coeffs[..n], root);
            }
            Ok(eigenvalues)
        }
    }
}

/// Refine a root estimate using Newton's method on the polynomial.
///
/// Performs up to 8 iterations of z ← z − p(z)/p'(z), stopping early when
/// the correction is below machine epsilon. For roots with negligible
/// imaginary part, uses real arithmetic to avoid complex rounding noise.
fn newton_polish(coeffs: &[f64], z: &mut Complex<f64>) {
    let n = coeffs.len(); // n = degree + 1

    // Evaluate |p(z)| to track whether Newton is improving
    let eval_poly = |z: &Complex<f64>| -> f64 {
        let mut p = Complex::new(coeffs[n - 1], 0.0);
        for i in (0..n - 1).rev() {
            p = p * *z + Complex::new(coeffs[i], 0.0);
        }
        p.norm()
    };

    // For nearly-real roots, polish in real arithmetic for better precision.
    // Use a generous threshold: QR can leave imaginary residuals up to ~1e-12
    // for real roots of ill-conditioned companion matrices.
    if z.im.abs() <= 1e-10 * z.re.abs().max(1.0) {
        let mut x = z.re;
        let mut best_x = x;
        let mut best_res = eval_poly(&Complex::new(x, 0.0));
        for _ in 0..8 {
            let mut p = coeffs[n - 1];
            let mut dp = 0.0;
            for i in (0..n - 1).rev() {
                dp = dp * x + p;
                p = p.mul_add(x, coeffs[i]);
            }
            if dp.abs() < f64::EPSILON * 1e-100 {
                break;
            }
            let correction = p / dp;
            let candidate = x - correction;
            let res = eval_poly(&Complex::new(candidate, 0.0));
            if res < best_res {
                best_x = candidate;
                best_res = res;
            }
            x = candidate;
            if correction.abs() <= f64::EPSILON * x.abs() * 2.0 {
                break;
            }
        }
        z.re = best_x;
        z.im = 0.0;
        return;
    }

    // Complex Newton for genuinely complex roots
    let mut best_z = *z;
    let mut best_res = eval_poly(z);
    for _ in 0..8 {
        let mut p = Complex::new(coeffs[n - 1], 0.0);
        let mut dp = Complex::new(0.0, 0.0);
        for i in (0..n - 1).rev() {
            dp = dp * *z + p;
            p = p * *z + Complex::new(coeffs[i], 0.0);
        }
        let dp_norm = dp.norm();
        if dp_norm < f64::EPSILON * 1e-100 {
            break;
        }
        let correction = p / dp;
        *z -= correction;
        let res = eval_poly(z);
        if res < best_res {
            best_z = *z;
            best_res = res;
        }
        if correction.norm() <= f64::EPSILON * z.norm() * 2.0 {
            break;
        }
    }
    *z = best_z;
}


#[cfg(test)]
mod tests {
    use super::*;

    fn sort_roots(roots: &mut [Complex<f64>]) {
        roots.sort_by(|a, b| {
            a.re.partial_cmp(&b.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
        });
    }

    #[test]
    fn roots_linear() {
        // 2 + x = 0 => x = -2
        let roots = find_roots_from_power_coeffs(&[2.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0].re - (-2.0)).abs() < 1e-12);
        assert!(roots[0].im.abs() < 1e-12);
    }

    #[test]
    fn roots_quadratic_real() {
        // AC-1: x^2 - 3x + 2 = (x-1)(x-2), coefficients [2, -3, 1]
        let mut roots = find_roots_from_power_coeffs(&[2.0, -3.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 2);
        sort_roots(&mut roots);
        assert!(
            (roots[0].re - 1.0).abs() < 1e-10,
            "root[0] = {:?}",
            roots[0]
        );
        assert!(
            (roots[1].re - 2.0).abs() < 1e-10,
            "root[1] = {:?}",
            roots[1]
        );
        assert!(roots[0].im.abs() < 1e-10);
        assert!(roots[1].im.abs() < 1e-10);
    }

    #[test]
    fn roots_quadratic_complex() {
        // x^2 + 1 = 0 => x = +/- i
        let mut roots = find_roots_from_power_coeffs(&[1.0, 0.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 2);
        sort_roots(&mut roots);
        assert!(roots[0].re.abs() < 1e-10);
        assert!((roots[0].im.abs() - 1.0).abs() < 1e-10);
        assert!(roots[1].re.abs() < 1e-10);
        assert!((roots[1].im.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn roots_cubic() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        // coefficients [-6, 11, -6, 1]
        let mut roots = find_roots_from_power_coeffs(&[-6.0, 11.0, -6.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 3);
        sort_roots(&mut roots);
        assert!((roots[0].re - 1.0).abs() < 1e-8, "root[0] = {:?}", roots[0]);
        assert!((roots[1].re - 2.0).abs() < 1e-8, "root[1] = {:?}", roots[1]);
        assert!((roots[2].re - 3.0).abs() < 1e-8, "root[2] = {:?}", roots[2]);
    }

    #[test]
    fn roots_quartic() {
        // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
        // coefficients [24, -50, 35, -10, 1]
        let mut roots = find_roots_from_power_coeffs(&[24.0, -50.0, 35.0, -10.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 4);
        sort_roots(&mut roots);
        for (i, &expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!(
                (roots[i].re - expected).abs() < 1e-6,
                "root[{}] = {:?}, expected {}",
                i,
                roots[i],
                expected
            );
            assert!(roots[i].im.abs() < 1e-6);
        }
    }

    #[test]
    fn roots_constant() {
        let roots = find_roots_from_power_coeffs(&[5.0]).unwrap();
        assert!(roots.is_empty());
    }

    #[test]
    fn roots_empty_err() {
        assert!(find_roots_from_power_coeffs(&[]).is_err());
    }

    #[test]
    fn roots_degree_5_known() {
        // (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120
        let coeffs = [-120.0, 274.0, -225.0, 85.0, -15.0, 1.0];
        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 5);
        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, &r) in real_roots.iter().enumerate() {
            assert!(
                (r - (i + 1) as f64).abs() < 1e-6,
                "root {i}: expected {}, got {r}",
                i + 1
            );
        }
    }

    #[test]
    fn roots_degree_10() {
        // (x-1)(x-2)...(x-10) — all roots are 1..10
        // Build coefficients by expanding the product
        let mut coeffs = vec![1.0_f64]; // start with constant polynomial "1"
        for k in 1..=10 {
            // Multiply by (x - k): new[i] = old[i-1] - k * old[i]
            let mut new_coeffs = vec![0.0; coeffs.len() + 1];
            for (i, &c) in coeffs.iter().enumerate() {
                new_coeffs[i + 1] += c; // x term
                new_coeffs[i] -= k as f64 * c; // -k term
            }
            coeffs = new_coeffs;
        }

        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 10);

        // All roots should be real and close to 1..10
        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, &r) in real_roots.iter().enumerate() {
            let expected = (i + 1) as f64;
            assert!(
                (r - expected).abs() < 0.01,
                "degree-10 root {i}: expected {expected}, got {r}"
            );
        }
        // Imaginary parts should be near zero
        for (i, root) in roots.iter().enumerate() {
            assert!(
                root.im.abs() < 0.01,
                "degree-10 root {i} has imaginary part {}",
                root.im
            );
        }
    }

    #[test]
    fn roots_degree_6_with_real_roots() {
        // (x-1)(x+1)(x-2)(x+2)(x-3)(x+3) = (x^2-1)(x^2-4)(x^2-9)
        // = x^6 - 14x^4 + 49x^2 - 36
        let coeffs = [-36.0, 0.0, 49.0, 0.0, -14.0, 0.0, 1.0];
        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 6);
        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        for (i, (&r, &e)) in real_roots.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "root {i}: expected {e}, got {r}");
        }
    }

    // ----- Repeated-root coverage (#250) ---------------------------------
    //
    // Companion-matrix eigendecomposition is numerically ill-conditioned
    // for polynomials with repeated roots — the LAPACK output can spread
    // a multiplicity-k root across a cluster of size ~eps^(1/k). Tests
    // here use bands wide enough to absorb that spread (~1e-3 for double
    // roots, ~1e-2 for triple roots) while still pinning the cluster
    // location and overall multiplicity.

    #[test]
    fn roots_double_at_one() {
        // (x-1)^2 = x^2 - 2x + 1
        let coeffs = [1.0, -2.0, 1.0];
        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 2);
        for (i, r) in roots.iter().enumerate() {
            assert!(
                (r.re - 1.0).abs() < 1e-3,
                "root {i} re = {} far from 1.0",
                r.re
            );
            assert!(
                r.im.abs() < 1e-3,
                "root {i} im = {} expected ~0",
                r.im
            );
        }
    }

    #[test]
    fn roots_triple_at_one() {
        // (x-1)^3 = x^3 - 3x^2 + 3x - 1
        let coeffs = [-1.0, 3.0, -3.0, 1.0];
        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 3);
        // Triple root: spread ~ eps^(1/3) ~ 6e-6, but each root's
        // distance to 1.0 (in 2-D, |re-1| + |im|) should still be small.
        for (i, r) in roots.iter().enumerate() {
            let dist = ((r.re - 1.0).powi(2) + r.im.powi(2)).sqrt();
            assert!(
                dist < 1e-2,
                "root {i} = ({}, {}) too far from cluster at 1.0 (dist = {dist})",
                r.re,
                r.im
            );
        }
        // The centroid of the three roots should match the algebraic
        // average — for a triple root at 1.0, the sum is 3.0. Tolerance
        // accommodates the eps^(1/3) ~ 6e-6 spread of the cluster.
        let re_sum: f64 = roots.iter().map(|r| r.re).sum();
        assert!(
            (re_sum - 3.0).abs() < 1e-4,
            "sum of root real parts = {re_sum}, expected 3.0"
        );
    }

    #[test]
    fn roots_double_plus_simple() {
        // (x-2)^2 (x+1) = (x^2 - 4x + 4)(x + 1) = x^3 - 3x^2 + 0*x + 4
        let coeffs = [4.0, 0.0, -3.0, 1.0];
        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 3);
        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // After sorting: [-1, 2, 2] with the simple root well-separated.
        assert!(
            (real_roots[0] - (-1.0)).abs() < 1e-6,
            "simple root: got {} expected -1.0",
            real_roots[0]
        );
        for r in &real_roots[1..] {
            assert!(
                (r - 2.0).abs() < 1e-3,
                "double root: got {r} expected 2.0"
            );
        }
        // Imaginary parts should all be small.
        for r in &roots {
            assert!(r.im.abs() < 1e-3);
        }
    }

    #[test]
    fn roots_quadruple_at_origin() {
        // x^4 = (x-0)^4 — quadruple root at 0.
        let coeffs = [0.0, 0.0, 0.0, 0.0, 1.0];
        let roots = find_roots_from_power_coeffs(&coeffs).unwrap();
        assert_eq!(roots.len(), 4);
        // Quadruple root spread ~ eps^(1/4) ~ 1e-4. Centroid is 0.
        let centroid_re: f64 = roots.iter().map(|r| r.re).sum::<f64>() / 4.0;
        let centroid_im: f64 = roots.iter().map(|r| r.im).sum::<f64>() / 4.0;
        assert!(centroid_re.abs() < 1e-9);
        assert!(centroid_im.abs() < 1e-9);
        for r in &roots {
            assert!(r.norm() < 1e-3);
        }
    }

    #[test]
    fn roots_repeated_complex_pair() {
        // (x^2 + 1)^2 = x^4 + 2x^2 + 1, repeated roots at +i and -i.
        // Repeated complex roots through a real-coefficient polynomial
        // make the companion matrix's QR iteration ill-conditioned —
        // it can either converge with a wider cluster than usual or
        // fail to converge entirely. Accept both outcomes.
        let coeffs = [1.0, 0.0, 2.0, 0.0, 1.0];
        match find_roots_from_power_coeffs(&coeffs) {
            Ok(roots) => {
                assert_eq!(roots.len(), 4);
                for r in &roots {
                    assert!(r.re.abs() < 1e-2, "real part {} should be ~0", r.re);
                    assert!(
                        (r.im.abs() - 1.0).abs() < 1e-2,
                        "|im| = {} should be ~1.0",
                        r.im.abs()
                    );
                }
                let im_sum: f64 = roots.iter().map(|r| r.im).sum();
                assert!(im_sum.abs() < 1e-6);
            }
            Err(FerrayError::ConvergenceFailure { .. }) => {
                // Acceptable: documents the known limitation. A future
                // multi-stage root-finder (deflation + nudging on
                // repeated roots) would tighten this branch up.
            }
            Err(e) => panic!("unexpected error variant: {e}"),
        }
    }

    // ----- NaN/Inf input rejection (#252) --------------------------------

    #[test]
    fn roots_rejects_nan_coefficient() {
        let coeffs = [1.0, f64::NAN, 1.0];
        let err = find_roots_from_power_coeffs(&coeffs).unwrap_err();
        assert!(
            err.to_string().contains("not finite"),
            "expected non-finite-coefficient error, got: {err}"
        );
    }

    #[test]
    fn roots_rejects_inf_coefficient() {
        let coeffs = [1.0, 2.0, f64::INFINITY];
        let err = find_roots_from_power_coeffs(&coeffs).unwrap_err();
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn roots_rejects_neg_inf_coefficient() {
        let coeffs = [f64::NEG_INFINITY, 0.0, 1.0];
        let err = find_roots_from_power_coeffs(&coeffs).unwrap_err();
        assert!(err.to_string().contains("not finite"));
    }
}
