// Property-based tests for ferray-polynomial
//
// Tests mathematical invariants of polynomial operations using proptest.

use ferray_polynomial::power::Polynomial;
use ferray_polynomial::traits::Poly;

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. Evaluation at roots: p(root) ~= 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_eval_at_roots(
        c0 in -10.0f64..10.0,
        c1 in -10.0f64..10.0,
        c2 in 0.1f64..10.0,
    ) {
        // Create a polynomial c0 + c1*x + c2*x^2 (degree 2, with c2 != 0)
        let p = Polynomial::new(&[c0, c1, c2]);
        let roots = p.roots().unwrap();

        for root in &roots {
            // Evaluate at the real part if the imaginary part is small
            // For complex roots, evaluate using manual computation
            let x_re = root.re;
            let x_im = root.im;

            // p(z) = c0 + c1*z + c2*z^2 where z = x_re + i*x_im
            let z = num_complex::Complex::new(x_re, x_im);
            let val = num_complex::Complex::new(c0, 0.0)
                + num_complex::Complex::new(c1, 0.0) * z
                + num_complex::Complex::new(c2, 0.0) * z * z;

            prop_assert!(
                val.norm() < 1e-4,
                "p(root) = {:?} should be ~0, root = {:?}, coeffs = [{}, {}, {}]",
                val, root, c0, c1, c2
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. Degree of product equals sum of degrees
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mul_degree_sum(
        a_coeffs in proptest::collection::vec(-10.0f64..10.0, 2..=4),
        b_coeffs in proptest::collection::vec(-10.0f64..10.0, 2..=4),
    ) {
        // Ensure leading coefficients are nonzero
        let mut a_c = a_coeffs;
        let mut b_c = b_coeffs;
        if a_c.last().map_or(true, |&v| v.abs() < 0.01) {
            *a_c.last_mut().unwrap() = 1.0;
        }
        if b_c.last().map_or(true, |&v| v.abs() < 0.01) {
            *b_c.last_mut().unwrap() = 1.0;
        }

        let a = Polynomial::new(&a_c);
        let b = Polynomial::new(&b_c);
        let product = a.mul(&b).unwrap();

        let expected_degree = a.degree() + b.degree();
        prop_assert_eq!(
            product.degree(), expected_degree,
            "deg(a*b)={} != deg(a)+deg(b)={} for a={:?}, b={:?}",
            product.degree(), expected_degree, a.coeffs(), b.coeffs()
        );
    }

    // -----------------------------------------------------------------------
    // 3. Identity: p * 1 == p
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mul_identity(
        coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
    ) {
        let p = Polynomial::new(&coeffs);
        let one = Polynomial::new(&[1.0]);
        let result = p.mul(&one).unwrap();

        // Compare evaluations at several points
        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let p_val = p.eval(x).unwrap();
            let r_val = result.eval(x).unwrap();
            prop_assert!(
                (p_val - r_val).abs() < 1e-10,
                "p({}) = {} != (p*1)({}) = {}",
                x, p_val, x, r_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. Addition commutativity: a + b == b + a
    // -----------------------------------------------------------------------
    #[test]
    fn prop_add_commutative(
        a_coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
        b_coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
    ) {
        let a = Polynomial::new(&a_coeffs);
        let b = Polynomial::new(&b_coeffs);
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.7 - 3.0;
            let ab_val = ab.eval(x).unwrap();
            let ba_val = ba.eval(x).unwrap();
            prop_assert!(
                (ab_val - ba_val).abs() < 1e-10,
                "(a+b)({}) = {} != (b+a)({}) = {}",
                x, ab_val, x, ba_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. Subtraction: a - a == 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sub_self_is_zero(
        coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
    ) {
        let p = Polynomial::new(&coeffs);
        let zero = p.sub(&p).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.3 - 1.5;
            let val = zero.eval(x).unwrap();
            prop_assert!(
                val.abs() < 1e-10,
                "(p-p)({}) = {} should be 0",
                x, val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. Derivative of integral is identity: (integ(p))' == p
    // -----------------------------------------------------------------------
    #[test]
    fn prop_deriv_integ_roundtrip(
        coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=4),
    ) {
        let p = Polynomial::new(&coeffs);
        let integrated = p.integ(1, &[0.0]).unwrap();
        let derived = integrated.deriv(1).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let p_val = p.eval(x).unwrap();
            let d_val = derived.eval(x).unwrap();
            prop_assert!(
                (p_val - d_val).abs() < 1e-8,
                "(integ(p))'({}) = {} != p({}) = {}",
                x, d_val, x, p_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 7. Power: p^0 == 1
    // -----------------------------------------------------------------------
    #[test]
    fn prop_pow_zero_is_one(
        coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=4),
    ) {
        let p = Polynomial::new(&coeffs);
        let result = p.pow(0).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let val = result.eval(x).unwrap();
            prop_assert!(
                (val - 1.0).abs() < 1e-10,
                "p^0({}) = {} should be 1",
                x, val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 8. Power: p^1 == p
    // -----------------------------------------------------------------------
    #[test]
    fn prop_pow_one_is_identity(
        coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=4),
    ) {
        let p = Polynomial::new(&coeffs);
        let result = p.pow(1).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let p_val = p.eval(x).unwrap();
            let r_val = result.eval(x).unwrap();
            prop_assert!(
                (p_val - r_val).abs() < 1e-10,
                "p^1({}) = {} != p({}) = {}",
                x, r_val, x, p_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. Divmod: quotient * divisor + remainder == dividend
    // -----------------------------------------------------------------------
    #[test]
    fn prop_divmod_reconstruction(
        a_coeffs in proptest::collection::vec(-5.0f64..5.0, 3..=5),
        b_coeffs in proptest::collection::vec(-5.0f64..5.0, 2..=3),
    ) {
        let mut b_c = b_coeffs;
        // Ensure leading coefficient is nonzero
        if b_c.last().map_or(true, |&v| v.abs() < 0.1) {
            *b_c.last_mut().unwrap() = 1.0;
        }

        let a = Polynomial::new(&a_coeffs);
        let b = Polynomial::new(&b_c);
        let (q, r) = a.divmod(&b).unwrap();

        // Verify: q * b + r == a at several points
        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let a_val = a.eval(x).unwrap();
            let qb = q.mul(&b).unwrap();
            let qb_val = qb.eval(x).unwrap();
            let r_val = r.eval(x).unwrap();
            let reconstructed = qb_val + r_val;
            let diff = (a_val - reconstructed).abs();
            let scale = a_val.abs().max(1.0);
            prop_assert!(
                diff / scale < 1e-6,
                "q*b+r({}) = {} != a({}) = {}, diff = {}",
                x, reconstructed, x, a_val, diff
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10. Trim does not change evaluation
    // -----------------------------------------------------------------------
    #[test]
    fn prop_trim_preserves_eval(
        coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
    ) {
        let p = Polynomial::new(&coeffs);
        let trimmed = p.trim(1e-12).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let p_val = p.eval(x).unwrap();
            let t_val = trimmed.eval(x).unwrap();
            prop_assert!(
                (p_val - t_val).abs() < 1e-8,
                "trim changed eval at {}: {} vs {}",
                x, p_val, t_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 11. Multiplication commutativity: a * b == b * a
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mul_commutative(
        a_coeffs in proptest::collection::vec(-5.0f64..5.0, 1..=3),
        b_coeffs in proptest::collection::vec(-5.0f64..5.0, 1..=3),
    ) {
        let a = Polynomial::new(&a_coeffs);
        let b = Polynomial::new(&b_coeffs);
        let ab = a.mul(&b).unwrap();
        let ba = b.mul(&a).unwrap();

        for i in 0..10 {
            let x = (i as f64) * 0.5 - 2.0;
            let ab_val = ab.eval(x).unwrap();
            let ba_val = ba.eval(x).unwrap();
            prop_assert!(
                (ab_val - ba_val).abs() < 1e-8,
                "(a*b)({}) = {} != (b*a)({}) = {}",
                x, ab_val, x, ba_val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 12. Zero polynomial evaluates to zero
    // -----------------------------------------------------------------------
    #[test]
    fn prop_zero_poly_evals_zero(x in -100.0f64..100.0) {
        let zero = Polynomial::new(&[0.0]);
        let val = zero.eval(x).unwrap();
        prop_assert!(
            val.abs() < 1e-15,
            "zero poly at {} = {} should be 0",
            x, val
        );
    }

    // --- Chebyshev property tests ---

    #[test]
    fn prop_chebyshev_add_commutative(
        a_coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
        b_coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
    ) {
        use ferray_polynomial::Chebyshev;
        use ferray_polynomial::Poly;
        let a = Chebyshev::new(&a_coeffs);
        let b = Chebyshev::new(&b_coeffs);
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();
        // Evaluate at a test point
        let x = 0.5;
        let val_ab = ab.eval(x).unwrap();
        let val_ba = ba.eval(x).unwrap();
        prop_assert!(
            (val_ab - val_ba).abs() < 1e-10,
            "Chebyshev add not commutative: {} vs {}",
            val_ab, val_ba
        );
    }

    #[test]
    fn prop_chebyshev_deriv_integ_roundtrip(
        coeffs in proptest::collection::vec(-5.0f64..5.0, 2..=6),
    ) {
        use ferray_polynomial::Chebyshev;
        use ferray_polynomial::Poly;
        let p = Chebyshev::new(&coeffs);
        let dp = p.deriv(1).unwrap();
        let idp = dp.integ(1, &[0.0]).unwrap();
        // integ(deriv(p)) should equal p minus constant term (up to constant)
        // Check at x=0.5
        let x = 0.5;
        let orig = p.eval(x).unwrap();
        let roundtrip = idp.eval(x).unwrap();
        // Difference should be a constant (the lost constant of integration)
        let orig0 = p.eval(0.0).unwrap();
        let roundtrip0 = idp.eval(0.0).unwrap();
        let const_diff = orig0 - roundtrip0;
        prop_assert!(
            ((orig - roundtrip) - const_diff).abs() < 1e-6,
            "Chebyshev deriv/integ roundtrip failed at x={}: {} vs {} (const={})",
            x, orig, roundtrip, const_diff
        );
    }

    // --- Legendre property tests ---

    #[test]
    fn prop_legendre_add_commutative(
        a_coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
        b_coeffs in proptest::collection::vec(-10.0f64..10.0, 1..=5),
    ) {
        use ferray_polynomial::Legendre;
        use ferray_polynomial::Poly;
        let a = Legendre::new(&a_coeffs);
        let b = Legendre::new(&b_coeffs);
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();
        let x = 0.3;
        let val_ab = ab.eval(x).unwrap();
        let val_ba = ba.eval(x).unwrap();
        prop_assert!(
            (val_ab - val_ba).abs() < 1e-10,
            "Legendre add not commutative: {} vs {}",
            val_ab, val_ba
        );
    }

    // --- Basis conversion property tests ---

    #[test]
    fn prop_power_chebyshev_roundtrip(
        coeffs in proptest::collection::vec(-5.0f64..5.0, 1..=4),
    ) {
        use ferray_polynomial::{Polynomial, Chebyshev, Poly};
        use ferray_polynomial::{FromPowerBasis, ToPowerBasis};
        let p = Polynomial::new(&coeffs);
        let cheb = Chebyshev::from_power_basis(p.coeffs()).unwrap();
        let back_coeffs = cheb.to_power_basis().unwrap();
        let back = Polynomial::new(&back_coeffs);
        // Evaluate both at a test point
        let x = 0.7;
        let orig = p.eval(x).unwrap();
        let roundtrip = back.eval(x).unwrap();
        prop_assert!(
            (orig - roundtrip).abs() < 1e-6,
            "Power->Chebyshev->Power roundtrip failed: {} vs {}",
            orig, roundtrip
        );
    }
}
