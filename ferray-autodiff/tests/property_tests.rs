use ferray_autodiff::{DualNumber, derivative, gradient};
use proptest::prelude::*;

const TOL: f64 = 1e-8;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // 1. d/dx sin(x) = cos(x)
    #[test]
    fn prop_derivative_sin_is_cos(x in -10.0..10.0_f64) {
        let d = derivative(|x| x.sin(), x);
        let expected = x.cos();
        prop_assert!(
            approx_eq(d, expected),
            "derivative(sin, {}) = {}, expected cos({}) = {}",
            x, d, x, expected
        );
    }

    // 2. d/dx cos(x) = -sin(x)
    #[test]
    fn prop_derivative_cos_is_neg_sin(x in -10.0..10.0_f64) {
        let d = derivative(|x| x.cos(), x);
        let expected = -x.sin();
        prop_assert!(
            approx_eq(d, expected),
            "derivative(cos, {}) = {}, expected -sin({}) = {}",
            x, d, x, expected
        );
    }

    // 3. d/dx exp(x) = exp(x)
    #[test]
    fn prop_derivative_exp_is_exp(x in -5.0..5.0_f64) {
        let d = derivative(|x| x.exp(), x);
        let expected = x.exp();
        prop_assert!(
            approx_eq(d, expected),
            "derivative(exp, {}) = {}, expected exp({}) = {}",
            x, d, x, expected
        );
    }

    // 4. d/dx ln(x) = 1/x
    #[test]
    fn prop_derivative_ln_is_recip(x in 0.1..100.0_f64) {
        let d = derivative(|x| x.ln(), x);
        let expected = 1.0 / x;
        prop_assert!(
            approx_eq(d, expected),
            "derivative(ln, {}) = {}, expected 1/{} = {}",
            x, d, x, expected
        );
    }

    // 5. d/dx sqrt(x) = 1/(2*sqrt(x))
    #[test]
    fn prop_derivative_sqrt_formula(x in 0.1..100.0_f64) {
        let d = derivative(|x| x.sqrt(), x);
        let expected = 1.0 / (2.0 * x.sqrt());
        prop_assert!(
            approx_eq(d, expected),
            "derivative(sqrt, {}) = {}, expected 1/(2*sqrt({})) = {}",
            x, d, x, expected
        );
    }

    // 6. d/dx constant = 0
    #[test]
    fn prop_derivative_constant_is_zero(x in -10.0..10.0_f64) {
        let d = derivative(|_x| DualNumber::constant(5.0), x);
        prop_assert!(
            d == 0.0,
            "derivative(const 5.0, {}) = {}, expected 0.0",
            x, d
        );
    }

    // 7. Chain rule: d/dx exp(sin(x)) = cos(x) * exp(sin(x))
    #[test]
    fn prop_chain_rule(x in -10.0..10.0_f64) {
        let d = derivative(|x| x.sin().exp(), x);
        let expected = x.cos() * x.sin().exp();
        prop_assert!(
            approx_eq(d, expected),
            "derivative(exp(sin(x)), {}) = {}, expected cos({})*exp(sin({})) = {}",
            x, d, x, x, expected
        );
    }

    // 8. Product rule / power rule: d/dx x^2 = 2x
    #[test]
    fn prop_product_rule(x in -10.0..10.0_f64) {
        let d = derivative(|x| x * x, x);
        let expected = 2.0 * x;
        prop_assert!(
            approx_eq(d, expected),
            "derivative(x*x, {}) = {}, expected 2*{} = {}",
            x, d, x, expected
        );
    }

    // 9. Power rule: d/dx x^3 = 3*x^2
    #[test]
    fn prop_power_rule(x in -10.0..10.0_f64) {
        let d = derivative(|x| x.powi(3), x);
        let expected = 3.0 * x * x;
        prop_assert!(
            approx_eq(d, expected),
            "derivative(x^3, {}) = {}, expected 3*{}^2 = {}",
            x, d, x, expected
        );
    }

    // 10. Quotient rule: d/dx (1/x) = -1/x^2
    #[test]
    fn prop_quotient_rule(x in 0.1..100.0_f64) {
        let d = derivative(|x| x.recip(), x);
        let expected = -1.0 / (x * x);
        prop_assert!(
            approx_eq(d, expected),
            "derivative(1/x, {}) = {}, expected -1/{}^2 = {}",
            x, d, x, expected
        );
    }

    // 11. Addition is commutative for DualNumbers
    #[test]
    fn prop_add_commutative(
        ar in -100.0..100.0_f64,
        ad in -100.0..100.0_f64,
        br in -100.0..100.0_f64,
        bd in -100.0..100.0_f64,
    ) {
        let a = DualNumber::new(ar, ad);
        let b = DualNumber::new(br, bd);
        let ab = a + b;
        let ba = b + a;
        prop_assert_eq!(ab.real, ba.real, "add commutative real: {} + {} != {} + {}", ar, br, br, ar);
        prop_assert_eq!(ab.dual, ba.dual, "add commutative dual: {} + {} != {} + {}", ad, bd, bd, ad);
    }

    // 12. Multiplication is commutative for DualNumbers
    #[test]
    fn prop_mul_commutative(
        ar in -100.0..100.0_f64,
        ad in -100.0..100.0_f64,
        br in -100.0..100.0_f64,
        bd in -100.0..100.0_f64,
    ) {
        let a = DualNumber::new(ar, ad);
        let b = DualNumber::new(br, bd);
        let ab = a * b;
        let ba = b * a;
        prop_assert!(
            approx_eq(ab.real, ba.real),
            "mul commutative real: ({} * {}) = {} != ({} * {}) = {}",
            ar, br, ab.real, br, ar, ba.real
        );
        prop_assert!(
            approx_eq(ab.dual, ba.dual),
            "mul commutative dual: {} != {}",
            ab.dual, ba.dual
        );
    }

    // 13. Gradient of f(x,y) = 2x + 3y is [2, 3] everywhere
    #[test]
    fn prop_gradient_linear(
        x in -100.0..100.0_f64,
        y in -100.0..100.0_f64,
    ) {
        let g = gradient(
            |v| {
                let two = DualNumber::constant(2.0);
                let three = DualNumber::constant(3.0);
                v[0] * two + v[1] * three
            },
            &[x, y],
        );
        prop_assert!(
            approx_eq(g[0], 2.0),
            "gradient[0] of 2x+3y at ({}, {}) = {}, expected 2.0",
            x, y, g[0]
        );
        prop_assert!(
            approx_eq(g[1], 3.0),
            "gradient[1] of 2x+3y at ({}, {}) = {}, expected 3.0",
            x, y, g[1]
        );
    }

    // 14. Pythagorean identity: d/dx (sin^2(x) + cos^2(x)) = 0
    #[test]
    fn prop_pythagorean_derivative(x in -10.0..10.0_f64) {
        let d = derivative(
            |x| {
                let s = x.sin();
                let c = x.cos();
                s * s + c * c
            },
            x,
        );
        prop_assert!(
            approx_eq(d, 0.0),
            "derivative(sin^2+cos^2, {}) = {}, expected 0.0",
            x, d
        );
    }

    // 15. Exp-log inverse: d/dx exp(ln(x)) = 1 for x > 0
    #[test]
    fn prop_exp_log_inverse_derivative(x in 0.1..100.0_f64) {
        let d = derivative(|x| x.ln().exp(), x);
        prop_assert!(
            approx_eq(d, 1.0),
            "derivative(exp(ln(x)), {}) = {}, expected 1.0",
            x, d
        );
    }

    // --- Unique tests moved from src/tests.rs internal proptests (#299) ---

    // 16. Derivative of a constant is zero
    #[test]
    fn prop_derivative_of_constant_is_zero(c in -100.0..100.0_f64) {
        let d = derivative(|_x| DualNumber::constant(c), 1.0_f64);
        prop_assert!(approx_eq(d, 0.0));
    }

    // 17. d/dx x = 1
    #[test]
    fn prop_derivative_of_identity_is_one(x in -100.0..100.0_f64) {
        let d = derivative(|v| v, x);
        prop_assert!(approx_eq(d, 1.0));
    }

    // 18. d/dx (a*x + b) = a
    #[test]
    fn prop_derivative_linear(a in -10.0..10.0_f64, b in -10.0..10.0_f64, x in -10.0..10.0_f64) {
        let d = derivative(|v| v * DualNumber::constant(a) + DualNumber::constant(b), x);
        prop_assert!(approx_eq(d, a), "d/dx ({a}*x + {b}) at x={x}: got {d}, expected {a}");
    }

    // 19. T op DualNumber (#184) — verify 2.0 * x == x * 2.0
    #[test]
    fn prop_scalar_mul_commutative(x in -10.0..10.0_f64) {
        let d1 = derivative(|v| 2.0 * v, x);
        let d2 = derivative(|v| v * DualNumber::constant(2.0), x);
        prop_assert!(approx_eq(d1, d2));
    }

    // 20. Sum trait works (#302)
    #[test]
    fn prop_sum_matches_fold(n in 1usize..10) {
        let vals: Vec<DualNumber<f64>> = (0..n)
            .map(|i| DualNumber::new(i as f64, 1.0))
            .collect();
        let sum: DualNumber<f64> = vals.iter().copied().sum();
        let expected_real = (0..n).map(|i| i as f64).sum::<f64>();
        prop_assert!(approx_eq(sum.real, expected_real));
        prop_assert!(approx_eq(sum.dual, n as f64));
    }
}
