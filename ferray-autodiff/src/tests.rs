#[cfg(test)]
mod unit_tests {
    use crate::*;
    use num_traits::{Float, One, Zero};

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // --- Constructors ---

    #[test]
    fn test_new() {
        let d = DualNumber::new(3.0_f64, 4.0);
        assert_eq!(d.real, 3.0);
        assert_eq!(d.dual, 4.0);
    }

    #[test]
    fn test_constant() {
        let c = DualNumber::constant(5.0_f64);
        assert_eq!(c.real, 5.0);
        assert_eq!(c.dual, 0.0);
    }

    #[test]
    fn test_variable() {
        let v = DualNumber::variable(2.0_f64);
        assert_eq!(v.real, 2.0);
        assert_eq!(v.dual, 1.0);
    }

    // --- Basic arithmetic ---

    #[test]
    fn test_add() {
        let a = DualNumber::new(2.0_f64, 3.0);
        let b = DualNumber::new(4.0, 5.0);
        let c = a + b;
        assert_eq!(c.real, 6.0);
        assert_eq!(c.dual, 8.0);
    }

    #[test]
    fn test_sub() {
        let a = DualNumber::new(5.0_f64, 3.0);
        let b = DualNumber::new(2.0, 1.0);
        let c = a - b;
        assert_eq!(c.real, 3.0);
        assert_eq!(c.dual, 2.0);
    }

    #[test]
    fn test_mul() {
        // (2 + 1*eps) * (3 + 1*eps) = 6 + (2*1 + 1*3)*eps = 6 + 5*eps
        let a = DualNumber::new(2.0_f64, 1.0);
        let b = DualNumber::new(3.0, 1.0);
        let c = a * b;
        assert_eq!(c.real, 6.0);
        assert_eq!(c.dual, 5.0);
    }

    #[test]
    fn test_div() {
        // (6 + 1*eps) / (3 + 0*eps) = 2 + (1*3 - 6*0)/9 * eps = 2 + 1/3 * eps
        let a = DualNumber::new(6.0_f64, 1.0);
        let b = DualNumber::constant(3.0);
        let c = a / b;
        assert!(approx_eq(c.real, 2.0));
        assert!(approx_eq(c.dual, 1.0 / 3.0));
    }

    #[test]
    fn test_neg() {
        let a = DualNumber::new(3.0_f64, 4.0);
        let b = -a;
        assert_eq!(b.real, -3.0);
        assert_eq!(b.dual, -4.0);
    }

    // --- Compound assignment (issue #542) ---

    #[test]
    fn test_add_assign_dual() {
        let mut a = DualNumber::new(2.0_f64, 3.0);
        let b = DualNumber::new(4.0, 5.0);
        a += b;
        assert_eq!(a.real, 6.0);
        assert_eq!(a.dual, 8.0);
    }

    #[test]
    fn test_sub_assign_dual() {
        let mut a = DualNumber::new(5.0_f64, 3.0);
        let b = DualNumber::new(2.0, 1.0);
        a -= b;
        assert_eq!(a.real, 3.0);
        assert_eq!(a.dual, 2.0);
    }

    #[test]
    fn test_mul_assign_dual() {
        // d/dx (x^2) at x=2.5: variable starts with dual=1
        let mut a = DualNumber::variable(2.5_f64);
        a *= DualNumber::variable(2.5_f64);
        // (2.5 + eps) * (2.5 + eps) = 6.25 + 5*eps
        assert!(approx_eq(a.real, 6.25));
        assert!(approx_eq(a.dual, 5.0));
    }

    #[test]
    fn test_div_assign_dual() {
        let mut a = DualNumber::new(10.0_f64, 1.0);
        let b = DualNumber::new(2.0, 0.0);
        a /= b;
        assert!(approx_eq(a.real, 5.0));
        assert!(approx_eq(a.dual, 0.5));
    }

    #[test]
    fn test_add_assign_scalar() {
        let mut a = DualNumber::new(2.0_f64, 3.0);
        a += 4.0;
        assert_eq!(a.real, 6.0);
        assert_eq!(a.dual, 3.0);
    }

    #[test]
    fn test_mul_assign_scalar() {
        let mut a = DualNumber::new(2.0_f64, 3.0);
        a *= 4.0;
        assert_eq!(a.real, 8.0);
        assert_eq!(a.dual, 12.0);
    }

    #[test]
    fn test_accumulation_loop_compiles() {
        // The motivating use case for compound assign: sum loops.
        let mut sum = DualNumber::constant(0.0_f64);
        for i in 1..=5 {
            sum += DualNumber::variable(f64::from(i));
        }
        assert_eq!(sum.real, 15.0);
        assert_eq!(sum.dual, 5.0); // each variable contributes dual=1
    }

    // --- Remainder ---

    #[test]
    fn test_rem_dual_dual() {
        // d/dx (x % 3) = 1 (for non-integer x/3)
        let d = derivative(|x| x % DualNumber::constant(3.0), 7.5_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_rem_dual_scalar() {
        // d/dx (x % 3.0) = 1
        let x = DualNumber::variable(7.5_f64);
        let r = x % 3.0;
        assert!((r.real - 1.5).abs() < 1e-10);
        assert!(approx_eq(r.dual, 1.0));
    }

    #[test]
    fn test_rem_values() {
        // 7.5 % 3.0 = 1.5
        let a = DualNumber::new(7.5_f64, 1.0);
        let b = DualNumber::constant(3.0);
        let c = a % b;
        assert!((c.real - 1.5).abs() < 1e-10);
    }

    // --- Scalar arithmetic ---

    #[test]
    fn test_add_scalar() {
        let a = DualNumber::new(2.0_f64, 3.0);
        let c = a + 5.0;
        assert_eq!(c.real, 7.0);
        assert_eq!(c.dual, 3.0);
    }

    #[test]
    fn test_mul_scalar() {
        let a = DualNumber::new(2.0_f64, 3.0);
        let c = a * 4.0;
        assert_eq!(c.real, 8.0);
        assert_eq!(c.dual, 12.0);
    }

    // --- Trigonometric derivatives ---

    #[test]
    fn test_sin_derivative() {
        // d/dx sin(x) = cos(x)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::sin, x);
        assert!(approx_eq(d, x.cos()));
    }

    #[test]
    fn test_cos_derivative() {
        // d/dx cos(x) = -sin(x)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::cos, x);
        assert!(approx_eq(d, -x.sin()));
    }

    #[test]
    fn test_tan_derivative() {
        // d/dx tan(x) = 1/cos^2(x)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::tan, x);
        let expected = 1.0 / (x.cos() * x.cos());
        assert!(approx_eq(d, expected));
    }

    // --- Exponential / logarithm derivatives ---

    #[test]
    fn test_exp_derivative() {
        // d/dx exp(x) = exp(x)
        let x = 1.0_f64;
        let d = derivative(super::super::dual::DualNumber::exp, x);
        assert!(approx_eq(d, x.exp()));
    }

    #[test]
    fn test_exp_at_zero() {
        let d = derivative(super::super::dual::DualNumber::exp, 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_ln_derivative() {
        // d/dx ln(x) = 1/x
        let x = 2.0_f64;
        let d = derivative(super::super::dual::DualNumber::ln, x);
        assert!(approx_eq(d, 1.0 / x));
    }

    #[test]
    fn test_log2_derivative() {
        // d/dx log2(x) = 1/(x * ln(2))
        let x = 3.0_f64;
        let d = derivative(super::super::dual::DualNumber::log2, x);
        assert!(approx_eq(d, 1.0 / (x * 2.0_f64.ln())));
    }

    #[test]
    fn test_log10_derivative() {
        // d/dx log10(x) = 1/(x * ln(10))
        let x = 5.0_f64;
        let d = derivative(super::super::dual::DualNumber::log10, x);
        assert!(approx_eq(d, 1.0 / (x * 10.0_f64.ln())));
    }

    // --- Power / root derivatives ---

    #[test]
    fn test_sqrt_derivative() {
        // d/dx sqrt(x) = 1/(2*sqrt(x))
        let x = 4.0_f64;
        let d = derivative(super::super::dual::DualNumber::sqrt, x);
        assert!(approx_eq(d, 1.0 / (2.0 * x.sqrt())));
    }

    #[test]
    fn test_powi_derivative() {
        // d/dx x^3 = 3*x^2
        let d = derivative(|v| v.powi(3), 2.0_f64);
        assert!(approx_eq(d, 12.0));
    }

    #[test]
    fn test_powf_derivative() {
        // d/dx x^2.5 = 2.5 * x^1.5 (when exponent is constant)
        let x = 2.0_f64;
        let d = derivative(|v| v.powf(DualNumber::constant(2.5)), x);
        let expected = 2.5 * x.powf(1.5);
        assert!(approx_eq(d, expected));
    }

    #[test]
    fn test_powf_both_variable() {
        // f(x) = x^x at x=2: value = 4, derivative = x^x * (ln(x) + 1) = 4 * (ln(2) + 1)
        let x = 2.0_f64;
        let d = derivative(|v| v.powf(v), x);
        let expected = x.powf(x) * (x.ln() + 1.0);
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_powf_variable_exponent() {
        // f(p) = 2^p at p=3: value = 8, derivative = 2^p * ln(2) = 8 * ln(2)
        let p = 3.0_f64;
        let d = derivative(|v| DualNumber::constant(2.0).powf(v), p);
        let expected = p.exp2() * 2.0_f64.ln();
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_powf_at_zero_p_gt_1() {
        // d/dx x^2 at x=0 is 0
        let d = derivative(|v| v.powf(DualNumber::constant(2.0)), 0.0_f64);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn test_powf_at_zero_p_eq_1() {
        // d/dx x^1 at x=0 is 1
        let d = derivative(|v| v.powf(DualNumber::constant(1.0)), 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_powf_at_zero_p_lt_1() {
        // d/dx x^0.5 at x=0 is infinite (singular)
        let d = derivative(|v| v.powf(DualNumber::constant(0.5)), 0.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    // --- ReLU pattern: max(x, 0) ---

    #[test]
    fn test_relu_positive() {
        // d/dx max(x, 0) at x=3 is 1
        let d = derivative(|v| v.max(DualNumber::constant(0.0)), 3.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_relu_negative() {
        // d/dx max(x, 0) at x=-3 is 0
        let d = derivative(|v| v.max(DualNumber::constant(0.0)), -3.0_f64);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn test_relu_at_zero() {
        // d/dx max(x, 0) at x=0: convention returns 1 (self.real >= other.real picks self)
        let d = derivative(|v| v.max(DualNumber::constant(0.0)), 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_min_derivative() {
        // d/dx min(x, 1) at x=0.5 is 1 (x is smaller)
        let d = derivative(|v| v.min(DualNumber::constant(1.0)), 0.5_f64);
        assert!(approx_eq(d, 1.0));

        // d/dx min(x, 1) at x=2.0 is 0 (constant is smaller)
        let d = derivative(|v| v.min(DualNumber::constant(1.0)), 2.0_f64);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn test_abs_derivative() {
        // d/dx |x| = signum(x)
        let d = derivative(super::super::dual::DualNumber::abs, 3.0_f64);
        assert!(approx_eq(d, 1.0));

        let d = derivative(super::super::dual::DualNumber::abs, -3.0_f64);
        assert!(approx_eq(d, -1.0));
    }

    // --- Mathematical singularities ---
    // These document the behavior at points where derivatives are
    // infinite or undefined. The results are NaN or Inf, which is
    // correct — the derivative genuinely doesn't exist or is unbounded.

    #[test]
    fn test_sqrt_at_zero_singular() {
        // d/dx sqrt(x) = 1/(2*sqrt(x)), which is +inf at x=0
        let d = derivative(super::super::dual::DualNumber::sqrt, 0.0_f64);
        assert!(d.is_infinite() && d.is_sign_positive());
    }

    #[test]
    fn test_ln_at_zero_singular() {
        // d/dx ln(x) = 1/x, which is +inf at x=0+
        let d = derivative(super::super::dual::DualNumber::ln, 0.0_f64);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_recip_at_zero_singular() {
        // d/dx 1/x = -1/x^2, which is -inf at x=0
        let d = derivative(super::super::dual::DualNumber::recip, 0.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_asin_at_boundary() {
        // d/dx asin(x) = 1/sqrt(1-x^2), which is +inf at x=1
        let d = derivative(super::super::dual::DualNumber::asin, 1.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_acosh_at_boundary() {
        // d/dx acosh(x) = 1/sqrt(x^2-1), which is +inf at x=1
        let d = derivative(super::super::dual::DualNumber::acosh, 1.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_atanh_at_boundary() {
        // d/dx atanh(x) = 1/(1-x^2), which is +inf at x=1
        let d = derivative(super::super::dual::DualNumber::atanh, 1.0_f64);
        assert!(d.is_infinite());
    }

    // --- Extended edge cases at domain boundaries (#303) -------------
    //
    // The basic singularities are pinned above; broaden the coverage to
    // the negative side and the out-of-domain cases that should produce
    // NaN, plus a few logarithmic-family boundaries that previously had
    // no coverage at all.

    #[test]
    fn test_asin_at_neg_one() {
        // d/dx asin(x) = 1/sqrt(1-x^2). At x=-1 the derivative is
        // also +inf (symmetric to x=1), since 1 - x^2 = 0 from the
        // positive side.
        let d = derivative(super::super::dual::DualNumber::asin, -1.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_asin_outside_domain_is_nan() {
        // |x| > 1: asin itself is NaN, derivative also NaN.
        let d = derivative(super::super::dual::DualNumber::asin, 1.5_f64);
        assert!(d.is_nan());
    }

    #[test]
    fn test_acos_at_one_is_singular() {
        // d/dx acos(x) = -1/sqrt(1-x^2), -inf at x=1.
        let d = derivative(super::super::dual::DualNumber::acos, 1.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_atanh_at_neg_one_is_singular() {
        // d/dx atanh(x) = 1/(1-x^2), +inf at x=-1 by symmetry with x=1.
        let d = derivative(super::super::dual::DualNumber::atanh, -1.0_f64);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_atanh_outside_domain_returns_finite_derivative_formula() {
        // |x| > 1: atanh's value is NaN, but the dual implementation
        // computes the derivative algebraically as 1/(1-x^2), which
        // evaluates to a finite value for any x != ±1. Pin that the
        // derivative pair stays computable here even though the
        // function value is NaN.
        let d = derivative(super::super::dual::DualNumber::atanh, 2.0_f64);
        // 1 / (1 - 4) = -1/3.
        assert!(d.is_finite());
        assert!((d - (-1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_acosh_outside_domain_is_nan() {
        // x < 1: acosh itself is NaN, derivative NaN.
        let d = derivative(super::super::dual::DualNumber::acosh, 0.5_f64);
        assert!(d.is_nan());
    }

    #[test]
    fn test_sqrt_negative_is_nan() {
        // sqrt of a negative is NaN; derivative NaN.
        let d = derivative(super::super::dual::DualNumber::sqrt, -1.0_f64);
        assert!(d.is_nan());
    }

    #[test]
    fn test_ln_negative_is_nan() {
        // ln(x<0) is NaN, derivative 1/x is finite but the chain
        // through ln makes the value/derivative pair both NaN.
        let d = derivative(super::super::dual::DualNumber::ln, -1.0_f64);
        // The derivative of ln at a negative number is 1/(-1) = -1
        // formally, but the function value ln(-1) is NaN; the dual
        // pair propagates NaN through the result.
        // We accept either: -1.0 (mathematical derivative formula) or
        // NaN (NaN-poisoned dual). Pin behaviour: must not panic.
        assert!(d.is_nan() || (d - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_log2_at_zero_is_singular() {
        // d/dx log2(x) = 1/(x*ln(2)), +inf at x=0+.
        let d = derivative(super::super::dual::DualNumber::log2, 0.0_f64);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_log10_at_zero_is_singular() {
        let d = derivative(super::super::dual::DualNumber::log10, 0.0_f64);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_ln_1p_at_neg_one_is_singular() {
        // d/dx ln(1+x) = 1/(1+x); at x=-1, ln(1+x)=ln(0), derivative -inf.
        let d = derivative(super::super::dual::DualNumber::ln_1p, -1.0_f64);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_powf_zero_base_negative_exponent_is_inf() {
        // 0^(-1) is undefined / +inf for the value; derivative is also
        // singular. Exact behaviour depends on f64::powf rules.
        let d = derivative(
            |x: super::super::dual::DualNumber<f64>| {
                x.powf(super::super::dual::DualNumber::new(-1.0, 0.0))
            },
            0.0_f64,
        );
        // Either inf or NaN is acceptable; pin no-panic.
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_powi_zero_base_negative_exponent_is_singular() {
        // 0^-2 is undefined; derivative also singular.
        let d = derivative(|x: super::super::dual::DualNumber<f64>| x.powi(-2), 0.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_recip_at_zero_negative_side() {
        // d/dx 1/x = -1/x^2 → -inf as x→0 from either side, but a
        // floating-point 0.0 has no sign distinction in the derivative
        // formula. Pin no-panic + non-finite result.
        let d = derivative(super::super::dual::DualNumber::recip, -0.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    // --- Hyperbolic derivatives ---

    #[test]
    fn test_sinh_derivative() {
        // d/dx sinh(x) = cosh(x)
        let x = 1.0_f64;
        let d = derivative(super::super::dual::DualNumber::sinh, x);
        assert!(approx_eq(d, x.cosh()));
    }

    #[test]
    fn test_cosh_derivative() {
        // d/dx cosh(x) = sinh(x)
        let x = 1.0_f64;
        let d = derivative(super::super::dual::DualNumber::cosh, x);
        assert!(approx_eq(d, x.sinh()));
    }

    #[test]
    fn test_tanh_derivative() {
        // d/dx tanh(x) = 1 - tanh^2(x)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::tanh, x);
        let t = x.tanh();
        assert!(approx_eq(d, t.mul_add(-t, 1.0)));
    }

    // --- Inverse trig derivatives ---

    #[test]
    fn test_asin_derivative() {
        // d/dx asin(x) = 1/sqrt(1 - x^2)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::asin, x);
        assert!(approx_eq(d, 1.0 / x.mul_add(-x, 1.0).sqrt()));
    }

    #[test]
    fn test_acos_derivative() {
        // d/dx acos(x) = -1/sqrt(1 - x^2)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::acos, x);
        assert!(approx_eq(d, -1.0 / x.mul_add(-x, 1.0).sqrt()));
    }

    #[test]
    fn test_atan_derivative() {
        // d/dx atan(x) = 1/(1 + x^2)
        let x = 1.0_f64;
        let d = derivative(super::super::dual::DualNumber::atan, x);
        assert!(approx_eq(d, 1.0 / x.mul_add(x, 1.0)));
    }

    #[test]
    fn test_atan2_derivative() {
        // atan2(y, x): d/dy = x / (x^2 + y^2)
        let y = 3.0_f64;
        let x_val = 4.0_f64;
        let denom = x_val.mul_add(x_val, y * y);

        // partial derivative w.r.t. y
        let dy = DualNumber::variable(y);
        let dx = DualNumber::constant(x_val);
        let result = dy.atan2(dx);
        assert!(approx_eq(result.dual, x_val / denom));

        // partial derivative w.r.t. x
        let dy2 = DualNumber::constant(y);
        let dx2 = DualNumber::variable(x_val);
        let result2 = dy2.atan2(dx2);
        assert!(approx_eq(result2.dual, -y / denom));
    }

    // --- Chain rule ---

    #[test]
    fn test_chain_rule_sin_x_squared() {
        // d/dx sin(x^2) = 2x * cos(x^2)
        let x = 1.0_f64;
        let d = derivative(|v| (v * v).sin(), x);
        let expected = 2.0 * x * (x * x).cos();
        assert!(approx_eq(d, expected));
    }

    #[test]
    fn test_chain_rule_exp_sin() {
        // d/dx exp(sin(x)) = exp(sin(x)) * cos(x)
        let x = 0.5_f64;
        let d = derivative(|v| v.sin().exp(), x);
        let expected = x.sin().exp() * x.cos();
        assert!(approx_eq(d, expected));
    }

    // --- Convenience function tests ---

    #[test]
    fn test_derivative_sin_at_zero() {
        let d = derivative(super::super::dual::DualNumber::sin, 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_derivative_exp_at_zero() {
        let d = derivative(super::super::dual::DualNumber::exp, 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_derivative_powi_3_at_2() {
        let d = derivative(|x| x.powi(3), 2.0_f64);
        assert!(approx_eq(d, 12.0));
    }

    // --- Gradient ---

    #[test]
    fn test_gradient_x_squared_plus_y_squared() {
        // f(x, y) = x^2 + y^2
        // grad f = (2x, 2y)
        let g = gradient(|v| v[0] * v[0] + v[1] * v[1], &[3.0_f64, 4.0]);
        assert!(approx_eq(g[0], 6.0));
        assert!(approx_eq(g[1], 8.0));
    }

    #[test]
    fn test_gradient_product() {
        // f(x, y) = x * y
        // grad f = (y, x)
        let g = gradient(|v| v[0] * v[1], &[3.0_f64, 5.0]);
        assert!(approx_eq(g[0], 5.0));
        assert!(approx_eq(g[1], 3.0));
    }

    #[test]
    fn test_gradient_three_vars() {
        // f(x, y, z) = x*y + y*z + x*z
        // df/dx = y + z, df/dy = x + z, df/dz = y + x
        let g = gradient(
            |v| v[0] * v[1] + v[1] * v[2] + v[0] * v[2],
            &[1.0_f64, 2.0, 3.0],
        );
        assert!(approx_eq(g[0], 5.0)); // y + z = 2 + 3
        assert!(approx_eq(g[1], 4.0)); // x + z = 1 + 3
        assert!(approx_eq(g[2], 3.0)); // y + x = 2 + 1
    }

    // --- Jacobian ---

    #[test]
    fn test_jacobian_2x2() {
        // f(x, y) = (x*y, x + y)
        // J = [[y, x], [1, 1]]
        let (jac, m) = jacobian(|v| vec![v[0] * v[1], v[0] + v[1]], &[3.0_f64, 4.0]);
        assert_eq!(m, 2);
        assert!(approx_eq(jac[0], 4.0)); // df0/dx = y
        assert!(approx_eq(jac[1], 3.0)); // df0/dy = x
        assert!(approx_eq(jac[2], 1.0)); // df1/dx = 1
        assert!(approx_eq(jac[3], 1.0)); // df1/dy = 1
    }

    #[test]
    fn test_jacobian_3x2() {
        // f(x, y) = (x^2, x*y, y^2)
        // J = [[2x, 0], [y, x], [0, 2y]]
        let (jac, m) = jacobian(
            |v| vec![v[0] * v[0], v[0] * v[1], v[1] * v[1]],
            &[2.0_f64, 3.0],
        );
        assert_eq!(m, 3);
        assert!(approx_eq(jac[0], 4.0)); // 2x
        assert!(approx_eq(jac[1], 0.0)); // 0
        assert!(approx_eq(jac[2], 3.0)); // y
        assert!(approx_eq(jac[3], 2.0)); // x
        assert!(approx_eq(jac[4], 0.0)); // 0
        assert!(approx_eq(jac[5], 6.0)); // 2y
    }

    #[test]
    fn test_jacobian_1x1_matches_derivative() {
        // f(x) = sin(x) — Jacobian is just the derivative
        let (jac, m) = jacobian(|v| vec![v[0].sin()], &[0.5_f64]);
        assert_eq!(m, 1);
        assert!(approx_eq(jac[0], 0.5_f64.cos()));
    }

    // --- Higher-order derivatives via nesting ---

    #[test]
    fn test_second_derivative_x_squared() {
        // d²/dx² x² = 2
        // Nest: derivative of (derivative of x²) at x
        let x = 2.0_f64;
        let d2 = derivative(
            |outer| {
                // Inner derivative: d/dx x² = 2x
                // We seed the inner DualNumber as a variable of the outer DualNumber
                let inner = DualNumber::variable(outer);
                let result = inner * inner; // x²
                result.dual // = 2x, which is itself a DualNumber<f64>
            },
            x,
        );
        // d/dx (2x) = 2
        assert!(approx_eq(d2, 2.0));
    }

    #[test]
    fn test_second_derivative_sin() {
        // d²/dx² sin(x) = -sin(x)
        let x = 1.0_f64;
        let d2 = derivative(
            |outer| {
                let inner = DualNumber::variable(outer);
                inner.sin().dual // d/dx sin(x) = cos(x)
            },
            x,
        );
        // d/dx cos(x) = -sin(x)
        assert!((d2 - (-x.sin())).abs() < 1e-10);
    }

    #[test]
    fn test_second_derivative_exp() {
        // d²/dx² exp(x) = exp(x)
        let x = 0.5_f64;
        let d2 = derivative(
            |outer| {
                let inner = DualNumber::variable(outer);
                inner.exp().dual // d/dx exp(x) = exp(x)
            },
            x,
        );
        assert!((d2 - x.exp()).abs() < 1e-10);
    }

    // --- num_traits implementations ---

    #[test]
    fn test_zero() {
        let z: DualNumber<f64> = Zero::zero();
        assert_eq!(z.real, 0.0);
        assert_eq!(z.dual, 0.0);
        assert!(z.is_zero());
    }

    #[test]
    fn test_one() {
        let o: DualNumber<f64> = One::one();
        assert_eq!(o.real, 1.0);
        assert_eq!(o.dual, 0.0);
        assert!(o.is_one());
    }

    #[test]
    fn test_float_trait_basics() {
        // Verify DualNumber<f64> implements Float
        fn uses_float<F: Float>(x: F) -> F {
            x.sin()
        }

        let result = uses_float(DualNumber::variable(0.0_f64));
        assert!(approx_eq(result.real, 0.0));
        assert!(approx_eq(result.dual, 1.0));
    }

    #[test]
    fn test_float_nan() {
        let nan: DualNumber<f64> = Float::nan();
        assert!(nan.is_nan());
        assert!(!nan.is_finite());
    }

    #[test]
    fn test_float_infinity() {
        let inf: DualNumber<f64> = Float::infinity();
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
    }

    #[test]
    fn test_recip_derivative() {
        // d/dx 1/x = -1/x^2
        let x = 2.0_f64;
        let d = derivative(super::super::dual::DualNumber::recip, x);
        assert!(approx_eq(d, -1.0 / (x * x)));
    }

    #[test]
    fn test_cbrt_derivative() {
        // d/dx cbrt(x) = 1/(3 * x^(2/3))
        let x = 8.0_f64;
        let d = derivative(super::super::dual::DualNumber::cbrt, x);
        let expected = 1.0 / (3.0 * x.cbrt() * x.cbrt());
        assert!(approx_eq(d, expected));
    }

    #[test]
    fn test_hypot_derivative() {
        // d/dx sqrt(x^2 + c^2) = x / sqrt(x^2 + c^2) when c is constant
        let x = 3.0_f64;
        let c = 4.0_f64;
        let dx = DualNumber::variable(x);
        let dc = DualNumber::constant(c);
        let result = dx.hypot(dc);
        assert!(approx_eq(result.real, 5.0));
        assert!(approx_eq(result.dual, 3.0 / 5.0));
    }

    #[test]
    fn test_exp2_derivative() {
        // d/dx 2^x = 2^x * ln(2)
        let x = 3.0_f64;
        let d = derivative(super::super::dual::DualNumber::exp2, x);
        let expected = x.exp2() * 2.0_f64.ln();
        assert!(approx_eq(d, expected));
    }

    #[test]
    fn test_exp_m1_derivative() {
        // d/dx (exp(x) - 1) = exp(x)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::exp_m1, x);
        assert!(approx_eq(d, x.exp()));
    }

    #[test]
    fn test_ln_1p_derivative() {
        // d/dx ln(1+x) = 1/(1+x)
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::ln_1p, x);
        assert!(approx_eq(d, 1.0 / (1.0 + x)));
    }

    #[test]
    fn test_asinh_derivative() {
        // d/dx asinh(x) = 1/sqrt(x^2 + 1)
        let x = 1.0_f64;
        let d = derivative(super::super::dual::DualNumber::asinh, x);
        assert!(approx_eq(d, 1.0 / x.mul_add(x, 1.0).sqrt()));
    }

    #[test]
    fn test_acosh_derivative() {
        // d/dx acosh(x) = 1/sqrt(x^2 - 1), x > 1
        let x = 2.0_f64;
        let d = derivative(super::super::dual::DualNumber::acosh, x);
        assert!(approx_eq(d, 1.0 / x.mul_add(x, -1.0).sqrt()));
    }

    #[test]
    fn test_atanh_derivative() {
        // d/dx atanh(x) = 1/(1 - x^2), |x| < 1
        let x = 0.5_f64;
        let d = derivative(super::super::dual::DualNumber::atanh, x);
        assert!(approx_eq(d, 1.0 / x.mul_add(-x, 1.0)));
    }

    #[test]
    fn test_display() {
        let d = DualNumber::new(3.0_f64, 4.0);
        assert_eq!(format!("{d}"), "(3 + 4*eps)");
    }

    #[test]
    fn test_partial_ord() {
        let a = DualNumber::new(2.0_f64, 10.0);
        let b = DualNumber::new(3.0, 1.0);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_f32_support() {
        let d = derivative(|x: DualNumber<f32>| x.sin(), 0.0_f32);
        assert!((d - 1.0_f32).abs() < 1e-5);
    }

    #[test]
    fn test_complex_expression() {
        // d/dx (x^2 * sin(x) + exp(x)) at x = 1
        // = 2x*sin(x) + x^2*cos(x) + exp(x)
        let x = 1.0_f64;
        let d = derivative(|v| v * v * v.sin() + v.exp(), x);
        let expected = (2.0 * x).mul_add(x.sin(), x * x * x.cos()) + x.exp();
        assert!(approx_eq(d, expected));
    }

    #[test]
    fn test_sin_cos() {
        let x = DualNumber::variable(0.5_f64);
        let (s, c) = x.sin_cos();
        assert!(approx_eq(s.real, 0.5_f64.sin()));
        assert!(approx_eq(s.dual, 0.5_f64.cos()));
        assert!(approx_eq(c.real, 0.5_f64.cos()));
        assert!(approx_eq(c.dual, -0.5_f64.sin()));
    }

    #[test]
    fn test_mul_add() {
        // mul_add(a, b) = self * a + b
        let x = DualNumber::variable(2.0_f64);
        let a = DualNumber::constant(3.0);
        let b = DualNumber::constant(1.0);
        let result = x.mul_add(a, b);
        assert!(approx_eq(result.real, 7.0)); // 2*3 + 1
        assert!(approx_eq(result.dual, 3.0)); // d/dx (3x + 1) = 3
    }

    #[test]
    fn test_floor_ceil_round() {
        let x = DualNumber::variable(2.7_f64);
        assert_eq!(x.floor().real, 2.0);
        assert_eq!(x.floor().dual, 0.0);
        assert_eq!(x.ceil().real, 3.0);
        assert_eq!(x.ceil().dual, 0.0);
        assert_eq!(x.round().real, 3.0);
        assert_eq!(x.round().dual, 0.0);
    }

    #[test]
    fn test_fract() {
        let x = DualNumber::variable(2.7_f64);
        let f = x.fract();
        assert!((f.real - 0.7).abs() < TOL);
        assert!(approx_eq(f.dual, 1.0));
    }

    #[test]
    fn test_to_degrees_to_radians() {
        let x = DualNumber::variable(std::f64::consts::PI);
        let deg = x.to_degrees();
        assert!(approx_eq(deg.real, 180.0));

        let y = DualNumber::variable(180.0_f64);
        let rad = y.to_radians();
        assert!(approx_eq(rad.real, std::f64::consts::PI));
    }
}

#[cfg(test)]
mod finite_difference_tests {
    use crate::*;

    /// Central finite difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    fn numerical_derivative(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-7;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// Verify AD derivative matches numerical finite difference within tolerance.
    fn check_ad_vs_fd(
        ad_fn: impl Fn(DualNumber<f64>) -> DualNumber<f64>,
        scalar_fn: impl Fn(f64) -> f64,
        x: f64,
        tol: f64,
    ) {
        let ad_result = derivative(&ad_fn, x);
        let fd_result = numerical_derivative(&scalar_fn, x);
        assert!(
            (ad_result - fd_result).abs() < tol,
            "AD vs FD mismatch at x={x}: AD={ad_result}, FD={fd_result}, diff={}",
            (ad_result - fd_result).abs()
        );
    }

    #[test]
    fn fd_sin() {
        check_ad_vs_fd(super::super::dual::DualNumber::sin, f64::sin, 1.0, 1e-8);
    }

    #[test]
    fn fd_cos() {
        check_ad_vs_fd(super::super::dual::DualNumber::cos, f64::cos, 1.0, 1e-8);
    }

    #[test]
    fn fd_exp() {
        check_ad_vs_fd(super::super::dual::DualNumber::exp, f64::exp, 1.0, 1e-8);
    }

    #[test]
    fn fd_ln() {
        check_ad_vs_fd(super::super::dual::DualNumber::ln, f64::ln, 2.0, 1e-8);
    }

    #[test]
    fn fd_sqrt() {
        check_ad_vs_fd(super::super::dual::DualNumber::sqrt, f64::sqrt, 4.0, 1e-8);
    }

    #[test]
    fn fd_tanh() {
        check_ad_vs_fd(super::super::dual::DualNumber::tanh, f64::tanh, 0.5, 1e-8);
    }

    #[test]
    fn fd_asin() {
        check_ad_vs_fd(super::super::dual::DualNumber::asin, f64::asin, 0.5, 1e-8);
    }

    #[test]
    fn fd_complex_expression() {
        // f(x) = x^2 * sin(x) + exp(-x)
        let ad_fn = |x: DualNumber<f64>| x * x * x.sin() + (-x).exp();
        let scalar_fn = |x: f64| (x * x).mul_add(x.sin(), (-x).exp());
        check_ad_vs_fd(ad_fn, scalar_fn, 1.5, 1e-7);
    }

    #[test]
    fn fd_gradient_multivariate() {
        // f(x, y) = x^2 * y + sin(x * y)
        let g = gradient(
            |v| v[0] * v[0] * v[1] + (v[0] * v[1]).sin(),
            &[1.0_f64, 2.0],
        );

        let h = 1e-7;
        let f = |x: f64, y: f64| (x * x).mul_add(y, (x * y).sin());

        let fd_dx = (f(1.0 + h, 2.0) - f(1.0 - h, 2.0)) / (2.0 * h);
        let fd_dy = (f(1.0, 2.0 + h) - f(1.0, 2.0 - h)) / (2.0 * h);

        assert!(
            (g[0] - fd_dx).abs() < 1e-6,
            "df/dx: AD={}, FD={fd_dx}",
            g[0]
        );
        assert!(
            (g[1] - fd_dy).abs() < 1e-6,
            "df/dy: AD={}, FD={fd_dy}",
            g[1]
        );
    }
}

// Internal proptests moved to tests/property_tests.rs (#299) to
// avoid running duplicate cases in both the lib-test and the
// integration-test binary.
