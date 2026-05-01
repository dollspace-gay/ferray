//! Second-order derivative tests via nested DualNumber (#305).
//!
//! `DualNumber<T>` is generic over its component type, so a nested
//! `DualNumber<DualNumber<f64>>` carries (value, first-derivative)
//! pairs in both the real and dual slots — collectively giving the
//! function value, two copies of the first derivative, and the
//! second derivative in one forward pass. These tests verify that
//! the nested-dual computation produces correct second derivatives
//! for known closed-form cases.
//!
//! Setup convention: to differentiate twice at `a`, build
//!
//!     x = DualNumber::new(
//!         DualNumber::new(a,   1.0),  // outer real, inner dual = "1 for d/dx"
//!         DualNumber::new(1.0, 0.0),  // outer dual = "1 for d/dx", inner dual unused
//!     );
//!
//! Then for `y = f(x)`:
//!   - y.real.real = f(a)
//!   - y.real.dual = f'(a)
//!   - y.dual.real = f'(a)
//!   - y.dual.dual = f''(a)

use ferray_autodiff::DualNumber;

const TOL: f64 = 1e-10;

fn second_deriv<F>(f: F, a: f64) -> (f64, f64, f64)
where
    F: Fn(DualNumber<DualNumber<f64>>) -> DualNumber<DualNumber<f64>>,
{
    let x = DualNumber::<DualNumber<f64>>::new(
        DualNumber::<f64>::new(a, 1.0),
        DualNumber::<f64>::new(1.0, 0.0),
    );
    let y = f(x);
    (y.real.real, y.real.dual, y.dual.dual)
}

#[test]
fn second_derivative_of_x_squared_is_two() {
    // f(x) = x^2 → f'(x) = 2x, f''(x) = 2.
    let (val, d1, d2) = second_deriv(|x| x * x, 3.0);
    assert!((val - 9.0).abs() < TOL);
    assert!((d1 - 6.0).abs() < TOL);
    assert!((d2 - 2.0).abs() < TOL);
}

#[test]
fn second_derivative_of_x_cubed_is_six_x() {
    // f(x) = x^3 → f'(x) = 3x^2, f''(x) = 6x.
    let (val, d1, d2) = second_deriv(|x| x * x * x, 2.0);
    assert!((val - 8.0).abs() < TOL);
    assert!((d1 - 12.0).abs() < TOL);
    assert!((d2 - 12.0).abs() < TOL);
}

#[test]
fn second_derivative_of_sin_is_neg_sin() {
    // f(x) = sin(x) → f'(x) = cos(x), f''(x) = -sin(x).
    let (val, d1, d2) = second_deriv(DualNumber::sin, 1.0);
    assert!((val - 1.0_f64.sin()).abs() < TOL);
    assert!((d1 - 1.0_f64.cos()).abs() < TOL);
    assert!((d2 - (-1.0_f64.sin())).abs() < TOL);
}

#[test]
fn second_derivative_of_cos_is_neg_cos() {
    let (val, d1, d2) = second_deriv(DualNumber::cos, 1.5);
    assert!((val - 1.5_f64.cos()).abs() < TOL);
    assert!((d1 - (-1.5_f64.sin())).abs() < TOL);
    assert!((d2 - (-1.5_f64.cos())).abs() < TOL);
}

#[test]
fn second_derivative_of_exp_is_exp() {
    // f(x) = exp(x) → f'(x) = exp(x), f''(x) = exp(x).
    let (val, d1, d2) = second_deriv(DualNumber::exp, 0.5);
    let expected = 0.5_f64.exp();
    assert!((val - expected).abs() < TOL);
    assert!((d1 - expected).abs() < TOL);
    assert!((d2 - expected).abs() < TOL);
}

#[test]
fn second_derivative_of_ln_is_neg_recip_squared() {
    // f(x) = ln(x) → f'(x) = 1/x, f''(x) = -1/x^2.
    let a = 2.0_f64;
    let (val, d1, d2) = second_deriv(DualNumber::ln, a);
    assert!((val - a.ln()).abs() < TOL);
    assert!((d1 - 1.0 / a).abs() < TOL);
    assert!((d2 - (-1.0 / (a * a))).abs() < TOL);
}

#[test]
fn second_derivative_of_sqrt() {
    // f(x) = sqrt(x) → f'(x) = 1/(2 sqrt(x)), f''(x) = -1/(4 x^1.5).
    let a = 4.0_f64;
    let (val, d1, d2) = second_deriv(DualNumber::sqrt, a);
    assert!((val - 2.0).abs() < TOL);
    assert!((d1 - 0.25).abs() < TOL);
    let expected_d2 = -1.0 / (4.0 * a.powf(1.5));
    assert!((d2 - expected_d2).abs() < TOL);
}

#[test]
fn second_derivative_chain_rule() {
    // f(x) = sin(x^2) → f'(x) = 2x cos(x^2),
    //                    f''(x) = 2 cos(x^2) - 4 x^2 sin(x^2).
    let a = 1.5_f64;
    let (_val, d1, d2) = second_deriv(|x| (x * x).sin(), a);
    let expected_d1 = 2.0 * a * (a * a).cos();
    let expected_d2 = 2.0 * (a * a).cos() - 4.0 * a * a * (a * a).sin();
    assert!((d1 - expected_d1).abs() < TOL);
    assert!((d2 - expected_d2).abs() < TOL);
}

#[test]
fn second_derivative_of_constant_is_zero() {
    // f(x) = c → f'(x) = 0, f''(x) = 0.
    let (val, d1, d2) = second_deriv(|_| DualNumber::constant(DualNumber::constant(7.0)), 3.0);
    assert!((val - 7.0).abs() < TOL);
    assert!(d1.abs() < TOL);
    assert!(d2.abs() < TOL);
}

#[test]
fn second_derivative_of_identity_is_zero() {
    // f(x) = x → f'(x) = 1, f''(x) = 0.
    let (val, d1, d2) = second_deriv(|x| x, 5.0);
    assert!((val - 5.0).abs() < TOL);
    assert!((d1 - 1.0).abs() < TOL);
    assert!(d2.abs() < TOL);
}

#[test]
fn second_derivative_of_x_to_the_fourth() {
    // f(x) = x^4 → f''(x) = 12 x^2.
    let a = 2.0_f64;
    let (val, _d1, d2) = second_deriv(|x| x * x * x * x, a);
    assert!((val - 16.0).abs() < TOL);
    assert!((d2 - 48.0).abs() < TOL);
}
