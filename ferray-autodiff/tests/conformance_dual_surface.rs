//! Conformance coverage for the public dual-number convenience surface.
//!
//! These APIs do not map to a distinct NumPy fixture because they are the
//! forward-mode scalar building blocks used by the fixture-backed autodiff
//! entry points. The checks below pin their value and derivative contracts
//! directly against the analytic rules.

use ferray_autodiff::DualNumber;

fn d(real: f64, dual: f64) -> DualNumber<f64> {
    DualNumber::new(real, dual)
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{context}: got {actual}, expected {expected}"
    );
}

fn assert_dual_close(actual: DualNumber<f64>, expected_real: f64, expected_dual: f64, name: &str) {
    assert_close(actual.real, expected_real, &format!("{name}:real"));
    assert_close(actual.dual, expected_dual, &format!("{name}:dual"));
}

/// Path anchors:
/// `ferray_autodiff::dual::DualNumber`,
/// `ferray_autodiff::dual::DualNumber::new`,
/// `ferray_autodiff::dual::DualNumber::constant`,
/// `ferray_autodiff::dual::DualNumber::variable`.
#[test]
fn dual_number_constructors_seed_forward_mode_lanes() {
    let explicit: ferray_autodiff::dual::DualNumber<f64> =
        ferray_autodiff::dual::DualNumber::new(2.5, -0.75);
    assert_dual_close(explicit, 2.5, -0.75, "new");

    let constant = ferray_autodiff::dual::DualNumber::constant(3.5);
    assert_dual_close(constant, 3.5, 0.0, "constant");

    let variable = ferray_autodiff::dual::DualNumber::variable(-1.25);
    assert_dual_close(variable, -1.25, 1.0, "variable");
}

/// Path anchors:
/// `ferray_autodiff::abs`, `ferray_autodiff::acos`,
/// `ferray_autodiff::asin`, `ferray_autodiff::atan`,
/// `ferray_autodiff::cos`, `ferray_autodiff::cosh`,
/// `ferray_autodiff::exp`, `ferray_autodiff::ln`,
/// `ferray_autodiff::log10`, `ferray_autodiff::log2`,
/// `ferray_autodiff::sin`, `ferray_autodiff::sinh`,
/// `ferray_autodiff::sqrt`, `ferray_autodiff::tan`,
/// `ferray_autodiff::tanh`, `ferray_autodiff::atan2`.
#[test]
fn crate_root_function_reexports_match_dual_methods() {
    let x = d(0.4, 1.25);
    assert_dual_close(ferray_autodiff::sin(x), x.sin().real, x.sin().dual, "sin");
    assert_dual_close(ferray_autodiff::cos(x), x.cos().real, x.cos().dual, "cos");
    assert_dual_close(ferray_autodiff::tan(x), x.tan().real, x.tan().dual, "tan");
    assert_dual_close(
        ferray_autodiff::asin(x),
        x.asin().real,
        x.asin().dual,
        "asin",
    );
    assert_dual_close(
        ferray_autodiff::acos(x),
        x.acos().real,
        x.acos().dual,
        "acos",
    );
    assert_dual_close(
        ferray_autodiff::atan(x),
        x.atan().real,
        x.atan().dual,
        "atan",
    );
    assert_dual_close(
        ferray_autodiff::sinh(x),
        x.sinh().real,
        x.sinh().dual,
        "sinh",
    );
    assert_dual_close(
        ferray_autodiff::cosh(x),
        x.cosh().real,
        x.cosh().dual,
        "cosh",
    );
    assert_dual_close(
        ferray_autodiff::tanh(x),
        x.tanh().real,
        x.tanh().dual,
        "tanh",
    );

    let positive = d(1.7, -0.5);
    assert_dual_close(
        ferray_autodiff::exp(positive),
        positive.exp().real,
        positive.exp().dual,
        "exp",
    );
    assert_dual_close(
        ferray_autodiff::ln(positive),
        positive.ln().real,
        positive.ln().dual,
        "ln",
    );
    assert_dual_close(
        ferray_autodiff::log2(positive),
        positive.log2().real,
        positive.log2().dual,
        "log2",
    );
    assert_dual_close(
        ferray_autodiff::log10(positive),
        positive.log10().real,
        positive.log10().dual,
        "log10",
    );
    assert_dual_close(
        ferray_autodiff::sqrt(positive),
        positive.sqrt().real,
        positive.sqrt().dual,
        "sqrt",
    );

    let signed = d(-2.0, 0.75);
    assert_dual_close(
        ferray_autodiff::abs(signed),
        signed.abs().real,
        signed.abs().dual,
        "abs",
    );

    let y = d(1.2, 0.5);
    let x_axis = d(2.3, -0.25);
    assert_dual_close(
        ferray_autodiff::atan2(y, x_axis),
        y.atan2(x_axis).real,
        y.atan2(x_axis).dual,
        "atan2",
    );
}

/// Path anchors:
/// `ferray_autodiff::functions::DualNumber::abs`,
/// `ferray_autodiff::functions::DualNumber::acos`,
/// `ferray_autodiff::functions::DualNumber::acosh`,
/// `ferray_autodiff::functions::DualNumber::asin`,
/// `ferray_autodiff::functions::DualNumber::asinh`,
/// `ferray_autodiff::functions::DualNumber::atan`,
/// `ferray_autodiff::functions::DualNumber::atan2`,
/// `ferray_autodiff::functions::DualNumber::atanh`,
/// `ferray_autodiff::functions::DualNumber::cbrt`,
/// `ferray_autodiff::functions::DualNumber::ceil`,
/// `ferray_autodiff::functions::DualNumber::cos`,
/// `ferray_autodiff::functions::DualNumber::cosh`,
/// `ferray_autodiff::functions::DualNumber::exp`,
/// `ferray_autodiff::functions::DualNumber::exp2`,
/// `ferray_autodiff::functions::DualNumber::exp_m1`,
/// `ferray_autodiff::functions::DualNumber::floor`,
/// `ferray_autodiff::functions::DualNumber::fract`,
/// `ferray_autodiff::functions::DualNumber::hypot`,
/// `ferray_autodiff::functions::DualNumber::ln`,
/// `ferray_autodiff::functions::DualNumber::ln_1p`,
/// `ferray_autodiff::functions::DualNumber::log`,
/// `ferray_autodiff::functions::DualNumber::log10`,
/// `ferray_autodiff::functions::DualNumber::log2`,
/// `ferray_autodiff::functions::DualNumber::mul_add`,
/// `ferray_autodiff::functions::DualNumber::powf`,
/// `ferray_autodiff::functions::DualNumber::powi`,
/// `ferray_autodiff::functions::DualNumber::recip`,
/// `ferray_autodiff::functions::DualNumber::round`,
/// `ferray_autodiff::functions::DualNumber::signum`,
/// `ferray_autodiff::functions::DualNumber::sin`,
/// `ferray_autodiff::functions::DualNumber::sin_cos`,
/// `ferray_autodiff::functions::DualNumber::sinh`,
/// `ferray_autodiff::functions::DualNumber::sqrt`,
/// `ferray_autodiff::functions::DualNumber::tan`,
/// `ferray_autodiff::functions::DualNumber::tanh`,
/// `ferray_autodiff::functions::DualNumber::to_degrees`,
/// `ferray_autodiff::functions::DualNumber::to_radians`,
/// `ferray_autodiff::functions::DualNumber::trunc`.
#[test]
fn dual_methods_match_analytic_forward_mode_rules() {
    let x = d(0.4, 1.25);
    assert_dual_close(x.sin(), x.real.sin(), x.dual * x.real.cos(), "sin");
    assert_dual_close(x.cos(), x.real.cos(), -x.dual * x.real.sin(), "cos");
    assert_dual_close(
        x.tan(),
        x.real.tan(),
        x.dual / (x.real.cos() * x.real.cos()),
        "tan",
    );
    assert_dual_close(
        x.asin(),
        x.real.asin(),
        x.dual / (1.0 - x.real * x.real).sqrt(),
        "asin",
    );
    assert_dual_close(
        x.acos(),
        x.real.acos(),
        -x.dual / (1.0 - x.real * x.real).sqrt(),
        "acos",
    );
    assert_dual_close(
        x.atan(),
        x.real.atan(),
        x.dual / (1.0 + x.real * x.real),
        "atan",
    );
    assert_dual_close(x.sinh(), x.real.sinh(), x.dual * x.real.cosh(), "sinh");
    assert_dual_close(x.cosh(), x.real.cosh(), x.dual * x.real.sinh(), "cosh");
    assert_dual_close(
        x.tanh(),
        x.real.tanh(),
        x.dual * (1.0 - x.real.tanh() * x.real.tanh()),
        "tanh",
    );
    assert_dual_close(
        x.asinh(),
        x.real.asinh(),
        x.dual / (x.real * x.real + 1.0).sqrt(),
        "asinh",
    );
    assert_dual_close(
        x.atanh(),
        x.real.atanh(),
        x.dual / (1.0 - x.real * x.real),
        "atanh",
    );

    let acosh_x = d(2.5, -0.75);
    assert_dual_close(
        acosh_x.acosh(),
        acosh_x.real.acosh(),
        acosh_x.dual / (acosh_x.real * acosh_x.real - 1.0).sqrt(),
        "acosh",
    );

    let positive = d(1.7, -0.5);
    assert_dual_close(
        positive.exp(),
        positive.real.exp(),
        positive.dual * positive.real.exp(),
        "exp",
    );
    assert_dual_close(
        positive.exp_m1(),
        positive.real.exp_m1(),
        positive.dual * positive.real.exp(),
        "exp_m1",
    );
    assert_dual_close(
        positive.exp2(),
        positive.real.exp2(),
        positive.dual * positive.real.exp2() * 2.0_f64.ln(),
        "exp2",
    );
    assert_dual_close(
        positive.ln(),
        positive.real.ln(),
        positive.dual / positive.real,
        "ln",
    );
    assert_dual_close(
        positive.log2(),
        positive.real.log2(),
        positive.dual / (positive.real * 2.0_f64.ln()),
        "log2",
    );
    assert_dual_close(
        positive.log10(),
        positive.real.log10(),
        positive.dual / (positive.real * 10.0_f64.ln()),
        "log10",
    );
    assert_dual_close(
        positive.ln_1p(),
        positive.real.ln_1p(),
        positive.dual / (1.0 + positive.real),
        "ln_1p",
    );
    assert_dual_close(
        positive.sqrt(),
        positive.real.sqrt(),
        positive.dual / (2.0 * positive.real.sqrt()),
        "sqrt",
    );
    assert_dual_close(
        positive.cbrt(),
        positive.real.cbrt(),
        positive.dual / (3.0 * positive.real.cbrt() * positive.real.cbrt()),
        "cbrt",
    );
    assert_dual_close(
        positive.recip(),
        positive.real.recip(),
        -positive.dual / (positive.real * positive.real),
        "recip",
    );

    let log_base = d(2.0, 0.2);
    let log_x = d(4.5, 0.7);
    let ln_x = log_x.real.ln();
    let ln_base = log_base.real.ln();
    assert_dual_close(
        log_x.log(log_base),
        ln_x / ln_base,
        log_x.dual / (log_x.real * ln_base)
            - log_base.dual * ln_x / (log_base.real * ln_base * ln_base),
        "log",
    );

    let pow_base = d(2.0, 0.7);
    let pow_exp = d(3.0, -0.2);
    let pow_value = pow_base.real.powf(pow_exp.real);
    assert_dual_close(
        pow_base.powf(pow_exp),
        pow_value,
        pow_value
            * (pow_exp.dual * pow_base.real.ln() + pow_exp.real * pow_base.dual / pow_base.real),
        "powf",
    );
    assert_dual_close(
        pow_base.powi(3),
        pow_base.real.powi(3),
        pow_base.dual * 3.0 * pow_base.real.powi(2),
        "powi",
    );

    let y = d(1.2, 0.5);
    let x_axis = d(2.3, -0.25);
    let denom = y.real * y.real + x_axis.real * x_axis.real;
    assert_dual_close(
        y.atan2(x_axis),
        y.real.atan2(x_axis.real),
        (y.dual * x_axis.real - x_axis.dual * y.real) / denom,
        "atan2",
    );
    let hypot = y.real.hypot(x_axis.real);
    assert_dual_close(
        y.hypot(x_axis),
        hypot,
        (y.real * y.dual + x_axis.real * x_axis.dual) / hypot,
        "hypot",
    );
    assert_dual_close(
        y.mul_add(x_axis, d(-0.4, 0.8)),
        y.real.mul_add(x_axis.real, -0.4),
        y.dual * x_axis.real + y.real * x_axis.dual + 0.8,
        "mul_add",
    );

    let signed = d(-2.0, 0.75);
    assert_dual_close(
        signed.abs(),
        signed.real.abs(),
        signed.dual * signed.real.signum(),
        "abs",
    );
    assert_dual_close(signed.signum(), signed.real.signum(), 0.0, "signum");

    let discrete = d(2.75, -0.5);
    assert_dual_close(discrete.floor(), discrete.real.floor(), 0.0, "floor");
    assert_dual_close(discrete.ceil(), discrete.real.ceil(), 0.0, "ceil");
    assert_dual_close(discrete.round(), discrete.real.round(), 0.0, "round");
    assert_dual_close(discrete.trunc(), discrete.real.trunc(), 0.0, "trunc");
    assert_dual_close(
        discrete.fract(),
        discrete.real.fract(),
        discrete.dual,
        "fract",
    );
    assert_dual_close(
        discrete.to_degrees(),
        discrete.real.to_degrees(),
        discrete.dual * 180.0 / std::f64::consts::PI,
        "to_degrees",
    );
    assert_dual_close(
        discrete.to_radians(),
        discrete.real.to_radians(),
        discrete.dual * std::f64::consts::PI / 180.0,
        "to_radians",
    );

    let (sin_x, cos_x) = x.sin_cos();
    assert_dual_close(sin_x, x.real.sin(), x.dual * x.real.cos(), "sin_cos:sin");
    assert_dual_close(cos_x, x.real.cos(), -x.dual * x.real.sin(), "sin_cos:cos");
}
