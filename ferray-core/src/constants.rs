// ferray-core: Mathematical constants (REQ-33)
//
// Mirrors numpy's constants module: np.pi, np.e, np.inf, np.nan, etc.

// `EULER_GAMMA` is a 16-digit canonical mathematical constant; underscore
// separators would diverge from the form quoted by every reference.
#![allow(clippy::unreadable_literal)]

/// The ratio of a circle's circumference to its diameter.
pub const PI: f64 = core::f64::consts::PI;

/// Euler's number, the base of natural logarithms.
pub const E: f64 = core::f64::consts::E;

/// Positive infinity.
pub const INF: f64 = f64::INFINITY;

/// Negative infinity.
pub const NEG_INF: f64 = f64::NEG_INFINITY;

/// Not a number (quiet NaN).
pub const NAN: f64 = f64::NAN;

/// The Euler-Mascheroni constant (gamma ~ 0.5772...).
///
/// This is the limiting difference between the harmonic series and the
/// natural logarithm: gamma = lim(n->inf) [sum(1/k, k=1..n) - ln(n)].
pub const EULER_GAMMA: f64 = 0.5772156649015329;

/// Positive zero (`+0.0`).
pub const PZERO: f64 = 0.0_f64;

/// Negative zero (`-0.0`).
pub const NZERO: f64 = -0.0_f64;

/// Sentinel value for `expand_dims` indicating a new axis should be inserted.
///
/// In `NumPy` this is `numpy.newaxis` (an alias for `None`). In ferray it is
/// a `usize` sentinel that is recognized by `expand_dims`.
pub const NEWAXIS: usize = usize::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_values() {
        assert_eq!(PI, std::f64::consts::PI);
        assert_eq!(E, std::f64::consts::E);
        assert_eq!(INF, f64::INFINITY);
        assert_eq!(NEG_INF, f64::NEG_INFINITY);
        assert!(NAN.is_nan());
        assert!(PZERO.is_sign_positive());
        assert!(NZERO.is_sign_negative());
        assert_eq!(PZERO, 0.0);
        assert_eq!(NZERO, 0.0);
    }

    #[test]
    fn euler_gamma_approximate() {
        // Known to ~16 decimal places
        assert!((EULER_GAMMA - 0.5772156649015329).abs() < 1e-15);
    }

    #[test]
    fn newaxis_is_sentinel() {
        assert_eq!(NEWAXIS, usize::MAX);
    }
}
