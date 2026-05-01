// ferray-polynomial: Domain/window affine mapping
//
// NumPy's polynomial classes carry `domain` and `window` attributes that
// describe an affine map between an input interval (where x lives) and a
// canonical interval (where the basis is well-conditioned).
//
// This is critical for numerical stability — for example, fitting a
// Chebyshev series to data on [0, 1000] without mapping to [-1, 1] produces
// catastrophic cancellation. The mapping is `u = offset + scale * x` where
// `(offset, scale) = mapparms(domain, window)`.
//
// See: https://github.com/dollspace-gay/ferray/issues/474

use ferray_core::error::FerrayError;

/// Compute the affine map parameters `(offset, scale)` such that
/// `u = offset + scale * x` maps `domain` onto `window`.
///
/// Specifically, if `domain = [a, b]` and `window = [c, d]`, the linear
/// map sending `a -> c` and `b -> d` is:
///
/// ```text
///     u = c + (d - c) * (x - a) / (b - a)
///       = (c - a*(d-c)/(b-a)) + ((d-c)/(b-a)) * x
/// ```
///
/// so `scale = (d-c)/(b-a)` and `offset = c - scale*a`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `domain[0] == domain[1]` (the
/// domain is degenerate and the map would divide by zero).
pub fn mapparms(domain: [f64; 2], window: [f64; 2]) -> Result<(f64, f64), FerrayError> {
    let span = domain[1] - domain[0];
    if span == 0.0 {
        return Err(FerrayError::invalid_value(format!(
            "polynomial domain is degenerate: domain[0] == domain[1] == {}",
            domain[0]
        )));
    }
    let scale = (window[1] - window[0]) / span;
    let offset = scale.mul_add(-domain[0], window[0]);
    Ok((offset, scale))
}

/// Apply the mapping `u = offset + scale * x` to a single value.
#[inline]
#[must_use]
pub fn map_x(x: f64, offset: f64, scale: f64) -> f64 {
    scale.mul_add(x, offset)
}

/// Validate a domain/window pair (both must have non-degenerate spans
/// and be finite — NaN/Inf endpoints would silently corrupt every
/// downstream eval, fit, and root-finding operation, #252).
pub fn validate_domain_window(domain: [f64; 2], window: [f64; 2]) -> Result<(), FerrayError> {
    for (label, pair) in [("domain", domain), ("window", window)] {
        if !pair[0].is_finite() || !pair[1].is_finite() {
            return Err(FerrayError::invalid_value(format!(
                "polynomial {label} contains a non-finite value: {pair:?}"
            )));
        }
    }
    if domain[0] == domain[1] {
        return Err(FerrayError::invalid_value(format!(
            "polynomial domain is degenerate: {domain:?}"
        )));
    }
    if window[0] == window[1] {
        return Err(FerrayError::invalid_value(format!(
            "polynomial window is degenerate: {window:?}"
        )));
    }
    Ok(())
}

/// Compute the auto domain `[min, max]` from a slice of x values, falling
/// back to `[-1, 1]` for empty input.
#[must_use]
pub fn auto_domain(x: &[f64]) -> [f64; 2] {
    if x.is_empty() {
        return [-1.0, 1.0];
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &xi in x {
        if xi < min {
            min = xi;
        }
        if xi > max {
            max = xi;
        }
    }
    // Guard against all-equal x: fall back to a unit-width domain.
    if min == max {
        [min - 1.0, min + 1.0]
    } else {
        [min, max]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapparms_identity() {
        // domain == window -> offset = 0, scale = 1
        let (off, sc) = mapparms([-1.0, 1.0], [-1.0, 1.0]).unwrap();
        assert_eq!(off, 0.0);
        assert_eq!(sc, 1.0);
    }

    #[test]
    fn mapparms_zero_to_one_to_canonical() {
        // domain [0, 4], window [-1, 1]: u = -1 + 0.5*x
        let (off, sc) = mapparms([0.0, 4.0], [-1.0, 1.0]).unwrap();
        assert_eq!(off, -1.0);
        assert_eq!(sc, 0.5);
        // Spot-check: x=0 -> u=-1, x=4 -> u=1, x=2 -> u=0
        assert_eq!(map_x(0.0, off, sc), -1.0);
        assert_eq!(map_x(4.0, off, sc), 1.0);
        assert_eq!(map_x(2.0, off, sc), 0.0);
    }

    #[test]
    fn mapparms_negative_window() {
        // domain [10, 20], window [0, 1]: scale=0.1, offset=-1
        let (off, sc) = mapparms([10.0, 20.0], [0.0, 1.0]).unwrap();
        assert!((sc - 0.1).abs() < 1e-15);
        assert!((off - (-1.0)).abs() < 1e-15);
        assert!((map_x(15.0, off, sc) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn mapparms_degenerate_domain_errors() {
        assert!(mapparms([5.0, 5.0], [-1.0, 1.0]).is_err());
    }

    #[test]
    fn auto_domain_basic() {
        assert_eq!(auto_domain(&[0.0, 1.0, 2.0, 3.0, 4.0]), [0.0, 4.0]);
        assert_eq!(auto_domain(&[-3.0, 2.0, -1.0]), [-3.0, 2.0]);
    }

    #[test]
    fn auto_domain_empty() {
        let empty: &[f64] = &[];
        assert_eq!(auto_domain(empty), [-1.0, 1.0]);
    }

    #[test]
    fn auto_domain_constant() {
        // All-equal x: fall back to a unit-width domain centered on the value.
        assert_eq!(auto_domain(&[3.0, 3.0, 3.0]), [2.0, 4.0]);
    }
}
