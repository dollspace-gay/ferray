//! `Display` implementations for the polynomial basis types (#486).
//!
//! Format mirrors numpy's text representation, with the basis-specific
//! term name (`x^i`, `T_i(x)`, `H_i(x)`, etc.). Examples:
//!
//! ```text
//! Polynomial: 1 + 2*x + 3*x^2
//! Chebyshev:  1 + 2*T_1(x) + 3*T_2(x)
//! Legendre:   1 + 2*P_1(x) + 3*P_2(x)
//! Hermite:    1 + 2*H_1(x) + 3*H_2(x)
//! HermiteE:   1 + 2*He_1(x) + 3*He_2(x)
//! Laguerre:   1 + 2*L_1(x) + 3*L_2(x)
//! ```
//!
//! Zero coefficients are skipped. Negative coefficients use ` - `
//! between terms and emit the absolute value of the coefficient.
//! The all-zero polynomial prints as `0`.

use core::fmt;

use crate::chebyshev::Chebyshev;
use crate::hermite::Hermite;
use crate::hermite_e::HermiteE;
use crate::laguerre::Laguerre;
use crate::legendre::Legendre;
use crate::power::Polynomial;
use crate::traits::Poly;

/// Format `coeffs` using the supplied per-degree term-name function
/// (e.g. for power basis, `term(0) = ""`, `term(1) = "x"`,
/// `term(i) = "x^i"`).
fn format_polynomial(
    f: &mut fmt::Formatter<'_>,
    coeffs: &[f64],
    term: impl Fn(usize) -> String,
) -> fmt::Result {
    let mut first = true;
    let mut wrote_anything = false;
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        let abs = c.abs();
        let sign_char = if c < 0.0 { '-' } else { '+' };
        let term_str = term(i);

        if first {
            // Leading term: no leading sign for positive, just "-" for negative.
            if c < 0.0 {
                f.write_str("-")?;
            }
            if i == 0 || abs != 1.0 {
                write!(f, "{abs}")?;
                if !term_str.is_empty() {
                    write!(f, "*{term_str}")?;
                }
            } else {
                write!(f, "{term_str}")?;
            }
            first = false;
        } else {
            write!(f, " {sign_char} ")?;
            if i == 0 || abs != 1.0 {
                write!(f, "{abs}")?;
                if !term_str.is_empty() {
                    write!(f, "*{term_str}")?;
                }
            } else {
                write!(f, "{term_str}")?;
            }
        }
        wrote_anything = true;
    }
    if !wrote_anything {
        f.write_str("0")?;
    }
    Ok(())
}

fn power_term(i: usize) -> String {
    match i {
        0 => String::new(),
        1 => "x".into(),
        n => format!("x^{n}"),
    }
}

fn basis_term(name: &str, i: usize) -> String {
    match i {
        0 => String::new(),
        n => format!("{name}_{n}(x)"),
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_polynomial(f, self.coeffs(), power_term)
    }
}

impl fmt::Display for Chebyshev {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_polynomial(f, self.coeffs(), |i| basis_term("T", i))
    }
}

impl fmt::Display for Legendre {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_polynomial(f, self.coeffs(), |i| basis_term("P", i))
    }
}

impl fmt::Display for Hermite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_polynomial(f, self.coeffs(), |i| basis_term("H", i))
    }
}

impl fmt::Display for HermiteE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_polynomial(f, self.coeffs(), |i| basis_term("He", i))
    }
}

impl fmt::Display for Laguerre {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_polynomial(f, self.coeffs(), |i| basis_term("L", i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_constant() {
        let p = Polynomial::new(&[5.0]);
        assert_eq!(p.to_string(), "5");
    }

    #[test]
    fn power_zero() {
        let p = Polynomial::new(&[0.0]);
        assert_eq!(p.to_string(), "0");
    }

    #[test]
    fn power_linear() {
        let p = Polynomial::new(&[1.0, 2.0]);
        assert_eq!(p.to_string(), "1 + 2*x");
    }

    #[test]
    fn power_quadratic_with_minus() {
        // 1 - 3x + 2x^2
        let p = Polynomial::new(&[1.0, -3.0, 2.0]);
        assert_eq!(p.to_string(), "1 - 3*x + 2*x^2");
    }

    #[test]
    fn power_skips_zero_terms() {
        // 1 + 0*x + 4*x^2 → "1 + 4*x^2"
        let p = Polynomial::new(&[1.0, 0.0, 4.0]);
        assert_eq!(p.to_string(), "1 + 4*x^2");
    }

    #[test]
    fn power_unit_coefficient_is_implicit() {
        // 1 + x + x^2
        let p = Polynomial::new(&[1.0, 1.0, 1.0]);
        assert_eq!(p.to_string(), "1 + x + x^2");
    }

    #[test]
    fn power_unit_negative_coefficient() {
        // 1 - x + x^3 (skipping x^2)
        let p = Polynomial::new(&[1.0, -1.0, 0.0, 1.0]);
        assert_eq!(p.to_string(), "1 - x + x^3");
    }

    #[test]
    fn chebyshev_format() {
        let p = Chebyshev::new(&[1.0, 2.0, 3.0]);
        assert_eq!(p.to_string(), "1 + 2*T_1(x) + 3*T_2(x)");
    }

    #[test]
    fn legendre_format() {
        let p = Legendre::new(&[0.0, 1.0, -1.0]);
        // First non-zero is the linear term, so "P_1(x) - P_2(x)".
        assert_eq!(p.to_string(), "P_1(x) - P_2(x)");
    }

    #[test]
    fn hermite_format() {
        let p = Hermite::new(&[2.0, 0.0, -1.0]);
        assert_eq!(p.to_string(), "2 - H_2(x)");
    }

    #[test]
    fn hermite_e_format() {
        let p = HermiteE::new(&[1.0, 1.0]);
        assert_eq!(p.to_string(), "1 + He_1(x)");
    }

    #[test]
    fn laguerre_format() {
        let p = Laguerre::new(&[3.0]);
        assert_eq!(p.to_string(), "3");
    }
}
