//! Differentiable methods on [`DualNumber`] plus flat free-function
//! mirrors (`sin`, `cos`, `exp`, …) for a numpy-style namespace.
//!
//! The free functions all delegate to the corresponding method on
//! `DualNumber<T>` — they exist so callers can write `autodiff::sin(x)`
//! without having to remember whether a given op is a method or a
//! function.

use num_traits::Float;

use crate::dual::DualNumber;

impl<T: Float> DualNumber<T> {
    /// Sine function. Derivative: `cos(x)`.
    #[inline]
    pub fn sin(self) -> Self {
        Self {
            real: self.real.sin(),
            dual: self.dual * self.real.cos(),
        }
    }

    /// Cosine function. Derivative: `-sin(x)`.
    #[inline]
    pub fn cos(self) -> Self {
        Self {
            real: self.real.cos(),
            dual: -self.dual * self.real.sin(),
        }
    }

    /// Tangent function. Derivative: `1/cos^2(x)`.
    #[inline]
    pub fn tan(self) -> Self {
        let c = self.real.cos();
        Self {
            real: self.real.tan(),
            dual: self.dual / (c * c),
        }
    }

    /// Exponential function. Derivative: `exp(x)`.
    #[inline]
    pub fn exp(self) -> Self {
        let e = self.real.exp();
        Self {
            real: e,
            dual: self.dual * e,
        }
    }

    /// `exp(x) - 1`, more accurate for small x. Derivative: `exp(x)`.
    #[inline]
    pub fn exp_m1(self) -> Self {
        Self {
            real: self.real.exp_m1(),
            dual: self.dual * self.real.exp(),
        }
    }

    /// `exp2(x) = 2^x`. Derivative: `2^x * ln(2)`.
    #[inline]
    pub fn exp2(self) -> Self {
        let e = self.real.exp2();
        let ln2 = T::from(2.0).unwrap().ln();
        Self {
            real: e,
            dual: self.dual * e * ln2,
        }
    }

    /// Natural logarithm. Derivative: `1/x`.
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            real: self.real.ln(),
            dual: self.dual / self.real,
        }
    }

    /// Logarithm with arbitrary base. Derivatives:
    /// - d/dx log_b(x) = 1/(x ln b)
    /// - d/db log_b(x) = -ln(x) / (b ln(b)^2)
    #[inline]
    pub fn log(self, base: Self) -> Self {
        let ln_self = self.real.ln();
        let ln_base = base.real.ln();
        let ln_base_sq = ln_base * ln_base;
        Self {
            real: ln_self / ln_base,
            dual: self.dual / (self.real * ln_base)
                - base.dual * ln_self / (base.real * ln_base_sq),
        }
    }

    /// Base-2 logarithm. Derivative: `1/(x * ln(2))`.
    #[inline]
    pub fn log2(self) -> Self {
        let ln2 = T::from(2.0).unwrap().ln();
        Self {
            real: self.real.log2(),
            dual: self.dual / (self.real * ln2),
        }
    }

    /// Base-10 logarithm. Derivative: `1/(x * ln(10))`.
    #[inline]
    pub fn log10(self) -> Self {
        let ln10 = T::from(10.0).unwrap().ln();
        Self {
            real: self.real.log10(),
            dual: self.dual / (self.real * ln10),
        }
    }

    /// `ln(1 + x)`, more accurate for small x. Derivative: `1/(1 + x)`.
    #[inline]
    pub fn ln_1p(self) -> Self {
        Self {
            real: self.real.ln_1p(),
            dual: self.dual / (T::one() + self.real),
        }
    }

    /// Square root. Derivative: `1/(2*sqrt(x))`.
    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.real.sqrt();
        Self {
            real: s,
            dual: self.dual / (T::from(2.0).unwrap() * s),
        }
    }

    /// Cube root. Derivative: `1/(3*x^(2/3))`.
    #[inline]
    pub fn cbrt(self) -> Self {
        let c = self.real.cbrt();
        let three = T::from(3.0).unwrap();
        Self {
            real: c,
            dual: self.dual / (three * c * c),
        }
    }

    /// Absolute value. Derivative: `signum(x)`.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            real: self.real.abs(),
            dual: self.dual * self.real.signum(),
        }
    }

    /// Signum function. Derivative is zero (piecewise constant).
    #[inline]
    pub fn signum(self) -> Self {
        Self {
            real: self.real.signum(),
            dual: T::zero(),
        }
    }

    /// Integer power. Derivative: `n * x^(n-1)`.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let nf = T::from(n).unwrap();
        Self {
            real: self.real.powi(n),
            dual: self.dual * nf * self.real.powi(n - 1),
        }
    }

    /// Floating-point power where the exponent is a DualNumber.
    /// `x^p` where both x and p may carry derivatives.
    ///
    /// Derivative components:
    /// - d/dx: `p * x^(p-1) * dx`
    /// - d/dp: `x^p * ln(x) * dp`
    ///
    /// At `x = 0`:
    /// - For `p > 1`: derivative is 0 (both components).
    /// - For `p == 1`: d/dx component is `dx`, d/dp component is 0.
    /// - For `0 < p < 1`: d/dx is infinite (returns NaN/Inf).
    /// - For `p <= 0`: undefined (returns NaN).
    ///
    /// For `x < 0` with non-integer `p`, the real part is NaN (as with `f64::powf`),
    /// and the dual part is also NaN.
    #[inline]
    pub fn powf(self, p: Self) -> Self {
        let x = self.real;
        let val = x.powf(p.real);

        // Handle x == 0 to avoid ln(0) and 0/0
        if x == T::zero() {
            let one = T::one();
            let zero = T::zero();
            if p.real > one {
                // x^p = 0, d/dx = p * 0^(p-1) = 0, d/dp = 0^p * ln(0) -> 0
                return Self {
                    real: val,
                    dual: zero,
                };
            } else if p.real == one {
                // x^1 = x, d/dx = 1 * dx, d/dp = 0^1 * ln(0) -> 0
                return Self {
                    real: val,
                    dual: self.dual,
                };
            }
            // p < 1 at x = 0: derivative is singular, let NaN/Inf propagate
            return Self {
                real: val,
                dual: val * (p.dual * x.ln() + p.real * self.dual / x),
            };
        }

        Self {
            real: val,
            dual: val * (p.dual * x.ln() + p.real * self.dual / x),
        }
    }

    /// Hyperbolic sine. Derivative: `cosh(x)`.
    #[inline]
    pub fn sinh(self) -> Self {
        Self {
            real: self.real.sinh(),
            dual: self.dual * self.real.cosh(),
        }
    }

    /// Hyperbolic cosine. Derivative: `sinh(x)`.
    #[inline]
    pub fn cosh(self) -> Self {
        Self {
            real: self.real.cosh(),
            dual: self.dual * self.real.sinh(),
        }
    }

    /// Hyperbolic tangent. Derivative: `1 - tanh^2(x)`.
    #[inline]
    pub fn tanh(self) -> Self {
        let t = self.real.tanh();
        Self {
            real: t,
            dual: self.dual * (T::one() - t * t),
        }
    }

    /// Inverse sine. Derivative: `1/sqrt(1 - x^2)`.
    #[inline]
    pub fn asin(self) -> Self {
        Self {
            real: self.real.asin(),
            dual: self.dual / (T::one() - self.real * self.real).sqrt(),
        }
    }

    /// Inverse cosine. Derivative: `-1/sqrt(1 - x^2)`.
    #[inline]
    pub fn acos(self) -> Self {
        Self {
            real: self.real.acos(),
            dual: -self.dual / (T::one() - self.real * self.real).sqrt(),
        }
    }

    /// Inverse tangent. Derivative: `1/(1 + x^2)`.
    #[inline]
    pub fn atan(self) -> Self {
        Self {
            real: self.real.atan(),
            dual: self.dual / (T::one() + self.real * self.real),
        }
    }

    /// Two-argument inverse tangent `atan2(y, x)`.
    ///
    /// Derivatives:
    /// - d/dy = x / (x^2 + y^2)
    /// - d/dx = -y / (x^2 + y^2)
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        let denom = self.real * self.real + other.real * other.real;
        Self {
            real: self.real.atan2(other.real),
            dual: (self.dual * other.real - other.dual * self.real) / denom,
        }
    }

    /// Simultaneously computes sine and cosine.
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real.sin_cos();
        (
            Self {
                real: s,
                dual: self.dual * c,
            },
            Self {
                real: c,
                dual: -self.dual * s,
            },
        )
    }

    /// Inverse hyperbolic sine. Derivative: `1/sqrt(x^2 + 1)`.
    #[inline]
    pub fn asinh(self) -> Self {
        Self {
            real: self.real.asinh(),
            dual: self.dual / (self.real * self.real + T::one()).sqrt(),
        }
    }

    /// Inverse hyperbolic cosine. Derivative: `1/sqrt(x^2 - 1)`.
    #[inline]
    pub fn acosh(self) -> Self {
        Self {
            real: self.real.acosh(),
            dual: self.dual / (self.real * self.real - T::one()).sqrt(),
        }
    }

    /// Inverse hyperbolic tangent. Derivative: `1/(1 - x^2)`.
    #[inline]
    pub fn atanh(self) -> Self {
        Self {
            real: self.real.atanh(),
            dual: self.dual / (T::one() - self.real * self.real),
        }
    }

    /// Hypotenuse: `sqrt(x^2 + y^2)`.
    #[inline]
    pub fn hypot(self, other: Self) -> Self {
        let h = self.real.hypot(other.real);
        Self {
            real: h,
            dual: (self.real * self.dual + other.real * other.dual) / h,
        }
    }

    /// Fused multiply-add: `self * a + b`.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // (self * a) + b
        Self {
            real: self.real.mul_add(a.real, b.real),
            dual: self.dual * a.real + self.real * a.dual + b.dual,
        }
    }

    /// Reciprocal: `1/x`. Derivative: `-1/x^2`.
    #[inline]
    pub fn recip(self) -> Self {
        Self {
            real: self.real.recip(),
            dual: -self.dual / (self.real * self.real),
        }
    }

    /// Floor function (derivative is zero almost everywhere).
    #[inline]
    pub fn floor(self) -> Self {
        Self {
            real: self.real.floor(),
            dual: T::zero(),
        }
    }

    /// Ceiling function (derivative is zero almost everywhere).
    #[inline]
    pub fn ceil(self) -> Self {
        Self {
            real: self.real.ceil(),
            dual: T::zero(),
        }
    }

    /// Round to nearest integer (derivative is zero almost everywhere).
    #[inline]
    pub fn round(self) -> Self {
        Self {
            real: self.real.round(),
            dual: T::zero(),
        }
    }

    /// Truncate to integer part (derivative is zero almost everywhere).
    #[inline]
    pub fn trunc(self) -> Self {
        Self {
            real: self.real.trunc(),
            dual: T::zero(),
        }
    }

    /// Fractional part. Derivative: `1` (same as `x - floor(x)`).
    #[inline]
    pub fn fract(self) -> Self {
        Self {
            real: self.real.fract(),
            dual: self.dual,
        }
    }

    /// Convert radians to degrees.
    #[inline]
    pub fn to_degrees(self) -> Self {
        let factor = T::from(180.0).unwrap() / T::from(std::f64::consts::PI).unwrap();
        Self {
            real: self.real.to_degrees(),
            dual: self.dual * factor,
        }
    }

    /// Convert degrees to radians.
    #[inline]
    pub fn to_radians(self) -> Self {
        let factor = T::from(std::f64::consts::PI).unwrap() / T::from(180.0).unwrap();
        Self {
            real: self.real.to_radians(),
            dual: self.dual * factor,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions mirroring DualNumber methods
//
// The previous hand-written list of 15 one-line wrappers is now
// generated by the local `unary_free_fn!` macro (#541). DualNumber
// already implements `num_traits::Float`, so these are just
// ergonomic fallbacks for call-sites that prefer `sin(x)` to
// `x.sin()`. `atan2` remains a hand-written two-argument function.
// ---------------------------------------------------------------------------

macro_rules! unary_free_fn {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[inline]
        pub fn $name<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
            x.$name()
        }
    };
}

unary_free_fn!(
    /// Sine of a dual number. Derivative: `cos(x)`.
    sin
);
unary_free_fn!(
    /// Cosine of a dual number. Derivative: `-sin(x)`.
    cos
);
unary_free_fn!(
    /// Tangent of a dual number. Derivative: `1/cos^2(x)`.
    tan
);
unary_free_fn!(
    /// Exponential of a dual number. Derivative: `exp(x)`.
    exp
);
unary_free_fn!(
    /// Natural logarithm of a dual number. Derivative: `1/x`.
    ln
);
unary_free_fn!(
    /// Base-2 logarithm of a dual number. Derivative: `1/(x * ln(2))`.
    log2
);
unary_free_fn!(
    /// Base-10 logarithm of a dual number. Derivative: `1/(x * ln(10))`.
    log10
);
unary_free_fn!(
    /// Square root of a dual number. Derivative: `1/(2*sqrt(x))`.
    sqrt
);
unary_free_fn!(
    /// Absolute value of a dual number. Derivative: `signum(x)`.
    abs
);
unary_free_fn!(
    /// Hyperbolic sine. Derivative: `cosh(x)`.
    sinh
);
unary_free_fn!(
    /// Hyperbolic cosine. Derivative: `sinh(x)`.
    cosh
);
unary_free_fn!(
    /// Hyperbolic tangent. Derivative: `1 - tanh^2(x)`.
    tanh
);
unary_free_fn!(
    /// Inverse sine. Derivative: `1/sqrt(1 - x^2)`.
    asin
);
unary_free_fn!(
    /// Inverse cosine. Derivative: `-1/sqrt(1 - x^2)`.
    acos
);
unary_free_fn!(
    /// Inverse tangent. Derivative: `1/(1 + x^2)`.
    atan
);

/// Two-argument inverse tangent.
#[inline]
pub fn atan2<T: Float>(y: DualNumber<T>, x: DualNumber<T>) -> DualNumber<T> {
    y.atan2(x)
}
