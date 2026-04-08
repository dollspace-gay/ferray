//! Forward-mode automatic differentiation for ferray.
//!
//! This crate provides a [`DualNumber`] type that tracks derivatives through
//! computations using the dual number algebra: `(a + b*eps)` where `eps^2 = 0`.
//!
//! # Quick Start
//!
//! ```
//! use ferray_autodiff::{DualNumber, derivative};
//!
//! // Compute d/dx sin(x) at x = 0
//! let d = derivative(|x| x.sin(), 0.0_f64);
//! assert!((d - 1.0).abs() < 1e-10);
//! ```

use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::fmt;
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Array-aware forward-mode AD primitives (derivative_elementwise,
/// gradient_vector, jacobian) that accept `ferray_core::Array` inputs.
/// See the module docstring for the rationale (Element sealing
/// prevents storing DualNumber in a ferray Array directly).
pub mod array_ops;

pub use array_ops::{
    derivative_elementwise, gradient_vector, jacobian_array, value_and_derivative_elementwise,
    value_and_gradient,
};

/// A dual number for forward-mode automatic differentiation.
///
/// `DualNumber { real: a, dual: b }` represents `a + b*eps` where `eps^2 = 0`.
/// The dual part tracks the derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualNumber<T> {
    /// The primal (real) value.
    pub real: T,
    /// The dual (derivative) part.
    pub dual: T,
}

impl<T: Float> DualNumber<T> {
    /// Creates a new dual number with the given real and dual parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferray_autodiff::DualNumber;
    /// let d = DualNumber::new(3.0_f64, 1.0);
    /// assert_eq!(d.real, 3.0);
    /// assert_eq!(d.dual, 1.0);
    /// ```
    #[inline]
    pub fn new(real: T, dual: T) -> Self {
        Self { real, dual }
    }

    /// Creates a constant dual number (dual part is zero).
    ///
    /// Use this for values that are not being differentiated with respect to.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferray_autodiff::DualNumber;
    /// let c = DualNumber::constant(5.0_f64);
    /// assert_eq!(c.real, 5.0);
    /// assert_eq!(c.dual, 0.0);
    /// ```
    #[inline]
    pub fn constant(real: T) -> Self {
        Self {
            real,
            dual: T::zero(),
        }
    }

    /// Creates a variable dual number (dual part is one).
    ///
    /// Use this to seed the variable you are differentiating with respect to.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferray_autodiff::DualNumber;
    /// let x = DualNumber::variable(2.0_f64);
    /// assert_eq!(x.real, 2.0);
    /// assert_eq!(x.dual, 1.0);
    /// ```
    #[inline]
    pub fn variable(real: T) -> Self {
        Self {
            real,
            dual: T::one(),
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for DualNumber<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}*eps)", self.real, self.dual)
    }
}

// ---------------------------------------------------------------------------
// Arithmetic: DualNumber op DualNumber
// ---------------------------------------------------------------------------

impl<T: Float> Add for DualNumber<T> {
    type Output = Self;

    /// `(a + b*eps) + (c + d*eps) = (a+c) + (b+d)*eps`
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl<T: Float> Sub for DualNumber<T> {
    type Output = Self;

    /// `(a + b*eps) - (c + d*eps) = (a-c) + (b-d)*eps`
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl<T: Float> Mul for DualNumber<T> {
    type Output = Self;

    /// `(a + b*eps) * (c + d*eps) = a*c + (a*d + b*c)*eps`
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl<T: Float> Div for DualNumber<T> {
    type Output = Self;

    /// `(a + b*eps) / (c + d*eps) = a/c + (b*c - a*d) / c^2 * eps`
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let c2 = rhs.real * rhs.real;
        Self {
            real: self.real / rhs.real,
            dual: (self.dual * rhs.real - self.real * rhs.dual) / c2,
        }
    }
}

impl<T: Float> Rem for DualNumber<T> {
    type Output = Self;

    /// Remainder: `a % b`. The derivative is `1` when `a/b` has no integer part change,
    /// and `0` for the rhs derivative component, following the identity `a % b = a - b * floor(a/b)`.
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        // a % b = a - b * floor(a / b)
        // derivative: da - db * floor(a/b)  (floor is locally constant)
        let q = (self.real / rhs.real).floor();
        Self {
            real: self.real % rhs.real,
            dual: self.dual - rhs.dual * q,
        }
    }
}

impl<T: Float> Neg for DualNumber<T> {
    type Output = Self;

    /// `-(a + b*eps) = -a + (-b)*eps`
    #[inline]
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic: DualNumber op scalar T
// ---------------------------------------------------------------------------

impl<T: Float> Add<T> for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self {
        Self {
            real: self.real + rhs,
            dual: self.dual,
        }
    }
}

impl<T: Float> Sub<T> for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self {
        Self {
            real: self.real - rhs,
            dual: self.dual,
        }
    }
}

impl<T: Float> Mul<T> for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self {
        Self {
            real: self.real * rhs,
            dual: self.dual * rhs,
        }
    }
}

impl<T: Float> Div<T> for DualNumber<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self {
        Self {
            real: self.real / rhs,
            dual: self.dual / rhs,
        }
    }
}

impl<T: Float> Rem<T> for DualNumber<T> {
    type Output = Self;

    /// Remainder with a scalar constant: `x % c`.
    /// Derivative: `d/dx (x % c) = 1` (almost everywhere, since `x % c = x - c * floor(x/c)`
    /// and `floor` is locally constant).
    #[inline]
    fn rem(self, rhs: T) -> Self {
        Self {
            real: self.real % rhs,
            dual: self.dual,
        }
    }
}

// ---------------------------------------------------------------------------
// Differentiable methods on DualNumber<T: Float>
// ---------------------------------------------------------------------------

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

    /// Natural logarithm (alias for `ln`). Derivative: `1/x`.
    #[inline]
    pub fn log(self, base: Self) -> Self {
        // log_b(x) = ln(x) / ln(b)
        // d/dx: 1/(x * ln(b))   for x part
        // d/db: -ln(x)/(b * ln(b)^2) for base part
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
// ---------------------------------------------------------------------------

/// Sine of a dual number. Derivative: `cos(x)`.
#[inline]
pub fn sin<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.sin()
}

/// Cosine of a dual number. Derivative: `-sin(x)`.
#[inline]
pub fn cos<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.cos()
}

/// Tangent of a dual number. Derivative: `1/cos^2(x)`.
#[inline]
pub fn tan<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.tan()
}

/// Exponential of a dual number. Derivative: `exp(x)`.
#[inline]
pub fn exp<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.exp()
}

/// Natural logarithm of a dual number. Derivative: `1/x`.
#[inline]
pub fn ln<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.ln()
}

/// Base-2 logarithm of a dual number. Derivative: `1/(x * ln(2))`.
#[inline]
pub fn log2<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.log2()
}

/// Base-10 logarithm of a dual number. Derivative: `1/(x * ln(10))`.
#[inline]
pub fn log10<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.log10()
}

/// Square root of a dual number. Derivative: `1/(2*sqrt(x))`.
#[inline]
pub fn sqrt<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.sqrt()
}

/// Absolute value of a dual number. Derivative: `signum(x)`.
#[inline]
pub fn abs<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.abs()
}

/// Hyperbolic sine. Derivative: `cosh(x)`.
#[inline]
pub fn sinh<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.sinh()
}

/// Hyperbolic cosine. Derivative: `sinh(x)`.
#[inline]
pub fn cosh<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.cosh()
}

/// Hyperbolic tangent. Derivative: `1 - tanh^2(x)`.
#[inline]
pub fn tanh<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.tanh()
}

/// Inverse sine. Derivative: `1/sqrt(1 - x^2)`.
#[inline]
pub fn asin<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.asin()
}

/// Inverse cosine. Derivative: `-1/sqrt(1 - x^2)`.
#[inline]
pub fn acos<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.acos()
}

/// Inverse tangent. Derivative: `1/(1 + x^2)`.
#[inline]
pub fn atan<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x.atan()
}

/// Two-argument inverse tangent.
#[inline]
pub fn atan2<T: Float>(y: DualNumber<T>, x: DualNumber<T>) -> DualNumber<T> {
    y.atan2(x)
}

// ---------------------------------------------------------------------------
// num_traits::Zero
// ---------------------------------------------------------------------------

impl<T: Float> Zero for DualNumber<T> {
    #[inline]
    fn zero() -> Self {
        Self {
            real: T::zero(),
            dual: T::zero(),
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

// ---------------------------------------------------------------------------
// num_traits::One
// ---------------------------------------------------------------------------

impl<T: Float> One for DualNumber<T> {
    #[inline]
    fn one() -> Self {
        Self {
            real: T::one(),
            dual: T::zero(),
        }
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.real.is_one() && self.dual.is_zero()
    }
}

// ---------------------------------------------------------------------------
// num_traits::ToPrimitive (delegates to real part)
// ---------------------------------------------------------------------------

impl<T: Float + ToPrimitive> ToPrimitive for DualNumber<T> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.real.to_i64()
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.real.to_u64()
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.real.to_f32()
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.real.to_f64()
    }
}

// ---------------------------------------------------------------------------
// num_traits::FromPrimitive
// ---------------------------------------------------------------------------

impl<T: Float + FromPrimitive> FromPrimitive for DualNumber<T> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(Self::constant)
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(Self::constant)
    }

    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        T::from_f32(n).map(Self::constant)
    }

    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        T::from_f64(n).map(Self::constant)
    }
}

// ---------------------------------------------------------------------------
// num_traits::NumCast
// ---------------------------------------------------------------------------

impl<T: Float> NumCast for DualNumber<T> {
    #[inline]
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        T::from(n).map(Self::constant)
    }
}

// ---------------------------------------------------------------------------
// num_traits::Num
// ---------------------------------------------------------------------------

impl<T: Float> Num for DualNumber<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(Self::constant)
    }
}

// ---------------------------------------------------------------------------
// num_traits::Float
// ---------------------------------------------------------------------------

impl<T: Float> Float for DualNumber<T> {
    #[inline]
    fn nan() -> Self {
        Self::constant(T::nan())
    }

    #[inline]
    fn infinity() -> Self {
        Self::constant(T::infinity())
    }

    #[inline]
    fn neg_infinity() -> Self {
        Self::constant(T::neg_infinity())
    }

    #[inline]
    fn neg_zero() -> Self {
        Self::constant(T::neg_zero())
    }

    #[inline]
    fn min_value() -> Self {
        Self::constant(T::min_value())
    }

    #[inline]
    fn min_positive_value() -> Self {
        Self::constant(T::min_positive_value())
    }

    #[inline]
    fn epsilon() -> Self {
        Self::constant(T::epsilon())
    }

    #[inline]
    fn max_value() -> Self {
        Self::constant(T::max_value())
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.real.is_nan()
    }

    #[inline]
    fn is_infinite(self) -> bool {
        self.real.is_infinite()
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.real.is_finite()
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.real.is_normal()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.real.classify()
    }

    #[inline]
    fn floor(self) -> Self {
        DualNumber::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        DualNumber::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        DualNumber::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        DualNumber::trunc(self)
    }

    #[inline]
    fn fract(self) -> Self {
        DualNumber::fract(self)
    }

    #[inline]
    fn abs(self) -> Self {
        DualNumber::abs(self)
    }

    #[inline]
    fn signum(self) -> Self {
        DualNumber::signum(self)
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        self.real.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        self.real.is_sign_negative()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        DualNumber::mul_add(self, a, b)
    }

    #[inline]
    fn recip(self) -> Self {
        DualNumber::recip(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        DualNumber::powi(self, n)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        DualNumber::powf(self, n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        DualNumber::sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        DualNumber::exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        DualNumber::exp2(self)
    }

    #[inline]
    fn ln(self) -> Self {
        DualNumber::ln(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        DualNumber::log(self, base)
    }

    #[inline]
    fn log2(self) -> Self {
        DualNumber::log2(self)
    }

    #[inline]
    fn log10(self) -> Self {
        DualNumber::log10(self)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.real >= other.real { self } else { other }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.real <= other.real { self } else { other }
    }

    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        if self.real > other.real {
            self - other
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn cbrt(self) -> Self {
        DualNumber::cbrt(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        DualNumber::hypot(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        DualNumber::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        DualNumber::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        DualNumber::tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        DualNumber::asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        DualNumber::acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        DualNumber::atan(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        DualNumber::atan2(self, other)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        DualNumber::sin_cos(self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        DualNumber::exp_m1(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        DualNumber::ln_1p(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        DualNumber::sinh(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        DualNumber::cosh(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        DualNumber::tanh(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        DualNumber::asinh(self)
    }

    #[inline]
    fn acosh(self) -> Self {
        DualNumber::acosh(self)
    }

    #[inline]
    fn atanh(self) -> Self {
        DualNumber::atanh(self)
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.real.integer_decode()
    }

    #[inline]
    fn to_degrees(self) -> Self {
        DualNumber::to_degrees(self)
    }

    #[inline]
    fn to_radians(self) -> Self {
        DualNumber::to_radians(self)
    }
}

// ---------------------------------------------------------------------------
// PartialOrd (based on real part only)
// ---------------------------------------------------------------------------

impl<T: Float> PartialOrd for DualNumber<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Compute the derivative of a univariate function at a point.
///
/// Seeds the input as a variable (dual part = 1), evaluates the function,
/// and extracts the dual part of the result, which is the derivative.
///
/// # Examples
///
/// ```
/// use ferray_autodiff::derivative;
///
/// // d/dx sin(x) at x=0 is cos(0) = 1
/// let d = derivative(|x| x.sin(), 0.0_f64);
/// assert!((d - 1.0).abs() < 1e-10);
///
/// // d/dx x^3 at x=2 is 3*4 = 12
/// let d = derivative(|x| x.powi(3), 2.0_f64);
/// assert!((d - 12.0).abs() < 1e-10);
/// ```
pub fn derivative<T: Float>(f: impl Fn(DualNumber<T>) -> DualNumber<T>, x: T) -> T {
    f(DualNumber::variable(x)).dual
}

/// Compute the gradient of a multivariate function at a point.
///
/// For each component of the input, seeds that component as a variable
/// (dual = 1) while keeping all others as constants (dual = 0), then
/// evaluates the function and collects the partial derivatives.
///
/// # Examples
///
/// ```
/// use ferray_autodiff::{gradient, DualNumber};
///
/// // f(x, y) = x^2 + y^2
/// // grad f = (2x, 2y)
/// let g = gradient(
///     |v| v[0] * v[0] + v[1] * v[1],
///     &[3.0_f64, 4.0],
/// );
/// assert!((g[0] - 6.0).abs() < 1e-10);
/// assert!((g[1] - 8.0).abs() < 1e-10);
/// ```
pub fn gradient<T: Float>(f: impl Fn(&[DualNumber<T>]) -> DualNumber<T>, point: &[T]) -> Vec<T> {
    let n = point.len();
    let mut grad = Vec::with_capacity(n);

    // Pre-allocate a single buffer of constants; toggle one variable at a time.
    let mut args: Vec<DualNumber<T>> = point.iter().map(|&val| DualNumber::constant(val)).collect();

    for i in 0..n {
        // Seed the i-th component as the variable
        args[i].dual = T::one();
        grad.push(f(&args).dual);
        // Reset back to constant
        args[i].dual = T::zero();
    }
    grad
}

/// Compute the Jacobian matrix of a vector-valued function at a point.
///
/// For `f: R^n -> R^m`, returns an `m x n` matrix (as a flat `Vec<T>` in
/// row-major order) where `J[i*n + j] = df_i/dx_j`.
///
/// Uses forward-mode AD: each evaluation with the j-th input seeded
/// produces column j of the Jacobian (all m partial derivatives with
/// respect to x_j).
///
/// # Examples
///
/// ```
/// use ferray_autodiff::{jacobian, DualNumber};
///
/// // f(x, y) = (x*y, x + y)
/// // J = [[y, x], [1, 1]]
/// let (jac, m) = jacobian(
///     |v| vec![v[0] * v[1], v[0] + v[1]],
///     &[3.0_f64, 4.0],
/// );
/// assert_eq!(m, 2); // 2 outputs
/// assert!((jac[0] - 4.0).abs() < 1e-10); // df0/dx = y = 4
/// assert!((jac[1] - 3.0).abs() < 1e-10); // df0/dy = x = 3
/// assert!((jac[2] - 1.0).abs() < 1e-10); // df1/dx = 1
/// assert!((jac[3] - 1.0).abs() < 1e-10); // df1/dy = 1
/// ```
pub fn jacobian<T: Float>(
    f: impl Fn(&[DualNumber<T>]) -> Vec<DualNumber<T>>,
    point: &[T],
) -> (Vec<T>, usize) {
    let n = point.len();
    let mut args: Vec<DualNumber<T>> = point.iter().map(|&val| DualNumber::constant(val)).collect();

    // First evaluation to determine output dimension m
    let first_out = f(&args);
    let m = first_out.len();

    // Jacobian stored row-major: J[i*n + j] = df_i/dx_j
    let mut jac = vec![T::zero(); m * n];

    for j in 0..n {
        args[j].dual = T::one();
        let out = f(&args);
        for i in 0..m {
            jac[i * n + j] = out[i].dual;
        }
        args[j].dual = T::zero();
    }

    (jac, m)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        let d = derivative(|v| v.sin(), x);
        assert!(approx_eq(d, x.cos()));
    }

    #[test]
    fn test_cos_derivative() {
        // d/dx cos(x) = -sin(x)
        let x = 0.5_f64;
        let d = derivative(|v| v.cos(), x);
        assert!(approx_eq(d, -x.sin()));
    }

    #[test]
    fn test_tan_derivative() {
        // d/dx tan(x) = 1/cos^2(x)
        let x = 0.5_f64;
        let d = derivative(|v| v.tan(), x);
        let expected = 1.0 / (x.cos() * x.cos());
        assert!(approx_eq(d, expected));
    }

    // --- Exponential / logarithm derivatives ---

    #[test]
    fn test_exp_derivative() {
        // d/dx exp(x) = exp(x)
        let x = 1.0_f64;
        let d = derivative(|v| v.exp(), x);
        assert!(approx_eq(d, x.exp()));
    }

    #[test]
    fn test_exp_at_zero() {
        let d = derivative(|v| v.exp(), 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_ln_derivative() {
        // d/dx ln(x) = 1/x
        let x = 2.0_f64;
        let d = derivative(|v| v.ln(), x);
        assert!(approx_eq(d, 1.0 / x));
    }

    #[test]
    fn test_log2_derivative() {
        // d/dx log2(x) = 1/(x * ln(2))
        let x = 3.0_f64;
        let d = derivative(|v| v.log2(), x);
        assert!(approx_eq(d, 1.0 / (x * 2.0_f64.ln())));
    }

    #[test]
    fn test_log10_derivative() {
        // d/dx log10(x) = 1/(x * ln(10))
        let x = 5.0_f64;
        let d = derivative(|v| v.log10(), x);
        assert!(approx_eq(d, 1.0 / (x * 10.0_f64.ln())));
    }

    // --- Power / root derivatives ---

    #[test]
    fn test_sqrt_derivative() {
        // d/dx sqrt(x) = 1/(2*sqrt(x))
        let x = 4.0_f64;
        let d = derivative(|v| v.sqrt(), x);
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
        let expected = 2.0_f64.powf(p) * 2.0_f64.ln();
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
        let d = derivative(|v| v.abs(), 3.0_f64);
        assert!(approx_eq(d, 1.0));

        let d = derivative(|v| v.abs(), -3.0_f64);
        assert!(approx_eq(d, -1.0));
    }

    // --- Mathematical singularities ---
    // These document the behavior at points where derivatives are
    // infinite or undefined. The results are NaN or Inf, which is
    // correct — the derivative genuinely doesn't exist or is unbounded.

    #[test]
    fn test_sqrt_at_zero_singular() {
        // d/dx sqrt(x) = 1/(2*sqrt(x)), which is +inf at x=0
        let d = derivative(|v| v.sqrt(), 0.0_f64);
        assert!(d.is_infinite() && d.is_sign_positive());
    }

    #[test]
    fn test_ln_at_zero_singular() {
        // d/dx ln(x) = 1/x, which is +inf at x=0+
        let d = derivative(|v| v.ln(), 0.0_f64);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_recip_at_zero_singular() {
        // d/dx 1/x = -1/x^2, which is -inf at x=0
        let d = derivative(|v| v.recip(), 0.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_asin_at_boundary() {
        // d/dx asin(x) = 1/sqrt(1-x^2), which is +inf at x=1
        let d = derivative(|v| v.asin(), 1.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_acosh_at_boundary() {
        // d/dx acosh(x) = 1/sqrt(x^2-1), which is +inf at x=1
        let d = derivative(|v| v.acosh(), 1.0_f64);
        assert!(d.is_infinite() || d.is_nan());
    }

    #[test]
    fn test_atanh_at_boundary() {
        // d/dx atanh(x) = 1/(1-x^2), which is +inf at x=1
        let d = derivative(|v| v.atanh(), 1.0_f64);
        assert!(d.is_infinite());
    }

    // --- Hyperbolic derivatives ---

    #[test]
    fn test_sinh_derivative() {
        // d/dx sinh(x) = cosh(x)
        let x = 1.0_f64;
        let d = derivative(|v| v.sinh(), x);
        assert!(approx_eq(d, x.cosh()));
    }

    #[test]
    fn test_cosh_derivative() {
        // d/dx cosh(x) = sinh(x)
        let x = 1.0_f64;
        let d = derivative(|v| v.cosh(), x);
        assert!(approx_eq(d, x.sinh()));
    }

    #[test]
    fn test_tanh_derivative() {
        // d/dx tanh(x) = 1 - tanh^2(x)
        let x = 0.5_f64;
        let d = derivative(|v| v.tanh(), x);
        let t = x.tanh();
        assert!(approx_eq(d, 1.0 - t * t));
    }

    // --- Inverse trig derivatives ---

    #[test]
    fn test_asin_derivative() {
        // d/dx asin(x) = 1/sqrt(1 - x^2)
        let x = 0.5_f64;
        let d = derivative(|v| v.asin(), x);
        assert!(approx_eq(d, 1.0 / (1.0 - x * x).sqrt()));
    }

    #[test]
    fn test_acos_derivative() {
        // d/dx acos(x) = -1/sqrt(1 - x^2)
        let x = 0.5_f64;
        let d = derivative(|v| v.acos(), x);
        assert!(approx_eq(d, -1.0 / (1.0 - x * x).sqrt()));
    }

    #[test]
    fn test_atan_derivative() {
        // d/dx atan(x) = 1/(1 + x^2)
        let x = 1.0_f64;
        let d = derivative(|v| v.atan(), x);
        assert!(approx_eq(d, 1.0 / (1.0 + x * x)));
    }

    #[test]
    fn test_atan2_derivative() {
        // atan2(y, x): d/dy = x / (x^2 + y^2)
        let y = 3.0_f64;
        let x_val = 4.0_f64;
        let denom = x_val * x_val + y * y;

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
        let d = derivative(|x| x.sin(), 0.0_f64);
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_derivative_exp_at_zero() {
        let d = derivative(|x| x.exp(), 0.0_f64);
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
        let (jac, m) = jacobian(
            |v| vec![v[0] * v[1], v[0] + v[1]],
            &[3.0_f64, 4.0],
        );
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
        let d = derivative(|v| v.recip(), x);
        assert!(approx_eq(d, -1.0 / (x * x)));
    }

    #[test]
    fn test_cbrt_derivative() {
        // d/dx cbrt(x) = 1/(3 * x^(2/3))
        let x = 8.0_f64;
        let d = derivative(|v| v.cbrt(), x);
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
        let d = derivative(|v| v.exp2(), x);
        let expected = x.exp2() * 2.0_f64.ln();
        assert!(approx_eq(d, expected));
    }

    #[test]
    fn test_exp_m1_derivative() {
        // d/dx (exp(x) - 1) = exp(x)
        let x = 0.5_f64;
        let d = derivative(|v| v.exp_m1(), x);
        assert!(approx_eq(d, x.exp()));
    }

    #[test]
    fn test_ln_1p_derivative() {
        // d/dx ln(1+x) = 1/(1+x)
        let x = 0.5_f64;
        let d = derivative(|v| v.ln_1p(), x);
        assert!(approx_eq(d, 1.0 / (1.0 + x)));
    }

    #[test]
    fn test_asinh_derivative() {
        // d/dx asinh(x) = 1/sqrt(x^2 + 1)
        let x = 1.0_f64;
        let d = derivative(|v| v.asinh(), x);
        assert!(approx_eq(d, 1.0 / (x * x + 1.0).sqrt()));
    }

    #[test]
    fn test_acosh_derivative() {
        // d/dx acosh(x) = 1/sqrt(x^2 - 1), x > 1
        let x = 2.0_f64;
        let d = derivative(|v| v.acosh(), x);
        assert!(approx_eq(d, 1.0 / (x * x - 1.0).sqrt()));
    }

    #[test]
    fn test_atanh_derivative() {
        // d/dx atanh(x) = 1/(1 - x^2), |x| < 1
        let x = 0.5_f64;
        let d = derivative(|v| v.atanh(), x);
        assert!(approx_eq(d, 1.0 / (1.0 - x * x)));
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
        let expected = 2.0 * x * x.sin() + x * x * x.cos() + x.exp();
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
    use super::*;

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
        check_ad_vs_fd(|x| x.sin(), f64::sin, 1.0, 1e-8);
    }

    #[test]
    fn fd_cos() {
        check_ad_vs_fd(|x| x.cos(), f64::cos, 1.0, 1e-8);
    }

    #[test]
    fn fd_exp() {
        check_ad_vs_fd(|x| x.exp(), f64::exp, 1.0, 1e-8);
    }

    #[test]
    fn fd_ln() {
        check_ad_vs_fd(|x| x.ln(), f64::ln, 2.0, 1e-8);
    }

    #[test]
    fn fd_sqrt() {
        check_ad_vs_fd(|x| x.sqrt(), f64::sqrt, 4.0, 1e-8);
    }

    #[test]
    fn fd_tanh() {
        check_ad_vs_fd(|x| x.tanh(), f64::tanh, 0.5, 1e-8);
    }

    #[test]
    fn fd_asin() {
        check_ad_vs_fd(|x| x.asin(), f64::asin, 0.5, 1e-8);
    }

    #[test]
    fn fd_complex_expression() {
        // f(x) = x^2 * sin(x) + exp(-x)
        let ad_fn = |x: DualNumber<f64>| x * x * x.sin() + (-x).exp();
        let scalar_fn = |x: f64| x * x * x.sin() + (-x).exp();
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
        let f = |x: f64, y: f64| x * x * y + (x * y).sin();

        let fd_dx = (f(1.0 + h, 2.0) - f(1.0 - h, 2.0)) / (2.0 * h);
        let fd_dy = (f(1.0, 2.0 + h) - f(1.0, 2.0 - h)) / (2.0 * h);

        assert!((g[0] - fd_dx).abs() < 1e-6, "df/dx: AD={}, FD={fd_dx}", g[0]);
        assert!((g[1] - fd_dy).abs() < 1e-6, "df/dy: AD={}, FD={fd_dy}", g[1]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_add_commutative(a_r in -100.0..100.0_f64, a_d in -100.0..100.0_f64,
                                 b_r in -100.0..100.0_f64, b_d in -100.0..100.0_f64) {
            let a = DualNumber::new(a_r, a_d);
            let b = DualNumber::new(b_r, b_d);
            let ab = a + b;
            let ba = b + a;
            prop_assert!((ab.real - ba.real).abs() < 1e-10);
            prop_assert!((ab.dual - ba.dual).abs() < 1e-10);
        }

        #[test]
        fn prop_mul_commutative(a_r in -100.0..100.0_f64, a_d in -100.0..100.0_f64,
                                 b_r in -100.0..100.0_f64, b_d in -100.0..100.0_f64) {
            let a = DualNumber::new(a_r, a_d);
            let b = DualNumber::new(b_r, b_d);
            let ab = a * b;
            let ba = b * a;
            prop_assert!((ab.real - ba.real).abs() < 1e-10);
            prop_assert!((ab.dual - ba.dual).abs() < 1e-10);
        }

        #[test]
        fn prop_derivative_of_constant_is_zero(c in -100.0..100.0_f64) {
            let d = derivative(|_x| DualNumber::constant(c), 1.0_f64);
            prop_assert!((d - 0.0).abs() < 1e-10);
        }

        #[test]
        fn prop_derivative_of_identity_is_one(x in -100.0..100.0_f64) {
            let d = derivative(|v| v, x);
            prop_assert!((d - 1.0).abs() < 1e-10);
        }

        #[test]
        fn prop_derivative_linear(a in -10.0..10.0_f64, b in -10.0..10.0_f64, x in -10.0..10.0_f64) {
            // d/dx (a*x + b) = a
            let d = derivative(|v| v * DualNumber::constant(a) + DualNumber::constant(b), x);
            prop_assert!((d - a).abs() < 1e-10);
        }

        #[test]
        fn prop_product_rule(x in 0.1..10.0_f64) {
            // d/dx [sin(x) * cos(x)] should equal cos(x)*cos(x) - sin(x)*sin(x) = cos(2x)
            let d = derivative(|v| v.sin() * v.cos(), x);
            let expected = (2.0 * x).cos();
            prop_assert!((d - expected).abs() < 1e-8,
                "product rule failed: got {}, expected {}", d, expected);
        }

        #[test]
        fn prop_chain_rule_exp_of_sin(x in -3.0..3.0_f64) {
            // d/dx exp(sin(x)) = exp(sin(x)) * cos(x)
            let d = derivative(|v| v.sin().exp(), x);
            let expected = x.sin().exp() * x.cos();
            prop_assert!((d - expected).abs() < 1e-8,
                "chain rule failed at x={}: got {}, expected {}", x, d, expected);
        }

        #[test]
        fn prop_power_rule(x in 0.5..10.0_f64, n in 1..8_i32) {
            // d/dx x^n = n * x^(n-1)
            let d = derivative(|v| v.powi(n), x);
            let expected = (n as f64) * x.powi(n - 1);
            prop_assert!((d - expected).abs() < 1e-6 * expected.abs().max(1.0),
                "power rule failed: d/dx x^{} at x={}: got {}, expected {}",
                n, x, d, expected);
        }

        #[test]
        fn prop_quotient_rule(x in 0.5..10.0_f64) {
            // d/dx [sin(x)/x] = (x*cos(x) - sin(x)) / x^2
            let d = derivative(|v| v.sin() / v, x);
            let expected = (x * x.cos() - x.sin()) / (x * x);
            prop_assert!((d - expected).abs() < 1e-8,
                "quotient rule failed at x={}: got {}, expected {}", x, d, expected);
        }
    }
}
