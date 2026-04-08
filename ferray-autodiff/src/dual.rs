//! Core [`DualNumber`] type, constructors, Display, arithmetic
//! operators, and `PartialOrd`. Differentiable methods live in
//! `functions.rs`; `num_traits` impls in `num_impls.rs`; the
//! `derivative` / `gradient` / `jacobian` convenience API in `api.rs`.

use num_traits::Float;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

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
// PartialOrd (based on real part only)
// ---------------------------------------------------------------------------

impl<T: Float> PartialOrd for DualNumber<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}
