//! Core [`DualNumber`] type, constructors, Display, arithmetic
//! operators, and `PartialOrd`. Differentiable methods live in
//! `functions.rs`; `num_traits` impls in `num_impls.rs`; the
//! `derivative` / `gradient` / `jacobian` convenience API in `api.rs`.

use num_traits::Float;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign,
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
// Arithmetic: scalar T op DualNumber (#184)
//
// Without these impls, `2.0 * x` doesn't compile when `x: DualNumber`
// because only `DualNumber * T` (not `T * DualNumber`) is defined.
// Rust's orphan rules allow us to implement `Add<DualNumber<T>> for T`
// etc. only for concrete T; we can't do a blanket impl. But Float
// covers f32 and f64 which are the types users actually write. For
// the non-commutative Sub and Div, the derivative flips sign / uses
// the quotient rule respectively.
// ---------------------------------------------------------------------------

/// Newtype macro for T op DualNumber<T> where op is commutative.
macro_rules! impl_scalar_lhs {
    (Add, add) => {
        impl<T: Float> Add<DualNumber<T>> for f64 {
            type Output = DualNumber<T>;
            #[inline]
            fn add(self, rhs: DualNumber<T>) -> DualNumber<T> {
                DualNumber {
                    real: T::from(self).unwrap() + rhs.real,
                    dual: rhs.dual,
                }
            }
        }
        impl<T: Float> Add<DualNumber<T>> for f32 {
            type Output = DualNumber<T>;
            #[inline]
            fn add(self, rhs: DualNumber<T>) -> DualNumber<T> {
                DualNumber {
                    real: T::from(self).unwrap() + rhs.real,
                    dual: rhs.dual,
                }
            }
        }
    };
    (Mul, mul) => {
        impl<T: Float> Mul<DualNumber<T>> for f64 {
            type Output = DualNumber<T>;
            #[inline]
            fn mul(self, rhs: DualNumber<T>) -> DualNumber<T> {
                let s = T::from(self).unwrap();
                DualNumber {
                    real: s * rhs.real,
                    dual: s * rhs.dual,
                }
            }
        }
        impl<T: Float> Mul<DualNumber<T>> for f32 {
            type Output = DualNumber<T>;
            #[inline]
            fn mul(self, rhs: DualNumber<T>) -> DualNumber<T> {
                let s = T::from(self).unwrap();
                DualNumber {
                    real: s * rhs.real,
                    dual: s * rhs.dual,
                }
            }
        }
    };
    (Sub, sub) => {
        impl<T: Float> Sub<DualNumber<T>> for f64 {
            type Output = DualNumber<T>;
            #[inline]
            fn sub(self, rhs: DualNumber<T>) -> DualNumber<T> {
                DualNumber {
                    real: T::from(self).unwrap() - rhs.real,
                    dual: -rhs.dual,
                }
            }
        }
        impl<T: Float> Sub<DualNumber<T>> for f32 {
            type Output = DualNumber<T>;
            #[inline]
            fn sub(self, rhs: DualNumber<T>) -> DualNumber<T> {
                DualNumber {
                    real: T::from(self).unwrap() - rhs.real,
                    dual: -rhs.dual,
                }
            }
        }
    };
    (Div, div) => {
        impl<T: Float> Div<DualNumber<T>> for f64 {
            type Output = DualNumber<T>;
            #[inline]
            fn div(self, rhs: DualNumber<T>) -> DualNumber<T> {
                let s = T::from(self).unwrap();
                let r2 = rhs.real * rhs.real;
                DualNumber {
                    real: s / rhs.real,
                    dual: -s * rhs.dual / r2,
                }
            }
        }
        impl<T: Float> Div<DualNumber<T>> for f32 {
            type Output = DualNumber<T>;
            #[inline]
            fn div(self, rhs: DualNumber<T>) -> DualNumber<T> {
                let s = T::from(self).unwrap();
                let r2 = rhs.real * rhs.real;
                DualNumber {
                    real: s / rhs.real,
                    dual: -s * rhs.dual / r2,
                }
            }
        }
    };
}

impl_scalar_lhs!(Add, add);
impl_scalar_lhs!(Mul, mul);
impl_scalar_lhs!(Sub, sub);
impl_scalar_lhs!(Div, div);

// ---------------------------------------------------------------------------
// Compound assignment: DualNumber op= DualNumber / scalar T
//
// Mirrors the existing `Add`/`Sub`/`Mul`/`Div` impls so accumulation
// loops like `sum += f(x_i)` compile without requiring `sum = sum + ...`
// (see issue #542). Each `_assign` method forwards to the pre-existing
// by-value operator to keep the differentiation rules in one place.
// ---------------------------------------------------------------------------

impl<T: Float> AddAssign for DualNumber<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Float> SubAssign for DualNumber<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Float> MulAssign for DualNumber<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Float> DivAssign for DualNumber<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Float> AddAssign<T> for DualNumber<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: Float> SubAssign<T> for DualNumber<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<T: Float> MulAssign<T> for DualNumber<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: Float> DivAssign<T> for DualNumber<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
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
// Iterator traits: Sum / Product (#302)
// ---------------------------------------------------------------------------

impl<T: Float> std::iter::Sum for DualNumber<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(T::zero()), |acc, x| acc + x)
    }
}

impl<T: Float> std::iter::Product for DualNumber<T> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(T::one()), |acc, x| acc * x)
    }
}
