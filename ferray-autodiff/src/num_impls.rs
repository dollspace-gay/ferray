//! `num_traits` implementations for [`DualNumber`]:
//! `Zero`, `One`, `ToPrimitive`, `FromPrimitive`, `NumCast`, `Num`,
//! and the big `Float` trait.
//!
//! These are kept separate from the differentiable methods so the
//! top-level `dual.rs` + `functions.rs` modules stay readable — this
//! file is almost entirely glue.

use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::num::FpCategory;

use crate::dual::DualNumber;

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
        Self::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        Self::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        Self::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        Self::trunc(self)
    }

    #[inline]
    fn fract(self) -> Self {
        Self::fract(self)
    }

    #[inline]
    fn abs(self) -> Self {
        Self::abs(self)
    }

    #[inline]
    fn signum(self) -> Self {
        Self::signum(self)
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
        Self::mul_add(self, a, b)
    }

    #[inline]
    fn recip(self) -> Self {
        Self::recip(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::powi(self, n)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Self::powf(self, n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        Self::exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        Self::exp2(self)
    }

    #[inline]
    fn ln(self) -> Self {
        Self::ln(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        Self::log(self, base)
    }

    #[inline]
    fn log2(self) -> Self {
        Self::log2(self)
    }

    #[inline]
    fn log10(self) -> Self {
        Self::log10(self)
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
        Self::cbrt(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::hypot(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        Self::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        Self::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        Self::tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        Self::asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        Self::acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        Self::atan(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::atan2(self, other)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        Self::sin_cos(self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::exp_m1(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self::ln_1p(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        Self::sinh(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        Self::cosh(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        Self::tanh(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        Self::asinh(self)
    }

    #[inline]
    fn acosh(self) -> Self {
        Self::acosh(self)
    }

    #[inline]
    fn atanh(self) -> Self {
        Self::atanh(self)
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.real.integer_decode()
    }

    #[inline]
    fn to_degrees(self) -> Self {
        Self::to_degrees(self)
    }

    #[inline]
    fn to_radians(self) -> Self {
        Self::to_radians(self)
    }
}
