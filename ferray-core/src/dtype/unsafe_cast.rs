// ferray-core: Unsafe (lossy) type casting
//
// Provides the `CastTo<U>` trait — a superset of `PromoteTo<U>` that also
// covers narrowing and kind-changing conversions like f64 -> i32, Complex -> f64,
// f32 -> u8, etc. The conversion semantics match Rust's `as` operator
// (which is what NumPy uses for `arr.astype(dtype, casting='unsafe')`).
//
// Unlike PromoteTo (which only allows information-preserving widening), CastTo
// is implemented for every (T, U) pair of supported Element types. The caller
// is responsible for checking via `can_cast(from, to, CastKind)` whether the
// cast is safe at the desired safety level.
//
// See: https://github.com/dollspace-gay/ferray/issues/361

use num_complex::Complex;

use crate::dtype::Element;

/// Trait for performing an unsafe (lossy) cast from `Self` to `U`.
///
/// Unlike [`crate::dtype::promotion::PromoteTo`], this trait covers all
/// element-type pairs including those that lose information. The semantics
/// match Rust's `as` operator and NumPy's `casting='unsafe'` default.
///
/// To check whether a cast is safe at a given level, use
/// [`crate::dtype::casting::can_cast`].
pub trait CastTo<U: Element>: Element {
    /// Convert this value to `U`. May lose information for narrowing
    /// or kind-changing casts.
    fn cast_to(self) -> U;
}

// ===========================================================================
// Identity casts (T -> T) for every Element type
// ===========================================================================

macro_rules! impl_cast_identity {
    ($ty:ty) => {
        impl CastTo<$ty> for $ty {
            #[inline]
            fn cast_to(self) -> $ty {
                self
            }
        }
    };
}

impl_cast_identity!(bool);
impl_cast_identity!(u8);
impl_cast_identity!(u16);
impl_cast_identity!(u32);
impl_cast_identity!(u64);
impl_cast_identity!(u128);
impl_cast_identity!(i8);
impl_cast_identity!(i16);
impl_cast_identity!(i32);
impl_cast_identity!(i64);
impl_cast_identity!(i128);
impl_cast_identity!(f32);
impl_cast_identity!(f64);
impl_cast_identity!(Complex<f32>);
impl_cast_identity!(Complex<f64>);

// ===========================================================================
// Numeric cross-casts via the `as` operator
//
// We generate impls for every (numeric, numeric) pair where the source and
// target are both primitive numeric types (signed/unsigned int, float). The
// `as` operator handles all the truncation, wraparound, and saturation
// semantics that NumPy expects under `casting='unsafe'`.
// ===========================================================================

macro_rules! impl_cast_as {
    ($from:ty => $($to:ty),* $(,)?) => {
        $(
            impl CastTo<$to> for $from {
                #[inline]
                fn cast_to(self) -> $to {
                    self as $to
                }
            }
        )*
    };
}

// Source: u8 (skip u8 -> u8 since identity is above)
impl_cast_as!(u8 => u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);
impl_cast_as!(u16 => u8, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);
impl_cast_as!(u32 => u8, u16, u64, u128, i8, i16, i32, i64, i128, f32, f64);
impl_cast_as!(u64 => u8, u16, u32, u128, i8, i16, i32, i64, i128, f32, f64);
impl_cast_as!(u128 => u8, u16, u32, u64, i8, i16, i32, i64, i128, f32, f64);

impl_cast_as!(i8 => u8, u16, u32, u64, u128, i16, i32, i64, i128, f32, f64);
impl_cast_as!(i16 => u8, u16, u32, u64, u128, i8, i32, i64, i128, f32, f64);
impl_cast_as!(i32 => u8, u16, u32, u64, u128, i8, i16, i64, i128, f32, f64);
impl_cast_as!(i64 => u8, u16, u32, u64, u128, i8, i16, i32, i128, f32, f64);
impl_cast_as!(i128 => u8, u16, u32, u64, u128, i8, i16, i32, i64, f32, f64);

impl_cast_as!(f32 => u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f64);
impl_cast_as!(f64 => u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32);

// ===========================================================================
// bool conversions
//
// bool -> integer: `as` works (true=1, false=0)
// bool -> float:   manual (Rust doesn't allow `bool as f64`)
// numeric -> bool: nonzero check (matches NumPy)
// ===========================================================================

macro_rules! impl_cast_bool_to_int {
    ($($to:ty),* $(,)?) => {
        $(
            impl CastTo<$to> for bool {
                #[inline]
                fn cast_to(self) -> $to {
                    self as $to
                }
            }
        )*
    };
}

impl_cast_bool_to_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl CastTo<f32> for bool {
    #[inline]
    fn cast_to(self) -> f32 {
        if self { 1.0 } else { 0.0 }
    }
}

impl CastTo<f64> for bool {
    #[inline]
    fn cast_to(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

impl CastTo<Complex<f32>> for bool {
    #[inline]
    fn cast_to(self) -> Complex<f32> {
        Complex::new(if self { 1.0 } else { 0.0 }, 0.0)
    }
}

impl CastTo<Complex<f64>> for bool {
    #[inline]
    fn cast_to(self) -> Complex<f64> {
        Complex::new(if self { 1.0 } else { 0.0 }, 0.0)
    }
}

// numeric -> bool (nonzero check)
macro_rules! impl_cast_num_to_bool {
    ($($from:ty),* $(,)?) => {
        $(
            impl CastTo<bool> for $from {
                #[inline]
                fn cast_to(self) -> bool {
                    self != 0 as $from
                }
            }
        )*
    };
}

impl_cast_num_to_bool!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl CastTo<bool> for f32 {
    #[inline]
    fn cast_to(self) -> bool {
        self != 0.0
    }
}

impl CastTo<bool> for f64 {
    #[inline]
    fn cast_to(self) -> bool {
        self != 0.0
    }
}

impl CastTo<bool> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

impl CastTo<bool> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

// ===========================================================================
// Real -> Complex (set imaginary part to zero)
// ===========================================================================

macro_rules! impl_cast_to_complex {
    ($($from:ty),* $(,)?) => {
        $(
            impl CastTo<Complex<f32>> for $from {
                #[inline]
                fn cast_to(self) -> Complex<f32> {
                    Complex::new(self as f32, 0.0)
                }
            }
            impl CastTo<Complex<f64>> for $from {
                #[inline]
                fn cast_to(self) -> Complex<f64> {
                    Complex::new(self as f64, 0.0)
                }
            }
        )*
    };
}

impl_cast_to_complex!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

// ===========================================================================
// Complex -> real (take real part — NumPy semantics)
// ===========================================================================

macro_rules! impl_cast_complex_to_real {
    ($($to:ty),* $(,)?) => {
        $(
            impl CastTo<$to> for Complex<f32> {
                #[inline]
                fn cast_to(self) -> $to {
                    self.re as $to
                }
            }
            impl CastTo<$to> for Complex<f64> {
                #[inline]
                fn cast_to(self) -> $to {
                    self.re as $to
                }
            }
        )*
    };
}

impl_cast_complex_to_real!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

// Complex<f32> <-> Complex<f64>
impl CastTo<Complex<f64>> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
}

impl CastTo<Complex<f32>> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Complex<f32> {
        Complex::new(self.re as f32, self.im as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_widen_int() {
        let a: i32 = 42;
        let b: i64 = a.cast_to();
        assert_eq!(b, 42);
    }

    #[test]
    fn cast_narrow_int_truncates() {
        let a: i64 = 300;
        let b: u8 = a.cast_to();
        assert_eq!(b, 44); // 300 % 256 = 44
    }

    #[test]
    fn cast_float_to_int_truncates() {
        let a: f64 = 3.7;
        let b: i32 = a.cast_to();
        assert_eq!(b, 3);

        let a: f64 = -2.9;
        let b: i32 = a.cast_to();
        assert_eq!(b, -2);
    }

    #[test]
    fn cast_int_to_float() {
        let a: i64 = 1234567;
        let b: f64 = a.cast_to();
        assert_eq!(b, 1234567.0);
    }

    #[test]
    fn cast_f64_to_f32_rounds() {
        let a: f64 = 1.0 + (1.0 / (1u64 << 30) as f64);
        let b: f32 = a.cast_to();
        // f32 has fewer mantissa bits — exact equality unlikely
        assert!((b as f64 - a).abs() < 1e-6);
    }

    #[test]
    fn cast_bool_to_int() {
        let t: bool = true;
        let f: bool = false;
        assert_eq!(<bool as CastTo<u8>>::cast_to(t), 1u8);
        assert_eq!(<bool as CastTo<u8>>::cast_to(f), 0u8);
        assert_eq!(<bool as CastTo<i64>>::cast_to(t), 1i64);
    }

    #[test]
    fn cast_bool_to_float() {
        assert_eq!(<bool as CastTo<f64>>::cast_to(true), 1.0);
        assert_eq!(<bool as CastTo<f64>>::cast_to(false), 0.0);
    }

    #[test]
    fn cast_int_to_bool() {
        assert!(<i32 as CastTo<bool>>::cast_to(1));
        assert!(<i32 as CastTo<bool>>::cast_to(-5));
        assert!(!<i32 as CastTo<bool>>::cast_to(0));
    }

    #[test]
    fn cast_float_to_bool() {
        assert!(<f64 as CastTo<bool>>::cast_to(1.5));
        assert!(<f64 as CastTo<bool>>::cast_to(-0.0001));
        assert!(!<f64 as CastTo<bool>>::cast_to(0.0));
        // NaN: NaN != 0.0 is true, so casts to true (matches NumPy)
        assert!(<f64 as CastTo<bool>>::cast_to(f64::NAN));
    }

    #[test]
    fn cast_real_to_complex() {
        let a: f64 = 2.5;
        let z: Complex<f64> = a.cast_to();
        assert_eq!(z, Complex::new(2.5, 0.0));

        let i: i32 = 42;
        let z2: Complex<f32> = i.cast_to();
        assert_eq!(z2, Complex::new(42.0f32, 0.0));
    }

    #[test]
    fn cast_complex_to_real_takes_re() {
        let z = Complex::new(2.5_f64, 5.5);
        let r: f64 = z.cast_to();
        assert_eq!(r, 2.5);

        let z = Complex::new(7.0_f32, 2.0);
        let i: i32 = z.cast_to();
        assert_eq!(i, 7);
    }

    #[test]
    fn cast_complex_to_complex() {
        let z = Complex::new(1.5_f32, -2.5);
        let z64: Complex<f64> = z.cast_to();
        assert_eq!(z64, Complex::new(1.5_f64, -2.5));

        let z = Complex::new(1.5_f64, -2.5);
        let z32: Complex<f32> = z.cast_to();
        assert_eq!(z32, Complex::new(1.5_f32, -2.5));
    }

    #[test]
    fn cast_complex_to_bool() {
        let z = Complex::new(0.0_f64, 0.0);
        assert!(!<Complex<f64> as CastTo<bool>>::cast_to(z));

        let z = Complex::new(0.0_f64, 1.0);
        assert!(<Complex<f64> as CastTo<bool>>::cast_to(z));
    }

    #[test]
    fn cast_identity() {
        let a: f64 = 2.5;
        let b: f64 = a.cast_to();
        assert_eq!(a, b);
    }
}
