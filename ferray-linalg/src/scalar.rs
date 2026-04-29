// ferray-linalg: LinalgFloat trait — sealed bound for f32/f64 generic linalg
//
// This trait combines the requirements from ferray-core (Element), faer (ComplexField),
// and num-traits (Float) into a single sealed bound. Only f32 and f64 implement it.

use ferray_core::dtype::Element;
use num_complex::Complex;
use num_traits::Float;

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Marker trait for floating-point types supported by ferray-linalg.
///
/// This is a sealed trait — only `f32` and `f64` implement it.
/// It provides the combined bounds needed to:
/// - Store elements in ferray arrays ([`Element`])
/// - Perform LAPACK-style operations via faer ([`faer_traits::ComplexField`])
/// - Use standard floating-point operations ([`num_traits::Float`])
pub trait LinalgFloat:
    Element
    + Float
    + faer_traits::ComplexField<Canonical = Self, Real = Self>
    + faer_traits::RealField
    + Copy
    + Send
    + Sync
    + std::ops::AddAssign
    + std::iter::Sum
    + 'static
    + private::Sealed
{
    /// The complex type corresponding to this real type.
    type Complex: Element;

    /// Machine epsilon for this type.
    fn machine_epsilon() -> Self;

    /// Convert from f64 (used for `NormOrder::P` parameter and literal constants).
    fn from_f64_const(v: f64) -> Self;

    /// Convert to f64.
    fn into_f64(self) -> f64;

    /// Convert from usize.
    fn from_usize(v: usize) -> Self;

    /// Create a Complex<Self> from real and imaginary parts.
    fn complex(re: Self, im: Self) -> Self::Complex;
}

impl LinalgFloat for f32 {
    type Complex = Complex<Self>;

    #[inline]
    fn machine_epsilon() -> Self {
        Self::EPSILON
    }

    #[inline]
    fn from_f64_const(v: f64) -> Self {
        v as Self
    }

    #[inline]
    fn into_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as Self
    }

    #[inline]
    fn complex(re: Self, im: Self) -> Self::Complex {
        Complex::new(re, im)
    }
}

impl LinalgFloat for f64 {
    type Complex = Complex<Self>;

    #[inline]
    fn machine_epsilon() -> Self {
        Self::EPSILON
    }

    #[inline]
    fn from_f64_const(v: f64) -> Self {
        v
    }

    #[inline]
    fn into_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as Self
    }

    #[inline]
    fn complex(re: Self, im: Self) -> Self::Complex {
        Complex::new(re, im)
    }
}
