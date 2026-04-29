// ferray-fft: FftFloat trait for f32/f64 generic FFTs
//
// Abstracts the plan cache lookup and typed normalization factor over
// scalar types. Implemented for f32 and f64; the public FFT functions
// (fft, ifft, rfft, irfft, ...) are generic over `T: FftFloat` so the
// same call site works with either precision.
//
// See: https://github.com/dollspace-gay/ferray/issues/426

use std::sync::Arc;

use ferray_core::dtype::Element;
use num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};
use rustfft::{Fft, FftNum};

use crate::norm::{FftDirection, FftNorm};
use crate::plan::{
    get_cached_plan_f32, get_cached_plan_f64, get_cached_real_forward_f32,
    get_cached_real_forward_f64, get_cached_real_inverse_f32, get_cached_real_inverse_f64,
};

/// Scalar type eligible for FFT computation.
///
/// Implemented for [`f32`] and [`f64`]. Carries the bounds required by
/// `rustfft` / `realfft` (both parameterize over `rustfft::FftNum`) plus
/// ferray-core's [`Element`] so values can live inside `Array<T, D>`.
///
/// The `where Complex<Self>: Element` bound propagates as an implied
/// bound at use sites, so generic code written with `T: FftFloat` can
/// freely construct `Array<Complex<T>, D>` without restating the
/// constraint on every function. ferray-core implements `Element` for
/// `Complex<f32>` and `Complex<f64>` which satisfies this for both
/// real impls.
///
/// The trait's methods look up cached plans and return a typed
/// normalization factor so the generic FFT pipeline can call through a
/// single interface without duplicating code per precision.
pub trait FftFloat: Element + FftNum + Copy + 'static
where
    Complex<Self>: Element,
{
    /// Return a cached `Fft<Self>` plan for the given transform size and
    /// direction (forward when `inverse = false`).
    fn cached_plan(size: usize, inverse: bool) -> Arc<dyn Fft<Self>>;

    /// Return a cached real-to-complex plan for forward `rfft`.
    fn cached_real_forward(size: usize) -> Arc<dyn RealToComplex<Self>>;

    /// Return a cached complex-to-real plan for inverse `irfft`.
    ///
    /// The `size` argument is the **real-output** length (not the
    /// complex input length, which is `size/2 + 1`).
    fn cached_real_inverse(size: usize) -> Arc<dyn ComplexToReal<Self>>;

    /// Normalization scale factor as a value of this precision.
    ///
    /// Thin wrapper over [`FftNorm::scale_factor_f64`] with a cast back
    /// to `Self`, so generic code avoids the `f64 → f32` conversion dance.
    fn scale_factor(norm: FftNorm, n: usize, direction: FftDirection) -> Self;
}

impl FftFloat for f64 {
    fn cached_plan(size: usize, inverse: bool) -> Arc<dyn Fft<Self>> {
        get_cached_plan_f64(size, inverse)
    }

    fn cached_real_forward(size: usize) -> Arc<dyn RealToComplex<Self>> {
        get_cached_real_forward_f64(size)
    }

    fn cached_real_inverse(size: usize) -> Arc<dyn ComplexToReal<Self>> {
        get_cached_real_inverse_f64(size)
    }

    fn scale_factor(norm: FftNorm, n: usize, direction: FftDirection) -> Self {
        norm.scale_factor_f64(n, direction)
    }
}

impl FftFloat for f32 {
    fn cached_plan(size: usize, inverse: bool) -> Arc<dyn Fft<Self>> {
        get_cached_plan_f32(size, inverse)
    }

    fn cached_real_forward(size: usize) -> Arc<dyn RealToComplex<Self>> {
        get_cached_real_forward_f32(size)
    }

    fn cached_real_inverse(size: usize) -> Arc<dyn ComplexToReal<Self>> {
        get_cached_real_inverse_f32(size)
    }

    fn scale_factor(norm: FftNorm, n: usize, direction: FftDirection) -> Self {
        norm.scale_factor_f64(n, direction) as Self
    }
}
