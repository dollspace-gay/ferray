// ferray-fft: Hermitian FFTs — hfft, ihfft (REQ-8)
//
// hfft: Takes a signal with Hermitian symmetry (in frequency domain)
//       and returns a real-valued result. This is the inverse of ihfft.
//       Equivalent to irfft but with different input interpretation.
//
// ihfft: Takes a real-valued signal and returns the Hermitian-symmetric
//        spectrum (n/2+1 complex values). This is the inverse of hfft.
//        Equivalent to rfft of the input divided by n (with appropriate norm).

use num_complex::Complex;

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::axes::resolve_axis;
use crate::float::FftFloat;
use crate::norm::FftNorm;

// ---------------------------------------------------------------------------
// hfft (REQ-8)
// ---------------------------------------------------------------------------

/// Compute the FFT of a signal with Hermitian symmetry (real spectrum).
///
/// Analogous to `numpy.fft.hfft`. The input is a Hermitian-symmetric
/// signal of length `n/2+1`, and the output is a real-valued array of
/// length `n`.
///
/// This is effectively the inverse of [`ihfft`]. The input is first
/// extended using Hermitian symmetry, then an inverse FFT is applied,
/// and the result is scaled by `n`.
///
/// # Parameters
/// - `a`: Input complex array (Hermitian-symmetric, length `n/2+1`).
/// - `n`: Output length. If `None`, uses `2 * (input_length - 1)`.
/// - `axis`: Axis along which to compute. Defaults to the last axis.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn hfft<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    // hfft is essentially irfft with Forward normalization swapped:
    // numpy defines hfft as the inverse of ihfft.
    // hfft(a, n) = irfft(conj(a), n) * n  (with backward norm)
    // But with proper norm parameter handling, we use irfft logic directly.

    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let half_len = shape[ax];
    let output_len = n.unwrap_or(2 * (half_len - 1));

    if output_len == 0 {
        return Err(FerrayError::invalid_value("hfft output length must be > 0"));
    }

    // Conjugate the input (hfft = ifft(conj(a)) * n effectively)
    let conj_data: Vec<Complex<T>> = a.iter().map(|c| c.conj()).collect();
    let conj_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), conj_data)?;

    // Use irfft on the conjugated data
    // For hfft with Backward norm: no scaling on forward, so we scale by n
    // For hfft, the normalization semantics are swapped vs irfft
    let hfft_norm = match norm {
        FftNorm::Backward => FftNorm::Forward,
        FftNorm::Forward => FftNorm::Backward,
        FftNorm::Ortho => FftNorm::Ortho,
    };

    crate::real::irfft::<T, IxDyn>(&conj_arr, Some(output_len), Some(ax), hfft_norm)
}

/// Compute the inverse FFT of a real-valued signal, returning
/// Hermitian-symmetric output.
///
/// Analogous to `numpy.fft.ihfft`. Takes `n` real values and returns
/// `n/2+1` complex values representing the Hermitian-symmetric spectrum.
///
/// This is the inverse of [`hfft`].
///
/// # Parameters
/// - `a`: Input real-valued array.
/// - `n`: Length of the FFT. If `None`, uses the input length.
/// - `axis`: Axis along which to compute. Defaults to the last axis.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn ihfft<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    n: Option<usize>,
    axis: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    // ihfft is essentially rfft with swapped normalization and conjugation
    // numpy: ihfft(a) = conj(rfft(a)) / n  (with backward norm)

    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;

    // Swap normalization (ihfft's forward is rfft's inverse semantics)
    let ihfft_norm = match norm {
        FftNorm::Backward => FftNorm::Forward,
        FftNorm::Forward => FftNorm::Backward,
        FftNorm::Ortho => FftNorm::Ortho,
    };

    let result = crate::real::rfft::<T, D>(a, n, Some(ax), ihfft_norm)?;

    // Conjugate the output
    let conj_data: Vec<Complex<T>> = result.iter().map(|c| c.conj()).collect();
    let out_shape = result.shape().to_vec();
    Array::from_vec(IxDyn::new(&out_shape), conj_data)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    #[test]
    fn hfft_ihfft_roundtrip() {
        // ihfft of a real signal, then hfft should recover the original
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = original.len();
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([n]), original.clone()).unwrap();

        let spectrum = ihfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[n / 2 + 1]);

        let recovered = hfft(&spectrum, Some(n), None, FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[n]);

        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in original.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-10, "mismatch: expected {}, got {}", o, r);
        }
    }

    #[test]
    fn ihfft_basic() {
        // ihfft of [1, 0, 0, 0] — DC impulse
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = ihfft(&a, None, None, FftNorm::Backward).unwrap();
        // n/2+1 = 3
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn hfft_hermitian_input() {
        // Create Hermitian-symmetric input and verify output is real
        let input = vec![c(10.0, 0.0), c(-2.0, 2.0), c(-2.0, 0.0)];
        let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([3]), input).unwrap();
        let result = hfft(&a, Some(4), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[4]);
        // Output should be real-valued (imaginary parts ~ 0)
        // This is guaranteed by Hermitian symmetry of input
    }
}
