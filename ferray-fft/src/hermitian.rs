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

use crate::axes::{resolve_axes, resolve_axis};
use crate::float::FftFloat;
use crate::norm::FftNorm;

/// Hermitian FFTs swap forward/backward normalization relative to their
/// rfft/irfft counterparts. Ortho is preserved.
fn swap_norm(n: FftNorm) -> FftNorm {
    match n {
        FftNorm::Backward => FftNorm::Forward,
        FftNorm::Forward => FftNorm::Backward,
        FftNorm::Ortho => FftNorm::Ortho,
    }
}

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
    axis: Option<isize>,
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
    let conj_data: Vec<Complex<T>> = a.iter().map(num_complex::Complex::conj).collect();
    let conj_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), conj_data)?;

    // `ax` is already validated against `ndim` here, so casting to
    // `isize` is lossless within usize::MAX/2.
    crate::real::irfft::<T, IxDyn>(
        &conj_arr,
        Some(output_len),
        Some(ax as isize),
        swap_norm(norm),
    )
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
    axis: Option<isize>,
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

    let result = crate::real::rfft::<T, D>(a, n, Some(ax as isize), swap_norm(norm))?;

    // Conjugate the output
    let conj_data: Vec<Complex<T>> = result.iter().map(num_complex::Complex::conj).collect();
    let out_shape = result.shape().to_vec();
    Array::from_vec(IxDyn::new(&out_shape), conj_data)
}

// ---------------------------------------------------------------------------
// N-D Hermitian FFTs (#582 — torch parity for hfftn / ihfftn / hfft2 / ihfft2)
// ---------------------------------------------------------------------------

/// N-dimensional FFT of a Hermitian-symmetric signal, returning a real array.
///
/// Analogous to `numpy.fft.hfftn` / `torch.fft.hfftn`. Generalizes [`hfft`]
/// to arbitrary axes. Implementation: conjugate the input, run [`irfftn`]
/// with swapped normalization.
///
/// # Parameters
/// - `a`: Input complex array (Hermitian-symmetric along the last
///   transform axis, length `n/2+1`).
/// - `s`: Output shape along transform axes. The last entry is the real
///   signal length along the last transform axis (defaults to
///   `2 * (input_last - 1)`).
/// - `axes`: Axes to transform. Defaults to all axes.
/// - `norm`: Normalization mode (swapped vs irfftn internally).
pub fn hfftn<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    let shape = a.shape().to_vec();
    let conj_data: Vec<Complex<T>> = a.iter().map(num_complex::Complex::conj).collect();
    let conj_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), conj_data)?;
    crate::real::irfftn::<T, IxDyn>(&conj_arr, s, axes, swap_norm(norm))
}

/// N-dimensional inverse FFT of a real signal, returning Hermitian-symmetric
/// complex output.
///
/// Analogous to `numpy.fft.ihfftn` / `torch.fft.ihfftn`. Generalizes [`ihfft`]
/// to arbitrary axes. Implementation: run [`rfftn`] with swapped
/// normalization, then conjugate the output.
pub fn ihfftn<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    let result = crate::real::rfftn::<T, D>(a, s, axes, swap_norm(norm))?;
    let conj_data: Vec<Complex<T>> = result.iter().map(num_complex::Complex::conj).collect();
    let out_shape = result.shape().to_vec();
    Array::from_vec(IxDyn::new(&out_shape), conj_data)
}

/// 2-D FFT of a Hermitian-symmetric signal, returning a real array. Defaults
/// the transform axes to the last two when `axes` is `None`.
///
/// Analogous to `numpy.fft.hfft2` / `torch.fft.hfft2`.
pub fn hfft2<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    let ndim = a.shape().len();
    let resolved: Vec<isize> = if let Some(ax) = axes {
        ax.to_vec()
    } else {
        if ndim < 2 {
            return Err(FerrayError::invalid_value(
                "hfft2 requires at least 2 dimensions",
            ));
        }
        vec![(ndim - 2) as isize, (ndim - 1) as isize]
    };
    let _ = resolve_axes(ndim, Some(&resolved))?;
    hfftn::<T, D>(a, s, Some(&resolved), norm)
}

/// 2-D inverse FFT of a real signal, returning Hermitian-symmetric complex
/// output. Defaults the transform axes to the last two.
pub fn ihfft2<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    let ndim = a.shape().len();
    let resolved: Vec<isize> = if let Some(ax) = axes {
        ax.to_vec()
    } else {
        if ndim < 2 {
            return Err(FerrayError::invalid_value(
                "ihfft2 requires at least 2 dimensions",
            ));
        }
        vec![(ndim - 2) as isize, (ndim - 1) as isize]
    };
    let _ = resolve_axes(ndim, Some(&resolved))?;
    ihfftn::<T, D>(a, s, Some(&resolved), norm)
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
            assert!((o - r).abs() < 1e-10, "mismatch: expected {o}, got {r}");
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

    // -----------------------------------------------------------------------
    // hfftn / ihfftn / hfft2 / ihfft2 (#582)
    // -----------------------------------------------------------------------

    use ferray_core::dimension::Ix2;

    #[test]
    fn hfftn_ihfftn_roundtrip_2d() {
        // Round-trip a 2-D real signal through ihfftn → hfftn.
        let original: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([4, 4]), original.clone()).unwrap();

        let spectrum = ihfftn(&a, None, None, FftNorm::Backward).unwrap();
        // Last axis becomes 4/2+1 = 3, others stay full → [4, 3].
        assert_eq!(spectrum.shape(), &[4, 3]);

        let recovered = hfftn(&spectrum, Some(&[4, 4]), None, FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[4, 4]);
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in original.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-9, "mismatch: expected {o}, got {r}");
        }
    }

    #[test]
    fn hfft2_ihfft2_roundtrip() {
        // hfft2 / ihfft2 default to the last two axes — same shape as hfftn
        // when input is 2-D.
        let original: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([4, 4]), original.clone()).unwrap();

        let spectrum = ihfft2(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[4, 3]);
        let recovered = hfft2(&spectrum, Some(&[4, 4]), None, FftNorm::Backward).unwrap();
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in original.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-9);
        }
    }

    #[test]
    fn ihfftn_collapses_to_ihfft_for_1d() {
        // For a 1-D input, ihfftn must agree with the 1-D ihfft.
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), original.clone()).unwrap();
        let nd = ihfftn(&a, None, None, FftNorm::Backward).unwrap();
        let one = ihfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(nd.shape(), one.shape());
        for (lhs, rhs) in nd.iter().zip(one.iter()) {
            assert!((lhs.re - rhs.re).abs() < 1e-12);
            assert!((lhs.im - rhs.im).abs() < 1e-12);
        }
    }

    #[test]
    fn hfftn_collapses_to_hfft_for_1d() {
        // For a 1-D input, hfftn must agree with the 1-D hfft.
        let input = vec![c(10.0, 0.0), c(-2.0, 2.0), c(-2.0, 0.0)];
        let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([3]), input).unwrap();
        let nd = hfftn(&a, Some(&[4]), None, FftNorm::Backward).unwrap();
        let one = hfft(&a, Some(4), None, FftNorm::Backward).unwrap();
        assert_eq!(nd.shape(), one.shape());
        for (lhs, rhs) in nd.iter().zip(one.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    #[test]
    fn hfft2_rejects_1d_input() {
        let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([4]), vec![c(0.0, 0.0); 4]).unwrap();
        let err = hfft2(&a, None, None, FftNorm::Backward);
        assert!(err.is_err());
    }
}
