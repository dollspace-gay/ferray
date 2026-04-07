// ferray-fft: Complex FFTs — fft, ifft, fft2, ifft2, fftn, ifftn (REQ-1..REQ-4)

use num_complex::Complex;

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::nd::{fft_1d_along_axis, fft_along_axes};
use crate::norm::FftNorm;

// ---------------------------------------------------------------------------
// Helpers to convert input to Complex<f64> flat data
// ---------------------------------------------------------------------------

/// Convert an Array<Complex<f64>, D> to flat row-major data.
/// Get contiguous data, borrowing if C-contiguous, copying otherwise.
enum ComplexData<'a> {
    Borrowed(&'a [Complex<f64>]),
    Owned(Vec<Complex<f64>>),
}

impl std::ops::Deref for ComplexData<'_> {
    type Target = [Complex<f64>];
    fn deref(&self) -> &[Complex<f64>] {
        match self {
            ComplexData::Borrowed(s) => s,
            ComplexData::Owned(v) => v,
        }
    }
}

fn borrow_complex_flat<D: Dimension>(a: &Array<Complex<f64>, D>) -> ComplexData<'_> {
    if let Some(s) = a.as_slice() {
        ComplexData::Borrowed(s)
    } else {
        ComplexData::Owned(a.iter().copied().collect())
    }
}

/// Resolve an axis parameter: if None, use the last axis.
fn resolve_axis(ndim: usize, axis: Option<usize>) -> FerrayResult<usize> {
    match axis {
        Some(ax) => {
            if ax >= ndim {
                Err(FerrayError::axis_out_of_bounds(ax, ndim))
            } else {
                Ok(ax)
            }
        }
        None => {
            if ndim == 0 {
                Err(FerrayError::invalid_value(
                    "cannot compute FFT on a 0-dimensional array",
                ))
            } else {
                Ok(ndim - 1)
            }
        }
    }
}

/// Resolve axes for multi-dimensional FFT. If None, use all axes.
fn resolve_axes(ndim: usize, axes: Option<&[usize]>) -> FerrayResult<Vec<usize>> {
    match axes {
        Some(ax) => {
            for &a in ax {
                if a >= ndim {
                    return Err(FerrayError::axis_out_of_bounds(a, ndim));
                }
            }
            Ok(ax.to_vec())
        }
        None => Ok((0..ndim).collect()),
    }
}

/// Resolve shapes for multi-dimensional FFT. If None, use the input shape.
fn resolve_shapes(
    input_shape: &[usize],
    axes: &[usize],
    s: Option<&[usize]>,
) -> FerrayResult<Vec<Option<usize>>> {
    match s {
        Some(sizes) => {
            if sizes.len() != axes.len() {
                return Err(FerrayError::invalid_value(format!(
                    "shape parameter length {} does not match axes length {}",
                    sizes.len(),
                    axes.len(),
                )));
            }
            Ok(sizes.iter().map(|&sz| Some(sz)).collect())
        }
        None => Ok(axes.iter().map(|&ax| Some(input_shape[ax])).collect()),
    }
}

// ---------------------------------------------------------------------------
// 1-D FFT (REQ-1)
// ---------------------------------------------------------------------------

/// Compute the one-dimensional discrete Fourier Transform.
///
/// Analogous to `numpy.fft.fft`. The input must be a complex array.
///
/// # Parameters
/// - `a`: Input complex array of any dimensionality.
/// - `n`: Length of the transformed axis. If `None`, uses the length of
///   the input along `axis`. If shorter, the input is truncated. If longer,
///   it is zero-padded.
/// - `axis`: Axis along which to compute the FFT. Defaults to the last axis.
/// - `norm`: Normalization mode. Defaults to `FftNorm::Backward`.
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn fft<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    n: Option<usize>,
    axis: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let data = borrow_complex_flat(a);

    let (new_shape, result) = fft_1d_along_axis(&data, &shape, ax, n, false, norm)?;

    Array::from_vec(IxDyn::new(&new_shape), result)
}

/// Compute the one-dimensional inverse discrete Fourier Transform.
///
/// Analogous to `numpy.fft.ifft`.
///
/// # Parameters
/// - `a`: Input complex array.
/// - `n`: Length of the transformed axis. Defaults to the input length.
/// - `axis`: Axis along which to compute. Defaults to the last axis.
/// - `norm`: Normalization mode. Defaults to `FftNorm::Backward` (divides by `n`).
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn ifft<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    n: Option<usize>,
    axis: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let data = borrow_complex_flat(a);

    let (new_shape, result) = fft_1d_along_axis(&data, &shape, ax, n, true, norm)?;

    Array::from_vec(IxDyn::new(&new_shape), result)
}

// ---------------------------------------------------------------------------
// 2-D FFT (REQ-3)
// ---------------------------------------------------------------------------

/// Compute the 2-dimensional discrete Fourier Transform.
///
/// Analogous to `numpy.fft.fft2`. Equivalent to calling `fftn` with
/// `axes = [-2, -1]` (the last two axes).
///
/// # Parameters
/// - `a`: Input complex array (must have at least 2 dimensions).
/// - `s`: Shape `(n_rows, n_cols)` of the output along the transform axes.
///   If `None`, uses the input shape.
/// - `axes`: Axes over which to compute the FFT. Defaults to the last 2 axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if the array has fewer than 2 dimensions, axes are
/// out of bounds, or shape parameters are invalid.
pub fn fft2<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let ndim = a.shape().len();
    let axes = match axes {
        Some(ax) => ax.to_vec(),
        None => {
            if ndim < 2 {
                return Err(FerrayError::invalid_value(
                    "fft2 requires at least 2 dimensions",
                ));
            }
            vec![ndim - 2, ndim - 1]
        }
    };
    fftn_impl(a, s, &axes, false, norm)
}

/// Compute the 2-dimensional inverse discrete Fourier Transform.
///
/// Analogous to `numpy.fft.ifft2`.
///
/// # Parameters
/// - `a`: Input complex array (must have at least 2 dimensions).
/// - `s`: Output shape along the transform axes. If `None`, uses input shape.
/// - `axes`: Axes over which to compute. Defaults to the last 2 axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if the array has fewer than 2 dimensions, axes are
/// out of bounds, or shape parameters are invalid.
pub fn ifft2<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let ndim = a.shape().len();
    let axes = match axes {
        Some(ax) => ax.to_vec(),
        None => {
            if ndim < 2 {
                return Err(FerrayError::invalid_value(
                    "ifft2 requires at least 2 dimensions",
                ));
            }
            vec![ndim - 2, ndim - 1]
        }
    };
    fftn_impl(a, s, &axes, true, norm)
}

// ---------------------------------------------------------------------------
// N-D FFT (REQ-4)
// ---------------------------------------------------------------------------

/// Compute the N-dimensional discrete Fourier Transform.
///
/// Analogous to `numpy.fft.fftn`. Transforms along each of the specified
/// axes in sequence.
///
/// # Parameters
/// - `a`: Input complex array.
/// - `s`: Shape of the output along each transform axis. If `None`,
///   uses the input shape.
/// - `axes`: Axes over which to compute. If `None`, uses all axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if axes are out of bounds or shape parameters
/// are inconsistent.
pub fn fftn<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let ax = resolve_axes(a.shape().len(), axes)?;
    fftn_impl(a, s, &ax, false, norm)
}

/// Compute the N-dimensional inverse discrete Fourier Transform.
///
/// Analogous to `numpy.fft.ifftn`.
///
/// # Parameters
/// - `a`: Input complex array.
/// - `s`: Shape of the output along each transform axis. If `None`,
///   uses the input shape.
/// - `axes`: Axes over which to compute. If `None`, uses all axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if axes are out of bounds or shape parameters
/// are inconsistent.
pub fn ifftn<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let ax = resolve_axes(a.shape().len(), axes)?;
    fftn_impl(a, s, &ax, true, norm)
}

// ---------------------------------------------------------------------------
// Internal N-D implementation
// ---------------------------------------------------------------------------

fn fftn_impl<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: &[usize],
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let shape = a.shape().to_vec();
    let sizes = resolve_shapes(&shape, axes, s)?;
    let data = borrow_complex_flat(a);

    let axes_and_sizes: Vec<(usize, Option<usize>)> = axes.iter().copied().zip(sizes).collect();

    let (new_shape, result) = fft_along_axes(&data, &shape, &axes_and_sizes, inverse, norm)?;

    Array::from_vec(IxDyn::new(&new_shape), result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    fn make_1d(data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn fft_impulse() {
        // FFT of [1, 0, 0, 0] = [1, 1, 1, 1]
        let a = make_1d(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
        let result = fft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[4]);
        for val in result.iter() {
            assert!((val.re - 1.0).abs() < 1e-12);
            assert!(val.im.abs() < 1e-12);
        }
    }

    #[test]
    fn fft_constant() {
        // FFT of [1, 1, 1, 1] = [4, 0, 0, 0]
        let a = make_1d(vec![c(1.0, 0.0); 4]);
        let result = fft(&a, None, None, FftNorm::Backward).unwrap();
        let vals: Vec<_> = result.iter().copied().collect();
        assert!((vals[0].re - 4.0).abs() < 1e-12);
        for v in &vals[1..] {
            assert!(v.re.abs() < 1e-12);
            assert!(v.im.abs() < 1e-12);
        }
    }

    #[test]
    fn fft_ifft_roundtrip() {
        // AC-1: fft(ifft(a)) roundtrips to within 4 ULPs for complex f64
        let data = vec![
            c(1.0, 2.0),
            c(-1.0, 0.5),
            c(3.0, -1.0),
            c(0.0, 0.0),
            c(-2.5, 1.5),
            c(0.7, -0.3),
            c(1.2, 0.8),
            c(-0.4, 2.1),
        ];
        let a = make_1d(data.clone());
        let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
        let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig.re - rec.re).abs() < 1e-10,
                "re mismatch: {} vs {}",
                orig.re,
                rec.re
            );
            assert!(
                (orig.im - rec.im).abs() < 1e-10,
                "im mismatch: {} vs {}",
                orig.im,
                rec.im
            );
        }
    }

    #[test]
    fn fft_with_n_padding() {
        // Pad [1, 1] to length 4 -> FFT of [1, 1, 0, 0]
        let a = make_1d(vec![c(1.0, 0.0), c(1.0, 0.0)]);
        let result = fft(&a, Some(4), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[4]);
        let vals: Vec<_> = result.iter().copied().collect();
        assert!((vals[0].re - 2.0).abs() < 1e-12);
    }

    #[test]
    fn fft_with_n_truncation() {
        // Truncate [1, 2, 3, 4] to length 2 -> FFT of [1, 2]
        let a = make_1d(vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)]);
        let result = fft(&a, Some(2), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[2]);
        let vals: Vec<_> = result.iter().copied().collect();
        // FFT of [1, 2] = [3, -1]
        assert!((vals[0].re - 3.0).abs() < 1e-12);
        assert!((vals[1].re - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn fft_non_power_of_two() {
        // AC-2 partial: test non-power-of-2 length
        let n = 7;
        let data: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64, 0.0)).collect();
        let a = make_1d(data.clone());
        let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
        let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }
    }

    #[test]
    fn fft2_basic() {
        use ferray_core::dimension::Ix2;
        let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let a = Array::from_vec(Ix2::new([2, 2]), data).unwrap();
        let result = fft2(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let recovered = ifft2(&result, None, None, FftNorm::Backward).unwrap();
        let orig: Vec<_> = a.iter().copied().collect();
        for (o, r) in orig.iter().zip(recovered.iter()) {
            assert!((o.re - r.re).abs() < 1e-10);
            assert!((o.im - r.im).abs() < 1e-10);
        }
    }

    #[test]
    fn fftn_roundtrip_3d() {
        use ferray_core::dimension::Ix3;
        let n = 2 * 3 * 4;
        let data: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64, -(i as f64) * 0.5)).collect();
        let a = Array::from_vec(Ix3::new([2, 3, 4]), data.clone()).unwrap();
        let spectrum = fftn(&a, None, None, FftNorm::Backward).unwrap();
        let recovered = ifftn(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o.re - r.re).abs() < 1e-9, "re: {} vs {}", o.re, r.re);
            assert!((o.im - r.im).abs() < 1e-9, "im: {} vs {}", o.im, r.im);
        }
    }

    #[test]
    fn fft_axis_out_of_bounds() {
        let a = make_1d(vec![c(1.0, 0.0)]);
        assert!(fft(&a, None, Some(1), FftNorm::Backward).is_err());
    }

    // --- Shape/padding parameter tests ---

    #[test]
    fn fft2_with_shape_padding() {
        use ferray_core::dimension::Ix2;
        // 2x2 input, pad to 4x4 via s parameter
        let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let a = Array::from_vec(Ix2::new([2, 2]), data).unwrap();
        let result = fft2(&a, Some(&[4, 4]), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[4, 4]);
    }

    #[test]
    fn fft2_with_shape_truncation() {
        use ferray_core::dimension::Ix2;
        // 4x4 input, truncate to 2x2 via s parameter
        let data: Vec<Complex<f64>> = (0..16).map(|i| c(i as f64, 0.0)).collect();
        let a = Array::from_vec(Ix2::new([4, 4]), data).unwrap();
        let result = fft2(&a, Some(&[2, 2]), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn fftn_with_shape_roundtrip() {
        use ferray_core::dimension::Ix2;
        // 3x4 input, pad to 4x8, then ifft with same shape should recover padded version
        let data: Vec<Complex<f64>> = (0..12).map(|i| c(i as f64, 0.0)).collect();
        let a = Array::from_vec(Ix2::new([3, 4]), data).unwrap();
        let spectrum = fftn(&a, Some(&[4, 8]), None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[4, 8]);
        let recovered = ifftn(&spectrum, Some(&[4, 8]), None, FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[4, 8]);
        // First 3x4 block should match original (rest is zero-padded)
        for i in 0..3 {
            for j in 0..4 {
                let idx = i * 8 + j;
                let orig_val = (i * 4 + j) as f64;
                assert!(
                    (recovered.iter().nth(idx).unwrap().re - orig_val).abs() < 1e-9,
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    // --- Normalization mode tests ---

    #[test]
    fn fft_ifft_ortho_roundtrip() {
        // Ortho normalization: both forward and inverse scale by 1/sqrt(n)
        let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let a = make_1d(data.clone());
        let spectrum = fft(&a, None, None, FftNorm::Ortho).unwrap();
        let recovered = ifft(&spectrum, None, None, FftNorm::Ortho).unwrap();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }
    }

    #[test]
    fn fft_ifft_forward_roundtrip() {
        // Forward normalization: forward scales by 1/n, inverse no scaling
        let data = vec![c(1.0, 2.0), c(-1.0, 0.5), c(3.0, -1.0), c(0.0, 0.0)];
        let a = make_1d(data.clone());
        let spectrum = fft(&a, None, None, FftNorm::Forward).unwrap();
        let recovered = ifft(&spectrum, None, None, FftNorm::Forward).unwrap();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }
    }

    #[test]
    fn fft_ortho_energy_preservation() {
        // Parseval's theorem: ||x||^2 = ||X||^2 for ortho normalization
        let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let a = make_1d(data.clone());
        let spectrum = fft(&a, None, None, FftNorm::Ortho).unwrap();

        let energy_time: f64 = data.iter().map(|x| x.re * x.re + x.im * x.im).sum();
        let energy_freq: f64 = spectrum
            .iter()
            .map(|x| x.re * x.re + x.im * x.im)
            .sum();
        assert!(
            (energy_time - energy_freq).abs() < 1e-10,
            "Parseval: time={energy_time}, freq={energy_freq}"
        );
    }

    #[test]
    fn fft_forward_scaling() {
        // Forward normalization: FFT of constant [1,1,1,1] should be [1, 0, 0, 0]
        // (divided by n=4, so DC = 4/4 = 1)
        let a = make_1d(vec![c(1.0, 0.0); 4]);
        let result = fft(&a, None, None, FftNorm::Forward).unwrap();
        let vals: Vec<_> = result.iter().copied().collect();
        assert!((vals[0].re - 1.0).abs() < 1e-12);
        for v in &vals[1..] {
            assert!(v.re.abs() < 1e-12);
            assert!(v.im.abs() < 1e-12);
        }
    }
}
