// ferray-fft: Complex FFTs — fft, ifft, fft2, ifft2, fftn, ifftn (REQ-1..REQ-4)
//
// Generic over the scalar precision via [`FftFloat`] — works for both
// `Complex<f64>` and `Complex<f32>` arrays. See issue #426.

use num_complex::Complex;

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::axes::{resolve_axes, resolve_axis};
use crate::float::FftFloat;
use crate::nd::{fft_along_axes, fft_along_axis};
use crate::norm::FftNorm;

// ---------------------------------------------------------------------------
// Helpers to convert input to flat complex data
// ---------------------------------------------------------------------------

/// Borrowed or owned contiguous complex data. Borrows when the input
/// array is C-contiguous; otherwise materializes into an owned `Vec`.
enum ComplexData<'a, T: FftFloat>
where
    Complex<T>: ferray_core::Element,
{
    Borrowed(&'a [Complex<T>]),
    Owned(Vec<Complex<T>>),
}

impl<T: FftFloat> std::ops::Deref for ComplexData<'_, T>
where
    Complex<T>: ferray_core::Element,
{
    type Target = [Complex<T>];
    fn deref(&self) -> &[Complex<T>] {
        match self {
            ComplexData::Borrowed(s) => s,
            ComplexData::Owned(v) => v,
        }
    }
}

fn borrow_complex_flat<T: FftFloat, D: Dimension>(a: &Array<Complex<T>, D>) -> ComplexData<'_, T>
where
    Complex<T>: ferray_core::Element,
{
    if let Some(s) = a.as_slice() {
        ComplexData::Borrowed(s)
    } else {
        ComplexData::Owned(a.iter().copied().collect())
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
pub fn fft<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let data = borrow_complex_flat(a);

    let (new_shape, result) = fft_along_axis::<T>(&data, &shape, ax, n, false, norm)?;

    Array::from_vec(IxDyn::new(&new_shape), result)
}

/// In-place variant of [`fft`] writing the result into a pre-allocated
/// `out` buffer (#429).
///
/// `out` must be a contiguous `Array<Complex<T>, IxDyn>` whose shape
/// matches the result of `fft(a, n, axis, norm)`. Returns an error if
/// the shapes mismatch or the buffer is non-contiguous. Useful for
/// real-time signal processing where the output shape is known and
/// repeated transforms should reuse the same allocation.
///
/// # Errors
/// - Same as [`fft`] for axis/n validation.
/// - `FerrayError::ShapeMismatch` if `out` has a different shape than
///   the FFT would produce.
/// - `FerrayError::InvalidValue` if `out` is non-contiguous.
pub fn fft_into<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
    out: &mut Array<Complex<T>, IxDyn>,
) -> FerrayResult<()>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let data = borrow_complex_flat(a);
    let (new_shape, result) = fft_along_axis::<T>(&data, &shape, ax, n, false, norm)?;
    if out.shape() != new_shape.as_slice() {
        return Err(ferray_core::error::FerrayError::shape_mismatch(format!(
            "fft_into: out shape {:?} does not match fft output shape {:?}",
            out.shape(),
            new_shape
        )));
    }
    let out_slice = out.as_slice_mut().ok_or_else(|| {
        ferray_core::error::FerrayError::invalid_value(
            "fft_into requires a contiguous out buffer",
        )
    })?;
    out_slice.copy_from_slice(&result);
    Ok(())
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
pub fn ifft<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let data = borrow_complex_flat(a);

    let (new_shape, result) = fft_along_axis::<T>(&data, &shape, ax, n, true, norm)?;

    Array::from_vec(IxDyn::new(&new_shape), result)
}

/// In-place variant of [`ifft`] writing the result into a
/// pre-allocated `out` buffer (#429). Same shape requirements and
/// error model as [`fft_into`].
///
/// # Errors
/// - Same as [`ifft`] for axis/n validation.
/// - `FerrayError::ShapeMismatch` if `out`'s shape doesn't match
///   the inverse FFT's output shape.
/// - `FerrayError::InvalidValue` if `out` is non-contiguous.
pub fn ifft_into<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
    out: &mut Array<Complex<T>, IxDyn>,
) -> FerrayResult<()>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let data = borrow_complex_flat(a);
    let (new_shape, result) = fft_along_axis::<T>(&data, &shape, ax, n, true, norm)?;
    if out.shape() != new_shape.as_slice() {
        return Err(ferray_core::error::FerrayError::shape_mismatch(format!(
            "ifft_into: out shape {:?} does not match ifft output shape {:?}",
            out.shape(),
            new_shape
        )));
    }
    let out_slice = out.as_slice_mut().ok_or_else(|| {
        ferray_core::error::FerrayError::invalid_value(
            "ifft_into requires a contiguous out buffer",
        )
    })?;
    out_slice.copy_from_slice(&result);
    Ok(())
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
pub fn fft2<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let ndim = a.shape().len();
    let axes = if let Some(ax) = axes {
        resolve_axes(ndim, Some(ax))?
    } else {
        if ndim < 2 {
            return Err(FerrayError::invalid_value(
                "fft2 requires at least 2 dimensions",
            ));
        }
        vec![ndim - 2, ndim - 1]
    };
    fftn_impl::<T, D>(a, s, &axes, false, norm)
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
pub fn ifft2<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let ndim = a.shape().len();
    let axes = if let Some(ax) = axes {
        resolve_axes(ndim, Some(ax))?
    } else {
        if ndim < 2 {
            return Err(FerrayError::invalid_value(
                "ifft2 requires at least 2 dimensions",
            ));
        }
        vec![ndim - 2, ndim - 1]
    };
    fftn_impl::<T, D>(a, s, &axes, true, norm)
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
pub fn fftn<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let ax = resolve_axes(a.shape().len(), axes)?;
    fftn_impl::<T, D>(a, s, &ax, false, norm)
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
pub fn ifftn<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let ax = resolve_axes(a.shape().len(), axes)?;
    fftn_impl::<T, D>(a, s, &ax, true, norm)
}

// ---------------------------------------------------------------------------
// Real-input convenience wrappers (issue #427)
//
// NumPy's `np.fft.fft(real_array)` accepts real arrays directly and
// promotes them to complex. ferray-fft's native `fft` requires
// `Array<Complex<T>, D>`, so these wrappers bridge real → complex by
// wrapping each element in `Complex::new(v, 0)` before calling the
// generic complex path.
//
// For the half-spectrum (Hermitian) form of a real transform use
// `rfft` / `rfftn` from `real.rs` instead — those return n/2+1 complex
// values and are ~2× faster for the real-input case.
// ---------------------------------------------------------------------------

/// Promote a real array to a `Vec<Complex<T>>` with zero imaginary parts.
fn real_to_complex_vec<T: FftFloat, D: Dimension>(a: &Array<T, D>) -> Vec<Complex<T>>
where
    Complex<T>: ferray_core::Element,
{
    a.iter()
        .map(|&v| Complex::new(v, <T as num_traits::Zero>::zero()))
        .collect()
}

/// 1-D FFT of a real-valued array. Equivalent to `np.fft.fft(real_array)`.
///
/// Auto-promotes the input to complex and delegates to [`fft`]. The
/// returned spectrum is the full-length complex FFT, not the
/// Hermitian-folded half-spectrum — use [`crate::rfft`] for that.
///
/// # Errors
/// Forwards any error from [`fft`].
pub fn fft_real<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let complex_data = real_to_complex_vec(a);
    let complex_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), complex_data)?;
    fft::<T, IxDyn>(&complex_arr, n, axis, norm)
}

/// 1-D inverse FFT returning only the real part. Equivalent to
/// `np.fft.ifft(arr).real` for the common case where the caller knows
/// the result is real-valued (e.g. the inverse of a real FFT done via
/// the full complex path).
///
/// For the Hermitian-folded inverse use [`crate::irfft`] instead.
///
/// # Errors
/// Forwards any error from [`ifft`].
pub fn ifft_real<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let spectrum = ifft::<T, D>(a, n, axis, norm)?;
    let shape = spectrum.shape().to_vec();
    let real_data: Vec<T> = spectrum.iter().map(|c| c.re).collect();
    Array::from_vec(IxDyn::new(&shape), real_data)
}

/// 2-D FFT of a real-valued array.
///
/// Auto-promotes the input to complex and delegates to [`fft2`]. For the
/// Hermitian-folded form use [`crate::rfft2`].
pub fn fft_real2<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let complex_data = real_to_complex_vec(a);
    let complex_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), complex_data)?;
    fft2::<T, IxDyn>(&complex_arr, s, axes, norm)
}

/// N-D FFT of a real-valued array.
///
/// Auto-promotes the input to complex and delegates to [`fftn`]. For the
/// Hermitian-folded form use [`crate::rfftn`].
pub fn fft_realn<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let complex_data = real_to_complex_vec(a);
    let complex_arr = Array::<Complex<T>, IxDyn>::from_vec(IxDyn::new(&shape), complex_data)?;
    fftn::<T, IxDyn>(&complex_arr, s, axes, norm)
}

// ---------------------------------------------------------------------------
// Internal N-D implementation
// ---------------------------------------------------------------------------

fn fftn_impl<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: &[usize],
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: ferray_core::Element,
{
    let shape = a.shape().to_vec();
    let sizes = resolve_shapes(&shape, axes, s)?;
    let data = borrow_complex_flat(a);

    let axes_and_sizes: Vec<(usize, Option<usize>)> = axes.iter().copied().zip(sizes).collect();

    let (new_shape, result) = fft_along_axes::<T>(&data, &shape, &axes_and_sizes, inverse, norm)?;

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
    fn fft_length_one() {
        // FFT of a length-1 array is the identity (#229).
        let a = make_1d(vec![c(7.0, -2.0)]);
        let result = fft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[1]);
        let v = result.iter().next().unwrap();
        assert!((v.re - 7.0).abs() < 1e-12);
        assert!((v.im + 2.0).abs() < 1e-12);

        // Roundtrip should also be exact for length 1.
        let recovered = ifft(&result, None, None, FftNorm::Backward).unwrap();
        let r = recovered.iter().next().unwrap();
        assert!((r.re - 7.0).abs() < 1e-12);
        assert!((r.im + 2.0).abs() < 1e-12);
    }

    #[test]
    fn fft_negative_axis_matches_explicit() {
        // axis=-1 on a 2-D array should match axis=1 (#434).
        use ferray_core::dimension::Ix2;
        let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let a = Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([2, 2]), data).unwrap();
        let neg = fft(&a, None, Some(-1), FftNorm::Backward).unwrap();
        let pos = fft(&a, None, Some(1), FftNorm::Backward).unwrap();
        assert_eq!(neg.shape(), pos.shape());
        for (n, p) in neg.iter().zip(pos.iter()) {
            assert!((n.re - p.re).abs() < 1e-12);
            assert!((n.im - p.im).abs() < 1e-12);
        }
    }

    #[test]
    fn fftn_negative_axes_matches_explicit() {
        // axes=[-2, -1] on a 2-D array should match axes=[0, 1] (#434).
        use ferray_core::dimension::Ix2;
        let data: Vec<Complex<f64>> = (0..6).map(|i| c(f64::from(i), 0.0)).collect();
        let a = Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([2, 3]), data).unwrap();
        let neg = fftn(&a, None, Some(&[-2, -1][..]), FftNorm::Backward).unwrap();
        let pos = fftn(&a, None, Some(&[0, 1][..]), FftNorm::Backward).unwrap();
        for (n, p) in neg.iter().zip(pos.iter()) {
            assert!((n.re - p.re).abs() < 1e-12);
            assert!((n.im - p.im).abs() < 1e-12);
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
    fn fft_into_writes_into_preallocated_buffer() {
        // #429: fft_into / ifft_into should match the allocating
        // versions exactly and reject mis-sized output buffers.
        let a = make_1d(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]);
        let expected = fft(&a, None, None, FftNorm::Backward).unwrap();

        let mut out: Array<Complex<f64>, IxDyn> =
            Array::from_vec(IxDyn::new(&[4]), vec![c(0.0, 0.0); 4]).unwrap();
        fft_into(&a, None, None, FftNorm::Backward, &mut out).unwrap();
        for (e, o) in expected.iter().zip(out.iter()) {
            assert!((e.re - o.re).abs() < 1e-12);
            assert!((e.im - o.im).abs() < 1e-12);
        }

        // Round-trip through ifft_into.
        let mut recovered: Array<Complex<f64>, IxDyn> =
            Array::from_vec(IxDyn::new(&[4]), vec![c(0.0, 0.0); 4]).unwrap();
        ifft_into(&out, None, None, FftNorm::Backward, &mut recovered).unwrap();
        for (orig, rec) in a.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }

        // Wrong-size out buffer is rejected.
        let mut wrong: Array<Complex<f64>, IxDyn> =
            Array::from_vec(IxDyn::new(&[3]), vec![c(0.0, 0.0); 3]).unwrap();
        let err = fft_into(&a, None, None, FftNorm::Backward, &mut wrong).unwrap_err();
        assert!(err.to_string().contains("does not match"));
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
        let data: Vec<Complex<f64>> = (0..n).map(|i| c(f64::from(i), 0.0)).collect();
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
        let data: Vec<Complex<f64>> = (0..n)
            .map(|i| c(f64::from(i), -f64::from(i) * 0.5))
            .collect();
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
        let data: Vec<Complex<f64>> = (0..16).map(|i| c(f64::from(i), 0.0)).collect();
        let a = Array::from_vec(Ix2::new([4, 4]), data).unwrap();
        let result = fft2(&a, Some(&[2, 2]), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn fftn_with_shape_roundtrip() {
        use ferray_core::dimension::Ix2;
        // 3x4 input, pad to 4x8, then ifft with same shape should recover padded version
        let data: Vec<Complex<f64>> = (0..12).map(|i| c(f64::from(i), 0.0)).collect();
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

        let energy_time: f64 = data.iter().map(|x| x.re.mul_add(x.re, x.im * x.im)).sum();
        let energy_freq: f64 = spectrum
            .iter()
            .map(|x| x.re.mul_add(x.re, x.im * x.im))
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

    // --- f32 generic path (#426) ---

    #[test]
    fn fft_ifft_f32_roundtrip() {
        // AC-426: The FFT functions must work for f32 as well as f64.
        let data: Vec<Complex<f32>> = (0..16)
            .map(|i| Complex::new(i as f32 * 0.25, (i as f32).sin()))
            .collect();
        let a = Array::from_vec(Ix1::new([16]), data.clone()).unwrap();
        let spectrum = fft::<f32, Ix1>(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[16]);
        let recovered = ifft::<f32, IxDyn>(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig.re - rec.re).abs() < 1e-4,
                "f32 re mismatch: {} vs {}",
                orig.re,
                rec.re
            );
            assert!(
                (orig.im - rec.im).abs() < 1e-4,
                "f32 im mismatch: {} vs {}",
                orig.im,
                rec.im
            );
        }
    }

    #[test]
    fn fft_f32_impulse() {
        // FFT of [1, 0, 0, 0] in f32 = [1, 1, 1, 1]
        let data = vec![
            Complex::<f32>::new(1.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
        ];
        let a = Array::from_vec(Ix1::new([4]), data).unwrap();
        let result = fft::<f32, Ix1>(&a, None, None, FftNorm::Backward).unwrap();
        for val in result.iter() {
            assert!((val.re - 1.0).abs() < 1e-6);
            assert!(val.im.abs() < 1e-6);
        }
    }

    #[test]
    fn fft2_f32_roundtrip() {
        use ferray_core::dimension::Ix2;
        let data: Vec<Complex<f32>> = (0..16)
            .map(|i| Complex::new(i as f32, -(i as f32) * 0.25))
            .collect();
        let a = Array::from_vec(Ix2::new([4, 4]), data.clone()).unwrap();
        let spectrum = fft2::<f32, Ix2>(&a, None, None, FftNorm::Backward).unwrap();
        let recovered = ifft2::<f32, IxDyn>(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o.re - r.re).abs() < 1e-4);
            assert!((o.im - r.im).abs() < 1e-4);
        }
    }

    // --- Real-input convenience wrappers (#427) ---

    #[test]
    fn fft_real_ifft_real_roundtrip_f64() {
        // AC-427: Real array should be transformable without manual promotion to Complex.
        let original = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([8]), original.clone()).unwrap();
        let spectrum = fft_real::<f64, Ix1>(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[8]);
        let recovered = ifft_real::<f64, IxDyn>(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-10, "mismatch: {o} vs {r}");
        }
    }

    #[test]
    fn fft_real_ifft_real_roundtrip_f32() {
        // Same real-input convenience, but on f32 to verify both layers work together.
        let original: Vec<f32> = (0..16).map(|i| (i as f32).mul_add(0.5, -2.0)).collect();
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([16]), original.clone()).unwrap();
        let spectrum = fft_real::<f32, Ix1>(&a, None, None, FftNorm::Backward).unwrap();
        let recovered = ifft_real::<f32, IxDyn>(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-4, "f32 mismatch: {o} vs {r}");
        }
    }

    #[test]
    fn fft_real_dc_component() {
        // FFT of real constant [1,1,1,1] should have DC = 4, rest zero.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let spectrum = fft_real::<f64, Ix1>(&a, None, None, FftNorm::Backward).unwrap();
        let vals: Vec<_> = spectrum.iter().copied().collect();
        assert!((vals[0].re - 4.0).abs() < 1e-12);
        assert!(vals[0].im.abs() < 1e-12);
        for v in &vals[1..] {
            assert!(v.re.abs() < 1e-12);
            assert!(v.im.abs() < 1e-12);
        }
    }

    #[test]
    fn fft_real2_roundtrip() {
        use ferray_core::dimension::Ix2;
        let data: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.3).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
        let spectrum = fft_real2::<f64, Ix2>(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[3, 4]);
        let recovered = ifft2::<f64, IxDyn>(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o - r.re).abs() < 1e-10);
            assert!(r.im.abs() < 1e-10);
        }
    }

    #[test]
    fn fft_realn_3d_roundtrip() {
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(|i| f64::from(i).sin()).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data.clone()).unwrap();
        let spectrum = fft_realn::<f64, Ix3>(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[2, 3, 4]);
        let recovered = ifftn::<f64, IxDyn>(&spectrum, None, None, FftNorm::Backward).unwrap();
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o - r.re).abs() < 1e-10);
            assert!(r.im.abs() < 1e-10);
        }
    }
}
