// ferray-fft: Real FFTs — rfft, irfft, rfft2, irfft2, rfftn, irfftn (REQ-5..REQ-7)
//
// Real FFTs exploit Hermitian symmetry: for a real input of length n, the
// output has only n/2+1 unique complex values. The inverse operation takes
// n/2+1 complex values and produces n real values.
//
// The 1-D real→complex transform is delegated to the `realfft` crate (via
// `rfft_along_axis` in nd.rs) which implements the half-size complex FFT
// trick — roughly 2× faster than promoting to complex and running a full
// complex FFT. See issue #432 for background.

use num_complex::Complex;

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::axes::{resolve_axes, resolve_axis};
use crate::float::FftFloat;
use crate::nd::{fft_along_axis, irfft_along_axis, rfft_along_axis};
use crate::norm::FftNorm;

// ---------------------------------------------------------------------------
// 1-D real FFT (REQ-5)
// ---------------------------------------------------------------------------

/// Compute the one-dimensional discrete Fourier Transform of a real-valued input.
///
/// Analogous to `numpy.fft.rfft`. Since the input is real, the output
/// exhibits Hermitian symmetry and only the first `n/2 + 1` complex
/// coefficients are returned.
///
/// Uses the dedicated real-to-complex FFT from the `realfft` crate
/// (issue #432), which is ~2× faster than promoting to complex first.
///
/// # Parameters
/// - `a`: Input real-valued array.
/// - `n`: Length of the transformed axis. If `None`, uses the input length.
/// - `axis`: Axis along which to compute. Defaults to the last axis.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn rfft<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;

    // Materialize the input data as a flat `&[T]`. The underlying array
    // may already be contiguous — if so we could borrow it, but `iter()`
    // handles non-contiguous views too.
    let real_data: Vec<T> = a.iter().copied().collect();

    // Delegate to the realfft-backed lane helper, which produces the
    // n/2+1 Hermitian-folded output directly — no full-size complex FFT
    // followed by truncation.
    let (out_shape, out_data) = rfft_along_axis::<T>(&real_data, &shape, ax, n, norm)?;
    Array::from_vec(IxDyn::new(&out_shape), out_data)
}

/// In-place variant of [`rfft`] writing into a pre-allocated `out`
/// buffer (#429).
///
/// # Errors
/// - Same as [`rfft`] for axis/n validation.
/// - `FerrayError::ShapeMismatch` if `out`'s shape doesn't match the
///   rfft output shape (`n/2 + 1` along the transform axis).
/// - `FerrayError::InvalidValue` if `out` is non-contiguous.
pub fn rfft_into<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
    out: &mut Array<Complex<T>, IxDyn>,
) -> FerrayResult<()>
where
    Complex<T>: Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let real_data: Vec<T> = a.iter().copied().collect();
    let (out_shape, out_data) = rfft_along_axis::<T>(&real_data, &shape, ax, n, norm)?;
    if out.shape() != out_shape.as_slice() {
        return Err(FerrayError::shape_mismatch(format!(
            "rfft_into: out shape {:?} does not match rfft output shape {:?}",
            out.shape(),
            out_shape
        )));
    }
    let out_slice = out
        .as_slice_mut()
        .ok_or_else(|| FerrayError::invalid_value("rfft_into requires a contiguous out buffer"))?;
    out_slice.copy_from_slice(&out_data);
    Ok(())
}

/// Compute the inverse of `rfft`, producing real-valued output.
///
/// Analogous to `numpy.fft.irfft`. Takes `n/2 + 1` complex values and
/// produces `n` real values by exploiting Hermitian symmetry.
///
/// Uses the dedicated complex-to-real FFT from the `realfft` crate
/// (issue #432), which is ~2× faster than extending to full Hermitian
/// length and running a full inverse complex FFT.
///
/// # Parameters
/// - `a`: Input complex array (Hermitian-symmetric spectrum, typically n/2+1 values).
/// - `n`: Length of the output along the transform axis. If `None`,
///   uses `2 * (input_length - 1)`.
/// - `axis`: Axis along which to compute. Defaults to the last axis.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn irfft<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;

    let half_len = shape[ax];
    let output_len = n.unwrap_or(2 * (half_len - 1));
    if output_len == 0 {
        return Err(FerrayError::invalid_value(
            "irfft output length must be > 0",
        ));
    }

    // Delegate to the realfft-backed lane helper, which runs the
    // complex-to-real transform directly without extending to full length.
    let complex_data: Vec<Complex<T>> = a.iter().copied().collect();
    let (out_shape, out_data) = irfft_along_axis::<T>(&complex_data, &shape, ax, output_len, norm)?;
    Array::from_vec(IxDyn::new(&out_shape), out_data)
}

/// In-place variant of [`irfft`] writing real-valued output into a
/// pre-allocated `out` buffer (#429).
///
/// # Errors
/// - Same as [`irfft`] for axis/n validation.
/// - `FerrayError::ShapeMismatch` if `out`'s shape doesn't match the
///   irfft output shape.
/// - `FerrayError::InvalidValue` if `out` is non-contiguous.
pub fn irfft_into<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: FftNorm,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
where
    Complex<T>: Element,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;
    let half_len = shape[ax];
    let output_len = n.unwrap_or(2 * (half_len - 1));
    if output_len == 0 {
        return Err(FerrayError::invalid_value(
            "irfft_into output length must be > 0",
        ));
    }
    let complex_data: Vec<Complex<T>> = a.iter().copied().collect();
    let (out_shape, out_data) =
        irfft_along_axis::<T>(&complex_data, &shape, ax, output_len, norm)?;
    if out.shape() != out_shape.as_slice() {
        return Err(FerrayError::shape_mismatch(format!(
            "irfft_into: out shape {:?} does not match irfft output shape {:?}",
            out.shape(),
            out_shape
        )));
    }
    let out_slice = out
        .as_slice_mut()
        .ok_or_else(|| FerrayError::invalid_value("irfft_into requires a contiguous out buffer"))?;
    out_slice.copy_from_slice(&out_data);
    Ok(())
}

// ---------------------------------------------------------------------------
// 2-D real FFT (REQ-7)
// ---------------------------------------------------------------------------

/// Compute the 2-dimensional real FFT.
///
/// Analogous to `numpy.fft.rfft2`. The last transform axis produces
/// `n/2+1` complex values (Hermitian symmetry).
///
/// # Parameters
/// - `a`: Input real-valued array (at least 2 dimensions).
/// - `s`: Output shape along transform axes. Defaults to input shape.
/// - `axes`: Axes to transform. Defaults to the last 2 axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if axes are invalid or the array has fewer than 2 dimensions.
pub fn rfft2<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    let ndim = a.shape().len();
    let axes = if let Some(ax) = axes {
        resolve_axes(ndim, Some(ax))?
    } else {
        if ndim < 2 {
            return Err(FerrayError::invalid_value(
                "rfft2 requires at least 2 dimensions",
            ));
        }
        vec![ndim - 2, ndim - 1]
    };
    rfftn_impl::<T, D>(a, s, &axes, norm)
}

/// Compute the 2-dimensional inverse real FFT.
///
/// Analogous to `numpy.fft.irfft2`.
///
/// # Parameters
/// - `a`: Input complex array.
/// - `s`: Output shape along transform axes.
/// - `axes`: Axes to transform. Defaults to the last 2 axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if axes are invalid or the array has fewer than 2 dimensions.
pub fn irfft2<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    let ndim = a.shape().len();
    let axes = if let Some(ax) = axes {
        resolve_axes(ndim, Some(ax))?
    } else {
        if ndim < 2 {
            return Err(FerrayError::invalid_value(
                "irfft2 requires at least 2 dimensions",
            ));
        }
        vec![ndim - 2, ndim - 1]
    };
    irfftn_impl::<T, D>(a, s, &axes, norm)
}

// ---------------------------------------------------------------------------
// N-D real FFT (REQ-7)
// ---------------------------------------------------------------------------

/// Compute the N-dimensional real FFT.
///
/// Analogous to `numpy.fft.rfftn`. The last axis in the transform
/// produces `n/2+1` complex values (Hermitian symmetry).
///
/// # Parameters
/// - `a`: Input real-valued array.
/// - `s`: Output shape along transform axes. Defaults to input shape.
/// - `axes`: Axes to transform. Defaults to all axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if axes are invalid.
pub fn rfftn<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    let ax = resolve_axes(a.shape().len(), axes)?;
    rfftn_impl::<T, D>(a, s, &ax, norm)
}

/// Compute the N-dimensional inverse real FFT.
///
/// Analogous to `numpy.fft.irfftn`.
///
/// # Parameters
/// - `a`: Input complex array.
/// - `s`: Output shape along transform axes.
/// - `axes`: Axes to transform. Defaults to all axes.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if axes are invalid.
pub fn irfftn<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    let ax = resolve_axes(a.shape().len(), axes)?;
    irfftn_impl::<T, D>(a, s, &ax, norm)
}

// ---------------------------------------------------------------------------
// Internal N-D implementations
// ---------------------------------------------------------------------------

fn rfftn_impl<T: FftFloat, D: Dimension>(
    a: &Array<T, D>,
    s: Option<&[usize]>,
    axes: &[usize],
    norm: FftNorm,
) -> FerrayResult<Array<Complex<T>, IxDyn>>
where
    Complex<T>: Element,
{
    if axes.is_empty() {
        // No axes to transform — just convert to complex
        let data: Vec<Complex<T>> = a
            .iter()
            .map(|&v| Complex::new(v, <T as num_traits::Zero>::zero()))
            .collect();
        return Array::from_vec(IxDyn::new(a.shape()), data);
    }

    let input_shape = a.shape().to_vec();
    let sizes: Vec<Option<usize>> = match s {
        Some(sizes) => {
            if sizes.len() != axes.len() {
                return Err(FerrayError::invalid_value(format!(
                    "shape parameter length {} does not match axes length {}",
                    sizes.len(),
                    axes.len(),
                )));
            }
            sizes.iter().map(|&sz| Some(sz)).collect()
        }
        None => axes.iter().map(|&ax| Some(input_shape[ax])).collect(),
    };

    // NumPy's rfftn runs a real→complex transform on the LAST axis first
    // (producing the Hermitian-folded dimension) and then complex transforms
    // on the remaining axes. We do the same: start with the real input,
    // apply rfft_along_axis on the final axis, then iterate the remaining
    // axes with the complex path.
    let last_idx = axes.len() - 1;
    let last_ax = axes[last_idx];
    let last_n = sizes[last_idx];

    let real_data: Vec<T> = a.iter().copied().collect();
    let (mut current_shape, mut current_data) =
        rfft_along_axis::<T>(&real_data, &input_shape, last_ax, last_n, norm)?;

    // Remaining axes run as complex forward FFTs on the intermediate.
    for i in 0..last_idx {
        let ax = axes[i];
        let n = sizes[i];
        let (new_shape, new_data) =
            fft_along_axis::<T>(&current_data, &current_shape, ax, n, false, norm)?;
        current_shape = new_shape;
        current_data = new_data;
    }

    Array::from_vec(IxDyn::new(&current_shape), current_data)
}

fn irfftn_impl<T: FftFloat, D: Dimension>(
    a: &Array<Complex<T>, D>,
    s: Option<&[usize]>,
    axes: &[usize],
    norm: FftNorm,
) -> FerrayResult<Array<T, IxDyn>>
where
    Complex<T>: Element,
{
    if axes.is_empty() {
        let data: Vec<T> = a.iter().map(|c| c.re).collect();
        return Array::from_vec(IxDyn::new(a.shape()), data);
    }

    let input_shape = a.shape().to_vec();
    let last_idx = axes.len() - 1;

    // Determine the output sizes for each axis.
    let sizes: Vec<usize> = if let Some(sizes) = s {
        if sizes.len() != axes.len() {
            return Err(FerrayError::invalid_value(format!(
                "shape parameter length {} does not match axes length {}",
                sizes.len(),
                axes.len(),
            )));
        }
        sizes.to_vec()
    } else {
        // For all axes except the last: use input shape (complex FFT
        // is shape-preserving along each axis unless overridden).
        // For the last axis: output is `2 * (input_len - 1)`.
        let mut result = Vec::with_capacity(axes.len());
        for (i, &ax) in axes.iter().enumerate() {
            if i < last_idx {
                result.push(input_shape[ax]);
            } else {
                result.push(2 * (input_shape[ax] - 1));
            }
        }
        result
    };

    // NumPy's irfftn runs complex inverse FFTs on axes[0..last] first, then
    // the complex-to-real inverse transform on axes[last]. We match that
    // order so the Hermitian-folded axis stays folded through the other
    // transforms and is only un-folded (to real) at the very end.
    let mut current_data: Vec<Complex<T>> = a.iter().copied().collect();
    let mut current_shape = input_shape;

    for i in 0..last_idx {
        let ax = axes[i];
        let n = Some(sizes[i]);
        let (new_shape, new_data) =
            fft_along_axis::<T>(&current_data, &current_shape, ax, n, true, norm)?;
        current_shape = new_shape;
        current_data = new_data;
    }

    // Final step: complex-to-real inverse transform on the last axis via
    // the realfft-backed lane helper.
    let last_ax = axes[last_idx];
    let output_len = sizes[last_idx];
    let (final_shape, final_data) =
        irfft_along_axis::<T>(&current_data, &current_shape, last_ax, output_len, norm)?;

    Array::from_vec(IxDyn::new(&final_shape), final_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn make_real_1d(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn rfft_basic() {
        // AC-3: rfft of a real signal returns n/2+1 complex values
        let a = make_real_1d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = rfft(&a, None, None, FftNorm::Backward).unwrap();
        // n=8, so output should have 8/2+1 = 5 values
        assert_eq!(result.shape(), &[5]);
    }

    #[test]
    fn rfft_impulse() {
        // rfft of [1, 0, 0, 0] should give [1, 1, 1] (n/2+1 = 3)
        let a = make_real_1d(vec![1.0, 0.0, 0.0, 0.0]);
        let result = rfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[3]);
        for val in result.iter() {
            assert!((val.re - 1.0).abs() < 1e-12);
            assert!(val.im.abs() < 1e-12);
        }
    }

    #[test]
    fn rfft_irfft_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = make_real_1d(original.clone());
        let spectrum = rfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[5]); // n/2+1

        let recovered = irfft(&spectrum, Some(8), None, FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[8]);
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in original.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-10, "{o} vs {r}");
        }
    }

    #[test]
    fn rfft_irfft_into_roundtrip() {
        // #429: pre-allocated _into variants for real FFTs.
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = make_real_1d(original.clone());

        let mut spectrum: Array<Complex<f64>, IxDyn> =
            Array::from_vec(IxDyn::new(&[5]), vec![Complex::new(0.0, 0.0); 5]).unwrap();
        rfft_into(&a, None, None, FftNorm::Backward, &mut spectrum).unwrap();

        let mut recovered: Array<f64, IxDyn> =
            Array::from_vec(IxDyn::new(&[8]), vec![0.0; 8]).unwrap();
        irfft_into(&spectrum, Some(8), None, FftNorm::Backward, &mut recovered).unwrap();
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-10);
        }

        // Wrong-size out buffer is rejected.
        let mut wrong: Array<Complex<f64>, IxDyn> =
            Array::from_vec(IxDyn::new(&[4]), vec![Complex::new(0.0, 0.0); 4]).unwrap();
        let err = rfft_into(&a, None, None, FftNorm::Backward, &mut wrong).unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn rfft_with_n() {
        // Zero-pad [1, 2] to length 8
        let a = make_real_1d(vec![1.0, 2.0]);
        let result = rfft(&a, Some(8), None, FftNorm::Backward).unwrap();
        assert_eq!(result.shape(), &[5]); // 8/2+1
    }

    #[test]
    fn rfft2_basic() {
        use ferray_core::dimension::Ix2;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = Array::from_vec(Ix2::new([2, 2]), data).unwrap();
        let result = rfft2(&a, None, None, FftNorm::Backward).unwrap();
        // Last axis: 2/2+1=2, first axis stays 2 -> shape [2, 2]
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn rfft_irfft_roundtrip_odd() {
        // Test with odd-length signal
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let a = make_real_1d(original.clone());
        let spectrum = rfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[3]); // 5/2+1 = 3

        let recovered = irfft(&spectrum, Some(5), None, FftNorm::Backward).unwrap();
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in original.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-10, "{o} vs {r}");
        }
    }

    #[test]
    fn rfft_along_axis0_2d() {
        use ferray_core::dimension::Ix2;
        // 2x4 real array, rfft along axis 0 (columns)
        // Each column is length 2, so rfft output has 2 bins (2/2+1=2)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 4]), data).unwrap();
        let result = rfft(&a, None, Some(0), FftNorm::Backward).unwrap();
        // Output shape: (2, 4) -> rfft along axis 0 -> (2/2+1, 4) = (2, 4)
        assert_eq!(result.shape(), &[2, 4]);
    }

    #[test]
    fn rfft_irfft_roundtrip_axis0() {
        use ferray_core::dimension::Ix2;
        // 3x4 array, rfft/irfft along axis 0
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
        let spectrum = rfft(&a, None, Some(0), FftNorm::Backward).unwrap();
        let recovered = irfft(&spectrum, Some(3), Some(0), FftNorm::Backward).unwrap();
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in data.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-9, "axis0 roundtrip: {o} vs {r}");
        }
    }

    #[test]
    fn rfft_irfft_roundtrip_axis1() {
        use ferray_core::dimension::Ix2;
        // 3x4 array, rfft/irfft along axis 1 (default)
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
        let spectrum = rfft(&a, None, Some(1), FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape()[0], 3);
        assert_eq!(spectrum.shape()[1], 3); // 4/2+1 = 3
        let recovered = irfft(&spectrum, Some(4), Some(1), FftNorm::Backward).unwrap();
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in data.iter().zip(rec_data.iter()) {
            assert!((o - r).abs() < 1e-9, "axis1 roundtrip: {o} vs {r}");
        }
    }

    // -----------------------------------------------------------------------
    // Regression guards for the realfft-backed rfft path (#432)
    //
    // These test the multi-lane par_chunks_mut code path in rfft_along_axis
    // with specific mathematical invariants that let us verify the
    // realfft-backed implementation matches the previous promote-to-complex
    // approach.
    // -----------------------------------------------------------------------

    /// Spectrum of a real sinusoid should have a single pair of spikes.
    /// For a cosine at frequency k in a signal of length n, rfft(cos) has:
    ///   - bin k: n/2 + 0j (real, positive)
    ///   - all other bins: ~0
    #[test]
    fn rfft_single_cosine_matches_analytical() {
        let n = 16;
        let k = 3; // frequency
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64).cos())
            .collect();
        let a = make_real_1d(data);
        let spectrum = rfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[n / 2 + 1]);

        let bins: Vec<Complex<f64>> = spectrum.iter().copied().collect();
        for (i, bin) in bins.iter().enumerate() {
            if i == k {
                // Bin k should have magnitude n/2 (real, positive).
                assert!(
                    (bin.re - (n as f64 / 2.0)).abs() < 1e-10,
                    "bin {} real part = {}, expected {}",
                    i,
                    bin.re,
                    n as f64 / 2.0
                );
                assert!(bin.im.abs() < 1e-10);
            } else {
                // All other bins should be near zero.
                assert!(bin.norm() < 1e-10, "bin {i} should be ~0, got {bin:?}");
            }
        }
    }

    /// Parseval's theorem: sum of |x[i]|^2 equals sum of |X[k]|^2 / n for
    /// "backward" normalization (the default NumPy/ferray convention).
    /// For rfft we need to account for the folded representation: bins 0 and
    /// n/2 (if present) contribute once; others contribute twice (since
    /// X[n-k] = conj(X[k]) is not stored but carries the same magnitude).
    #[test]
    fn rfft_parseval_holds_for_multi_lane() {
        use ferray_core::dimension::Ix2;
        // 4 lanes of length 16 each. Each lane is a different signal.
        let rows = 4usize;
        let cols = 16usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i as f64).sin() + (i as f64 * 0.3).cos()) * 2.0)
            .collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data.clone()).unwrap();
        let spectrum = rfft(&a, None, Some(1), FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[rows, cols / 2 + 1]);

        for row in 0..rows {
            // Sum of |x[i]|^2 for this lane.
            let lane_start = row * cols;
            let time_energy: f64 = data[lane_start..lane_start + cols]
                .iter()
                .map(|&v| v * v)
                .sum();

            // Sum of |X[k]|^2 for this lane in the folded representation.
            let spec_row = row * (cols / 2 + 1);
            let half_len = cols / 2 + 1;
            let mut freq_energy = 0.0;
            for k in 0..half_len {
                let bin = spectrum.iter().nth(spec_row + k).copied().unwrap();
                let mag_sq = bin.norm_sqr();
                // Bins 0 and cols/2 (for even n) count once; others count
                // twice to account for their unstored conjugate.
                if k == 0 || (cols % 2 == 0 && k == cols / 2) {
                    freq_energy += mag_sq;
                } else {
                    freq_energy += 2.0 * mag_sq;
                }
            }
            // Parseval: sum |X[k]|^2 = n * sum |x[i]|^2 for "backward" norm.
            let expected = cols as f64 * time_energy;
            assert!(
                ((freq_energy - expected) / expected).abs() < 1e-9,
                "lane {row}: freq energy {freq_energy} vs expected {expected} (time energy = {time_energy})"
            );
        }
    }

    /// Odd-length signal: realfft handles odd n via the halved representation
    /// `n/2 + 1 = (n+1)/2`. Validate against a delta input.
    #[test]
    fn rfft_odd_length_impulse() {
        // Odd-length impulse [1, 0, 0, 0, 0] → all 3 bins are 1.0+0j
        let a = make_real_1d(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let spectrum = rfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[3]);
        for bin in spectrum.iter() {
            assert!((bin.re - 1.0).abs() < 1e-12);
            assert!(bin.im.abs() < 1e-12);
        }
    }

    /// Zero-padding through the multi-lane path.
    #[test]
    fn rfft_multi_lane_with_zero_padding() {
        use ferray_core::dimension::Ix2;
        // 3 lanes of length 2, padded to 8 → output shape (3, 5).
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), data).unwrap();
        let spectrum = rfft(&a, Some(8), Some(1), FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape(), &[3, 5]);

        // DC bin of row r is (data[2r] + data[2r+1]) since other positions
        // are zero-padded.
        let bins: Vec<Complex<f64>> = spectrum.iter().copied().collect();
        assert!((bins[0].re - (1.0 + 2.0)).abs() < 1e-12);
        assert!((bins[5].re - (3.0 + 4.0)).abs() < 1e-12);
        assert!((bins[10].re - (5.0 + 6.0)).abs() < 1e-12);
    }

    /// Roundtrip through the multi-lane inverse path (irfft along axis 0).
    #[test]
    fn irfft_multi_lane_axis0_roundtrip() {
        use ferray_core::dimension::Ix2;
        // 6×4 array, rfft/irfft along axis 0 → lanes of length 6.
        let rows = 6usize;
        let cols = 4usize;
        let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64).sqrt()).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data.clone()).unwrap();

        let spectrum = rfft(&a, None, Some(0), FftNorm::Backward).unwrap();
        // rfft along axis 0 with n=6 → output shape (4, 4) since 6/2+1 = 4
        assert_eq!(spectrum.shape(), &[4, 4]);

        let recovered = irfft(&spectrum, Some(rows), Some(0), FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[rows, cols]);
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (i, (o, r)) in data.iter().zip(rec_data.iter()).enumerate() {
            assert!((o - r).abs() < 1e-10, "index {i}: expected {o}, got {r}");
        }
    }

    // --- f32 generic path (#426) ---

    #[test]
    fn rfft_irfft_f32_roundtrip() {
        // Real FFT should work for f32 as well as f64.
        let original: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).cos()).collect();
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([16]), original.clone()).unwrap();
        let spectrum = rfft::<f32, Ix1>(&a, None, None, FftNorm::Backward).unwrap();
        // n/2+1 = 9
        assert_eq!(spectrum.shape(), &[9]);
        let recovered = irfft::<f32, IxDyn>(&spectrum, Some(16), None, FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[16]);
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-5, "f32 rfft/irfft mismatch: {o} vs {r}");
        }
    }

    #[test]
    fn rfft2_f32_roundtrip() {
        use ferray_core::dimension::Ix2;
        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25).collect();
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([4, 4]), data.clone()).unwrap();
        let spectrum = rfft2::<f32, Ix2>(&a, None, None, FftNorm::Backward).unwrap();
        // rfft2 on 4×4 → 4 × (4/2+1) = 4 × 3
        assert_eq!(spectrum.shape(), &[4, 3]);
        let recovered =
            irfft2::<f32, IxDyn>(&spectrum, Some(&[4, 4]), None, FftNorm::Backward).unwrap();
        assert_eq!(recovered.shape(), &[4, 4]);
        for (o, r) in data.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-5);
        }
    }
}
