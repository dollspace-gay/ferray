// ferray-fft: Real FFTs — rfft, irfft, rfft2, irfft2, rfftn, irfftn (REQ-5..REQ-7)
//
// Real FFTs exploit Hermitian symmetry: for a real input of length n,
// the output has only n/2+1 unique complex values. The inverse operation
// takes n/2+1 complex values and produces n real values.

use num_complex::Complex;

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::nd::fft_along_axis;
use crate::norm::FftNorm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a real f64 array to flat Complex<f64> data.
fn real_to_complex_flat<D: Dimension>(a: &Array<f64, D>) -> Vec<Complex<f64>> {
    a.iter().map(|&v| Complex::new(v, 0.0)).collect()
}

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

/// Truncate the FFT output along `axis` from `n` to `n/2+1` (Hermitian symmetry).
fn truncate_hermitian(
    data: &[Complex<f64>],
    shape: &[usize],
    axis: usize,
) -> (Vec<usize>, Vec<Complex<f64>>) {
    let full_len = shape[axis];
    let half_len = full_len / 2 + 1;

    let mut new_shape = shape.to_vec();
    new_shape[axis] = half_len;

    let strides = compute_strides(shape);
    let new_strides = compute_strides(&new_shape);
    let new_total: usize = new_shape.iter().product();

    let output: Vec<Complex<f64>> = (0..new_total)
        .map(|flat_idx| {
            let multi = flat_to_multi(flat_idx, &new_shape, &new_strides);
            // Same multi-index in the source (only axis dimension is truncated)
            let src_idx: usize = multi
                .iter()
                .zip(strides.iter())
                .map(|(&m, &s)| m * s as usize)
                .sum();
            data[src_idx]
        })
        .collect();

    (new_shape, output)
}

/// Extend Hermitian-symmetric data from n/2+1 complex values back to n.
fn extend_hermitian(
    data: &[Complex<f64>],
    shape: &[usize],
    axis: usize,
    n: usize,
) -> (Vec<usize>, Vec<Complex<f64>>) {
    let half_len = shape[axis];

    let mut new_shape = shape.to_vec();
    new_shape[axis] = n;

    let strides = compute_strides(shape);
    let new_strides = compute_strides(&new_shape);
    let new_total: usize = new_shape.iter().product();

    let output: Vec<Complex<f64>> = (0..new_total)
        .map(|flat_idx| {
            let multi = flat_to_multi(flat_idx, &new_shape, &new_strides);
            let axis_idx = multi[axis];

            if axis_idx < half_len {
                // Direct copy
                let src_idx: usize = multi
                    .iter()
                    .zip(strides.iter())
                    .map(|(&m, &s)| m * s as usize)
                    .sum();
                data[src_idx]
            } else {
                // Hermitian symmetry: X[n-k] = conj(X[k])
                let mirror_axis_idx = n - axis_idx;
                if mirror_axis_idx < half_len {
                    let mut src_multi = multi;
                    src_multi[axis] = mirror_axis_idx;
                    let src_idx: usize = src_multi
                        .iter()
                        .zip(strides.iter())
                        .map(|(&m, &s)| m * s as usize)
                        .sum();
                    data[src_idx].conj()
                } else {
                    // Zero (shouldn't happen for valid Hermitian data)
                    Complex::new(0.0, 0.0)
                }
            }
        })
        .collect();

    (new_shape, output)
}

fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    let mut strides = vec![0isize; ndim];
    if ndim == 0 {
        return strides;
    }
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

fn flat_to_multi(flat_idx: usize, shape: &[usize], strides: &[isize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut multi = vec![0usize; ndim];
    let mut remaining = flat_idx;
    for d in 0..ndim {
        if strides[d] != 0 {
            multi[d] = remaining / strides[d] as usize;
            remaining %= strides[d] as usize;
        }
    }
    multi
}

// ---------------------------------------------------------------------------
// 1-D real FFT (REQ-5)
// ---------------------------------------------------------------------------

/// Compute the one-dimensional discrete Fourier Transform of a real-valued input.
///
/// Analogous to `numpy.fft.rfft`. Since the input is real, the output
/// exhibits Hermitian symmetry and only the first `n/2 + 1` complex
/// coefficients are returned.
///
/// # Parameters
/// - `a`: Input real-valued array.
/// - `n`: Length of the transformed axis. If `None`, uses the input length.
/// - `axis`: Axis along which to compute. Defaults to the last axis.
/// - `norm`: Normalization mode.
///
/// # Errors
/// Returns an error if `axis` is out of bounds or `n` is 0.
pub fn rfft<D: Dimension>(
    a: &Array<f64, D>,
    n: Option<usize>,
    axis: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    let ax = resolve_axis(ndim, axis)?;

    let fft_len = n.unwrap_or(shape[ax]);
    if fft_len == 0 {
        return Err(FerrayError::invalid_value("FFT length must be > 0"));
    }

    // Convert real to complex and compute full FFT
    let complex_data = real_to_complex_flat(a);
    let (full_shape, full_result) =
        fft_along_axis(&complex_data, &shape, ax, Some(fft_len), false, norm)?;

    // Truncate to n/2+1 along the transform axis
    let (out_shape, out_data) = truncate_hermitian(&full_result, &full_shape, ax);

    Array::from_vec(IxDyn::new(&out_shape), out_data)
}

/// Compute the inverse of `rfft`, producing real-valued output.
///
/// Analogous to `numpy.fft.irfft`. Takes `n/2 + 1` complex values and
/// produces `n` real values by exploiting Hermitian symmetry.
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
pub fn irfft<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    n: Option<usize>,
    axis: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<Array<f64, IxDyn>> {
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

    // Extend Hermitian-symmetric data to full length
    let complex_data: Vec<Complex<f64>> = a.iter().copied().collect();
    let (extended_shape, extended_data) = extend_hermitian(&complex_data, &shape, ax, output_len);

    // Compute inverse FFT on the full-length data
    let (result_shape, result_data) =
        fft_along_axis(&extended_data, &extended_shape, ax, None, true, norm)?;

    // Extract real parts
    let real_data: Vec<f64> = result_data.iter().map(|c| c.re).collect();

    Array::from_vec(IxDyn::new(&result_shape), real_data)
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
pub fn rfft2<D: Dimension>(
    a: &Array<f64, D>,
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
                    "rfft2 requires at least 2 dimensions",
                ));
            }
            vec![ndim - 2, ndim - 1]
        }
    };
    rfftn_impl(a, s, &axes, norm)
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
pub fn irfft2<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<f64, IxDyn>> {
    let ndim = a.shape().len();
    let axes = match axes {
        Some(ax) => ax.to_vec(),
        None => {
            if ndim < 2 {
                return Err(FerrayError::invalid_value(
                    "irfft2 requires at least 2 dimensions",
                ));
            }
            vec![ndim - 2, ndim - 1]
        }
    };
    irfftn_impl(a, s, &axes, norm)
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
pub fn rfftn<D: Dimension>(
    a: &Array<f64, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    let ax = resolve_axes(a.shape().len(), axes)?;
    rfftn_impl(a, s, &ax, norm)
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
pub fn irfftn<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: FftNorm,
) -> FerrayResult<Array<f64, IxDyn>> {
    let ax = resolve_axes(a.shape().len(), axes)?;
    irfftn_impl(a, s, &ax, norm)
}

// ---------------------------------------------------------------------------
// Internal N-D implementations
// ---------------------------------------------------------------------------

fn rfftn_impl<D: Dimension>(
    a: &Array<f64, D>,
    s: Option<&[usize]>,
    axes: &[usize],
    norm: FftNorm,
) -> FerrayResult<Array<Complex<f64>, IxDyn>> {
    if axes.is_empty() {
        // No axes to transform — just convert to complex
        let data: Vec<Complex<f64>> = a.iter().map(|&v| Complex::new(v, 0.0)).collect();
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

    let complex_data = real_to_complex_flat(a);
    let mut current_data = complex_data;
    let mut current_shape = input_shape;

    // For all axes except the last: do a full complex FFT
    for (i, &ax) in axes.iter().enumerate() {
        let n = sizes[i];
        if i < axes.len() - 1 {
            // Full complex FFT
            let (new_shape, new_data) =
                fft_along_axis(&current_data, &current_shape, ax, n, false, norm)?;
            current_shape = new_shape;
            current_data = new_data;
        } else {
            // Last axis: full FFT then truncate
            let fft_len = n.unwrap_or(current_shape[ax]);
            let (full_shape, full_data) = fft_along_axis(
                &current_data,
                &current_shape,
                ax,
                Some(fft_len),
                false,
                norm,
            )?;
            let (out_shape, out_data) = truncate_hermitian(&full_data, &full_shape, ax);
            current_shape = out_shape;
            current_data = out_data;
        }
    }

    Array::from_vec(IxDyn::new(&current_shape), current_data)
}

fn irfftn_impl<D: Dimension>(
    a: &Array<Complex<f64>, D>,
    s: Option<&[usize]>,
    axes: &[usize],
    norm: FftNorm,
) -> FerrayResult<Array<f64, IxDyn>> {
    if axes.is_empty() {
        let data: Vec<f64> = a.iter().map(|c| c.re).collect();
        return Array::from_vec(IxDyn::new(a.shape()), data);
    }

    let input_shape = a.shape().to_vec();

    // Determine the output sizes for each axis
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
        None => {
            // For all axes except the last: use input shape
            // For the last axis: n = 2*(input_len - 1)
            let mut result = Vec::with_capacity(axes.len());
            for (i, &ax) in axes.iter().enumerate() {
                if i < axes.len() - 1 {
                    result.push(Some(input_shape[ax]));
                } else {
                    result.push(Some(2 * (input_shape[ax] - 1)));
                }
            }
            result
        }
    };

    let complex_data: Vec<Complex<f64>> = a.iter().copied().collect();
    let mut current_data = complex_data;
    let mut current_shape = input_shape;

    // Process axes in order
    for (i, &ax) in axes.iter().enumerate() {
        let n = sizes[i];
        if i < axes.len() - 1 {
            // All but last axis: inverse complex FFT
            let (new_shape, new_data) =
                fft_along_axis(&current_data, &current_shape, ax, n, true, norm)?;
            current_shape = new_shape;
            current_data = new_data;
        } else {
            // Last axis: extend Hermitian then inverse FFT
            let output_len = n.unwrap_or(2 * (current_shape[ax] - 1));
            let (ext_shape, ext_data) =
                extend_hermitian(&current_data, &current_shape, ax, output_len);
            let (result_shape, result_data) =
                fft_along_axis(&ext_data, &ext_shape, ax, None, true, norm)?;
            current_shape = result_shape;
            current_data = result_data;
        }
    }

    // Extract real parts
    let real_data: Vec<f64> = current_data.iter().map(|c| c.re).collect();
    Array::from_vec(IxDyn::new(&current_shape), real_data)
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
            assert!((o - r).abs() < 1e-10, "{} vs {}", o, r);
        }
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
            assert!((o - r).abs() < 1e-10, "{} vs {}", o, r);
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
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
        let spectrum = rfft(&a, None, Some(0), FftNorm::Backward).unwrap();
        let recovered = irfft(&spectrum, Some(3), Some(0), FftNorm::Backward).unwrap();
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in data.iter().zip(rec_data.iter()) {
            assert!(
                (o - r).abs() < 1e-9,
                "axis0 roundtrip: {} vs {}",
                o,
                r
            );
        }
    }

    #[test]
    fn rfft_irfft_roundtrip_axis1() {
        use ferray_core::dimension::Ix2;
        // 3x4 array, rfft/irfft along axis 1 (default)
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
        let spectrum = rfft(&a, None, Some(1), FftNorm::Backward).unwrap();
        assert_eq!(spectrum.shape()[0], 3);
        assert_eq!(spectrum.shape()[1], 3); // 4/2+1 = 3
        let recovered = irfft(&spectrum, Some(4), Some(1), FftNorm::Backward).unwrap();
        let rec_data: Vec<f64> = recovered.iter().copied().collect();
        for (o, r) in data.iter().zip(rec_data.iter()) {
            assert!(
                (o - r).abs() < 1e-9,
                "axis1 roundtrip: {} vs {}",
                o,
                r
            );
        }
    }
}
