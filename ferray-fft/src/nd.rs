// ferray-fft: Multi-dimensional FFT via iterated 1D transforms along axes (REQ-3, REQ-4)
//
// The core strategy: for each axis in the list, extract every 1-D "lane"
// along that axis, run the 1-D FFT on each lane, and write the results
// back. For N-D arrays this is done sequentially per axis but lanes
// within an axis are processed in parallel via Rayon.

use num_complex::Complex;
use rayon::prelude::*;

use ferray_core::error::{FerrayError, FerrayResult};

use crate::norm::FftNorm;
use crate::plan::get_cached_plan;

/// Apply a 1-D FFT along a single axis of a multi-dimensional array
/// stored in row-major (C) order.
///
/// `shape` and `data` describe the array. The transform is applied
/// along `axis`, optionally with zero-padding/truncation to `n` points.
/// The returned `(new_shape, new_data)` reflect the (possibly changed)
/// size along the transformed axis.
pub(crate) fn fft_along_axis(
    data: &[Complex<f64>],
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<f64>>)> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }

    let axis_len = shape[axis];
    let fft_len = n.unwrap_or(axis_len);
    if fft_len == 0 {
        return Err(FerrayError::invalid_value("FFT length must be > 0"));
    }

    // Total elements
    let total = shape.iter().product::<usize>();
    if total == 0 {
        let mut new_shape = shape.to_vec();
        new_shape[axis] = fft_len;
        let new_total: usize = new_shape.iter().product();
        return Ok((new_shape, vec![Complex::new(0.0, 0.0); new_total]));
    }

    // Fast path: 1D array — skip all lane machinery
    if ndim == 1 {
        return fft_1d_fast(data, fft_len, axis_len, inverse, norm);
    }

    let num_lanes = total / axis_len;

    // Compute strides for the input shape (row-major)
    let strides = compute_strides(shape);

    // Build output shape
    let mut new_shape = shape.to_vec();
    new_shape[axis] = fft_len;
    let new_strides = compute_strides(&new_shape);
    let new_total: usize = new_shape.iter().product();

    // Extract lane indices: for each lane, compute the starting multi-index
    // (with axis dimension = 0), then iterate along axis.
    let lane_starts = compute_lane_starts(shape, &strides, axis, num_lanes);

    // Get the cached plan once for this size
    let plan = get_cached_plan(fft_len, inverse);

    // Compute normalization scale
    let direction = if inverse {
        crate::norm::FftDirection::Inverse
    } else {
        crate::norm::FftDirection::Forward
    };
    let scale = norm.scale_factor(fft_len, direction);

    // Pre-compute scratch size once
    let scratch_len = plan.get_inplace_scratch_len();

    // Process all lanes in parallel, using thread-local scratch buffers
    let lane_results: Vec<Vec<Complex<f64>>> = lane_starts
        .par_iter()
        .map_init(
            || vec![Complex::new(0.0, 0.0); scratch_len],
            |scratch, &start_offset| {
                // Extract lane from input
                let mut buffer = Vec::with_capacity(fft_len);
                let stride = strides[axis] as usize;
                for i in 0..axis_len.min(fft_len) {
                    buffer.push(data[start_offset + i * stride]);
                }
                // Zero-pad if needed
                buffer.resize(fft_len, Complex::new(0.0, 0.0));

                // Execute FFT reusing thread-local scratch
                plan.process_with_scratch(&mut buffer, scratch);

                // Apply normalization
                if (scale - 1.0).abs() > f64::EPSILON {
                    for c in &mut buffer {
                        *c *= scale;
                    }
                }

                buffer
            },
        )
        .collect();

    // Write results back into a flat output array
    let mut output = vec![Complex::new(0.0, 0.0); new_total];
    let out_stride = new_strides[axis] as usize;

    for (lane_idx, lane_data) in lane_results.iter().enumerate() {
        // Compute the start offset in the output array for this lane
        let out_start = compute_lane_output_start(
            &new_shape,
            &new_strides,
            axis,
            &lane_starts[lane_idx],
            &strides,
            shape,
        );

        for (i, &val) in lane_data.iter().enumerate() {
            output[out_start + i * out_stride] = val;
        }
    }

    Ok((new_shape, output))
}

/// Fast path for 1D FFT: no lane extraction, no parallel overhead.
/// Operates directly on a contiguous buffer in-place.
fn fft_1d_fast(
    data: &[Complex<f64>],
    fft_len: usize,
    input_len: usize,
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<f64>>)> {
    // Build buffer: copy input (truncated or padded)
    let mut buffer = Vec::with_capacity(fft_len);
    let copy_len = input_len.min(fft_len);
    buffer.extend_from_slice(&data[..copy_len]);
    buffer.resize(fft_len, Complex::new(0.0, 0.0));

    // Get cached plan and execute in-place
    let plan = get_cached_plan(fft_len, inverse);
    let mut scratch = vec![Complex::new(0.0, 0.0); plan.get_inplace_scratch_len()];
    plan.process_with_scratch(&mut buffer, &mut scratch);

    // Apply normalization
    let direction = if inverse {
        crate::norm::FftDirection::Inverse
    } else {
        crate::norm::FftDirection::Forward
    };
    let scale = norm.scale_factor(fft_len, direction);
    if (scale - 1.0).abs() > f64::EPSILON {
        for c in &mut buffer {
            *c *= scale;
        }
    }

    Ok((vec![fft_len], buffer))
}

/// Apply 1-D FFTs along multiple axes sequentially.
///
/// `shapes_and_sizes` is a list of `(axis, optional_n)` pairs.
/// Each axis is transformed in order, feeding the output of one
/// as the input to the next.
pub(crate) fn fft_along_axes(
    data: &[Complex<f64>],
    shape: &[usize],
    axes_and_sizes: &[(usize, Option<usize>)],
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<f64>>)> {
    let mut current_data = data.to_vec();
    let mut current_shape = shape.to_vec();

    for &(axis, n) in axes_and_sizes {
        let (new_shape, new_data) =
            fft_along_axis(&current_data, &current_shape, axis, n, inverse, norm)?;
        current_shape = new_shape;
        current_data = new_data;
    }

    Ok((current_shape, current_data))
}

/// Simpler entry point: FFT along a single axis using raw execute_fft_1d.
/// Used by the 1-D fft/ifft functions that don't need multi-axis iteration.
pub(crate) fn fft_1d_along_axis(
    data: &[Complex<f64>],
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<f64>>)> {
    fft_along_axis(data, shape, axis, n, inverse, norm)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute row-major strides for a given shape.
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

/// Compute the flat offset for each lane's starting position.
///
/// A "lane" is a 1-D slice along `axis`. The number of lanes equals
/// `total_elements / shape[axis]`. We need to enumerate all
/// combinations of indices for the non-axis dimensions.
fn compute_lane_starts(
    shape: &[usize],
    strides: &[isize],
    axis: usize,
    num_lanes: usize,
) -> Vec<usize> {
    let ndim = shape.len();
    let mut lane_starts = Vec::with_capacity(num_lanes);

    // Build the "outer shape" — shape with axis dimension removed
    let mut outer_dims: Vec<(usize, isize)> = Vec::with_capacity(ndim - 1);
    for (d, (&s, &st)) in shape.iter().zip(strides.iter()).enumerate() {
        if d != axis {
            outer_dims.push((s, st));
        }
    }

    // Enumerate all multi-indices in the outer dimensions
    let outer_total = outer_dims.iter().map(|&(s, _)| s).product::<usize>();
    debug_assert_eq!(outer_total, num_lanes);

    for lane_idx in 0..num_lanes {
        let mut offset = 0usize;
        let mut remainder = lane_idx;
        // Convert flat lane_idx to multi-index in outer dims (row-major)
        for &(dim_size, stride) in outer_dims.iter().rev() {
            let idx = remainder % dim_size;
            remainder /= dim_size;
            offset += idx * stride as usize;
        }
        lane_starts.push(offset);
    }

    lane_starts
}

/// Compute the output start offset for a lane, given its input start offset.
///
/// When the axis size changes (due to zero-padding/truncation), the strides
/// change, so we need to recompute the flat offset in the output array.
fn compute_lane_output_start(
    new_shape: &[usize],
    new_strides: &[isize],
    axis: usize,
    input_start: &usize,
    input_strides: &[isize],
    input_shape: &[usize],
) -> usize {
    let ndim = new_shape.len();

    // Recover the multi-index from the input start offset
    let mut remaining = *input_start as isize;
    let mut multi_idx = vec![0usize; ndim];
    for d in 0..ndim {
        if d == axis {
            continue;
        }
        if input_strides[d] != 0 {
            multi_idx[d] = (remaining / input_strides[d]) as usize;
            remaining -= (multi_idx[d] as isize) * input_strides[d];
        }
    }

    // Compute output offset from multi-index using new strides
    let mut offset = 0usize;
    for d in 0..ndim {
        if d == axis {
            continue;
        }
        offset += multi_idx[d] * new_strides[d] as usize;
    }

    let _ = input_shape; // used implicitly via strides
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_1d() {
        assert_eq!(compute_strides(&[8]), vec![1]);
    }

    #[test]
    fn strides_2d() {
        assert_eq!(compute_strides(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn strides_3d() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn fft_1d_simple() {
        // FFT of [1, 0, 0, 0] should give [1, 1, 1, 1]
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let (shape, result) =
            fft_along_axis(&data, &[4], 0, None, false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![4]);
        for c in &result {
            assert!((c.re - 1.0).abs() < 1e-12);
            assert!(c.im.abs() < 1e-12);
        }
    }

    #[test]
    fn fft_2d_along_axis0() {
        // 2x2 identity-ish test
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let (shape, result) =
            fft_along_axis(&data, &[2, 2], 0, None, false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![2, 2]);
        // Along axis 0: column 0 = [1,0] -> FFT -> [1, 1]
        //                column 1 = [0,1] -> FFT -> [1, -1]
        assert!((result[0].re - 1.0).abs() < 1e-12); // (0,0)
        assert!((result[1].re - 1.0).abs() < 1e-12); // (0,1)
        assert!((result[2].re - 1.0).abs() < 1e-12); // (1,0)
        assert!((result[3].re - (-1.0)).abs() < 1e-12); // (1,1)
    }

    #[test]
    fn fft_axis_out_of_bounds() {
        let data = vec![Complex::new(1.0, 0.0)];
        assert!(fft_along_axis(&data, &[1], 1, None, false, FftNorm::Backward).is_err());
    }

    #[test]
    fn fft_with_zero_padding() {
        // Input of length 2, padded to 4
        let data = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let (shape, result) =
            fft_along_axis(&data, &[2], 0, Some(4), false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(result.len(), 4);
        // FFT of [1, 1, 0, 0]
        assert!((result[0].re - 2.0).abs() < 1e-12);
    }
}
