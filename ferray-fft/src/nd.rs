// ferray-fft: Multi-dimensional FFT via iterated 1D transforms along axes (REQ-3, REQ-4)
//
// The core strategy: for each axis in the list, extract every 1-D "lane"
// along that axis, run the 1-D FFT on each lane, and write the results
// back. For N-D arrays this is done sequentially per axis but lanes
// within an axis are processed in parallel via Rayon.

use num_complex::Complex;
use num_traits::{One, Zero};
use rayon::prelude::*;

use ferray_core::error::{FerrayError, FerrayResult};

use crate::axes::compute_strides;
use crate::float::FftFloat;
use crate::norm::FftNorm;

/// Apply a 1-D FFT along a single axis of a multi-dimensional array
/// stored in row-major (C) order.
///
/// `shape` and `data` describe the array. The transform is applied
/// along `axis`, optionally with zero-padding/truncation to `n` points.
/// The returned `(new_shape, new_data)` reflect the (possibly changed)
/// size along the transformed axis. Generic over the scalar precision
/// via [`FftFloat`].
pub fn fft_along_axis<T: FftFloat>(
    data: &[Complex<T>],
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<T>>)>
where
    Complex<T>: ferray_core::Element,
{
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
        return Ok((new_shape, vec![Complex::zero(); new_total]));
    }

    // Fast path: 1D array — skip all lane machinery
    if ndim == 1 {
        return Ok(fft_1d_fast(data, fft_len, axis_len, inverse, norm));
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

    // Get the cached plan once for this size (dispatched by precision).
    let plan = T::cached_plan(fft_len, inverse);

    // Compute normalization scale in the target precision.
    let direction = if inverse {
        crate::norm::FftDirection::Inverse
    } else {
        crate::norm::FftDirection::Forward
    };
    let scale = T::scale_factor(norm, fft_len, direction);
    let one = <T as One>::one();

    // Pre-compute scratch size once
    let scratch_len = plan.get_inplace_scratch_len();

    // Process all lanes in parallel.
    //
    // Old approach allocated a fresh `Vec<Complex<f64>>` per lane and
    // collected them into a `Vec<Vec<Complex<f64>>>` — `num_lanes` separate
    // heap allocations plus `num_lanes` Vec headers. For an (M, N) array
    // with FFT along axis 1, that's M allocations.
    //
    // New approach: a single contiguous buffer of `num_lanes * fft_len`
    // complex values (one allocation total), partitioned via
    // `par_chunks_mut(fft_len)` so each rayon worker writes directly into
    // its assigned chunk. The lane buffer for the FFT is the chunk itself,
    // so there's zero per-lane allocation. Scratch for the FFT is reused
    // per-thread via `for_each_init`.
    let stride = strides[axis] as usize;
    let copy_len = axis_len.min(fft_len);
    let mut lane_outputs: Vec<Complex<T>> = vec![Complex::zero(); num_lanes * fft_len];
    lane_outputs
        .par_chunks_mut(fft_len)
        .zip(lane_starts.par_iter())
        .for_each_init(
            || vec![Complex::zero(); scratch_len],
            |scratch, (out_chunk, &start_offset)| {
                // Copy this lane's strided slice into the contiguous output
                // chunk. The chunk is already zero-initialized above, so
                // positions beyond `copy_len` act as zero-padding when
                // `fft_len > axis_len`.
                for (i, slot) in out_chunk.iter_mut().take(copy_len).enumerate() {
                    *slot = data[start_offset + i * stride];
                }
                // The remaining positions (copy_len..fft_len) are already
                // zero from the initial allocation.

                // Execute FFT in-place on the chunk, reusing thread-local scratch.
                plan.process_with_scratch(out_chunk, scratch);

                // Apply normalization in place.
                if scale != one {
                    for c in out_chunk.iter_mut() {
                        *c = *c * scale;
                    }
                }
            },
        );

    // Scatter step: copy each lane's contiguous chunk into the final output
    // buffer at its strided position. This is sequential but is essentially
    // a series of indexed writes — no allocation, no compute.
    let mut output: Vec<Complex<T>> = vec![Complex::zero(); new_total];
    let out_stride = new_strides[axis] as usize;

    for (lane_idx, lane_chunk) in lane_outputs.chunks(fft_len).enumerate() {
        let out_start = compute_lane_output_start(
            &new_shape,
            &new_strides,
            axis,
            lane_starts[lane_idx],
            &strides,
        );

        for (i, &val) in lane_chunk.iter().enumerate() {
            output[out_start + i * out_stride] = val;
        }
    }

    Ok((new_shape, output))
}

/// Fast path for 1D FFT: no lane extraction, no parallel overhead.
/// Operates directly on a contiguous buffer in-place.
fn fft_1d_fast<T: FftFloat>(
    data: &[Complex<T>],
    fft_len: usize,
    input_len: usize,
    inverse: bool,
    norm: FftNorm,
) -> (Vec<usize>, Vec<Complex<T>>)
where
    Complex<T>: ferray_core::Element,
{
    // Build buffer: copy input (truncated or padded)
    let mut buffer: Vec<Complex<T>> = Vec::with_capacity(fft_len);
    let copy_len = input_len.min(fft_len);
    buffer.extend_from_slice(&data[..copy_len]);
    buffer.resize(fft_len, Complex::zero());

    // Get cached plan and execute in-place
    let plan = T::cached_plan(fft_len, inverse);
    let mut scratch: Vec<Complex<T>> = vec![Complex::zero(); plan.get_inplace_scratch_len()];
    plan.process_with_scratch(&mut buffer, &mut scratch);

    // Apply normalization
    let direction = if inverse {
        crate::norm::FftDirection::Inverse
    } else {
        crate::norm::FftDirection::Forward
    };
    let scale = T::scale_factor(norm, fft_len, direction);
    let one = <T as One>::one();
    if scale != one {
        for c in &mut buffer {
            *c = *c * scale;
        }
    }

    (vec![fft_len], buffer)
}

/// Apply 1-D FFTs along multiple axes sequentially.
///
/// `shapes_and_sizes` is a list of `(axis, optional_n)` pairs.
/// Each axis is transformed in order, feeding the output of one
/// as the input to the next.
pub fn fft_along_axes<T: FftFloat>(
    data: &[Complex<T>],
    shape: &[usize],
    axes_and_sizes: &[(usize, Option<usize>)],
    inverse: bool,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<T>>)>
where
    Complex<T>: ferray_core::Element,
{
    // No axes specified — fall back to a copy of the input.
    if axes_and_sizes.is_empty() {
        return Ok((shape.to_vec(), data.to_vec()));
    }

    // The first pass reads from the caller-owned `data` slice (no
    // upfront copy needed — `fft_along_axis` allocates its own output).
    // Subsequent passes feed the owned `current_data` from one pass to
    // the next so we never re-copy the full array between axes (#439).
    let (first_axis, first_n) = axes_and_sizes[0];
    let (mut current_shape, mut current_data) =
        fft_along_axis(data, shape, first_axis, first_n, inverse, norm)?;

    for &(axis, n) in &axes_and_sizes[1..] {
        let (new_shape, new_data) =
            fft_along_axis(&current_data, &current_shape, axis, n, inverse, norm)?;
        current_shape = new_shape;
        current_data = new_data;
    }

    Ok((current_shape, current_data))
}

// ---------------------------------------------------------------------------
// Real-input FFT (issue #432)
//
// Dedicated real-to-complex transform via the realfft crate. The output
// has length `fft_len/2 + 1` along the transformed axis (Hermitian
// symmetry — the second half is the conjugate of the first and would be
// redundant). This avoids the 2× overhead of promoting to complex first
// and running a full complex FFT.
// ---------------------------------------------------------------------------

/// Apply a 1-D real-to-complex FFT along a single axis. The output axis
/// length is `fft_len / 2 + 1` (Hermitian-folded representation).
///
/// `data` is a flat real array in row-major order, `shape` describes its
/// nominal multi-dimensional layout, and `axis` is the transform axis.
/// Optional `n` truncates or zero-pads the input axis to `n` real values
/// before the transform; if `None`, the existing axis length is used.
///
/// Mirrors [`fft_along_axis`] but uses the cached real plan and the
/// `par_chunks_mut` allocation pattern from #433. Generic over the scalar
/// precision via [`FftFloat`].
pub fn rfft_along_axis<T: FftFloat>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<Complex<T>>)>
where
    Complex<T>: ferray_core::Element,
{
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }

    let axis_len = shape[axis];
    let fft_len = n.unwrap_or(axis_len);
    if fft_len == 0 {
        return Err(FerrayError::invalid_value("FFT length must be > 0"));
    }

    let half_len = fft_len / 2 + 1;

    // Output shape: same as input but with `half_len` along the transform axis.
    let mut new_shape = shape.to_vec();
    new_shape[axis] = half_len;
    let new_total: usize = new_shape.iter().product();

    let total = shape.iter().product::<usize>();
    if total == 0 {
        return Ok((new_shape, vec![Complex::zero(); new_total]));
    }

    let plan = T::cached_real_forward(fft_len);
    let scratch_len = plan.get_scratch_len();

    // Compute normalization scale (forward direction).
    let scale = T::scale_factor(norm, fft_len, crate::norm::FftDirection::Forward);
    let one = <T as One>::one();
    let t_zero = <T as Zero>::zero();

    let strides = compute_strides(shape);
    let new_strides = compute_strides(&new_shape);
    let stride = strides[axis] as usize;
    let copy_len = axis_len.min(fft_len);

    // 1-D fast path: skip lane machinery entirely.
    if ndim == 1 {
        let mut input_buf: Vec<T> = vec![t_zero; fft_len];
        input_buf[..copy_len].copy_from_slice(&data[..copy_len]);
        // input_buf[copy_len..] already zero from the initial vec! allocation.
        let mut output_buf: Vec<Complex<T>> = vec![Complex::zero(); half_len];
        let mut scratch = plan.make_scratch_vec();
        plan.process_with_scratch(&mut input_buf, &mut output_buf, &mut scratch)
            .map_err(|e| FerrayError::invalid_value(format!("real FFT process failed: {e}")))?;
        if scale != one {
            for c in &mut output_buf {
                *c = *c * scale;
            }
        }
        return Ok((new_shape, output_buf));
    }

    let num_lanes = total / axis_len;
    let lane_starts = compute_lane_starts(shape, &strides, axis, num_lanes);

    // Pre-allocate one contiguous buffer for all lane outputs (#433 pattern).
    // Each chunk is `half_len` complex values; total is num_lanes * half_len.
    let mut lane_outputs: Vec<Complex<T>> = vec![Complex::zero(); num_lanes * half_len];

    lane_outputs
        .par_chunks_mut(half_len)
        .zip(lane_starts.par_iter())
        .for_each_init(
            || (vec![t_zero; fft_len], vec![Complex::zero(); scratch_len]),
            |(input_buf, scratch), (out_chunk, &start_offset)| {
                // Copy this lane's strided real values into the contiguous
                // input buffer; pad the rest with zeros.
                for (i, slot) in input_buf.iter_mut().take(copy_len).enumerate() {
                    *slot = data[start_offset + i * stride];
                }
                for slot in input_buf.iter_mut().skip(copy_len) {
                    *slot = t_zero;
                }
                // Run the real-to-complex transform directly into the
                // chunk's slice of the contiguous lane_outputs buffer.
                plan.process_with_scratch(input_buf, out_chunk, scratch)
                    .expect("real FFT process failed");
                if scale != one {
                    for c in out_chunk.iter_mut() {
                        *c = *c * scale;
                    }
                }
            },
        );

    // Scatter the contiguous chunks into the strided positions in the final
    // output buffer.
    let mut output: Vec<Complex<T>> = vec![Complex::zero(); new_total];
    let out_stride = new_strides[axis] as usize;

    for (lane_idx, lane_chunk) in lane_outputs.chunks(half_len).enumerate() {
        let out_start = compute_lane_output_start(
            &new_shape,
            &new_strides,
            axis,
            lane_starts[lane_idx],
            &strides,
        );
        for (i, &val) in lane_chunk.iter().enumerate() {
            output[out_start + i * out_stride] = val;
        }
    }

    Ok((new_shape, output))
}

/// Apply a 1-D complex-to-real FFT along a single axis. The input has
/// `half_len = output_len/2 + 1` complex values along the transform axis;
/// the output has `output_len` real values.
///
/// `data` is a flat complex array, `shape` is its multi-dimensional
/// layout (with `shape[axis] == half_len`), and `output_len` is the
/// desired length of the real output along that axis. Generic over the
/// scalar precision via [`FftFloat`].
pub fn irfft_along_axis<T: FftFloat>(
    data: &[Complex<T>],
    shape: &[usize],
    axis: usize,
    output_len: usize,
    norm: FftNorm,
) -> FerrayResult<(Vec<usize>, Vec<T>)>
where
    Complex<T>: ferray_core::Element,
{
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    if output_len == 0 {
        return Err(FerrayError::invalid_value(
            "irfft output length must be > 0",
        ));
    }

    let half_len = output_len / 2 + 1;
    let input_axis_len = shape[axis];

    // realfft requires exactly `output_len/2 + 1` complex bins. If the user
    // supplied a different number, we truncate or pad with zeros to fit.
    // (This matches numpy.fft.irfft semantics where extra bins beyond
    // n/2+1 are ignored.)

    let mut new_shape = shape.to_vec();
    new_shape[axis] = output_len;
    let new_total: usize = new_shape.iter().product();

    let total = shape.iter().product::<usize>();
    let t_zero = <T as Zero>::zero();
    if total == 0 {
        return Ok((new_shape, vec![t_zero; new_total]));
    }

    let plan = T::cached_real_inverse(output_len);
    let scratch_len = plan.get_scratch_len();

    let scale = T::scale_factor(norm, output_len, crate::norm::FftDirection::Inverse);
    let one = <T as One>::one();

    let strides = compute_strides(shape);
    let new_strides = compute_strides(&new_shape);
    let stride = strides[axis] as usize;
    let copy_len = input_axis_len.min(half_len);

    // 1-D fast path.
    if ndim == 1 {
        let mut input_buf: Vec<Complex<T>> = vec![Complex::zero(); half_len];
        input_buf[..copy_len].copy_from_slice(&data[..copy_len]);
        // input_buf[copy_len..] already zero — pads with zero bins.
        let mut output_buf: Vec<T> = vec![t_zero; output_len];
        let mut scratch = plan.make_scratch_vec();
        plan.process_with_scratch(&mut input_buf, &mut output_buf, &mut scratch)
            .map_err(|e| {
                FerrayError::invalid_value(format!("inverse real FFT process failed: {e}"))
            })?;
        if scale != one {
            for v in &mut output_buf {
                *v = *v * scale;
            }
        }
        return Ok((new_shape, output_buf));
    }

    let num_lanes = total / input_axis_len;
    let lane_starts = compute_lane_starts(shape, &strides, axis, num_lanes);

    // Per-lane output is `output_len` real values, packed contiguously.
    let mut lane_outputs: Vec<T> = vec![t_zero; num_lanes * output_len];

    lane_outputs
        .par_chunks_mut(output_len)
        .zip(lane_starts.par_iter())
        .for_each_init(
            || {
                (
                    vec![Complex::zero(); half_len],
                    vec![Complex::zero(); scratch_len],
                )
            },
            |(input_buf, scratch), (out_chunk, &start_offset)| {
                // Copy this lane's strided complex bins into the input buffer.
                for (i, slot) in input_buf.iter_mut().take(copy_len).enumerate() {
                    *slot = data[start_offset + i * stride];
                }
                for slot in input_buf.iter_mut().skip(copy_len) {
                    *slot = Complex::zero();
                }
                plan.process_with_scratch(input_buf, out_chunk, scratch)
                    .expect("inverse real FFT process failed");
                if scale != one {
                    for v in out_chunk.iter_mut() {
                        *v = *v * scale;
                    }
                }
            },
        );

    // Scatter into strided output positions.
    let mut output: Vec<T> = vec![t_zero; new_total];
    let out_stride = new_strides[axis] as usize;

    for (lane_idx, lane_chunk) in lane_outputs.chunks(output_len).enumerate() {
        let out_start = compute_lane_output_start(
            &new_shape,
            &new_strides,
            axis,
            lane_starts[lane_idx],
            &strides,
        );
        for (i, &val) in lane_chunk.iter().enumerate() {
            output[out_start + i * out_stride] = val;
        }
    }

    Ok((new_shape, output))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
/// The input shape is not needed because it's already encoded in
/// `input_strides`.
fn compute_lane_output_start(
    new_shape: &[usize],
    new_strides: &[isize],
    axis: usize,
    input_start: usize,
    input_strides: &[isize],
) -> usize {
    let ndim = new_shape.len();

    // Recover the multi-index from the input start offset
    let mut remaining = input_start as isize;
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
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_1d_simple() {
        // FFT of [1, 0, 0, 0] should give [1, 1, 1, 1]
        let data = vec![
            Complex::<f64>::new(1.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
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
            Complex::<f64>::new(1.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
            Complex::<f64>::new(1.0, 0.0),
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
        let data = vec![Complex::<f64>::new(1.0, 0.0)];
        assert!(fft_along_axis(&data, &[1], 1, None, false, FftNorm::Backward).is_err());
    }

    #[test]
    fn fft_with_zero_padding() {
        // Input of length 2, padded to 4
        let data = vec![Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(1.0, 0.0)];
        let (shape, result) =
            fft_along_axis(&data, &[2], 0, Some(4), false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(result.len(), 4);
        // FFT of [1, 1, 0, 0]
        assert!((result[0].re - 2.0).abs() < 1e-12);
    }

    /// Exercises the multi-lane parallel path with enough lanes to actually
    /// hit the `par_chunks_mut` codegen and verify the lane→output scatter
    /// step lines up. Regression guard for the #433 refactor.
    #[test]
    fn fft_2d_many_lanes_along_axis1() {
        // 4×8 array. FFT along axis 1 → 4 lanes of length 8 each.
        // Use a pattern where each row has a single 1.0 at column `row % 8`,
        // so the FFT of each row is e^{-2πi k r / 8} for k = 0..8.
        let rows = 4usize;
        let cols = 8usize;
        let mut data = vec![Complex::<f64>::new(0.0, 0.0); rows * cols];
        for r in 0..rows {
            data[r * cols + r] = Complex::<f64>::new(1.0, 0.0);
        }
        let (shape, result) =
            fft_along_axis(&data, &[rows, cols], 1, None, false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![rows, cols]);
        // For row 0 (delta at col 0), result is all 1.0+0.0j across the lane.
        for c in result.iter().take(cols) {
            assert!((c.re - 1.0).abs() < 1e-12);
            assert!(c.im.abs() < 1e-12);
        }
        // For row r, the first bin (k=0) is always 1.0 (sum of input).
        for r in 0..rows {
            assert!((result[r * cols].re - 1.0).abs() < 1e-12);
        }
    }

    /// Same array, FFT along axis 0 (the strided lanes case). Verifies the
    /// scatter back to non-contiguous output positions works after
    /// switching to `par_chunks_mut` on a separate buffer.
    #[test]
    fn fft_2d_many_lanes_along_axis0() {
        // 8×4 array, FFT along axis 0 → 4 lanes of length 8, each strided
        // by `cols` in the input layout.
        let rows = 8usize;
        let cols = 4usize;
        let mut data = vec![Complex::<f64>::new(0.0, 0.0); rows * cols];
        // Put a 1.0 at row 0 of every column.
        for slot in data.iter_mut().take(cols) {
            *slot = Complex::<f64>::new(1.0, 0.0);
        }
        let (shape, result) =
            fft_along_axis(&data, &[rows, cols], 0, None, false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![rows, cols]);
        // Each column had [1, 0, 0, 0, 0, 0, 0, 0] → FFT gives all 1.0+0.0j.
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                assert!(
                    (result[idx].re - 1.0).abs() < 1e-12,
                    "result[{r},{c}].re = {}",
                    result[idx].re
                );
                assert!(result[idx].im.abs() < 1e-12);
            }
        }
    }

    /// Exercise zero-padding through the multi-lane path.
    #[test]
    fn fft_2d_zero_padding_multi_lane() {
        // 3×2 input, FFT along axis 1 with n=4 → 3 lanes of length 4 each
        // (each input lane padded from 2 to 4 elements).
        let data = vec![
            Complex::<f64>::new(1.0, 0.0),
            Complex::<f64>::new(1.0, 0.0),
            Complex::<f64>::new(2.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
            Complex::<f64>::new(0.0, 0.0),
            Complex::<f64>::new(3.0, 0.0),
        ];
        let (shape, result) =
            fft_along_axis(&data, &[3, 2], 1, Some(4), false, FftNorm::Backward).unwrap();
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(result.len(), 12);
        // Row 0: [1, 1, 0, 0] → DC = 2, others computed by FFT.
        assert!((result[0].re - 2.0).abs() < 1e-12);
        // Row 1: [2, 0, 0, 0] → all bins = 2.
        assert!((result[4].re - 2.0).abs() < 1e-12);
        assert!((result[5].re - 2.0).abs() < 1e-12);
        assert!((result[6].re - 2.0).abs() < 1e-12);
        assert!((result[7].re - 2.0).abs() < 1e-12);
        // Row 2: [0, 3, 0, 0] → DC = 3.
        assert!((result[8].re - 3.0).abs() < 1e-12);
    }
}
