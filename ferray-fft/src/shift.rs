// ferray-fft: fftshift, ifftshift (REQ-11)
//
// fftshift moves the zero-frequency component to the center of the spectrum.
// ifftshift undoes this operation.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;

use crate::axes::{compute_strides, resolve_axes};

/// Shift the zero-frequency component to the center of the spectrum.
///
/// Analogous to `numpy.fft.fftshift`. For each specified axis, the
/// array is rolled by `n/2` positions (where `n` is the length along
/// that axis).
///
/// This is useful for visualizing the Fourier transform with the
/// zero-frequency component in the middle.
///
/// # Parameters
/// - `a`: Input array.
/// - `axes`: Axes over which to shift. If `None`, shifts all axes.
///
/// # Errors
/// Returns an error if any axis is out of bounds.
pub fn fftshift<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    let axes = resolve_axes(ndim, axes)?;

    // Compute shift amounts: n // 2 for each axis
    let shifts: Vec<isize> = axes.iter().map(|&ax| (shape[ax] / 2) as isize).collect();

    roll_along_axes(a, &axes, &shifts)
}

/// Inverse of `fftshift`.
///
/// Analogous to `numpy.fft.ifftshift`. For each specified axis, the
/// array is rolled by `-(n+1)/2` positions (undoing `fftshift`).
///
/// Note: `fftshift` and `ifftshift` are the same for even-length axes
/// but differ for odd-length axes.
///
/// # Parameters
/// - `a`: Input array.
/// - `axes`: Axes over which to shift. If `None`, shifts all axes.
///
/// # Errors
/// Returns an error if any axis is out of bounds.
pub fn ifftshift<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    let axes = resolve_axes(ndim, axes)?;

    // Compute shift amounts: -(n//2) for each axis
    let shifts: Vec<isize> = axes.iter().map(|&ax| -((shape[ax] / 2) as isize)).collect();

    roll_along_axes(a, &axes, &shifts)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Roll an array along the specified axes by the given shift amounts.
///
/// This implements circular shifting (like `numpy.roll`) along multiple
/// axes simultaneously.
fn roll_along_axes<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axes: &[usize],
    shifts: &[isize],
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    let total: usize = shape.iter().product();

    if total == 0 {
        let data: Vec<T> = Vec::new();
        return Array::from_vec(IxDyn::new(shape), data);
    }

    // `compute_strides` returns signed strides (matching the lane machinery
    // in nd.rs); the roll loop below only uses them as non-negative
    // row-major strides, so cast once up front.
    let strides: Vec<usize> = compute_strides(shape).iter().map(|&s| s as usize).collect();

    // Build a shift lookup: for each dimension, the shift amount (mod axis_len)
    let mut axis_shifts = vec![0isize; ndim];
    for (&ax, &sh) in axes.iter().zip(shifts.iter()) {
        let n = shape[ax] as isize;
        if n > 0 {
            axis_shifts[ax] = ((sh % n) + n) % n;
        }
    }

    // Allocate output and fill
    let input: Vec<T> = a.iter().cloned().collect();
    let mut output = Vec::with_capacity(total);

    for out_flat in 0..total {
        // Convert flat index to multi-index
        let mut src_flat = 0usize;
        let mut remaining = out_flat;
        for d in 0..ndim {
            let idx = remaining / strides[d];
            remaining %= strides[d];

            // Compute the source index by un-shifting
            let n = shape[d] as isize;
            let src_idx = ((idx as isize - axis_shifts[d]) % n + n) % n;
            src_flat += src_idx as usize * strides[d];
        }
        output.push(input[src_flat].clone());
    }

    Array::from_vec(IxDyn::new(shape), output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;
    use ferray_core::dimension::Ix2;

    #[test]
    fn fftshift_even() {
        // [0, 1, 2, 3, 4, 5, 6, 7] -> [4, 5, 6, 7, 0, 1, 2, 3]
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([8]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        let shifted = fftshift(&a, None).unwrap();
        let data: Vec<f64> = shifted.iter().copied().collect();
        assert_eq!(data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn fftshift_odd() {
        // [0, 1, 2, 3, 4] -> [3, 4, 0, 1, 2]  (shift by 5//2 = 2)
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let shifted = fftshift(&a, None).unwrap();
        let data: Vec<f64> = shifted.iter().copied().collect();
        assert_eq!(data, vec![3.0, 4.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn ifftshift_even() {
        // Inverse of fftshift for even length
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([8]),
            vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let unshifted = ifftshift(&a, None).unwrap();
        let data: Vec<f64> = unshifted.iter().copied().collect();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn ifftshift_odd() {
        // Inverse of fftshift for odd length
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, 4.0, 0.0, 1.0, 2.0]).unwrap();
        let unshifted = ifftshift(&a, None).unwrap();
        let data: Vec<f64> = unshifted.iter().copied().collect();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn fftshift_ifftshift_roundtrip_even() {
        // AC-5: roundtrip
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([8]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        let shifted = fftshift(&a, None).unwrap();
        let recovered = ifftshift(&shifted, None).unwrap();
        let data: Vec<f64> = recovered.iter().copied().collect();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn fftshift_ifftshift_roundtrip_odd() {
        // AC-5: roundtrip for odd-length
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let shifted = fftshift(&a, None).unwrap();
        let recovered = ifftshift(&shifted, None).unwrap();
        let data: Vec<f64> = recovered.iter().copied().collect();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn fftshift_2d() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 4]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        let shifted = fftshift(&a, None).unwrap();
        let data: Vec<f64> = shifted.iter().copied().collect();
        // Shift by [1, 2]: row 1 becomes row 0, cols shift by 2
        assert_eq!(data, vec![6.0, 7.0, 4.0, 5.0, 2.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn fftshift_specific_axis() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 4]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        // Shift only along axis 1
        let shifted = fftshift(&a, Some(&[1])).unwrap();
        let data: Vec<f64> = shifted.iter().copied().collect();
        assert_eq!(data, vec![2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0]);
    }

    #[test]
    fn fftshift_axis_out_of_bounds() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.0; 4]).unwrap();
        assert!(fftshift(&a, Some(&[1])).is_err());
    }
}
