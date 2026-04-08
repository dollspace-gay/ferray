// ferray-fft: Shared axis/stride helpers used across all FFT modules.
//
// These utilities used to be copy-pasted into `complex.rs`, `real.rs`,
// `hermitian.rs`, `shift.rs`, and `nd.rs` — four-way duplication of
// `resolve_axis`/`resolve_axes` plus `compute_strides` in two slightly
// different flavors (`Vec<isize>` vs `Vec<usize>`). They now live in
// one place so the public FFT surfaces share a single source of truth
// for axis validation (see issues #223, #224, #225, #438).

use ferray_core::error::{FerrayError, FerrayResult};

/// Normalize a (possibly negative) axis into the range `[0, ndim)`.
///
/// Mirrors NumPy's behaviour where `axis=-1` selects the last axis,
/// `axis=-2` the one before it, and so on. Returns
/// `FerrayError::AxisOutOfBounds` if the value is out of range
/// (after normalization). See issue #434.
#[inline]
pub(crate) fn normalize_axis(ndim: usize, axis: isize) -> FerrayResult<usize> {
    let n = ndim as isize;
    let normalized = if axis < 0 { axis + n } else { axis };
    if normalized < 0 || normalized >= n {
        Err(FerrayError::axis_out_of_bounds(axis as usize, ndim))
    } else {
        Ok(normalized as usize)
    }
}

/// Resolve an axis parameter: if `None`, use the last axis. Negative
/// values are accepted and normalized via [`normalize_axis`].
///
/// Returns an error if `ndim == 0` or the given axis is out of bounds.
#[inline]
pub(crate) fn resolve_axis(ndim: usize, axis: Option<isize>) -> FerrayResult<usize> {
    match axis {
        Some(ax) => normalize_axis(ndim, ax),
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

/// Resolve axes for a multi-dimensional FFT. If `None`, use all axes.
/// Each axis may be negative; values are normalized via
/// [`normalize_axis`] (#434).
///
/// Returns an error if any axis is out of bounds.
#[inline]
pub(crate) fn resolve_axes(
    ndim: usize,
    axes: Option<&[isize]>,
) -> FerrayResult<Vec<usize>> {
    match axes {
        Some(ax) => ax.iter().map(|&a| normalize_axis(ndim, a)).collect(),
        None => Ok((0..ndim).collect()),
    }
}

/// Compute row-major (C-order) strides for a given shape, returned as
/// `isize`. Callers that need `usize` strides (e.g. the roll helper in
/// `shift.rs`) can cast element-wise at the call site; returning `isize`
/// here matches the form used by the lane machinery in `nd.rs`, which
/// multiplies with other signed offsets.
#[inline]
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<isize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_axis_defaults_to_last() {
        assert_eq!(resolve_axis(3, None).unwrap(), 2);
        assert_eq!(resolve_axis(1, None).unwrap(), 0);
    }

    #[test]
    fn resolve_axis_zero_dim_errors() {
        assert!(resolve_axis(0, None).is_err());
    }

    #[test]
    fn resolve_axis_out_of_bounds() {
        assert!(resolve_axis(2, Some(3)).is_err());
        assert!(resolve_axis(2, Some(-3)).is_err());
    }

    #[test]
    fn resolve_axis_negative() {
        // axis = -1 is the last axis
        assert_eq!(resolve_axis(3, Some(-1)).unwrap(), 2);
        // axis = -2 is the second-to-last
        assert_eq!(resolve_axis(3, Some(-2)).unwrap(), 1);
        // axis = -3 wraps to the first
        assert_eq!(resolve_axis(3, Some(-3)).unwrap(), 0);
    }

    #[test]
    fn resolve_axes_defaults_to_all() {
        assert_eq!(resolve_axes(3, None).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn resolve_axes_validates_each() {
        assert!(resolve_axes(3, Some(&[0, 5])).is_err());
        assert_eq!(resolve_axes(3, Some(&[0, 2])).unwrap(), vec![0, 2]);
    }

    #[test]
    fn resolve_axes_negative_mix() {
        // mix of positive and negative axes
        assert_eq!(resolve_axes(4, Some(&[-2, -1])).unwrap(), vec![2, 3]);
        assert_eq!(resolve_axes(4, Some(&[0, -1])).unwrap(), vec![0, 3]);
    }

    #[test]
    fn normalize_axis_round_trip() {
        for ndim in 1..=4usize {
            for ax in 0..ndim {
                assert_eq!(normalize_axis(ndim, ax as isize).unwrap(), ax);
                let neg = (ax as isize) - (ndim as isize);
                assert_eq!(normalize_axis(ndim, neg).unwrap(), ax);
            }
        }
    }

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
    fn strides_empty() {
        assert_eq!(compute_strides(&[]), Vec::<isize>::new());
    }
}
