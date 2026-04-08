// ferray-fft: Shared axis/stride helpers used across all FFT modules.
//
// These utilities used to be copy-pasted into `complex.rs`, `real.rs`,
// `hermitian.rs`, `shift.rs`, and `nd.rs` — four-way duplication of
// `resolve_axis`/`resolve_axes` plus `compute_strides` in two slightly
// different flavors (`Vec<isize>` vs `Vec<usize>`). They now live in
// one place so the public FFT surfaces share a single source of truth
// for axis validation (see issues #223, #224, #225, #438).

use ferray_core::error::{FerrayError, FerrayResult};

/// Resolve an axis parameter: if `None`, use the last axis.
///
/// Returns an error if `ndim == 0` or the given axis is out of bounds.
#[inline]
pub(crate) fn resolve_axis(ndim: usize, axis: Option<usize>) -> FerrayResult<usize> {
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

/// Resolve axes for a multi-dimensional FFT. If `None`, use all axes.
///
/// Returns an error if any axis is out of bounds.
#[inline]
pub(crate) fn resolve_axes(
    ndim: usize,
    axes: Option<&[usize]>,
) -> FerrayResult<Vec<usize>> {
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
