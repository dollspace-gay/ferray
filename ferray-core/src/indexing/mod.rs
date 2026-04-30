// ferray-core: Indexing module (REQ-12 through REQ-15a)
//
// Provides basic indexing (integer + slice → views), advanced indexing
// (fancy + boolean → copies), and extended indexing functions (take, put,
// choose, compress, etc.).

pub mod advanced;
pub mod basic;
pub mod extended;

// Flat re-exports of the most NumPy-namespace-adjacent extended
// indexing functions so users don't need to reach into the `extended`
// submodule for the common cases. Matches NumPy's flat top-level
// namespace (`np.ravel_multi_index`, `np.unravel_index`, `np.nonzero`,
// `np.argwhere`, `np.flatnonzero`).
pub use extended::{
    MaskKind, argwhere, extract, flatnonzero, mask_indices, nonzero, place, putmask,
    ravel_multi_index, unravel_index,
};

use crate::error::{FerrayError, FerrayResult};

/// Normalize a (possibly negative) index into a non-negative `usize`
/// suitable for array access along `axis`. Negative indices count from the
/// end (`-1` is the last element, etc.). Returns `IndexOutOfBounds` when
/// the normalized index falls outside `[0, size)`.
///
/// Shared by all three indexing submodules; see issue #121 for the
/// original triplication.
#[inline]
pub(crate) const fn normalize_index(index: isize, size: usize, axis: usize) -> FerrayResult<usize> {
    if index < 0 {
        let pos = size as isize + index;
        if pos < 0 {
            return Err(FerrayError::index_out_of_bounds(index, axis, size));
        }
        Ok(pos as usize)
    } else {
        let idx = index as usize;
        if idx >= size {
            return Err(FerrayError::index_out_of_bounds(index, axis, size));
        }
        Ok(idx)
    }
}
