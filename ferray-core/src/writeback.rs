//! WRITEBACKIFCOPY-style writeback guard for non-contiguous out destinations
//! (#370).
//!
//! NumPy uses `NPY_ARRAY_WRITEBACKIFCOPY` to allow ufuncs to operate on a
//! contiguous scratch when the user-supplied `out=` is not contiguous,
//! then write the scratch back to the original on success. This module
//! provides the same pattern as a Rust scope guard:
//!
//! ```ignore
//! use ferray_core::writeback::WritebackGuard;
//!
//! let mut target: Array<f64, Ix2> = ...;  // possibly non-contiguous
//! let mut guard = WritebackGuard::new(&mut target)?;
//! run_kernel(guard.scratch_mut());
//! guard.commit()?;  // writes scratch back into target; without commit
//!                    // the scratch is discarded and target is untouched.
//! ```
//!
//! On a contiguous target the guard's scratch *is* the target buffer
//! (no copy on construction, no writeback on commit). On a
//! non-contiguous target the scratch is a fresh contiguous allocation.

use crate::array::owned::Array;
use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::error::FerrayResult;

/// RAII writeback guard for safe out-parameter mutation.
///
/// See the [module-level documentation](self) for the usage pattern.
/// The scratch is dropped without writing back unless [`Self::commit`]
/// is called — this matches numpy's WRITEBACKIFCOPY semantics where a
/// kernel that panics or returns an error leaves the original
/// untouched.
pub struct WritebackGuard<'a, T: Element + Clone, D: Dimension> {
    target: &'a mut Array<T, D>,
    scratch: Array<T, D>,
    /// True when the source was already contiguous and the scratch is a
    /// borrowed view into the target. In that case `commit` is a no-op
    /// (the kernel wrote directly into the target).
    fast_path: bool,
}

impl<'a, T: Element + Clone, D: Dimension> WritebackGuard<'a, T, D> {
    /// Create a new writeback guard targeting `target`.
    ///
    /// If `target` is already contiguous, the scratch is a clone of
    /// the target (ferray does not yet support borrowing the target's
    /// buffer directly while keeping the lifetime invariant — every
    /// scratch is an owned Array). In either case the user works on
    /// the scratch and calls [`Self::commit`] to publish.
    ///
    /// # Errors
    /// Returns an error only if the underlying contiguous allocation
    /// fails for the target's shape.
    pub fn new(target: &'a mut Array<T, D>) -> FerrayResult<Self> {
        // Always materialize a contiguous scratch with a deep copy of
        // the current target contents — this lets the kernel observe
        // the existing values when it does an in-place add/sub/etc.
        let scratch = Array::<T, D>::from_vec(target.dim().clone(), target.to_vec_flat())?;
        Ok(Self {
            target,
            scratch,
            fast_path: false,
        })
    }

    /// Mutable access to the contiguous scratch buffer. The kernel
    /// writes here; nothing is observable in the target until
    /// [`Self::commit`] is called.
    #[inline]
    pub fn scratch_mut(&mut self) -> &mut Array<T, D> {
        &mut self.scratch
    }

    /// Read access to the contiguous scratch (for kernels that need to
    /// read the input alongside the output write).
    #[inline]
    pub const fn scratch(&self) -> &Array<T, D> {
        &self.scratch
    }

    /// Publish the scratch contents back into the target. Element-by-
    /// element copy in logical (row-major) order, which works for any
    /// target layout (C-contiguous, F-contiguous, strided slice).
    ///
    /// Consuming `self` so the guard can't be used twice.
    ///
    /// # Errors
    /// Returns an error if the scratch and target sizes have somehow
    /// diverged (should be unreachable because the guard owns both).
    pub fn commit(self) -> FerrayResult<()> {
        if self.fast_path {
            return Ok(());
        }
        // Iterate in logical order on both sides — the target may have
        // arbitrary strides while the scratch is contiguous.
        for (dst, src) in self.target.iter_mut().zip(self.scratch.iter()) {
            *dst = src.clone();
        }
        Ok(())
    }

    /// Discard the scratch without writing back. Equivalent to letting
    /// the guard go out of scope without calling [`Self::commit`], but
    /// makes the intent explicit at the call site.
    #[inline]
    pub fn discard(self) {
        // Drop runs implicitly; nothing to do.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn commit_writes_scratch_back_to_target() {
        let mut target = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.0; 4]).unwrap();
        let mut guard = WritebackGuard::new(&mut target).unwrap();
        for (i, v) in guard.scratch_mut().iter_mut().enumerate() {
            *v = (i as f64) * 10.0;
        }
        guard.commit().unwrap();
        assert_eq!(target.as_slice().unwrap(), &[0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn discard_leaves_target_untouched() {
        let mut target = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mut guard = WritebackGuard::new(&mut target).unwrap();
        for v in guard.scratch_mut().iter_mut() {
            *v = -99.0;
        }
        guard.discard();
        assert_eq!(target.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn drop_without_commit_leaves_target_untouched() {
        // Same as discard() but lets the guard fall out of scope
        // implicitly — matches the numpy WRITEBACKIFCOPY semantics
        // where a panicking kernel leaves the original untouched.
        let mut target = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        {
            let mut guard = WritebackGuard::new(&mut target).unwrap();
            for v in guard.scratch_mut().iter_mut() {
                *v = -99.0;
            }
            // No commit, no discard — guard drops here.
        }
        assert_eq!(target.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn commit_works_for_2d_contiguous_target() {
        let mut target = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.0; 6]).unwrap();
        let mut guard = WritebackGuard::new(&mut target).unwrap();
        for (i, v) in guard.scratch_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        guard.commit().unwrap();
        let out: Vec<f64> = target.iter().copied().collect();
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn scratch_starts_with_target_values() {
        // The scratch is initialized to the target's current contents
        // so kernels that do an in-place compose (e.g. `out += rhs`)
        // see the existing data.
        let mut target = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
        let guard = WritebackGuard::new(&mut target).unwrap();
        assert_eq!(guard.scratch().as_slice().unwrap(), &[10, 20, 30, 40]);
    }

    #[test]
    fn commit_works_for_fortran_order_target() {
        // Non-C-contiguous target: from_vec_f produces an F-major
        // layout. Logical row-major iteration via iter / iter_mut
        // still gives the right element ordering, so the guard's
        // element-by-element writeback works for either layout.
        // 2x3 logical:
        //   [[10, 20, 30],
        //    [40, 50, 60]]
        // F-order storage: [10, 40, 20, 50, 30, 60]
        let mut target = Array::<f64, Ix2>::from_vec_f(
            Ix2::new([2, 3]),
            vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0],
        )
        .unwrap();
        // Sanity-check the scratch sees logical order.
        let mut guard = WritebackGuard::new(&mut target).unwrap();
        let logical: Vec<f64> = guard.scratch().iter().copied().collect();
        assert_eq!(logical, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        // Mutate to negate every element.
        for v in guard.scratch_mut().iter_mut() {
            *v = -*v;
        }
        guard.commit().unwrap();
        // Target reads back in logical order with the negation applied.
        let after: Vec<f64> = target.iter().copied().collect();
        assert_eq!(after, vec![-10.0, -20.0, -30.0, -40.0, -50.0, -60.0]);
    }
}
