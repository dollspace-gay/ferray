// ferray-core: NdIter — general-purpose multi-array broadcast iterator
//
// Implements the core functionality of NumPy's nditer:
// - Multi-array iteration with automatic broadcasting
// - Contiguous inner-loop access for SIMD-friendly processing
// - Binary and unary iteration patterns
//
// This is the foundational building block for ufunc-style operations.

use crate::array::owned::Array;
use crate::array::view::ArrayView;
use crate::dimension::broadcast::broadcast_shapes;
use crate::dimension::{Dimension, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

// ---------------------------------------------------------------------------
// NdIter for two arrays (binary broadcast iteration)
// ---------------------------------------------------------------------------

/// A broadcast iterator over two arrays, yielding element pairs.
///
/// The two input arrays are broadcast to a common shape. Iteration
/// proceeds in row-major (C) order over the broadcast shape.
///
/// # Example
/// ```ignore
/// let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
/// let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
/// let result = NdIter::binary_map(&a, &b, |x, y| x + y).unwrap();
/// assert_eq!(result.shape(), &[3, 4]);
/// ```
pub struct NdIter;

impl NdIter {
    /// Apply a binary function elementwise over two broadcast arrays,
    /// returning a new array with the broadcast shape.
    ///
    /// This is the workhorse for ufunc-style binary operations. It:
    /// 1. Computes the broadcast shape
    /// 2. Broadcasts both arrays (zero-copy via stride tricks)
    /// 3. Iterates in row-major order, applying `f` to each pair
    /// 4. Returns the result as `Array<U, IxDyn>`
    ///
    /// For contiguous inner loops, the implementation uses slice access
    /// when both broadcast views are contiguous.
    ///
    /// # Errors
    /// Returns `FerrayError::BroadcastFailure` if shapes are incompatible.
    pub fn binary_map<T, U, D1, D2, F>(
        a: &Array<T, D1>,
        b: &Array<T, D2>,
        f: F,
    ) -> FerrayResult<Array<U, IxDyn>>
    where
        T: Element + Copy,
        U: Element,
        D1: Dimension,
        D2: Dimension,
        F: Fn(T, T) -> U,
    {
        let shape = broadcast_shapes(a.shape(), b.shape())?;
        let a_view = a.broadcast_to(&shape)?;
        let b_view = b.broadcast_to(&shape)?;

        let total: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total);

        // Fast path: both views are contiguous (same-shape, no broadcasting)
        if let (Some(a_slice), Some(b_slice)) = (a_view.as_slice(), b_view.as_slice()) {
            for (&ai, &bi) in a_slice.iter().zip(b_slice.iter()) {
                data.push(f(ai, bi));
            }
        } else {
            // General path: iterate via broadcast views
            for (&ai, &bi) in a_view.iter().zip(b_view.iter()) {
                data.push(f(ai, bi));
            }
        }

        Array::from_vec(IxDyn::from(&shape[..]), data)
    }

    /// Apply a binary function where inputs may have different element types.
    ///
    /// Unlike [`binary_map`] which requires both arrays to have the same `T`,
    /// this accepts `Array<A, D1>` and `Array<B, D2>` with different element
    /// types, mapping `(A, B) -> U`.
    ///
    /// This is essential for mixed-type operations like comparing an integer
    /// array against a float threshold, or operations after type promotion.
    pub fn binary_map_mixed<A, B, U, D1, D2, F>(
        a: &Array<A, D1>,
        b: &Array<B, D2>,
        f: F,
    ) -> FerrayResult<Array<U, IxDyn>>
    where
        A: Element + Copy,
        B: Element + Copy,
        U: Element,
        D1: Dimension,
        D2: Dimension,
        F: Fn(A, B) -> U,
    {
        let shape = broadcast_shapes(a.shape(), b.shape())?;
        let a_view = a.broadcast_to(&shape)?;
        let b_view = b.broadcast_to(&shape)?;

        let total: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total);

        for (&ai, &bi) in a_view.iter().zip(b_view.iter()) {
            data.push(f(ai, bi));
        }

        Array::from_vec(IxDyn::from(&shape[..]), data)
    }

    /// Apply a binary function elementwise, writing results into a
    /// pre-allocated output array. Avoids allocation for repeated operations.
    ///
    /// The output array must have the broadcast shape.
    ///
    /// # Errors
    /// Returns errors on shape mismatch or broadcast failure.
    pub fn binary_map_into<T, D1, D2>(
        a: &Array<T, D1>,
        b: &Array<T, D2>,
        out: &mut Array<T, IxDyn>,
        f: impl Fn(T, T) -> T,
    ) -> FerrayResult<()>
    where
        T: Element + Copy,
        D1: Dimension,
        D2: Dimension,
    {
        let shape = broadcast_shapes(a.shape(), b.shape())?;
        if out.shape() != &shape[..] {
            return Err(FerrayError::shape_mismatch(format!(
                "output shape {:?} does not match broadcast shape {:?}",
                out.shape(),
                shape
            )));
        }

        let a_view = a.broadcast_to(&shape)?;
        let b_view = b.broadcast_to(&shape)?;

        if let Some(out_slice) = out.as_slice_mut() {
            for ((o, &ai), &bi) in out_slice
                .iter_mut()
                .zip(a_view.iter())
                .zip(b_view.iter())
            {
                *o = f(ai, bi);
            }
        } else {
            // Non-contiguous output — use indexed iteration
            for ((&ai, &bi), o) in a_view.iter().zip(b_view.iter()).zip(out.iter_mut()) {
                *o = f(ai, bi);
            }
        }

        Ok(())
    }

    /// Apply a unary function elementwise, returning a new array.
    ///
    /// This is simpler than the binary case (no broadcasting needed) but
    /// provides the same contiguous-slice optimization.
    pub fn unary_map<T, U, D, F>(
        a: &Array<T, D>,
        f: F,
    ) -> FerrayResult<Array<U, IxDyn>>
    where
        T: Element + Copy,
        U: Element,
        D: Dimension,
        F: Fn(T) -> U,
    {
        let shape = a.shape().to_vec();
        let total: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total);

        if let Some(slice) = a.as_slice() {
            for &x in slice {
                data.push(f(x));
            }
        } else {
            for &x in a.iter() {
                data.push(f(x));
            }
        }

        Array::from_vec(IxDyn::from(&shape[..]), data)
    }

    /// Apply a unary function elementwise into a pre-allocated output.
    pub fn unary_map_into<T, D>(
        a: &Array<T, D>,
        out: &mut Array<T, IxDyn>,
        f: impl Fn(T) -> T,
    ) -> FerrayResult<()>
    where
        T: Element + Copy,
        D: Dimension,
    {
        if a.shape() != out.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "input shape {:?} does not match output shape {:?}",
                a.shape(),
                out.shape()
            )));
        }

        if let (Some(in_slice), Some(out_slice)) = (a.as_slice(), out.as_slice_mut()) {
            for (o, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                *o = f(x);
            }
        } else {
            for (o, &x) in out.iter_mut().zip(a.iter()) {
                *o = f(x);
            }
        }

        Ok(())
    }

    /// Compute the broadcast shape of two arrays without iterating.
    ///
    /// Useful for pre-allocating output arrays.
    pub fn broadcast_shape(
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> FerrayResult<Vec<usize>> {
        broadcast_shapes(a_shape, b_shape)
    }

    /// Iterate over two broadcast arrays, yielding `(&T, &T)` pairs.
    ///
    /// This returns an iterator that can be consumed lazily. The arrays
    /// are broadcast to a common shape and iterated in row-major order.
    pub fn binary_iter<'a, T, D1, D2>(
        a: &'a Array<T, D1>,
        b: &'a Array<T, D2>,
    ) -> FerrayResult<BinaryBroadcastIter<'a, T>>
    where
        T: Element + Copy,
        D1: Dimension,
        D2: Dimension,
    {
        let shape = broadcast_shapes(a.shape(), b.shape())?;
        let a_view = a.broadcast_to(&shape)?;
        let b_view = b.broadcast_to(&shape)?;

        // Collect into vecs to own the data for the iterator lifetime
        // (broadcast views borrow from the originals, which live for 'a)
        // Materialize broadcast data for owned iteration via next()
        let a_data: Vec<T> = a_view.iter().copied().collect();
        let b_data: Vec<T> = b_view.iter().copied().collect();

        Ok(BinaryBroadcastIter {
            a_view,
            b_view,
            a_data,
            b_data,
            index: 0,
        })
    }
}

/// A broadcast iterator over two arrays that implements `Iterator`.
///
/// Created by [`NdIter::binary_iter`]. Yields `(T, T)` value pairs
/// in row-major order over the broadcast shape. Both broadcast views
/// are materialized into owned `Vec<T>` at construction time so that
/// `next()` can yield owned values without lifetime issues from
/// stride-0 broadcast dimensions.
///
/// For zero-copy functional iteration without materialization, use
/// [`NdIter::binary_map`] or the `map_collect`/`for_each` methods.
pub struct BinaryBroadcastIter<'a, T: Element> {
    /// Broadcast view of the first array (kept for shape/metadata access).
    a_view: ArrayView<'a, T, IxDyn>,
    /// Broadcast view of the second array.
    b_view: ArrayView<'a, T, IxDyn>,
    /// Materialized data from a_view for owned iteration.
    a_data: Vec<T>,
    /// Materialized data from b_view for owned iteration.
    b_data: Vec<T>,
    /// Current position in the iteration.
    index: usize,
}

impl<T: Element + Copy> Iterator for BinaryBroadcastIter<'_, T> {
    type Item = (T, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.a_data.len() {
            return None;
        }
        let i = self.index;
        self.index += 1;
        Some((self.a_data[i], self.b_data[i]))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.a_data.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: Element + Copy> ExactSizeIterator for BinaryBroadcastIter<'_, T> {}

impl<'a, T: Element> BinaryBroadcastIter<'a, T> {
    /// Apply a function to each element pair and collect the results.
    ///
    /// This uses the broadcast views directly (no materialization overhead
    /// beyond what was done at construction).
    pub fn map_collect<U, F>(self, f: F) -> Vec<U>
    where
        T: Copy,
        F: Fn(T, T) -> U,
    {
        self.a_view
            .iter()
            .zip(self.b_view.iter())
            .map(|(&a, &b)| f(a, b))
            .collect()
    }

    /// Apply a function to each element pair without collecting results.
    pub fn for_each<F>(self, mut f: F)
    where
        T: Copy,
        F: FnMut(T, T),
    {
        for (&a, &b) in self.a_view.iter().zip(self.b_view.iter()) {
            f(a, b);
        }
    }

    /// The broadcast shape of this iterator.
    pub fn shape(&self) -> &[usize] {
        self.a_view.shape()
    }

    /// Total number of elements in the broadcast shape.
    pub fn size(&self) -> usize {
        self.a_view.size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn binary_map_same_shape() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let c = NdIter::binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[3]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn binary_map_broadcast_1d_to_2d() {
        // (3, 1) + (4,) -> (3, 4)
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = NdIter::binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(
            data,
            vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0]
        );
    }

    #[test]
    fn binary_map_broadcast_scalar() {
        // (3,) + (1,) -> (3,)
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![100.0]).unwrap();
        let c = NdIter::binary_map(&a, &b, |x, y| x * y).unwrap();
        assert_eq!(c.shape(), &[3]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn binary_map_incompatible_shapes() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = NdIter::binary_map(&a, &b, |x, y| x + y);
        assert!(result.is_err());
    }

    #[test]
    fn binary_map_to_bool() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 5.0, 3.0, 7.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![2.0, 3.0, 3.0, 6.0]).unwrap();
        let c = NdIter::binary_map(&a, &b, |x, y| x > y).unwrap();
        assert_eq!(c.shape(), &[4]);
        let data: Vec<bool> = c.iter().copied().collect();
        assert_eq!(data, vec![false, true, false, true]);
    }

    #[test]
    fn binary_map_into_preallocated() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let mut out = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
        NdIter::binary_map_into(&a, &b, &mut out, |x, y| x + y).unwrap();
        let data: Vec<f64> = out.iter().copied().collect();
        assert_eq!(data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn binary_map_into_wrong_shape_error() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let mut out = Array::<f64, IxDyn>::zeros(IxDyn::new(&[5])).unwrap();
        let result = NdIter::binary_map_into(&a, &b, &mut out, |x, y| x + y);
        assert!(result.is_err());
    }

    #[test]
    fn unary_map_basic() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 4.0, 9.0, 16.0]).unwrap();
        let c = NdIter::unary_map(&a, |x| x.sqrt()).unwrap();
        assert_eq!(c.shape(), &[4]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unary_map_into_preallocated() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 4.0, 9.0]).unwrap();
        let mut out = Array::<f64, IxDyn>::zeros(IxDyn::new(&[3])).unwrap();
        NdIter::unary_map_into(&a, &mut out, |x| x * 2.0).unwrap();
        let data: Vec<f64> = out.iter().copied().collect();
        assert_eq!(data, vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn binary_iter_shape() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let iter = NdIter::binary_iter(&a, &b).unwrap();
        assert_eq!(iter.shape(), &[2, 3]);
    }

    #[test]
    fn binary_iter_map_collect() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let iter = NdIter::binary_iter(&a, &b).unwrap();
        let result: Vec<f64> = iter.map_collect(|x, y| x + y);
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn binary_map_3d_broadcast() {
        // (2, 1, 4) + (3, 1) -> (2, 3, 4)
        use crate::dimension::Ix3;
        let a = Array::<i32, Ix3>::from_vec(
            Ix3::new([2, 1, 4]),
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )
        .unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([3, 1]), vec![10, 20, 30]).unwrap();
        let c = NdIter::binary_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.shape(), &[2, 3, 4]);
        // First element: a[0,0,0] + b[0,0] = 1 + 10 = 11
        assert_eq!(*c.iter().next().unwrap(), 11);
    }

    // --- Iterator trait tests ---

    #[test]
    fn binary_iter_next() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let mut iter = NdIter::binary_iter(&a, &b).unwrap();
        assert_eq!(iter.next(), Some((1.0, 10.0)));
        assert_eq!(iter.next(), Some((2.0, 20.0)));
        assert_eq!(iter.next(), Some((3.0, 30.0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn binary_iter_for_loop() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let b = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let iter = NdIter::binary_iter(&a, &b).unwrap();
        let sums: Vec<i32> = iter.map(|(x, y)| x + y).collect();
        assert_eq!(sums, vec![11, 22, 33]);
    }

    #[test]
    fn binary_iter_broadcast_with_next() {
        // (2, 1) + (3,) -> (2, 3) — iterator yields 6 pairs
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let iter = NdIter::binary_iter(&a, &b).unwrap();
        assert_eq!(iter.len(), 6);
        let pairs: Vec<(f64, f64)> = iter.collect();
        assert_eq!(
            pairs,
            vec![
                (1.0, 10.0), (1.0, 20.0), (1.0, 30.0),
                (2.0, 10.0), (2.0, 20.0), (2.0, 30.0),
            ]
        );
    }

    #[test]
    fn binary_iter_exact_size() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0; 5]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![2.0; 5]).unwrap();
        let iter = NdIter::binary_iter(&a, &b).unwrap();
        assert_eq!(iter.len(), 5);
    }

    #[test]
    fn binary_iter_for_each_method() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let iter = NdIter::binary_iter(&a, &b).unwrap();
        let mut sum = 0.0;
        iter.for_each(|x, y| sum += x + y);
        assert!((sum - 66.0).abs() < 1e-10); // 11 + 22 + 33 = 66
    }

    // --- binary_map_mixed tests ---

    #[test]
    fn binary_map_mixed_i32_f64() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 1.5, 2.5]).unwrap();
        let c = NdIter::binary_map_mixed(&a, &b, |x, y| x as f64 + y).unwrap();
        assert_eq!(c.shape(), &[3]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn binary_map_mixed_broadcast() {
        // (3, 1) i32 + (4,) f64 -> (3, 4) f64
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([3, 1]), vec![1, 2, 3]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let c = NdIter::binary_map_mixed(&a, &b, |x, y| x as f64 + y).unwrap();
        assert_eq!(c.shape(), &[3, 4]);
        let first: f64 = *c.iter().next().unwrap();
        assert!((first - 1.1).abs() < 1e-10);
    }

    #[test]
    fn binary_map_mixed_to_bool() {
        // Compare i32 array > f64 threshold
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 5, 3, 7]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![2.0, 3.0, 3.0, 6.0]).unwrap();
        let c = NdIter::binary_map_mixed(&a, &b, |x, y| (x as f64) > y).unwrap();
        let data: Vec<bool> = c.iter().copied().collect();
        assert_eq!(data, vec![false, true, false, true]);
    }
}
