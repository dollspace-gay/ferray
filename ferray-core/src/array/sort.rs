// ferray-core: Array sort / argsort methods (#366)
//
// Minimal NumPy parity for ndarray.sort() / np.argsort() at the core
// level. Both methods flatten first and return a 1-D result. Axis-aware
// sorting with multiple algorithms lives in ferray-stats for callers
// that need the richer API (and for property tests that want to pin
// the stable-sort invariant against per-axis reductions).

use std::cmp::Ordering;

use crate::dimension::{Dimension, Ix1};
use crate::dtype::Element;

use super::owned::Array;

/// Total order comparator that sends "unorderable" (NaN) values to the
/// end, matching NumPy's `np.sort` convention. Uses self-comparison to
/// detect NaN without needing a `Float` bound on `T`: for any sane
/// `PartialOrd`, `x.partial_cmp(&x)` is `Some(Equal)`; only NaN-like
/// values break that invariant and return `None`.
fn nan_last_cmp<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    match a.partial_cmp(b) {
        Some(ord) => ord,
        None => {
            let a_nan = a.partial_cmp(a).is_none();
            let b_nan = b.partial_cmp(b).is_none();
            match (a_nan, b_nan) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                // Genuinely incomparable non-NaN values shouldn't exist
                // for any numeric Element type, but keep sort_by total
                // by treating them as equal.
                (false, false) => Ordering::Equal,
            }
        }
    }
}

impl<T, D> Array<T, D>
where
    T: Element + PartialOrd,
    D: Dimension,
{
    /// Return a sorted 1-D copy of the flattened array in ascending order.
    ///
    /// Equivalent to `np.sort(a.ravel())`. NaN values (for floating-point
    /// element types) are ordered last, matching NumPy's convention.
    ///
    /// For axis-aware sorting or algorithm selection (mergesort, heapsort),
    /// use `ferray_stats::sorting::sort`.
    pub fn sorted(&self) -> Array<T, Ix1> {
        let mut data: Vec<T> = self.iter().cloned().collect();
        data.sort_by(nan_last_cmp);
        let n = data.len();
        Array::<T, Ix1>::from_vec(Ix1::new([n]), data)
            .expect("from_vec with exact length cannot fail")
    }

    /// Return the indices that would sort the flattened array in ascending
    /// order.
    ///
    /// Equivalent to `np.argsort(a.ravel())`. The returned indices are
    /// `u64` to match NumPy's default index dtype. NaN values are sent
    /// to the tail of the sort order, matching [`Array::sorted`].
    ///
    /// For axis-aware argsort, use `ferray_stats::sorting::argsort`.
    pub fn argsort(&self) -> Array<u64, Ix1> {
        let data: Vec<T> = self.iter().cloned().collect();
        let mut idx: Vec<u64> = (0..data.len() as u64).collect();
        idx.sort_by(|&i, &j| nan_last_cmp(&data[i as usize], &data[j as usize]));
        let n = idx.len();
        Array::<u64, Ix1>::from_vec(Ix1::new([n]), idx)
            .expect("from_vec with exact length cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn sorted_ascending_1d() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![3, 1, 4, 1, 5]).unwrap();
        let s = a.sorted();
        assert_eq!(s.shape(), &[5]);
        assert_eq!(s.as_slice().unwrap(), &[1, 1, 3, 4, 5]);
        // Source unchanged.
        assert_eq!(a.as_slice().unwrap(), &[3, 1, 4, 1, 5]);
    }

    #[test]
    fn sorted_flattens_2d() {
        // Row-major flatten, then sort.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![5, 2, 8, 1, 9, 4]).unwrap();
        let s = a.sorted();
        assert_eq!(s.shape(), &[6]);
        assert_eq!(s.as_slice().unwrap(), &[1, 2, 4, 5, 8, 9]);
    }

    #[test]
    fn sorted_f64_nans_go_last() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, f64::NAN, 1.0, f64::NAN, 2.0])
            .unwrap();
        let s = a.sorted();
        let data = s.as_slice().unwrap();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
        assert!(data[3].is_nan());
        assert!(data[4].is_nan());
    }

    #[test]
    fn sorted_empty() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let s = a.sorted();
        assert_eq!(s.shape(), &[0]);
        assert_eq!(s.size(), 0);
    }

    #[test]
    fn argsort_matches_sorted_1d() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![3, 1, 4, 1, 5]).unwrap();
        let idx = a.argsort();
        assert_eq!(idx.shape(), &[5]);
        // Applying the index permutation to `a` must yield the sorted array.
        let data = a.as_slice().unwrap();
        let picked: Vec<i32> = idx
            .as_slice()
            .unwrap()
            .iter()
            .map(|&i| data[i as usize])
            .collect();
        assert_eq!(picked, vec![1, 1, 3, 4, 5]);
    }

    #[test]
    fn argsort_f64_sends_nans_last() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![2.0, f64::NAN, 0.5, 5.0]).unwrap();
        let idx = a.argsort();
        let data = a.as_slice().unwrap();
        let picked: Vec<f64> = idx
            .as_slice()
            .unwrap()
            .iter()
            .map(|&i| data[i as usize])
            .collect();
        assert_eq!(picked[0], 0.5);
        assert_eq!(picked[1], 2.0);
        assert_eq!(picked[2], 5.0);
        assert!(picked[3].is_nan());
    }

    #[test]
    fn argsort_flattens_2d() {
        // [[5,2,8],[1,9,4]] -> flat [5,2,8,1,9,4]
        // sorted ascending: [1,2,4,5,8,9] at flat indices [3,1,5,0,2,4]
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![5, 2, 8, 1, 9, 4]).unwrap();
        let idx = a.argsort();
        assert_eq!(idx.shape(), &[6]);
        assert_eq!(idx.as_slice().unwrap(), &[3, 1, 5, 0, 2, 4]);
    }
}
