// ferray-core: Reduction methods for Array<T, D>
//
// Provides NumPy-equivalent reduction methods directly on Array and ArrayView:
//   sum, prod, min, max, mean, var, std, any, all
//
// Each reduction has a whole-array variant and an axis variant:
//   .sum()             -> T
//   .sum_axis(Axis(0)) -> Array<T, IxDyn>
//
// These methods complement the lower-level fold_axis primitive in methods.rs
// and the free-function reductions in ferray-stats. The instance-method form
// matches NumPy's `arr.sum()` ergonomics so users don't need to import a
// separate crate just to compute a sum.
//
// ## REQ status (reductions, NumPy parity)
//  - sum / prod / sum_axis / prod_axis accumulator promotion — SHIPPED (#780):
//    the `ReduceAcc` trait (this file) maps narrow signed ints → i64, narrow
//    unsigned ints → u64, and bool → i64 before reducing, so narrow-int
//    reductions never overflow and the result dtype matches numpy
//    (numpy/_core/fromnumeric.py:2321-2327). Consumers: `Array::sum`/`prod`/
//    `sum_axis`/`prod_axis` and the `ArrayView` mirrors (all in this file).
//  - min / max / mean / var / std / any / all — SHIPPED (#368), NaN-propagating.
//  - empty-array min/max raising ValueError — NOT-STARTED (open blocker #782).
//  - cumsum / cumprod live as ferray-stats free functions; their narrow-int
//    promotion is a separate ferray-stats blocker that reuses `ReduceAcc`.
//
// See: https://github.com/dollspace-gay/ferray/issues/368, /issues/780

use num_traits::Float;

use crate::array::owned::Array;
use crate::array::view::ArrayView;
use crate::dimension::{Axis, Dimension, IxDyn};
use crate::dtype::Element;
use crate::error::FerrayResult;

// ---------------------------------------------------------------------------
// ReduceAcc — NumPy's sum/prod/cumsum/cumprod accumulator-and-result dtype.
// ---------------------------------------------------------------------------

/// Maps an element type `T` to the type NumPy uses to *accumulate* (and
/// return) `sum` / `prod` / `cumsum` / `cumprod` over it.
///
/// NumPy promotes any integer dtype of *less precision than the default
/// platform integer* before reducing, so a narrow-int reduction can never
/// overflow and the result dtype is the platform integer:
///
/// > "The dtype of `a` is used by default unless `a` has an integer dtype of
/// > less precision than the default platform integer.  In that case, if `a`
/// > is signed then the platform integer is used while if `a` is unsigned then
/// > an unsigned integer of the same precision as the platform integer is
/// > used."
/// > — `numpy/_core/fromnumeric.py:2321-2327` (sum), `:3306-3312` (prod)
///
/// The reduction itself is `umr_sum = um.add.reduce` /
/// `umr_prod = um.multiply.reduce` (`numpy/_core/_methods.py:20-21`), i.e. the
/// add/multiply ufunc whose *loop dtype* is the promoted accumulator.
///
/// The mapping (platform integer = 64-bit, matching ferray's `intp`/`int64`):
///   - `i8 / i16 / i32 → i64`,  `i64 → i64`,  `i128 → i128`
///   - `u8 / u16 / u32 → u64`,  `u64 → u64`,  `u128 → u128`
///   - `bool → i64` (NumPy reduces bool as the platform integer, counting `true`)
///   - `f32 → f32`, `f64 → f64`, complex stays itself (no promotion)
///
/// Wider-or-equal dtypes map to themselves, so existing `f64`/`i64`/complex
/// reductions are unchanged — only narrow-int callers observe the promoted
/// return type.
pub trait ReduceAcc: Element + Copy {
    /// The accumulator-and-result element type for reductions over `Self`.
    type Acc: Element + Copy + std::ops::Add<Output = Self::Acc> + std::ops::Mul<Output = Self::Acc>;

    /// Widen one element into the accumulator type before reducing, matching
    /// NumPy's promotion of the loop dtype (`true → 1` for `bool`).
    fn widen(self) -> Self::Acc;
}

macro_rules! impl_reduce_acc {
    ($($t:ty => $acc:ty),* $(,)?) => {
        $(
            impl ReduceAcc for $t {
                type Acc = $acc;
                #[inline]
                fn widen(self) -> $acc {
                    self as $acc
                }
            }
        )*
    };
}

// Narrow signed ints promote to i64; i64/i128 stay themselves.
impl_reduce_acc! {
    i8 => i64, i16 => i64, i32 => i64, i64 => i64, i128 => i128,
    u8 => u64, u16 => u64, u32 => u64, u64 => u64, u128 => u128,
    f32 => f32, f64 => f64,
}

// bool reduces as the platform integer, counting `true` (numpy:
// `np.sum(np.array([True, True, True])).dtype == int64`). `as i64` maps
// false→0, true→1.
impl ReduceAcc for bool {
    type Acc = i64;
    #[inline]
    fn widen(self) -> i64 {
        i64::from(self)
    }
}

// Complex stays itself — numpy never promotes a complex reduction.
impl ReduceAcc for num_complex::Complex<f32> {
    type Acc = num_complex::Complex<f32>;
    #[inline]
    fn widen(self) -> Self {
        self
    }
}

impl ReduceAcc for num_complex::Complex<f64> {
    type Acc = num_complex::Complex<f64>;
    #[inline]
    fn widen(self) -> Self {
        self
    }
}

/// Generic min/max fold step that propagates NaN per `NumPy` semantics.
///
/// Once any NaN enters the fold, all subsequent steps return NaN. Detected
/// generically via `x.partial_cmp(&x).is_none()`, which is true iff `x` is
/// NaN (or any other value that violates `PartialOrd` reflexivity, e.g.
/// `Complex` types — but those don't implement `PartialOrd` so this is moot).
#[inline]
fn reduce_step<T: PartialOrd + Copy>(acc: T, x: T, take_min: bool) -> T {
    let acc_is_nan = acc.partial_cmp(&acc).is_none();
    if acc_is_nan {
        return acc;
    }
    let x_is_nan = x.partial_cmp(&x).is_none();
    if x_is_nan {
        return x;
    }
    match (take_min, x.partial_cmp(&acc)) {
        (true, Some(std::cmp::Ordering::Less)) => x,
        (false, Some(std::cmp::Ordering::Greater)) => x,
        _ => acc,
    }
}

// ---------------------------------------------------------------------------
// Sum / Prod (work for any Element with Add/Mul, using Element::zero/one)
// ---------------------------------------------------------------------------

impl<T, D> Array<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Sum of all elements (whole-array reduction).
    ///
    /// The result type is the NumPy reduction accumulator
    /// [`ReduceAcc::Acc`]: narrow signed ints widen to `i64`, narrow unsigned
    /// ints to `u64`, `bool` to `i64`, and `f32`/`f64`/complex stay
    /// themselves. This means a narrow-int sum can never overflow and its
    /// dtype matches `np.sum`'s promoted result
    /// (`numpy/_core/fromnumeric.py:2321-2327`).
    ///
    /// Returns `Acc::zero()` for an empty array.
    ///
    /// # Examples
    /// ```
    /// # use ferray_core::Array;
    /// # use ferray_core::dimension::Ix1;
    /// let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(a.sum(), 6.0);
    /// // i8 sums promote to i64 and never overflow (numpy parity):
    /// let b = Array::<i8, Ix1>::from_vec(Ix1::new([3]), vec![100, 100, 100]).unwrap();
    /// assert_eq!(b.sum(), 300_i64);
    /// ```
    pub fn sum(&self) -> <T as ReduceAcc>::Acc
    where
        T: ReduceAcc,
    {
        let mut acc = <T as ReduceAcc>::Acc::zero();
        for &x in self.iter() {
            acc = acc + x.widen();
        }
        acc
    }

    /// Sum along the given axis. Returns an array with one fewer dimension,
    /// whose element type is the promoted [`ReduceAcc::Acc`] (same numpy
    /// narrow-int promotion as the whole-array [`Array::sum`]).
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
    pub fn sum_axis(&self, axis: Axis) -> FerrayResult<Array<<T as ReduceAcc>::Acc, IxDyn>>
    where
        T: ReduceAcc,
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let widened = self.map_to::<<T as ReduceAcc>::Acc>(ReduceAcc::widen);
        widened.fold_axis(axis, <T as ReduceAcc>::Acc::zero(), |acc, &x| *acc + x)
    }

    /// Product of all elements.
    ///
    /// The result type is the promoted [`ReduceAcc::Acc`] (same numpy
    /// narrow-int promotion as [`Array::sum`]; see
    /// `numpy/_core/fromnumeric.py:3306-3312`).
    ///
    /// Returns `Acc::one()` for an empty array.
    pub fn prod(&self) -> <T as ReduceAcc>::Acc
    where
        T: ReduceAcc,
    {
        let mut acc = <T as ReduceAcc>::Acc::one();
        for &x in self.iter() {
            acc = acc * x.widen();
        }
        acc
    }

    /// Product along the given axis. Element type is the promoted
    /// [`ReduceAcc::Acc`].
    pub fn prod_axis(&self, axis: Axis) -> FerrayResult<Array<<T as ReduceAcc>::Acc, IxDyn>>
    where
        T: ReduceAcc,
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let widened = self.map_to::<<T as ReduceAcc>::Acc>(ReduceAcc::widen);
        widened.fold_axis(axis, <T as ReduceAcc>::Acc::one(), |acc, &x| *acc * x)
    }
}

// ---------------------------------------------------------------------------
// Min / Max — require PartialOrd
// ---------------------------------------------------------------------------

impl<T, D> Array<T, D>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    /// Minimum value across the entire array.
    ///
    /// Returns `None` if the array is empty. NaN values follow `NumPy` semantics:
    /// once a NaN is seen the result stays NaN, detected via self-comparison
    /// (`x.partial_cmp(&x).is_none()`).
    pub fn min(&self) -> Option<T> {
        let mut iter = self.iter().copied();
        let first = iter.next()?;
        Some(iter.fold(first, |acc, x| reduce_step(acc, x, true)))
    }

    /// Maximum value across the entire array.
    ///
    /// Returns `None` if the array is empty. NaN values propagate per `NumPy`.
    pub fn max(&self) -> Option<T> {
        let mut iter = self.iter().copied();
        let first = iter.next()?;
        Some(iter.fold(first, |acc, x| reduce_step(acc, x, false)))
    }

    /// Minimum value along an axis.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`, or
    /// `FerrayError::ShapeMismatch` if the resulting axis would be empty.
    pub fn min_axis(&self, axis: Axis) -> FerrayResult<Array<T, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        // Use the first element along the axis as init by sentinel: pull the
        // first lane and fold the rest. fold_axis applies init to every lane,
        // but min has no neutral identity for arbitrary T. We sidestep by
        // folding starting from any element of `self` — the per-lane init is
        // overwritten by the first comparison, which is correct iff every lane
        // has at least one element. Empty axes would yield uninitialized data.
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(crate::error::FerrayError::axis_out_of_bounds(
                axis.index(),
                ndim,
            ));
        }
        if self.shape()[axis.index()] == 0 {
            return Err(crate::error::FerrayError::shape_mismatch(
                "cannot compute min along empty axis",
            ));
        }
        // Manual lane iteration: fold_axis can't be used here because min has
        // no neutral identity that works for arbitrary `T: PartialOrd` (no
        // T::infinity for ints).
        self.fold_axis_min_max(axis, true)
    }

    /// Maximum value along an axis.
    ///
    /// See [`Array::min_axis`] for error semantics.
    pub fn max_axis(&self, axis: Axis) -> FerrayResult<Array<T, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(crate::error::FerrayError::axis_out_of_bounds(
                axis.index(),
                ndim,
            ));
        }
        if self.shape()[axis.index()] == 0 {
            return Err(crate::error::FerrayError::shape_mismatch(
                "cannot compute max along empty axis",
            ));
        }
        self.fold_axis_min_max(axis, false)
    }

    /// Internal: per-lane min/max via manual lane iteration. Avoids the
    /// init-bias problem of `fold_axis` (which applies a single init to every
    /// lane, even though min/max have no identity element).
    fn fold_axis_min_max(&self, axis: Axis, take_min: bool) -> FerrayResult<Array<T, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let nd_axis = ndarray::Axis(axis.index());
        // Use ndarray's lane iteration directly via the inner ndarray::ArrayBase.
        // Each lane is a 1D view orthogonal to the chosen axis.
        let lanes = self.inner.lanes(nd_axis);
        let mut out: Vec<T> = Vec::with_capacity(lanes.into_iter().len());
        for lane in self.inner.lanes(nd_axis) {
            let mut iter = lane.iter().copied();
            let first = iter.next().unwrap(); // safe: empty axis already rejected
            let result = iter.fold(first, |acc, x| reduce_step(acc, x, take_min));
            out.push(result);
        }

        // Output shape: drop the reduced axis from the input shape.
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        out_shape.remove(axis.index());
        Array::from_vec(IxDyn::from(&out_shape[..]), out)
    }
}

// ---------------------------------------------------------------------------
// Mean / Var / Std — require Float
// ---------------------------------------------------------------------------

impl<T, D> Array<T, D>
where
    T: Element + Float,
    D: Dimension,
{
    /// Arithmetic mean of all elements. Returns `None` for an empty array.
    pub fn mean(&self) -> Option<T> {
        let n = self.size();
        if n == 0 {
            return None;
        }
        let sum: T = self
            .iter()
            .copied()
            .fold(<T as Element>::zero(), |acc, x| acc + x);
        Some(sum / T::from(n).unwrap())
    }

    /// Mean along an axis.
    pub fn mean_axis(&self, axis: Axis) -> FerrayResult<Array<T, IxDyn>>
    where
        T: ReduceAcc<Acc = T>,
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(crate::error::FerrayError::axis_out_of_bounds(
                axis.index(),
                ndim,
            ));
        }
        let n = self.shape()[axis.index()];
        if n == 0 {
            return Err(crate::error::FerrayError::shape_mismatch(
                "cannot compute mean along empty axis",
            ));
        }
        let sums = self.sum_axis(axis)?;
        let n_t = T::from(n).unwrap();
        Ok(sums.mapv(|x| x / n_t))
    }

    /// Variance with `ddof` degrees of freedom (Bessel's correction = 1).
    ///
    /// Returns `None` for an empty array, or when `ddof >= n`.
    pub fn var(&self, ddof: usize) -> Option<T> {
        let n = self.size();
        if n == 0 || ddof >= n {
            return None;
        }
        let mean = self.mean()?;
        let sum_sq: T = self.iter().copied().fold(<T as Element>::zero(), |acc, x| {
            acc + (x - mean) * (x - mean)
        });
        Some(sum_sq / T::from(n - ddof).unwrap())
    }

    /// Standard deviation with `ddof` degrees of freedom.
    pub fn std(&self, ddof: usize) -> Option<T> {
        self.var(ddof).map(num_traits::Float::sqrt)
    }
}

// ---------------------------------------------------------------------------
// any / all — for bool arrays
// ---------------------------------------------------------------------------

impl<D> Array<bool, D>
where
    D: Dimension,
{
    /// Returns `true` if any element is `true`.
    pub fn any(&self) -> bool {
        self.iter().any(|&x| x)
    }

    /// Returns `true` if all elements are `true`. Vacuously `true` for empty arrays.
    pub fn all(&self) -> bool {
        self.iter().all(|&x| x)
    }
}

// ---------------------------------------------------------------------------
// ArrayView mirrors — same methods on borrowed views
// ---------------------------------------------------------------------------

impl<T, D> ArrayView<'_, T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Sum of all elements. See [`Array::sum`] — returns the promoted
    /// [`ReduceAcc::Acc`].
    pub fn sum(&self) -> <T as ReduceAcc>::Acc
    where
        T: ReduceAcc,
    {
        let mut acc = <T as ReduceAcc>::Acc::zero();
        for &x in self.iter() {
            acc = acc + x.widen();
        }
        acc
    }

    /// Product of all elements. See [`Array::prod`] — returns the promoted
    /// [`ReduceAcc::Acc`].
    pub fn prod(&self) -> <T as ReduceAcc>::Acc
    where
        T: ReduceAcc,
    {
        let mut acc = <T as ReduceAcc>::Acc::one();
        for &x in self.iter() {
            acc = acc * x.widen();
        }
        acc
    }
}

impl<T, D> ArrayView<'_, T, D>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    /// Minimum value. See [`Array::min`].
    pub fn min(&self) -> Option<T> {
        let mut iter = self.iter().copied();
        let first = iter.next()?;
        Some(iter.fold(first, |acc, x| reduce_step(acc, x, true)))
    }

    /// Maximum value. See [`Array::max`].
    pub fn max(&self) -> Option<T> {
        let mut iter = self.iter().copied();
        let first = iter.next()?;
        Some(iter.fold(first, |acc, x| reduce_step(acc, x, false)))
    }
}

impl<T, D> ArrayView<'_, T, D>
where
    T: Element + Float,
    D: Dimension,
{
    /// Mean. See [`Array::mean`].
    pub fn mean(&self) -> Option<T> {
        let n = self.size();
        if n == 0 {
            return None;
        }
        let sum: T = self
            .iter()
            .copied()
            .fold(<T as Element>::zero(), |acc, x| acc + x);
        Some(sum / T::from(n).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr2(rows: usize, cols: usize, data: Vec<f64>) -> Array<f64, Ix2> {
        Array::from_vec(Ix2::new([rows, cols]), data).unwrap()
    }

    // ----- sum / prod -----

    #[test]
    fn sum_1d() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.sum(), 10.0);
    }

    #[test]
    fn sum_empty_returns_zero() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert_eq!(a.sum(), 0.0);
    }

    #[test]
    fn sum_axis_2d() {
        let a = arr2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Sum across rows (axis 0): [1+4, 2+5, 3+6] = [5, 7, 9]
        let s0 = a.sum_axis(Axis(0)).unwrap();
        assert_eq!(s0.shape(), &[3]);
        assert_eq!(s0.iter().copied().collect::<Vec<_>>(), vec![5.0, 7.0, 9.0]);

        // Sum across columns (axis 1): [1+2+3, 4+5+6] = [6, 15]
        let s1 = a.sum_axis(Axis(1)).unwrap();
        assert_eq!(s1.shape(), &[2]);
        assert_eq!(s1.iter().copied().collect::<Vec<_>>(), vec![6.0, 15.0]);
    }

    #[test]
    fn prod_1d() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.prod(), 24.0);
    }

    #[test]
    fn prod_empty_returns_one() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert_eq!(a.prod(), 1.0);
    }

    #[test]
    fn prod_axis_2d() {
        let a = arr2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p0 = a.prod_axis(Axis(0)).unwrap();
        assert_eq!(
            p0.iter().copied().collect::<Vec<_>>(),
            vec![4.0, 10.0, 18.0]
        );

        let p1 = a.prod_axis(Axis(1)).unwrap();
        assert_eq!(p1.iter().copied().collect::<Vec<_>>(), vec![6.0, 120.0]);
    }

    // ----- min / max -----

    #[test]
    fn min_max_1d() {
        let a = arr1(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]);
        assert_eq!(a.min(), Some(1.0));
        assert_eq!(a.max(), Some(9.0));
    }

    #[test]
    fn min_max_empty_returns_none() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert_eq!(a.min(), None);
        assert_eq!(a.max(), None);
    }

    #[test]
    fn min_max_int() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![3, -1, 4, -5, 2]).unwrap();
        assert_eq!(a.min(), Some(-5));
        assert_eq!(a.max(), Some(4));
    }

    #[test]
    fn min_max_axis_2d() {
        let a = arr2(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        // axis 0: min/max per column
        let mn0 = a.min_axis(Axis(0)).unwrap();
        assert_eq!(mn0.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        let mx0 = a.max_axis(Axis(0)).unwrap();
        assert_eq!(mx0.iter().copied().collect::<Vec<_>>(), vec![4.0, 5.0, 6.0]);

        // axis 1: min/max per row
        let mn1 = a.min_axis(Axis(1)).unwrap();
        assert_eq!(mn1.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0]);
        let mx1 = a.max_axis(Axis(1)).unwrap();
        assert_eq!(mx1.iter().copied().collect::<Vec<_>>(), vec![5.0, 6.0]);
    }

    // ----- mean / var / std -----

    #[test]
    fn mean_1d() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean(), Some(2.5));
    }

    #[test]
    fn mean_empty_returns_none() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert_eq!(a.mean(), None);
    }

    #[test]
    fn mean_axis_2d() {
        let a = arr2(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m0 = a.mean_axis(Axis(0)).unwrap();
        assert_eq!(m0.iter().copied().collect::<Vec<_>>(), vec![2.5, 3.5, 4.5]);
        let m1 = a.mean_axis(Axis(1)).unwrap();
        assert_eq!(m1.iter().copied().collect::<Vec<_>>(), vec![2.0, 5.0]);
    }

    #[test]
    fn var_population() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // population variance (ddof=0): ((1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2)/5 = 10/5 = 2
        assert_eq!(a.var(0), Some(2.0));
    }

    #[test]
    fn var_sample() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // sample variance (ddof=1): 10/4 = 2.5
        assert_eq!(a.var(1), Some(2.5));
    }

    #[test]
    fn std_basic() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = a.std(0).unwrap();
        assert!((s - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn var_ddof_too_large_returns_none() {
        let a = arr1(vec![1.0, 2.0]);
        assert_eq!(a.var(2), None);
        assert_eq!(a.var(5), None);
    }

    // ----- any / all -----

    #[test]
    fn any_all_bool() {
        let true_arr = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        let mixed = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let false_arr =
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let empty = Array::<bool, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();

        assert!(true_arr.all());
        assert!(true_arr.any());

        assert!(!mixed.all());
        assert!(mixed.any());

        assert!(!false_arr.all());
        assert!(!false_arr.any());

        // Vacuous truth for empty
        assert!(empty.all());
        assert!(!empty.any());
    }

    // ----- ArrayView mirrors -----

    #[test]
    fn view_sum_min_max_mean() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let v = a.view();
        assert_eq!(v.sum(), 10.0);
        assert_eq!(v.min(), Some(1.0));
        assert_eq!(v.max(), Some(4.0));
        assert_eq!(v.mean(), Some(2.5));
    }

    #[test]
    fn nan_propagates_in_min_max() {
        // NaN somewhere in the middle
        let a = arr1(vec![1.0, f64::NAN, 3.0]);
        assert!(a.min().unwrap().is_nan());
        assert!(a.max().unwrap().is_nan());

        // NaN at the start
        let b = arr1(vec![f64::NAN, 1.0, 3.0]);
        assert!(b.min().unwrap().is_nan());
        assert!(b.max().unwrap().is_nan());

        // NaN at the end
        let c = arr1(vec![1.0, 3.0, f64::NAN]);
        assert!(c.min().unwrap().is_nan());
        assert!(c.max().unwrap().is_nan());
    }
}
