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
//  - empty-array min/max raising ValueError — SHIPPED (#782, resolved R-DEV-4):
//    `Array::min`/`max` return `Option<T>` = `None` on an empty array (the
//    idiomatic Rust analog of numpy's no-identity error — same boundary contract
//    as `argmax`/`argmin`, REQ-40/41; test `min_max_empty_returns_none` in this
//    file). The ferray-python boundary maps `None` -> `ValueError`, matching
//    numpy's `ValueError` ("zero-size array to reduction operation minimum which
//    has no identity", `fromnumeric.py:3150` `min`/`:3266` `amin`). Verified:
//    `fr.min(fr.array([], dtype=fr.float64))` raises `ValueError` vs numpy 2.4.5.
//  - cumsum / cumprod live as ferray-stats free functions; their narrow-int
//    promotion is a separate ferray-stats blocker that reuses `ReduceAcc`.
//  - argmax / argmin (REQ-40, REQ-41) — SHIPPED: `Array::argmax`/`argmin`
//    (flattened, returning `Option<i64>`) and `argmax_axis`/`argmin_axis`
//    (returning `Array<i64, IxDyn>`) in this file. First-occurrence on ties and
//    NaN-first propagation (numpy/_core/fromnumeric.py:1222 `argmax`,
//    :1261-1262 ties; :1322 `argmin`, :1361-1362 ties). Empty flattened form
//    returns `None` (mirroring `min`/`max`'s `Option` analog; the ferray-python
//    boundary maps `None`→`ValueError` as numpy does). Result index dtype is
//    `i64` (ferray's `intp` analog), independent of element dtype. Consumers:
//    the boundary methods themselves are the public API surface (like
//    `sum`/`min`), exercised by the `ArrayView` mirror and the in-file tests.
//  - integer/bool mean → f64 (REQ-42) — SHIPPED: the `MeanAcc` trait (this
//    file) maps bool/integer element types to an `f64` accumulator-and-result,
//    while `f32`/`f64`/complex stay themselves, so `Array::<i32, _>::mean()`
//    returns `f64` matching numpy, which casts bool/unsigned/signed int to
//    float64 before averaging (numpy/_core/_methods.py:124-127). `mean`/
//    `mean_axis` and the `ArrayView::mean` mirror are now bounded by `MeanAcc`
//    instead of `Float`; existing `f32`/`f64` means are unchanged
//    (`MeanAcc::Mean == Self`). Consumers: `Array::var`/`std` (`self.mean()?`)
//    and the `ArrayView` mirror, all in this file.
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

// ---------------------------------------------------------------------------
// MeanAcc — NumPy's mean accumulator-and-result dtype.
// ---------------------------------------------------------------------------

/// Maps an element type `T` to the type NumPy uses to *accumulate* (and
/// return) `mean` over it.
///
/// NumPy casts a bool / unsigned-int / signed-int input to `float64` before
/// averaging:
///
/// > "Cast bool, unsigned int, and int to float64 by default ...
/// > `dtype = mu.dtype('f8')`"
/// > — `numpy/_core/_methods.py:124-127`
///
/// so `np.mean(np.array([1, 2, 3], np.int32))` is `float64 2.0` and
/// `np.mean([True, False, True])` is `float64 0.6666…`. Floating-point inputs
/// keep their own dtype (`f32`→`f32`, `f64`→`f64`), and complex stays itself.
///
/// The mapping:
///   - `bool / i8.. / u8.. → f64`
///   - `f32 → f32`, `f64 → f64` (unchanged — `Mean == Self`)
///   - `Complex<f32> → Complex<f32>`, `Complex<f64> → Complex<f64>`
pub trait MeanAcc: Element + Copy {
    /// The accumulator-and-result element type for `mean` over `Self`.
    type Mean: Element
        + Copy
        + std::ops::Add<Output = Self::Mean>
        + std::ops::Div<Output = Self::Mean>;

    /// Widen one element into the mean accumulator type, matching NumPy's
    /// pre-average cast (`true → 1.0`, `false → 0.0` for `bool`).
    fn widen_mean(self) -> Self::Mean;

    /// Construct the divisor (element count `n`) in the accumulator type.
    fn count(n: usize) -> Self::Mean;
}

macro_rules! impl_mean_acc_to_f64 {
    ($($t:ty),* $(,)?) => {
        $(
            impl MeanAcc for $t {
                type Mean = f64;
                #[inline]
                fn widen_mean(self) -> f64 {
                    self as f64
                }
                #[inline]
                fn count(n: usize) -> f64 {
                    n as f64
                }
            }
        )*
    };
}

// bool / all integer dtypes average in f64 (numpy/_core/_methods.py:124-127).
impl_mean_acc_to_f64!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

impl MeanAcc for bool {
    type Mean = f64;
    #[inline]
    fn widen_mean(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
    #[inline]
    fn count(n: usize) -> f64 {
        n as f64
    }
}

// Floating-point inputs keep their own dtype (Mean == Self).
impl MeanAcc for f32 {
    type Mean = f32;
    #[inline]
    fn widen_mean(self) -> f32 {
        self
    }
    #[inline]
    fn count(n: usize) -> f32 {
        n as f32
    }
}

impl MeanAcc for f64 {
    type Mean = f64;
    #[inline]
    fn widen_mean(self) -> f64 {
        self
    }
    #[inline]
    fn count(n: usize) -> f64 {
        n as f64
    }
}

impl MeanAcc for num_complex::Complex<f32> {
    type Mean = num_complex::Complex<f32>;
    #[inline]
    fn widen_mean(self) -> Self {
        self
    }
    #[inline]
    fn count(n: usize) -> Self {
        num_complex::Complex::new(n as f32, 0.0)
    }
}

impl MeanAcc for num_complex::Complex<f64> {
    type Mean = num_complex::Complex<f64>;
    #[inline]
    fn widen_mean(self) -> Self {
        self
    }
    #[inline]
    fn count(n: usize) -> Self {
        num_complex::Complex::new(n as f64, 0.0)
    }
}

/// First-occurrence, NaN-first arg-reduction over a flat element iterator.
///
/// Mirrors NumPy's `argmax`/`argmin` (`numpy/_core/fromnumeric.py:1222`,
/// `:1322`): on ties the *first* occurrence wins (`:1261-1262`, `:1361-1362`),
/// and when any NaN is present the index of the *first* NaN is returned
/// (NaN-propagating, NaN-first — live oracle numpy 2.4.5:
/// `np.argmax([1.0, nan, 3.0, nan]) == 1`).
///
/// Returns `None` for an empty iterator, the `Option` analog `min`/`max` use;
/// the ferray-python boundary maps `None`→`ValueError` as numpy does.
#[inline]
fn arg_reduce<T: PartialOrd + Copy>(iter: impl Iterator<Item = T>, take_min: bool) -> Option<i64> {
    let mut best_idx: Option<i64> = None;
    let mut best: Option<T> = None;
    for (i, x) in iter.enumerate() {
        let i = i as i64;
        // NaN-first: the first NaN seen wins immediately and is never beaten.
        if x.partial_cmp(&x).is_none() {
            return Some(i);
        }
        match best {
            None => {
                best = Some(x);
                best_idx = Some(i);
            }
            Some(b) => {
                // Strict comparison => first occurrence wins on ties.
                let replace = match x.partial_cmp(&b) {
                    Some(std::cmp::Ordering::Less) => take_min,
                    Some(std::cmp::Ordering::Greater) => !take_min,
                    _ => false,
                };
                if replace {
                    best = Some(x);
                    best_idx = Some(i);
                }
            }
        }
    }
    best_idx
}

/// Generic min/max fold step that propagates NaN per `NumPy` semantics.
///
/// Once any NaN enters the fold, all subsequent steps return NaN. Detected
/// generically via `x.partial_cmp(&x).is_none()`, which is true iff `x` is
/// NaN (or any other value that violates `PartialOrd` reflexivity, e.g.
/// `Complex` types — but those don't implement `PartialOrd` so this is moot).
///
/// On an equal compare (`Ordering::Equal`) the NEW element `x` is kept, not the
/// accumulator. This mirrors `numpy`'s `maximum.reduce`/`minimum.reduce`
/// (`numpy/_core/_methods.py:38-44`, `umr_maximum`/`umr_minimum`), whose
/// underlying scalar `maximum(a, b)`/`minimum(a, b)` loops return the *later*
/// operand on ties — observable only for signed zeros (`+0.0 == -0.0`), where
/// numpy keeps the LAST seen zero's sign bit. For any non-signed-zero equal
/// pair the values are identical, so this changes nothing. This is the VALUE
/// min/max reduce; `argmin`/`argmax` use first-occurrence on ties and live on a
/// separate code path (they do not call `reduce_step`).
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
        // Tie: keep the LAST operand (numpy maximum/minimum.reduce semantics).
        (_, Some(std::cmp::Ordering::Equal)) => x,
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

    /// Flat index of the maximum element (whole-array reduction).
    ///
    /// Returns `None` for an empty array (the `Option` analog `min`/`max`
    /// use; the ferray-python boundary maps `None`→`ValueError`, matching
    /// `np.argmax([])`). On ties the **first** occurrence wins, and when any
    /// NaN is present the index of the **first** NaN is returned (NaN-first),
    /// matching `np.argmax` (`numpy/_core/fromnumeric.py:1222`, ties at
    /// `:1261-1262`; live oracle `np.argmax([1.0, nan, 3.0, nan]) == 1`). The
    /// index type is `i64` (ferray's `intp` analog), independent of `T`.
    pub fn argmax(&self) -> Option<i64> {
        arg_reduce(self.iter().copied(), false)
    }

    /// Flat index of the minimum element (whole-array reduction).
    ///
    /// Mirror of [`Array::argmax`] with min substituted for max: first
    /// occurrence on ties, NaN-first, `None` on empty, `i64` index
    /// (`numpy/_core/fromnumeric.py:1322`, ties at `:1361-1362`; live oracle
    /// `np.argmin([1.0, nan, 3.0]) == 1`).
    pub fn argmin(&self) -> Option<i64> {
        arg_reduce(self.iter().copied(), true)
    }

    /// Indices of the maxima along `axis`, as an `Array<i64, IxDyn>` with the
    /// reduced axis removed. First-occurrence on ties, NaN-first per lane
    /// (`numpy/_core/fromnumeric.py:1222`).
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`, or
    /// `FerrayError::ShapeMismatch` if the reduced axis is empty (matching
    /// numpy's `ValueError` on an empty argmax axis).
    pub fn argmax_axis(&self, axis: Axis) -> FerrayResult<Array<i64, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        self.arg_axis(axis, false)
    }

    /// Indices of the minima along `axis`. See [`Array::argmax_axis`].
    pub fn argmin_axis(&self, axis: Axis) -> FerrayResult<Array<i64, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        self.arg_axis(axis, true)
    }

    /// Internal: per-lane arg-reduction along `axis`. Each lane is a 1D view
    /// orthogonal to `axis`; the reduced index is the position within the lane.
    fn arg_axis(&self, axis: Axis, take_min: bool) -> FerrayResult<Array<i64, IxDyn>>
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
                "attempt to get argmax/argmin of an empty axis",
            ));
        }
        let nd_axis = ndarray::Axis(axis.index());
        let lanes = self.inner.lanes(nd_axis);
        let mut out: Vec<i64> = Vec::with_capacity(lanes.into_iter().len());
        for lane in self.inner.lanes(nd_axis) {
            // Lane is non-empty (empty axis already rejected), so arg_reduce
            // returns Some; default 0 is unreachable but keeps the code panic-free.
            let idx = arg_reduce(lane.iter().copied(), take_min).unwrap_or(0);
            out.push(idx);
        }
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        out_shape.remove(axis.index());
        Array::from_vec(IxDyn::from(&out_shape[..]), out)
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
    T: MeanAcc,
    D: Dimension,
{
    /// Arithmetic mean of all elements. Returns `None` for an empty array.
    ///
    /// The result type is the NumPy mean accumulator [`MeanAcc::Mean`]:
    /// bool / integer inputs average in (and return) `f64`, while `f32`/`f64`/
    /// complex keep their own dtype. This matches numpy, which casts bool /
    /// unsigned / signed int to `float64` before averaging
    /// (`numpy/_core/_methods.py:124-127`), so `Array::<i32, _>::mean()` is
    /// `Some(f64)` and `Array::<bool, _>::mean()` is `Some(f64)` (e.g.
    /// `0.666…`), matching `np.mean`.
    pub fn mean(&self) -> Option<<T as MeanAcc>::Mean> {
        let n = self.size();
        if n == 0 {
            return None;
        }
        let sum = self
            .iter()
            .copied()
            .fold(<T as MeanAcc>::Mean::zero(), |acc, x| acc + x.widen_mean());
        Some(sum / <T as MeanAcc>::count(n))
    }

    /// Mean along an axis. Element type is the promoted [`MeanAcc::Mean`]
    /// (bool / integer lanes average in `f64`; `f32`/`f64` stay themselves).
    pub fn mean_axis(&self, axis: Axis) -> FerrayResult<Array<<T as MeanAcc>::Mean, IxDyn>>
    where
        <T as MeanAcc>::Mean: ReduceAcc<Acc = <T as MeanAcc>::Mean>,
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
        // Widen each element into the mean accumulator, then sum along the axis
        // and divide by the lane length.
        let widened = self.map_to::<<T as MeanAcc>::Mean>(MeanAcc::widen_mean);
        let sums = widened.sum_axis(axis)?;
        let n_t = <T as MeanAcc>::count(n);
        Ok(sums.mapv(|x| x / n_t))
    }
}

impl<T, D> Array<T, D>
where
    T: Element + Float + MeanAcc<Mean = T>,
    D: Dimension,
{
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

    /// Flat index of the maximum element. See [`Array::argmax`].
    pub fn argmax(&self) -> Option<i64> {
        arg_reduce(self.iter().copied(), false)
    }

    /// Flat index of the minimum element. See [`Array::argmin`].
    pub fn argmin(&self) -> Option<i64> {
        arg_reduce(self.iter().copied(), true)
    }
}

impl<T, D> ArrayView<'_, T, D>
where
    T: MeanAcc,
    D: Dimension,
{
    /// Mean. See [`Array::mean`] — returns the promoted [`MeanAcc::Mean`].
    pub fn mean(&self) -> Option<<T as MeanAcc>::Mean> {
        let n = self.size();
        if n == 0 {
            return None;
        }
        let sum = self
            .iter()
            .copied()
            .fold(<T as MeanAcc>::Mean::zero(), |acc, x| acc + x.widen_mean());
        Some(sum / <T as MeanAcc>::count(n))
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
