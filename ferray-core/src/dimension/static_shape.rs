// ferray-core: Compile-time static shape types (Phase 4)
//
// Provides Shape1<N> through Shape6<A,B,C,D,E,F> for opt-in
// compile-time shape verification. Each type implements Dimension
// and can interoperate with the dynamic runtime types (Ix1..Ix6, IxDyn).
//
// Feature-gated behind `const_shapes`.

use std::fmt;

use crate::array::owned::Array;
use crate::dimension::{Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

// ===========================================================================
// StaticSize trait — compile-time total element count
// ===========================================================================

/// Trait for dimension types that have a compile-time-known total size.
///
/// Used with [`static_reshape`](Array::static_reshape) to enable
/// compile-time verification that a reshape preserves element count.
pub trait StaticSize {
    /// The total number of elements (product of all dimensions).
    const SIZE: usize;
}

// ===========================================================================
// Shape1<N>
// ===========================================================================

/// A 1-dimensional shape with compile-time extent `N`.
///
/// Implements [`Dimension`] so it can be used as the `D` parameter
/// of [`Array<T, D>`](crate::Array).
///
/// # Examples
/// ```
/// # use ferray_core::dimension::static_shape::Shape1;
/// # use ferray_core::dimension::Dimension;
/// let s = Shape1::<5>::new();
/// assert_eq!(s.as_slice(), &[5]);
/// assert_eq!(s.size(), 5);
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape1<const N: usize> {
    shape: [usize; 1],
}

impl<const N: usize> Shape1<N> {
    /// Create a new `Shape1` from its const generic parameter.
    #[inline]
    pub const fn new() -> Self {
        Self { shape: [N] }
    }
}

impl<const N: usize> Default for Shape1<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> fmt::Debug for Shape1<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape1<{}>", N)
    }
}

impl<const N: usize> StaticSize for Shape1<N> {
    const SIZE: usize = N;
}

impl<const N: usize> Dimension for Shape1<N> {
    const NDIM: Option<usize> = Some(1);
    type NdarrayDim = ndarray::Ix1;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(self.shape)
    }

    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let _ = dim; // The shape is known at compile time
        Self::new()
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape == [N] {
            Some(Self::new())
        } else {
            None
        }
    }
}

// ===========================================================================
// Shape2<M, N>
// ===========================================================================

/// A 2-dimensional shape with compile-time extents `M` and `N`.
///
/// Implements [`Dimension`] so it can be used as the `D` parameter
/// of [`Array<T, D>`](crate::Array).
///
/// # Examples
/// ```
/// # use ferray_core::dimension::static_shape::Shape2;
/// # use ferray_core::dimension::Dimension;
/// let s = Shape2::<3, 4>::new();
/// assert_eq!(s.as_slice(), &[3, 4]);
/// assert_eq!(s.size(), 12);
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape2<const M: usize, const N: usize> {
    shape: [usize; 2],
}

impl<const M: usize, const N: usize> Shape2<M, N> {
    /// Create a new `Shape2` from its const generic parameters.
    #[inline]
    pub const fn new() -> Self {
        Self { shape: [M, N] }
    }
}

impl<const M: usize, const N: usize> Default for Shape2<M, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize, const N: usize> fmt::Debug for Shape2<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape2<{}, {}>", M, N)
    }
}

impl<const M: usize, const N: usize> StaticSize for Shape2<M, N> {
    const SIZE: usize = M * N;
}

impl<const M: usize, const N: usize> Dimension for Shape2<M, N> {
    const NDIM: Option<usize> = Some(2);
    type NdarrayDim = ndarray::Ix2;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(self.shape)
    }

    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let _ = dim;
        Self::new()
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape == [M, N] {
            Some(Self::new())
        } else {
            None
        }
    }
}

// ===========================================================================
// Shape3<A, B, C>
// ===========================================================================

/// A 3-dimensional shape with compile-time extents `A`, `B`, and `C`.
///
/// Implements [`Dimension`] so it can be used as the `D` parameter
/// of [`Array<T, D>`](crate::Array).
///
/// # Examples
/// ```
/// # use ferray_core::dimension::static_shape::Shape3;
/// # use ferray_core::dimension::Dimension;
/// let s = Shape3::<2, 3, 4>::new();
/// assert_eq!(s.as_slice(), &[2, 3, 4]);
/// assert_eq!(s.size(), 24);
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape3<const A: usize, const B: usize, const C: usize> {
    shape: [usize; 3],
}

impl<const A: usize, const B: usize, const C: usize> Shape3<A, B, C> {
    /// Create a new `Shape3` from its const generic parameters.
    #[inline]
    pub const fn new() -> Self {
        Self { shape: [A, B, C] }
    }
}

impl<const A: usize, const B: usize, const C: usize> Default for Shape3<A, B, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const A: usize, const B: usize, const C: usize> fmt::Debug for Shape3<A, B, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape3<{}, {}, {}>", A, B, C)
    }
}

impl<const A: usize, const B: usize, const C: usize> StaticSize for Shape3<A, B, C> {
    const SIZE: usize = A * B * C;
}

impl<const A: usize, const B: usize, const C: usize> Dimension for Shape3<A, B, C> {
    const NDIM: Option<usize> = Some(3);
    type NdarrayDim = ndarray::Ix3;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(self.shape)
    }

    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let _ = dim;
        Self::new()
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape == [A, B, C] {
            Some(Self::new())
        } else {
            None
        }
    }
}

// ===========================================================================
// Shape4<A, B, C, D>
// ===========================================================================

/// A 4-dimensional shape with compile-time extents `A`, `B`, `C`, and `D_`.
///
/// Implements [`Dimension`] so it can be used as the `D` parameter
/// of [`Array<T, D>`](crate::Array).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape4<const A: usize, const B: usize, const C: usize, const D: usize> {
    shape: [usize; 4],
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> Shape4<A, B, C, D> {
    /// Create a new `Shape4` from its const generic parameters.
    #[inline]
    pub const fn new() -> Self {
        Self {
            shape: [A, B, C, D],
        }
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> Default
    for Shape4<A, B, C, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> fmt::Debug
    for Shape4<A, B, C, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape4<{}, {}, {}, {}>", A, B, C, D)
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> StaticSize
    for Shape4<A, B, C, D>
{
    const SIZE: usize = A * B * C * D;
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> Dimension
    for Shape4<A, B, C, D>
{
    const NDIM: Option<usize> = Some(4);
    type NdarrayDim = ndarray::Ix4;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(self.shape)
    }

    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let _ = dim;
        Self::new()
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape == [A, B, C, D] {
            Some(Self::new())
        } else {
            None
        }
    }
}

// ===========================================================================
// Shape5<A, B, C, D, E>
// ===========================================================================

/// A 5-dimensional shape with compile-time extents.
///
/// Implements [`Dimension`] so it can be used as the `D` parameter
/// of [`Array<T, D>`](crate::Array).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape5<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> {
    shape: [usize; 5],
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    Shape5<A, B, C, D, E>
{
    /// Create a new `Shape5` from its const generic parameters.
    #[inline]
    pub const fn new() -> Self {
        Self {
            shape: [A, B, C, D, E],
        }
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> Default
    for Shape5<A, B, C, D, E>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> fmt::Debug
    for Shape5<A, B, C, D, E>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape5<{}, {}, {}, {}, {}>", A, B, C, D, E)
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> StaticSize
    for Shape5<A, B, C, D, E>
{
    const SIZE: usize = A * B * C * D * E;
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize> Dimension
    for Shape5<A, B, C, D, E>
{
    const NDIM: Option<usize> = Some(5);
    type NdarrayDim = ndarray::Ix5;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(self.shape)
    }

    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let _ = dim;
        Self::new()
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape == [A, B, C, D, E] {
            Some(Self::new())
        } else {
            None
        }
    }
}

// ===========================================================================
// Shape6<A, B, C, D, E, F>
// ===========================================================================

/// A 6-dimensional shape with compile-time extents.
///
/// Implements [`Dimension`] so it can be used as the `D` parameter
/// of [`Array<T, D>`](crate::Array).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape6<
    const A: usize,
    const B: usize,
    const C: usize,
    const D: usize,
    const E: usize,
    const F: usize,
> {
    shape: [usize; 6],
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    Shape6<A, B, C, D, E, F>
{
    /// Create a new `Shape6` from its const generic parameters.
    #[inline]
    pub const fn new() -> Self {
        Self {
            shape: [A, B, C, D, E, F],
        }
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    Default for Shape6<A, B, C, D, E, F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    fmt::Debug for Shape6<A, B, C, D, E, F>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape6<{}, {}, {}, {}, {}, {}>", A, B, C, D, E, F)
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    StaticSize for Shape6<A, B, C, D, E, F>
{
    const SIZE: usize = A * B * C * D * E * F;
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    Dimension for Shape6<A, B, C, D, E, F>
{
    const NDIM: Option<usize> = Some(6);
    type NdarrayDim = ndarray::Ix6;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(self.shape)
    }

    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let _ = dim;
        Self::new()
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape == [A, B, C, D, E, F] {
            Some(Self::new())
        } else {
            None
        }
    }
}

// ===========================================================================
// From conversions: Static -> Dynamic (REQ-9)
// ===========================================================================

impl<const N: usize> From<Shape1<N>> for Ix1 {
    /// Convert a static `Shape1<N>` to a dynamic `Ix1`.
    fn from(_s: Shape1<N>) -> Self {
        Ix1::new([N])
    }
}

impl<const M: usize, const N: usize> From<Shape2<M, N>> for Ix2 {
    /// Convert a static `Shape2<M, N>` to a dynamic `Ix2`.
    fn from(_s: Shape2<M, N>) -> Self {
        Ix2::new([M, N])
    }
}

impl<const A: usize, const B: usize, const C: usize> From<Shape3<A, B, C>> for Ix3 {
    /// Convert a static `Shape3<A, B, C>` to a dynamic `Ix3`.
    fn from(_s: Shape3<A, B, C>) -> Self {
        Ix3::new([A, B, C])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> From<Shape4<A, B, C, D>>
    for Ix4
{
    /// Convert a static `Shape4` to a dynamic `Ix4`.
    fn from(_s: Shape4<A, B, C, D>) -> Self {
        Ix4::new([A, B, C, D])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    From<Shape5<A, B, C, D, E>> for Ix5
{
    /// Convert a static `Shape5` to a dynamic `Ix5`.
    fn from(_s: Shape5<A, B, C, D, E>) -> Self {
        Ix5::new([A, B, C, D, E])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    From<Shape6<A, B, C, D, E, F>> for Ix6
{
    /// Convert a static `Shape6` to a dynamic `Ix6`.
    fn from(_s: Shape6<A, B, C, D, E, F>) -> Self {
        Ix6::new([A, B, C, D, E, F])
    }
}

// From static shapes to IxDyn

impl<const N: usize> From<Shape1<N>> for IxDyn {
    /// Convert a static `Shape1<N>` to a dynamic `IxDyn`.
    fn from(_s: Shape1<N>) -> Self {
        IxDyn::new(&[N])
    }
}

impl<const M: usize, const N: usize> From<Shape2<M, N>> for IxDyn {
    /// Convert a static `Shape2<M, N>` to a dynamic `IxDyn`.
    fn from(_s: Shape2<M, N>) -> Self {
        IxDyn::new(&[M, N])
    }
}

impl<const A: usize, const B: usize, const C: usize> From<Shape3<A, B, C>> for IxDyn {
    /// Convert a static `Shape3<A, B, C>` to a dynamic `IxDyn`.
    fn from(_s: Shape3<A, B, C>) -> Self {
        IxDyn::new(&[A, B, C])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> From<Shape4<A, B, C, D>>
    for IxDyn
{
    /// Convert a static `Shape4` to a dynamic `IxDyn`.
    fn from(_s: Shape4<A, B, C, D>) -> Self {
        IxDyn::new(&[A, B, C, D])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    From<Shape5<A, B, C, D, E>> for IxDyn
{
    /// Convert a static `Shape5` to a dynamic `IxDyn`.
    fn from(_s: Shape5<A, B, C, D, E>) -> Self {
        IxDyn::new(&[A, B, C, D, E])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    From<Shape6<A, B, C, D, E, F>> for IxDyn
{
    /// Convert a static `Shape6` to a dynamic `IxDyn`.
    fn from(_s: Shape6<A, B, C, D, E, F>) -> Self {
        IxDyn::new(&[A, B, C, D, E, F])
    }
}

// ===========================================================================
// Array conversion methods: static-shaped -> dynamic-shaped
// ===========================================================================

// Note: `into_dyn` for static-shape arrays is provided by the generic
// `Array::into_dyn` impl in `array/methods.rs` — it already works for any
// `D: Dimension`, including `Shape1..Shape6`. We only need the narrower
// `into_dynamic_ix` helpers here, which produce `Ix1`/`Ix2`/`Ix3` rather
// than fully dynamic `IxDyn`.

impl<T: Element, const N: usize> Array<T, Shape1<N>> {
    /// Convert this statically-shaped array into a dynamically-shaped `Array<T, Ix1>`.
    ///
    /// This is a zero-cost operation that reinterprets the dimension type.
    pub fn into_dynamic_ix(self) -> Array<T, Ix1> {
        Array::from_ndarray(self.inner)
    }
}

impl<T: Element, const M: usize, const N: usize> Array<T, Shape2<M, N>> {
    /// Convert this statically-shaped array into a dynamically-shaped `Array<T, Ix2>`.
    ///
    /// This is a zero-cost operation that reinterprets the dimension type.
    pub fn into_dynamic_ix(self) -> Array<T, Ix2> {
        Array::from_ndarray(self.inner)
    }
}

impl<T: Element, const A: usize, const B: usize, const C: usize> Array<T, Shape3<A, B, C>> {
    /// Convert this statically-shaped array into a dynamically-shaped `Array<T, Ix3>`.
    pub fn into_dynamic_ix(self) -> Array<T, Ix3> {
        Array::from_ndarray(self.inner)
    }
}

// ===========================================================================
// StaticMatMul (REQ-10): Compile-time matrix multiply shape checking
// ===========================================================================

/// Trait for compile-time-verified matrix multiplication.
///
/// The shared inner dimension `K` is enforced at the type level:
/// `Array<T, Shape2<M, K>> * Array<T, Shape2<K, N>>` produces
/// `Array<T, Shape2<M, N>>`. If the inner dimensions don't match
/// the code will not compile.
///
/// # Examples
/// ```
/// # use ferray_core::dimension::static_shape::{Shape2, StaticMatMul};
/// # use ferray_core::array::owned::Array;
/// let a = Array::<f64, Shape2<2, 3>>::from_vec(
///     Shape2::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ).unwrap();
/// let b = Array::<f64, Shape2<3, 2>>::from_vec(
///     Shape2::new(), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
/// ).unwrap();
/// let c = a.static_matmul(b).unwrap();
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
///
/// Mismatched inner dimensions produce a compile error:
///
/// ```compile_fail
/// use ferray_core::dimension::static_shape::{Shape2, StaticMatMul};
/// use ferray_core::array::owned::Array;
///
/// let a = Array::<f64, Shape2<3, 4>>::ones(Shape2::new()).unwrap();
/// let b = Array::<f64, Shape2<5, 2>>::ones(Shape2::new()).unwrap();
/// // K=4 vs K=5 — should not compile
/// let _c = a.static_matmul(b);
/// ```
pub trait StaticMatMul<Rhs> {
    /// The output array type with a statically-known shape.
    type Output;

    /// Perform matrix multiplication with compile-time shape verification.
    ///
    /// # Errors
    /// Returns `FerrayError` if the underlying computation fails (e.g., allocation).
    fn static_matmul(self, rhs: Rhs) -> FerrayResult<Self::Output>;
}

impl<T, const M: usize, const K: usize, const N: usize> StaticMatMul<Array<T, Shape2<K, N>>>
    for Array<T, Shape2<M, K>>
where
    T: Element + num_traits::Float,
{
    type Output = Array<T, Shape2<M, N>>;

    fn static_matmul(self, rhs: Array<T, Shape2<K, N>>) -> FerrayResult<Self::Output> {
        // Delegate to a straightforward matmul implementation.
        // Both arrays share ndarray::Ix2 as NdarrayDim.
        let a = &self.inner;
        let b = &rhs.inner;
        let c = a.dot(b);
        Ok(Array::from_ndarray(c))
    }
}

// ===========================================================================
// StaticBroadcast (REQ-10a): Compile-time broadcast shape resolution
// ===========================================================================

/// Trait for compile-time broadcast shape resolution.
///
/// When two static shapes are broadcast-compatible, this trait's
/// associated `Output` type gives the result shape at compile time.
///
/// Only the most common broadcast patterns are covered:
/// - Same shape -> same shape
/// - Dimension of size 1 broadcasts with any size
///
/// Incompatible shapes are rejected at compile time — there is simply
/// no `StaticBroadcast` impl connecting them:
///
/// ```compile_fail
/// use ferray_core::dimension::static_shape::{Shape2, StaticBroadcast};
///
/// // Shape2<3, 4> and Shape2<5, 6> are not broadcast-compatible
/// fn check<A: StaticBroadcast<B>, B>() {}
/// check::<Shape2<3, 4>, Shape2<5, 6>>();
/// ```
pub trait StaticBroadcast<Rhs> {
    /// The resulting shape after broadcasting.
    type Output;
}

// ---------------------------------------------------------------------------
// Same shape -> same shape
// These are the safest impls: no overlap is possible since both sides
// share exactly the same const generic parameters.
// ---------------------------------------------------------------------------

// Shape1<N> + Shape1<N> -> Shape1<N>
impl<const N: usize> StaticBroadcast<Shape1<N>> for Shape1<N> {
    type Output = Shape1<N>;
}

// Shape2<M, N> + Shape2<M, N> -> Shape2<M, N>
impl<const M: usize, const N: usize> StaticBroadcast<Shape2<M, N>> for Shape2<M, N> {
    type Output = Shape2<M, N>;
}

// Shape3<A, B, C> + Shape3<A, B, C> -> Shape3<A, B, C>
impl<const A: usize, const B: usize, const C: usize> StaticBroadcast<Shape3<A, B, C>>
    for Shape3<A, B, C>
{
    type Output = Shape3<A, B, C>;
}

// ---------------------------------------------------------------------------
// 1D broadcasts to 2D
// These do not conflict because Shape1 and Shape2 are distinct types.
// ---------------------------------------------------------------------------

// Shape1<N> + Shape2<M, N> -> Shape2<M, N>
impl<const M: usize, const N: usize> StaticBroadcast<Shape2<M, N>> for Shape1<N> {
    type Output = Shape2<M, N>;
}

// Shape2<M, N> + Shape1<N> -> Shape2<M, N>
impl<const M: usize, const N: usize> StaticBroadcast<Shape1<N>> for Shape2<M, N> {
    type Output = Shape2<M, N>;
}

// ---------------------------------------------------------------------------
// Concrete broadcast impls for the most common patterns with literal 1.
// On stable Rust, generic impls like Shape2<M, 1> + Shape2<1, N> overlap
// with Shape2<M, N> + Shape2<M, N> at the point M=1, N=1.
// We provide concrete (non-generic) macro-generated impls instead.
// ---------------------------------------------------------------------------

/// Inner helper: generate impls for one fixed `$m` against all `$n` values.
macro_rules! impl_broadcast_cross_inner {
    ($m:expr, [$($n:expr),+]) => {
        $(
            impl StaticBroadcast<Shape2<1, $n>> for Shape2<$m, 1> {
                type Output = Shape2<$m, $n>;
            }
            impl StaticBroadcast<Shape2<$m, 1>> for Shape2<1, $n> {
                type Output = Shape2<$m, $n>;
            }
        )+
    };
}

/// Outer driver: iterate over `$m` values, calling inner for each.
macro_rules! impl_broadcast_cross_2d {
    ([$($m:expr),+], $ns:tt) => {
        $(
            impl_broadcast_cross_inner!($m, $ns);
        )+
    };
}

// Cover common dimension sizes. All values >= 2 to avoid overlap with
// the same-shape impl at Shape2<1, 1>.
impl_broadcast_cross_2d!(
    [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024
    ],
    [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024
    ]
);

// ===========================================================================
// Static Reshape (REQ-10b): Compile-time element count verification
// ===========================================================================

/// Compile-time assertion helper for static reshape.
///
/// This struct exists solely to be used with the [`IsTrue`] trait
/// to enforce that two static sizes are equal at compile time.
pub struct Assert<const COND: bool>;

/// Marker trait that is only implemented for `Assert<true>`.
///
/// Used with [`Assert`] to create compile-time assertions:
/// ```rust,ignore
/// where Assert<{ ShapeA::SIZE == ShapeB::SIZE }>: IsTrue
/// ```
pub trait IsTrue {}

impl IsTrue for Assert<true> {}

// We need a helper to construct the target ndarray dimension from a static shape.
// Since all static shapes have `new()`, we use a trait-based approach so that
// `static_reshape` and `static_reshape_array` can create the target dimension
// without knowing which concrete ShapeN type is in use.

/// Helper trait for creating a default ndarray dimension from a static shape.
///
/// This is used internally by [`static_reshape_array`] and
/// [`Array::static_reshape`] to construct the target dimension.
pub trait DefaultNdarrayDim: Dimension {
    /// Create the default ndarray dimension for this static shape.
    fn default_ndarray_dim() -> Self::NdarrayDim;
}

impl<T: Element, D: Dimension + StaticSize> Array<T, D> {
    /// Reshape this array to a new static shape, verifying that
    /// the total element counts match.
    ///
    /// Both the source and target shapes must implement [`StaticSize`].
    /// Because both sizes are const-evaluable, the optimizer eliminates
    /// the runtime check when the sizes are equal.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if the element counts differ.
    ///
    /// # Examples
    /// ```
    /// # use ferray_core::dimension::static_shape::{Shape1, Shape2};
    /// # use ferray_core::array::owned::Array;
    /// let a = Array::<f64, Shape1<12>>::from_vec(
    ///     Shape1::new(), (0..12).map(|i| i as f64).collect(),
    /// ).unwrap();
    /// let b: Array<f64, Shape2<3, 4>> = a.static_reshape().unwrap();
    /// assert_eq!(b.shape(), &[3, 4]);
    /// ```
    pub fn static_reshape<NewD>(self) -> FerrayResult<Array<T, NewD>>
    where
        NewD: Dimension + StaticSize + DefaultNdarrayDim,
    {
        static_reshape_array(self)
    }
}

impl<const N: usize> DefaultNdarrayDim for Shape1<N> {
    fn default_ndarray_dim() -> ndarray::Ix1 {
        ndarray::Dim([N])
    }
}

impl<const M: usize, const N: usize> DefaultNdarrayDim for Shape2<M, N> {
    fn default_ndarray_dim() -> ndarray::Ix2 {
        ndarray::Dim([M, N])
    }
}

impl<const A: usize, const B: usize, const C: usize> DefaultNdarrayDim for Shape3<A, B, C> {
    fn default_ndarray_dim() -> ndarray::Ix3 {
        ndarray::Dim([A, B, C])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> DefaultNdarrayDim
    for Shape4<A, B, C, D>
{
    fn default_ndarray_dim() -> ndarray::Ix4 {
        ndarray::Dim([A, B, C, D])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    DefaultNdarrayDim for Shape5<A, B, C, D, E>
{
    fn default_ndarray_dim() -> ndarray::Ix5 {
        ndarray::Dim([A, B, C, D, E])
    }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize, const F: usize>
    DefaultNdarrayDim for Shape6<A, B, C, D, E, F>
{
    fn default_ndarray_dim() -> ndarray::Ix6 {
        ndarray::Dim([A, B, C, D, E, F])
    }
}

// ===========================================================================
// Cleaner static_reshape using DefaultNdarrayDim
// ===========================================================================

// Override the previous static_reshape with a cleaner implementation.
// We'll actually use a free function to avoid the orphan/duplicate impl issues.

/// Reshape a statically-shaped array to a new static shape.
///
/// Both the source and target shapes must implement [`StaticSize`], and
/// the element counts must match (verified at runtime but const-evaluable).
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if sizes differ.
pub fn static_reshape_array<T, OldD, NewD>(arr: Array<T, OldD>) -> FerrayResult<Array<T, NewD>>
where
    T: Element,
    OldD: Dimension + StaticSize,
    NewD: Dimension + StaticSize + DefaultNdarrayDim,
{
    if OldD::SIZE != NewD::SIZE {
        return Err(FerrayError::shape_mismatch(format!(
            "cannot reshape array of size {} into shape with size {}",
            OldD::SIZE,
            NewD::SIZE,
        )));
    }
    let data: Vec<T> = arr.inner.into_iter().collect();
    let new_dim = NewD::from_ndarray_dim(&NewD::default_ndarray_dim());
    Array::from_vec(new_dim, data)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Basic shape construction and Dimension trait
    // -----------------------------------------------------------------------

    #[test]
    fn shape1_basics() {
        let s = Shape1::<5>::new();
        assert_eq!(s.as_slice(), &[5]);
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.size(), 5);
        assert_eq!(Shape1::<5>::NDIM, Some(1));
    }

    #[test]
    fn shape2_basics() {
        let s = Shape2::<3, 4>::new();
        assert_eq!(s.as_slice(), &[3, 4]);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.size(), 12);
        assert_eq!(Shape2::<3, 4>::NDIM, Some(2));
    }

    #[test]
    fn shape3_basics() {
        let s = Shape3::<2, 3, 4>::new();
        assert_eq!(s.as_slice(), &[2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.size(), 24);
    }

    #[test]
    fn shape4_basics() {
        let s = Shape4::<2, 3, 4, 5>::new();
        assert_eq!(s.as_slice(), &[2, 3, 4, 5]);
        assert_eq!(s.ndim(), 4);
        assert_eq!(s.size(), 120);
    }

    #[test]
    fn shape5_basics() {
        let s = Shape5::<2, 3, 4, 5, 6>::new();
        assert_eq!(s.as_slice(), &[2, 3, 4, 5, 6]);
        assert_eq!(s.ndim(), 5);
        assert_eq!(s.size(), 720);
    }

    #[test]
    fn shape6_basics() {
        let s = Shape6::<2, 3, 4, 5, 6, 7>::new();
        assert_eq!(s.as_slice(), &[2, 3, 4, 5, 6, 7]);
        assert_eq!(s.ndim(), 6);
        assert_eq!(s.size(), 5040);
    }

    // -----------------------------------------------------------------------
    // StaticSize
    // -----------------------------------------------------------------------

    #[test]
    fn static_size_values() {
        assert_eq!(<Shape1<5> as StaticSize>::SIZE, 5);
        assert_eq!(<Shape2<3, 4> as StaticSize>::SIZE, 12);
        assert_eq!(<Shape3<2, 3, 4> as StaticSize>::SIZE, 24);
        assert_eq!(<Shape4<2, 3, 4, 5> as StaticSize>::SIZE, 120);
        assert_eq!(<Shape5<2, 3, 4, 5, 6> as StaticSize>::SIZE, 720);
        assert_eq!(<Shape6<2, 3, 4, 5, 6, 7> as StaticSize>::SIZE, 5040);
    }

    // -----------------------------------------------------------------------
    // ndarray roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn shape1_ndarray_roundtrip() {
        let s = Shape1::<5>::new();
        let nd = s.to_ndarray_dim();
        let s2 = Shape1::<5>::from_ndarray_dim(&nd);
        assert_eq!(s, s2);
    }

    #[test]
    fn shape2_ndarray_roundtrip() {
        let s = Shape2::<3, 4>::new();
        let nd = s.to_ndarray_dim();
        let s2 = Shape2::<3, 4>::from_ndarray_dim(&nd);
        assert_eq!(s, s2);
    }

    // -----------------------------------------------------------------------
    // From conversions: static -> dynamic
    // -----------------------------------------------------------------------

    #[test]
    fn shape1_to_ix1() {
        let ix: Ix1 = Shape1::<5>::new().into();
        assert_eq!(ix.as_slice(), &[5]);
    }

    #[test]
    fn shape2_to_ix2() {
        let ix: Ix2 = Shape2::<3, 4>::new().into();
        assert_eq!(ix.as_slice(), &[3, 4]);
    }

    #[test]
    fn shape3_to_ix3() {
        let ix: Ix3 = Shape3::<2, 3, 4>::new().into();
        assert_eq!(ix.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn shape4_to_ix4() {
        let ix: Ix4 = Shape4::<2, 3, 4, 5>::new().into();
        assert_eq!(ix.as_slice(), &[2, 3, 4, 5]);
    }

    #[test]
    fn shape5_to_ix5() {
        let ix: Ix5 = Shape5::<2, 3, 4, 5, 6>::new().into();
        assert_eq!(ix.as_slice(), &[2, 3, 4, 5, 6]);
    }

    #[test]
    fn shape6_to_ix6() {
        let ix: Ix6 = Shape6::<2, 3, 4, 5, 6, 7>::new().into();
        assert_eq!(ix.as_slice(), &[2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn shape1_to_ixdyn() {
        let ix: IxDyn = Shape1::<5>::new().into();
        assert_eq!(ix.as_slice(), &[5]);
    }

    #[test]
    fn shape2_to_ixdyn() {
        let ix: IxDyn = Shape2::<3, 4>::new().into();
        assert_eq!(ix.as_slice(), &[3, 4]);
    }

    #[test]
    fn shape3_to_ixdyn() {
        let ix: IxDyn = Shape3::<2, 3, 4>::new().into();
        assert_eq!(ix.as_slice(), &[2, 3, 4]);
    }

    // -----------------------------------------------------------------------
    // Array creation with static shapes
    // -----------------------------------------------------------------------

    #[test]
    fn create_array_with_shape1() {
        let arr =
            Array::<f64, Shape1<4>>::from_vec(Shape1::new(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(arr.shape(), &[4]);
        assert_eq!(arr.size(), 4);
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn create_array_with_shape2() {
        let arr =
            Array::<f64, Shape2<2, 3>>::from_vec(Shape2::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.size(), 6);
    }

    #[test]
    fn create_zeros_with_shape2() {
        let arr = Array::<f64, Shape2<3, 4>>::zeros(Shape2::new()).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.size(), 12);
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn create_ones_with_shape3() {
        let arr = Array::<f64, Shape3<2, 3, 4>>::ones(Shape3::new()).unwrap();
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert_eq!(arr.size(), 24);
        assert!(arr.iter().all(|&x| x == 1.0));
    }

    // -----------------------------------------------------------------------
    // Static to dynamic array conversion
    // -----------------------------------------------------------------------

    #[test]
    fn into_dynamic_ix_shape1() {
        let arr = Array::<f64, Shape1<3>>::from_vec(Shape1::new(), vec![1.0, 2.0, 3.0]).unwrap();
        let dyn_arr: Array<f64, Ix1> = arr.into_dynamic_ix();
        assert_eq!(dyn_arr.shape(), &[3]);
        assert_eq!(dyn_arr.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn into_dynamic_ix_shape2() {
        let arr =
            Array::<f64, Shape2<2, 3>>::from_vec(Shape2::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let dyn_arr: Array<f64, Ix2> = arr.into_dynamic_ix();
        assert_eq!(dyn_arr.shape(), &[2, 3]);
    }

    #[test]
    fn into_dyn_shape2() {
        let arr =
            Array::<f64, Shape2<2, 3>>::from_vec(Shape2::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let dyn_arr: Array<f64, IxDyn> = arr.into_dyn();
        assert_eq!(dyn_arr.shape(), &[2, 3]);
        assert_eq!(dyn_arr.ndim(), 2);
    }

    // -----------------------------------------------------------------------
    // StaticMatMul
    // -----------------------------------------------------------------------

    #[test]
    fn static_matmul_2x3_times_3x2() {
        let a =
            Array::<f64, Shape2<2, 3>>::from_vec(Shape2::new(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, Shape2<3, 2>>::from_vec(
            Shape2::new(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c: Array<f64, Shape2<2, 2>> = a.static_matmul(b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[58, 64], [139, 154]]
        let data = c.as_slice().unwrap();
        assert_eq!(data, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn static_matmul_3x4_times_4x5() {
        let a = Array::<f64, Shape2<3, 4>>::ones(Shape2::new()).unwrap();
        let b = Array::<f64, Shape2<4, 5>>::ones(Shape2::new()).unwrap();
        let c: Array<f64, Shape2<3, 5>> = a.static_matmul(b).unwrap();
        assert_eq!(c.shape(), &[3, 5]);
        // Each element should be 4.0 (dot product of ones vectors of length 4)
        assert!(c.iter().all(|&x| (x - 4.0).abs() < 1e-10));
    }

    #[test]
    fn static_matmul_1x1() {
        let a = Array::<f64, Shape2<1, 1>>::from_vec(Shape2::new(), vec![3.0]).unwrap();
        let b = Array::<f64, Shape2<1, 1>>::from_vec(Shape2::new(), vec![5.0]).unwrap();
        let c: Array<f64, Shape2<1, 1>> = a.static_matmul(b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[15.0]);
    }

    #[test]
    fn static_matmul_f32() {
        let a = Array::<f32, Shape2<2, 2>>::from_vec(Shape2::new(), vec![1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let b = Array::<f32, Shape2<2, 2>>::from_vec(Shape2::new(), vec![5.0f32, 6.0, 7.0, 8.0])
            .unwrap();
        let c: Array<f32, Shape2<2, 2>> = a.static_matmul(b).unwrap();
        let data = c.as_slice().unwrap();
        assert_eq!(data, &[19.0f32, 22.0, 43.0, 50.0]);
    }

    // -----------------------------------------------------------------------
    // StaticBroadcast (type-level tests)
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_same_shape() {
        // Verify the type resolves
        fn assert_broadcast<A, B>()
        where
            A: StaticBroadcast<B>,
        {
        }
        assert_broadcast::<Shape1<5>, Shape1<5>>();
        assert_broadcast::<Shape2<3, 4>, Shape2<3, 4>>();
        assert_broadcast::<Shape3<2, 3, 4>, Shape3<2, 3, 4>>();
    }

    #[test]
    fn broadcast_shape2_with_ones() {
        fn check_output<A, B>()
        where
            A: StaticBroadcast<B>,
        {
        }

        // Shape2<3, 1> + Shape2<1, 4> -> Shape2<3, 4>
        // Uses the concrete macro-generated impl for sizes 3 and 4
        check_output::<Shape2<3, 1>, Shape2<1, 4>>();

        // Shape2<1, 4> + Shape2<3, 1> -> Shape2<3, 4>
        check_output::<Shape2<1, 4>, Shape2<3, 1>>();

        // Shape2<1, 4> + Shape2<1, 4> -> Shape2<1, 4> (same shape, generic impl)
        check_output::<Shape2<1, 4>, Shape2<1, 4>>();

        // Shape2<10, 1> + Shape2<1, 16> -> Shape2<10, 16>
        check_output::<Shape2<10, 1>, Shape2<1, 16>>();
    }

    #[test]
    fn broadcast_1d_to_2d() {
        fn check_output<A, B>()
        where
            A: StaticBroadcast<B>,
        {
        }

        // Shape1<4> + Shape2<3, 4> -> Shape2<3, 4>
        check_output::<Shape1<4>, Shape2<3, 4>>();

        // Shape2<3, 4> + Shape1<4> -> Shape2<3, 4>
        check_output::<Shape2<3, 4>, Shape1<4>>();
    }

    // -----------------------------------------------------------------------
    // Static reshape
    // -----------------------------------------------------------------------

    #[test]
    fn static_reshape_1d_to_2d() {
        let a =
            Array::<f64, Shape1<12>>::from_vec(Shape1::new(), (0..12).map(|i| i as f64).collect())
                .unwrap();
        let b: Array<f64, Shape2<3, 4>> = static_reshape_array(a).unwrap();
        assert_eq!(b.shape(), &[3, 4]);
        assert_eq!(b.size(), 12);
        // Data should be preserved in row-major order
        let data = b.as_slice().unwrap();
        for i in 0..12 {
            assert_eq!(data[i], i as f64);
        }
    }

    #[test]
    fn static_reshape_2d_to_1d() {
        let a = Array::<f64, Shape2<3, 4>>::from_vec(
            Shape2::new(),
            (0..12).map(|i| i as f64).collect(),
        )
        .unwrap();
        let b: Array<f64, Shape1<12>> = static_reshape_array(a).unwrap();
        assert_eq!(b.shape(), &[12]);
    }

    #[test]
    fn static_reshape_2d_to_2d() {
        let a = Array::<f64, Shape2<2, 6>>::from_vec(
            Shape2::new(),
            (0..12).map(|i| i as f64).collect(),
        )
        .unwrap();
        let b: Array<f64, Shape2<3, 4>> = static_reshape_array(a).unwrap();
        assert_eq!(b.shape(), &[3, 4]);
    }

    #[test]
    fn static_reshape_to_3d() {
        let a =
            Array::<f64, Shape1<24>>::from_vec(Shape1::new(), (0..24).map(|i| i as f64).collect())
                .unwrap();
        let b: Array<f64, Shape3<2, 3, 4>> = static_reshape_array(a).unwrap();
        assert_eq!(b.shape(), &[2, 3, 4]);
    }

    #[test]
    fn static_reshape_size_mismatch_returns_error() {
        let a =
            Array::<f64, Shape1<12>>::from_vec(Shape1::new(), (0..12).map(|i| i as f64).collect())
                .unwrap();
        // 3*5 = 15 != 12, so this should fail
        let result: FerrayResult<Array<f64, Shape2<3, 5>>> = static_reshape_array(a);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("12"));
        assert!(err_msg.contains("15"));
    }

    // -----------------------------------------------------------------------
    // Clone and equality
    // -----------------------------------------------------------------------

    #[test]
    fn shape_clone_and_eq() {
        let s1 = Shape2::<3, 4>::new();
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    #[test]
    fn shape_debug_format() {
        let s = Shape2::<3, 4>::new();
        let dbg = format!("{:?}", s);
        assert_eq!(dbg, "Shape2<3, 4>");
    }

    // -----------------------------------------------------------------------
    // Default trait
    // -----------------------------------------------------------------------

    #[test]
    fn shape_default() {
        let s = Shape2::<3, 4>::default();
        assert_eq!(s.as_slice(), &[3, 4]);
    }

    // The compile_fail proofs for matmul-K-mismatch and broadcast-shape-mismatch
    // live as doctests on `StaticMatMul` and `StaticBroadcast` themselves
    // (see their trait definitions above).
}
