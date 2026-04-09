// ferray-core: Dimension trait and concrete dimension types
//
// These types mirror ndarray's Ix1..Ix6 and IxDyn but live in ferray-core's
// namespace so that ndarray never appears in the public API.

#[cfg(feature = "std")]
pub mod broadcast;
// static_shape depends on Array, Vec, format!, and the ndarray-gated methods
// on Dimension, so it's only available when std is in scope.
#[cfg(all(feature = "const_shapes", feature = "std"))]
pub mod static_shape;

use core::fmt;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// We need ndarray's Dimension trait in scope for `as_array_view()` etc.
#[cfg(feature = "std")]
use ndarray::Dimension as NdDimension;

/// Trait for types that describe the dimensionality of an array.
///
/// Each dimension type knows its number of axes at the type level
/// (except [`IxDyn`] which carries it at runtime).
pub trait Dimension: Clone + PartialEq + Eq + fmt::Debug + Send + Sync + 'static {
    /// The number of axes, or `None` for dynamic-rank arrays.
    const NDIM: Option<usize>;

    /// The corresponding `ndarray` dimension type (private, not exposed in public API).
    #[doc(hidden)]
    #[cfg(feature = "std")]
    type NdarrayDim: ndarray::Dimension;

    /// Return the shape as a slice.
    fn as_slice(&self) -> &[usize];

    /// Return the shape as a mutable slice.
    fn as_slice_mut(&mut self) -> &mut [usize];

    /// Number of dimensions.
    fn ndim(&self) -> usize {
        self.as_slice().len()
    }

    /// Total number of elements (product of all dimension sizes).
    fn size(&self) -> usize {
        self.as_slice().iter().product()
    }

    /// Convert to the internal ndarray dimension type.
    #[doc(hidden)]
    #[cfg(feature = "std")]
    fn to_ndarray_dim(&self) -> Self::NdarrayDim;

    /// Create from the internal ndarray dimension type.
    #[doc(hidden)]
    #[cfg(feature = "std")]
    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self;

    /// Construct a dimension from a slice of axis lengths.
    ///
    /// Returns `None` if the slice length does not match `Self::NDIM`
    /// for fixed-rank dimensions. Always succeeds for [`IxDyn`].
    ///
    /// This is the inverse of [`Dimension::as_slice`].
    fn from_dim_slice(shape: &[usize]) -> Option<Self>;
}

// ---------------------------------------------------------------------------
// Fixed-rank dimension types
// ---------------------------------------------------------------------------

macro_rules! impl_fixed_dimension {
    ($name:ident, $n:expr, $ndarray_ty:ty) => {
        /// A fixed-rank dimension with
        #[doc = concat!(stringify!($n), " axes.")]
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            shape: [usize; $n],
        }

        impl $name {
            /// Create a new dimension from a fixed-size array.
            #[inline]
            pub fn new(shape: [usize; $n]) -> Self {
                Self { shape }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}", &self.shape[..])
            }
        }

        impl From<[usize; $n]> for $name {
            #[inline]
            fn from(shape: [usize; $n]) -> Self {
                Self::new(shape)
            }
        }

        impl Dimension for $name {
            const NDIM: Option<usize> = Some($n);

            #[cfg(feature = "std")]
            type NdarrayDim = $ndarray_ty;

            #[inline]
            fn as_slice(&self) -> &[usize] {
                &self.shape
            }

            #[inline]
            fn as_slice_mut(&mut self) -> &mut [usize] {
                &mut self.shape
            }

            #[cfg(feature = "std")]
            fn to_ndarray_dim(&self) -> Self::NdarrayDim {
                // ndarray::Dim implements From<[usize; N]> for N=1..6
                ndarray::Dim(self.shape)
            }

            #[cfg(feature = "std")]
            fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
                let view = dim.as_array_view();
                let s = view.as_slice().expect("ndarray dim should be contiguous");
                let mut shape = [0usize; $n];
                shape.copy_from_slice(s);
                Self { shape }
            }

            fn from_dim_slice(shape: &[usize]) -> Option<Self> {
                if shape.len() != $n {
                    return None;
                }
                let mut arr = [0usize; $n];
                arr.copy_from_slice(shape);
                Some(Self { shape: arr })
            }
        }
    };
}

impl_fixed_dimension!(Ix1, 1, ndarray::Ix1);
impl_fixed_dimension!(Ix2, 2, ndarray::Ix2);
impl_fixed_dimension!(Ix3, 3, ndarray::Ix3);
impl_fixed_dimension!(Ix4, 4, ndarray::Ix4);
impl_fixed_dimension!(Ix5, 5, ndarray::Ix5);
impl_fixed_dimension!(Ix6, 6, ndarray::Ix6);

// ---------------------------------------------------------------------------
// Ix0: scalar (0-dimensional)
// ---------------------------------------------------------------------------

/// A zero-dimensional (scalar) dimension.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Ix0;

impl fmt::Debug for Ix0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[]")
    }
}

impl Dimension for Ix0 {
    const NDIM: Option<usize> = Some(0);

    #[cfg(feature = "std")]
    type NdarrayDim = ndarray::Ix0;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &[]
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut []
    }

    #[cfg(feature = "std")]
    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::Dim(())
    }

    #[cfg(feature = "std")]
    fn from_ndarray_dim(_dim: &Self::NdarrayDim) -> Self {
        Ix0
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        if shape.is_empty() { Some(Ix0) } else { None }
    }
}

// ---------------------------------------------------------------------------
// IxDyn: dynamic-rank dimension
// ---------------------------------------------------------------------------

/// A dynamic-rank dimension whose number of axes is determined at runtime.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IxDyn {
    shape: Vec<usize>,
}

impl IxDyn {
    /// Create a new dynamic dimension from a slice.
    pub fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl fmt::Debug for IxDyn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.shape[..])
    }
}

impl From<Vec<usize>> for IxDyn {
    fn from(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl From<&[usize]> for IxDyn {
    fn from(shape: &[usize]) -> Self {
        Self::new(shape)
    }
}

impl Dimension for IxDyn {
    const NDIM: Option<usize> = None;

    #[cfg(feature = "std")]
    type NdarrayDim = ndarray::IxDyn;

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.shape
    }

    #[cfg(feature = "std")]
    fn to_ndarray_dim(&self) -> Self::NdarrayDim {
        ndarray::IxDyn(&self.shape)
    }

    #[cfg(feature = "std")]
    fn from_ndarray_dim(dim: &Self::NdarrayDim) -> Self {
        let view = dim.as_array_view();
        let s = view.as_slice().expect("ndarray IxDyn should be contiguous");
        Self { shape: s.to_vec() }
    }

    fn from_dim_slice(shape: &[usize]) -> Option<Self> {
        Some(Self {
            shape: shape.to_vec(),
        })
    }
}

/// Newtype for axis indices used throughout ferray.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis(pub usize);

impl Axis {
    /// Return the axis index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ix1_basics() {
        let d = Ix1::new([5]);
        assert_eq!(d.ndim(), 1);
        assert_eq!(d.size(), 5);
        assert_eq!(d.as_slice(), &[5]);
    }

    #[test]
    fn ix2_basics() {
        let d = Ix2::new([3, 4]);
        assert_eq!(d.ndim(), 2);
        assert_eq!(d.size(), 12);
    }

    #[test]
    fn ix0_basics() {
        let d = Ix0;
        assert_eq!(d.ndim(), 0);
        assert_eq!(d.size(), 1);
    }

    #[test]
    fn ixdyn_basics() {
        let d = IxDyn::new(&[2, 3, 4]);
        assert_eq!(d.ndim(), 3);
        assert_eq!(d.size(), 24);
    }

    #[test]
    fn roundtrip_ix2_ndarray() {
        let d = Ix2::new([3, 7]);
        let nd = d.to_ndarray_dim();
        let d2 = Ix2::from_ndarray_dim(&nd);
        assert_eq!(d, d2);
    }

    #[test]
    fn roundtrip_ixdyn_ndarray() {
        let d = IxDyn::new(&[2, 5, 3]);
        let nd = d.to_ndarray_dim();
        let d2 = IxDyn::from_ndarray_dim(&nd);
        assert_eq!(d, d2);
    }

    #[test]
    fn axis_index() {
        let a = Axis(2);
        assert_eq!(a.index(), 2);
    }
}
