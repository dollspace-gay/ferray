//! `PyO3` `NumPy` <-> ferray array conversions (feature-gated behind `"python"`).
//!
//! Provides [`AsFerray`] and [`IntoNumPy`] traits for converting between
//! `PyO3` `NumPy` arrays and ferray arrays.
//!
//! # Memory semantics
//!
//! Both directions **copy** the data buffer. The previous docstring
//! claimed zero-copy borrows, but the implementation iterates and
//! clones (`py_arr.iter().cloned().collect()` on the way in,
//! `to_vec_flat()` + `PyArray1::from_vec` on the way out). True
//! zero-copy would require sharing the raw buffer with the Python GC
//! via a pinned refcount handshake — a larger design change that is
//! tracked as a follow-up.
//!
//! - **`NumPy` -> ferray**: [`AsFerray::as_ferray`] allocates a fresh
//!   ferray [`Array`] and copies each element from the `NumPy` array.
//! - **ferray -> `NumPy`**: [`IntoNumPy::into_pyarray`] allocates a new
//!   Python-owned [`PyArray`] and copies the flattened ferray buffer
//!   into it.

use numpy::Element as NumpyElement;
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;

use ferray_core::array::aliases::{Array1, Array2, ArrayD};
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::{Array, Element, FerrayError};

// ---------------------------------------------------------------------------
// Marker: ferray Element that is also a NumPy element
// ---------------------------------------------------------------------------

/// Sealed marker associating a ferray [`Element`] type with the corresponding
/// `NumPy` element type.
///
/// Implemented for all numeric types that both ferray and `NumPy` support:
/// `bool`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
///
/// Complex types require special handling and are not covered by this
/// trait — use [`crate::arrow_conv`] for those.
pub trait NpElement: Element + NumpyElement {}

macro_rules! impl_np_element {
    ($($ty:ty),*) => {
        $( impl NpElement for $ty {} )*
    };
}

impl_np_element!(bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// f16 / bf16 — ferray-core supports both behind feature flags;
// pyo3-numpy supports half::f16 (Float16Type) but not bf16 in
// the underlying NumPy ABI. Implement only what numpy-rs has
// `NumpyElement` for. Tracked in #739.
#[cfg(feature = "f16")]
impl NpElement for half::f16 {}

// ---------------------------------------------------------------------------
// NumPy -> ferray  (REQ-1)
// ---------------------------------------------------------------------------

/// Extension trait for converting a `NumPy` readonly array into a ferray
/// array. The data is copied (see the module docstring for why).
pub trait AsFerray<T: Element, D: ferray_core::Dimension> {
    /// Copy the `NumPy` array's contents into a fresh ferray [`Array`].
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the `NumPy` dtype does not
    /// match `T`, or [`FerrayError::ShapeMismatch`] if the dimensions do
    /// not match `D`.
    fn as_ferray(&self) -> Result<Array<T, D>, FerrayError>;
}

impl<T: NpElement> AsFerray<T, Ix1> for PyReadonlyArray1<'_, T> {
    fn as_ferray(&self) -> Result<Array1<T>, FerrayError> {
        let py_arr = self.as_array();
        let shape = py_arr.shape();
        let dim = Ix1::new([shape[0]]);
        let data: Vec<T> = py_arr.iter().cloned().collect();
        Array1::<T>::from_vec(dim, data)
    }
}

impl<T: NpElement> AsFerray<T, Ix2> for PyReadonlyArray2<'_, T> {
    fn as_ferray(&self) -> Result<Array2<T>, FerrayError> {
        let py_arr = self.as_array();
        let shape = py_arr.shape();
        let dim = Ix2::new([shape[0], shape[1]]);
        let data: Vec<T> = py_arr.iter().cloned().collect();
        Array2::<T>::from_vec(dim, data)
    }
}

impl<T: NpElement> AsFerray<T, IxDyn> for PyReadonlyArrayDyn<'_, T> {
    fn as_ferray(&self) -> Result<ArrayD<T>, FerrayError> {
        let py_arr = self.as_array();
        let shape = py_arr.shape();
        let dim = IxDyn::new(shape);
        let data: Vec<T> = py_arr.iter().cloned().collect();
        ArrayD::<T>::from_vec(dim, data)
    }
}

// ---------------------------------------------------------------------------
// ferray -> NumPy  (REQ-2)
// ---------------------------------------------------------------------------

/// Extension trait for converting an owned ferray array to a `NumPy` array.
///
/// The data is copied into a fresh Python-owned `PyArray` (see the module
/// docstring for why this is not yet zero-copy).
pub trait IntoNumPy<T: Element, D: ferray_core::Dimension> {
    /// The `PyO3` `NumPy` array type produced.
    type PyArrayType;

    /// Copy this ferray array's contents into a new Python-owned `NumPy`
    /// array. The input array's buffer is flattened, the resulting Vec
    /// is handed to `PyArray::from_vec`, and the result is reshaped.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the element type has no
    /// `NumPy` equivalent, or [`FerrayError::ShapeMismatch`] if the
    /// reshape step fails.
    fn into_pyarray(self, py: Python<'_>) -> Result<Bound<'_, Self::PyArrayType>, FerrayError>;
}

impl<T: NpElement> IntoNumPy<T, Ix1> for Array1<T> {
    type PyArrayType = PyArray1<T>;

    fn into_pyarray(self, py: Python<'_>) -> Result<Bound<'_, PyArray1<T>>, FerrayError> {
        // 1-D has no reshape overhead to worry about; hand the flat Vec
        // straight to PyArray1::from_vec.
        let data = self.to_vec_flat();
        Ok(PyArray1::from_vec(py, data))
    }
}

impl<T: NpElement> IntoNumPy<T, Ix2> for Array2<T> {
    type PyArrayType = PyArray2<T>;

    fn into_pyarray(self, py: Python<'_>) -> Result<Bound<'_, PyArray2<T>>, FerrayError> {
        // #310: hand the underlying ndarray::Array2 directly to
        // PyArray2::from_owned_array. This avoids the prior
        // flatten-into-PyArray1-then-reshape round-trip, and PyO3's
        // numpy crate preserves the source layout automatically:
        // ferray Array2 → ferray::into_ndarray() (a move, no copy)
        // → ndarray::Array2 (carries C/F-contiguity in its strides)
        // → PyArray2::from_owned_array (sees the same strides and
        // produces a PyArray2 with matching F_CONTIGUOUS flag for
        // F-major sources, addressing #698 without a manual branch).
        Ok(PyArray2::from_owned_array(py, self.into_ndarray()))
    }
}

impl<T: NpElement> IntoNumPy<T, IxDyn> for ArrayD<T> {
    type PyArrayType = PyArrayDyn<T>;

    fn into_pyarray(self, py: Python<'_>) -> Result<Bound<'_, PyArrayDyn<T>>, FerrayError> {
        // #310: same direct-handoff as the Ix2 impl. PyArrayDyn maps to
        // numpy's IxDyn ndarray, so passing through ferray's
        // into_ndarray() preserves the source contiguity (#698) without
        // a manual F/C branch and skips the flatten-then-reshape pass.
        Ok(PyArrayDyn::from_owned_array(py, self.into_ndarray()))
    }
}

// Note: Tests for PyO3/NumPy require a Python interpreter and are best run
// with `cargo test --features python` in an environment where Python + numpy
// are available. The #[cfg(test)] module below contains tests that use
// pyo3::prepare_freethreaded_python().

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unreadable_literal)] // Roundtrip tests assert exact equality on hand-picked NumPy values.
mod tests {
    use super::*;
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    fn with_python<F: for<'py> FnOnce(Python<'py>)>(f: F) {
        Python::initialize();
        Python::attach(f);
    }

    macro_rules! test_roundtrip_1d {
        ($name:ident, $ty:ty, $values:expr) => {
            #[test]
            fn $name() {
                with_python(|py| {
                    let data: Vec<$ty> = $values;
                    let len = data.len();
                    let arr = Array1::<$ty>::from_vec(Ix1::new([len]), data.clone()).unwrap();

                    // ferray -> numpy
                    let py_arr = arr.into_pyarray(py).unwrap();
                    assert_eq!(py_arr.shape(), [len]);

                    // numpy -> ferray
                    let readonly = py_arr.readonly();
                    let back: Array1<$ty> = readonly.as_ferray().unwrap();
                    assert_eq!(back.shape(), &[len]);
                    assert_eq!(back.as_slice().unwrap(), &data[..]);
                });
            }
        };
    }

    test_roundtrip_1d!(roundtrip_f64, f64, vec![1.0, 2.5, -4.75, 0.0]);
    test_roundtrip_1d!(roundtrip_f32, f32, vec![1.0f32, -2.5, 0.0]);
    test_roundtrip_1d!(roundtrip_i32, i32, vec![0, 1, -1, i32::MAX, i32::MIN]);
    test_roundtrip_1d!(roundtrip_i64, i64, vec![0i64, 42, -99]);
    test_roundtrip_1d!(roundtrip_u8, u8, vec![0u8, 128, 255]);
    test_roundtrip_1d!(roundtrip_u32, u32, vec![0u32, 1, u32::MAX]);
    // #196: bool round-trip — previously NpElement excluded bool even
    // though numpy::Element implements it, so any Array<bool> → NumPy
    // path was a type error.
    test_roundtrip_1d!(roundtrip_bool, bool, vec![true, false, true, true, false]);

    #[test]
    fn roundtrip_2d_bool() {
        with_python(|py| {
            let data = vec![true, false, true, false, true, false];
            let arr = Array2::<bool>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
            let py_arr = arr.into_pyarray(py).unwrap();
            assert_eq!(py_arr.shape(), [2, 3]);

            let readonly = py_arr.readonly();
            let back: Array2<bool> = readonly.as_ferray().unwrap();
            assert_eq!(back.shape(), &[2, 3]);
            assert_eq!(back.to_vec_flat(), data);
        });
    }

    #[test]
    fn roundtrip_2d_f64() {
        with_python(|py| {
            let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
            let arr = Array2::<f64>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();

            let py_arr = arr.into_pyarray(py).unwrap();
            assert_eq!(py_arr.shape(), [2, 3]);

            let readonly = py_arr.readonly();
            let back: Array2<f64> = readonly.as_ferray().unwrap();
            assert_eq!(back.shape(), &[2, 3]);
            assert_eq!(back.to_vec_flat(), data);
        });
    }

    #[test]
    fn roundtrip_dyn_f64() {
        with_python(|py| {
            let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
            let arr = ArrayD::<f64>::from_vec(IxDyn::new(&[2, 3]), data.clone()).unwrap();

            let py_arr = arr.into_pyarray(py).unwrap();
            assert_eq!(py_arr.shape(), [2, 3]);

            let readonly = py_arr.readonly();
            let back: ArrayD<f64> = readonly.as_ferray().unwrap();
            assert_eq!(back.shape(), &[2, 3]);
            assert_eq!(back.to_vec_flat(), data);
        });
    }

    #[test]
    fn empty_array_roundtrip() {
        with_python(|py| {
            let arr = Array1::<f64>::from_vec(Ix1::new([0]), vec![]).unwrap();
            let py_arr = arr.into_pyarray(py).unwrap();
            assert_eq!(py_arr.shape(), [0]);

            let readonly = py_arr.readonly();
            let back: Array1<f64> = readonly.as_ferray().unwrap();
            assert_eq!(back.shape(), &[0]);
        });
    }

    #[test]
    fn bit_identical_roundtrip() {
        with_python(|py| {
            let original: Vec<f64> = vec![
                1.0,
                -0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.23456789012345e-300,
            ];
            let len = original.len();
            let arr = Array1::<f64>::from_vec(Ix1::new([len]), original.clone()).unwrap();
            let py_arr = arr.into_pyarray(py).unwrap();
            let readonly = py_arr.readonly();
            let back: Array1<f64> = readonly.as_ferray().unwrap();

            let back_slice = back.as_slice().unwrap();
            for (i, (a, b)) in original.iter().zip(back_slice.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bit mismatch at index {i}: {a} vs {b}"
                );
            }
        });
    }
}
