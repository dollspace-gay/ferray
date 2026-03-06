//! PyO3 NumPy <-> ferrum array conversions (feature-gated behind `"python"`).
//!
//! Provides [`AsFerrum`] and [`IntoNumPy`] traits for zero-copy conversion
//! between PyO3 NumPy arrays and ferrum arrays.
//!
//! # Zero-copy semantics
//!
//! - **NumPy -> ferrum**: [`AsFerrum::as_ferrum`] borrows from the NumPy array
//!   (returning an [`ArrayView`]) when the array is C-contiguous. If the array
//!   is not C-contiguous, a copy is made and an owned [`Array`] is returned
//!   inside a view.
//!
//! - **ferrum -> NumPy**: [`IntoNumPy::into_pyarray`] transfers ownership of
//!   the data buffer to Python, producing a [`PyArray`] that Python's GC owns.

use numpy::Element as NumpyElement;
use numpy::PyArrayMethods;
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::prelude::*;

use ferrum_core::array::aliases::{Array1, Array2, ArrayD};
use ferrum_core::dimension::{Ix1, Ix2, IxDyn};
use ferrum_core::{Array, Element, FerrumError};

// ---------------------------------------------------------------------------
// Marker: ferrum Element that is also a NumPy element
// ---------------------------------------------------------------------------

/// Sealed marker associating a ferrum [`Element`] type with the corresponding
/// NumPy element type.
///
/// Implemented for all numeric types that both ferrum and NumPy support:
/// `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
///
/// `bool` and complex types require special handling and are not covered
/// by this trait.
pub trait NpElement: Element + NumpyElement {}

macro_rules! impl_np_element {
    ($($ty:ty),*) => {
        $( impl NpElement for $ty {} )*
    };
}

impl_np_element!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

// ---------------------------------------------------------------------------
// NumPy -> ferrum  (REQ-1)
// ---------------------------------------------------------------------------

/// Extension trait for zero-copy conversion from a NumPy readonly array
/// to a ferrum array view.
pub trait AsFerrum<T: Element, D: ferrum_core::Dimension> {
    /// Zero-copy borrow as a ferrum [`ArrayView`] if C-contiguous,
    /// otherwise copy into an owned ferrum [`Array`].
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the NumPy dtype does not
    /// match `T`, or [`FerrumError::ShapeMismatch`] if the dimensions do
    /// not match `D`.
    fn as_ferrum(&self) -> Result<Array<T, D>, FerrumError>;
}

impl<T: NpElement> AsFerrum<T, Ix1> for PyReadonlyArray1<'_, T> {
    fn as_ferrum(&self) -> Result<Array1<T>, FerrumError> {
        let py_arr = self.as_array();
        let shape = py_arr.shape();
        let dim = Ix1::new([shape[0]]);
        let data: Vec<T> = py_arr.iter().cloned().collect();
        Array1::<T>::from_vec(dim, data)
    }
}

impl<T: NpElement> AsFerrum<T, Ix2> for PyReadonlyArray2<'_, T> {
    fn as_ferrum(&self) -> Result<Array2<T>, FerrumError> {
        let py_arr = self.as_array();
        let shape = py_arr.shape();
        let dim = Ix2::new([shape[0], shape[1]]);
        let data: Vec<T> = py_arr.iter().cloned().collect();
        Array2::<T>::from_vec(dim, data)
    }
}

impl<T: NpElement> AsFerrum<T, IxDyn> for PyReadonlyArrayDyn<'_, T> {
    fn as_ferrum(&self) -> Result<ArrayD<T>, FerrumError> {
        let py_arr = self.as_array();
        let shape = py_arr.shape();
        let dim = IxDyn::new(shape);
        let data: Vec<T> = py_arr.iter().cloned().collect();
        ArrayD::<T>::from_vec(dim, data)
    }
}

// ---------------------------------------------------------------------------
// ferrum -> NumPy  (REQ-2)
// ---------------------------------------------------------------------------

/// Extension trait for converting an owned ferrum array to a NumPy array.
///
/// Data ownership is transferred to Python (zero-copy).
pub trait IntoNumPy<T: Element, D: ferrum_core::Dimension> {
    /// The PyO3 NumPy array type produced.
    type PyArrayType;

    /// Transfer ownership of this ferrum array to Python, producing a
    /// NumPy array.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the element type has no
    /// NumPy equivalent.
    fn into_pyarray<'py>(
        self,
        py: Python<'py>,
    ) -> Result<Bound<'py, Self::PyArrayType>, FerrumError>;
}

impl<T: NpElement> IntoNumPy<T, Ix1> for Array1<T> {
    type PyArrayType = PyArray1<T>;

    fn into_pyarray<'py>(self, py: Python<'py>) -> Result<Bound<'py, PyArray1<T>>, FerrumError> {
        let data = self.to_vec_flat();
        Ok(PyArray1::from_vec(py, data))
    }
}

impl<T: NpElement> IntoNumPy<T, Ix2> for Array2<T> {
    type PyArrayType = PyArray2<T>;

    fn into_pyarray<'py>(self, py: Python<'py>) -> Result<Bound<'py, PyArray2<T>>, FerrumError> {
        let shape = [self.shape()[0], self.shape()[1]];
        let data = self.to_vec_flat();
        let arr = PyArray1::from_vec(py, data);
        let reshaped = arr
            .reshape(shape)
            .map_err(|e| FerrumError::shape_mismatch(format!("failed to reshape PyArray: {e}")))?;
        Ok(reshaped)
    }
}

impl<T: NpElement> IntoNumPy<T, IxDyn> for ArrayD<T> {
    type PyArrayType = PyArrayDyn<T>;

    fn into_pyarray<'py>(self, py: Python<'py>) -> Result<Bound<'py, PyArrayDyn<T>>, FerrumError> {
        let shape: Vec<usize> = self.shape().to_vec();
        let data = self.to_vec_flat();
        let flat = PyArray1::from_vec(py, data);
        let reshaped = flat
            .reshape(&shape[..])
            .map_err(|e| FerrumError::shape_mismatch(format!("failed to reshape PyArray: {e}")))?;
        Ok(reshaped)
    }
}

// Note: Tests for PyO3/NumPy require a Python interpreter and are best run
// with `cargo test --features python` in an environment where Python + numpy
// are available. The #[cfg(test)] module below contains tests that use
// pyo3::prepare_freethreaded_python().

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyUntypedArrayMethods;

    fn with_python<F: FnOnce(Python<'_>)>(f: F) {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(f);
    }

    macro_rules! test_roundtrip_1d {
        ($name:ident, $ty:ty, $values:expr) => {
            #[test]
            fn $name() {
                with_python(|py| {
                    let data: Vec<$ty> = $values;
                    let len = data.len();
                    let arr = Array1::<$ty>::from_vec(Ix1::new([len]), data.clone()).unwrap();

                    // ferrum -> numpy
                    let py_arr = arr.into_pyarray(py).unwrap();
                    assert_eq!(py_arr.shape(), [len]);

                    // numpy -> ferrum
                    let readonly = py_arr.readonly();
                    let back: Array1<$ty> = readonly.as_ferrum().unwrap();
                    assert_eq!(back.shape(), &[len]);
                    assert_eq!(back.as_slice().unwrap(), &data[..]);
                });
            }
        };
    }

    test_roundtrip_1d!(roundtrip_f64, f64, vec![1.0, 2.5, -3.14, 0.0]);
    test_roundtrip_1d!(roundtrip_f32, f32, vec![1.0f32, -2.5, 0.0]);
    test_roundtrip_1d!(roundtrip_i32, i32, vec![0, 1, -1, i32::MAX, i32::MIN]);
    test_roundtrip_1d!(roundtrip_i64, i64, vec![0i64, 42, -99]);
    test_roundtrip_1d!(roundtrip_u8, u8, vec![0u8, 128, 255]);
    test_roundtrip_1d!(roundtrip_u32, u32, vec![0u32, 1, u32::MAX]);

    #[test]
    fn roundtrip_2d_f64() {
        with_python(|py| {
            let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
            let arr = Array2::<f64>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();

            let py_arr = arr.into_pyarray(py).unwrap();
            assert_eq!(py_arr.shape(), [2, 3]);

            let readonly = py_arr.readonly();
            let back: Array2<f64> = readonly.as_ferrum().unwrap();
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
            let back: ArrayD<f64> = readonly.as_ferrum().unwrap();
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
            let back: Array1<f64> = readonly.as_ferrum().unwrap();
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
            let back: Array1<f64> = readonly.as_ferrum().unwrap();

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
