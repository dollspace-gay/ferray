//! Arrow <-> ferray array conversions (feature-gated behind `"arrow"`).
//!
//! Provides [`ToArrow`] and [`FromArrow`] traits for converting between
//! ferray 1-D arrays and Arrow [`PrimitiveArray`] / [`BooleanArray`].
//!
//! # Memory semantics
//!
//! Both directions copy the data buffer. The previous documentation
//! claimed "zero-copy when the source is C-contiguous", but the
//! implementation unconditionally calls `slice.to_vec()` on the ferray
//! side (which allocates and copies) and `PrimitiveArray::values()`
//! cloned into a Vec on the Arrow side. True zero-copy would require
//! Arrow's [`Buffer::from_custom_allocation`] with a ferray-specific
//! Drop hook on the outbound path, and unsafe pointer sharing on the
//! inbound path — a non-trivial design change tracked as a follow-up.
//! See the crate-level docstring for the full zero-copy discussion.

use arrow::array::{Array as ArrowArray, BooleanArray, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};

use ferray_core::array::aliases::Array1;
use ferray_core::{Element, FerrayError, Ix1};

use crate::dtype_map;

// ---------------------------------------------------------------------------
// Helper: marker trait linking ferray Element to ArrowPrimitiveType
// ---------------------------------------------------------------------------

/// Sealed marker associating a ferray [`Element`] type with its Arrow
/// [`ArrowPrimitiveType`] counterpart.
///
/// This is implemented for every numeric type that both ferray and Arrow
/// support. `bool` is handled separately because Arrow uses a bit-packed
/// [`BooleanArray`] rather than a `PrimitiveArray`.
pub trait ArrowElement: Element + ArrowNativeType {
    /// The Arrow primitive type tag.
    type ArrowType: ArrowPrimitiveType<Native = Self>;
}

macro_rules! impl_arrow_element {
    ($rust_ty:ty, $arrow_ty:ty) => {
        impl ArrowElement for $rust_ty {
            type ArrowType = $arrow_ty;
        }
    };
}

impl_arrow_element!(u8, arrow::datatypes::UInt8Type);
impl_arrow_element!(u16, arrow::datatypes::UInt16Type);
impl_arrow_element!(u32, arrow::datatypes::UInt32Type);
impl_arrow_element!(u64, arrow::datatypes::UInt64Type);
impl_arrow_element!(i8, arrow::datatypes::Int8Type);
impl_arrow_element!(i16, arrow::datatypes::Int16Type);
impl_arrow_element!(i32, arrow::datatypes::Int32Type);
impl_arrow_element!(i64, arrow::datatypes::Int64Type);
impl_arrow_element!(f32, arrow::datatypes::Float32Type);
impl_arrow_element!(f64, arrow::datatypes::Float64Type);

// ---------------------------------------------------------------------------
// ferray -> Arrow  (REQ-4)
// ---------------------------------------------------------------------------

/// Extension trait for converting a ferray array to an Arrow array.
///
/// Multi-dimensional inputs flatten to a 1-D Arrow primitive in
/// row-major (C) order — Arrow's columnar primitive format is
/// inherently 1-D, so any dimensionality information is dropped on
/// the way out. The reverse path returns a 1-D ferray array; callers
/// who need to recover the original shape should `.reshape()` after
/// [`FromArrow::into_ferray`] (#307).
pub trait ToArrow {
    /// The Arrow array type produced by the conversion.
    type ArrowArray;

    /// Convert this ferray array to an Arrow array. This copies the
    /// underlying buffer — see the module docstring for the rationale.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the element type has no
    /// Arrow equivalent.
    fn to_arrow(&self) -> Result<Self::ArrowArray, FerrayError>;
}

impl<T, D> ToArrow for ferray_core::Array<T, D>
where
    T: ArrowElement,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
    D: ferray_core::dimension::Dimension,
{
    type ArrowArray = PrimitiveArray<T::ArrowType>;

    fn to_arrow(&self) -> Result<Self::ArrowArray, FerrayError> {
        // Validate that the ferray dtype has an Arrow equivalent.
        let _ = dtype_map::dtype_to_arrow(self.dtype())?;

        // Copy the data into a Vec<T>. `as_slice()` succeeds for
        // C-contiguous arrays (fast single-copy path); otherwise
        // fall back to a logical-order iteration via `to_vec_flat`.
        let data: Vec<T> = match self.as_slice() {
            Some(slice) => slice.to_vec(),
            None => self.to_vec_flat(),
        };

        // Hand the Vec to Arrow, which takes ownership of the allocation.
        let buffer = Buffer::from_vec(data);
        let array = PrimitiveArray::<T::ArrowType>::new(buffer.into(), None);
        Ok(array)
    }
}

/// Extension trait for converting a ferray boolean array to an Arrow
/// [`BooleanArray`]. Multi-dimensional inputs flatten to 1-D in
/// row-major order (#307).
pub trait ToArrowBool {
    /// Convert this ferray bool array to an Arrow [`BooleanArray`].
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] on internal inconsistency.
    fn to_arrow(&self) -> Result<BooleanArray, FerrayError>;
}

impl<D: ferray_core::dimension::Dimension> ToArrowBool for ferray_core::Array<bool, D> {
    fn to_arrow(&self) -> Result<BooleanArray, FerrayError> {
        let values: Vec<bool> = self.to_vec_flat();
        Ok(BooleanArray::from(values))
    }
}

// ---------------------------------------------------------------------------
// Arrow -> ferray  (REQ-5)
// ---------------------------------------------------------------------------

/// Extension trait for converting an Arrow primitive array to a ferray
/// 1-D array.
pub trait FromArrow<T: Element>: Sized {
    /// Convert an Arrow array to a ferray `Array1<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the Arrow array dtype does
    /// not match `T`, or [`FerrayError::InvalidValue`] if the Arrow array
    /// contains nulls (ferray arrays do not support null values).
    fn into_ferray(self) -> Result<Array1<T>, FerrayError>;
}

impl<T: ArrowElement> FromArrow<T> for PrimitiveArray<T::ArrowType>
where
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    fn into_ferray(self) -> Result<Array1<T>, FerrayError> {
        // Validate no nulls
        if self.null_count() > 0 {
            return Err(FerrayError::invalid_value(format!(
                "Arrow array contains {} null values; ferray arrays do not support nulls",
                self.null_count()
            )));
        }

        // Validate dtype correspondence
        let arrow_dt = self.data_type();
        let ferray_dt = dtype_map::arrow_to_dtype(arrow_dt)?;
        if ferray_dt != T::dtype() {
            return Err(FerrayError::invalid_dtype(format!(
                "Arrow dtype {arrow_dt:?} maps to ferray {ferray_dt}, but requested {}",
                T::dtype()
            )));
        }

        let values = self.values();
        let data: Vec<T> = values.iter().copied().collect();
        let len = data.len();
        Array1::<T>::from_vec(Ix1::new([len]), data)
    }
}

/// Extension trait for converting an Arrow [`BooleanArray`] to a ferray
/// `Array1<bool>`.
pub trait FromArrowBool: Sized {
    /// Convert an Arrow boolean array to a ferray `Array1<bool>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidValue`] if the array contains nulls.
    fn into_ferray_bool(self) -> Result<Array1<bool>, FerrayError>;
}

impl FromArrowBool for BooleanArray {
    fn into_ferray_bool(self) -> Result<Array1<bool>, FerrayError> {
        if self.null_count() > 0 {
            return Err(FerrayError::invalid_value(format!(
                "Arrow BooleanArray contains {} null values; ferray arrays do not support nulls",
                self.null_count()
            )));
        }

        let data: Vec<bool> = self.iter().map(|v| v.unwrap_or(false)).collect();
        let len = data.len();
        Array1::<bool>::from_vec(Ix1::new([len]), data)
    }
}

// ---------------------------------------------------------------------------
// 2-D and dynamic-rank interop (#549)
// ---------------------------------------------------------------------------
//
// Arrow's primitive data model is inherently 1-D: a `PrimitiveArray` is
// a flat buffer. Higher-rank ferray arrays can be represented in two
// natural ways:
//
//   1. "flattened": one Arrow array containing the row-major bytes.
//      Round-trip requires the caller to remember the shape.
//   2. "columnar": a `Vec<PrimitiveArray>`, one per column of the 2-D
//      ferray array. This is the natural fit for a downstream Arrow
//      consumer that treats each column as a field.
//
// We provide both. Callers pick whichever matches their pipeline.

use ferray_core::array::aliases::{Array2, ArrayD};
use ferray_core::dimension::{Ix2, IxDyn};

/// Convert an `Array2<T>` into a `Vec` of 1-D Arrow arrays, one per
/// column of the input. This is the natural shape for a downstream
/// consumer that turns each column into an Arrow field.
///
/// # Errors
/// - `FerrayError::InvalidDtype` if the element type has no Arrow equivalent.
pub fn array2_to_arrow_columns<T>(
    a: &Array2<T>,
) -> Result<Vec<PrimitiveArray<T::ArrowType>>, FerrayError>
where
    T: ArrowElement,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    let _ = dtype_map::dtype_to_arrow(a.dtype())?;
    let shape = a.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    let mut out: Vec<PrimitiveArray<T::ArrowType>> = Vec::with_capacity(ncols);
    for c in 0..ncols {
        let mut col: Vec<T> = Vec::with_capacity(nrows);
        for r in 0..nrows {
            // Use iter+nth (slow path) because we need strided access.
            // For C-contiguous arrays this still walks via as_slice below.
            if let Some(slice) = a.as_slice() {
                col.push(slice[r * ncols + c]);
            } else {
                // Non-contiguous fallback — one-off; build from scratch
                // via iter indexing. Rare in practice.
                col.push(*a.iter().nth(r * ncols + c).unwrap());
            }
        }
        let buffer = Buffer::from_vec(col);
        out.push(PrimitiveArray::<T::ArrowType>::new(buffer.into(), None));
    }
    Ok(out)
}

/// Stack a `Vec` of 1-D Arrow arrays into a single `Array2<T>`, one
/// column per input array. All input arrays must have the same length;
/// the result has shape `(len, ncols)`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the columns have different lengths.
/// - `FerrayError::InvalidDtype` / `InvalidValue` as per `FromArrow`.
pub fn array2_from_arrow_columns<T>(
    cols: &[PrimitiveArray<T::ArrowType>],
) -> Result<Array2<T>, FerrayError>
where
    T: ArrowElement,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    if cols.is_empty() {
        return Array2::<T>::from_vec(Ix2::new([0, 0]), Vec::new());
    }
    let nrows = cols[0].len();
    let ncols = cols.len();

    // Validate shapes and dtypes up front so we fail fast.
    for (i, c) in cols.iter().enumerate() {
        if c.len() != nrows {
            return Err(FerrayError::shape_mismatch(format!(
                "array2_from_arrow_columns: column {i} has length {} but column 0 has length {nrows}",
                c.len()
            )));
        }
        if c.null_count() > 0 {
            return Err(FerrayError::invalid_value(format!(
                "Arrow array column {i} contains {} nulls; ferray arrays do not support nulls",
                c.null_count()
            )));
        }
        let arrow_dt = c.data_type();
        let ferray_dt = dtype_map::arrow_to_dtype(arrow_dt)?;
        if ferray_dt != T::dtype() {
            return Err(FerrayError::invalid_dtype(format!(
                "Arrow dtype {arrow_dt:?} on column {i} maps to ferray {ferray_dt}, but requested {}",
                T::dtype()
            )));
        }
    }

    // Interleave into row-major order.
    let mut data: Vec<T> = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in cols {
            data.push(c.values()[r]);
        }
    }
    Array2::<T>::from_vec(Ix2::new([nrows, ncols]), data)
}

/// Flatten a dynamic-rank ferray array into a single 1-D Arrow
/// primitive array. The shape is NOT preserved in the returned Arrow
/// array — pass the shape alongside it (or use
/// [`arrayd_from_arrow_flat`] with an explicit target shape to
/// reshape on the round-trip).
///
/// # Errors
/// - `FerrayError::InvalidDtype` if the element type has no Arrow equivalent.
pub fn arrayd_to_arrow_flat<T>(a: &ArrayD<T>) -> Result<PrimitiveArray<T::ArrowType>, FerrayError>
where
    T: ArrowElement,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    let _ = dtype_map::dtype_to_arrow(a.dtype())?;
    let data: Vec<T> = match a.as_slice() {
        Some(slice) => slice.to_vec(),
        None => a.to_vec_flat(),
    };
    let buffer = Buffer::from_vec(data);
    Ok(PrimitiveArray::<T::ArrowType>::new(buffer.into(), None))
}

/// Reshape a flat Arrow primitive array into a dynamic-rank ferray
/// array with the given shape.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `shape.iter().product()` does not
///   match the Arrow array's length.
pub fn arrayd_from_arrow_flat<T>(
    arr: &PrimitiveArray<T::ArrowType>,
    shape: &[usize],
) -> Result<ArrayD<T>, FerrayError>
where
    T: ArrowElement,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    if arr.null_count() > 0 {
        return Err(FerrayError::invalid_value(format!(
            "Arrow array contains {} null values; ferray arrays do not support nulls",
            arr.null_count()
        )));
    }
    let arrow_dt = arr.data_type();
    let ferray_dt = dtype_map::arrow_to_dtype(arrow_dt)?;
    if ferray_dt != T::dtype() {
        return Err(FerrayError::invalid_dtype(format!(
            "Arrow dtype {arrow_dt:?} maps to ferray {ferray_dt}, but requested {}",
            T::dtype()
        )));
    }
    let expected: usize = shape.iter().product();
    if arr.len() != expected {
        return Err(FerrayError::shape_mismatch(format!(
            "arrayd_from_arrow_flat: arrow length {} does not match shape {:?} (product {})",
            arr.len(),
            shape,
            expected
        )));
    }
    let data: Vec<T> = arr.values().iter().copied().collect();
    ArrayD::<T>::from_vec(IxDyn::new(shape), data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::type_repetition_in_bounds,
    clippy::trait_duplication_in_bounds
)]
mod tests {
    use super::*;

    // ----- ferray -> Arrow -> ferray roundtrip (AC-2) -----

    macro_rules! test_roundtrip {
        ($name:ident, $ty:ty, $values:expr) => {
            #[test]
            fn $name() {
                let data: Vec<$ty> = $values;
                let len = data.len();
                let arr = Array1::<$ty>::from_vec(Ix1::new([len]), data.clone()).unwrap();

                // ferray -> arrow
                let arrow_arr = arr.to_arrow().unwrap();
                assert_eq!(arrow_arr.len(), len);

                // arrow -> ferray
                let back: Array1<$ty> = arrow_arr.into_ferray().unwrap();
                assert_eq!(back.shape(), &[len]);
                assert_eq!(back.as_slice().unwrap(), &data[..]);
            }
        };
    }

    test_roundtrip!(roundtrip_f64, f64, vec![1.0, 2.5, -4.75, 0.0, f64::MAX]);
    test_roundtrip!(roundtrip_f32, f32, vec![1.0f32, -2.5, 0.0, f32::MIN]);
    test_roundtrip!(roundtrip_i32, i32, vec![0, 1, -1, i32::MAX, i32::MIN]);
    test_roundtrip!(roundtrip_i64, i64, vec![0i64, 42, -99]);
    test_roundtrip!(roundtrip_i8, i8, vec![0i8, 127, -128]);
    test_roundtrip!(roundtrip_i16, i16, vec![0i16, 32767, -32768]);
    test_roundtrip!(roundtrip_u8, u8, vec![0u8, 128, 255]);
    test_roundtrip!(roundtrip_u16, u16, vec![0u16, 1000, 65535]);
    test_roundtrip!(roundtrip_u32, u32, vec![0u32, 1, u32::MAX]);
    test_roundtrip!(roundtrip_u64, u64, vec![0u64, 1, u64::MAX]);

    #[test]
    fn roundtrip_bool() {
        let data = vec![true, false, true, true, false];
        let len = data.len();
        let arr = Array1::<bool>::from_vec(Ix1::new([len]), data.clone()).unwrap();

        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), len);

        let back = arrow_arr.into_ferray_bool().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn empty_array_roundtrip() {
        let arr = Array1::<f64>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), 0);

        let back: Array1<f64> = arrow_arr.into_ferray().unwrap();
        assert_eq!(back.shape(), &[0]);
    }

    #[test]
    fn arrow_with_nulls_rejected() {
        // Create an Arrow array with a null
        let arr =
            PrimitiveArray::<arrow::datatypes::Float64Type>::from(vec![Some(1.0), None, Some(3.0)]);
        let result: Result<Array1<f64>, _> = arr.into_ferray();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("null"));
    }

    #[test]
    fn bool_arrow_with_nulls_rejected() {
        let arr = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let result = arr.into_ferray_bool();
        assert!(result.is_err());
    }

    // AC-3: dtype mismatch returns clear error
    #[test]
    fn dtype_mismatch_arrow_to_dtype() {
        // arrow::datatypes::Utf8 has no ferray equivalent
        let result = dtype_map::arrow_to_dtype(&arrow::datatypes::DataType::Utf8);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no ferray equivalent"), "got: {msg}");
    }

    #[test]
    fn bit_identical_roundtrip() {
        // AC-2: ferray -> arrow -> ferray produces bit-identical data
        let original: Vec<f64> = vec![
            1.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.23456789012345e-300,
            9.87654321098765e+300,
        ];
        let len = original.len();
        let arr = Array1::<f64>::from_vec(Ix1::new([len]), original.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        let back: Array1<f64> = arrow_arr.into_ferray().unwrap();

        let back_slice = back.as_slice().unwrap();
        for (i, (a, b)) in original.iter().zip(back_slice.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    // ----- 2-D and dynamic-rank interop (#549) -----

    #[test]
    fn array2_arrow_columns_roundtrip_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Array2::<f64>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();

        let cols = array2_to_arrow_columns(&a).unwrap();
        // 3 columns of 2 rows each
        assert_eq!(cols.len(), 3);
        for c in &cols {
            assert_eq!(c.len(), 2);
        }
        // Column 0: rows are elements [0] and [3]
        assert_eq!(cols[0].values()[0], 1.0);
        assert_eq!(cols[0].values()[1], 4.0);
        // Column 2: elements [2] and [5]
        assert_eq!(cols[2].values()[0], 3.0);
        assert_eq!(cols[2].values()[1], 6.0);

        let back = array2_from_arrow_columns::<f64>(&cols).unwrap();
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn array2_arrow_columns_roundtrip_i32() {
        let data = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
        let a = Array2::<i32>::from_vec(Ix2::new([4, 2]), data.clone()).unwrap();
        let cols = array2_to_arrow_columns(&a).unwrap();
        assert_eq!(cols.len(), 2);
        let back = array2_from_arrow_columns::<i32>(&cols).unwrap();
        assert_eq!(back.shape(), &[4, 2]);
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn array2_from_arrow_columns_inconsistent_length_errors() {
        let c0 = PrimitiveArray::<arrow::datatypes::Float64Type>::new(
            Buffer::from_vec(vec![1.0_f64, 2.0, 3.0]).into(),
            None,
        );
        let c1 = PrimitiveArray::<arrow::datatypes::Float64Type>::new(
            Buffer::from_vec(vec![4.0_f64, 5.0]).into(),
            None,
        );
        assert!(array2_from_arrow_columns::<f64>(&[c0, c1]).is_err());
    }

    #[test]
    fn array2_from_arrow_columns_empty_is_empty() {
        let cols: Vec<PrimitiveArray<arrow::datatypes::Float64Type>> = Vec::new();
        let a = array2_from_arrow_columns::<f64>(&cols).unwrap();
        assert_eq!(a.shape(), &[0, 0]);
    }

    #[test]
    fn arrayd_flat_roundtrip() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = ArrayD::<f64>::from_vec(IxDyn::new(&[2, 2, 2]), data.clone()).unwrap();
        let flat = arrayd_to_arrow_flat(&a).unwrap();
        assert_eq!(flat.len(), 8);
        let back = arrayd_from_arrow_flat::<f64>(&flat, &[2, 2, 2]).unwrap();
        assert_eq!(back.shape(), &[2, 2, 2]);
        assert_eq!(back.to_vec_flat(), data);
    }

    #[test]
    fn arrayd_flat_reshape_to_different_rank() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = ArrayD::<f64>::from_vec(IxDyn::new(&[6]), data.clone()).unwrap();
        let flat = arrayd_to_arrow_flat(&a).unwrap();
        // Re-interpret the same 6 values as (2, 3).
        let reshaped = arrayd_from_arrow_flat::<f64>(&flat, &[2, 3]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.to_vec_flat(), data);
    }

    #[test]
    fn arrayd_from_arrow_flat_wrong_shape_errors() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = ArrayD::<f64>::from_vec(IxDyn::new(&[6]), data).unwrap();
        let flat = arrayd_to_arrow_flat(&a).unwrap();
        // 6 elements cannot reshape to (2, 4).
        assert!(arrayd_from_arrow_flat::<f64>(&flat, &[2, 4]).is_err());
    }

    // ----- ToArrow now generic over Dimension (#307) ---------------------

    #[test]
    fn to_arrow_from_array2_flattens_row_major() {
        use ferray_core::Array;
        use ferray_core::dimension::Ix2;
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), 6);
        let values: Vec<f64> = arrow_arr.values().iter().copied().collect();
        // Row-major flattening: original [[1,2,3],[4,5,6]] → [1,2,3,4,5,6].
        assert_eq!(values, data);
    }

    #[test]
    fn to_arrow_from_arrayd_flattens_row_major() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let arr = ArrayD::<f32>::from_vec(IxDyn::new(&[2, 3, 4]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), 24);
        let values: Vec<f32> = arrow_arr.values().iter().copied().collect();
        assert_eq!(values, data);
    }

    #[test]
    fn to_arrow_bool_from_array2_flattens() {
        use ferray_core::Array;
        use ferray_core::dimension::Ix2;
        let data = vec![true, false, false, true, false, true];
        let arr = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), 6);
        let got: Vec<bool> = (0..arrow_arr.len()).map(|i| arrow_arr.value(i)).collect();
        assert_eq!(got, data);
    }
}
