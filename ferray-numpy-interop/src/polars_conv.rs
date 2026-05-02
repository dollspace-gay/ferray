//! Polars <-> ferray array conversions (feature-gated behind `"polars"`).
//!
//! Provides [`ToPolars`] and [`FromPolars`] traits for converting between
//! ferray 1-D arrays and Polars [`Series`].

use polars::prelude::*;

use ferray_core::array::aliases::Array1;
use ferray_core::{Element, FerrayError, Ix1};

use crate::dtype_map;

// ---------------------------------------------------------------------------
// Sealed marker: ferray Element <-> Polars PolarsNumericType
// ---------------------------------------------------------------------------

/// Sealed marker associating a ferray [`Element`] type with Polars
/// chunked-array extraction.
///
/// Implemented for all numeric types that both ferray and Polars support.
/// `bool` is handled separately.
pub trait PolarsElement: Element {
    /// The Polars numeric type tag, whose `Native` associated type is `Self`.
    type PolarsType: PolarsNumericType<Native = Self>;

    /// Extract a reference to the typed [`ChunkedArray`] from a [`Series`].
    ///
    /// # Errors
    ///
    /// Returns an error if the Series dtype does not match.
    fn extract_ca(series: &Series) -> Result<&ChunkedArray<Self::PolarsType>, FerrayError>;
}

macro_rules! impl_polars_element {
    ($rust_ty:ty, $polars_ty:ty, $extractor:ident) => {
        impl PolarsElement for $rust_ty {
            type PolarsType = $polars_ty;

            fn extract_ca(series: &Series) -> Result<&ChunkedArray<Self::PolarsType>, FerrayError> {
                series.$extractor().map_err(|e| {
                    FerrayError::invalid_dtype(format!("Polars Series dtype mismatch: {e}"))
                })
            }
        }
    };
}

impl_polars_element!(u8, UInt8Type, u8);
impl_polars_element!(u16, UInt16Type, u16);
impl_polars_element!(u32, UInt32Type, u32);
impl_polars_element!(u64, UInt64Type, u64);
impl_polars_element!(i8, Int8Type, i8);
impl_polars_element!(i16, Int16Type, i16);
impl_polars_element!(i32, Int32Type, i32);
impl_polars_element!(i64, Int64Type, i64);
impl_polars_element!(f32, Float32Type, f32);
impl_polars_element!(f64, Float64Type, f64);

// ---------------------------------------------------------------------------
// ferray -> Polars  (REQ-6)
// ---------------------------------------------------------------------------

/// Extension trait for converting a ferray array to a Polars [`Series`].
///
/// Multi-dimensional inputs flatten to a 1-D Series in row-major (C)
/// order — a Polars Series is inherently a single column, so shape
/// information is dropped on the way out. Callers who need to
/// recover an N-D ferray array should `.reshape()` after
/// [`FromPolars::from_polars_series`] (#307).
pub trait ToPolars {
    /// Convert this ferray array to a Polars [`Series`] with the given name.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the element type has no
    /// Polars equivalent.
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrayError>;
}

impl<T, D> ToPolars for ferray_core::Array<T, D>
where
    T: PolarsElement,
    D: ferray_core::dimension::Dimension,
    ChunkedArray<T::PolarsType>: IntoSeries,
{
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrayError> {
        // Validate dtype mapping exists
        let _ = dtype_map::dtype_to_polars(self.dtype())?;

        let data: Vec<T> = self.to_vec_flat();
        let ca = ChunkedArray::<T::PolarsType>::from_slice(name.into(), &data);
        Ok(ca.into_series())
    }
}

/// Extension trait for converting a ferray boolean array to a Polars
/// [`Series`]. Multi-dimensional inputs flatten to 1-D in row-major
/// order (#307).
///
/// # Why a separate trait? (#311)
///
/// Polars represents booleans as a [`BooleanChunked`], not as a
/// `ChunkedArray<BooleanType>`-via-`PolarsNumericType`. The
/// machinery used by [`ToPolars`] dispatches through
/// `PolarsNumericType::Native`, which has no boolean slot, so bool
/// gets its own parallel trait that builds the `BooleanChunked`
/// directly. Unifying with [`ToPolars`] would require an additional
/// abstraction over "numeric vs. boolean", which is more API
/// surface than the savings warrant — the Arrow side uses the same
/// trait split for the same reason.
pub trait ToPolarsBool {
    /// Convert this ferray bool array to a Polars [`Series`].
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] on internal inconsistency.
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrayError>;
}

impl<D: ferray_core::dimension::Dimension> ToPolarsBool for ferray_core::Array<bool, D> {
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrayError> {
        let data: Vec<bool> = self.to_vec_flat();
        let ca = BooleanChunked::new(name.into(), &data);
        Ok(ca.into_series())
    }
}

// ---------------------------------------------------------------------------
// Polars -> ferray  (REQ-7)
// ---------------------------------------------------------------------------

/// Extension trait for converting a Polars [`Series`] to a ferray 1-D array.
pub trait FromPolars<T: Element>: Sized {
    /// Convert a Polars Series to a ferray `Array1<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the Series dtype does not
    /// match `T`, or [`FerrayError::InvalidValue`] if the Series contains
    /// null values.
    fn into_ferray(self) -> Result<Array1<T>, FerrayError>;
}

impl<T: PolarsElement> FromPolars<T> for Series {
    fn into_ferray(self) -> Result<Array1<T>, FerrayError> {
        // Validate dtype
        let polars_dt = self.dtype().clone();
        let ferray_dt = dtype_map::polars_to_dtype(&polars_dt)?;
        if ferray_dt != T::dtype() {
            return Err(FerrayError::invalid_dtype(format!(
                "Polars Series has dtype {polars_dt:?} (ferray {ferray_dt}), but requested {}",
                T::dtype()
            )));
        }

        // Check for nulls
        if self.null_count() > 0 {
            return Err(FerrayError::invalid_value(format!(
                "Polars Series contains {} null values; ferray arrays do not support nulls",
                self.null_count()
            )));
        }

        let ca = T::extract_ca(&self)?;
        let data: Vec<T> = ca.into_no_null_iter().collect();
        let len = data.len();
        Array1::<T>::from_vec(Ix1::new([len]), data)
    }
}

/// Extension trait for converting a Polars [`Series`] with boolean dtype
/// to a ferray `Array1<bool>`.
///
/// # Why a separate trait? (#311)
///
/// Same rationale as [`ToPolarsBool`]: Polars' bool path goes
/// through `BooleanChunked`, not through a numeric `ChunkedArray<T>`,
/// so the [`FromPolars`] trait machinery (parameterized over
/// `PolarsNumericType`) doesn't fit. Hence a separate FromPolars-
/// shaped trait for the bool conversion.
pub trait FromPolarsBool: Sized {
    /// Convert a Polars boolean Series to a ferray `Array1<bool>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrayError::InvalidDtype`] if the Series is not boolean,
    /// or [`FerrayError::InvalidValue`] if it contains nulls.
    fn into_ferray_bool(self) -> Result<Array1<bool>, FerrayError>;
}

impl FromPolarsBool for Series {
    fn into_ferray_bool(self) -> Result<Array1<bool>, FerrayError> {
        if *self.dtype() != DataType::Boolean {
            return Err(FerrayError::invalid_dtype(format!(
                "expected Boolean Series, got {:?}",
                self.dtype()
            )));
        }

        if self.null_count() > 0 {
            return Err(FerrayError::invalid_value(format!(
                "Polars Series contains {} null values; ferray arrays do not support nulls",
                self.null_count()
            )));
        }

        let ca = self.bool().map_err(|e| {
            FerrayError::invalid_dtype(format!("failed to extract BooleanChunked: {e}"))
        })?;
        let data: Vec<bool> = ca.into_no_null_iter().collect();
        let len = data.len();
        Array1::<bool>::from_vec(Ix1::new([len]), data)
    }
}

// ---------------------------------------------------------------------------
// 2-D interop via Polars DataFrame (#549)
// ---------------------------------------------------------------------------
//
// A Polars `DataFrame` is a heterogeneous collection of named 1-D
// `Series`. The natural 2-D ferray mapping is "one column of the array
// per Series", which lets callers hand an `Array2<f64>` to a data
// pipeline that expects a DataFrame without any intermediate pandas-
// like shuffling.
//
// Only f32/f64/integer columns are supported here; mixed-dtype
// DataFrames need a per-column cast on the caller side.

use ferray_core::array::aliases::Array2;
use ferray_core::dimension::Ix2;

/// Convert an `Array2<T>` into a Polars [`DataFrame`] with one `Series`
/// per column of the input array. The `column_names` slice must have
/// exactly `a.shape()[1]` entries.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `column_names.len() != ncols`.
/// - `FerrayError::InvalidDtype` on dtype-map failure.
pub fn array2_to_polars_dataframe<T>(
    a: &Array2<T>,
    column_names: &[&str],
) -> Result<DataFrame, FerrayError>
where
    T: PolarsElement + Copy,
    ChunkedArray<T::PolarsType>: IntoSeries,
{
    let shape = a.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    if column_names.len() != ncols {
        return Err(FerrayError::shape_mismatch(format!(
            "array2_to_polars_dataframe: got {} column names but array has {ncols} columns",
            column_names.len()
        )));
    }
    let _ = dtype_map::dtype_to_polars(a.dtype())?;

    let mut columns: Vec<Column> = Vec::with_capacity(ncols);
    for c in 0..ncols {
        // Extract column c into a fresh Vec<T> in row order.
        let mut col: Vec<T> = Vec::with_capacity(nrows);
        for r in 0..nrows {
            if let Some(slice) = a.as_slice() {
                col.push(slice[r * ncols + c]);
            } else {
                col.push(*a.iter().nth(r * ncols + c).unwrap());
            }
        }
        // Build the Column via Array1 → Series → Column.
        let col_arr = Array1::<T>::from_vec(Ix1::new([nrows]), col)?;
        let series = col_arr.to_polars_series(column_names[c])?;
        columns.push(series.into_column());
    }

    DataFrame::new(columns)
        .map_err(|e| FerrayError::invalid_value(format!("DataFrame::new failed: {e}")))
}

/// Convert a Polars [`DataFrame`] back into an `Array2<T>` by stacking
/// its columns. Every column must have the same Polars dtype (which
/// must match `T`), the same length, and contain no nulls.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if columns have different lengths.
/// - `FerrayError::InvalidDtype` if any column's dtype doesn't match `T`.
/// - `FerrayError::InvalidValue` if any column contains nulls.
pub fn array2_from_polars_dataframe<T>(df: &DataFrame) -> Result<Array2<T>, FerrayError>
where
    T: PolarsElement + Copy,
{
    let ncols = df.width();
    if ncols == 0 {
        return Array2::<T>::from_vec(Ix2::new([0, 0]), Vec::new());
    }
    let nrows = df.height();

    // Extract each column as a typed ChunkedArray, collecting into row-major order.
    let mut per_col: Vec<Vec<T>> = Vec::with_capacity(ncols);
    for col_series in df.get_columns() {
        let series = col_series.as_materialized_series();
        if series.null_count() > 0 {
            return Err(FerrayError::invalid_value(format!(
                "column '{}' contains {} null values; ferray arrays do not support nulls",
                col_series.name(),
                series.null_count()
            )));
        }
        let ca = T::extract_ca(series)?;
        let values: Vec<T> = ca.into_no_null_iter().collect();
        if values.len() != nrows {
            return Err(FerrayError::shape_mismatch(format!(
                "column '{}' has length {} but DataFrame height is {nrows}",
                col_series.name(),
                values.len()
            )));
        }
        per_col.push(values);
    }

    // Interleave into row-major order.
    let mut data: Vec<T> = Vec::with_capacity(nrows * ncols);
    for r in 0..nrows {
        for c in &per_col {
            data.push(c[r]);
        }
    }
    Array2::<T>::from_vec(Ix2::new([nrows, ncols]), data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unreadable_literal)] // Roundtrip tests assert exact equality on hand-picked Polars values.
mod tests {
    use super::*;

    macro_rules! test_roundtrip {
        ($name:ident, $ty:ty, $values:expr) => {
            #[test]
            fn $name() {
                let data: Vec<$ty> = $values;
                let len = data.len();
                let arr = Array1::<$ty>::from_vec(Ix1::new([len]), data.clone()).unwrap();

                let series = arr.to_polars_series("test").unwrap();
                assert_eq!(series.len(), len);

                let back: Array1<$ty> = series.into_ferray().unwrap();
                assert_eq!(back.shape(), &[len]);
                assert_eq!(back.as_slice().unwrap(), &data[..]);
            }
        };
    }

    test_roundtrip!(roundtrip_f64, f64, vec![1.0, 2.5, -4.75, 0.0]);
    test_roundtrip!(roundtrip_f32, f32, vec![1.0f32, -2.5, 0.0]);
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

        let series = arr.to_polars_series("flags").unwrap();
        assert_eq!(series.len(), len);
        assert_eq!(*series.dtype(), DataType::Boolean);

        let back = series.into_ferray_bool().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn empty_series_roundtrip() {
        let arr = Array1::<f64>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let series = arr.to_polars_series("empty").unwrap();
        assert_eq!(series.len(), 0);

        let back: Array1<f64> = series.into_ferray().unwrap();
        assert_eq!(back.shape(), &[0]);
    }

    #[test]
    fn series_name_preserved() {
        let arr = Array1::<i32>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let series = arr.to_polars_series("my_column").unwrap();
        assert_eq!(series.name().as_str(), "my_column");
    }

    #[test]
    fn dtype_mismatch_rejected() {
        // Create an i32 series and try to extract as f64
        let series = Series::new("test".into(), &[1i32, 2, 3]);
        let result: Result<Array1<f64>, _> = series.into_ferray();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("dtype") || msg.contains("mismatch"),
            "expected dtype error, got: {msg}"
        );
    }

    #[test]
    fn series_with_nulls_rejected() {
        let series = Series::new("test".into(), &[Some(1.0f64), None, Some(3.0)]);
        let result: Result<Array1<f64>, _> = series.into_ferray();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("null"), "expected null error, got: {msg}");
    }

    #[test]
    fn bool_series_with_nulls_rejected() {
        let series = Series::new("test".into(), &[Some(true), None, Some(false)]);
        let result = series.into_ferray_bool();
        assert!(result.is_err());
    }

    #[test]
    fn non_bool_series_into_ferray_bool_rejected() {
        let series = Series::new("test".into(), &[1i32, 2, 3]);
        let result = series.into_ferray_bool();
        assert!(result.is_err());
    }

    #[test]
    fn bit_identical_f64_roundtrip() {
        // AC (#198): Polars roundtrip must be bit-identical for every
        // f64 corner case that the Arrow roundtrip already covers, incl.
        // NaN. Previously `f64::NAN` was omitted here, hiding the fact
        // that bit-preservation through Polars was never verified for
        // non-finite values. Matches arrow_conv::tests::bit_identical_roundtrip.
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
        let series = arr.to_polars_series("precise").unwrap();
        let back: Array1<f64> = series.into_ferray().unwrap();

        let back_slice = back.as_slice().unwrap();
        for (i, (a, b)) in original.iter().zip(back_slice.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn nan_f64_roundtrip_preserves_nan_ness() {
        // Weaker but more portable check: Polars may canonicalize the
        // exact NaN payload on roundtrip, which is allowed by IEEE 754.
        // At minimum the result must still test is_nan().
        let original: Vec<f64> = vec![f64::NAN, 1.0, f64::NAN];
        let arr = Array1::<f64>::from_vec(Ix1::new([3]), original).unwrap();
        let series = arr.to_polars_series("nans").unwrap();
        let back: Array1<f64> = series.into_ferray().unwrap();
        let back_slice = back.as_slice().unwrap();
        assert!(back_slice[0].is_nan(), "index 0 should still be NaN");
        assert_eq!(back_slice[1], 1.0);
        assert!(back_slice[2].is_nan(), "index 2 should still be NaN");
    }

    // ----- 2-D DataFrame interop (#549) -----

    #[test]
    fn array2_to_polars_dataframe_f64() {
        // shape (3, 2) — 3 rows, 2 columns
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Array2::<f64>::from_vec(Ix2::new([3, 2]), data).unwrap();
        let df = array2_to_polars_dataframe(&a, &["x", "y"]).unwrap();
        assert_eq!(df.width(), 2);
        assert_eq!(df.height(), 3);
        // Column "x" = rows[0] across all 3 rows = [1, 3, 5]
        let col_x: Vec<f64> = df
            .column("x")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(col_x, vec![1.0, 3.0, 5.0]);
        let col_y: Vec<f64> = df
            .column("y")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(col_y, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn array2_polars_dataframe_roundtrip_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Array2::<f64>::from_vec(Ix2::new([4, 2]), data.clone()).unwrap();
        let df = array2_to_polars_dataframe(&a, &["c0", "c1"]).unwrap();
        let back = array2_from_polars_dataframe::<f64>(&df).unwrap();
        assert_eq!(back.shape(), &[4, 2]);
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn array2_polars_dataframe_roundtrip_i32() {
        let data = vec![1i32, 2, 3, 4, 5, 6];
        let a = Array2::<i32>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
        let df = array2_to_polars_dataframe(&a, &["a", "b", "c"]).unwrap();
        let back = array2_from_polars_dataframe::<i32>(&df).unwrap();
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn array2_to_polars_dataframe_wrong_column_count_errors() {
        let a = Array2::<f64>::from_vec(Ix2::new([2, 3]), vec![0.0; 6]).unwrap();
        // 3 columns but only 2 names provided.
        assert!(array2_to_polars_dataframe(&a, &["x", "y"]).is_err());
    }

    #[test]
    fn array2_from_polars_dataframe_empty_is_empty() {
        let df = DataFrame::empty();
        let a = array2_from_polars_dataframe::<f64>(&df).unwrap();
        assert_eq!(a.shape(), &[0, 0]);
    }

    #[test]
    fn array2_from_polars_dataframe_wrong_dtype_errors() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let a = Array2::<f64>::from_vec(Ix2::new([2, 2]), data).unwrap();
        let df = array2_to_polars_dataframe(&a, &["c0", "c1"]).unwrap();
        // Request the wrong type on the way back.
        assert!(array2_from_polars_dataframe::<i32>(&df).is_err());
    }

    // ----- ToPolars now generic over Dimension (#307) -------------------

    #[test]
    fn to_polars_series_from_array2_flattens_row_major() {
        use ferray_core::Array;
        use ferray_core::dimension::Ix2;
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
        let series = arr.to_polars_series("flat").unwrap();
        assert_eq!(series.len(), 6);
        let ca = series.f64().unwrap();
        let got: Vec<f64> = ca.into_no_null_iter().collect();
        assert_eq!(got, data);
    }

    #[test]
    fn to_polars_series_from_arrayd_flattens() {
        use ferray_core::dimension::IxDyn;
        let data: Vec<i64> = (0..24).collect();
        let arr = ferray_core::Array::<i64, IxDyn>::from_vec(IxDyn::new(&[2, 3, 4]), data.clone())
            .unwrap();
        let series = arr.to_polars_series("flat3d").unwrap();
        assert_eq!(series.len(), 24);
        let ca = series.i64().unwrap();
        let got: Vec<i64> = ca.into_no_null_iter().collect();
        assert_eq!(got, data);
    }

    #[test]
    fn to_polars_bool_from_array2_flattens() {
        use ferray_core::Array;
        use ferray_core::dimension::Ix2;
        let data = vec![true, false, false, true, false, true];
        let arr = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
        let series = arr.to_polars_series("bools").unwrap();
        assert_eq!(series.len(), 6);
        let ca = series.bool().unwrap();
        let got: Vec<bool> = ca.into_no_null_iter().collect();
        assert_eq!(got, data);
    }
}
