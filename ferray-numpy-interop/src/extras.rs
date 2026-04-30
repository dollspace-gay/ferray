// ferray-numpy-interop: high-level helpers that round out the parity gaps.
//
// This module sits above the per-backend conversion modules and provides:
//   - DynArray <-> Arrow / NumPy dispatch (#644)
//   - Arrow RecordBatch from named ferray columns (#648)
//   - Polars DataFrame from named ferray columns (#647)
//   - Null-bitmap interop: build Arrow arrays with a validity buffer from
//     a ferray Array + Vec<bool> mask, and read them back (#646). The
//     Vec<bool> primitive lets ferray-ma callers route through this
//     module without ferray-numpy-interop taking a hard ferray-ma dep.
//   - bool round-trip helpers (#643)
//
// What is intentionally NOT covered here (still tracked as deferred):
//   - true zero-copy buffer sharing across the FFI boundary (large
//     refcount handshake design change)
//   - Arrow Float16 ↔ ferray F16 round-trip via PrimitiveArray<Float16Type>
//     (the half-rs / arrow type-system glue lives upstream and ferray's
//     ArrowElement trait would need a Float16-aware impl_arrow_element
//     branch keyed on `cfg(feature = "f16")`)

#[cfg(feature = "arrow")]
use ferray_core::{Array, DType, DynArray, FerrayError, FerrayResult, Ix1};

// ===========================================================================
// DynArray <-> Arrow dispatch (#644)
// ===========================================================================

/// Convert a [`DynArray`] (runtime-typed ferray array) to an Arrow
/// [`arrow::array::ArrayRef`] by dispatching on the dtype. The result is
/// always 1-D — Arrow's columnar model collapses higher-rank ferray
/// arrays to a flat sequence; callers that need to preserve shape should
/// pass shape metadata alongside (e.g. via [`array2_to_record_batch`]).
///
/// # Errors
/// - `FerrayError::InvalidDtype` for dtypes Arrow doesn't model
///   (Complex32/64, U128/I128, F16 without the `f16` feature).
#[cfg(feature = "arrow")]
pub fn dynarray_to_arrow(arr: &DynArray) -> FerrayResult<arrow::array::ArrayRef> {
    use arrow::array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };
    use std::sync::Arc;

    macro_rules! flat_typed {
        ($variant:ident, $arrow_array:ty, $rust:ty) => {{
            let typed = arr.clone().$variant()?;
            let data: Vec<$rust> = typed.iter().copied().collect();
            let a: $arrow_array = data.into();
            Arc::new(a) as ArrayRef
        }};
    }

    // DynArray currently exposes try_into accessors only for the
    // dtypes shipped at the public boundary: bool, f32, f64, i32, i64.
    // The remaining integer widths (i8/i16/u8/u16/u32/u64) need a wider
    // DynArray API, which is the structural part of #644 — we surface
    // them as InvalidDtype with a clear message until the core gets
    // those accessors.
    let result: ArrayRef = match arr.dtype() {
        DType::Bool => {
            let typed = arr.clone().try_into_bool()?;
            let data: Vec<bool> = typed.iter().copied().collect();
            let a: BooleanArray = data.into();
            Arc::new(a) as ArrayRef
        }
        DType::I32 => flat_typed!(try_into_i32, Int32Array, i32),
        DType::I64 => flat_typed!(try_into_i64, Int64Array, i64),
        DType::F32 => flat_typed!(try_into_f32, Float32Array, f32),
        DType::F64 => flat_typed!(try_into_f64, Float64Array, f64),
        // datetime64 / timedelta64: unwrap to the i64 storage and emit
        // an Arrow Timestamp / Duration of the matching unit. The
        // schema-side mapping is in dtype_map::dtype_to_arrow.
        DType::DateTime64(unit) => {
            use arrow::array::TimestampMicrosecondArray;
            use arrow::array::TimestampMillisecondArray;
            use arrow::array::TimestampNanosecondArray;
            use arrow::array::TimestampSecondArray;
            use ferray_core::dtype::TimeUnit;
            let (typed, _) = arr.clone().try_into_datetime64()?;
            let data: Vec<i64> = typed.iter().map(|v| v.0).collect();
            match unit {
                TimeUnit::Ns => Arc::new(TimestampNanosecondArray::from(data)) as ArrayRef,
                TimeUnit::Us => Arc::new(TimestampMicrosecondArray::from(data)) as ArrayRef,
                TimeUnit::Ms => Arc::new(TimestampMillisecondArray::from(data)) as ArrayRef,
                TimeUnit::S => Arc::new(TimestampSecondArray::from(data)) as ArrayRef,
                other => {
                    return Err(FerrayError::invalid_dtype(format!(
                        "dynarray_to_arrow: Arrow has no Timestamp variant for TimeUnit::{other:?}"
                    )));
                }
            }
        }
        DType::Timedelta64(unit) => {
            use arrow::array::DurationMicrosecondArray;
            use arrow::array::DurationMillisecondArray;
            use arrow::array::DurationNanosecondArray;
            use arrow::array::DurationSecondArray;
            use ferray_core::dtype::TimeUnit;
            let (typed, _) = arr.clone().try_into_timedelta64()?;
            let data: Vec<i64> = typed.iter().map(|v| v.0).collect();
            match unit {
                TimeUnit::Ns => Arc::new(DurationNanosecondArray::from(data)) as ArrayRef,
                TimeUnit::Us => Arc::new(DurationMicrosecondArray::from(data)) as ArrayRef,
                TimeUnit::Ms => Arc::new(DurationMillisecondArray::from(data)) as ArrayRef,
                TimeUnit::S => Arc::new(DurationSecondArray::from(data)) as ArrayRef,
                other => {
                    return Err(FerrayError::invalid_dtype(format!(
                        "dynarray_to_arrow: Arrow has no Duration variant for TimeUnit::{other:?}"
                    )));
                }
            }
        }
        other => {
            return Err(FerrayError::invalid_dtype(format!(
                "dynarray_to_arrow: dtype {other} not yet supported \
                 (DynArray accessor missing — tracked under #644 / ferray-core widening)"
            )));
        }
    };
    // Suppress unused-import warnings for the array types we don't dispatch
    // to until DynArray gains the corresponding accessors.
    let _ = (
        std::marker::PhantomData::<UInt8Array>,
        std::marker::PhantomData::<UInt16Array>,
        std::marker::PhantomData::<UInt32Array>,
        std::marker::PhantomData::<UInt64Array>,
        std::marker::PhantomData::<Int8Array>,
        std::marker::PhantomData::<Int16Array>,
    );
    Ok(result)
}

/// Construct an Arrow [`arrow::record_batch::RecordBatch`] from a list of
/// `(column_name, DynArray)` pairs.
///
/// All columns must have the same length (Arrow's RecordBatch contract).
/// The resulting batch's schema is inferred from each column's dtype via
/// [`crate::dtype_map::dtype_to_arrow`].
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if columns have inconsistent lengths.
/// - `FerrayError::InvalidDtype` if any column has a dtype Arrow can't
///   represent.
#[cfg(feature = "arrow")]
pub fn record_batch_from_columns(
    columns: &[(&str, &DynArray)],
) -> FerrayResult<arrow::record_batch::RecordBatch> {
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    if columns.is_empty() {
        return Err(FerrayError::invalid_value(
            "record_batch_from_columns: at least one column is required",
        ));
    }
    let n = columns[0].1.size();
    for (name, arr) in columns {
        if arr.size() != n {
            return Err(FerrayError::shape_mismatch(format!(
                "record_batch_from_columns: column {name:?} has {} elements, expected {n}",
                arr.size()
            )));
        }
    }
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays = Vec::with_capacity(columns.len());
    for (name, arr) in columns {
        let arrow_dtype = crate::dtype_map::dtype_to_arrow(arr.dtype())?;
        fields.push(Field::new(*name, arrow_dtype, false));
        arrays.push(dynarray_to_arrow(arr)?);
    }
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| {
        FerrayError::invalid_value(format!(
            "record_batch_from_columns: RecordBatch::try_new failed: {e}"
        ))
    })
}

/// Decompose an Arrow [`RecordBatch`] into a `Vec<(name, DynArray)>`.
///
/// Each column is converted to a [`DynArray`] of the appropriate dtype.
/// String, struct, list, and other complex Arrow types are surfaced as
/// `FerrayError::InvalidDtype`.
///
/// # Errors
/// - `FerrayError::InvalidDtype` for Arrow types ferray doesn't model.
#[cfg(feature = "arrow")]
pub fn record_batch_to_columns(
    batch: &arrow::record_batch::RecordBatch,
) -> FerrayResult<Vec<(String, DynArray)>> {
    use arrow::array::{
        Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };
    use ferray_core::IxDyn;

    let mut out = Vec::with_capacity(batch.num_columns());
    for (i, field) in batch.schema().fields().iter().enumerate() {
        let col = batch.column(i);
        let dt = crate::dtype_map::arrow_to_dtype(col.data_type())?;
        let n = col.len();
        let dyn_arr: DynArray = match dt {
            DType::Bool => {
                let arr = col
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| FerrayError::invalid_dtype("bool downcast failed"))?;
                let v: Vec<bool> = (0..n).map(|j| arr.value(j)).collect();
                let typed = ferray_core::Array::<bool, IxDyn>::from_vec(IxDyn::new(&[n]), v)?;
                typed.into()
            }
            DType::I32 => prim_to_dyn::<Int32Array, i32>(col, n)?,
            DType::I64 => prim_to_dyn::<Int64Array, i64>(col, n)?,
            DType::F32 => prim_to_dyn::<Float32Array, f32>(col, n)?,
            DType::F64 => prim_to_dyn::<Float64Array, f64>(col, n)?,
            other => {
                return Err(FerrayError::invalid_dtype(format!(
                    "record_batch_to_columns: column {} (dtype {other}) not yet supported \
                     (DynArray accessor missing — tracked under #644)",
                    field.name()
                )));
            }
        };
        // Suppress unused-import for the integer-array types we'd dispatch
        // to once #644 widens DynArray. The phantom block keeps the imports
        // live without contributing to the runtime path.
        let _ = (
            std::marker::PhantomData::<Int8Array>,
            std::marker::PhantomData::<Int16Array>,
            std::marker::PhantomData::<UInt8Array>,
            std::marker::PhantomData::<UInt16Array>,
            std::marker::PhantomData::<UInt32Array>,
            std::marker::PhantomData::<UInt64Array>,
        );
        out.push((field.name().clone(), dyn_arr));
    }
    Ok(out)
}

#[cfg(feature = "arrow")]
fn prim_to_dyn<A, T>(col: &arrow::array::ArrayRef, n: usize) -> FerrayResult<DynArray>
where
    A: arrow::array::Array + 'static,
    T: ferray_core::Element + Copy,
    for<'a> &'a A: IntoIterator<Item = Option<T>>,
    ferray_core::Array<T, ferray_core::IxDyn>: Into<DynArray>,
{
    let arr = col
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| FerrayError::invalid_dtype("primitive downcast failed"))?;
    let v: Vec<T> = arr
        .into_iter()
        .map(|opt| opt.unwrap_or_else(|| panic!("null in non-null Arrow column")))
        .collect();
    let _ = n; // length comes from the iterator
    let typed = ferray_core::Array::<T, ferray_core::IxDyn>::from_vec(
        ferray_core::IxDyn::new(&[v.len()]),
        v,
    )?;
    Ok(typed.into())
}

// ===========================================================================
// Polars DataFrame builders (#647)
// ===========================================================================

/// Build a Polars [`polars::frame::DataFrame`] from named ferray columns.
///
/// Accepts a slice of `(name, &DynArray)` pairs; each column becomes a
/// Polars Series of the corresponding dtype. All columns must have the
/// same length.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if columns have inconsistent lengths.
/// - `FerrayError::InvalidDtype` for dtypes Polars can't model.
#[cfg(feature = "polars")]
pub fn dataframe_from_columns(
    columns: &[(&str, &ferray_core::DynArray)],
) -> ferray_core::FerrayResult<polars::frame::DataFrame> {
    use ferray_core::DType;
    use polars::prelude::{DataFrame, NamedFrom, Series};

    if columns.is_empty() {
        return Err(ferray_core::FerrayError::invalid_value(
            "dataframe_from_columns: at least one column is required",
        ));
    }
    let n = columns[0].1.size();
    for (name, arr) in columns {
        if arr.size() != n {
            return Err(ferray_core::FerrayError::shape_mismatch(format!(
                "dataframe_from_columns: column {name:?} has {} elements, expected {n}",
                arr.size()
            )));
        }
    }
    let mut series_vec = Vec::with_capacity(columns.len());
    for (name, arr) in columns {
        let s = match arr.dtype() {
            DType::Bool => {
                let typed = (*arr).clone().try_into_bool()?;
                let v: Vec<bool> = typed.iter().copied().collect();
                Series::new((*name).into(), v)
            }
            DType::I32 => {
                let typed = (*arr).clone().try_into_i32()?;
                let v: Vec<i32> = typed.iter().copied().collect();
                Series::new((*name).into(), v)
            }
            DType::I64 => {
                let typed = (*arr).clone().try_into_i64()?;
                let v: Vec<i64> = typed.iter().copied().collect();
                Series::new((*name).into(), v)
            }
            DType::F32 => {
                let typed = (*arr).clone().try_into_f32()?;
                let v: Vec<f32> = typed.iter().copied().collect();
                Series::new((*name).into(), v)
            }
            DType::F64 => {
                let typed = (*arr).clone().try_into_f64()?;
                let v: Vec<f64> = typed.iter().copied().collect();
                Series::new((*name).into(), v)
            }
            other => {
                return Err(ferray_core::FerrayError::invalid_dtype(format!(
                    "dataframe_from_columns: dtype {other} not supported"
                )));
            }
        };
        series_vec.push(s.into());
    }
    DataFrame::new(series_vec).map_err(|e| {
        ferray_core::FerrayError::invalid_value(format!(
            "dataframe_from_columns: DataFrame::new failed: {e}"
        ))
    })
}

// ===========================================================================
// Null-bitmap interop (#646)
// ===========================================================================
//
// Arrow's columnar model carries a per-array "validity buffer": a packed
// bitmap where bit `i` = 1 means "value `i` is valid", and bit `i` = 0
// means "value `i` is null". ferray's MaskedArray uses an opposite
// convention — bit `i` = 1 means "value `i` is masked / invalid".
//
// These helpers route data + a Vec<bool> mask (ferray's convention,
// matching `MaskedArray::mask().iter().copied().collect()`) through to
// Arrow with the bits inverted, and read them back with the inversion
// undone. The Vec<bool> primitive lets ferray-ma callers use this
// interface without ferray-numpy-interop taking a hard ferray-ma
// dependency.

/// Build an Arrow primitive array from a 1-D ferray array and an optional
/// mask. `mask[i] = true` means the position should be marked null in the
/// resulting Arrow array; `None` means "no nulls" (all-valid).
///
/// Currently supports the dtypes that have NpElement coverage:
/// Bool/I32/I64/F32/F64. Other dtypes return InvalidDtype until
/// DynArray gains the corresponding accessors (#644).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `mask.len()` differs from the
///   array length.
/// - `FerrayError::InvalidDtype` for unsupported dtypes.
#[cfg(feature = "arrow")]
pub fn dynarray_to_arrow_with_mask(
    arr: &DynArray,
    mask: Option<&[bool]>,
) -> FerrayResult<arrow::array::ArrayRef> {
    use arrow::array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    };
    use std::sync::Arc;

    if let Some(m) = mask {
        if m.len() != arr.size() {
            return Err(FerrayError::shape_mismatch(format!(
                "dynarray_to_arrow_with_mask: mask length {} != array size {}",
                m.len(),
                arr.size()
            )));
        }
    }

    /// Build a `Vec<Option<T>>` from a typed ferray Array and a mask.
    /// Where `mask[i]` is `true`, we emit `None`. Otherwise `Some(value)`.
    fn paired<T: Copy>(values: &[T], mask: Option<&[bool]>) -> Vec<Option<T>> {
        match mask {
            None => values.iter().map(|v| Some(*v)).collect(),
            Some(m) => values
                .iter()
                .zip(m.iter())
                .map(|(v, &masked)| if masked { None } else { Some(*v) })
                .collect(),
        }
    }

    let result: ArrayRef = match arr.dtype() {
        DType::Bool => {
            let typed = arr.clone().try_into_bool()?;
            let v: Vec<bool> = typed.iter().copied().collect();
            let pairs = paired(&v, mask);
            let a: BooleanArray = pairs.into_iter().collect();
            Arc::new(a) as ArrayRef
        }
        DType::I32 => {
            let typed = arr.clone().try_into_i32()?;
            let v: Vec<i32> = typed.iter().copied().collect();
            let pairs = paired(&v, mask);
            let a: Int32Array = pairs.into_iter().collect();
            Arc::new(a) as ArrayRef
        }
        DType::I64 => {
            let typed = arr.clone().try_into_i64()?;
            let v: Vec<i64> = typed.iter().copied().collect();
            let pairs = paired(&v, mask);
            let a: Int64Array = pairs.into_iter().collect();
            Arc::new(a) as ArrayRef
        }
        DType::F32 => {
            let typed = arr.clone().try_into_f32()?;
            let v: Vec<f32> = typed.iter().copied().collect();
            let pairs = paired(&v, mask);
            let a: Float32Array = pairs.into_iter().collect();
            Arc::new(a) as ArrayRef
        }
        DType::F64 => {
            let typed = arr.clone().try_into_f64()?;
            let v: Vec<f64> = typed.iter().copied().collect();
            let pairs = paired(&v, mask);
            let a: Float64Array = pairs.into_iter().collect();
            Arc::new(a) as ArrayRef
        }
        other => {
            return Err(FerrayError::invalid_dtype(format!(
                "dynarray_to_arrow_with_mask: dtype {other} not yet supported \
                 (DynArray accessor missing — tracked under #644)"
            )));
        }
    };
    Ok(result)
}

/// Inverse of [`dynarray_to_arrow_with_mask`]: read an Arrow array and
/// return both the values (with masked positions filled by the type's
/// default) and the ferray-style mask (`true` = was null in the Arrow
/// validity buffer, `false` = valid).
///
/// Returns `(DynArray, Option<Vec<bool>>)`. The mask is `None` when the
/// Arrow array has no validity buffer at all; otherwise it's `Some(vec)`
/// of length `arr.len()`.
///
/// # Errors
/// - `FerrayError::InvalidDtype` for Arrow types ferray doesn't model.
#[cfg(feature = "arrow")]
pub fn arrow_to_dynarray_with_mask(
    arr: &arrow::array::ArrayRef,
) -> FerrayResult<(DynArray, Option<Vec<bool>>)> {
    use arrow::array::{
        Array as _, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    };
    use ferray_core::IxDyn;

    let dt = crate::dtype_map::arrow_to_dtype(arr.data_type())?;
    let n = arr.len();

    // Build the mask in ferray convention: true = was null in Arrow.
    let mask: Option<Vec<bool>> = arr
        .nulls()
        .map(|nb| (0..n).map(|i| !nb.is_valid(i)).collect::<Vec<bool>>());

    let dyn_arr: DynArray = match dt {
        DType::Bool => {
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| FerrayError::invalid_dtype("bool downcast failed"))?;
            // For null positions emit `false`; the mask carries the truth.
            let v: Vec<bool> = (0..n)
                .map(|i| if a.is_null(i) { false } else { a.value(i) })
                .collect();
            let typed = Array::<bool, IxDyn>::from_vec(IxDyn::new(&[n]), v)?;
            typed.into()
        }
        DType::I32 => prim_with_default::<Int32Array, i32>(arr, n)?,
        DType::I64 => prim_with_default::<Int64Array, i64>(arr, n)?,
        DType::F32 => prim_with_default::<Float32Array, f32>(arr, n)?,
        DType::F64 => prim_with_default::<Float64Array, f64>(arr, n)?,
        other => {
            return Err(FerrayError::invalid_dtype(format!(
                "arrow_to_dynarray_with_mask: dtype {other} not yet supported (#644)"
            )));
        }
    };
    Ok((dyn_arr, mask))
}

#[cfg(feature = "arrow")]
fn prim_with_default<A, T>(arr: &arrow::array::ArrayRef, n: usize) -> FerrayResult<DynArray>
where
    A: arrow::array::Array + 'static,
    T: ferray_core::Element + Copy + Default,
    for<'a> &'a A: IntoIterator<Item = Option<T>>,
    Array<T, ferray_core::IxDyn>: Into<DynArray>,
{
    use ferray_core::IxDyn;
    let typed_arr = arr
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| FerrayError::invalid_dtype("primitive downcast failed"))?;
    // Null entries get `T::default()` in the data buffer; the caller
    // should consult the mask for truth.
    let v: Vec<T> = typed_arr
        .into_iter()
        .map(|opt| opt.unwrap_or_default())
        .collect();
    let _ = n;
    let typed = Array::<T, IxDyn>::from_vec(IxDyn::new(&[v.len()]), v)?;
    Ok(typed.into())
}

/// Build an Arrow primitive array from a typed [`Array<T, Ix1>`] and a
/// matching `Vec<bool>` mask (true = null). Convenience wrapper for
/// callers that already hold a typed Array; goes through Arrow's
/// `from_iter` path so the resulting array carries a proper validity
/// buffer.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `mask.len() != arr.size()`.
#[cfg(feature = "arrow")]
pub fn array1_to_arrow_with_mask<T>(
    arr: &Array<T, Ix1>,
    mask: &[bool],
) -> FerrayResult<arrow::array::PrimitiveArray<<T as crate::arrow_conv::ArrowElement>::ArrowType>>
where
    T: crate::arrow_conv::ArrowElement,
    <T as crate::arrow_conv::ArrowElement>::ArrowType: arrow::array::ArrowPrimitiveType<Native = T>,
{
    if mask.len() != arr.size() {
        return Err(FerrayError::shape_mismatch(format!(
            "array1_to_arrow_with_mask: mask length {} != array size {}",
            mask.len(),
            arr.size()
        )));
    }
    let pairs: Vec<Option<T>> = arr
        .iter()
        .zip(mask.iter())
        .map(|(v, &masked)| if masked { None } else { Some(*v) })
        .collect();
    Ok(pairs.into_iter().collect())
}

// ===========================================================================
// Complex round-trip via Arrow FixedSizeList[2] (#642)
// ===========================================================================
//
// Arrow has no native complex dtype. The two community conventions are:
//   1. FixedSizeList<float, 2>  — a [real, imag] pair per value
//   2. Struct{ "re": float, "im": float }
//
// We implement (1) — it's the canonical numerical representation used by
// Arrow's geospatial / vector extensions and pandas' Arrow round-trip.
// `complex_to_arrow` writes f32/f64 complex arrays as FixedSizeList<2>
// and `arrow_to_complex` reads them back. The helper is gated on the
// `complex` cargo feature.

#[cfg(all(feature = "arrow", feature = "complex"))]
mod complex_arrow {
    use super::*;
    use arrow::array::{
        Array as _, ArrayRef, FixedSizeListArray, Float32Array, Float32Builder, Float64Array,
        Float64Builder,
    };
    use arrow::datatypes::{DataType, Field};
    use ferray_core::Ix1;
    use num_complex::Complex;
    use std::sync::Arc;

    /// Convert a 1-D `Array<Complex<f64>, Ix1>` to an Arrow
    /// FixedSizeList<Float64, 2> array. Each complex value emits two
    /// child floats: real then imaginary.
    ///
    /// # Errors
    /// Returns errors only if Arrow's builder rejects the values
    /// (should not happen for finite inputs).
    pub fn complex64_to_arrow(arr: &Array<Complex<f64>, Ix1>) -> FerrayResult<ArrayRef> {
        let n = arr.size();
        let mut child = Float64Builder::with_capacity(n * 2);
        for c in arr.iter() {
            child.append_value(c.re);
            child.append_value(c.im);
        }
        let values: Float64Array = child.finish();
        let field = Arc::new(Field::new("item", DataType::Float64, false));
        let list = FixedSizeListArray::try_new(field, 2, Arc::new(values), None).map_err(|e| {
            FerrayError::invalid_value(format!("complex64_to_arrow: build failed: {e}"))
        })?;
        Ok(Arc::new(list) as ArrayRef)
    }

    /// f32 variant of [`complex64_to_arrow`].
    ///
    /// # Errors
    /// Returns errors only on Arrow builder failures.
    pub fn complex32_to_arrow(arr: &Array<Complex<f32>, Ix1>) -> FerrayResult<ArrayRef> {
        let n = arr.size();
        let mut child = Float32Builder::with_capacity(n * 2);
        for c in arr.iter() {
            child.append_value(c.re);
            child.append_value(c.im);
        }
        let values: Float32Array = child.finish();
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let list = FixedSizeListArray::try_new(field, 2, Arc::new(values), None).map_err(|e| {
            FerrayError::invalid_value(format!("complex32_to_arrow: build failed: {e}"))
        })?;
        Ok(Arc::new(list) as ArrayRef)
    }

    /// Read an Arrow FixedSizeList<Float64, 2> back into
    /// `Array<Complex<f64>, Ix1>`.
    ///
    /// # Errors
    /// - `FerrayError::InvalidDtype` if the array isn't a
    ///   FixedSizeList<Float64, 2>.
    pub fn arrow_to_complex64(arr: &ArrayRef) -> FerrayResult<Array<Complex<f64>, Ix1>> {
        let list = arr
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                FerrayError::invalid_dtype(
                    "arrow_to_complex64: expected FixedSizeList<Float64, 2> array",
                )
            })?;
        if list.value_length() != 2 {
            return Err(FerrayError::invalid_dtype(format!(
                "arrow_to_complex64: list length {} != 2",
                list.value_length()
            )));
        }
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                FerrayError::invalid_dtype("arrow_to_complex64: child type must be Float64")
            })?;
        let n = list.len();
        let mut data: Vec<Complex<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let re = values.value(i * 2);
            let im = values.value(i * 2 + 1);
            data.push(Complex::new(re, im));
        }
        Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([n]), data)
    }

    /// f32 variant of [`arrow_to_complex64`].
    ///
    /// # Errors
    /// - `FerrayError::InvalidDtype` if the array isn't a
    ///   FixedSizeList<Float32, 2>.
    pub fn arrow_to_complex32(arr: &ArrayRef) -> FerrayResult<Array<Complex<f32>, Ix1>> {
        let list = arr
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                FerrayError::invalid_dtype(
                    "arrow_to_complex32: expected FixedSizeList<Float32, 2> array",
                )
            })?;
        if list.value_length() != 2 {
            return Err(FerrayError::invalid_dtype(format!(
                "arrow_to_complex32: list length {} != 2",
                list.value_length()
            )));
        }
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                FerrayError::invalid_dtype("arrow_to_complex32: child type must be Float32")
            })?;
        let n = list.len();
        let mut data: Vec<Complex<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let re = values.value(i * 2);
            let im = values.value(i * 2 + 1);
            data.push(Complex::new(re, im));
        }
        Array::<Complex<f32>, Ix1>::from_vec(Ix1::new([n]), data)
    }
}

#[cfg(all(feature = "arrow", feature = "complex"))]
pub use complex_arrow::{
    arrow_to_complex32, arrow_to_complex64, complex32_to_arrow, complex64_to_arrow,
};

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(all(test, feature = "arrow"))]
mod arrow_tests {
    use super::*;
    use arrow::array::Array as _;
    use ferray_core::{Array, IxDyn};

    #[test]
    fn dynarray_to_arrow_f64_roundtrip() {
        let arr: Array<f64, IxDyn> =
            Array::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let arrow_ref = dynarray_to_arrow(&dyn_arr).unwrap();
        assert_eq!(arrow_ref.len(), 3);
        let f64_arr = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert_eq!(f64_arr.value(0), 1.0);
        assert_eq!(f64_arr.value(2), 3.0);
    }

    #[test]
    fn dynarray_to_arrow_bool_roundtrip() {
        let arr: Array<bool, IxDyn> =
            Array::from_vec(IxDyn::new(&[3]), vec![true, false, true]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let arrow_ref = dynarray_to_arrow(&dyn_arr).unwrap();
        let b = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .unwrap();
        assert_eq!(b.len(), 3);
        assert!(b.value(0) && !b.value(1) && b.value(2));
    }

    #[test]
    fn record_batch_roundtrip() {
        let i_arr: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![10, 20, 30]).unwrap();
        let f_arr: Array<f64, IxDyn> =
            Array::from_vec(IxDyn::new(&[3]), vec![1.5, 2.5, 3.5]).unwrap();
        let i_dyn: DynArray = i_arr.into();
        let f_dyn: DynArray = f_arr.into();
        let batch = record_batch_from_columns(&[("ints", &i_dyn), ("floats", &f_dyn)]).unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let back = record_batch_to_columns(&batch).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, "ints");
        assert_eq!(back[0].1.dtype(), DType::I32);
        assert_eq!(back[1].0, "floats");
        assert_eq!(back[1].1.dtype(), DType::F64);
    }

    #[test]
    fn record_batch_inconsistent_lengths_errs() {
        let a: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[2]), vec![1, 2]).unwrap();
        let b: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![3, 4, 5]).unwrap();
        let a_dyn: DynArray = a.into();
        let b_dyn: DynArray = b.into();
        assert!(record_batch_from_columns(&[("a", &a_dyn), ("b", &b_dyn)]).is_err());
    }

    // ---- Null-bitmap interop tests ----

    #[test]
    fn dynarray_to_arrow_with_mask_no_nulls_when_mask_is_none() {
        let arr: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![1, 2, 3]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let arrow_ref = dynarray_to_arrow_with_mask(&dyn_arr, None).unwrap();
        let typed = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(typed.null_count(), 0);
        assert_eq!(typed.value(0), 1);
        assert_eq!(typed.value(2), 3);
    }

    #[test]
    fn dynarray_to_arrow_with_mask_emits_nulls() {
        let arr: Array<f64, IxDyn> =
            Array::from_vec(IxDyn::new(&[4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let mask = vec![false, true, false, true];
        let arrow_ref = dynarray_to_arrow_with_mask(&dyn_arr, Some(&mask)).unwrap();
        let typed = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert_eq!(typed.null_count(), 2);
        assert!(typed.is_valid(0));
        assert!(typed.is_null(1));
        assert!(typed.is_valid(2));
        assert!(typed.is_null(3));
        assert_eq!(typed.value(0), 10.0);
        assert_eq!(typed.value(2), 30.0);
    }

    #[test]
    fn arrow_to_dynarray_with_mask_recovers_null_positions() {
        // Round-trip: ferray array + mask → Arrow with nulls → ferray + mask
        let arr: Array<i64, IxDyn> =
            Array::from_vec(IxDyn::new(&[5]), vec![100, 200, 300, 400, 500]).unwrap();
        let original: DynArray = arr.into();
        let mask = vec![false, false, true, false, true];

        let arrow_ref = dynarray_to_arrow_with_mask(&original, Some(&mask)).unwrap();
        let (back, recovered_mask) = arrow_to_dynarray_with_mask(&arrow_ref).unwrap();

        assert_eq!(back.shape(), &[5]);
        assert_eq!(back.dtype(), DType::I64);
        assert_eq!(recovered_mask.as_deref(), Some(&mask[..]));

        let typed = back.try_into_i64().unwrap();
        let v: Vec<i64> = typed.iter().copied().collect();
        // Null positions get the type default (0); the mask carries truth.
        assert_eq!(v, vec![100, 200, 0, 400, 0]);
    }

    #[test]
    fn arrow_to_dynarray_with_mask_returns_none_for_no_validity_buffer() {
        let arr: Array<f32, IxDyn> =
            Array::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let arrow_ref = dynarray_to_arrow_with_mask(&dyn_arr, None).unwrap();
        let (_back, mask) = arrow_to_dynarray_with_mask(&arrow_ref).unwrap();
        assert!(mask.is_none());
    }

    #[test]
    fn dynarray_to_arrow_with_mask_length_mismatch_errs() {
        let arr: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![1, 2, 3]).unwrap();
        let dyn_arr: DynArray = arr.into();
        assert!(dynarray_to_arrow_with_mask(&dyn_arr, Some(&[false, true])).is_err());
    }

    #[test]
    fn array1_to_arrow_with_mask_typed_path() {
        use ferray_core::Ix1;
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let mask = vec![false, true, false];
        let prim = array1_to_arrow_with_mask(&arr, &mask).unwrap();
        assert_eq!(prim.null_count(), 1);
        assert!(prim.is_valid(0));
        assert!(prim.is_null(1));
        assert!(prim.is_valid(2));
    }

    #[test]
    fn datetime64_dynarray_to_arrow_emits_timestamp() {
        use ferray_core::dtype::{DateTime64, TimeUnit};
        let arr = Array::<DateTime64, IxDyn>::from_vec(
            IxDyn::new(&[3]),
            vec![
                DateTime64(0),
                DateTime64(1_700_000_000_000_000_000),
                DateTime64(-1),
            ],
        )
        .unwrap();
        let dyn_arr = DynArray::from_datetime64(arr, TimeUnit::Ns);
        let arrow_ref = dynarray_to_arrow(&dyn_arr).unwrap();
        // Validate the Arrow array is a TimestampNanosecondArray with the
        // right values.
        let ts = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::TimestampNanosecondArray>()
            .expect("expected TimestampNanosecondArray");
        assert_eq!(ts.value(0), 0);
        assert_eq!(ts.value(1), 1_700_000_000_000_000_000);
        assert_eq!(ts.value(2), -1);
    }

    #[test]
    fn timedelta64_dynarray_to_arrow_emits_duration() {
        use ferray_core::dtype::{TimeUnit, Timedelta64};
        let arr = Array::<Timedelta64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![Timedelta64(60_000), Timedelta64(120_000)],
        )
        .unwrap();
        let dyn_arr = DynArray::from_timedelta64(arr, TimeUnit::Us);
        let arrow_ref = dynarray_to_arrow(&dyn_arr).unwrap();
        let dur = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::DurationMicrosecondArray>()
            .expect("expected DurationMicrosecondArray");
        assert_eq!(dur.value(0), 60_000);
        assert_eq!(dur.value(1), 120_000);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn complex64_arrow_roundtrip() {
        use ferray_core::Ix1;
        use num_complex::Complex;
        let original: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.5, 4.25),
            Complex::new(0.0, -1.0),
        ];
        let arr = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([3]), original.clone()).unwrap();
        let arrow_ref = complex64_to_arrow(&arr).unwrap();
        let back = arrow_to_complex64(&arrow_ref).unwrap();
        let v: Vec<Complex<f64>> = back.iter().copied().collect();
        assert_eq!(v, original);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn complex32_arrow_roundtrip() {
        use ferray_core::Ix1;
        use num_complex::Complex;
        let original: Vec<Complex<f32>> = vec![Complex::new(1.5, 2.5), Complex::new(-3.0, 4.0)];
        let arr = Array::<Complex<f32>, Ix1>::from_vec(Ix1::new([2]), original.clone()).unwrap();
        let arrow_ref = complex32_to_arrow(&arr).unwrap();
        let back = arrow_to_complex32(&arrow_ref).unwrap();
        let v: Vec<Complex<f32>> = back.iter().copied().collect();
        assert_eq!(v, original);
    }

    #[cfg(feature = "complex")]
    #[test]
    fn arrow_to_complex_rejects_non_fsl() {
        let arr: Array<f64, IxDyn> = Array::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let arrow_ref = dynarray_to_arrow(&dyn_arr).unwrap();
        // Plain Float64Array — not FixedSizeList<2>.
        assert!(arrow_to_complex64(&arrow_ref).is_err());
    }

    #[test]
    fn dynarray_to_arrow_with_mask_bool_emits_nulls() {
        let arr: Array<bool, IxDyn> =
            Array::from_vec(IxDyn::new(&[3]), vec![true, false, true]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let mask = vec![false, true, false];
        let arrow_ref = dynarray_to_arrow_with_mask(&dyn_arr, Some(&mask)).unwrap();
        let typed = arrow_ref
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .unwrap();
        assert_eq!(typed.null_count(), 1);
        assert!(typed.is_valid(0));
        assert!(typed.is_null(1));
        assert!(typed.is_valid(2));
    }
}
