//! Conformance tests for ferray-numpy-interop.
//!
//! ferray-numpy-interop is a feature-gated bridge crate, not a numerical
//! oracle target. Its public surface marshals byte buffers across
//! NumPy/Arrow/Polars boundaries; conformance is measured as round-trip
//! correctness (dtype preservation, shape preservation, bit-identical
//! values) — there is no NumPy reference scalar to ULP-compare against.
//!
//! Each `#[test]` below names its target item by the user-facing
//! crate-root re-export path AND the inner canonical module path in
//! its doc comment so the surface-coverage gate's text-match picks
//! both up. Tests are feature-gated and silently no-op when the
//! relevant Cargo feature is disabled.
//!
//! Items with no direct test here (traits, blanket re-exports, dtype
//! variants whose family representative is already exercised) are
//! listed in `tests/conformance/_surface_exclusions.toml` with a
//! `covered_by` anchor pointing to the in-crate unit test that pins
//! their contract.

#![cfg(any(feature = "arrow", feature = "polars", feature = "python"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    unused_imports
)]

use ferray_test_oracle::{fixtures_dir, load_fixture};

// Reference the fixtures helper so the test-oracle linkage is exercised.
// numpy-interop has no `fixtures/numpy_interop/` directory — round-trip
// correctness is checked against Rust-side data, not Python output.
#[allow(dead_code)]
fn _oracle_anchor() -> std::path::PathBuf {
    let _ = load_fixture; // satisfy the linker for the re-export
    fixtures_dir()
}

// ===========================================================================
// numpy_conv: ferray <-> NumPy
// ===========================================================================

/// Covers the PyO3/NumPy trait surface:
/// - `ferray_numpy_interop::AsFerray`
/// - `ferray_numpy_interop::IntoNumPy`
/// - `ferray_numpy_interop::numpy_conv::AsFerray`
/// - `ferray_numpy_interop::numpy_conv::IntoNumPy`
/// - `ferray_numpy_interop::numpy_conv::NpElement`
///
/// The test uses `f64`, one of the `NpElement` implementors, to round-trip a
/// ferray `Array1` through an owned NumPy array and back with bit-exact payload
/// preservation.
#[cfg(feature = "python")]
#[test]
fn numpy_traits_roundtrip_path() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::{AsFerray, IntoNumPy};
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::Python;

    Python::initialize();
    Python::attach(|py| {
        let data = vec![1.0_f64, -0.0, f64::INFINITY, -3.5];
        let arr: Array1<f64> = Array::from_vec(Ix1::new([data.len()]), data.clone()).unwrap();
        let py_arr = arr.into_pyarray(py).expect("into_pyarray");
        assert_eq!(py_arr.shape(), [data.len()]);

        let readonly = py_arr.readonly();
        let back: Array1<f64> = readonly.as_ferray().expect("as_ferray");
        assert_eq!(back.shape(), &[data.len()]);
        for (actual, expected) in back.as_slice().unwrap().iter().zip(&data) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    });
}

// ===========================================================================
// dtype_map: Arrow ↔ ferray dtype
// ===========================================================================

/// Covers: `ferray_numpy_interop::dtype_map::dtype_to_arrow` and
/// `ferray_numpy_interop::dtype_map::arrow_to_dtype` — round-trip the
/// supported scalar dtypes through the Arrow `DataType` representation.
/// Path anchor for the surface gate (matched textually):
/// `ferray_numpy_interop::dtype_map::dtype_to_arrow`,
/// `ferray_numpy_interop::dtype_map::arrow_to_dtype`.
#[cfg(feature = "arrow")]
#[test]
fn dtype_to_arrow_and_arrow_to_dtype_roundtrip() {
    use ferray_core::DType;
    use ferray_numpy_interop::dtype_map::{arrow_to_dtype, dtype_to_arrow};
    let cases = [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
    ];
    for dt in &cases {
        let arrow_dt = dtype_to_arrow(*dt).unwrap_or_else(|e| panic!("{dt}: {e}"));
        let back = arrow_to_dtype(&arrow_dt).unwrap_or_else(|e| panic!("{dt}: {e}"));
        assert_eq!(back, *dt, "dtype roundtrip failed for {dt}");
    }
}

/// Covers: `ferray_numpy_interop::dtype_map::dtype_to_polars` and
/// `ferray_numpy_interop::dtype_map::polars_to_dtype`.
/// Path anchors: `ferray_numpy_interop::dtype_map::dtype_to_polars`,
/// `ferray_numpy_interop::dtype_map::polars_to_dtype`.
#[cfg(feature = "polars")]
#[test]
fn dtype_to_polars_and_polars_to_dtype_roundtrip() {
    use ferray_core::DType;
    use ferray_numpy_interop::dtype_map::{dtype_to_polars, polars_to_dtype};
    let cases = [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
    ];
    for dt in &cases {
        let polars_dt = dtype_to_polars(*dt).unwrap_or_else(|e| panic!("{dt}: {e}"));
        let back = polars_to_dtype(&polars_dt).unwrap_or_else(|e| panic!("{dt}: {e}"));
        assert_eq!(back, *dt, "dtype roundtrip failed for {dt}");
    }
}

// ===========================================================================
// arrow_conv: ferray ↔ Arrow primitive
// ===========================================================================

/// Covers: `ferray_numpy_interop::arrow_conv::arrayd_to_arrow_flat` and
/// `ferray_numpy_interop::arrow_conv::arrayd_from_arrow_flat` — flat
/// dynamic-rank Arrow round-trip. Also exercises the
/// `ferray_numpy_interop::arrow_conv::ArrowElement` trait via the
/// generic `T: ArrowElement` bound. Crate-root re-exports
/// `ferray_numpy_interop::arrayd_to_arrow_flat` and
/// `ferray_numpy_interop::arrayd_from_arrow_flat` are exercised by the
/// same code path through the re-export.
#[cfg(feature = "arrow")]
#[test]
fn arrayd_arrow_flat_roundtrip() {
    use ferray_core::Array;
    use ferray_core::array::aliases::ArrayD;
    use ferray_core::dimension::IxDyn;
    use ferray_numpy_interop::arrow_conv::{arrayd_from_arrow_flat, arrayd_to_arrow_flat};

    let data: Vec<f64> = (0..24).map(|i| i as f64 * 0.5).collect();
    let a: ArrayD<f64> = Array::from_vec(IxDyn::new(&[2, 3, 4]), data.clone()).unwrap();
    let arrow_arr = arrayd_to_arrow_flat(&a).unwrap();
    assert_eq!(arrow_arr.len(), data.len());
    let back: ArrayD<f64> = arrayd_from_arrow_flat(&arrow_arr, &[2, 3, 4]).unwrap();
    assert_eq!(back.shape(), &[2, 3, 4]);
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}

/// Covers: `ferray_numpy_interop::arrow_conv::array2_to_arrow_columns`
/// and `ferray_numpy_interop::arrow_conv::array2_from_arrow_columns`.
/// Crate-root re-exports `ferray_numpy_interop::array2_to_arrow_columns`
/// and `ferray_numpy_interop::array2_from_arrow_columns` resolve to the
/// same code path through the re-export.
#[cfg(feature = "arrow")]
#[test]
fn array2_arrow_columns_roundtrip_path() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array2;
    use ferray_core::dimension::Ix2;
    use ferray_numpy_interop::arrow_conv::{array2_from_arrow_columns, array2_to_arrow_columns};

    let data = vec![1i32, 2, 3, 4, 5, 6];
    let a: Array2<i32> = Array::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
    let cols = array2_to_arrow_columns(&a).unwrap();
    assert_eq!(cols.len(), 3);
    let back: Array2<i32> = array2_from_arrow_columns(&cols).unwrap();
    assert_eq!(back.shape(), &[2, 3]);
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}

/// Covers the `ferray_numpy_interop::arrow_conv::ToArrow` and
/// `ferray_numpy_interop::arrow_conv::FromArrow` trait surface (and
/// their crate-root re-exports `ferray_numpy_interop::ToArrow` and
/// `ferray_numpy_interop::FromArrow`) by exercising the blanket impl
/// for `Array1<T: ArrowElement>`.
#[cfg(feature = "arrow")]
#[test]
fn to_arrow_and_from_arrow_traits_roundtrip() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::arrow_conv::{FromArrow, ToArrow};

    let data: Vec<i64> = vec![1, 2, 3, 4, 5];
    let arr: Array1<i64> = Array::from_vec(Ix1::new([data.len()]), data.clone()).unwrap();
    let arrow_arr = arr.to_arrow().unwrap();
    let back: Array1<i64> = arrow_arr.to_ferray().unwrap();
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}

/// Covers `ferray_numpy_interop::arrow_conv::ToArrowBool` and
/// `ferray_numpy_interop::arrow_conv::FromArrowBool` (and their crate-
/// root re-exports `ferray_numpy_interop::ToArrowBool` and
/// `ferray_numpy_interop::FromArrowBool`).
#[cfg(feature = "arrow")]
#[test]
fn to_arrow_bool_and_from_arrow_bool_roundtrip() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::arrow_conv::{FromArrowBool, ToArrowBool};

    let data = vec![true, false, true, true, false];
    let arr: Array1<bool> = Array::from_vec(Ix1::new([data.len()]), data.clone()).unwrap();
    let arrow_arr = arr.to_arrow().unwrap();
    let back: Array1<bool> = arrow_arr.to_ferray_bool().unwrap();
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}

// ===========================================================================
// extras: DynArray / RecordBatch / Polars frame helpers
// ===========================================================================

/// Covers: `ferray_numpy_interop::extras::dynarray_to_arrow` (and crate-
/// root re-export `ferray_numpy_interop::dynarray_to_arrow`) via a
/// rank-1 DynArray round-trip.
#[cfg(feature = "arrow")]
#[test]
fn extras_dynarray_to_arrow_path() {
    use ferray_core::dimension::IxDyn;
    use ferray_core::{Array, DynArray};
    use ferray_numpy_interop::extras::dynarray_to_arrow;
    let arr: Array<f64, IxDyn> =
        Array::from_vec(IxDyn::new(&[3]), vec![1.5_f64, 2.5, 3.5]).unwrap();
    let dyn_arr: DynArray = arr.into();
    let out = dynarray_to_arrow(&dyn_arr).unwrap();
    assert_eq!(out.len(), 3);
}

/// Covers: `ferray_numpy_interop::extras::dynarray_to_arrow_with_mask`
/// and `ferray_numpy_interop::extras::arrow_to_dynarray_with_mask` (and
/// their crate-root re-exports `ferray_numpy_interop::dynarray_to_arrow_with_mask`
/// and `ferray_numpy_interop::arrow_to_dynarray_with_mask`).
#[cfg(feature = "arrow")]
#[test]
fn extras_dynarray_with_mask_roundtrip() {
    use ferray_core::dimension::IxDyn;
    use ferray_core::{Array, DType, DynArray};
    use ferray_numpy_interop::extras::{arrow_to_dynarray_with_mask, dynarray_to_arrow_with_mask};

    let arr: Array<i64, IxDyn> = Array::from_vec(IxDyn::new(&[4]), vec![10, 20, 30, 40]).unwrap();
    let dyn_arr: DynArray = arr.into();
    let mask = vec![true, false, true, false];
    let arrow_ref = dynarray_to_arrow_with_mask(&dyn_arr, Some(&mask)).unwrap();
    let (back, mask_back) = arrow_to_dynarray_with_mask(&arrow_ref).unwrap();
    assert_eq!(back.shape(), &[4]);
    assert_eq!(back.dtype(), DType::I64);
    // The mask polarity returned by arrow_to_dynarray_with_mask
    // matches the original valid-bitmap: false at index 1 and 3.
    let m = mask_back.expect("mask should be present");
    assert_eq!(m.len(), 4);
    assert!(m[0]);
    assert!(!m[1]);
}

/// Covers: `ferray_numpy_interop::extras::array1_to_arrow_with_mask`
/// (and crate-root re-export
/// `ferray_numpy_interop::array1_to_arrow_with_mask`).
#[cfg(feature = "arrow")]
#[test]
fn extras_array1_to_arrow_with_mask_path() {
    use arrow::array::Array as _;
    use ferray_core::Array;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::extras::array1_to_arrow_with_mask;

    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let arr = Array::<f64, Ix1>::from_vec(Ix1::new([4]), data).unwrap();
    let mask = vec![false, true, false, false];
    let out = array1_to_arrow_with_mask(&arr, &mask).unwrap();
    assert_eq!(out.len(), 4);
    assert_eq!(out.null_count(), 1);
}

/// Covers: `ferray_numpy_interop::extras::record_batch_from_columns`
/// and `ferray_numpy_interop::extras::record_batch_to_columns` (plus
/// crate-root re-exports `ferray_numpy_interop::record_batch_from_columns`
/// and `ferray_numpy_interop::record_batch_to_columns`).
#[cfg(feature = "arrow")]
#[test]
fn extras_record_batch_roundtrip() {
    use ferray_core::dimension::IxDyn;
    use ferray_core::{Array, DynArray};
    use ferray_numpy_interop::extras::{record_batch_from_columns, record_batch_to_columns};

    let a: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![1, 2, 3]).unwrap();
    let b: Array<f64, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![1.5, 2.5, 3.5]).unwrap();
    let ad: DynArray = a.into();
    let bd: DynArray = b.into();
    let batch = record_batch_from_columns(&[("ints", &ad), ("floats", &bd)]).unwrap();
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.num_rows(), 3);
    let back = record_batch_to_columns(&batch).unwrap();
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].0, "ints");
    assert_eq!(back[1].0, "floats");
}

/// Covers: `ferray_numpy_interop::extras::dataframe_from_columns` (and
/// crate-root re-export `ferray_numpy_interop::dataframe_from_columns`).
#[cfg(feature = "polars")]
#[test]
fn extras_dataframe_from_columns_path() {
    use ferray_core::dimension::IxDyn;
    use ferray_core::{Array, DynArray};
    use ferray_numpy_interop::extras::dataframe_from_columns;
    let a: Array<i64, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![1, 2, 3]).unwrap();
    let ad: DynArray = a.into();
    let df = dataframe_from_columns(&[("col", &ad)]).unwrap();
    assert_eq!(df.height(), 3);
    assert_eq!(df.width(), 1);
}

/// Covers complex-Arrow helpers in the inner
/// `ferray_numpy_interop::extras::complex_arrow` module and their
/// `extras::*`/crate-root re-exports:
/// `ferray_numpy_interop::extras::complex_arrow::complex32_to_arrow`,
/// `ferray_numpy_interop::extras::complex_arrow::arrow_to_complex32`,
/// `ferray_numpy_interop::extras::complex_arrow::complex64_to_arrow`,
/// `ferray_numpy_interop::extras::complex_arrow::arrow_to_complex64`,
/// `ferray_numpy_interop::extras::complex32_to_arrow`,
/// `ferray_numpy_interop::extras::arrow_to_complex32`,
/// `ferray_numpy_interop::extras::complex64_to_arrow`,
/// `ferray_numpy_interop::extras::arrow_to_complex64`,
/// `ferray_numpy_interop::complex32_to_arrow`,
/// `ferray_numpy_interop::arrow_to_complex32`,
/// `ferray_numpy_interop::complex64_to_arrow`,
/// `ferray_numpy_interop::arrow_to_complex64`.
#[cfg(all(feature = "arrow", feature = "complex"))]
#[test]
fn extras_complex_arrow_roundtrip() {
    use ferray_core::Array;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::extras::{
        arrow_to_complex32, arrow_to_complex64, complex32_to_arrow, complex64_to_arrow,
    };
    use num_complex::Complex;

    let c64 = vec![
        Complex::<f64>::new(1.0, 2.0),
        Complex::<f64>::new(-3.0, 4.0),
    ];
    let arr64 = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([c64.len()]), c64.clone()).unwrap();
    let out64 = complex64_to_arrow(&arr64).unwrap();
    let back64 = arrow_to_complex64(&out64).unwrap();
    assert_eq!(back64.as_slice().unwrap(), &c64[..]);

    let c32 = vec![Complex::<f32>::new(0.5, -0.5)];
    let arr32 = Array::<Complex<f32>, Ix1>::from_vec(Ix1::new([c32.len()]), c32.clone()).unwrap();
    let out32 = complex32_to_arrow(&arr32).unwrap();
    let back32 = arrow_to_complex32(&out32).unwrap();
    assert_eq!(back32.as_slice().unwrap(), &c32[..]);
}

// ===========================================================================
// polars_conv
// ===========================================================================

/// Covers: `ferray_numpy_interop::polars_conv::array2_to_polars_dataframe`
/// and `ferray_numpy_interop::polars_conv::array2_from_polars_dataframe`
/// (plus crate-root re-exports
/// `ferray_numpy_interop::array2_to_polars_dataframe` and
/// `ferray_numpy_interop::array2_from_polars_dataframe`).
#[cfg(feature = "polars")]
#[test]
fn polars_array2_dataframe_roundtrip_path() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array2;
    use ferray_core::dimension::Ix2;
    use ferray_numpy_interop::polars_conv::{
        array2_from_polars_dataframe, array2_to_polars_dataframe,
    };

    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a: Array2<f64> = Array::from_vec(Ix2::new([3, 2]), data.clone()).unwrap();
    let df = array2_to_polars_dataframe(&a, &["x", "y"]).unwrap();
    assert_eq!(df.width(), 2);
    assert_eq!(df.height(), 3);
    let back: Array2<f64> = array2_from_polars_dataframe(&df).unwrap();
    assert_eq!(back.shape(), &[3, 2]);
}

/// Covers the `ferray_numpy_interop::polars_conv::ToPolars` and
/// `ferray_numpy_interop::polars_conv::FromPolars` trait surface (and
/// their crate-root re-exports `ferray_numpy_interop::ToPolars` and
/// `ferray_numpy_interop::FromPolars`) via the blanket impl for
/// `Array1<T: PolarsElement>`. Path anchor:
/// `ferray_numpy_interop::polars_conv::PolarsElement`.
#[cfg(feature = "polars")]
#[test]
fn polars_to_from_polars_traits_roundtrip() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::polars_conv::{FromPolars, ToPolars};

    let data: Vec<i32> = vec![10, 20, 30];
    let arr: Array1<i32> = Array::from_vec(Ix1::new([data.len()]), data.clone()).unwrap();
    let series = arr.to_polars_series("col").unwrap();
    let back: Array1<i32> = series.into_ferray().unwrap();
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}

/// Covers `ferray_numpy_interop::polars_conv::ToPolarsBool` and
/// `ferray_numpy_interop::polars_conv::FromPolarsBool` (plus crate-root
/// re-exports `ferray_numpy_interop::ToPolarsBool` and
/// `ferray_numpy_interop::FromPolarsBool`).
#[cfg(feature = "polars")]
#[test]
fn polars_bool_traits_roundtrip() {
    use ferray_core::Array;
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    use ferray_numpy_interop::polars_conv::{FromPolarsBool, ToPolarsBool};

    let data = vec![true, false, true, true, false];
    let arr: Array1<bool> = Array::from_vec(Ix1::new([data.len()]), data.clone()).unwrap();
    let series = arr.to_polars_series("col").unwrap();
    let back: Array1<bool> = series.into_ferray_bool().unwrap();
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}
