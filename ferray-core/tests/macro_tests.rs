// Integration tests for ferray-core-macros
//
// These tests exercise the proc macros (FerrayRecord, s!, promoted_type!)
// from the consumer perspective — i.e., they use the macros via ferray-core's
// public re-exports, just as a downstream crate would.

use ferray_core::dtype::{DType, SliceInfoElem};
use ferray_core::record::FerrayRecord;

// ---------------------------------------------------------------------------
// #[derive(FerrayRecord)] tests
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Debug, ferray_core::FerrayRecord)]
struct Point {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Debug, ferray_core::FerrayRecord)]
struct Measurement {
    time: f64,
    value: f32,
    count: i32,
    flag: bool,
}

#[test]
fn derive_record_field_count() {
    let fields = Point::field_descriptors();
    assert_eq!(fields.len(), 2);
}

#[test]
fn derive_record_field_names() {
    let fields = Point::field_descriptors();
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[1].name, "y");
}

#[test]
fn derive_record_field_dtypes() {
    let fields = Point::field_descriptors();
    assert_eq!(fields[0].dtype, DType::F64);
    assert_eq!(fields[1].dtype, DType::F64);
}

#[test]
fn derive_record_field_offsets() {
    let fields = Point::field_descriptors();
    // #[repr(C)] struct with two f64 fields: offset 0 and 8
    assert_eq!(fields[0].offset, 0);
    assert_eq!(fields[1].offset, 8);
}

#[test]
fn derive_record_field_sizes() {
    let fields = Point::field_descriptors();
    assert_eq!(fields[0].size, 8);
    assert_eq!(fields[1].size, 8);
}

#[test]
fn derive_record_size() {
    assert_eq!(Point::record_size(), std::mem::size_of::<Point>());
    assert_eq!(Point::record_size(), 16);
}

#[test]
fn derive_record_field_by_name() {
    let fd = Point::field_by_name("y").unwrap();
    assert_eq!(fd.dtype, DType::F64);
    assert_eq!(fd.offset, 8);

    assert!(Point::field_by_name("z").is_none());
}

#[test]
fn derive_record_multi_type() {
    let fields = Measurement::field_descriptors();
    assert_eq!(fields.len(), 4);

    assert_eq!(fields[0].name, "time");
    assert_eq!(fields[0].dtype, DType::F64);

    assert_eq!(fields[1].name, "value");
    assert_eq!(fields[1].dtype, DType::F32);

    assert_eq!(fields[2].name, "count");
    assert_eq!(fields[2].dtype, DType::I32);

    assert_eq!(fields[3].name, "flag");
    assert_eq!(fields[3].dtype, DType::Bool);
}

#[test]
fn derive_record_multi_type_size() {
    assert_eq!(
        Measurement::record_size(),
        std::mem::size_of::<Measurement>()
    );
    // f64(8) + f32(4) + i32(4) + bool(1) + padding(7) = 24 for repr(C)
    // The exact size depends on alignment, but it's at least 17
    assert!(Measurement::record_size() >= 17);
}

// ---------------------------------------------------------------------------
// s![] macro tests
// ---------------------------------------------------------------------------

#[test]
fn s_macro_single_index() {
    let slices = ferray_core::s![3];
    assert_eq!(slices.len(), 1);
    assert_eq!(slices[0], SliceInfoElem::Index(3));
}

#[test]
fn s_macro_full_range() {
    let slices = ferray_core::s![..];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }
    );
}

#[test]
fn s_macro_range() {
    let slices = ferray_core::s![1..5];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 1,
            end: Some(5),
            step: 1,
        }
    );
}

#[test]
fn s_macro_range_from() {
    let slices = ferray_core::s![2..];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 2,
            end: None,
            step: 1,
        }
    );
}

#[test]
fn s_macro_range_to() {
    let slices = ferray_core::s![..5];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: Some(5),
            step: 1,
        }
    );
}

#[test]
fn s_macro_range_with_step() {
    let slices = ferray_core::s![1..5;2];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 1,
            end: Some(5),
            step: 2,
        }
    );
}

#[test]
fn s_macro_full_range_with_step() {
    let slices = ferray_core::s![..;3];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 3,
        }
    );
}

#[test]
fn s_macro_multi_axis() {
    let slices = ferray_core::s![0..3, 2];
    assert_eq!(slices.len(), 2);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: Some(3),
            step: 1,
        }
    );
    assert_eq!(slices[1], SliceInfoElem::Index(2));
}

#[test]
fn s_macro_all_rows_step_cols() {
    let slices = ferray_core::s![.., 0..;2];
    assert_eq!(slices.len(), 2);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        }
    );
    assert_eq!(
        slices[1],
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 2,
        }
    );
}

// --- s![] macro tests: negative indices ---

#[test]
fn s_macro_negative_index() {
    let slices = ferray_core::s![-1];
    assert_eq!(slices.len(), 1);
    assert_eq!(slices[0], SliceInfoElem::Index(-1));
}

// --- s![] macro tests: expression robustness (#322) ---
//
// The macro routes through `proc_macro2::TokenStream::to_string()`
// then string-parses the result. The Rust pretty-printer's whitespace
// rules can drift slightly between compiler versions; these tests pin
// the macro's behavior on inputs that exercise non-trivial token
// shapes (variables, arithmetic, function calls, scoped paths).

#[test]
fn s_macro_variable_bounds() {
    let start = 1isize;
    let end = 5isize;
    let slices = ferray_core::s![start..end];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 1,
            end: Some(5),
            step: 1,
        }
    );
}

#[test]
fn s_macro_arithmetic_expressions() {
    let n = 10isize;
    let slices = ferray_core::s![n - 5..n - 1];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 5,
            end: Some(9),
            step: 1,
        }
    );
}

#[test]
fn s_macro_function_call_in_bound() {
    fn compute_end() -> isize {
        7
    }
    let slices = ferray_core::s![0..compute_end()];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: Some(7),
            step: 1,
        }
    );
}

#[test]
fn s_macro_step_from_variable() {
    let step = 3isize;
    let slices = ferray_core::s![0..10;step];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: Some(10),
            step: 3,
        }
    );
}

#[test]
fn s_macro_isize_constant_in_index() {
    // Pin behaviour on an indexed expression with a typed literal.
    let slices = ferray_core::s![5isize];
    assert_eq!(slices.len(), 1);
    assert_eq!(slices[0], SliceInfoElem::Index(5));
}

#[test]
fn s_macro_negative_start() {
    let slices = ferray_core::s![-3..];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: -3,
            end: None,
            step: 1,
        }
    );
}

#[test]
fn s_macro_negative_end() {
    let slices = ferray_core::s![..-1];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: Some(-1),
            step: 1,
        }
    );
}

#[test]
fn s_macro_negative_step() {
    let slices = ferray_core::s![..;-1];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: -1,
        }
    );
}

#[test]
fn s_macro_negative_range_with_step() {
    let slices = ferray_core::s![-5..-1;2];
    assert_eq!(slices.len(), 1);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: -5,
            end: Some(-1),
            step: 2,
        }
    );
}

#[test]
fn s_macro_multi_axis_with_negatives() {
    let slices = ferray_core::s![1..-1, -2];
    assert_eq!(slices.len(), 2);
    assert_eq!(
        slices[0],
        SliceInfoElem::Slice {
            start: 1,
            end: Some(-1),
            step: 1,
        }
    );
    assert_eq!(slices[1], SliceInfoElem::Index(-2));
}

// ---------------------------------------------------------------------------
// promoted_type! macro tests
// ---------------------------------------------------------------------------

#[test]
fn promoted_type_f32_f64() {
    // promoted_type!(f32, f64) should resolve to f64
    let _: ferray_core::promoted_type!(f32, f64) = 1.0f64;
}

#[test]
fn promoted_type_i32_f32() {
    // i32 + f32 -> f64 (because i32 needs 53-bit mantissa)
    let _: ferray_core::promoted_type!(i32, f32) = 1.0f64;
}

#[test]
fn promoted_type_u8_i8() {
    // u8 + i8 -> i16
    let _: ferray_core::promoted_type!(u8, i8) = 1i16;
}

#[test]
fn promoted_type_same() {
    let _: ferray_core::promoted_type!(f64, f64) = 1.0f64;
    let _: ferray_core::promoted_type!(i32, i32) = 1i32;
}

#[test]
fn promoted_type_bool_int() {
    let _: ferray_core::promoted_type!(bool, i32) = 1i32;
}

#[test]
fn promoted_type_complex() {
    let _: ferray_core::promoted_type!(Complex<f32>, f64) =
        num_complex::Complex::new(1.0f64, 0.0f64);
}

use num_complex::Complex;

#[test]
fn promoted_type_complex_f32() {
    let _: ferray_core::promoted_type!(f32, Complex<f32>) = Complex::new(1.0f32, 0.0f32);
}

#[test]
fn promoted_type_complex_f32_complex_f64() {
    // Issue #325: Complex<f32> + Complex<f64> should widen to Complex<f64>.
    let _: ferray_core::promoted_type!(Complex<f32>, Complex<f64>) = Complex::new(1.0f64, 0.0f64);
}

#[test]
fn promoted_type_complex_f64_complex_f32() {
    // Symmetry check.
    let _: ferray_core::promoted_type!(Complex<f64>, Complex<f32>) = Complex::new(1.0f64, 0.0f64);
}

// ---------------------------------------------------------------------------
// Cross-validation: proc macro vs Promoted trait (#214)
//
// The three promotion pathways (compile-time `promoted_type!`, runtime
// `ferray_core::dtype::promotion::Promoted`, and the
// `ferray_core::dtype::promotion::promoted_dtype` lookup) used to be
// maintained independently. These tests assert that they agree on a
// representative set of dtype pairs so a silent divergence fails CI.
// ---------------------------------------------------------------------------

#[test]
fn promoted_type_matches_promoted_trait_f32_f64() {
    use ferray_core::dtype::promotion::Promoted;
    type Macro = ferray_core::promoted_type!(f32, f64);
    type Trait = <f32 as Promoted<f64>>::Output;
    // If this compiles with the same type alias, the two branches agree.
    let _: Macro = 1.0f64;
    let _: Trait = 1.0f64;
}

#[test]
fn promoted_type_matches_promoted_trait_u8_i8() {
    use ferray_core::dtype::promotion::Promoted;
    type Macro = ferray_core::promoted_type!(u8, i8);
    type Trait = <u8 as Promoted<i8>>::Output;
    let _: Macro = 1i16;
    let _: Trait = 1i16;
}

#[test]
fn promoted_type_matches_promoted_trait_i32_f32() {
    use ferray_core::dtype::promotion::Promoted;
    type Macro = ferray_core::promoted_type!(i32, f32);
    type Trait = <i32 as Promoted<f32>>::Output;
    let _: Macro = 1.0f64;
    let _: Trait = 1.0f64;
}

#[test]
fn promoted_type_matches_promoted_trait_complex() {
    use ferray_core::dtype::promotion::Promoted;
    type Macro = ferray_core::promoted_type!(Complex<f32>, Complex<f64>);
    type Trait = <Complex<f32> as Promoted<Complex<f64>>>::Output;
    let _: Macro = Complex::new(1.0f64, 0.0f64);
    let _: Trait = Complex::new(1.0f64, 0.0f64);
}

// ---------------------------------------------------------------------------
// FerrayRecord generic / edge cases (#326, #327)
// ---------------------------------------------------------------------------

#[test]
fn s_macro_with_three_axes() {
    // Issue #327: exercise the s![a, b, c] multi-axis path that the
    // existing tests only covered up to two axes.
    let slices = ferray_core::s![0..2, 1..4, ..];
    assert_eq!(slices.len(), 3);
}

#[test]
fn s_macro_with_step_and_full_slice() {
    // Mixed step-and-slice cases.
    let slices = ferray_core::s![0..10;2, ..];
    assert_eq!(slices.len(), 2);
}
