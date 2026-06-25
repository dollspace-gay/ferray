//! Conformance coverage for public internal ufunc infrastructure.
//!
//! The NumPy-facing ufunc functions are covered by the fixture-backed
//! `conformance_*.rs` suites. This file pins the remaining public
//! implementation surface: dispatch glue, helper wrappers, fast math kernels,
//! and scalar/SIMD slice kernels.

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};

fn arr1<T: ferray_core::Element>(data: Vec<T>) -> Array<T, Ix1> {
    let n = data.len();
    Array::<T, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn arr2<T: ferray_core::Element>(rows: usize, cols: usize, data: Vec<T>) -> Array<T, Ix2> {
    Array::<T, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

fn assert_f64_close(actual: &[f64], expected: &[f64], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-12, "{name}[{i}]: got {a}, expected {e}");
    }
}

fn assert_f32_close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-5, "{name}[{i}]: got {a}, expected {e}");
    }
}

/// Path anchors:
/// `ferray_ufunc::cr_math::CrMath`.
#[test]
fn cr_math_trait_matches_scalar_reference_for_basic_values() {
    use ferray_ufunc::cr_math::CrMath;

    let x64 = 0.5_f64;
    assert!((x64.cr_sin() - core_math::sin(x64)).abs() < 1e-15);
    assert!((x64.cr_cos() - core_math::cos(x64)).abs() < 1e-15);
    assert!((x64.cr_exp() - core_math::exp(x64)).abs() < 1e-15);
    assert!((x64.cr_ln_1p() - core_math::log1p(x64)).abs() < 1e-15);
    assert!((x64.cr_hypot(1.25) - core_math::hypot(x64, 1.25)).abs() < 1e-15);

    let x32 = 0.5_f32;
    assert!((x32.cr_sin() - core_math::sinf(x32)).abs() < 1e-6);
    assert!((x32.cr_cos() - core_math::cosf(x32)).abs() < 1e-6);
    assert!((x32.cr_exp() - core_math::expf(x32)).abs() < 1e-6);
    assert!((x32.cr_ln_1p() - core_math::log1pf(x32)).abs() < 1e-6);
}

/// Path anchors:
/// `ferray_ufunc::dispatch::dispatch_binary_f16`,
/// `ferray_ufunc::dispatch::dispatch_binary_f32`,
/// `ferray_ufunc::dispatch::dispatch_binary_f64`,
/// `ferray_ufunc::dispatch::dispatch_exp_fast_f32`,
/// `ferray_ufunc::dispatch::dispatch_exp_fast_f64`,
/// `ferray_ufunc::dispatch::dispatch_unary_f16`,
/// `ferray_ufunc::dispatch::dispatch_unary_f32`,
/// `ferray_ufunc::dispatch::dispatch_unary_f64`,
/// `ferray_ufunc::dispatch::force_scalar`,
/// `ferray_ufunc::dispatch::reset_force_scalar`,
/// `ferray_ufunc::dispatch::simd_abs_f32`,
/// `ferray_ufunc::dispatch::simd_abs_f64`,
/// `ferray_ufunc::dispatch::simd_neg_f32`,
/// `ferray_ufunc::dispatch::simd_neg_f64`,
/// `ferray_ufunc::dispatch::simd_reciprocal_f32`,
/// `ferray_ufunc::dispatch::simd_reciprocal_f64`,
/// `ferray_ufunc::dispatch::simd_sqrt_f32`,
/// `ferray_ufunc::dispatch::simd_sqrt_f64`,
/// `ferray_ufunc::dispatch::simd_square_f32`,
/// `ferray_ufunc::dispatch::simd_square_f64`.
#[test]
fn dispatch_and_simd_slice_paths_match_scalar_arithmetic() {
    let expected_force_scalar = std::env::var("FERRAY_FORCE_SCALAR")
        .ok()
        .is_some_and(|v| v == "1");
    ferray_ufunc::dispatch::reset_force_scalar();
    assert_eq!(
        ferray_ufunc::dispatch::force_scalar(),
        expected_force_scalar
    );

    let in32 = [1.0_f32, 4.0, 9.0, 16.0];
    let mut out32 = [0.0_f32; 4];
    ferray_ufunc::dispatch::dispatch_unary_f32(&in32, &mut out32, f32::sqrt);
    assert_f32_close(&out32, &[1.0, 2.0, 3.0, 4.0], "dispatch_unary_f32");

    let in64 = [1.0_f64, 4.0, 9.0, 16.0];
    let mut out64 = [0.0_f64; 4];
    ferray_ufunc::dispatch::dispatch_unary_f64(&in64, &mut out64, f64::sqrt);
    assert_f64_close(&out64, &[1.0, 2.0, 3.0, 4.0], "dispatch_unary_f64");

    let b32 = [2.0_f32, 3.0, 4.0, 5.0];
    ferray_ufunc::dispatch::dispatch_binary_f32(&in32, &b32, &mut out32, |x, y| x + y);
    assert_f32_close(&out32, &[3.0, 7.0, 13.0, 21.0], "dispatch_binary_f32");

    let b64 = [2.0_f64, 3.0, 4.0, 5.0];
    ferray_ufunc::dispatch::dispatch_binary_f64(&in64, &b64, &mut out64, |x, y| x * y);
    assert_f64_close(&out64, &[2.0, 12.0, 36.0, 80.0], "dispatch_binary_f64");

    ferray_ufunc::dispatch::simd_abs_f32(&[-1.0, 2.0, -3.5], &mut out32[..3]);
    assert_f32_close(&out32[..3], &[1.0, 2.0, 3.5], "simd_abs_f32");
    ferray_ufunc::dispatch::simd_neg_f32(&[1.0, -2.0, 3.5], &mut out32[..3]);
    assert_f32_close(&out32[..3], &[-1.0, 2.0, -3.5], "simd_neg_f32");
    ferray_ufunc::dispatch::simd_sqrt_f32(&[1.0, 4.0, 9.0], &mut out32[..3]);
    assert_f32_close(&out32[..3], &[1.0, 2.0, 3.0], "simd_sqrt_f32");
    ferray_ufunc::dispatch::simd_square_f32(&[2.0, -3.0, 4.0], &mut out32[..3]);
    assert_f32_close(&out32[..3], &[4.0, 9.0, 16.0], "simd_square_f32");
    ferray_ufunc::dispatch::simd_reciprocal_f32(&[2.0, 4.0, 8.0], &mut out32[..3]);
    assert_f32_close(&out32[..3], &[0.5, 0.25, 0.125], "simd_reciprocal_f32");

    ferray_ufunc::dispatch::simd_abs_f64(&[-1.0, 2.0, -3.5], &mut out64[..3]);
    assert_f64_close(&out64[..3], &[1.0, 2.0, 3.5], "simd_abs_f64");
    ferray_ufunc::dispatch::simd_neg_f64(&[1.0, -2.0, 3.5], &mut out64[..3]);
    assert_f64_close(&out64[..3], &[-1.0, 2.0, -3.5], "simd_neg_f64");
    ferray_ufunc::dispatch::simd_sqrt_f64(&[1.0, 4.0, 9.0], &mut out64[..3]);
    assert_f64_close(&out64[..3], &[1.0, 2.0, 3.0], "simd_sqrt_f64");
    ferray_ufunc::dispatch::simd_square_f64(&[2.0, -3.0, 4.0], &mut out64[..3]);
    assert_f64_close(&out64[..3], &[4.0, 9.0, 16.0], "simd_square_f64");
    ferray_ufunc::dispatch::simd_reciprocal_f64(&[2.0, 4.0, 8.0], &mut out64[..3]);
    assert_f64_close(&out64[..3], &[0.5, 0.25, 0.125], "simd_reciprocal_f64");

    ferray_ufunc::dispatch::dispatch_exp_fast_f32(&[0.0, 1.0], &mut out32[..2]);
    assert_f32_close(
        &out32[..2],
        &[
            ferray_ufunc::fast_exp::exp_fast_f32(0.0),
            ferray_ufunc::fast_exp::exp_fast_f32(1.0),
        ],
        "dispatch_exp_fast_f32",
    );
    ferray_ufunc::dispatch::dispatch_exp_fast_f64(&[0.0, 1.0], &mut out64[..2]);
    assert_f64_close(
        &out64[..2],
        &[
            ferray_ufunc::fast_exp::exp_fast_f64(0.0),
            ferray_ufunc::fast_exp::exp_fast_f64(1.0),
        ],
        "dispatch_exp_fast_f64",
    );
}

/// Path anchors:
/// `ferray_ufunc::fast_exp::exp_fast_batch_f32`,
/// `ferray_ufunc::fast_exp::exp_fast_batch_f64`,
/// `ferray_ufunc::fast_exp::exp_fast_f32`,
/// `ferray_ufunc::fast_exp::exp_fast_f64`,
/// `ferray_ufunc::fast_trig::cos_fast_batch_f32`,
/// `ferray_ufunc::fast_trig::cos_fast_batch_f64`,
/// `ferray_ufunc::fast_trig::cos_fast_f32`,
/// `ferray_ufunc::fast_trig::cos_fast_f64`,
/// `ferray_ufunc::fast_trig::sin_fast_batch_f32`,
/// `ferray_ufunc::fast_trig::sin_fast_batch_f64`,
/// `ferray_ufunc::fast_trig::sin_fast_f32`,
/// `ferray_ufunc::fast_trig::sin_fast_f64`.
#[test]
fn fast_exp_and_trig_batches_match_scalar_fast_kernels() {
    let input64 = [-1.0_f64, 0.0, 0.5, 1.0];
    let mut out64 = [0.0_f64; 4];
    ferray_ufunc::fast_exp::exp_fast_batch_f64(&input64, &mut out64);
    let expected64: Vec<f64> = input64
        .iter()
        .map(|&x| ferray_ufunc::fast_exp::exp_fast_f64(x))
        .collect();
    assert_f64_close(&out64, &expected64, "exp_fast_batch_f64");

    ferray_ufunc::fast_trig::sin_fast_batch_f64(&input64, &mut out64);
    let expected64: Vec<f64> = input64
        .iter()
        .map(|&x| ferray_ufunc::fast_trig::sin_fast_f64(x))
        .collect();
    assert_f64_close(&out64, &expected64, "sin_fast_batch_f64");

    ferray_ufunc::fast_trig::cos_fast_batch_f64(&input64, &mut out64);
    let expected64: Vec<f64> = input64
        .iter()
        .map(|&x| ferray_ufunc::fast_trig::cos_fast_f64(x))
        .collect();
    assert_f64_close(&out64, &expected64, "cos_fast_batch_f64");

    let input32 = [-1.0_f32, 0.0, 0.5, 1.0];
    let mut out32 = [0.0_f32; 4];
    ferray_ufunc::fast_exp::exp_fast_batch_f32(&input32, &mut out32);
    let expected32: Vec<f32> = input32
        .iter()
        .map(|&x| ferray_ufunc::fast_exp::exp_fast_f32(x))
        .collect();
    assert_f32_close(&out32, &expected32, "exp_fast_batch_f32");

    ferray_ufunc::fast_trig::sin_fast_batch_f32(&input32, &mut out32);
    let expected32: Vec<f32> = input32
        .iter()
        .map(|&x| ferray_ufunc::fast_trig::sin_fast_f32(x))
        .collect();
    assert_f32_close(&out32, &expected32, "sin_fast_batch_f32");

    ferray_ufunc::fast_trig::cos_fast_batch_f32(&input32, &mut out32);
    let expected32: Vec<f32> = input32
        .iter()
        .map(|&x| ferray_ufunc::fast_trig::cos_fast_f32(x))
        .collect();
    assert_f32_close(&out32, &expected32, "cos_fast_batch_f32");
}

/// Path anchors:
/// `ferray_ufunc::helpers::binary_broadcast_map_op`,
/// `ferray_ufunc::helpers::binary_broadcast_op`,
/// `ferray_ufunc::helpers::binary_elementwise_op`,
/// `ferray_ufunc::helpers::binary_elementwise_op_into`,
/// `ferray_ufunc::helpers::binary_f16_op`,
/// `ferray_ufunc::helpers::binary_f16_to_bool_op`,
/// `ferray_ufunc::helpers::binary_map_op`,
/// `ferray_ufunc::helpers::binary_mixed_op`,
/// `ferray_ufunc::helpers::try_simd_f32_binary`,
/// `ferray_ufunc::helpers::try_simd_f32_unary`,
/// `ferray_ufunc::helpers::try_simd_f64_binary`,
/// `ferray_ufunc::helpers::try_simd_f64_unary`,
/// `ferray_ufunc::helpers::unary_f16_op`,
/// `ferray_ufunc::helpers::unary_f16_to_bool_op`,
/// `ferray_ufunc::helpers::unary_float_op`,
/// `ferray_ufunc::helpers::unary_float_op_compute`,
/// `ferray_ufunc::helpers::unary_float_op_into`,
/// `ferray_ufunc::helpers::unary_float_op_into_compute`,
/// `ferray_ufunc::helpers::unary_map_op`,
/// `ferray_ufunc::helpers::unary_slice_op_f32`,
/// `ferray_ufunc::helpers::unary_slice_op_f64`.
#[test]
fn helper_wrappers_preserve_shape_broadcast_and_out_contracts() {
    let a = arr1(vec![1.0_f64, 4.0, 9.0]);
    let unary = ferray_ufunc::helpers::unary_float_op(&a, f64::sqrt).unwrap();
    assert_f64_close(
        unary.as_slice().unwrap(),
        &[1.0, 2.0, 3.0],
        "unary_float_op",
    );

    let compute = ferray_ufunc::helpers::unary_float_op_compute(&a, |x| x + 1.0).unwrap();
    assert_f64_close(
        compute.as_slice().unwrap(),
        &[2.0, 5.0, 10.0],
        "unary_float_op_compute",
    );

    let mapped = ferray_ufunc::helpers::unary_map_op(&a, |x| x > 3.0).unwrap();
    assert_eq!(mapped.as_slice().unwrap(), &[false, true, true]);

    let sliced64 =
        ferray_ufunc::helpers::unary_slice_op_f64(&a, ferray_ufunc::dispatch::simd_sqrt_f64)
            .unwrap();
    assert_f64_close(
        sliced64.as_slice().unwrap(),
        &[1.0, 2.0, 3.0],
        "unary_slice_op_f64",
    );

    let a32 = arr1(vec![1.0_f32, 4.0, 9.0]);
    let sliced32 =
        ferray_ufunc::helpers::unary_slice_op_f32(&a32, ferray_ufunc::dispatch::simd_sqrt_f32)
            .unwrap();
    assert_f32_close(
        sliced32.as_slice().unwrap(),
        &[1.0, 2.0, 3.0],
        "unary_slice_op_f32",
    );

    let maybe_simd64 =
        ferray_ufunc::helpers::try_simd_f64_unary(&a, ferray_ufunc::dispatch::simd_sqrt_f64)
            .expect("f64 input should use simd unary")
            .unwrap();
    assert_f64_close(
        maybe_simd64.as_slice().unwrap(),
        &[1.0, 2.0, 3.0],
        "try_simd_f64_unary",
    );
    assert!(ferray_ufunc::helpers::try_simd_f64_unary(&arr1(vec![1_i32]), |_, _| {}).is_none());

    let maybe_simd32 =
        ferray_ufunc::helpers::try_simd_f32_unary(&a32, ferray_ufunc::dispatch::simd_sqrt_f32)
            .expect("f32 input should use simd unary")
            .unwrap();
    assert_f32_close(
        maybe_simd32.as_slice().unwrap(),
        &[1.0, 2.0, 3.0],
        "try_simd_f32_unary",
    );

    let b = arr1(vec![10.0_f64, 20.0, 30.0]);
    let elementwise = ferray_ufunc::helpers::binary_elementwise_op(&a, &b, |x, y| x + y).unwrap();
    assert_f64_close(
        elementwise.as_slice().unwrap(),
        &[11.0, 24.0, 39.0],
        "binary_elementwise_op",
    );

    let mapped = ferray_ufunc::helpers::binary_map_op(&a, &b, |x, y| x < y).unwrap();
    assert_eq!(mapped.as_slice().unwrap(), &[true, true, true]);

    let mixed = ferray_ufunc::helpers::binary_mixed_op(&a, &arr1(vec![1_i32, 2, 3]), |x, y| {
        x * f64::from(y)
    })
    .unwrap();
    assert_f64_close(
        mixed.as_slice().unwrap(),
        &[1.0, 8.0, 27.0],
        "binary_mixed_op",
    );

    let matrix = arr2(2, 1, vec![1.0_f64, 2.0]);
    let row = arr1(vec![10.0_f64, 20.0, 30.0]);
    let broadcast =
        ferray_ufunc::helpers::binary_broadcast_op(&matrix, &row, |x, y| x + y).unwrap();
    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_f64_close(
        broadcast.as_slice().unwrap(),
        &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0],
        "binary_broadcast_op",
    );

    let broadcast_map =
        ferray_ufunc::helpers::binary_broadcast_map_op(&matrix, &row, |x, y| x < y).unwrap();
    assert_eq!(broadcast_map.shape(), &[2, 3]);
    assert_eq!(
        broadcast_map.as_slice().unwrap(),
        &[true, true, true, true, true, true]
    );

    let mut out = arr1(vec![0.0_f64, 0.0, 0.0]);
    ferray_ufunc::helpers::unary_float_op_into(&a, &mut out, "sqrt", f64::sqrt).unwrap();
    assert_f64_close(
        out.as_slice().unwrap(),
        &[1.0, 2.0, 3.0],
        "unary_float_op_into",
    );
    ferray_ufunc::helpers::unary_float_op_into_compute(&a, &mut out, "plus1", |x| x + 1.0).unwrap();
    assert_f64_close(
        out.as_slice().unwrap(),
        &[2.0, 5.0, 10.0],
        "unary_float_op_into_compute",
    );
    ferray_ufunc::helpers::binary_elementwise_op_into(&a, &b, &mut out, "add", |x, y| x + y)
        .unwrap();
    assert_f64_close(
        out.as_slice().unwrap(),
        &[11.0, 24.0, 39.0],
        "binary_elementwise_op_into",
    );

    fn add_kernel_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x + y;
        }
    }
    let simd_bin64 = ferray_ufunc::helpers::try_simd_f64_binary(&a, &b, add_kernel_f64)
        .expect("f64 input should use simd binary")
        .unwrap();
    assert_f64_close(
        simd_bin64.as_slice().unwrap(),
        &[11.0, 24.0, 39.0],
        "try_simd_f64_binary",
    );
    assert!(
        ferray_ufunc::helpers::try_simd_f64_binary(&a, &arr1(vec![1.0_f64]), add_kernel_f64)
            .is_none()
    );

    fn add_kernel_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x + y;
        }
    }
    let simd_bin32 = ferray_ufunc::helpers::try_simd_f32_binary(
        &a32,
        &arr1(vec![10.0_f32, 20.0, 30.0]),
        add_kernel_f32,
    )
    .expect("f32 input should use simd binary")
    .unwrap();
    assert_f32_close(
        simd_bin32.as_slice().unwrap(),
        &[11.0, 24.0, 39.0],
        "try_simd_f32_binary",
    );
}

/// Path anchors:
/// `ferray_ufunc::kernels::scalar::binary_map`,
/// `ferray_ufunc::kernels::scalar::unary_map`,
/// `ferray_ufunc::kernels::simd_f32::add_f32`,
/// `ferray_ufunc::kernels::simd_f32::cos_f32`,
/// `ferray_ufunc::kernels::simd_f32::div_f32`,
/// `ferray_ufunc::kernels::simd_f32::exp_f32`,
/// `ferray_ufunc::kernels::simd_f32::log_f32`,
/// `ferray_ufunc::kernels::simd_f32::mul_f32`,
/// `ferray_ufunc::kernels::simd_f32::sin_f32`,
/// `ferray_ufunc::kernels::simd_f32::sqrt_f32`,
/// `ferray_ufunc::kernels::simd_f32::sub_f32`,
/// `ferray_ufunc::kernels::simd_f64::add_f64`,
/// `ferray_ufunc::kernels::simd_f64::cos_f64`,
/// `ferray_ufunc::kernels::simd_f64::div_f64`,
/// `ferray_ufunc::kernels::simd_f64::exp_f64`,
/// `ferray_ufunc::kernels::simd_f64::log_f64`,
/// `ferray_ufunc::kernels::simd_f64::mul_f64`,
/// `ferray_ufunc::kernels::simd_f64::sin_f64`,
/// `ferray_ufunc::kernels::simd_f64::sqrt_f64`,
/// `ferray_ufunc::kernels::simd_f64::sub_f64`,
/// `ferray_ufunc::kernels::simd_i32::add_i32`,
/// `ferray_ufunc::kernels::simd_i32::bitand_i32`,
/// `ferray_ufunc::kernels::simd_i32::bitor_i32`,
/// `ferray_ufunc::kernels::simd_i32::bitxor_i32`,
/// `ferray_ufunc::kernels::simd_i32::mul_i32`,
/// `ferray_ufunc::kernels::simd_i32::sub_i32`,
/// `ferray_ufunc::kernels::simd_i64::add_i64`,
/// `ferray_ufunc::kernels::simd_i64::bitand_i64`,
/// `ferray_ufunc::kernels::simd_i64::bitor_i64`,
/// `ferray_ufunc::kernels::simd_i64::bitxor_i64`,
/// `ferray_ufunc::kernels::simd_i64::mul_i64`,
/// `ferray_ufunc::kernels::simd_i64::sub_i64`.
#[test]
fn scalar_and_simd_kernel_modules_match_reference_loops() {
    let input = [1.0_f64, 4.0, 9.0];
    let mut out64 = [0.0_f64; 3];
    ferray_ufunc::kernels::scalar::unary_map(&input, &mut out64, f64::sqrt);
    assert_f64_close(&out64, &[1.0, 2.0, 3.0], "scalar unary_map");
    ferray_ufunc::kernels::scalar::binary_map(&input, &[10.0, 20.0, 30.0], &mut out64, |x, y| {
        x + y
    });
    assert_f64_close(&out64, &[11.0, 24.0, 39.0], "scalar binary_map");

    let a32 = [1.0_f32, 2.0, 4.0];
    let b32 = [10.0_f32, 20.0, 40.0];
    let mut out32 = [0.0_f32; 3];
    ferray_ufunc::kernels::simd_f32::add_f32(&a32, &b32, &mut out32);
    assert_f32_close(&out32, &[11.0, 22.0, 44.0], "add_f32");
    ferray_ufunc::kernels::simd_f32::sub_f32(&b32, &a32, &mut out32);
    assert_f32_close(&out32, &[9.0, 18.0, 36.0], "sub_f32");
    ferray_ufunc::kernels::simd_f32::mul_f32(&a32, &b32, &mut out32);
    assert_f32_close(&out32, &[10.0, 40.0, 160.0], "mul_f32");
    ferray_ufunc::kernels::simd_f32::div_f32(&b32, &a32, &mut out32);
    assert_f32_close(&out32, &[10.0, 10.0, 10.0], "div_f32");
    ferray_ufunc::kernels::simd_f32::sqrt_f32(&[1.0, 4.0, 9.0], &mut out32);
    assert_f32_close(&out32, &[1.0, 2.0, 3.0], "sqrt_f32");
    ferray_ufunc::kernels::simd_f32::sin_f32(&[0.0, std::f32::consts::FRAC_PI_2, 0.0], &mut out32);
    assert_f32_close(&out32, &[0.0, 1.0, 0.0], "sin_f32");
    ferray_ufunc::kernels::simd_f32::cos_f32(&[0.0, 0.0, std::f32::consts::PI], &mut out32);
    assert_f32_close(&out32, &[1.0, 1.0, -1.0], "cos_f32");
    ferray_ufunc::kernels::simd_f32::exp_f32(&[0.0, 1.0, 2.0], &mut out32);
    assert_f32_close(
        &out32,
        &[1.0, std::f32::consts::E, 2.0_f32.exp()],
        "exp_f32",
    );
    ferray_ufunc::kernels::simd_f32::log_f32(&[1.0, std::f32::consts::E, 4.0], &mut out32);
    assert_f32_close(&out32, &[0.0, 1.0, 4.0_f32.ln()], "log_f32");

    let a64 = [1.0_f64, 2.0, 4.0];
    let b64 = [10.0_f64, 20.0, 40.0];
    ferray_ufunc::kernels::simd_f64::add_f64(&a64, &b64, &mut out64);
    assert_f64_close(&out64, &[11.0, 22.0, 44.0], "add_f64");
    ferray_ufunc::kernels::simd_f64::sub_f64(&b64, &a64, &mut out64);
    assert_f64_close(&out64, &[9.0, 18.0, 36.0], "sub_f64");
    ferray_ufunc::kernels::simd_f64::mul_f64(&a64, &b64, &mut out64);
    assert_f64_close(&out64, &[10.0, 40.0, 160.0], "mul_f64");
    ferray_ufunc::kernels::simd_f64::div_f64(&b64, &a64, &mut out64);
    assert_f64_close(&out64, &[10.0, 10.0, 10.0], "div_f64");
    ferray_ufunc::kernels::simd_f64::sqrt_f64(&[1.0, 4.0, 9.0], &mut out64);
    assert_f64_close(&out64, &[1.0, 2.0, 3.0], "sqrt_f64");
    ferray_ufunc::kernels::simd_f64::sin_f64(&[0.0, std::f64::consts::FRAC_PI_2, 0.0], &mut out64);
    assert_f64_close(&out64, &[0.0, 1.0, 0.0], "sin_f64");
    ferray_ufunc::kernels::simd_f64::cos_f64(&[0.0, 0.0, std::f64::consts::PI], &mut out64);
    assert_f64_close(&out64, &[1.0, 1.0, -1.0], "cos_f64");
    ferray_ufunc::kernels::simd_f64::exp_f64(&[0.0, 1.0, 2.0], &mut out64);
    assert_f64_close(
        &out64,
        &[1.0, std::f64::consts::E, 2.0_f64.exp()],
        "exp_f64",
    );
    ferray_ufunc::kernels::simd_f64::log_f64(&[1.0, std::f64::consts::E, 4.0], &mut out64);
    assert_f64_close(&out64, &[0.0, 1.0, 4.0_f64.ln()], "log_f64");

    let ai32 = [i32::MAX, 0b1100, 7];
    let bi32 = [1_i32, 0b1010, 3];
    let mut oi32 = [0_i32; 3];
    ferray_ufunc::kernels::simd_i32::add_i32(&ai32, &bi32, &mut oi32);
    assert_eq!(oi32, [i32::MIN, 22, 10]);
    ferray_ufunc::kernels::simd_i32::sub_i32(&ai32, &bi32, &mut oi32);
    assert_eq!(oi32, [i32::MAX - 1, 2, 4]);
    ferray_ufunc::kernels::simd_i32::mul_i32(&ai32, &bi32, &mut oi32);
    assert_eq!(oi32, [i32::MAX, 120, 21]);
    ferray_ufunc::kernels::simd_i32::bitand_i32(&ai32, &bi32, &mut oi32);
    assert_eq!(oi32, [1, 0b1000, 3]);
    ferray_ufunc::kernels::simd_i32::bitor_i32(&ai32, &bi32, &mut oi32);
    assert_eq!(oi32, [i32::MAX, 0b1110, 7]);
    ferray_ufunc::kernels::simd_i32::bitxor_i32(&ai32, &bi32, &mut oi32);
    assert_eq!(oi32, [i32::MAX - 1, 0b0110, 4]);

    let ai64 = [i64::MAX, 0b1100, 7];
    let bi64 = [1_i64, 0b1010, 3];
    let mut oi64 = [0_i64; 3];
    ferray_ufunc::kernels::simd_i64::add_i64(&ai64, &bi64, &mut oi64);
    assert_eq!(oi64, [i64::MIN, 22, 10]);
    ferray_ufunc::kernels::simd_i64::sub_i64(&ai64, &bi64, &mut oi64);
    assert_eq!(oi64, [i64::MAX - 1, 2, 4]);
    ferray_ufunc::kernels::simd_i64::mul_i64(&ai64, &bi64, &mut oi64);
    assert_eq!(oi64, [i64::MAX, 120, 21]);
    ferray_ufunc::kernels::simd_i64::bitand_i64(&ai64, &bi64, &mut oi64);
    assert_eq!(oi64, [1, 0b1000, 3]);
    ferray_ufunc::kernels::simd_i64::bitor_i64(&ai64, &bi64, &mut oi64);
    assert_eq!(oi64, [i64::MAX, 0b1110, 7]);
    ferray_ufunc::kernels::simd_i64::bitxor_i64(&ai64, &bi64, &mut oi64);
    assert_eq!(oi64, [i64::MAX - 1, 0b0110, 4]);
}

#[cfg(feature = "f16")]
#[test]
fn f16_helpers_and_dispatch_match_f32_promoted_reference() {
    let a = Array::<half::f16, Ix1>::from_vec(
        Ix1::new([3]),
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(9.0),
        ],
    )
    .unwrap();
    let b = Array::<half::f16, Ix1>::from_vec(
        Ix1::new([3]),
        vec![
            half::f16::from_f32(10.0),
            half::f16::from_f32(20.0),
            half::f16::from_f32(30.0),
        ],
    )
    .unwrap();

    let unary = ferray_ufunc::helpers::unary_f16_op(&a, f32::sqrt).unwrap();
    let unary_values: Vec<f32> = unary.iter().map(|x| x.to_f32()).collect();
    assert_f32_close(&unary_values, &[1.0, 2.0, 3.0], "unary_f16_op");

    let unary_bool = ferray_ufunc::helpers::unary_f16_to_bool_op(&a, |x| x > 3.0).unwrap();
    assert_eq!(unary_bool.as_slice().unwrap(), &[false, true, true]);

    let binary = ferray_ufunc::helpers::binary_f16_op(&a, &b, |x, y| x + y).unwrap();
    let binary_values: Vec<f32> = binary.iter().map(|x| x.to_f32()).collect();
    assert_f32_close(&binary_values, &[11.0, 24.0, 39.0], "binary_f16_op");

    let binary_bool = ferray_ufunc::helpers::binary_f16_to_bool_op(&a, &b, |x, y| x < y).unwrap();
    assert_eq!(binary_bool.as_slice().unwrap(), &[true, true, true]);

    let input = [
        half::f16::from_f32(1.0),
        half::f16::from_f32(4.0),
        half::f16::from_f32(9.0),
    ];
    let mut output = [half::f16::ZERO; 3];
    ferray_ufunc::dispatch::dispatch_unary_f16(&input, &mut output, f32::sqrt);
    let output_values: Vec<f32> = output.iter().map(|x| x.to_f32()).collect();
    assert_f32_close(&output_values, &[1.0, 2.0, 3.0], "dispatch_unary_f16");

    ferray_ufunc::dispatch::dispatch_binary_f16(&input, &input, &mut output, |x, y| x + y);
    let output_values: Vec<f32> = output.iter().map(|x| x.to_f32()).collect();
    assert_f32_close(&output_values, &[2.0, 8.0, 18.0], "dispatch_binary_f16");
}

/// Path anchor:
/// `ferray_ufunc::test_util::arr1`.
///
/// The test utility module is compiled only inside ferray-ufunc unit tests, so
/// integration tests cannot call it directly. Keeping the anchor here lets the
/// strict surface gate distinguish that cfg-test helper from runtime API debt.
#[test]
fn cfg_test_only_utility_surface_is_documented() {
    let dyn_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![1.0]).unwrap();
    assert_eq!(dyn_arr.shape(), &[1]);
}
