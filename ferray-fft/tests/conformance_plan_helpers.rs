//! Conformance coverage for FFT plan handles, extension helpers, and
//! public-inventory helper paths that do not map to a direct NumPy fixture.

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_fft::FftNorm;
use num_complex::Complex;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (actual.re - expected.re).abs() < 1e-10,
        "real mismatch: got {}, expected {}",
        actual.re,
        expected.re
    );
    assert!(
        (actual.im - expected.im).abs() < 1e-10,
        "imag mismatch: got {}, expected {}",
        actual.im,
        expected.im
    );
}

fn assert_real_slices_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-9,
            "real slice mismatch at {i}: got {a}, expected {e}"
        );
    }
}

/// Path anchors:
/// `ferray_fft::FftPlan`, `ferray_fft::plan::FftPlan`,
/// `ferray_fft::plan::FftPlan::new`,
/// `ferray_fft::plan::FftPlan::size`,
/// `ferray_fft::plan::FftPlan::execute`,
/// `ferray_fft::plan::FftPlan::execute_inverse`,
/// `ferray_fft::plan::FftPlan::execute_with_norm`,
/// `ferray_fft::plan::FftPlan::execute_inverse_with_norm`,
/// `ferray_fft::FftPlanND`, `ferray_fft::plan::FftPlanND`,
/// `ferray_fft::plan::FftPlanND::new`,
/// `ferray_fft::plan::FftPlanND::shape`,
/// `ferray_fft::plan::FftPlanND::axes`,
/// `ferray_fft::plan::FftPlanND::num_axis_plans`,
/// `ferray_fft::plan::FftPlanND::forward_axis_plan`,
/// `ferray_fft::plan::FftPlanND::inverse_axis_plan`,
/// `ferray_fft::plan::FftPlanND::execute`,
/// `ferray_fft::plan::FftPlanND::execute_inverse`.
#[test]
fn plan_surfaces_match_uncached_fft_paths() {
    let data = vec![c(1.0, 0.0), c(0.0, 1.0), c(-2.0, 0.5), c(3.0, -1.0)];
    let signal = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([4]), data.clone()).unwrap();

    let root_plan: ferray_fft::FftPlan = ferray_fft::FftPlan::new(4).unwrap();
    let inner_plan: ferray_fft::plan::FftPlan = ferray_fft::plan::FftPlan::new(4).unwrap();
    assert_eq!(root_plan.size(), 4);
    assert_eq!(inner_plan.size(), 4);

    let from_plan = root_plan.execute(&signal).unwrap();
    let from_inner = inner_plan
        .execute_with_norm(&signal, FftNorm::Backward)
        .unwrap();
    let from_fn = ferray_fft::fft(&signal, None, None, FftNorm::Backward).unwrap();
    for ((p, i), f) in from_plan.iter().zip(from_inner.iter()).zip(from_fn.iter()) {
        assert_complex_close(*p, *f);
        assert_complex_close(*i, *f);
    }

    let recovered = root_plan.execute_inverse(&from_plan).unwrap();
    let recovered_inner = inner_plan
        .execute_inverse_with_norm(&from_inner, FftNorm::Backward)
        .unwrap();
    for ((r, ri), expected) in recovered
        .iter()
        .zip(recovered_inner.iter())
        .zip(data.iter())
    {
        assert_complex_close(*r, *expected);
        assert_complex_close(*ri, *expected);
    }

    let matrix_data: Vec<Complex<f64>> = (0..6).map(|i| c(i as f64, -(i as f64) / 2.0)).collect();
    let matrix =
        Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([2, 3]), matrix_data.clone()).unwrap();

    let nd_plan: ferray_fft::FftPlanND = ferray_fft::FftPlanND::new(&[2, 3], None).unwrap();
    let inner_nd_plan: ferray_fft::plan::FftPlanND =
        ferray_fft::plan::FftPlanND::new(&[2, 3], Some(&[-2, -1])).unwrap();
    assert_eq!(nd_plan.shape(), &[2, 3]);
    assert_eq!(nd_plan.axes(), &[0, 1]);
    assert_eq!(inner_nd_plan.axes(), &[0, 1]);
    assert_eq!(nd_plan.num_axis_plans(), nd_plan.axes().len());
    assert!(nd_plan.forward_axis_plan(0).is_some());
    assert!(nd_plan.inverse_axis_plan(1).is_some());
    assert!(nd_plan.forward_axis_plan(2).is_none());

    let planned_nd = nd_plan.execute(&matrix, FftNorm::Backward).unwrap();
    let planned_inner_nd = inner_nd_plan.execute(&matrix, FftNorm::Backward).unwrap();
    let direct_nd = ferray_fft::fftn(&matrix, None, None, FftNorm::Backward).unwrap();
    for ((p, pi), d) in planned_nd
        .iter()
        .zip(planned_inner_nd.iter())
        .zip(direct_nd.iter())
    {
        assert_complex_close(*p, *d);
        assert_complex_close(*pi, *d);
    }

    let inverse_nd = nd_plan
        .execute_inverse(&planned_nd, FftNorm::Backward)
        .unwrap();
    for (actual, expected) in inverse_nd.iter().zip(matrix_data.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

/// Path anchors:
/// `ferray_fft::hfft2`, `ferray_fft::hermitian::hfft2`,
/// `ferray_fft::ihfft2`, `ferray_fft::hermitian::ihfft2`,
/// `ferray_fft::hfftn`, `ferray_fft::hermitian::hfftn`,
/// `ferray_fft::ihfftn`, `ferray_fft::hermitian::ihfftn`.
#[test]
fn hermitian_extension_helpers_roundtrip_real_inputs() {
    let original: Vec<f64> = (0..16).map(|i| i as f64 - 3.0).collect();
    let real = Array::<f64, Ix2>::from_vec(Ix2::new([4, 4]), original.clone()).unwrap();

    let root_spectrum = ferray_fft::ihfft2(&real, None, None, FftNorm::Backward).unwrap();
    let root_recovered =
        ferray_fft::hfft2(&root_spectrum, Some(&[4, 4]), None, FftNorm::Backward).unwrap();
    assert_eq!(root_spectrum.shape(), &[4, 3]);
    assert_eq!(root_recovered.shape(), &[4, 4]);
    assert_real_slices_close(root_recovered.as_slice().unwrap(), &original);

    let inner_spectrum =
        ferray_fft::hermitian::ihfftn(&real, None, None, FftNorm::Backward).unwrap();
    let inner_recovered =
        ferray_fft::hermitian::hfftn(&inner_spectrum, Some(&[4, 4]), None, FftNorm::Backward)
            .unwrap();
    assert_eq!(inner_spectrum.shape(), &[4, 3]);
    assert_eq!(inner_recovered.shape(), &[4, 4]);
    assert_real_slices_close(inner_recovered.as_slice().unwrap(), &original);

    let inner_2d_spectrum =
        ferray_fft::hermitian::ihfft2(&real, None, None, FftNorm::Backward).unwrap();
    let inner_2d_recovered =
        ferray_fft::hermitian::hfft2(&inner_2d_spectrum, Some(&[4, 4]), None, FftNorm::Backward)
            .unwrap();
    assert_real_slices_close(inner_2d_recovered.as_slice().unwrap(), &original);
}

/// Path anchors:
/// `ferray_fft::axes::normalize_axis`, `ferray_fft::axes::resolve_axis`,
/// `ferray_fft::axes::resolve_axes`, `ferray_fft::axes::compute_strides`,
/// `ferray_fft::nd::fft_along_axis`, `ferray_fft::nd::fft_along_axes`,
/// `ferray_fft::nd::rfft_along_axis`, `ferray_fft::nd::irfft_along_axis`.
///
/// The `axes` and `nd` modules are private implementation modules; this
/// test drives their public behavior through negative-axis, default-axis,
/// multi-axis, and real-transform public FFT entry points.
#[test]
fn axis_and_nd_helpers_are_exercised_by_public_fft_paths() {
    let data: Vec<Complex<f64>> = (0..12).map(|i| c(i as f64, i as f64 / 10.0)).collect();
    let complex = Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([3, 4]), data).unwrap();

    let last_axis_negative = ferray_fft::fft(&complex, None, Some(-1), FftNorm::Backward).unwrap();
    let last_axis_positive = ferray_fft::fft(&complex, None, Some(1), FftNorm::Backward).unwrap();
    for (negative, positive) in last_axis_negative.iter().zip(last_axis_positive.iter()) {
        assert_complex_close(*negative, *positive);
    }

    let all_axes_default = ferray_fft::fftn(&complex, None, None, FftNorm::Backward).unwrap();
    let all_axes_explicit =
        ferray_fft::fftn(&complex, None, Some(&[0, -1]), FftNorm::Backward).unwrap();
    for (default, explicit) in all_axes_default.iter().zip(all_axes_explicit.iter()) {
        assert_complex_close(*default, *explicit);
    }

    let real_data: Vec<f64> = (0..12).map(|i| i as f64 - 2.0).collect();
    let real = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), real_data.clone()).unwrap();
    let spectrum = ferray_fft::rfftn(&real, None, None, FftNorm::Backward).unwrap();
    assert_eq!(spectrum.shape(), &[3, 3]);
    let recovered = ferray_fft::irfftn(&spectrum, Some(&[3, 4]), None, FftNorm::Backward).unwrap();
    assert_eq!(recovered.shape(), &[3, 4]);
    assert_real_slices_close(recovered.as_slice().unwrap(), &real_data);
}

/// Path anchors:
/// `ferray_fft::float::FftFloat`, `ferray_fft::norm::FftNorm`,
/// `ferray_fft::norm::FftNorm::scale_factor_f64`,
/// `ferray_fft::norm::FftDirection`.
#[test]
fn norm_and_float_type_system_surface_is_exercised() {
    use ferray_fft::float::FftFloat;
    use ferray_fft::norm::{FftDirection, FftNorm as InnerNorm};

    fn requires_fft_float<T>()
    where
        T: FftFloat,
        Complex<T>: ferray_core::dtype::Element,
    {
    }
    requires_fft_float::<f32>();
    requires_fft_float::<f64>();

    assert_eq!(
        InnerNorm::Backward.scale_factor_f64(8, FftDirection::Forward),
        1.0
    );
    assert_eq!(
        InnerNorm::Backward.scale_factor_f64(8, FftDirection::Inverse),
        0.125
    );
    assert_eq!(
        InnerNorm::Forward.scale_factor_f64(8, FftDirection::Forward),
        0.125
    );
    assert_eq!(
        InnerNorm::Forward.scale_factor_f64(8, FftDirection::Inverse),
        1.0
    );
    assert!(
        (InnerNorm::Ortho.scale_factor_f64(9, FftDirection::Forward) - (1.0 / 3.0)).abs() < 1e-15
    );
}
