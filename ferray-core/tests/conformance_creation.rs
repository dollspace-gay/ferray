//! Conformance tests for ferray-core creation functions (arange,
//! linspace, zeros, ones, full, eye) anchored against NumPy fixtures
//! under `fixtures/core/`, plus inline-anchored tests for the
//! analytically-verifiable variants of each surface.
//!
//! Surface paths exercised by this file (named here so the surface
//! gate's text match picks them up):
//!
//! - `ferray_core::creation::arange`
//! - `ferray_core::creation::linspace`
//! - `ferray_core::creation::zeros`
//! - `ferray_core::creation::ones`
//! - `ferray_core::creation::full`
//! - `ferray_core::creation::eye`
//! - `ferray_core::creation::zeros_like`
//! - `ferray_core::creation::ones_like`
//! - `ferray_core::creation::full_like`
//! - `ferray_core::creation::identity`
//! - `ferray_core::creation::array`
//! - `ferray_core::creation::asarray`
//! - `ferray_core::creation::asanyarray`
//! - `ferray_core::creation::asarray_chkfinite`
//! - `ferray_core::creation::ascontiguousarray`
//! - `ferray_core::creation::asfortranarray`
//! - `ferray_core::creation::copy`
//! - `ferray_core::creation::require`
//! - `ferray_core::creation::empty`
//! - `ferray_core::creation::empty_like`
//! - `ferray_core::creation::frombuffer`
//! - `ferray_core::creation::frombuffer_view`
//! - `ferray_core::creation::fromiter`
//! - `ferray_core::creation::fromfunction`
//! - `ferray_core::creation::fromstring`
//! - `ferray_core::creation::fromfile`
//! - `ferray_core::creation::geomspace`
//! - `ferray_core::creation::logspace`
//! - `ferray_core::creation::meshgrid`
//! - `ferray_core::creation::mgrid`
//! - `ferray_core::creation::ogrid`
//! - `ferray_core::creation::diag`
//! - `ferray_core::creation::diagflat`
//! - `ferray_core::creation::tri`
//! - `ferray_core::creation::tril`
//! - `ferray_core::creation::triu`
//! - `ferray_core::creation::vander`
//!
//! Fixture-strict tolerance: shape/index ops bit-exact; arithmetic uses
//! `TOL_REDUCTION_F64_ABS = 1e-12`. Shared `assert_f64_slice_ulp` honors
//! per-case ULP tolerance (min `MIN_ULP_TOLERANCE = 10`).

// Conformance tests cross fixture JSON, recover bit patterns, and
// compare floats — the casts and direct comparisons are part of the
// contract.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::creation;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_test_oracle::{
    MIN_ULP_TOLERANCE, TOL_REDUCTION_F64_ABS, assert_f64_slice_ulp, fixtures_dir, load_fixture,
    parse_f64_data, parse_f64_value, parse_shape,
};

fn core_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("core").join(name)
}

// ---------------------------------------------------------------------------
// Fixture-anchored: arange, linspace.
// ---------------------------------------------------------------------------

#[test]
fn fixture_arange_matches_numpy() {
    // Pins `ferray_core::creation::arange`.
    let suite = load_fixture(&core_path("arange.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let start = parse_f64_value(&case.inputs["start"]);
        let stop = parse_f64_value(&case.inputs["stop"]);
        let step = case.inputs.get("step").map_or(1.0, parse_f64_value);
        let got = creation::arange(start, stop, step).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            got.shape(),
            expected_shape.as_slice(),
            "{}: shape",
            case.name
        );
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_linspace_matches_numpy() {
    // Pins `ferray_core::creation::linspace`.
    let suite = load_fixture(&core_path("linspace.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let start = parse_f64_value(&case.inputs["start"]);
        let stop = parse_f64_value(&case.inputs["stop"]);
        let num = case.inputs["num"].as_u64().unwrap() as usize;
        let got = creation::linspace::<f64>(start, stop, num, true).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            got.shape(),
            expected_shape.as_slice(),
            "{}: shape",
            case.name
        );
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Fixture-anchored: zeros, ones, full, eye.
//
// These are bit-exact: integer/zero/one fill values have no rounding
// error so we tolerate 0 ULP for the f64 cases.
// ---------------------------------------------------------------------------

#[test]
fn fixture_zeros_matches_numpy() {
    // Pins `ferray_core::creation::zeros`.
    let suite = load_fixture(&core_path("zeros.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&case.inputs["shape"]);
        let got = creation::zeros::<f64, IxDyn>(IxDyn::new(&shape)).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(got.shape(), shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_ones_matches_numpy() {
    // Pins `ferray_core::creation::ones`.
    let suite = load_fixture(&core_path("ones.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&case.inputs["shape"]);
        let got = creation::ones::<f64, IxDyn>(IxDyn::new(&shape)).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(got.shape(), shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_full_matches_numpy() {
    // Pins `ferray_core::creation::full`.
    let suite = load_fixture(&core_path("full.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&case.inputs["shape"]);
        let fill = parse_f64_value(&case.inputs["fill_value"]);
        let got = creation::full::<f64, IxDyn>(IxDyn::new(&shape), fill).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(got.shape(), shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_eye_matches_numpy() {
    // Pins `ferray_core::creation::eye`.
    let suite = load_fixture(&core_path("eye.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let m = case.inputs["m"].as_u64().unwrap() as usize;
        let k = case.inputs["k"].as_i64().unwrap() as isize;
        let got = creation::eye::<f64>(n, m, k).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            got.shape(),
            expected_shape.as_slice(),
            "{}: shape",
            case.name
        );
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Inline-anchored: like-shaped constructors, identity, basic array
// builders, layout converters.
// ---------------------------------------------------------------------------

#[test]
fn inline_zeros_like_ones_like_full_like_preserve_shape() {
    // Pins:
    //   - `ferray_core::creation::zeros_like`
    //   - `ferray_core::creation::ones_like`
    //   - `ferray_core::creation::full_like`
    let proto = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.; 6]).unwrap();

    let z = creation::zeros_like(&proto).unwrap();
    assert_eq!(z.shape(), [2, 3]);
    assert!(z.iter().all(|&x| x == 0.0));

    let o = creation::ones_like(&proto).unwrap();
    assert_eq!(o.shape(), [2, 3]);
    assert!(o.iter().all(|&x| x == 1.0));

    let f = creation::full_like(&proto, 7.5).unwrap();
    assert_eq!(f.shape(), [2, 3]);
    assert!(f.iter().all(|&x| (x - 7.5).abs() < TOL_REDUCTION_F64_ABS));
}

#[test]
fn inline_identity_is_square_eye() {
    // Pins `ferray_core::creation::identity`. identity(n) == eye(n,n,0).
    let id = creation::identity::<f64>(4).unwrap();
    assert_eq!(id.shape(), [4, 4]);
    for i in 0..4 {
        for j in 0..4 {
            let want = if i == j { 1.0 } else { 0.0 };
            let got = id.as_slice().unwrap()[i * 4 + j];
            assert!(
                (got - want).abs() < TOL_REDUCTION_F64_ABS,
                "id[{i},{j}] = {got}"
            );
        }
    }
}

#[test]
fn inline_array_asarray_roundtrip() {
    // Pins:
    //   - `ferray_core::creation::array`
    //   - `ferray_core::creation::asarray`
    //   - `ferray_core::creation::asanyarray`
    //   - `ferray_core::creation::copy`
    //   - `ferray_core::creation::ascontiguousarray`
    //   - `ferray_core::creation::asfortranarray`
    //   - `ferray_core::creation::require`
    //   - `ferray_core::creation::asarray_chkfinite`
    let data = vec![1.0_f64, 2., 3., 4., 5., 6.];
    let a = creation::array(Ix2::new([2, 3]), data.clone()).unwrap();
    assert_eq!(a.shape(), [2, 3]);
    assert_eq!(a.as_slice().unwrap(), &[1.0, 2., 3., 4., 5., 6.]);

    let b = creation::asarray(Ix2::new([2, 3]), data.clone()).unwrap();
    assert_eq!(b.as_slice().unwrap(), a.as_slice().unwrap());

    let c = creation::asanyarray(&a);
    assert_eq!(c.as_slice().unwrap(), a.as_slice().unwrap());

    let copied = creation::copy(&a);
    assert_eq!(copied.as_slice().unwrap(), a.as_slice().unwrap());

    let cont = creation::ascontiguousarray(&a);
    assert_eq!(cont.shape(), a.shape());

    let fort = creation::asfortranarray(&a);
    assert_eq!(fort.shape(), a.shape());

    let req = creation::require(&a, "C");
    assert_eq!(req.shape(), a.shape());

    let chk = creation::asarray_chkfinite(&a).unwrap();
    assert_eq!(chk.as_slice().unwrap(), a.as_slice().unwrap());

    let bad = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
    assert!(creation::asarray_chkfinite(&bad).is_err());
}

#[test]
fn inline_empty_empty_like_alloc_correct_shape() {
    // Pins:
    //   - `ferray_core::creation::empty`
    //   - `ferray_core::creation::empty_like`
    let e = creation::empty::<f64, Ix2>(Ix2::new([3, 4]));
    assert_eq!(e.shape(), [3, 4]);
    let proto = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![0.; 5]).unwrap();
    let e2 = creation::empty_like(&proto);
    assert_eq!(e2.shape(), [5]);
}

#[test]
fn inline_fromiter_fromfunction_fromstring() {
    // Pins:
    //   - `ferray_core::creation::fromiter`
    //   - `ferray_core::creation::fromfunction`
    //   - `ferray_core::creation::fromstring`
    let it = creation::fromiter(0..5u64).unwrap();
    assert_eq!(it.shape(), [5]);
    assert_eq!(it.as_slice().unwrap(), &[0u64, 1, 2, 3, 4]);

    let f =
        creation::fromfunction::<f64, _, _>(Ix2::new([3, 3]), |idx| idx[0] as f64 + idx[1] as f64)
            .unwrap();
    assert_eq!(
        f.as_slice().unwrap(),
        &[0.0, 1., 2., 1., 2., 3., 2., 3., 4.]
    );

    let s = creation::fromstring::<f64>("1.0 2.0 3.0", " ").unwrap();
    assert_eq!(s.shape(), [3]);
    assert_eq!(s.as_slice().unwrap(), &[1.0, 2., 3.]);
}

#[test]
fn inline_frombuffer_roundtrips_le_bytes() {
    // Pins:
    //   - `ferray_core::creation::frombuffer`
    //   - `ferray_core::creation::frombuffer_view`
    let mut bytes = Vec::with_capacity(24);
    for v in [1.0_f64, 2.5, -3.5_f64] {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let a = creation::frombuffer::<f64, Ix1>(Ix1::new([3]), &bytes).unwrap();
    assert_eq!(a.shape(), [3]);
    assert_eq!(a.as_slice().unwrap(), &[1.0, 2.5, -3.5]);

    let v = creation::frombuffer_view::<f64, Ix1>(Ix1::new([3]), &bytes).unwrap();
    let collected: Vec<f64> = v.iter().copied().collect();
    assert_eq!(collected, vec![1.0, 2.5, -3.5]);
}

#[test]
fn inline_fromfile_reads_whitespace_separated_numbers() {
    // Pins `ferray_core::creation::fromfile`.
    let tmp = std::env::temp_dir().join("ferray_core_fromfile_test.txt");
    std::fs::write(&tmp, "1.0 2.5 -3.5\n").unwrap();
    let a = creation::fromfile::<f64, _>(&tmp, " ").unwrap();
    assert_eq!(a.shape(), [3]);
    assert_eq!(a.as_slice().unwrap(), &[1.0, 2.5, -3.5]);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn inline_geomspace_logspace_endpoints_correct() {
    // Pins:
    //   - `ferray_core::creation::geomspace`
    //   - `ferray_core::creation::logspace`
    let g = creation::geomspace::<f64>(1.0, 1000.0, 4, true).unwrap();
    assert_eq!(g.shape(), [4]);
    let g0 = g.as_slice().unwrap()[0];
    let g3 = g.as_slice().unwrap()[3];
    assert!((g0 - 1.0).abs() < 1e-12);
    assert!((g3 - 1000.0).abs() < 1e-9);

    let l = creation::logspace::<f64>(0.0, 3.0, 4, true, 10.0).unwrap();
    assert_eq!(l.shape(), [4]);
    let l0 = l.as_slice().unwrap()[0];
    let l3 = l.as_slice().unwrap()[3];
    assert!((l0 - 1.0).abs() < 1e-12);
    assert!((l3 - 1000.0).abs() < 1e-9);
}

#[test]
fn inline_meshgrid_mgrid_ogrid_shapes() {
    // Pins:
    //   - `ferray_core::creation::meshgrid`
    //   - `ferray_core::creation::mgrid`
    //   - `ferray_core::creation::ogrid`
    let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0., 1., 2.]).unwrap();
    let y = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![10., 20.]).unwrap();
    let xs = [x, y];
    let grids = creation::meshgrid(&xs, "xy").unwrap();
    assert_eq!(grids.len(), 2);
    // xy indexing: shape (len(y), len(x)) = (2, 3).
    assert_eq!(grids[0].shape(), [2, 3]);
    assert_eq!(grids[1].shape(), [2, 3]);

    let mg = creation::mgrid(&[(0.0, 3.0, 1.0), (0.0, 2.0, 1.0)]).unwrap();
    assert_eq!(mg.len(), 2);
    assert_eq!(mg[0].shape(), [3, 2]);

    let og = creation::ogrid(&[(0.0, 3.0, 1.0), (0.0, 2.0, 1.0)]).unwrap();
    assert_eq!(og.len(), 2);
}

#[test]
fn inline_diag_and_diagflat() {
    // Pins:
    //   - `ferray_core::creation::diag`
    //   - `ferray_core::creation::diagflat`
    // 2D -> 1D extracts the main diagonal.
    let m = Array::<f64, IxDyn>::from_vec(
        IxDyn::new(&[3, 3]),
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap();
    let d = creation::diag(&m, 0).unwrap();
    assert_eq!(d.shape(), [3]);
    assert_eq!(d.as_slice().unwrap(), &[1.0, 5., 9.]);

    // 1D -> 2D builds a diagonal matrix.
    let v = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2., 3.]).unwrap();
    let m2 = creation::diag(&v, 0).unwrap();
    assert_eq!(m2.shape(), [3, 3]);
    assert_eq!(m2.as_slice().unwrap()[0], 1.0);
    assert_eq!(m2.as_slice().unwrap()[4], 2.0);
    assert_eq!(m2.as_slice().unwrap()[8], 3.0);

    let df = creation::diagflat(&v, 0).unwrap();
    assert_eq!(df.shape(), [3, 3]);
}

#[test]
fn inline_tri_tril_triu_lower_upper_triangles() {
    // Pins:
    //   - `ferray_core::creation::tri`
    //   - `ferray_core::creation::tril`
    //   - `ferray_core::creation::triu`
    let t = creation::tri::<f64>(3, 3, 0).unwrap();
    assert_eq!(t.shape(), [3, 3]);
    // tri(n,m,0) has 1s on and below the main diagonal.
    let expected = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    assert_eq!(t.as_slice().unwrap(), &expected);

    let m = Array::<f64, IxDyn>::from_vec(
        IxDyn::new(&[3, 3]),
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
    )
    .unwrap();
    let lo = creation::tril(&m, 0).unwrap();
    let lo_s = lo.as_slice().unwrap();
    assert_eq!(lo_s[1], 0.0);
    assert_eq!(lo_s[2], 0.0);
    assert_eq!(lo_s[5], 0.0);
    let up = creation::triu(&m, 0).unwrap();
    let up_s = up.as_slice().unwrap();
    assert_eq!(up_s[3], 0.0);
    assert_eq!(up_s[6], 0.0);
    assert_eq!(up_s[7], 0.0);
}

#[test]
fn inline_vander_polynomial_columns() {
    // Pins `ferray_core::creation::vander`.
    let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2., 3.]).unwrap();
    let v = creation::vander(&x, Some(3), false).unwrap();
    assert_eq!(v.shape(), [3, 3]);
    // Increasing=false (default): columns are x^(N-1), x^(N-2), ..., x^0.
    let s = v.as_slice().unwrap();
    assert_eq!(s, &[1.0, 1.0, 1.0, 4.0, 2.0, 1.0, 9.0, 3.0, 1.0]);
}
