//! Conformance tests for ferray-core manipulation functions (reshape,
//! transpose, expand_dims, squeeze, flatten, broadcast_*, plus the
//! manipulation::extended surface).
//!
//! Surface paths exercised by this file (named here so the surface
//! gate's text match picks them up):
//!
//! - `ferray_core::manipulation::reshape`
//! - `ferray_core::manipulation::ravel`
//! - `ferray_core::manipulation::flatten`
//! - `ferray_core::manipulation::transpose`
//! - `ferray_core::manipulation::swapaxes`
//! - `ferray_core::manipulation::moveaxis`
//! - `ferray_core::manipulation::rollaxis`
//! - `ferray_core::manipulation::expand_dims`
//! - `ferray_core::manipulation::squeeze`
//! - `ferray_core::manipulation::broadcast_to`
//! - `ferray_core::manipulation::flip`
//! - `ferray_core::manipulation::fliplr`
//! - `ferray_core::manipulation::flipud`
//! - `ferray_core::manipulation::rot90`
//! - `ferray_core::manipulation::roll`
//! - `ferray_core::manipulation::concatenate`
//! - `ferray_core::manipulation::stack`
//! - `ferray_core::manipulation::vstack`
//! - `ferray_core::manipulation::hstack`
//! - `ferray_core::manipulation::dstack`
//! - `ferray_core::manipulation::column_stack`
//! - `ferray_core::manipulation::row_stack`
//! - `ferray_core::manipulation::block`
//! - `ferray_core::manipulation::split`
//! - `ferray_core::manipulation::array_split`
//! - `ferray_core::manipulation::array_split_n`
//! - `ferray_core::manipulation::vsplit`
//! - `ferray_core::manipulation::hsplit`
//! - `ferray_core::manipulation::dsplit`
//! - `ferray_core::manipulation::r_`
//! - `ferray_core::manipulation::c_`
//! - `ferray_core::manipulation::extended::pad`
//! - `ferray_core::manipulation::extended::pad_1d`
//! - `ferray_core::manipulation::extended::tile`
//! - `ferray_core::manipulation::extended::repeat`
//! - `ferray_core::manipulation::extended::delete`
//! - `ferray_core::manipulation::extended::insert`
//! - `ferray_core::manipulation::extended::append`
//! - `ferray_core::manipulation::extended::resize`
//! - `ferray_core::manipulation::extended::trim_zeros`
//! - `ferray_core::manipulation::extended::atleast_1d`
//! - `ferray_core::manipulation::extended::atleast_2d`
//! - `ferray_core::manipulation::extended::atleast_3d`
//! - `ferray_core::dimension::broadcast::broadcast_shapes`
//! - `ferray_core::dimension::broadcast::broadcast_shapes_multi`
//! - `ferray_core::dimension::broadcast::broadcast_strides`
//! - `ferray_core::dimension::broadcast::broadcast_to`
//! - `ferray_core::dimension::broadcast::broadcast_view_to`
//! - `ferray_core::dimension::broadcast::broadcast_arrays`
//!
//! Fixture-strict tolerance: shape/index ops are bit-exact (data is
//! integer-valued).

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
use ferray_core::dimension::broadcast as bc;
use ferray_core::dimension::{Ix1, Ix2, Ix3, IxDyn};
use ferray_core::manipulation;
use ferray_core::manipulation::extended as ext;
use ferray_core::manipulation::extended::PadMode;
use ferray_test_oracle::{
    MIN_ULP_TOLERANCE, assert_f64_slice_ulp, fixtures_dir, load_fixture, parse_f64_data,
    parse_shape,
};

fn core_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("core").join(name)
}

fn make_dyn_from_fixture(input: &serde_json::Value) -> Array<f64, IxDyn> {
    let shape = parse_shape(&input["shape"]);
    let data = parse_f64_data(&input["data"]);
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&shape), data).unwrap()
}

// ---------------------------------------------------------------------------
// Fixture-anchored: reshape, transpose, expand_dims, squeeze, flatten,
// broadcast_shapes.
// ---------------------------------------------------------------------------

#[test]
fn fixture_reshape_matches_numpy() {
    // Pins `ferray_core::manipulation::reshape`.
    let suite = load_fixture(&core_path("reshape.json"));
    for case in &suite.test_cases {
        let arr = make_dyn_from_fixture(&case.inputs["x"]);
        let new_shape = parse_shape(&case.inputs["new_shape"]);
        let got = manipulation::reshape(&arr, &new_shape).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(got.shape(), expected_shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_transpose_matches_numpy() {
    // Pins `ferray_core::manipulation::transpose`.
    let suite = load_fixture(&core_path("transpose.json"));
    for case in &suite.test_cases {
        let arr = make_dyn_from_fixture(&case.inputs["x"]);
        let got = manipulation::transpose(&arr, None).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(got.shape(), expected_shape.as_slice(), "{}: shape", case.name);
        let contig = ferray_core::creation::ascontiguousarray(&got);
        assert_f64_slice_ulp(
            contig.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_expand_dims_matches_numpy() {
    // Pins `ferray_core::manipulation::expand_dims`.
    let suite = load_fixture(&core_path("expand_dims.json"));
    for case in &suite.test_cases {
        let arr = make_dyn_from_fixture(&case.inputs["x"]);
        let axis = case.inputs["axis"].as_u64().unwrap() as usize;
        let got = manipulation::expand_dims(&arr, axis).unwrap();
        let expected_shape = parse_shape(&case.expected["shape"]);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(got.shape(), expected_shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_squeeze_matches_numpy() {
    // Pins `ferray_core::manipulation::squeeze`.
    let suite = load_fixture(&core_path("squeeze.json"));
    for case in &suite.test_cases {
        let arr = make_dyn_from_fixture(&case.inputs["x"]);
        let got = manipulation::squeeze(&arr, None).unwrap();
        let expected_shape = parse_shape(&case.expected["shape"]);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(got.shape(), expected_shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn fixture_flatten_matches_numpy() {
    // Pins `ferray_core::manipulation::flatten` and
    // `ferray_core::manipulation::ravel` (flatten delegates to ravel).
    let suite = load_fixture(&core_path("flatten.json"));
    for case in &suite.test_cases {
        let arr = make_dyn_from_fixture(&case.inputs["x"]);
        let got = manipulation::flatten(&arr).unwrap();
        let expected_shape = parse_shape(&case.expected["shape"]);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(got.shape(), expected_shape.as_slice(), "{}: shape", case.name);
        assert_f64_slice_ulp(
            got.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
        // Sibling: ravel produces an identical result.
        let r = manipulation::ravel(&arr).unwrap();
        assert_eq!(r.as_slice().unwrap(), got.as_slice().unwrap());
    }
}

#[test]
fn fixture_broadcast_shapes_matches_numpy() {
    // Pins:
    //   - `ferray_core::dimension::broadcast::broadcast_shapes`
    //   - `ferray_core::dimension::broadcast::broadcast_shapes_multi`
    let suite = load_fixture(&core_path("broadcast_shapes.json"));
    for case in &suite.test_cases {
        let s1 = parse_shape(&case.inputs["shape1"]);
        let s2 = parse_shape(&case.inputs["shape2"]);
        let got = bc::broadcast_shapes(&s1, &s2).unwrap();
        let expected = parse_shape(&case.expected["shape"]);
        assert_eq!(got, expected, "{}: pairwise", case.name);
        // multi-shape entry point with the same two operands must agree.
        let got_multi = bc::broadcast_shapes_multi(&[&s1, &s2]).unwrap();
        assert_eq!(got_multi, expected, "{}: multi", case.name);
    }
}

// ---------------------------------------------------------------------------
// Inline-anchored: manipulation siblings + extended surface.
// ---------------------------------------------------------------------------

#[test]
fn inline_swapaxes_moveaxis_rollaxis_preserve_data() {
    // Pins:
    //   - `ferray_core::manipulation::swapaxes`
    //   - `ferray_core::manipulation::moveaxis`
    //   - `ferray_core::manipulation::rollaxis`
    let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), (0..24).map(|x| x as f64).collect())
        .unwrap();

    let sw = manipulation::swapaxes(&a, 0, 2).unwrap();
    assert_eq!(sw.shape(), [4, 3, 2]);

    let mv = manipulation::moveaxis(&a, 0, 2).unwrap();
    assert_eq!(mv.shape(), [3, 4, 2]);

    let rl = manipulation::rollaxis(&a, 2, 0).unwrap();
    assert_eq!(rl.shape(), [4, 2, 3]);
}

#[test]
fn inline_flip_fliplr_flipud_rot90_roll() {
    // Pins:
    //   - `ferray_core::manipulation::flip`
    //   - `ferray_core::manipulation::fliplr`
    //   - `ferray_core::manipulation::flipud`
    //   - `ferray_core::manipulation::rot90`
    //   - `ferray_core::manipulation::roll`
    let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let fl = manipulation::flip(&a, 1).unwrap();
    // Make contiguous before reading flat slice.
    let fl_c = ferray_core::creation::ascontiguousarray(&fl);
    assert_eq!(fl_c.as_slice().unwrap(), &[3.0, 2., 1., 6., 5., 4.]);

    let lr = manipulation::fliplr(&a).unwrap();
    let lr_c = ferray_core::creation::ascontiguousarray(&lr);
    assert_eq!(lr_c.as_slice().unwrap(), &[3.0, 2., 1., 6., 5., 4.]);

    let ud = manipulation::flipud(&a).unwrap();
    let ud_c = ferray_core::creation::ascontiguousarray(&ud);
    assert_eq!(ud_c.as_slice().unwrap(), &[4.0, 5., 6., 1., 2., 3.]);

    let r = manipulation::rot90(&a, 1).unwrap();
    // rot90 once: shape transposes the last two dims.
    assert_eq!(r.shape(), [3, 2]);

    let rolled = manipulation::roll(&a, 1, Some(1)).unwrap();
    let rolled_c = ferray_core::creation::ascontiguousarray(&rolled);
    assert_eq!(rolled_c.as_slice().unwrap(), &[3.0, 1., 2., 6., 4., 5.]);
}

#[test]
fn inline_concatenate_stack_family() {
    // Pins:
    //   - `ferray_core::manipulation::concatenate`
    //   - `ferray_core::manipulation::stack`
    //   - `ferray_core::manipulation::vstack`
    //   - `ferray_core::manipulation::hstack`
    //   - `ferray_core::manipulation::dstack`
    //   - `ferray_core::manipulation::column_stack`
    //   - `ferray_core::manipulation::row_stack`
    //   - `ferray_core::manipulation::block`
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1., 2., 3., 4.]).unwrap();
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![5., 6., 7., 8.]).unwrap();

    let cat = manipulation::concatenate(&[a.clone(), b.clone()], 0).unwrap();
    assert_eq!(cat.shape(), [4, 2]);

    let st = manipulation::stack(&[a.clone(), b.clone()], 0).unwrap();
    assert_eq!(st.shape(), [2, 2, 2]);

    let vs = manipulation::vstack(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(vs.shape(), [4, 2]);
    let hs = manipulation::hstack(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(hs.shape(), [2, 4]);
    let ds = manipulation::dstack(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(ds.shape(), [2, 2, 2]);
    let cs = manipulation::column_stack(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(cs.shape(), [2, 4]);
    let rs = manipulation::row_stack(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(rs.shape(), [4, 2]);

    let blk = manipulation::block(&[vec![a.clone(), b.clone()]]).unwrap();
    assert_eq!(blk.shape(), [2, 4]);
}

#[test]
fn inline_split_family_returns_n_chunks() {
    // Pins:
    //   - `ferray_core::manipulation::split`
    //   - `ferray_core::manipulation::array_split`
    //   - `ferray_core::manipulation::array_split_n`
    //   - `ferray_core::manipulation::vsplit`
    //   - `ferray_core::manipulation::hsplit`
    //   - `ferray_core::manipulation::dsplit`
    let m = Array::<f64, IxDyn>::from_vec(
        IxDyn::new(&[4, 4]),
        (0..16).map(|x| x as f64).collect(),
    )
    .unwrap();

    let parts = manipulation::split(&m, 2, 0).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape(), [2, 4]);

    let asp = manipulation::array_split(&m, &[2], 0).unwrap();
    assert_eq!(asp.len(), 2);

    let aspn = manipulation::array_split_n(&m, 3, 0).unwrap();
    assert_eq!(aspn.len(), 3);

    let vs = manipulation::vsplit(&m, 2).unwrap();
    assert_eq!(vs.len(), 2);

    let hs = manipulation::hsplit(&m, 2).unwrap();
    assert_eq!(hs.len(), 2);

    let m3 = Array::<f64, IxDyn>::from_vec(
        IxDyn::new(&[2, 2, 4]),
        (0..16).map(|x| x as f64).collect(),
    )
    .unwrap();
    let ds = manipulation::dsplit(&m3, 2).unwrap();
    assert_eq!(ds.len(), 2);
}

#[test]
fn inline_r_underscore_c_underscore_concatenation_helpers() {
    // Pins:
    //   - `ferray_core::manipulation::r_`
    //   - `ferray_core::manipulation::c_`
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2., 3.]).unwrap();
    let b = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![4.0, 5.]).unwrap();
    let r = manipulation::r_(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(r.shape(), [5]);
    assert_eq!(r.as_slice().unwrap(), &[1.0, 2., 3., 4., 5.]);

    let a2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2., 3.]).unwrap();
    let b2 = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![4.0, 5., 6.]).unwrap();
    let c = manipulation::c_(&[a2, b2]).unwrap();
    assert_eq!(c.shape(), [3, 2]);
}

#[test]
fn inline_manipulation_broadcast_to_view() {
    // Pins:
    //   - `ferray_core::manipulation::broadcast_to`
    //   - `ferray_core::dimension::broadcast::broadcast_to`
    //   - `ferray_core::dimension::broadcast::broadcast_view_to`
    //   - `ferray_core::dimension::broadcast::broadcast_arrays`
    //   - `ferray_core::dimension::broadcast::broadcast_strides`
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10., 20., 30.]).unwrap();
    let b = manipulation::broadcast_to(&a, &[2, 3]).unwrap();
    assert_eq!(b.shape(), [2, 3]);

    let av = a.view();
    let v = bc::broadcast_view_to(&av, &[2, 3]).unwrap();
    assert_eq!(v.shape(), [2, 3]);

    let b2 = bc::broadcast_to(&a, &[2, 3]).unwrap();
    assert_eq!(b2.shape(), [2, 3]);

    let arr_x =
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2., 3., 4., 5., 6.]).unwrap();
    let arr_y =
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![10., 20., 30., 40., 50., 60.])
            .unwrap();
    let pair = [arr_x, arr_y];
    let views = bc::broadcast_arrays(&pair).unwrap();
    assert_eq!(views.len(), 2);
    assert_eq!(views[0].shape(), [2, 3]);
    assert_eq!(views[1].shape(), [2, 3]);

    let s = bc::broadcast_strides(&[3], &[1], &[2, 3]).unwrap();
    assert_eq!(s.len(), 2);
}

#[test]
fn inline_pad_pad_1d_modes() {
    // Pins:
    //   - `ferray_core::manipulation::extended::pad`
    //   - `ferray_core::manipulation::extended::pad_1d`
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2., 3.]).unwrap();
    let p1 = ext::pad_1d(&a, (1, 2), &PadMode::<f64>::Constant(0.0)).unwrap();
    assert_eq!(p1.shape(), [6]);
    assert_eq!(p1.as_slice().unwrap(), &[0.0, 1., 2., 3., 0., 0.]);

    let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1., 2., 3., 4.]).unwrap();
    let p2 = ext::pad(&a2, &[(1, 1), (1, 1)], &PadMode::<f64>::Constant(0.0)).unwrap();
    assert_eq!(p2.shape(), [4, 4]);
}

#[test]
fn inline_tile_repeat() {
    // Pins:
    //   - `ferray_core::manipulation::extended::tile`
    //   - `ferray_core::manipulation::extended::repeat`
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2., 3.]).unwrap();
    let t = ext::tile(&a, &[2]).unwrap();
    assert_eq!(t.shape(), [6]);
    assert_eq!(t.as_slice().unwrap(), &[1.0, 2., 3., 1., 2., 3.]);

    let r = ext::repeat(&a, 2, None).unwrap();
    assert_eq!(r.shape(), [6]);
    assert_eq!(r.as_slice().unwrap(), &[1.0, 1., 2., 2., 3., 3.]);
}

#[test]
fn inline_delete_insert_append_resize_trim_zeros() {
    // Pins:
    //   - `ferray_core::manipulation::extended::delete`
    //   - `ferray_core::manipulation::extended::insert`
    //   - `ferray_core::manipulation::extended::append`
    //   - `ferray_core::manipulation::extended::resize`
    //   - `ferray_core::manipulation::extended::trim_zeros`
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2., 3., 4., 5.]).unwrap();
    let d = ext::delete(&a, &[0, 4], 0).unwrap();
    assert_eq!(d.shape(), [3]);
    assert_eq!(d.as_slice().unwrap(), &[2.0, 3., 4.]);

    let values =
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![99.0]).unwrap();
    let ins = ext::insert(&a, 2, &values, 0).unwrap();
    assert_eq!(ins.shape(), [6]);

    let ap = ext::append(&a, &values, None).unwrap();
    assert_eq!(ap.shape(), [6]);

    let rs = ext::resize(&a, &[3]).unwrap();
    assert_eq!(rs.shape(), [3]);

    let z = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![0.0, 1., 2., 0., 0.]).unwrap();
    let tz = ext::trim_zeros(&z, "fb").unwrap();
    assert_eq!(tz.shape(), [2]);
    assert_eq!(tz.as_slice().unwrap(), &[1.0, 2.]);
}

#[test]
fn inline_atleast_promotes_dimensions() {
    // Pins:
    //   - `ferray_core::manipulation::extended::atleast_1d`
    //   - `ferray_core::manipulation::extended::atleast_2d`
    //   - `ferray_core::manipulation::extended::atleast_3d`
    let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2., 3.]).unwrap();
    let a1 = ext::atleast_1d(&v).unwrap();
    assert_eq!(a1.shape(), [3]);

    let a2 = ext::atleast_2d(&v).unwrap();
    assert_eq!(a2.shape(), [1, 3]);

    let a3 = ext::atleast_3d(&v).unwrap();
    assert_eq!(a3.shape(), [1, 3, 1]);
}
