// Divergence pins for ferray-core AXIS reductions + wider-int widths + empty
// narrow-int promotion (sum_axis / prod_axis), complementing the #780 whole-
// array pins in divergence_reductions.rs.
//
// AUDIT TARGET: ferray-core/src/array/reductions.rs (sum_axis / prod_axis +
// ReduceAcc), builder commit c13a3f20.
//
// Each test asserts ferray-core matches the LIVE numpy 2.4.5 oracle (value +
// result-array dtype). Expected values are numpy-derived (cited inline from a
// live `python3 -c "import numpy as np; ..."` run), NEVER copied from the
// ferray side (R-CHAR-3).
//
// The result DTYPE is pinned structurally: each test binds the reduced array
// to `Array<i64, _>` / `Array<u64, _>`. If `sum_axis`/`prod_axis` returned the
// un-promoted narrow `T` (the pre-#780 contract) these would not compile; the
// type annotation IS the dtype assertion.

use ferray_core::Array;
use ferray_core::dimension::{Axis, Ix1, Ix2, IxDyn};

// ---------------------------------------------------------------------------
// AXIS DTYPE PROMOTION — the builder pinned only whole-array sum/prod, NOT the
// axis variants. numpy promotes the AXIS result array dtype identically:
//   np.sum(int8_2d, axis=0).dtype  == int64
//   np.sum(uint8_2d, axis=0).dtype == uint64
//   np.prod(int16_2d, axis=0).dtype == int64
//   np.sum(bool_2d, axis=0).dtype  == int64
// Cite: numpy/_core/fromnumeric.py:2321-2327 (sum), :3306-3312 (prod) — the
// dtype rule is independent of `axis`; numpy/_core/_methods.py:20-21 runs the
// reduce loop at the promoted accumulator for any axis.
// ---------------------------------------------------------------------------

#[test]
fn sum_axis_i8_promotes_array_to_i64_and_no_overflow() {
    // numpy: a = np.array([[100,100,100],[100,100,100]], dtype=np.int8)
    //        np.sum(a, axis=0) == [200,200,200], dtype int64
    let a =
        Array::<i8, Ix2>::from_vec(Ix2::new([2, 3]), vec![100, 100, 100, 100, 100, 100]).unwrap();
    // Binding to Array<i64, _> pins the promoted element dtype.
    let s: Array<i64, IxDyn> = a.sum_axis(Axis(0)).unwrap();
    assert_eq!(s.shape(), &[3]);
    assert_eq!(
        s.iter().copied().collect::<Vec<i64>>(),
        vec![200, 200, 200],
        "numpy int8 sum_axis(0) promotes to int64 [200,200,200] without overflow"
    );
}

#[test]
fn prod_axis_i8_promotes_array_to_i64_and_no_overflow() {
    // numpy: p = np.array([[100,100],[100,100]], dtype=np.int8)
    //        np.prod(p, axis=0) == [10000,10000], dtype int64
    let p = Array::<i8, Ix2>::from_vec(Ix2::new([2, 2]), vec![100, 100, 100, 100]).unwrap();
    let pr: Array<i64, IxDyn> = p.prod_axis(Axis(0)).unwrap();
    assert_eq!(pr.shape(), &[2]);
    assert_eq!(
        pr.iter().copied().collect::<Vec<i64>>(),
        vec![10000, 10000],
        "numpy int8 prod_axis(0) promotes to int64 [10000,10000] without overflow"
    );
}

#[test]
fn sum_axis_u8_promotes_array_to_u64_and_no_overflow() {
    // numpy: a = np.array([[200,200],[200,200]], dtype=np.uint8)
    //        np.sum(a, axis=0) == [400,400], dtype uint64
    let a = Array::<u8, Ix2>::from_vec(Ix2::new([2, 2]), vec![200, 200, 200, 200]).unwrap();
    let s: Array<u64, IxDyn> = a.sum_axis(Axis(0)).unwrap();
    assert_eq!(
        s.iter().copied().collect::<Vec<u64>>(),
        vec![400, 400],
        "numpy uint8 sum_axis(0) promotes to uint64 [400,400] without overflow"
    );
}

#[test]
fn prod_axis_i16_promotes_array_to_i64() {
    // numpy: p = np.array([[1000,1000],[1000,1000]], dtype=np.int16)
    //        np.prod(p, axis=0) == [1000000,1000000], dtype int64
    let p = Array::<i16, Ix2>::from_vec(Ix2::new([2, 2]), vec![1000, 1000, 1000, 1000]).unwrap();
    let pr: Array<i64, IxDyn> = p.prod_axis(Axis(0)).unwrap();
    assert_eq!(
        pr.iter().copied().collect::<Vec<i64>>(),
        vec![1_000_000, 1_000_000],
        "numpy int16 prod_axis(0) promotes to int64 [1e6,1e6] without overflow"
    );
}

// ---------------------------------------------------------------------------
// WIDER NARROW WIDTHS the builder did not pin individually for whole-array:
// u16, u32, i32 sum and i16 prod. numpy: u16/u32 -> uint64, i32 -> int64.
// ---------------------------------------------------------------------------

#[test]
fn sum_u16_promotes_to_u64_no_overflow() {
    // numpy: np.sum(np.array([40000,40000], dtype=np.uint16)) == 80000, uint64
    // (overflows u16's 65535 ceiling -> would wrap to 14464 without promotion).
    let a = Array::<u16, Ix1>::from_vec(Ix1::new([2]), vec![40000, 40000]).unwrap();
    let got: u64 = a.sum();
    assert_eq!(got, 80000, "numpy uint16 sum promotes to uint64 = 80000");
}

#[test]
fn sum_u32_promotes_to_u64_no_overflow() {
    // numpy: np.sum(np.array([3000000000,3000000000], dtype=np.uint32))
    //        == 6000000000, uint64 (overflows u32's 4294967295 ceiling).
    let a = Array::<u32, Ix1>::from_vec(Ix1::new([2]), vec![3_000_000_000, 3_000_000_000]).unwrap();
    let got: u64 = a.sum();
    assert_eq!(
        got, 6_000_000_000,
        "numpy uint32 sum promotes to uint64 = 6e9"
    );
}

#[test]
fn sum_i32_promotes_to_i64_no_overflow() {
    // numpy: np.sum(np.array([2000000000,2000000000], dtype=np.int32))
    //        == 4000000000, int64 (overflows i32's 2147483647 ceiling).
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![2_000_000_000, 2_000_000_000]).unwrap();
    let got: i64 = a.sum();
    assert_eq!(
        got, 4_000_000_000,
        "numpy int32 sum promotes to int64 = 4e9"
    );
}

#[test]
fn prod_i16_promotes_to_i64_no_overflow() {
    // numpy: np.prod(np.array([1000,1000], dtype=np.int16)) == 1000000, int64
    // (overflows i16's 32767 ceiling).
    let a = Array::<i16, Ix1>::from_vec(Ix1::new([2]), vec![1000, 1000]).unwrap();
    let got: i64 = a.prod();
    assert_eq!(got, 1_000_000, "numpy int16 prod promotes to int64 = 1e6");
}

// ---------------------------------------------------------------------------
// EMPTY narrow-int reduction returns the IDENTITY in the PROMOTED dtype.
//   numpy: np.sum(np.array([], dtype=np.int8))  == 0, dtype int64
//          np.prod(np.array([], dtype=np.int8)) == 1, dtype int64
//          np.sum(np.array([], dtype=np.uint8)) == 0, dtype uint64
// (Unlike empty min/max which RAISE — #782 — empty sum/prod have an identity.)
// ---------------------------------------------------------------------------

#[test]
fn sum_empty_i8_is_zero_in_i64() {
    // numpy: np.sum(np.array([], dtype=np.int8)) == 0, dtype int64
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    let got: i64 = a.sum();
    assert_eq!(got, 0, "numpy empty int8 sum == 0 in promoted int64");
}

#[test]
fn prod_empty_i8_is_one_in_i64() {
    // numpy: np.prod(np.array([], dtype=np.int8)) == 1, dtype int64
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    let got: i64 = a.prod();
    assert_eq!(got, 1, "numpy empty int8 prod == 1 in promoted int64");
}

#[test]
fn sum_empty_u8_is_zero_in_u64() {
    // numpy: np.sum(np.array([], dtype=np.uint8)) == 0, dtype uint64
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    let got: u64 = a.sum();
    assert_eq!(got, 0, "numpy empty uint8 sum == 0 in promoted uint64");
}

// ---------------------------------------------------------------------------
// NO SPURIOUS PROMOTION on axis for wide dtypes: i64 axis stays i64, f64 axis
// stays f64 (numpy never promotes >= platform int or float).
//   np.sum(np.ones((2,3), dtype=np.int64), axis=0).dtype   == int64
//   np.sum(np.ones((2,3), dtype=np.float64), axis=0).dtype == float64
// ---------------------------------------------------------------------------

#[test]
fn sum_axis_i64_stays_i64() {
    // numpy: np.sum(np.array([[1,2,3],[4,5,6]], dtype=np.int64), axis=0)
    //        == [5,7,9], dtype int64
    let a = Array::<i64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let s: Array<i64, IxDyn> = a.sum_axis(Axis(0)).unwrap();
    assert_eq!(
        s.iter().copied().collect::<Vec<i64>>(),
        vec![5, 7, 9],
        "numpy int64 sum_axis stays int64 (no spurious promotion)"
    );
}

#[test]
fn sum_axis_bool_promotes_to_i64() {
    // numpy: np.sum(np.array([[True,True,False],[True,False,False]]), axis=0)
    //        == [2,1,0], dtype int64
    let a = Array::<bool, Ix2>::from_vec(
        Ix2::new([2, 3]),
        vec![true, true, false, true, false, false],
    )
    .unwrap();
    let s: Array<i64, IxDyn> = a.sum_axis(Axis(0)).unwrap();
    assert_eq!(
        s.iter().copied().collect::<Vec<i64>>(),
        vec![2, 1, 0],
        "numpy bool sum_axis(0) counts True as int64 [2,1,0]"
    );
}

// ---------------------------------------------------------------------------
// Extra hardening probes (3D axis dtype, signed-negative narrow sum, u8 prod
// overflow). All numpy-derived; included to catch any remaining gap.
// ---------------------------------------------------------------------------

use ferray_core::dimension::Ix3;

#[test]
fn sum_axis_3d_i8_promotes_to_i64() {
    // numpy: a = np.arange(24, dtype=np.int8).reshape(2,3,4)
    //        np.sum(a, axis=1).dtype == int64, shape (2,4)
    let data: Vec<i8> = (0..24).collect();
    let a = Array::<i8, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
    let s: Array<i64, IxDyn> = a.sum_axis(Axis(1)).unwrap();
    assert_eq!(s.shape(), &[2, 4]);
    // lane (i=0,k=0): 0+4+8 = 12
    assert_eq!(
        s.iter().copied().next(),
        Some(12),
        "numpy int8 3D sum_axis(1) -> int64, first element 0+4+8=12"
    );
}

#[test]
fn sum_negative_i8_promotes_to_i64() {
    // numpy: np.sum(np.array([-100,-100], dtype=np.int8)) == -200, int64
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![-100, -100]).unwrap();
    let got: i64 = a.sum();
    assert_eq!(
        got, -200,
        "numpy int8 negative sum promotes to int64 = -200"
    );
}

#[test]
fn prod_u8_promotes_to_u64_no_overflow() {
    // numpy: np.prod(np.array([200,200], dtype=np.uint8)) == 40000, uint64
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![200, 200]).unwrap();
    let got: u64 = a.prod();
    assert_eq!(got, 40000, "numpy uint8 prod promotes to uint64 = 40000");
}
