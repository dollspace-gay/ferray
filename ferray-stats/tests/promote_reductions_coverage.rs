//! Extended coverage for narrow-integer accumulator promotion in
//! `sum`/`prod`/`cumsum`/`cumprod` (ferray-stats + ferray-ufunc reuse of
//! ferray-core's `ReduceAcc`).
//!
//! Companion to `divergence_reductions.rs` (which pins the i8 cases). These
//! tests cover the rest of the promotion table — i16/u8/u32/bool inputs, the
//! axis path, and the wide-type identity cases (i64/f32/f64 stay themselves).
//!
//! R-CHAR-3: every expected value is a live numpy-2.4.5 result, reproduced by
//! the commands in each block, NOT copied from the ferray side.

use ferray_core::{Array, Ix1, Ix2};
use ferray_stats::{cumprod, cumsum, prod, sum};

// ---------------------------------------------------------------------------
// sum — narrow-int promotion across the table.
// ---------------------------------------------------------------------------

#[test]
fn sum_int16_promotes_to_int64() {
    // Live numpy: np.sum(np.array([20000,20000,20000],dtype=np.int16)) -> 60000 int64
    let a = Array::<i16, Ix1>::from_vec(Ix1::new([3]), vec![20000, 20000, 20000]).unwrap();
    let r = sum(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 60000_i64);
}

#[test]
fn sum_uint8_promotes_to_uint64() {
    // Live numpy: np.sum(np.array([200,200,200],dtype=np.uint8)) -> 600 uint64
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![200, 200, 200]).unwrap();
    let r = sum(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 600_u64);
}

#[test]
fn sum_uint32_promotes_to_uint64() {
    // Live numpy: np.sum(np.array([3000000000],dtype=np.uint32)) -> 3000000000 uint64
    let a = Array::<u32, Ix1>::from_vec(Ix1::new([1]), vec![3_000_000_000]).unwrap();
    let r = sum(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 3_000_000_000_u64);
}

#[test]
fn sum_bool_promotes_to_int64() {
    // Live numpy: np.sum(np.array([True,True,False,True])) -> 3 int64
    let a = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, true, false, true]).unwrap();
    let r = sum(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 3_i64);
}

#[test]
fn sum_int32_axis_promotes_to_int64() {
    // Live numpy: np.sum(np.array([[1,2,3],[4,5,6]],dtype=np.int32),axis=0) -> [5,7,9] int64
    let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let r = sum(&a, Some(0)).unwrap();
    assert_eq!(r.iter().copied().collect::<Vec<i64>>(), vec![5, 7, 9]);
}

// ---------------------------------------------------------------------------
// prod — narrow-int promotion.
// ---------------------------------------------------------------------------

#[test]
fn prod_uint8_promotes_to_uint64() {
    // Live numpy: np.prod(np.array([10,10,10],dtype=np.uint8)) -> 1000 uint64
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![10, 10, 10]).unwrap();
    let r = prod(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 1000_u64);
}

#[test]
fn prod_bool_promotes_to_int64() {
    // Live numpy: np.prod(np.array([True,True,True])) -> 1 int64
    let a = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
    let r = prod(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 1_i64);
}

// ---------------------------------------------------------------------------
// cumsum / cumprod — narrow-int promotion, flat + axis.
// ---------------------------------------------------------------------------

#[test]
fn cumsum_uint8_promotes_to_uint64() {
    // Live numpy: np.cumsum(np.array([200,200,200],dtype=np.uint8)) -> [200,400,600] uint64
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![200, 200, 200]).unwrap();
    let r = cumsum(&a, None).unwrap();
    assert_eq!(r.iter().copied().collect::<Vec<u64>>(), vec![200, 400, 600]);
}

#[test]
fn cumsum_bool_promotes_to_int64() {
    // Live numpy: np.cumsum(np.array([True,True,False,True])) -> [1,2,2,3] int64
    let a = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, true, false, true]).unwrap();
    let r = cumsum(&a, None).unwrap();
    assert_eq!(r.iter().copied().collect::<Vec<i64>>(), vec![1, 2, 2, 3]);
}

#[test]
fn cumsum_int16_axis_promotes_to_int64() {
    // Live numpy:
    //   np.cumsum(np.array([[20000,20000],[20000,20000]],dtype=np.int16),axis=0)
    //   -> [[20000,20000],[40000,40000]] int64
    let a =
        Array::<i16, Ix2>::from_vec(Ix2::new([2, 2]), vec![20000, 20000, 20000, 20000]).unwrap();
    let r = cumsum(&a, Some(0)).unwrap();
    assert_eq!(
        r.iter().copied().collect::<Vec<i64>>(),
        vec![20000, 20000, 40000, 40000]
    );
}

#[test]
fn cumprod_int16_promotes_to_int64() {
    // Live numpy: np.cumprod(np.array([200,200,200],dtype=np.int16)) -> [200,40000,8000000] int64
    let a = Array::<i16, Ix1>::from_vec(Ix1::new([3]), vec![200, 200, 200]).unwrap();
    let r = cumprod(&a, None).unwrap();
    assert_eq!(
        r.iter().copied().collect::<Vec<i64>>(),
        vec![200, 40000, 8_000_000]
    );
}

#[test]
fn cumprod_bool_promotes_to_int64() {
    // Live numpy: np.cumprod(np.array([True,False,True])) -> [1,0,0] int64
    let a = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
    let r = cumprod(&a, None).unwrap();
    assert_eq!(r.iter().copied().collect::<Vec<i64>>(), vec![1, 0, 0]);
}

// ---------------------------------------------------------------------------
// Wide types are unchanged (Acc == Self) — the promotion is a no-op.
// ---------------------------------------------------------------------------

#[test]
fn sum_int64_stays_int64() {
    // Live numpy: np.sum(np.array([1,2,3],dtype=np.int64)).dtype == int64, value 6
    let a = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
    let r = sum(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 6_i64);
}

#[test]
fn sum_f64_stays_f64() {
    // Live numpy: np.sum(np.array([1.,2.,3.])) == 6.0 float64
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    let r = sum(&a, None).unwrap();
    assert_eq!(*r.iter().next().unwrap(), 6.0_f64);
}

#[test]
fn cumsum_f32_stays_f32() {
    // Live numpy: np.cumsum(np.array([1,2,3],dtype=np.float32)) -> [1,3,6] float32
    let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    let r = cumsum(&a, None).unwrap();
    assert_eq!(r.iter().copied().collect::<Vec<f32>>(), vec![1.0, 3.0, 6.0]);
}
