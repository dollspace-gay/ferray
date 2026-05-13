//! Conformance tests for the `ferray_core::indexing::extended` surface.
//! No NumPy fixtures exist for these helpers; the analytic reference
//! values are computed inline against documented NumPy semantics.
//!
//! Surface paths exercised by this file:
//!
//! - `ferray_core::indexing::extended::argwhere`
//! - `ferray_core::indexing::extended::choose`
//! - `ferray_core::indexing::extended::compress`
//! - `ferray_core::indexing::extended::diag_indices`
//! - `ferray_core::indexing::extended::diag_indices_from`
//! - `ferray_core::indexing::extended::extract`
//! - `ferray_core::indexing::extended::flatnonzero`
//! - `ferray_core::indexing::extended::indices`
//! - `ferray_core::indexing::extended::ix_`
//! - `ferray_core::indexing::extended::mask_indices`
//! - `ferray_core::indexing::extended::ndenumerate`
//! - `ferray_core::indexing::extended::ndindex`
//! - `ferray_core::indexing::extended::nonzero`
//! - `ferray_core::indexing::extended::place`
//! - `ferray_core::indexing::extended::putmask`
//! - `ferray_core::indexing::extended::ravel_multi_index`
//! - `ferray_core::indexing::extended::select`
//! - `ferray_core::indexing::extended::take`
//! - `ferray_core::indexing::extended::take_along_axis`
//! - `ferray_core::indexing::extended::tril_indices`
//! - `ferray_core::indexing::extended::tril_indices_from`
//! - `ferray_core::indexing::extended::triu_indices`
//! - `ferray_core::indexing::extended::triu_indices_from`
//! - `ferray_core::indexing::extended::unravel_index`
//! - `ferray_core::indexing::extended::where_select`
//! - `ferray_core::indexing::extended::Array::fill_diagonal`
//! - `ferray_core::indexing::extended::Array::put`
//! - `ferray_core::indexing::extended::Array::put_along_axis`
//!
//! Tolerance: shape and index ops are bit-exact. Integer values are
//! compared with `assert_eq!`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use ferray_core::Array;
use ferray_core::dimension::{Axis, Ix1, Ix2, IxDyn};
use ferray_core::indexing::extended::{self as ext, MaskKind};

fn i32_2d(rows: usize, cols: usize, data: Vec<i32>) -> Array<i32, Ix2> {
    Array::<i32, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

#[test]
fn take_and_take_along_axis_pick_elements() {
    // Pins `ferray_core::indexing::extended::take` and
    // `ferray_core::indexing::extended::take_along_axis`.
    let a = i32_2d(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let row = ext::take(&a, &[0, 2], Axis(0)).unwrap();
    assert_eq!(row.shape(), [2, 3]);
    assert_eq!(
        row.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 7, 8, 9]
    );
    let cols = ext::take_along_axis(&a, &[1], Axis(1)).unwrap();
    assert_eq!(cols.shape(), [3, 1]);
    assert_eq!(cols.iter().copied().collect::<Vec<_>>(), vec![2, 5, 8]);
}

#[test]
fn argwhere_flatnonzero_nonzero_locate_nonzeros() {
    // Pins:
    //   - `ferray_core::indexing::extended::argwhere`
    //   - `ferray_core::indexing::extended::flatnonzero`
    //   - `ferray_core::indexing::extended::nonzero`
    let a = i32_2d(2, 3, vec![0, 1, 0, 2, 0, 3]);
    let aw = ext::argwhere(&a).unwrap();
    assert_eq!(aw.shape(), [3, 2]);
    let fl = ext::flatnonzero(&a);
    assert_eq!(fl, vec![1, 3, 5]);
    let nz = ext::nonzero(&a);
    // 3 nonzeros, 2 axes — each axis vector has length 3.
    assert_eq!(nz.len(), 2);
    assert_eq!(nz[0], vec![0, 1, 1]);
    assert_eq!(nz[1], vec![1, 0, 2]);
}

#[test]
fn choose_compress_select_dispatch_correctly() {
    // Pins:
    //   - `ferray_core::indexing::extended::choose`
    //   - `ferray_core::indexing::extended::compress`
    //   - `ferray_core::indexing::extended::select`
    let idx = Array::<u64, Ix1>::from_vec(Ix1::new([4]), vec![0, 1, 0, 1]).unwrap();
    let c0 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
    let c1 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![100, 200, 300, 400]).unwrap();
    let ch = ext::choose(&idx, &[c0, c1]).unwrap();
    assert_eq!(ch.iter().copied().collect::<Vec<_>>(), vec![10, 200, 30, 400]);

    let a = i32_2d(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let comp = ext::compress(&[true, false, true], &a, Axis(1)).unwrap();
    assert_eq!(comp.shape(), [2, 2]);

    let cond1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, false]).unwrap();
    let cond2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
    let ch1 = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
    let ch2 = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![100, 200, 300]).unwrap();
    let s = ext::select(&[cond1, cond2], &[ch1, ch2], -1).unwrap();
    assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![10, 200, -1]);
}

#[test]
fn indices_ix_underscore_generate_grid_arrays() {
    // Pins:
    //   - `ferray_core::indexing::extended::indices`
    //   - `ferray_core::indexing::extended::ix_`
    let i = ext::indices(&[2, 3]).unwrap();
    assert_eq!(i.len(), 2);
    assert_eq!(i[0].shape(), [2, 3]);
    assert_eq!(i[1].shape(), [2, 3]);

    let g = ext::ix_(&[&[0u64, 1], &[2u64, 4]]).unwrap();
    assert_eq!(g.len(), 2);
    assert_eq!(g[0].shape(), [2, 1]);
    assert_eq!(g[1].shape(), [1, 2]);
}

#[test]
fn diag_tril_triu_indices_match_documented_geometry() {
    // Pins:
    //   - `ferray_core::indexing::extended::diag_indices`
    //   - `ferray_core::indexing::extended::diag_indices_from`
    //   - `ferray_core::indexing::extended::tril_indices`
    //   - `ferray_core::indexing::extended::tril_indices_from`
    //   - `ferray_core::indexing::extended::triu_indices`
    //   - `ferray_core::indexing::extended::triu_indices_from`
    let d = ext::diag_indices(3, 2);
    assert_eq!(d, vec![vec![0, 1, 2], vec![0, 1, 2]]);

    let a = i32_2d(3, 3, vec![0; 9]);
    let df = ext::diag_indices_from(&a).unwrap();
    assert_eq!(df, vec![vec![0, 1, 2], vec![0, 1, 2]]);

    let (lr, lc) = ext::tril_indices(3, 0, None);
    assert_eq!(lr.len(), lc.len());
    assert!(!lr.is_empty());

    let (lr2, lc2) = ext::tril_indices_from(&a, 0).unwrap();
    assert_eq!(lr2.len(), lc2.len());

    let (ur, uc) = ext::triu_indices(3, 0, None);
    assert_eq!(ur.len(), uc.len());

    let (ur2, uc2) = ext::triu_indices_from(&a, 0).unwrap();
    assert_eq!(ur2.len(), uc2.len());
}

#[test]
fn ravel_unravel_round_trip_flat_indices() {
    // Pins:
    //   - `ferray_core::indexing::extended::ravel_multi_index`
    //   - `ferray_core::indexing::extended::unravel_index`
    let rows = [0usize, 1, 2];
    let cols = [0usize, 1, 2];
    let flat = ext::ravel_multi_index(&[&rows, &cols], &[3, 4]).unwrap();
    assert_eq!(flat, vec![0, 5, 10]);
    let back = ext::unravel_index(&flat, &[3, 4]).unwrap();
    assert_eq!(back, vec![vec![0, 1, 2], vec![0, 1, 2]]);
}

#[test]
fn ndindex_ndenumerate_iterate_all_positions() {
    // Pins:
    //   - `ferray_core::indexing::extended::ndindex`
    //   - `ferray_core::indexing::extended::ndenumerate`
    let it = ext::ndindex(&[2, 2]);
    let positions: Vec<_> = it.collect();
    assert_eq!(positions.len(), 4);
    assert_eq!(positions[0], vec![0, 0]);
    assert_eq!(positions[3], vec![1, 1]);

    let a = i32_2d(2, 2, vec![10, 20, 30, 40]);
    let pairs: Vec<_> = ext::ndenumerate(&a).map(|(i, v)| (i, *v)).collect();
    assert_eq!(pairs.len(), 4);
    assert_eq!(pairs[0], (vec![0, 0], 10));
    assert_eq!(pairs[3], (vec![1, 1], 40));
}

#[test]
fn where_select_place_putmask_extract_mask_ops() {
    // Pins:
    //   - `ferray_core::indexing::extended::where_select`
    //   - `ferray_core::indexing::extended::place`
    //   - `ferray_core::indexing::extended::putmask`
    //   - `ferray_core::indexing::extended::extract`
    let cond = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
    let x = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
    let y = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
    let w = ext::where_select(&cond, &x, &y).unwrap();
    assert_eq!(w.iter().copied().collect::<Vec<_>>(), vec![1, 20, 3, 40]);

    let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![0, 0, 0, 0]).unwrap();
    let mask = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
    ext::place(&mut arr, &mask, &[99, 88]).unwrap();
    assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![99, 0, 88, 0]);

    let mut arr2 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![0, 0, 0, 0]).unwrap();
    ext::putmask(&mut arr2, &mask, &[7]).unwrap();
    assert_eq!(arr2.iter().copied().collect::<Vec<_>>(), vec![7, 0, 7, 0]);

    let v = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
    let ex = ext::extract(&mask, &v).unwrap();
    assert_eq!(ex.iter().copied().collect::<Vec<_>>(), vec![10, 30]);
}

#[test]
fn mask_indices_enumerates_triangle_positions() {
    // Pins:
    //   - `ferray_core::indexing::extended::mask_indices`
    //   - `ferray_core::indexing::extended::MaskKind`
    let lo = ext::mask_indices(3, MaskKind::Tril, 0);
    let up = ext::mask_indices(3, MaskKind::Triu, 0);
    // For a 3x3 matrix flattened in row-major: tril has 6 entries
    // (lower triangle including diagonal), triu has 6 (upper triangle
    // including diagonal). Both sets contain the diagonal indices
    // (0,4,8).
    assert_eq!(lo.len(), 6);
    assert_eq!(up.len(), 6);
    assert!(lo.contains(&0) && lo.contains(&4) && lo.contains(&8));
    assert!(up.contains(&0) && up.contains(&4) && up.contains(&8));
}

#[test]
fn array_put_put_along_axis_fill_diagonal_mutate_in_place() {
    // Pins:
    //   - `ferray_core::indexing::extended::Array::put`
    //   - `ferray_core::indexing::extended::Array::put_along_axis`
    //   - `ferray_core::indexing::extended::Array::fill_diagonal`
    let mut a = i32_2d(3, 3, vec![0; 9]);
    a.put(&[0, 4, 8], &[1, 2, 3]).unwrap();
    let s: Vec<i32> = a.iter().copied().collect();
    assert_eq!(s, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);

    let mut b = i32_2d(3, 3, vec![0; 9]);
    let vals = Array::<i32, IxDyn>::from_vec(IxDyn::new(&[3, 1]), vec![10, 20, 30]).unwrap();
    b.put_along_axis(&[1], &vals, Axis(1)).unwrap();
    let bs: Vec<i32> = b.iter().copied().collect();
    assert_eq!(bs, vec![0, 10, 0, 0, 20, 0, 0, 30, 0]);

    let mut c = i32_2d(3, 3, vec![0; 9]);
    c.fill_diagonal(7);
    let cs: Vec<i32> = c.iter().copied().collect();
    assert_eq!(cs, vec![7, 0, 0, 0, 7, 0, 0, 0, 7]);
}
