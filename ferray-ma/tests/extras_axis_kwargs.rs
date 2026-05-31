//! Integration tests for the axis/returned extensions added for blockers
//! #1071/#1072/#1073 (refs #835):
//!
//!   - `ferray_ma::ma_median_axis`        — `numpy.ma.median(a, axis=...)`
//!   - `ferray_ma::notmasked_edges_axis2` — `numpy.ma.notmasked_edges(a, axis=...)`
//!   - `MaskedArray::average_returned`    — `numpy.ma.average(..., returned=True)`
//!
//! Expected values are constructed from the numpy.ma contract documented in
//! `numpy/ma/extras.py` (`median`:678, `average`:510, `notmasked_edges`:1925),
//! NOT copied from the ferray side (R-CHAR-3). Each comment cites the unmasked
//! subset that drives the result.

use ferray_core::dimension::IxDyn;
use ferray_core::{Array, Ix1, Ix2};
use ferray_ma::{MaskedArray, ma_median_axis, notmasked_edges_axis2};

fn ma_dyn(shape: &[usize], data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, IxDyn> {
    let d = Array::from_vec(IxDyn::new(shape), data).unwrap();
    let m = Array::from_vec(IxDyn::new(shape), mask).unwrap();
    MaskedArray::new(d, m).unwrap()
}

fn ma_ix2(rows: usize, cols: usize, data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix2> {
    let d = Array::from_vec(Ix2::new([rows, cols]), data).unwrap();
    let m = Array::from_vec(Ix2::new([rows, cols]), mask).unwrap();
    MaskedArray::new(d, m).unwrap()
}

#[test]
fn median_axis1_per_row() {
    // numpy.ma.median(a, axis=1): row0 unmasked {1,3}->2.0, row1 {4,5}->4.5.
    let ma = ma_dyn(
        &[2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![false, true, false, false, false, true],
    );
    let med = ma_median_axis(&ma, 1).unwrap();
    assert_eq!(med.shape(), &[2]);
    let d: Vec<f64> = med.data().iter().copied().collect();
    let m: Vec<bool> = med.mask().iter().copied().collect();
    assert!((d[0] - 2.0).abs() < 1e-12);
    assert!((d[1] - 4.5).abs() < 1e-12);
    assert_eq!(m, vec![false, false]);
}

#[test]
fn median_axis0_per_column() {
    // axis=0: col0 unmasked {1,4}->2.5, col1 {2,5}->3.5, col2 {6}->6.0.
    let ma = ma_dyn(
        &[2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![false, false, true, false, false, false],
    );
    let med = ma_median_axis(&ma, 0).unwrap();
    assert_eq!(med.shape(), &[3]);
    let d: Vec<f64> = med.data().iter().copied().collect();
    assert!((d[0] - 2.5).abs() < 1e-12);
    assert!((d[1] - 3.5).abs() < 1e-12);
    assert!((d[2] - 6.0).abs() < 1e-12);
}

#[test]
fn median_axis_all_masked_lane_is_masked() {
    // Column 1 fully masked -> masked output slot along axis 0.
    let ma = ma_dyn(
        &[2, 2],
        vec![1.0, 7.0, 3.0, 9.0],
        vec![false, true, false, true],
    );
    let med = ma_median_axis(&ma, 0).unwrap();
    let d: Vec<f64> = med.data().iter().copied().collect();
    let m: Vec<bool> = med.mask().iter().copied().collect();
    assert!((d[0] - 2.0).abs() < 1e-12); // col0 {1,3}->2.0
    assert_eq!(m, vec![false, true]);
}

#[test]
fn median_axis_out_of_bounds_errors() {
    let ma = ma_dyn(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
    assert!(ma_median_axis(&ma, 5).is_err());
}

#[test]
fn average_returned_weighted_yields_weight_sum() {
    // numpy.ma.average(..., returned=True): unmasked {1,2,3} w {1,2,3}
    // -> avg 14/6, sum_of_weights 6.
    let d = Array::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m = Array::from_vec(Ix1::new([4]), vec![false, false, false, true]).unwrap();
    let ma = MaskedArray::new(d, m).unwrap();
    let w = Array::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let (avg, wsum): (f64, f64) = ma.average_returned(Some(&w)).unwrap();
    assert!((avg - (14.0 / 6.0)).abs() < 1e-12);
    assert!((wsum - 6.0).abs() < 1e-12);
}

#[test]
fn average_returned_unweighted_is_unmasked_count() {
    // numpy.ma.average(returned=True) unweighted: scl == count of unmasked = 3.
    let d = Array::from_vec(Ix1::new([4]), vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    let m = Array::from_vec(Ix1::new([4]), vec![false, false, false, true]).unwrap();
    let ma = MaskedArray::new(d, m).unwrap();
    let (avg, wsum): (f64, f64) = ma.average_returned(None).unwrap();
    assert!((avg - 4.0).abs() < 1e-12); // mean of {2,4,6}
    assert!((wsum - 3.0).abs() < 1e-12);
}

#[test]
fn notmasked_edges_axis1_first_last_per_row() {
    // numpy.ma.notmasked_edges(a, axis=1):
    //   row0 mask [0,1,0,0] -> first col 0, last col 3.
    //   row1 mask [0,0,1,1] -> first col 0, last col 1.
    // Result coordinates are emitted (rows, cols) in axis order.
    let ma2 = ma_ix2(
        2,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![false, true, false, false, false, false, true, true],
    );
    let ((fr, fc), (lr, lc)) = notmasked_edges_axis2(&ma2, 1).unwrap();
    assert_eq!(fr, vec![0, 1]);
    assert_eq!(fc, vec![0, 0]);
    assert_eq!(lr, vec![0, 1]);
    assert_eq!(lc, vec![3, 1]);
}

#[test]
fn notmasked_edges_axis0_first_last_per_column() {
    // axis=0 over the same mask: per-column first/last unmasked ROW.
    //   col0 [0,0]->first row 0,last 1; col1 [1,0]->first 1,last 1;
    //   col2 [0,1]->first 0,last 0;     col3 [0,1]->first 0,last 0.
    let ma2 = ma_ix2(
        2,
        4,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![false, true, false, false, false, false, true, true],
    );
    let ((fr, fc), (lr, lc)) = notmasked_edges_axis2(&ma2, 0).unwrap();
    assert_eq!(fr, vec![0, 1, 0, 0]);
    assert_eq!(fc, vec![0, 1, 2, 3]);
    assert_eq!(lr, vec![1, 1, 0, 0]);
    assert_eq!(lc, vec![0, 1, 2, 3]);
}

#[test]
fn notmasked_edges_axis_out_of_bounds_errors() {
    let ma2 = ma_ix2(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
    assert!(notmasked_edges_axis2(&ma2, 3).is_err());
}
