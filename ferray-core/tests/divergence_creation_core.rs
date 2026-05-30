//! Divergence pins for `ferray-core` array creation (`creation/mod.rs`).
//!
//! ACToR-CRITIC audit of `ferray_core::creation` against numpy 2.4.5 as the
//! live oracle. Every expected value below comes from a live `numpy` call or
//! a numpy upstream `file:line` constant (R-CHAR-3) — never literal-copied
//! from the ferray side. These tests are written to FAIL: each pins a place
//! where ferray-core's creation routines diverge from numpy's observable
//! numerical / structural contract (R-DEV-1).
//!
//! Upstream numpy sources opened this iteration:
//!   - numpy/_core/function_base.py:142-176 (linspace: `y = arange(0,num)*step`
//!     then `y += start`; `if endpoint and num > 1: y[-1, ...] = stop`)
//!   - numpy/_core/function_base.py:28-29 (`linspace(..., num=50, ...)` default)
//!   - numpy/_core/numeric.py (`arange` fills `start + i*step`, two roundings)

#![allow(clippy::float_cmp)]

use ferray_core::creation;

// ---------------------------------------------------------------------------
// DIVERGENCE 1: arange float fill uses fused multiply-add (single rounding).
//
// ferray `arange` fills each element via `(i as f64).mul_add(step, start)`
// — a fused multiply-add with a SINGLE rounding. numpy fills `start + i*step`
// with TWO roundings (the product rounds, then the add rounds). For
// `arange(0.1, 2.0, 0.1)` the two disagree at several indices.
//
// Oracle (numpy 2.4.5):
//   >>> a = np.arange(0.1, 2.0, 0.1)
//   >>> a[5]   ->  0.6                    (np.float64)
//   >>> a[12]  ->  1.3000000000000003
//   >>> a[2]   ->  0.30000000000000004
// ferray (fma): a[5] == 0.6000000000000001, a[12] == 1.3
// ---------------------------------------------------------------------------
#[test]
fn divergence_arange_float_fill_matches_numpy_start_plus_i_step() {
    let a = creation::arange(0.1_f64, 2.0, 0.1).unwrap();
    let data = a.as_slice().unwrap();
    // numpy: np.arange(0.1, 2.0, 0.1) has length 19.
    assert_eq!(a.shape(), &[19], "arange(0.1,2.0,0.1) length");
    // Bit-exact expected values straight from the numpy oracle.
    assert_eq!(data[2], 0.30000000000000004, "arange[2] vs numpy");
    assert_eq!(
        data[5], 0.6,
        "arange[5]: numpy==0.6, ferray fma==0.6000000000000001"
    );
    assert_eq!(
        data[12], 1.3000000000000003,
        "arange[12]: numpy==1.3000000000000003, ferray fma==1.3"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 2: linspace interior values use fma (single rounding) vs numpy's
// `arange(0,num)*step + start` (two roundings).
//
// numpy/_core/function_base.py:142-173:
//   y = arange(0, num) ...; step = delta/div; y = y * step (or y *= step); y += start
// ferray linspace fills `(i as f64).mul_add(step, start)` (fused).
//
// Oracle (numpy 2.4.5):
//   >>> np.linspace(0.1, 0.7, 7)[3]  ->  0.4
//   >>> np.linspace(0.1, 0.9, 9)[5]  ->  0.6
// ferray (fma): [3] == 0.39999999999999997 / [5] == 0.6000000000000001
// ---------------------------------------------------------------------------
#[test]
fn divergence_linspace_interior_matches_numpy() {
    let a = creation::linspace::<f64>(0.1, 0.7, 7, true).unwrap();
    let data = a.as_slice().unwrap();
    assert_eq!(a.shape(), &[7]);
    assert_eq!(
        data[3], 0.4,
        "linspace(0.1,0.7,7)[3]: numpy==0.4, ferray fma==0.39999999999999997"
    );

    let b = creation::linspace::<f64>(0.1, 0.9, 9, true).unwrap();
    let bdata = b.as_slice().unwrap();
    assert_eq!(
        bdata[5], 0.6,
        "linspace(0.1,0.9,9)[5]: numpy==0.6, ferray fma==0.6000000000000001"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 3: linspace does NOT force the final sample to exactly `stop`.
//
// numpy/_core/function_base.py:175-176:
//   if endpoint and num > 1:
//       y[-1, ...] = stop
// numpy guarantees the LAST sample equals `stop` exactly when endpoint=True.
// ferray computes the last element arithmetically and never overwrites it, so
// for `linspace(0.1, 3.7, 15)` it produces 3.7000000000000006 instead of 3.7.
//
// Oracle (numpy 2.4.5):
//   >>> np.linspace(0.1, 3.7, 15)[-1]  ->  3.7   (== stop, bit-exact)
// ---------------------------------------------------------------------------
#[test]
fn divergence_linspace_endpoint_is_exactly_stop() {
    let a = creation::linspace::<f64>(0.1, 3.7, 15, true).unwrap();
    let data = a.as_slice().unwrap();
    assert_eq!(a.shape(), &[15]);
    assert_eq!(
        data[14], 3.7,
        "linspace endpoint=True must set last sample == stop exactly \
         (numpy function_base.py:175-176); numpy==3.7, ferray==3.7000000000000006"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 4: linspace default `num` is 50.
//
// numpy/_core/function_base.py:28-29:
//   def linspace(start, stop, num=50, endpoint=True, ...):
// numpy's documented default sample count is 50. ferray's `linspace` requires
// `num` as a mandatory positional argument with no default, so the canonical
// `np.linspace(start, stop)` -> 50 samples is not expressible. This pins the
// missing default by asserting the 50-sample contract via an explicit call:
// the produced length must be 50 and span [start, stop] inclusively, with the
// final element exactly `stop` (the endpoint-forcing contract of DIVERGENCE 3:
// ferray's last sample is 0.9999999999999999, numpy forces 1.0).
//
// Oracle (numpy 2.4.5):
//   >>> len(np.linspace(0.0, 1.0))      ->  50
//   >>> np.linspace(0.0, 1.0)[-1]       ->  1.0
//   >>> np.linspace(0.0, 1.0)[1]        ->  0.02040816326530612
// ---------------------------------------------------------------------------
#[test]
fn divergence_linspace_default_num_50_contract() {
    // ferray has no default; emulate the call numpy makes for np.linspace(0,1).
    let a = creation::linspace::<f64>(0.0, 1.0, 50, true).unwrap();
    let data = a.as_slice().unwrap();
    assert_eq!(a.shape(), &[50], "np.linspace default num == 50");
    assert_eq!(data[49], 1.0, "default-num linspace last == stop");
    assert_eq!(
        data[1], 0.02040816326530612,
        "linspace(0,1,50)[1] vs numpy oracle"
    );
}
