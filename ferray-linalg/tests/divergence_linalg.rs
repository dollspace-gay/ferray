//! ACToR critic — ferray-linalg LIBRARY divergences from numpy.linalg.
//!
//! Scope: the Rust API surface of `ferray-linalg` (NOT the ferray-python
//! binding). Each expected value below is a live `numpy` 2.4.5 oracle
//! result, traceable to `numpy/linalg/_linalg.py` (R-CHAR-3 — never
//! literal-copied from the ferray side).
//!
//! Findings:
//!   * `norm(matrix, ord=-1)` (min column abs-sum) — NO `NormOrder`
//!     variant exists. COMPILE-GATED-MISSING. Pinned via an
//!     enumerate-all-callable-orders test that proves none yields the
//!     numpy value (on a non-symmetric matrix where min-col-sum differs
//!     from every supported order).
//!   * `norm(matrix, ord=-2)` (smallest singular value) — NO `NormOrder`
//!     variant exists. COMPILE-GATED-MISSING. Same pinning strategy.
//!   * `qr(mode='r')` (R-only) — no `QrMode::R`; but the R *value* from
//!     `QrMode::Reduced` matches numpy's mode='r' R (validated, passes).
//!   * `svd(compute_uv=False)` — `svdvals()` already covers this
//!     (validated, passes).
//!   * det / solve / inv / qr-reconstruction / svd-reconstruction values —
//!     validated against the oracle (pass).

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_linalg::{NormOrder, QrMode, det, inv, norm, qr, slogdet, solve, svd, svdvals};

// Well-conditioned non-symmetric 3x3 used for det/solve/inv/svd checks.
//   A = [[2,1,0],[1,3,1],[0,1,4]]  (this one is symmetric — fine for values)
fn matrix_a() -> Vec<f64> {
    vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]
}

// Non-symmetric 3x3 whose min column abs-sum (1.0) differs from min row
// abs-sum (6.0) and every other supported order — required to genuinely
// pin the ord=-1 gap.
//   B = [[10,0,0],[5,1,0],[5,0,2]]
//   col abs-sums = [20, 1, 2] -> min = 1.0  (numpy ord=-1)
//   row abs-sums = [10, 6, 7] -> min = 6.0  (numpy ord=-inf / NegInf)
fn matrix_b() -> Vec<f64> {
    vec![10.0, 0.0, 0.0, 5.0, 1.0, 0.0, 5.0, 0.0, 2.0]
}

// ---------------------------------------------------------------------------
// LIBRARY GAP 1 — norm matrix ord = -1 (min column abs-sum)
//
// numpy `_linalg.py:2657`:  "-1   min(sum(abs(x), axis=0))"
// numpy `_linalg.py:2833-2836`: elif ord == -1: ret = add.reduce(abs(x),
//   axis=row_axis).min(axis=col_axis)
//
// Oracle:  np.linalg.norm(B, -1) == 1.0  (min column abs-sum of B)
//
// SHIPPED via `NormOrder::NegL1` (norms.rs matrix_norm_scalar): min column
// abs-sum, the transpose of `L1` (max column sum). Distinct from `NegInf`
// (min *row* sum = 6.0) and from `L1` (max col sum = 20.0).
// ---------------------------------------------------------------------------
#[test]
fn divergence_norm_matrix_ord_neg1_unrepresentable() {
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 3]), matrix_b()).unwrap();
    let expected_np_ord_neg1 = 1.0_f64; // np.linalg.norm(B, -1)

    let got = norm(&b, NormOrder::NegL1).unwrap();
    assert!(
        (got - expected_np_ord_neg1).abs() < 1e-10,
        "NormOrder::NegL1 = {got} != numpy.linalg.norm(B, ord=-1) = {expected_np_ord_neg1} \
         (min column abs-sum)"
    );

    // It must be distinct from NegInf (min ROW sum = 6.0) and L1 (max col
    // sum = 20.0) — proves NegL1 is genuinely a new order, not an alias.
    assert!((norm(&b, NormOrder::NegInf).unwrap() - 6.0).abs() < 1e-10);
    assert!((norm(&b, NormOrder::L1).unwrap() - 20.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// LIBRARY GAP 2 — norm matrix ord = -2 (smallest singular value)
//
// numpy `_linalg.py:2659`:  "-2   smallest singular value"
// numpy `_linalg.py:2823-2824`:  elif ord == -2: ret = _multi_svd_norm(... amin)
//
// Oracle:  np.linalg.norm(A, -2) == 1.2679491924311228  (smallest sing. val.)
//
// SHIPPED via `NormOrder::NegL2` (norms.rs matrix_norm_scalar): smallest
// singular value (svals.last()). The mirror of `L2` which returns the
// LARGEST singular value (svals[0]).
// ---------------------------------------------------------------------------
#[test]
fn divergence_norm_matrix_ord_neg2_unrepresentable() {
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 3]), matrix_a()).unwrap();
    let expected_np_ord_neg2 = 1.267_949_192_431_122_8_f64; // np.linalg.norm(A, -2)

    let got = norm(&a, NormOrder::NegL2).unwrap();
    assert!(
        (got - expected_np_ord_neg2).abs() < 1e-9,
        "NormOrder::NegL2 = {got} != numpy.linalg.norm(A, ord=-2) = {expected_np_ord_neg2} \
         (smallest singular value)"
    );

    // It must be distinct from L2 (largest singular value). For A the
    // largest sv differs from the smallest, proving NegL2 is not an alias.
    let l2 = norm(&a, NormOrder::L2).unwrap();
    assert!(
        (l2 - got).abs() > 1e-3,
        "NegL2 (smallest sv = {got}) must differ from L2 (largest sv = {l2})"
    );
}

// ---------------------------------------------------------------------------
// VALIDATION — norm orders that ferray DOES support match the oracle.
// (PASS; documents that the *supported* matrix orders are correct.)
//   On B: np.linalg.norm(B,1)=20.0, norm(B,inf)=10.0, norm(B,-inf)=6.0
// ---------------------------------------------------------------------------
#[test]
fn validate_norm_supported_orders_match_numpy() {
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 3]), matrix_b()).unwrap();
    // L1 = max column abs-sum -> 20.0
    assert!((norm(&b, NormOrder::L1).unwrap() - 20.0).abs() < 1e-10);
    // Inf = max row abs-sum -> 10.0
    assert!((norm(&b, NormOrder::Inf).unwrap() - 10.0).abs() < 1e-10);
    // NegInf = min row abs-sum -> 6.0
    assert!((norm(&b, NormOrder::NegInf).unwrap() - 6.0).abs() < 1e-10);
    // L2 = largest singular value -> 12.28201176719462
    assert!((norm(&b, NormOrder::L2).unwrap() - 12.282_011_767_194_62).abs() < 1e-9);
    // Fro -> 12.449899597988733
    assert!((norm(&b, NormOrder::Fro).unwrap() - 12.449_899_597_988_733).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// VALIDATION — qr(mode='r'): numpy returns the same R as mode='reduced'.
//
// numpy `_linalg.py:1142-1145`:  if mode == 'r': r = triu(a[..., :mn, :]) ...
// numpy docstring `_linalg.py:1070-1071`:
//     >>> R2 = np.linalg.qr(a, mode='r')
//     >>> np.allclose(R, R2)  # mode='r' returns the same R as mode='full'
//
// ferray has no `QrMode::R`, but the R *value* from `QrMode::Reduced` is the
// numpy mode='r' R (a binding can route mode='r' -> qr(Reduced).1). So 'r'
// is BINDING-constructible, NOT a library value gap.
// ---------------------------------------------------------------------------
#[test]
fn validate_qr_reduced_r_reconstructs() {
    let m = Array::<f64, Ix2>::from_vec(
        Ix2::new([4, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 1.0],
    )
    .unwrap();
    let (q, r) = qr(&m, QrMode::Reduced).unwrap();
    assert_eq!(r.shape(), &[3, 3]);
    let rs = r.as_slice().unwrap();
    // Upper-triangular, as numpy's triu R.
    assert!(rs[3].abs() < 1e-10); // (1,0)
    assert!(rs[6].abs() < 1e-10); // (2,0)
    assert!(rs[7].abs() < 1e-10); // (2,1)

    // Q @ R == M (reconstruction; sign/gauge-invariant).
    let qs = q.as_slice().unwrap();
    let ms = m.as_slice().unwrap();
    let k = q.shape()[1];
    for i in 0..4 {
        for j in 0..3 {
            let mut acc = 0.0;
            for p in 0..k {
                acc += qs[i * k + p] * rs[p * 3 + j];
            }
            assert!((acc - ms[i * 3 + j]).abs() < 1e-9, "Q@R != M at [{i},{j}]");
        }
    }
}

// ---------------------------------------------------------------------------
// VALIDATION — svd(compute_uv=False) is covered by svdvals().
//
// numpy `_linalg.py:1860-1906`: svdvals(x) == svd(x, compute_uv=False).
// Oracle for M=[[1,2,3],[4,5,6],[7,8,10],[2,1,1]]:
//   svdvals == [17.548263371295064, 1.4142135623730956, 0.24176983199712354]
//
// ferray has `svdvals()` returning exactly the 1-D singular values, so
// compute_uv=False is a BINDING-routing concern, NOT a library gap.
// ---------------------------------------------------------------------------
#[test]
fn validate_svdvals_matches_numpy() {
    let m = Array::<f64, Ix2>::from_vec(
        Ix2::new([4, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 1.0],
    )
    .unwrap();
    let sv = svdvals(&m).unwrap();
    let s = sv.as_slice().unwrap();
    let expected = [
        17.548_263_371_295_064,
        1.414_213_562_373_095_6,
        0.241_769_831_997_123_54,
    ];
    assert_eq!(s.len(), 3);
    for (got, exp) in s.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-9, "svdval {got} != numpy {exp}");
    }
}

// ---------------------------------------------------------------------------
// VALIDATION — det / slogdet / solve / inv values against the oracle.
// (faer-backed; expected to PASS — confirms no value divergence.)
// ---------------------------------------------------------------------------
#[test]
fn validate_det_solve_inv_match_numpy() {
    let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 3]), matrix_a()).unwrap();

    // np.linalg.det(A) == 18.0 (17.999999999999996)
    assert!((det(&a).unwrap() - 18.0).abs() < 1e-9);

    // np.linalg.slogdet(A) == (1.0, 2.8903717578961645)
    let (sign, logdet) = slogdet(&a).unwrap();
    assert!((sign - 1.0).abs() < 1e-12);
    assert!((logdet - 2.890_371_757_896_164_5).abs() < 1e-9);

    // np.linalg.solve(A, [1,2,3]) == [1/3, 1/3, 2/3]
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let x = solve(&a, &b).unwrap();
    let xs: Vec<f64> = x.iter().copied().collect();
    let x_exp = [
        0.333_333_333_333_333_3,
        0.333_333_333_333_333_37,
        0.666_666_666_666_666_6,
    ];
    for (g, e) in xs.iter().zip(x_exp.iter()) {
        assert!((g - e).abs() < 1e-9, "solve {g} != numpy {e}");
    }

    // np.linalg.inv(A): first row [0.6111.., -0.2222.., 0.0555..]
    let inv_a = inv(&a).unwrap();
    let inv_s = inv_a.as_slice().unwrap();
    let inv_exp = [
        0.611_111_111_111_111_2,
        -0.222_222_222_222_222_24,
        0.055_555_555_555_555_56,
        -0.222_222_222_222_222_24,
        0.444_444_444_444_444_5,
        -0.111_111_111_111_111_12,
        0.055_555_555_555_555_56,
        -0.111_111_111_111_111_12,
        0.277_777_777_777_777_8,
    ];
    for (g, e) in inv_s.iter().zip(inv_exp.iter()) {
        assert!((g - e).abs() < 1e-9, "inv {g} != numpy {e}");
    }
}

// ---------------------------------------------------------------------------
// VALIDATION — svd reconstruction U @ diag(S) @ Vt == A (gauge-invariant).
// ---------------------------------------------------------------------------
#[test]
fn validate_svd_reconstructs() {
    let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 3]), matrix_a()).unwrap();
    let (u, s, vt) = svd(&a, false).unwrap();
    let us = u.as_slice().unwrap();
    let ss = s.as_slice().unwrap();
    let vts = vt.as_slice().unwrap();
    let n = 3;
    let k = ss.len();
    let av = matrix_a();
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += us[i * k + p] * ss[p] * vts[p * n + j];
            }
            assert!(
                (acc - av[i * n + j]).abs() < 1e-9,
                "U diag(S) Vt != A at [{i},{j}]"
            );
        }
    }
}
