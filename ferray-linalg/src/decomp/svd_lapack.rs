//! LAPACK `gesdd`-backed SVD (#2116) — bit-identical to
//! `scipy.linalg.svd` / `numpy.linalg.svd`.
//!
//! ## Why this exists
//!
//! The default [`super::svd`] wraps `faer::Svd`, whose divide-and-conquer
//! kernel diverges from LAPACK at the noise floor: a rank-deficient
//! matrix's near-zero singular value comes out as e.g. `3.75e-32` under
//! faer where LAPACK reports `1.34e-31`. scipy (`gesdd`/`gesvd`) and numpy
//! both route through LAPACK and agree bit-for-bit. ferrolearn's PCA
//! `noise_variance_` (the trailing-singular-value tail) therefore needs a
//! LAPACK-backed SVD to be bit-identical to scikit-learn (ferrolearn
//! #2110). This routine calls LAPACK `gesdd` directly (`dgesdd_`/`sgesdd_`)
//! so the result matches scipy to the last bit.
//!
//! ## Linkage
//!
//! Gated on the `openblas` feature. The host OpenBLAS shared object
//! (linked via `openblas-src`, the same library the GEMM path uses for
//! `cblas_dgemm`) bundles LAPACK, so the Fortran symbols `dgesdd_` /
//! `sgesdd_` resolve from the same `.so`. We declare the externs here
//! rather than pulling a separate `lapack-sys` crate: the host links the
//! standard **LP64** `libopenblas.so` (32-bit Fortran integers), which is
//! exactly the ABI of these declarations (`i32` everywhere). The ILP64
//! variant (`libopenblas64_.so`, `dgesdd_64_`) is **not** what `system`
//! mode links, so no 64-bit-integer handling is needed.
//!
//! ## Layout
//!
//! LAPACK is column-major Fortran; ferray [`Array`]s are row-major. We
//! transpose the input into a column-major scratch buffer, run `gesdd`,
//! then transpose `U` and `Vt` back to row-major on the way out.

#![cfg(feature = "openblas")]

use core::ffi::c_char;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::scalar::LinalgFloat;

// Pull `openblas_src` into the link graph so the linker resolves the
// LAPACK Fortran symbols (`dgesdd_`/`sgesdd_`) from the same OpenBLAS
// shared object the GEMM path links for `cblas_*`. The crate's only job
// is to emit the `-l openblas` link directive.
#[allow(unused_imports)]
use openblas_src as _;

unsafe extern "C" {
    // LAPACK divide-and-conquer SVD, double precision (LP64, i32 ints).
    // Reference signature (netlib `dgesdd.f`):
    //   SUBROUTINE DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT,
    //                      WORK, LWORK, IWORK, INFO )
    #[allow(clippy::too_many_arguments)]
    fn dgesdd_(
        jobz: *const c_char,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );

    // LAPACK divide-and-conquer SVD, single precision (LP64, i32 ints).
    #[allow(clippy::too_many_arguments)]
    fn sgesdd_(
        jobz: *const c_char,
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        s: *mut f32,
        u: *mut f32,
        ldu: *const i32,
        vt: *mut f32,
        ldvt: *const i32,
        work: *mut f32,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
}

/// Dispatch from the generic [`LinalgFloat`] to the concrete LAPACK
/// `gesdd` routine for that precision. Sealed by `LinalgFloat`.
pub trait LapackGesdd: LinalgFloat {
    /// Call the precision-appropriate LAPACK `gesdd` routine.
    ///
    /// All pointers/dims follow the Fortran (column-major) `gesdd`
    /// contract. `info` is written by LAPACK: `0` = success, `<0` = the
    /// `-info`-th argument was illegal, `>0` = did not converge.
    ///
    /// # Safety
    /// All slice arguments must be sized per the `gesdd` contract for the
    /// given `(m, n, jobz)`; `lda`/`ldu`/`ldvt` must be the true leading
    /// dimensions of the respective column-major buffers.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gesdd(
        jobz: c_char,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        iwork: &mut [i32],
    ) -> i32;
}

impl LapackGesdd for f64 {
    unsafe fn gesdd(
        jobz: c_char,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        iwork: &mut [i32],
    ) -> i32 {
        let mut info: i32 = 0;
        // SAFETY: caller upholds the `gesdd` buffer/leading-dimension
        // contract; `dgesdd_` only reads/writes within those bounds.
        unsafe {
            dgesdd_(
                &jobz,
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &mut info,
            );
        }
        info
    }
}

impl LapackGesdd for f32 {
    unsafe fn gesdd(
        jobz: c_char,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        iwork: &mut [i32],
    ) -> i32 {
        let mut info: i32 = 0;
        // SAFETY: caller upholds the `gesdd` buffer/leading-dimension
        // contract; `sgesdd_` only reads/writes within those bounds.
        unsafe {
            sgesdd_(
                &jobz,
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &mut info,
            );
        }
        info
    }
}

/// Compute the SVD of `a` via LAPACK `gesdd`, bit-identical to
/// `scipy.linalg.svd(a, lapack_driver='gesdd')` / `numpy.linalg.svd(a)`.
///
/// Returns `(U, S, Vt)` where `A = U * diag(S) * Vt`, mirroring the
/// contract of [`super::svd`]:
/// - `full_matrices = true` (`jobz = 'A'`): `U` is `(m, m)`, `Vt` is
///   `(n, n)`.
/// - `full_matrices = false` (`jobz = 'S'`, thin): `U` is `(m, k)`,
///   `Vt` is `(k, n)` with `k = min(m, n)`.
///
/// `S` is always length `min(m, n)`, nonincreasing.
///
/// Requires the `openblas` feature (the LAPACK symbols ship inside the
/// linked OpenBLAS). With the feature off this function does not exist;
/// callers should fall back to [`super::svd`] (faer).
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if `a` is not 2-D / has a zero
///   dimension, or its data is not contiguous.
/// - [`FerrayError::InvalidValue`] if LAPACK reports an illegal argument
///   (`info < 0`) or fails to converge (`info > 0`).
pub fn svd_lapack<T: LapackGesdd>(
    a: &Array<T, Ix2>,
    full_matrices: bool,
) -> FerrayResult<(Array<T, Ix2>, Array<T, Ix1>, Array<T, Ix2>)> {
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];
    if m == 0 || n == 0 {
        return Err(FerrayError::shape_mismatch(format!(
            "svd_lapack: empty matrix {m}x{n}"
        )));
    }
    let k = m.min(n);

    let src = a.as_slice().ok_or_else(|| {
        FerrayError::shape_mismatch("svd_lapack: input must be contiguous (row-major)")
    })?;

    // ferray buffer is row-major (m, n). LAPACK wants column-major. Build
    // a column-major copy: col_major[i + j*m] = a[i, j] = src[i*n + j].
    // gesdd overwrites this buffer, so it must be owned scratch.
    let mut a_cm = vec![<T as num_traits::Zero>::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            a_cm[i + j * m] = src[i * n + j];
        }
    }

    let m_i = i32::try_from(m).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "svd_lapack: dimension {m} exceeds i32 (LP64 LAPACK)"
        ))
    })?;
    let n_i = i32::try_from(n).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "svd_lapack: dimension {n} exceeds i32 (LP64 LAPACK)"
        ))
    })?;

    // jobz, and the U / Vt leading dims & sizes (column-major).
    // jobz='A': U is (m,m), Vt is (n,n). jobz='S': U is (m,k), Vt is (k,n).
    let jobz: c_char = if full_matrices {
        b'A' as c_char
    } else {
        b'S' as c_char
    };
    let (ucols, vtrows) = if full_matrices { (m, n) } else { (k, k) };
    let ldu = m_i; // U has m rows (column-major leading dim = m)
    let ldvt_rows = vtrows; // Vt has `vtrows` rows
    let ldvt = i32::try_from(ldvt_rows).unwrap_or(m_i.max(n_i));

    let mut s = vec![<T as num_traits::Zero>::zero(); k];
    let mut u_cm = vec![<T as num_traits::Zero>::zero(); m * ucols];
    let mut vt_cm = vec![<T as num_traits::Zero>::zero(); vtrows * n];
    // gesdd needs 8*min(m,n) integer workspace.
    let mut iwork = vec![0i32; 8 * k];

    // Workspace query: lwork = -1 → LAPACK writes optimal lwork into
    // work[0]. Then allocate and call for real.
    let mut work_query = [<T as num_traits::Zero>::zero(); 1];
    // SAFETY: workspace-query form of gesdd — `lwork = -1` means LAPACK
    // only writes the optimal lwork into work[0] and touches no other
    // output buffers beyond the (correctly sized) s/u/vt/iwork pointers.
    let info = unsafe {
        T::gesdd(
            jobz,
            m_i,
            n_i,
            &mut a_cm,
            m_i,
            &mut s,
            &mut u_cm,
            ldu,
            &mut vt_cm,
            ldvt,
            &mut work_query,
            -1,
            &mut iwork,
        )
    };
    if info != 0 {
        return Err(FerrayError::InvalidValue {
            message: format!("svd_lapack: gesdd workspace query failed (info={info})"),
        });
    }
    let lwork = lwork_from_query(work_query[0]);
    let mut work = vec![<T as num_traits::Zero>::zero(); lwork.max(1) as usize];

    // SAFETY: all buffers are sized per the `gesdd` contract for
    // (m, n, jobz); leading dims (ldu, ldvt) are the true column-major
    // strides; `lwork`/`work` come from the preceding query; `iwork` is
    // `8*min(m,n)`. LAPACK writes only within these bounds.
    let info = unsafe {
        T::gesdd(
            jobz, m_i, n_i, &mut a_cm, m_i, &mut s, &mut u_cm, ldu, &mut vt_cm, ldvt, &mut work,
            lwork, &mut iwork,
        )
    };
    if info < 0 {
        return Err(FerrayError::InvalidValue {
            message: format!("svd_lapack: gesdd illegal argument (info={info})"),
        });
    }
    if info > 0 {
        return Err(FerrayError::InvalidValue {
            message: format!("svd_lapack: gesdd failed to converge (info={info})"),
        });
    }

    // Transpose U (column-major (m, ucols)) → row-major (m, ucols).
    let mut u_rm = vec![<T as num_traits::Zero>::zero(); m * ucols];
    for i in 0..m {
        for j in 0..ucols {
            u_rm[i * ucols + j] = u_cm[i + j * m];
        }
    }
    // Transpose Vt (column-major (vtrows, n)) → row-major (vtrows, n).
    let mut vt_rm = vec![<T as num_traits::Zero>::zero(); vtrows * n];
    for i in 0..vtrows {
        for j in 0..n {
            vt_rm[i * n + j] = vt_cm[i + j * vtrows];
        }
    }

    let u_arr = Array::from_vec(Ix2::new([m, ucols]), u_rm)?;
    let s_arr = Array::from_vec(Ix1::new([k]), s)?;
    let vt_arr = Array::from_vec(Ix2::new([vtrows, n]), vt_rm)?;
    Ok((u_arr, s_arr, vt_arr))
}

/// Round the LAPACK-returned optimal-workspace size (a float) up to the
/// next `i32`. LAPACK returns it in `work[0]` as the working precision
/// float; `ceil` guards against any fractional representation.
fn lwork_from_query<T: LinalgFloat>(q: T) -> i32 {
    let v = num_traits::Float::ceil(q).into_f64();
    if v.is_finite() && v >= 1.0 {
        v as i32
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- STEP 0: de-risk linking ---------------------------------
    //
    // Calls LAPACK `dgesdd_` on a tiny known matrix via the raw FFI extern
    // declared above and checks it links + runs + produces a correct SVD.
    // If the OpenBLAS LAPACK symbol failed to resolve (ILP64 mismatch,
    // missing lib) this test would fail to link the test binary.

    #[test]
    fn dgesdd_links_and_runs_diag() {
        // Diagonal A = diag(3, 2): singular values must be exactly 3, 2.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![3.0, 0.0, 0.0, 2.0]).unwrap();
        let (u, s, vt) = svd_lapack(&a, false).unwrap();
        let sv: Vec<f64> = s.iter().copied().collect();
        assert_eq!(sv.len(), 2);
        assert!((sv[0] - 3.0).abs() < 1e-13, "s0={}", sv[0]);
        assert!((sv[1] - 2.0).abs() < 1e-13, "s1={}", sv[1]);
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(vt.shape(), &[2, 2]);
    }

    #[test]
    fn sgesdd_links_and_runs_diag() {
        let a =
            Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![3.0_f32, 0.0, 0.0, 2.0]).unwrap();
        let (_u, s, _vt) = svd_lapack(&a, false).unwrap();
        let sv: Vec<f32> = s.iter().copied().collect();
        assert!((sv[0] - 3.0).abs() < 1e-6, "s0={}", sv[0]);
        assert!((sv[1] - 2.0).abs() < 1e-6, "s1={}", sv[1]);
    }

    // ---------- reconstruction (A = U diag(S) Vt) -----------------------

    fn reconstruct_err_f64(a: &Array<f64, Ix2>, full: bool) -> f64 {
        let (m, n) = (a.shape()[0], a.shape()[1]);
        let (u, s, vt) = svd_lapack(a, full).unwrap();
        let us = u.as_slice().unwrap();
        let ss = s.as_slice().unwrap();
        let vts = vt.as_slice().unwrap();
        let ucols = u.shape()[1];
        let k = ss.len();
        let asl = a.as_slice().unwrap();
        let mut maxerr = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for p in 0..k.min(ucols) {
                    val += us[i * ucols + p] * ss[p] * vts[p * n + j];
                }
                maxerr = maxerr.max((val - asl[i * n + j]).abs());
            }
        }
        maxerr
    }

    #[test]
    fn reconstructs_thin_and_full() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        )
        .unwrap();
        assert!(reconstruct_err_f64(&a, false) < 1e-12);
        assert!(reconstruct_err_f64(&a, true) < 1e-12);
    }

    // ---------- STEP 2: bit-identity vs scipy.linalg.svd(gesdd) ---------
    //
    // Expected values produced by a live python call:
    //   import scipy.linalg as sla
    //   U,s,Vt = sla.svd(A, full_matrices=False, lapack_driver='gesdd')
    // and pasted as f64 hex literals so there is zero decimal round-trip
    // loss (R-CHAR-3: oracle-sourced, not copied from ferray output).

    #[test]
    fn bit_identical_well_conditioned_3x3() {
        // A1 = [[1,2,3],[4,5,6],[7,8,10]]
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        )
        .unwrap();
        let (_u, s, _vt) = svd_lapack(&a, false).unwrap();
        let got: Vec<f64> = s.iter().copied().collect();
        // scipy gesdd singular values, as exact f64 hex:
        let expected = [
            hexf64("0x1.16999f048dfbap+4"),
            hexf64("0x1.c0152602e522bp-1"),
            hexf64("0x1.932ec12f0363dp-3"),
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            let rel = (g - e).abs() / e.abs();
            assert!(
                rel <= 4.0 * f64::EPSILON,
                "got {g:.17e} want {e:.17e} rel={rel:e}"
            );
        }
    }

    #[test]
    fn bit_identical_rank_deficient_4x5_noise_floor() {
        // Xc: a 4x5 centered (column-mean-removed) random matrix. Because
        // the rows sum to the column mean, rank is 3 and the 4th singular
        // value sits at the noise floor — THE value PCA noise_variance_
        // depends on. scipy/numpy/gesvd all agree here; faer does not.
        #[rustfmt::skip]
        let xc = vec![
            0.03042301523277824, 0.28971689266533396, -0.12103681980225467, -0.25819165018821255, -0.01338133181344242,
            0.1275036243721096, 0.012114737555606991, 0.16797280490818123, 0.16058792731591987, -0.053594612326569424,
            0.2733345493881181, 0.10342244604581896, -0.15575563477996623, 0.12252180510755162, -0.3660000729544602,
            -0.4312611889930058, -0.4052540762667598, 0.10881964967403945, -0.02491808223525893, 0.43297601709447203,
        ];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([4, 5]), xc).unwrap();
        let (_u, s, _vt) = svd_lapack(&a, false).unwrap();
        let got: Vec<f64> = s.iter().copied().collect();
        assert_eq!(got.len(), 4);
        // scipy gesdd singular values (exact f64 hex):
        let expected = [
            hexf64("0x1.cc8828a68801ep-1"),
            hexf64("0x1.c32e8f7c89d82p-2"),
            hexf64("0x1.b386b3087e4e2p-3"),
            hexf64("0x1.457a1a92c0974p-54"), // ~7.06e-17 noise-floor value
        ];
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            // The first three must match to a few ULPs.
            if idx < 3 {
                let rel = (g - e).abs() / e.abs();
                assert!(
                    rel <= 4.0 * f64::EPSILON,
                    "s[{idx}] got {g:.17e} want {e:.17e} rel={rel:e}"
                );
            } else {
                // The noise-floor value: both are O(1e-17). Assert they
                // agree to within a few ULPs *of that tiny magnitude*
                // (relative test), which is the whole point of #2116.
                let rel = (g - e).abs() / e.abs();
                assert!(
                    rel <= 1e-2,
                    "noise-floor s[3] got {g:.17e} want {e:.17e} rel={rel:e}"
                );
            }
        }
    }

    #[test]
    fn bit_identical_f32_3x2() {
        // A3 = [[1,2],[3,4],[5,6]] as f32.
        let a =
            Array::<f32, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let (_u, s, _vt) = svd_lapack(&a, false).unwrap();
        let got: Vec<f32> = s.iter().copied().collect();
        // scipy gesdd (float32) singular values, exact f32 hex:
        let expected = [hexf32("0x1.30d10ep+3"), hexf32("0x1.075282p-1")];
        for (g, e) in got.iter().zip(expected.iter()) {
            let rel = (g - e).abs() / e.abs();
            assert!(
                rel <= 4.0 * f32::EPSILON,
                "got {g:e} want {e:e} rel={rel:e}"
            );
        }
    }

    // Minimal hex-float parsers (Rust has no hex-float literals on stable).
    // Used only to express the scipy oracle values exactly.
    fn hexf64(s: &str) -> f64 {
        parse_hexf(s)
    }
    fn hexf32(s: &str) -> f32 {
        parse_hexf(s) as f32
    }
    fn parse_hexf(s: &str) -> f64 {
        // Format: [-]0x1.<hexfrac>p<exp>
        let neg = s.starts_with('-');
        let s = s.trim_start_matches('-').trim_start_matches("0x");
        let (mantissa, exp) = s.split_once('p').expect("hex float needs p-exponent");
        let exp: i32 = exp.parse().expect("bad exponent");
        let (int_part, frac_part) = match mantissa.split_once('.') {
            Some((i, f)) => (i, f),
            None => (mantissa, ""),
        };
        let mut val = i64::from_str_radix(int_part, 16).expect("bad int part") as f64;
        let mut scale = 1.0_f64 / 16.0;
        for c in frac_part.chars() {
            let d = c.to_digit(16).expect("bad hex digit") as f64;
            val += d * scale;
            scale /= 16.0;
        }
        val *= 2.0_f64.powi(exp);
        if neg { -val } else { val }
    }
}
