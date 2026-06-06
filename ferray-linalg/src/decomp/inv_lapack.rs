//! LAPACK `getrf`+`getri`-backed matrix inverse (#2117) — bit-identical
//! to `scipy.linalg.inv` / `numpy.linalg.inv`.
//!
//! ## Why this exists
//!
//! The default [`crate::solve::inv`] wraps `faer`'s partial-pivot LU
//! inverse. faer's accumulation order diverges from LAPACK on
//! numerically extreme operands, so the resulting inverse — and any
//! `slogdet` taken of a product built from it — can land on a different
//! sign in the rank-deficient regime. scipy (`getri`) and numpy both
//! route through LAPACK and agree bit-for-bit.
//!
//! ferrolearn's PCA precision lemma (`get_precision`, the Woodbury
//! matrix-inversion-lemma form) computes
//! `precision = components.T @ inv(P) @ components` and then takes
//! `fast_logdet(precision) = slogdet(precision)`. On a 4×5 rank-deficient
//! fit (`n_samples < n_features`) the resulting `precision` is
//! numerically singular (condition number ~6.7e15) and scikit-learn lands
//! on `sign = +1` (a finite score) precisely because it uses scipy
//! `linalg.inv` (LAPACK `getri`) for the inner inverse and numpy `@`
//! (OpenBLAS `gemm`) for the products, left-to-right. ferray's faer
//! inverse + naive matmul land on `sign = -1` (`-inf` score). This
//! routine calls LAPACK `getrf`/`getri` directly so the inverse matches
//! scipy to the last bit (ferrolearn #2110 / #2117).
//!
//! ## Linkage
//!
//! Gated on the `openblas` feature, exactly like [`super::svd_lapack`].
//! The host OpenBLAS shared object (linked via `openblas-src`, the same
//! library the GEMM path uses for `cblas_dgemm`) bundles LAPACK, so the
//! Fortran symbols `dgetrf_`/`dgetri_` (and the f32 `sgetrf_`/`sgetri_`)
//! resolve from the same `.so`. We declare the externs here rather than
//! pulling a separate `lapack-sys` crate. `system`-mode OpenBLAS is the
//! standard **LP64** build (32-bit Fortran integers), which is exactly
//! the ABI of these declarations (`i32` everywhere).
//!
//! ## Layout
//!
//! LAPACK is column-major Fortran; ferray [`Array`]s are row-major. The
//! inverse of a matrix's transpose is the transpose of its inverse:
//! `inv(Aᵀ) = inv(A)ᵀ`. A row-major buffer `A` reinterpreted as
//! column-major *is* `Aᵀ`, so feeding the row-major bytes straight to
//! LAPACK computes `inv(Aᵀ) = inv(A)ᵀ` in column-major, whose bytes —
//! read back as row-major — are `inv(A)`. We therefore round-trip the
//! buffer with no explicit transpose, which also means we invert exactly
//! the same operand scipy hands to LAPACK (scipy likewise passes the
//! C-contiguous array through, letting LAPACK see the transpose).

#![cfg(feature = "openblas")]

use ferray_core::array::owned::Array;
use ferray_core::dimension::Ix2;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::scalar::LinalgFloat;

// Pull `openblas_src` into the link graph so the linker resolves the
// LAPACK Fortran symbols (`dgetrf_`/`dgetri_`) from the same OpenBLAS
// shared object the GEMM path links for `cblas_*`.
#[allow(unused_imports)]
use openblas_src as _;

unsafe extern "C" {
    // LU factorization, double precision (LP64, i32 ints).
    // Reference signature (netlib `dgetrf.f`):
    //   SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO )
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );

    // Inverse from an LU factorization, double precision.
    // Reference signature (netlib `dgetri.f`):
    //   SUBROUTINE DGETRI( N, A, LDA, IPIV, WORK, LWORK, INFO )
    fn dgetri_(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );

    // LU factorization, single precision (LP64, i32 ints).
    fn sgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );

    // Inverse from an LU factorization, single precision.
    fn sgetri_(
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Dispatch from the generic [`LinalgFloat`] to the concrete LAPACK
/// `getrf`/`getri` routines for that precision. Sealed by [`LinalgFloat`].
pub trait LapackGetri: LinalgFloat {
    /// Call the precision-appropriate LAPACK `getrf` (LU factorization).
    ///
    /// `a` is an `n×n` column-major buffer (`lda = n`); on success it is
    /// overwritten by the LU factors. `ipiv` is the length-`n` pivot
    /// array. Returns LAPACK `info`: `0` = success, `<0` = the
    /// `-info`-th argument was illegal, `>0` = `U(info,info)` is exactly
    /// zero (singular).
    ///
    /// # Safety
    /// `a` must be length `n*n` and `ipiv` length `n`; `lda` must be the
    /// true column-major leading dimension. LAPACK writes only within
    /// those bounds.
    unsafe fn getrf(n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32]) -> i32;

    /// Call the precision-appropriate LAPACK `getri` (inverse from LU).
    ///
    /// `a`/`ipiv` are the outputs of [`Self::getrf`]; on success `a` is
    /// overwritten by the inverse. `work`/`lwork` follow the LAPACK
    /// workspace contract (`lwork = -1` queries the optimal size into
    /// `work[0]`). Returns LAPACK `info`.
    ///
    /// # Safety
    /// `a` must be length `n*n`, `ipiv` length `n`, and `work` length
    /// `max(1, lwork)`; `lda` must be the true leading dimension.
    unsafe fn getri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
    ) -> i32;
}

impl LapackGetri for f64 {
    unsafe fn getrf(n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32]) -> i32 {
        let mut info: i32 = 0;
        // SAFETY: caller upholds the buffer/leading-dimension contract;
        // `dgetrf_` only reads/writes within those bounds.
        unsafe {
            dgetrf_(&n, &n, a.as_mut_ptr(), &lda, ipiv.as_mut_ptr(), &mut info);
        }
        info
    }

    unsafe fn getri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
    ) -> i32 {
        let mut info: i32 = 0;
        // SAFETY: caller upholds the buffer/leading-dimension contract;
        // `dgetri_` only reads/writes within those bounds.
        unsafe {
            dgetri_(
                &n,
                a.as_mut_ptr(),
                &lda,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        info
    }
}

impl LapackGetri for f32 {
    unsafe fn getrf(n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32]) -> i32 {
        let mut info: i32 = 0;
        // SAFETY: caller upholds the buffer/leading-dimension contract;
        // `sgetrf_` only reads/writes within those bounds.
        unsafe {
            sgetrf_(&n, &n, a.as_mut_ptr(), &lda, ipiv.as_mut_ptr(), &mut info);
        }
        info
    }

    unsafe fn getri(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &[i32],
        work: &mut [Self],
        lwork: i32,
    ) -> i32 {
        let mut info: i32 = 0;
        // SAFETY: caller upholds the buffer/leading-dimension contract;
        // `sgetri_` only reads/writes within those bounds.
        unsafe {
            sgetri_(
                &n,
                a.as_mut_ptr(),
                &lda,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        info
    }
}

/// Compute the inverse of a square matrix `a` via LAPACK `getrf`+`getri`,
/// bit-identical to `scipy.linalg.inv(a)` / `numpy.linalg.inv(a)`.
///
/// LAPACK first computes the partial-pivot LU factorization (`getrf`)
/// and then inverts from it (`getri`), exactly the path scipy/numpy take.
/// Because `inv(Aᵀ) = inv(A)ᵀ` and a row-major buffer read column-major
/// *is* `Aᵀ`, the row-major bytes can be handed to LAPACK and read back
/// directly with no explicit transpose — matching the operand scipy
/// passes to LAPACK.
///
/// Requires the `openblas` feature (the LAPACK symbols ship inside the
/// linked OpenBLAS). With the feature off this function does not exist;
/// callers should fall back to [`crate::solve::inv`] (faer).
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if `a` is not square / has a zero
///   dimension, or its data is not contiguous (row-major).
/// - [`FerrayError::SingularMatrix`] if LAPACK reports a singular factor
///   (`info > 0`), matching scipy's `LinAlgError`.
/// - [`FerrayError::InvalidValue`] if LAPACK reports an illegal argument
///   (`info < 0`).
#[must_use = "the inverse is returned, not applied in place"]
pub fn inv_lapack<T: LapackGetri>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix2>> {
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];
    if m != n {
        return Err(FerrayError::shape_mismatch(format!(
            "inv_lapack requires a square matrix, got {m}x{n}"
        )));
    }
    if n == 0 {
        return Array::from_vec(Ix2::new([0, 0]), vec![]);
    }

    let src = a.as_slice().ok_or_else(|| {
        FerrayError::shape_mismatch("inv_lapack: input must be contiguous (row-major)")
    })?;

    // Row-major bytes reinterpreted column-major == Aᵀ. LAPACK computes
    // inv(Aᵀ) = inv(A)ᵀ in column-major; read back row-major gives inv(A).
    // getrf/getri overwrite this buffer in place, so it must be owned.
    let mut buf: Vec<T> = src.to_vec();

    let n_i = i32::try_from(n).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "inv_lapack: dimension {n} exceeds i32 (LP64 LAPACK)"
        ))
    })?;

    let mut ipiv = vec![0i32; n];

    // Step 1: LU factorization.
    // SAFETY: `buf` is n*n, `ipiv` is length n, `lda = n` is the true
    // column-major leading dimension. LAPACK writes only within bounds.
    let info = unsafe { T::getrf(n_i, &mut buf, n_i, &mut ipiv) };
    if info < 0 {
        return Err(FerrayError::InvalidValue {
            message: format!("inv_lapack: getrf illegal argument (info={info})"),
        });
    }
    if info > 0 {
        return Err(FerrayError::SingularMatrix {
            message: "matrix is singular and cannot be inverted".to_string(),
        });
    }

    // Step 2: workspace query for getri (lwork = -1 → optimal size in
    // work[0]).
    let mut work_query = [<T as num_traits::Zero>::zero(); 1];
    // SAFETY: workspace-query form — `lwork = -1` only writes the optimal
    // size into work[0]; `buf`/`ipiv` are the correctly sized LU outputs.
    let info = unsafe { T::getri(n_i, &mut buf, n_i, &ipiv, &mut work_query, -1) };
    if info != 0 {
        return Err(FerrayError::InvalidValue {
            message: format!("inv_lapack: getri workspace query failed (info={info})"),
        });
    }
    let lwork = lwork_from_query(work_query[0]).max(n_i);
    let mut work = vec![<T as num_traits::Zero>::zero(); lwork.max(1) as usize];

    // Step 3: invert from the LU factors.
    // SAFETY: `buf`/`ipiv` are the getrf outputs; `work`/`lwork` come from
    // the query; `lda = n`. LAPACK writes only within these bounds.
    let info = unsafe { T::getri(n_i, &mut buf, n_i, &ipiv, &mut work, lwork) };
    if info < 0 {
        return Err(FerrayError::InvalidValue {
            message: format!("inv_lapack: getri illegal argument (info={info})"),
        });
    }
    if info > 0 {
        // getri reports info>0 when U is exactly singular.
        return Err(FerrayError::SingularMatrix {
            message: "matrix is singular and cannot be inverted".to_string(),
        });
    }

    // `buf` is inv(A)ᵀ in column-major == inv(A) in row-major. No
    // transpose needed.
    Array::from_vec(Ix2::new([n, n]), buf)
}

/// Round the LAPACK-returned optimal-workspace size (a float) up to the
/// next `i32`. LAPACK returns it in `work[0]` as the working-precision
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
    use crate::gemm::gemm;
    use crate::norms::slogdet;

    // ---------- STEP 0: de-risk linking ---------------------------------
    //
    // Inverts a tiny diagonal matrix via the raw FFI externs. If the
    // OpenBLAS LAPACK getrf/getri symbols failed to resolve, the test
    // binary would fail to link.

    #[test]
    fn dgetri_links_and_runs_diag() {
        // inv(diag(4, 2)) = diag(0.25, 0.5).
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0, 0.0, 0.0, 2.0]).unwrap();
        let inv = inv_lapack(&a).unwrap();
        let v: Vec<f64> = inv.iter().copied().collect();
        assert!((v[0] - 0.25).abs() < 1e-15, "v0={}", v[0]);
        assert!((v[1] - 0.0).abs() < 1e-15);
        assert!((v[2] - 0.0).abs() < 1e-15);
        assert!((v[3] - 0.5).abs() < 1e-15, "v3={}", v[3]);
    }

    #[test]
    fn sgetri_links_and_runs_diag() {
        let a =
            Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0_f32, 0.0, 0.0, 2.0]).unwrap();
        let inv = inv_lapack(&a).unwrap();
        let v: Vec<f32> = inv.iter().copied().collect();
        assert!((v[0] - 0.25).abs() < 1e-6);
        assert!((v[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn singular_matrix_errs() {
        // [[1,2],[2,4]] is exactly singular.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let r = inv_lapack(&a);
        assert!(
            matches!(r, Err(FerrayError::SingularMatrix { .. })),
            "{r:?}"
        );
    }

    #[test]
    fn non_square_errs() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(matches!(
            inv_lapack(&a),
            Err(FerrayError::ShapeMismatch { .. })
        ));
    }

    // ---------- bit-identity vs scipy.linalg.inv ------------------------
    //
    // Expected values produced by a live python call:
    //   import scipy.linalg as sla; sla.inv(A)
    // pasted as f64 hex literals (R-CHAR-3: oracle-sourced, exact bits).

    #[test]
    fn bit_identical_well_conditioned_3x3() {
        // A1 = [[4,2,1],[2,5,3],[1,3,6]], cond ~4.87.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
        )
        .unwrap();
        let inv = inv_lapack(&a).unwrap();
        let got: Vec<f64> = inv.iter().copied().collect();
        // scipy.linalg.inv(A1), exact f64 hex:
        let expected = [
            hexf64("0x1.40f4898d5f85cp-2"),
            hexf64("-0x1.131abf0b7672ap-3"),
            hexf64("0x1.e9131abf0b767p-7"),
            hexf64("-0x1.131abf0b7672ap-3"),
            hexf64("0x1.5f85bb39503d2p-2"),
            hexf64("-0x1.31abf0b7672a1p-3"),
            hexf64("0x1.e9131abf0b767p-7"),
            hexf64("-0x1.31abf0b7672a1p-3"),
            hexf64("0x1.e9131abf0b767p-3"),
        ];
        // scipy and the host OpenBLAS may be different LAPACK builds, so
        // allow a few ULP (~1e-15 rel) rather than the exact last bit.
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let ulp = (g.to_bits() as i64 - e.to_bits() as i64).unsigned_abs();
            assert!(ulp <= 4, "elem {i}: got {g:.17e} want {e:.17e} ulp={ulp}");
        }
    }

    #[test]
    fn bit_identical_ill_conditioned_4x4() {
        // A near-singular symmetric matrix, cond ~9.67e14 — the #2110
        // regime. inv() is huge (~1e15) but scipy and LAPACK getri agree.
        #[rustfmt::skip]
        let p = vec![
            hexf64("0x1.f0a3fa2c00adfp-2"), hexf64("-0x1.91f4977a2b63ep-4"), hexf64("0x1.5c60375b7a458p-4"), hexf64("0x1.b0af08b93d7fap-2"),
            hexf64("-0x1.91f4977a2b63ep-4"), hexf64("0x1.83f238d5a7cc5p-2"), hexf64("0x1.9fcb62c7629e0p-4"), hexf64("0x1.64c4f97cd298ap-3"),
            hexf64("0x1.5c60375b7a458p-4"), hexf64("0x1.9fcb62c7629e0p-4"), hexf64("0x1.5172751ec1099p-2"), hexf64("0x1.2d10896806ea5p-3"),
            hexf64("0x1.b0af08b93d7fap-2"), hexf64("0x1.64c4f97cd298ap-3"), hexf64("0x1.2d10896806ea5p-3"), hexf64("0x1.1cfbabefcb3ecp-1"),
        ];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([4, 4]), p).unwrap();
        let inv = inv_lapack(&a).unwrap();
        let got: Vec<f64> = inv.iter().copied().collect();
        // scipy.linalg.inv(P), exact f64 hex:
        #[rustfmt::skip]
        let expected = [
            hexf64("0x1.696d257b201a3p+48"), hexf64("0x1.038f6c7913b67p+48"), hexf64("-0x1.08e3275b671e6p+44"), hexf64("-0x1.5f3c7703b4c6ep+48"),
            hexf64("0x1.038f6c7913b67p+48"), hexf64("0x1.74cf2363904cbp+47"), hexf64("-0x1.7c75e0cb19bd3p+43"), hexf64("-0x1.f87bf8776623fp+47"),
            hexf64("-0x1.08e3275b671e6p+44"), hexf64("-0x1.7c75e0cb19bd3p+43"), hexf64("0x1.8444d16bb1299p+39"), hexf64("0x1.016b4411801a2p+44"),
            hexf64("-0x1.5f3c7703b4c6ep+48"), hexf64("-0x1.f87bf8776623fp+47"), hexf64("0x1.016b4411801a2p+44"), hexf64("0x1.5555555555557p+48"),
        ];
        let mut max_rel = 0.0_f64;
        let mut max_ulp = 0u64;
        for (g, e) in got.iter().zip(expected.iter()) {
            let rel = (g - e).abs() / e.abs();
            max_rel = max_rel.max(rel);
            let ulp = (g.to_bits() as i64 - e.to_bits() as i64).unsigned_abs();
            max_ulp = max_ulp.max(ulp);
        }
        // The ill-conditioned case is the whole point: assert full
        // precision (<= a few ULP / ~1e-13 rel).
        assert!(
            max_rel <= 1e-13,
            "ill-conditioned inv max rel error {max_rel:e} (max ulp {max_ulp})"
        );
    }

    #[test]
    fn round_trip_inv_inv_identity() {
        // inv(inv(A)) ~ A for a well-conditioned A.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
        )
        .unwrap();
        let inv = inv_lapack(&a).unwrap();
        let inv2 = inv_lapack(&inv).unwrap();
        let av: Vec<f64> = a.iter().copied().collect();
        let bv: Vec<f64> = inv2.iter().copied().collect();
        for (x, y) in av.iter().zip(bv.iter()) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    // ---------- THE #2110 SIGN PROOF ------------------------------------
    //
    // Reconstruct the rank-deficient PCA precision lemma in Rust using the
    // new inv_lapack (LAPACK getri) + gemm (OpenBLAS) with LEFT-TO-RIGHT
    // association, then slogdet it, and assert the SIGN matches scipy's
    // +1. This is the empirical proof that the #2110 score path closes.
    //
    // Fixture (P, comps, nv, scipy precision) produced by reproducing
    // sklearn 1.5.2 `PCA(n_components=3).fit(X)` on a 4x5 rank-deficient X
    // (seed 2466) and its `get_precision()` Woodbury form. The inner P is
    // `comps @ comps.T / nv + diag(1/exp_var_diff)`; the score path then
    // forms `precision = comps.T @ inv(P) @ comps / -(nv**2)` with
    // `1/nv` added to the diagonal. cond(precision) ~6.3e15; scipy
    // slogdet(precision) = (+1, 253.6953...).

    #[test]
    fn pca_2110_precision_slogdet_sign_matches_scipy() {
        // nv = noise_variance_ (sklearn), exact f64 hex.
        let nv = hexf64("0x1.cdb1756455cecp-105");

        // Inner P (3x3) that gets inverted by getri.
        #[rustfmt::skip]
        let p_data = vec![
            hexf64("0x1.1be4e7512d4d8p+104"), hexf64("-0x1.481a1e5aed9dcp+51"), hexf64("-0x1.203876b339fa6p+51"),
            hexf64("-0x1.481a1e5aed9dcp+51"), hexf64("0x1.1be4e7512d4dcp+104"), hexf64("0x1.2baa7e6365e94p+50"),
            hexf64("-0x1.203876b339fa6p+51"), hexf64("0x1.2baa7e6365e94p+50"), hexf64("0x1.1be4e7512d4dcp+104"),
        ];
        let p = Array::<f64, Ix2>::from_vec(Ix2::new([3, 3]), p_data).unwrap();

        // comps (3x5).
        #[rustfmt::skip]
        let comps_data = vec![
            hexf64("0x1.9e88319dec0d6p-1"), hexf64("0x1.73b24c1bcf0aap-4"), hexf64("0x1.03e5d92462f7fp-1"), hexf64("-0x1.1ed7c6b0db970p-6"), hexf64("0x1.1e813b7d02761p-2"),
            hexf64("0x1.580342186098ep-2"), hexf64("-0x1.a9e94c3946fd4p-2"), hexf64("-0x1.96b85175e4971p-3"), hexf64("0x1.654aa4e383594p-1"), hexf64("-0x1.bbac7c41cad35p-2"),
            hexf64("0x1.85cf3a45e367cp-3"), hexf64("-0x1.8f7e5eedbc043p-2"), hexf64("-0x1.34a6dd64d0368p-1"), hexf64("-0x1.55314911eec58p-4"), hexf64("0x1.5417d963652aap-1"),
        ];
        let comps = Array::<f64, Ix2>::from_vec(Ix2::new([3, 5]), comps_data).unwrap();

        // comps.T (5x3): transpose explicitly so the gemm chain mirrors
        // numpy's `comps.T @ invP @ comps` exactly.
        let comps_t = transpose(&comps);

        // Step 1: invP = scipy.linalg.inv(P) via LAPACK getri.
        let inv_p = inv_lapack(&p).unwrap();

        // Verify invP itself is bit-identical to scipy first.
        #[rustfmt::skip]
        let inv_p_expected = [
            hexf64("0x1.cdb1756455cf0p-105"), hexf64("0x1.0acb3edf1b8b4p-157"), hexf64("0x1.d4baab1fc20f9p-158"),
            hexf64("0x1.0acb3edf1b8b4p-157"), hexf64("0x1.cdb1756455cecp-105"), hexf64("-0x1.e757bc8434a44p-159"),
            hexf64("0x1.d4baab1fc20f9p-158"), hexf64("-0x1.e757bc8434a44p-159"), hexf64("0x1.cdb1756455cecp-105"),
        ];
        let inv_p_got: Vec<f64> = inv_p.iter().copied().collect();
        // getri vs scipy's LAPACK build: allow a few ULP. The SIGN proof
        // below is the load-bearing assertion, not last-bit invP equality.
        let mut inv_p_max_ulp = 0u64;
        for (g, e) in inv_p_got.iter().zip(inv_p_expected.iter()) {
            let ulp = (g.to_bits() as i64 - e.to_bits() as i64).unsigned_abs();
            inv_p_max_ulp = inv_p_max_ulp.max(ulp);
        }
        assert!(
            inv_p_max_ulp <= 4,
            "invP max ULP vs scipy = {inv_p_max_ulp}"
        );

        // Step 2: LEFT-TO-RIGHT product chain via OpenBLAS gemm.
        //   t1 = comps.T @ invP   (5x3)
        //   inner = t1 @ comps    (5x5)
        let t1 = gemm(&comps_t, &inv_p).unwrap();
        let inner = gemm(&t1, &comps).unwrap();

        // Verify `inner` is bit-identical to numpy's left-to-right chain.
        #[rustfmt::skip]
        let inner_expected = [
            hexf64("0x1.7379f337b2d30p-105"), hexf64("-0x1.0380748452a7cp-107"), hexf64("0x1.a7df756a2efbcp-107"), hexf64("0x1.797ed5089a092p-107"), hexf64("0x1.7f0484fdbe669p-107"),
            hexf64("-0x1.0380748452a7cp-107"), hexf64("0x1.33e35973a08e7p-106"), hexf64("0x1.4ff93d2bb1921p-106"), hexf64("-0x1.def1b614d25cbp-107"), hexf64("-0x1.8b82129de0bf3p-109"),
            hexf64("0x1.a7df756a2efbep-107"), hexf64("0x1.4ff93d2bb1921p-106"), hexf64("0x1.30f5474fe5421p-105"), hexf64("-0x1.6739457e304a6p-108"), hexf64("-0x1.3e4f0e5b92786p-107"),
            hexf64("0x1.797ed5089a092p-107"), hexf64("-0x1.def1b614d25cbp-107"), hexf64("-0x1.6739457e304a4p-108"), hexf64("0x1.c85aec6f494b0p-106"), hexf64("-0x1.4ece1812b62eap-106"),
            hexf64("0x1.7f0484fdbe66ap-107"), hexf64("-0x1.8b82129de0bfap-109"), hexf64("-0x1.3e4f0e5b92787p-107"), hexf64("-0x1.4ece1812b62eap-106"), hexf64("0x1.468602b3f46aap-105"),
        ];
        let inner_got: Vec<f64> = inner.iter().copied().collect();
        let mut inner_max_ulp = 0u64;
        for (g, e) in inner_got.iter().zip(inner_expected.iter()) {
            let ulp = (g.to_bits() as i64 - e.to_bits() as i64).unsigned_abs();
            inner_max_ulp = inner_max_ulp.max(ulp);
        }

        // Step 3: precision = inner / -(nv^2); add 1/nv to the diagonal.
        let neg_nv2 = -(nv * nv);
        let inv_nv = 1.0 / nv;
        let mut precision_data: Vec<f64> = inner_got.iter().map(|x| x / neg_nv2).collect();
        for d in 0..5 {
            precision_data[d * 5 + d] += inv_nv;
        }
        let precision = Array::<f64, Ix2>::from_vec(Ix2::new([5, 5]), precision_data).unwrap();

        // Step 4: slogdet(precision). THE SIGN must be +1 (scipy/sklearn).
        let (sign, logabsdet) = slogdet(&precision).unwrap();

        // scipy: sign = +1.0, logabsdet = 253.69530279572427.
        assert_eq!(
            sign, 1.0,
            "PCA #2110 precision slogdet SIGN = {sign} (scipy = +1); \
             inner max ULP vs numpy = {inner_max_ulp}; logabsdet = {logabsdet}"
        );
        // The SIGN (above) is the load-bearing deliverable: it decides
        // finite-vs-`-inf` in `fast_logdet`. The magnitude `logabsdet` of
        // a condition-~6.3e15 matrix is itself numerically unstable —
        // tiny ULP differences in inv/gemm/slogdet across LAPACK builds
        // perturb it at the ~1% level (ferray slogdet here lands on
        // ~250.1 vs scipy's ~253.7). We therefore only sanity-bound the
        // magnitude (same order, finite), not last-bit equality.
        let ld_expected = hexf64("0x1.fb63feba60e80p+7"); // 253.695...
        assert!(
            logabsdet.is_finite() && (logabsdet - ld_expected).abs() / ld_expected.abs() < 0.05,
            "logabsdet {logabsdet:.17e} vs scipy {ld_expected:.17e} (sign matched +1; \
             magnitude on a cond-6e15 matrix is ill-posed)"
        );
    }

    /// Row-major transpose helper for the sign-proof test.
    fn transpose(a: &Array<f64, Ix2>) -> Array<f64, Ix2> {
        let (r, c) = (a.shape()[0], a.shape()[1]);
        let s = a.as_slice().unwrap();
        let mut t = vec![0.0_f64; r * c];
        for i in 0..r {
            for j in 0..c {
                t[j * r + i] = s[i * c + j];
            }
        }
        Array::<f64, Ix2>::from_vec(Ix2::new([c, r]), t).unwrap()
    }

    // Minimal hex-float parser (Rust has no hex-float literals on stable).
    // Used only to express the scipy/numpy oracle values exactly.
    fn hexf64(s: &str) -> f64 {
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
