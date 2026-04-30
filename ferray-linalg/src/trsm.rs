//! Native triangular solve (TRSM) primitives.
//!
//! Solves `op(A) * X = alpha * B` in-place where A is triangular. The
//! block-recursive algorithm splits the problem into a small unblocked
//! solve plus a trailing GEMM update, so the bulk of the FLOPs go
//! through the hand-tuned [`crate::gemm`] kernels and TRSM inherits
//! their AVX2/AVX-512 performance.
//!
//! Public surface — the 4 commonly-used left-side variants (the right-
//! side variants and the explicit-transpose variants are uncommon
//! enough that callers can transpose the inputs themselves):
//!
//! - [`trsm_left_lower_f64`] / [`trsm_left_lower_f32`]
//! - [`trsm_left_upper_f64`] / [`trsm_left_upper_f32`]
//!
//! Layout convention is row-major (matches the rest of ferray-linalg);
//! `lda` and `ldb` are row strides in elements.
//!
//! Diagonal handling: `unit_diag = true` skips the `1 / a[i,i]` divide,
//! used by LU's `L` factor (which has implicit unit diagonal).

#![allow(clippy::too_many_arguments)]

/// Block size for the recursive split. Below this row count the
/// unblocked scalar solve is used; above it the algorithm splits and
/// recurses, with the trailing `B -= L21 X1` (or `U12 X2`) update done
/// row-major scalar. The trailing update doesn't go through the hand-
/// tuned GEMM kernels because those expect tightly-packed contiguous
/// inputs (lda = k) — the strided sub-blocks here would require a
/// strided-input GEMM wrapper that doesn't exist yet. NB=32 keeps the
/// unblocked solve dominant for medium m and lets the cache absorb the
/// scalar trailing updates for large m.
const NB: usize = 32;

/// Solves `L * X = alpha * B` in-place, lower-triangular. Result
/// overwrites `B`. `m` is the rows of B (= dim of L), `n` is the cols
/// of B. `lda` ≥ `m`, `ldb` ≥ `n`. `unit_diag = true` ignores the
/// stored diagonal of A and treats it as 1.0 (LU's L factor convention).
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * lda` f64s representing an m×m matrix
/// - `b` points to at least `m * ldb` f64s, writable
/// - The buffers don't overlap
pub unsafe fn trsm_left_lower_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
    unit_diag: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha != 1.0 {
        unsafe { scale_f64(m, n, alpha, b, ldb) };
    }
    unsafe { trsm_lln_recurse_f64(m, n, a, lda, b, ldb, unit_diag) };
}

/// Solves `U * X = alpha * B` in-place, upper-triangular.
///
/// # Safety
///
/// Same as [`trsm_left_lower_f64`].
pub unsafe fn trsm_left_upper_f64(
    m: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
    unit_diag: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha != 1.0 {
        unsafe { scale_f64(m, n, alpha, b, ldb) };
    }
    unsafe { trsm_lun_recurse_f64(m, n, a, lda, b, ldb, unit_diag) };
}

/// f32 sibling of [`trsm_left_lower_f64`].
///
/// # Safety
///
/// Same as [`trsm_left_lower_f64`] but for f32 buffers.
pub unsafe fn trsm_left_lower_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *mut f32,
    ldb: usize,
    unit_diag: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha != 1.0 {
        unsafe { scale_f32(m, n, alpha, b, ldb) };
    }
    unsafe { trsm_lln_recurse_f32(m, n, a, lda, b, ldb, unit_diag) };
}

/// f32 sibling of [`trsm_left_upper_f64`].
///
/// # Safety
///
/// Same as [`trsm_left_upper_f64`] but for f32 buffers.
pub unsafe fn trsm_left_upper_f32(
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *mut f32,
    ldb: usize,
    unit_diag: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha != 1.0 {
        unsafe { scale_f32(m, n, alpha, b, ldb) };
    }
    unsafe { trsm_lun_recurse_f32(m, n, a, lda, b, ldb, unit_diag) };
}

unsafe fn scale_f64(m: usize, n: usize, alpha: f64, b: *mut f64, ldb: usize) {
    for i in 0..m {
        let row = unsafe { b.add(i * ldb) };
        for j in 0..n {
            unsafe { *row.add(j) *= alpha };
        }
    }
}

unsafe fn scale_f32(m: usize, n: usize, alpha: f32, b: *mut f32, ldb: usize) {
    for i in 0..m {
        let row = unsafe { b.add(i * ldb) };
        for j in 0..n {
            unsafe { *row.add(j) *= alpha };
        }
    }
}

// --- Block-recursive solve, lower triangular, no transpose ---
//
// Partition L (m×m), X = B (m×n) as
//   L = [[L11, 0  ],   X = [[X1],   B = [[B1],
//        [L21, L22]]       [X2]]       [B2]]
// where L11 is m1×m1 (m1 = m/2 rounded to NB).
// Steps:
//   1) solve L11 X1 = B1            (recurse on top half)
//   2) B2 -= L21 X1                  (GEMM)
//   3) solve L22 X2 = B2            (recurse on bottom half)
unsafe fn trsm_lln_recurse_f64(
    m: usize,
    n: usize,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
    unit_diag: bool,
) {
    if m <= NB {
        unsafe { trsm_lln_unblocked_f64(m, n, a, lda, b, ldb, unit_diag) };
        return;
    }
    let m1 = (m / 2).next_multiple_of(NB).min(m - NB);
    let m2 = m - m1;

    // Step 1: L11 X1 = B1
    unsafe {
        trsm_lln_recurse_f64(m1, n, a, lda, b, ldb, unit_diag);
    }

    // Step 2: B2 -= L21 @ X1, scalar trailing update (see NB doc above
    // for why we don't dispatch to gemm_f64 here).
    unsafe { gemm_update_lln_f64(m2, n, m1, a.add(m1 * lda), lda, b, ldb, b.add(m1 * ldb)) };

    // Step 3: L22 X2 = B2.
    unsafe {
        trsm_lln_recurse_f64(
            m2,
            n,
            a.add(m1 * lda + m1),
            lda,
            b.add(m1 * ldb),
            ldb,
            unit_diag,
        );
    }
}

/// B2 -= L21 @ X1, scalar fallback. L21 is m2×k, X1 is k×n, B2 is m2×n.
unsafe fn gemm_update_lln_f64(
    m2: usize,
    n: usize,
    k: usize,
    l21: *const f64,
    lda: usize,
    x1: *const f64,
    ldb: usize,
    b2: *mut f64,
) {
    for i in 0..m2 {
        for p in 0..k {
            let l_ip = unsafe { *l21.add(i * lda + p) };
            for j in 0..n {
                unsafe {
                    *b2.add(i * ldb + j) -= l_ip * *x1.add(p * ldb + j);
                }
            }
        }
    }
}

unsafe fn trsm_lln_unblocked_f64(
    m: usize,
    n: usize,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
    unit_diag: bool,
) {
    for i in 0..m {
        // Subtract earlier rows' contributions: b[i,:] -= sum_{p<i} a[i,p] * b[p,:]
        for p in 0..i {
            let a_ip = unsafe { *a.add(i * lda + p) };
            for j in 0..n {
                unsafe {
                    *b.add(i * ldb + j) -= a_ip * *b.add(p * ldb + j);
                }
            }
        }
        if !unit_diag {
            let inv_a_ii = 1.0 / unsafe { *a.add(i * lda + i) };
            for j in 0..n {
                unsafe { *b.add(i * ldb + j) *= inv_a_ii };
            }
        }
    }
}

// --- Block-recursive solve, upper triangular, no transpose ---
//
//   U = [[U11, U12],   X = [[X1],   B = [[B1],
//        [0,   U22]]       [X2]]       [B2]]
// where U22 is m2×m2.
// Steps:
//   1) solve U22 X2 = B2            (recurse on bottom half, smaller)
//   2) B1 -= U12 X2                  (GEMM)
//   3) solve U11 X1 = B1            (recurse on top half)
unsafe fn trsm_lun_recurse_f64(
    m: usize,
    n: usize,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
    unit_diag: bool,
) {
    if m <= NB {
        unsafe { trsm_lun_unblocked_f64(m, n, a, lda, b, ldb, unit_diag) };
        return;
    }
    let m1 = (m / 2).next_multiple_of(NB).min(m - NB);
    let m2 = m - m1;

    // Step 1: U22 X2 = B2. U22 at a[m1*lda+m1], B2 at b[m1*ldb].
    unsafe {
        trsm_lun_recurse_f64(
            m2,
            n,
            a.add(m1 * lda + m1),
            lda,
            b.add(m1 * ldb),
            ldb,
            unit_diag,
        );
    }

    // Step 2: B1 -= U12 @ X2. U12 at a[m1], X2 at b[m1*ldb], B1 at b.
    unsafe {
        gemm_update_lln_f64(m1, n, m2, a.add(m1), lda, b.add(m1 * ldb), ldb, b);
    }

    // Step 3: U11 X1 = B1.
    unsafe {
        trsm_lun_recurse_f64(m1, n, a, lda, b, ldb, unit_diag);
    }
}

unsafe fn trsm_lun_unblocked_f64(
    m: usize,
    n: usize,
    a: *const f64,
    lda: usize,
    b: *mut f64,
    ldb: usize,
    unit_diag: bool,
) {
    // Back-substitute: row i = m-1, m-2, ..., 0.
    for i in (0..m).rev() {
        for p in (i + 1)..m {
            let a_ip = unsafe { *a.add(i * lda + p) };
            for j in 0..n {
                unsafe {
                    *b.add(i * ldb + j) -= a_ip * *b.add(p * ldb + j);
                }
            }
        }
        if !unit_diag {
            let inv_a_ii = 1.0 / unsafe { *a.add(i * lda + i) };
            for j in 0..n {
                unsafe { *b.add(i * ldb + j) *= inv_a_ii };
            }
        }
    }
}

// --- f32 mirrors ---

unsafe fn trsm_lln_recurse_f32(
    m: usize,
    n: usize,
    a: *const f32,
    lda: usize,
    b: *mut f32,
    ldb: usize,
    unit_diag: bool,
) {
    if m <= NB {
        unsafe { trsm_lln_unblocked_f32(m, n, a, lda, b, ldb, unit_diag) };
        return;
    }
    let m1 = (m / 2).next_multiple_of(NB).min(m - NB);
    let m2 = m - m1;
    unsafe { trsm_lln_recurse_f32(m1, n, a, lda, b, ldb, unit_diag) };
    unsafe {
        gemm_update_lln_f32(m2, n, m1, a.add(m1 * lda), lda, b, ldb, b.add(m1 * ldb));
    }
    unsafe {
        trsm_lln_recurse_f32(
            m2,
            n,
            a.add(m1 * lda + m1),
            lda,
            b.add(m1 * ldb),
            ldb,
            unit_diag,
        );
    }
}

unsafe fn gemm_update_lln_f32(
    m2: usize,
    n: usize,
    k: usize,
    l21: *const f32,
    lda: usize,
    x1: *const f32,
    ldb: usize,
    b2: *mut f32,
) {
    for i in 0..m2 {
        for p in 0..k {
            let l_ip = unsafe { *l21.add(i * lda + p) };
            for j in 0..n {
                unsafe {
                    *b2.add(i * ldb + j) -= l_ip * *x1.add(p * ldb + j);
                }
            }
        }
    }
}

unsafe fn trsm_lln_unblocked_f32(
    m: usize,
    n: usize,
    a: *const f32,
    lda: usize,
    b: *mut f32,
    ldb: usize,
    unit_diag: bool,
) {
    for i in 0..m {
        for p in 0..i {
            let a_ip = unsafe { *a.add(i * lda + p) };
            for j in 0..n {
                unsafe {
                    *b.add(i * ldb + j) -= a_ip * *b.add(p * ldb + j);
                }
            }
        }
        if !unit_diag {
            let inv_a_ii = 1.0 / unsafe { *a.add(i * lda + i) };
            for j in 0..n {
                unsafe { *b.add(i * ldb + j) *= inv_a_ii };
            }
        }
    }
}

unsafe fn trsm_lun_recurse_f32(
    m: usize,
    n: usize,
    a: *const f32,
    lda: usize,
    b: *mut f32,
    ldb: usize,
    unit_diag: bool,
) {
    if m <= NB {
        unsafe { trsm_lun_unblocked_f32(m, n, a, lda, b, ldb, unit_diag) };
        return;
    }
    let m1 = (m / 2).next_multiple_of(NB).min(m - NB);
    let m2 = m - m1;
    unsafe {
        trsm_lun_recurse_f32(
            m2,
            n,
            a.add(m1 * lda + m1),
            lda,
            b.add(m1 * ldb),
            ldb,
            unit_diag,
        );
    }
    unsafe {
        gemm_update_lln_f32(m1, n, m2, a.add(m1), lda, b.add(m1 * ldb), ldb, b);
    }
    unsafe {
        trsm_lun_recurse_f32(m1, n, a, lda, b, ldb, unit_diag);
    }
}

unsafe fn trsm_lun_unblocked_f32(
    m: usize,
    n: usize,
    a: *const f32,
    lda: usize,
    b: *mut f32,
    ldb: usize,
    unit_diag: bool,
) {
    for i in (0..m).rev() {
        for p in (i + 1)..m {
            let a_ip = unsafe { *a.add(i * lda + p) };
            for j in 0..n {
                unsafe {
                    *b.add(i * ldb + j) -= a_ip * *b.add(p * ldb + j);
                }
            }
        }
        if !unit_diag {
            let inv_a_ii = 1.0 / unsafe { *a.add(i * lda + i) };
            for j in 0..n {
                unsafe { *b.add(i * ldb + j) *= inv_a_ii };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_matmul_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    fn make_lower_triangular(m: usize) -> Vec<f64> {
        let mut a = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in 0..=i {
                if i == j {
                    a[i * m + j] = 2.0 + (i as f64) * 0.5;
                } else {
                    a[i * m + j] = 0.1 * ((i + j + 1) as f64);
                }
            }
        }
        a
    }

    fn make_upper_triangular(m: usize) -> Vec<f64> {
        let mut a = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in i..m {
                if i == j {
                    a[i * m + j] = 2.0 + (i as f64) * 0.5;
                } else {
                    a[i * m + j] = 0.1 * ((i + j + 1) as f64);
                }
            }
        }
        a
    }

    #[test]
    fn trsm_lln_f64_small() {
        let m = 5;
        let n = 3;
        let a = make_lower_triangular(m);
        let b_orig: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.7 + 1.0).collect();
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_lower_f64(m, n, 1.0, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        // Verify A @ X == B (within tolerance).
        let mut check = vec![0.0_f64; m * n];
        naive_matmul_f64(&a, &x, &mut check, m, n, m);
        for i in 0..m * n {
            assert!(
                (check[i] - b_orig[i]).abs() < 1e-10,
                "lln f64 mismatch at {i}: got {} expected {}",
                check[i],
                b_orig[i]
            );
        }
    }

    #[test]
    fn trsm_lln_f64_large_blocked() {
        let m = 80; // > NB to exercise the recursion + trailing update.
        let n = 7;
        let a = make_lower_triangular(m);
        let b_orig: Vec<f64> = (0..m * n)
            .map(|i| ((i as f64) * 0.13).sin() + 1.0)
            .collect();
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_lower_f64(m, n, 1.0, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        let mut check = vec![0.0_f64; m * n];
        naive_matmul_f64(&a, &x, &mut check, m, n, m);
        for i in 0..m * n {
            assert!(
                (check[i] - b_orig[i]).abs() < 1e-9,
                "lln-blocked f64 mismatch at {i}: got {} expected {}",
                check[i],
                b_orig[i]
            );
        }
    }

    #[test]
    fn trsm_lln_f64_unit_diag() {
        let m = 6;
        let n = 4;
        let mut a = make_lower_triangular(m);
        // Force unit diagonal in storage so non-unit produces same result
        // as unit_diag=true.
        for i in 0..m {
            a[i * m + i] = 1.0;
        }
        let b_orig: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.3 + 0.5).collect();

        let mut x_unit = b_orig.clone();
        let mut x_explicit = b_orig.clone();
        unsafe {
            trsm_left_lower_f64(m, n, 1.0, a.as_ptr(), m, x_unit.as_mut_ptr(), n, true);
            trsm_left_lower_f64(m, n, 1.0, a.as_ptr(), m, x_explicit.as_mut_ptr(), n, false);
        }
        for i in 0..m * n {
            assert!(
                (x_unit[i] - x_explicit[i]).abs() < 1e-12,
                "unit-vs-explicit mismatch at {i}",
            );
        }
    }

    #[test]
    fn trsm_lun_f64_small() {
        let m = 5;
        let n = 3;
        let a = make_upper_triangular(m);
        let b_orig: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.7 + 1.0).collect();
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_upper_f64(m, n, 1.0, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        let mut check = vec![0.0_f64; m * n];
        naive_matmul_f64(&a, &x, &mut check, m, n, m);
        for i in 0..m * n {
            assert!(
                (check[i] - b_orig[i]).abs() < 1e-10,
                "lun f64 mismatch at {i}: got {} expected {}",
                check[i],
                b_orig[i]
            );
        }
    }

    #[test]
    fn trsm_lun_f64_large_blocked() {
        let m = 100;
        let n = 5;
        let a = make_upper_triangular(m);
        let b_orig: Vec<f64> = (0..m * n).map(|i| ((i as f64) * 0.17).cos()).collect();
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_upper_f64(m, n, 1.0, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        let mut check = vec![0.0_f64; m * n];
        naive_matmul_f64(&a, &x, &mut check, m, n, m);
        for i in 0..m * n {
            assert!(
                (check[i] - b_orig[i]).abs() < 1e-9,
                "lun-blocked f64 mismatch at {i}: got {} expected {}",
                check[i],
                b_orig[i]
            );
        }
    }

    #[test]
    fn trsm_alpha_scales_b() {
        let m = 4;
        let n = 2;
        let a = make_lower_triangular(m);
        let b_orig: Vec<f64> = (0..m * n).map(|i| (i as f64) + 1.0).collect();
        let alpha = 3.0;
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_lower_f64(m, n, alpha, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        let mut check = vec![0.0_f64; m * n];
        naive_matmul_f64(&a, &x, &mut check, m, n, m);
        for i in 0..m * n {
            let expected = alpha * b_orig[i];
            assert!(
                (check[i] - expected).abs() < 1e-10,
                "alpha-scaled lln mismatch at {i}: got {} expected {}",
                check[i],
                expected,
            );
        }
    }

    #[test]
    fn trsm_lln_f32_blocked() {
        let m = 65;
        let n = 4;
        let a64 = make_lower_triangular(m);
        let a: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
        let b_orig: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.13 + 0.5).collect();
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_lower_f32(m, n, 1.0, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        // Reconstruct A @ X and compare to B.
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for p in 0..m {
                    s += a[i * m + p] * x[p * n + j];
                }
                assert!(
                    (s - b_orig[i * n + j]).abs() < 1e-3,
                    "lln f32 mismatch at ({i},{j}): got {s} expected {}",
                    b_orig[i * n + j]
                );
            }
        }
    }

    #[test]
    fn trsm_lun_f32_blocked() {
        let m = 70;
        let n = 3;
        let a64 = make_upper_triangular(m);
        let a: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
        let b_orig: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.07 - 0.5).collect();
        let mut x = b_orig.clone();
        unsafe {
            trsm_left_upper_f32(m, n, 1.0, a.as_ptr(), m, x.as_mut_ptr(), n, false);
        }
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for p in 0..m {
                    s += a[i * m + p] * x[p * n + j];
                }
                assert!(
                    (s - b_orig[i * n + j]).abs() < 1e-3,
                    "lun f32 mismatch at ({i},{j}): got {s} expected {}",
                    b_orig[i * n + j]
                );
            }
        }
    }
}
