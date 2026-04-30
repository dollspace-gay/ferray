//! OpenBLAS backend — thin wrappers over `cblas_{s,d,c,z}gemm` that
//! match the signatures of the hand-tuned `gemm_{f32,f64,c32,c64}`
//! entries in [`super`]. Cfg-gated on the `openblas` feature.
//!
//! Linking expectation: the `openblas-src` crate is configured to use
//! the system OpenBLAS (`features = ["system", "cblas"]`). Users who
//! prefer a vendored static build can override at the workspace level
//! with `cargo build --features openblas openblas-src/static`.

use cblas_sys::{
    CBLAS_LAYOUT, CBLAS_TRANSPOSE, c_double_complex, c_float_complex, cblas_cgemm, cblas_dgemm,
    cblas_sgemm, cblas_zgemm,
};

// Pull `openblas_src` into the link graph so the linker resolves the
// cblas_* symbols. The crate's only purpose is to emit the link
// directive — the symbols themselves are declared by `cblas_sys`.
#[allow(unused_imports)]
use openblas_src as _;

/// `cblas_dgemm` wrapper. Computes `C = alpha * A @ B + beta * C` for
/// row-major f64 matrices.
///
/// # Safety
///
/// Caller must ensure the buffer-length contract matches the (m, n, k)
/// dims — same contract as the hand-tuned `gemm_f64`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_f64_openblas(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
) {
    unsafe {
        cblas_dgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            k as i32,
            b,
            n as i32,
            beta,
            c,
            n as i32,
        );
    }
}

/// `cblas_sgemm` wrapper. Computes `C = alpha * A @ B + beta * C` for
/// row-major f32 matrices.
///
/// # Safety
///
/// Caller must ensure the buffer-length contract matches the (m, n, k)
/// dims — same contract as the hand-tuned `gemm_f32`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_f32_openblas(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            k as i32,
            b,
            n as i32,
            beta,
            c,
            n as i32,
        );
    }
}

/// `cblas_zgemm` wrapper. Computes `C = alpha * A @ B + beta * C` for
/// row-major Complex<f64> matrices. `a`, `b`, `c` are passed as
/// `*const f64` / `*mut f64` where each Complex<f64> element is two
/// contiguous f64s `[re, im]`.
///
/// # Safety
///
/// Caller must ensure the buffer-length contract matches the (m, n, k)
/// dims — same contract as the hand-tuned `gemm_c64`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_c64_openblas(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
) {
    let alpha: c_double_complex = [alpha_re, alpha_im];
    let beta: c_double_complex = [beta_re, beta_im];
    unsafe {
        cblas_zgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            a as *const c_double_complex,
            k as i32,
            b as *const c_double_complex,
            n as i32,
            &beta,
            c as *mut c_double_complex,
            n as i32,
        );
    }
}

/// `cblas_cgemm` wrapper. Computes `C = alpha * A @ B + beta * C` for
/// row-major Complex<f32> matrices.
///
/// # Safety
///
/// Caller must ensure the buffer-length contract matches the (m, n, k)
/// dims — same contract as the hand-tuned `gemm_c32`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_c32_openblas(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
) {
    let alpha: c_float_complex = [alpha_re, alpha_im];
    let beta: c_float_complex = [beta_re, beta_im];
    unsafe {
        cblas_cgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            &alpha,
            a as *const c_float_complex,
            k as i32,
            b as *const c_float_complex,
            n as i32,
            &beta,
            c as *mut c_float_complex,
            n as i32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dgemm(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f64;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    #[test]
    fn openblas_dgemm_matches_naive() {
        let m = 8;
        let n = 12;
        let k = 16;
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.13).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.17 - 0.5).collect();
        let mut c_ours = vec![0.0_f64; m * n];
        let mut c_ref = vec![0.0_f64; m * n];
        unsafe {
            gemm_f64_openblas(
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
            );
        }
        naive_dgemm(m, n, k, &a, &b, &mut c_ref);
        for i in 0..m * n {
            assert!(
                (c_ours[i] - c_ref[i]).abs() < 1e-10,
                "openblas dgemm mismatch at {i}: {} vs {}",
                c_ours[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn openblas_sgemm_matches_naive() {
        let m = 7;
        let n = 9;
        let k = 11;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.13).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.17 - 0.5).collect();
        let mut c_ours = vec![0.0_f32; m * n];
        let mut c_ref = vec![0.0_f32; m * n];
        unsafe {
            gemm_f32_openblas(
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
            );
        }
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = s;
            }
        }
        for i in 0..m * n {
            assert!(
                (c_ours[i] - c_ref[i]).abs() < 1e-4,
                "openblas sgemm mismatch at {i}",
            );
        }
    }
}
