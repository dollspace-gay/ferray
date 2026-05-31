// Faer-backed GEMM fallback for architectures without a hand-tuned kernel.
//
// Rationale: ferray's API contract is universal numpy parity. x86_64 and
// aarch64 ship hand-tuned kernels for the perf bar. RISC-V, WASM, big-
// endian PowerPC, etc. would otherwise have no GEMM at all. faer's
// portable Rust GEMM is already in the workspace — we route all dtypes
// through it as a correctness backstop.
//
// Performance: faer is 4-9× slower than hand-tuned BLAS at N∈[128, 512]
// per the issue tracker — but the perf goal here is "correct, not fast".
// A native NEON-class kernel for any future arch is a follow-up.
//
// Integer dtypes (i8 / i16) are handled with a plain scalar triple-loop
// since faer is float-only. The faer fallback is correctness-only and
// these matrices on non-major arches are rare enough that perf doesn't
// dominate.

use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, Mat, Par};

/// f64 GEMM via faer.
pub unsafe fn gemm_fallback_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c.add(i * ldc + j) = 0.0;
                    }
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c.add(i * ldc + j) *= beta;
                    }
                }
            }
        }
        return;
    }
    // Build column-major faer Mats by copying the row-major ferray buffers.
    let lhs = Mat::<f64>::from_fn(m, k, |i, j| unsafe { *a.add(i * lda + j) });
    let rhs = Mat::<f64>::from_fn(k, n, |i, j| unsafe { *b.add(i * ldb + j) });

    // Compute alpha * A @ B + beta * C. faer's `matmul` does dst = beta_acc + alpha * lhs * rhs.
    let mut dst = Mat::<f64>::from_fn(m, n, |i, j| {
        if beta == 0.0 {
            0.0
        } else {
            beta * unsafe { *c.add(i * ldc + j) }
        }
    });
    faer_matmul(&mut dst, Accum::Add, &lhs, &rhs, alpha, Par::Seq);
    // Write back row-major.
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *c.add(i * ldc + j) = dst[(i, j)];
            }
        }
    }
}

/// f32 GEMM via faer.
pub unsafe fn gemm_fallback_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c.add(i * ldc + j) = 0.0;
                    }
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *c.add(i * ldc + j) *= beta;
                    }
                }
            }
        }
        return;
    }
    let lhs = Mat::<f32>::from_fn(m, k, |i, j| unsafe { *a.add(i * lda + j) });
    let rhs = Mat::<f32>::from_fn(k, n, |i, j| unsafe { *b.add(i * ldb + j) });
    let mut dst = Mat::<f32>::from_fn(m, n, |i, j| {
        if beta == 0.0 {
            0.0
        } else {
            beta * unsafe { *c.add(i * ldc + j) }
        }
    });
    faer_matmul(&mut dst, Accum::Add, &lhs, &rhs, alpha, Par::Seq);
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *c.add(i * ldc + j) = dst[(i, j)];
            }
        }
    }
}

/// Complex<f64> GEMM via scalar triple-loop fallback (faer's matmul
/// supports complex but the API surface to plumb it through is
/// nontrivial; the scalar loop is correctness-only and adequate for
/// rare non-major-arch use).
pub unsafe fn gemm_fallback_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc_re = 0.0_f64;
            let mut acc_im = 0.0_f64;
            for p in 0..k {
                let ar = unsafe { *a.add(i * lda * 2 + p * 2) };
                let ai = unsafe { *a.add(i * lda * 2 + p * 2 + 1) };
                let br = unsafe { *b.add(p * ldb * 2 + j * 2) };
                let bi = unsafe { *b.add(p * ldb * 2 + j * 2 + 1) };
                acc_re += ar * br - ai * bi;
                acc_im += ar * bi + ai * br;
            }
            // alpha * acc
            let new_re = alpha_re * acc_re - alpha_im * acc_im;
            let new_im = alpha_re * acc_im + alpha_im * acc_re;
            let p_re = unsafe { c.add(i * ldc * 2 + j * 2) };
            let p_im = unsafe { p_re.add(1) };
            if beta_re == 0.0 && beta_im == 0.0 {
                unsafe {
                    *p_re = new_re;
                    *p_im = new_im;
                }
            } else {
                let cur_re = unsafe { *p_re };
                let cur_im = unsafe { *p_im };
                let scaled_re = beta_re * cur_re - beta_im * cur_im;
                let scaled_im = beta_re * cur_im + beta_im * cur_re;
                unsafe {
                    *p_re = scaled_re + new_re;
                    *p_im = scaled_im + new_im;
                }
            }
        }
    }
}

/// Complex<f32> GEMM via scalar triple-loop fallback.
pub unsafe fn gemm_fallback_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc_re = 0.0_f32;
            let mut acc_im = 0.0_f32;
            for p in 0..k {
                let ar = unsafe { *a.add(i * lda * 2 + p * 2) };
                let ai = unsafe { *a.add(i * lda * 2 + p * 2 + 1) };
                let br = unsafe { *b.add(p * ldb * 2 + j * 2) };
                let bi = unsafe { *b.add(p * ldb * 2 + j * 2 + 1) };
                acc_re += ar * br - ai * bi;
                acc_im += ar * bi + ai * br;
            }
            let new_re = alpha_re * acc_re - alpha_im * acc_im;
            let new_im = alpha_re * acc_im + alpha_im * acc_re;
            let p_re = unsafe { c.add(i * ldc * 2 + j * 2) };
            let p_im = unsafe { p_re.add(1) };
            if beta_re == 0.0 && beta_im == 0.0 {
                unsafe {
                    *p_re = new_re;
                    *p_im = new_im;
                }
            } else {
                let cur_re = unsafe { *p_re };
                let cur_im = unsafe { *p_im };
                let scaled_re = beta_re * cur_re - beta_im * cur_im;
                let scaled_im = beta_re * cur_im + beta_im * cur_re;
                unsafe {
                    *p_re = scaled_re + new_re;
                    *p_im = scaled_im + new_im;
                }
            }
        }
    }
}

/// i16 × i16 → i32 GEMM via scalar triple-loop (integer types are not
/// in faer's matmul codepath; scalar is correctness-only).
pub unsafe fn gemm_fallback_i16(
    m: usize,
    n: usize,
    k: usize,
    a: *const i16,
    lda: usize,
    b: *const i16,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                let av = unsafe { *a.add(i * lda + p) } as i32;
                let bv = unsafe { *b.add(p * ldb + j) } as i32;
                acc = acc.wrapping_add(av.wrapping_mul(bv));
            }
            let cell = unsafe { c.add(i * ldc + j) };
            unsafe {
                *cell = if accumulate {
                    (*cell).wrapping_add(acc)
                } else {
                    acc
                };
            }
        }
    }
}

/// u8 × i8 → i32 GEMM via scalar triple-loop.
pub unsafe fn gemm_fallback_i8(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                let av = unsafe { *a.add(i * lda + p) } as i32;
                let bv = unsafe { *b.add(p * ldb + j) } as i32;
                acc = acc.wrapping_add(av.wrapping_mul(bv));
            }
            let cell = unsafe { c.add(i * ldc + j) };
            unsafe {
                *cell = if accumulate {
                    (*cell).wrapping_add(acc)
                } else {
                    acc
                };
            }
        }
    }
}

/// i8 × i8 → i32 GEMM via scalar triple-loop.
pub unsafe fn gemm_fallback_i8s(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                let av = unsafe { *a.add(i * lda + p) } as i32;
                let bv = unsafe { *b.add(p * ldb + j) } as i32;
                acc = acc.wrapping_add(av.wrapping_mul(bv));
            }
            let cell = unsafe { c.add(i * ldc + j) };
            unsafe {
                *cell = if accumulate {
                    (*cell).wrapping_add(acc)
                } else {
                    acc
                };
            }
        }
    }
}
