// AArch64 NEON Goto-style 5-loop GEMM drivers.
//
// Loop nesting (outer to inner): jc / pc / ic / jr / ir.
// Block sizes (matching the AVX2 driver — these target the typical
// Cortex-A78 / Neoverse-N1 L2 of 1-2 MB per core):
//   MC = 64
//   KC = 256
//   NC = 512
//
// One sequential driver per dtype. Multi-threading via rayon is a
// follow-up — at the dispatch-1 perf bar (≥2× scalar reference), the
// kernel itself is the dominant bottleneck.

use super::kernel_neon_complex_f32::{
    MR_NEON_C32, NR_NEON_C32, kernel_4x4_neon_c32, kernel_edge_neon_c32,
};
use super::kernel_neon_complex_f64::{
    MR_NEON_C64, NR_NEON_C64, kernel_4x2_neon_c64, kernel_edge_neon_c64,
};
use super::kernel_neon_f32::{MR_NEON_F32, NR_NEON_F32, kernel_4x16_neon_f32, kernel_edge_neon_f32};
use super::kernel_neon_f64::{MR_NEON_F64, NR_NEON_F64, kernel_4x8_neon_f64, kernel_edge_neon_f64};
use super::kernel_neon_i8::{MR_NEON_I8, NR_NEON_I8, kernel_4x8_neon_i8, kernel_edge_neon_i8};
use super::kernel_neon_i8s::{
    MR_NEON_I8S, NR_NEON_I8S, kernel_4x8_neon_i8_signed, kernel_edge_neon_i8_signed,
};
use super::kernel_neon_i16::{MR_NEON_I16, NR_NEON_I16, kernel_4x8_neon_i16, kernel_edge_neon_i16};
use super::pack_neon::{
    pack_a_neon_c32, pack_a_neon_c64, pack_a_neon_f32, pack_a_neon_f64, pack_a_neon_i8s,
    pack_a_neon_i16, pack_a_neon_u8, pack_b_neon_c32, pack_b_neon_c64, pack_b_neon_f32,
    pack_b_neon_f64, pack_b_neon_i8, pack_b_neon_i8s, pack_b_neon_i16,
};

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 512;

// ============================================================
// f64
// ============================================================

/// Compute C = alpha * A @ B + beta * C for row-major f64 matrices.
///
/// SAFETY: caller upholds buffer-length invariants for `a/b/c`. NEON
/// availability is implied by `target_arch = "aarch64"` (NEON is
/// mandatory in ARMv8).
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_f64(
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
                    *c.add(i * ldc + j) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let p = c.add(i * ldc + j);
                    *p *= beta;
                }
            }
        }
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_F64);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let eff_beta = if first_pc { beta } else { 1.0 };

            let mut b_pack = vec![0.0_f64; nr_strips * NR_NEON_F64 * kc];
            pack_b_neon_f64(kc, nc, b.add(pc * ldb + jc), ldb, &mut b_pack);

            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_F64);
                let mut a_pack = vec![0.0_f64; n_strips_m * MR_NEON_F64 * kc];
                pack_a_neon_f64(mc, kc, a.add(ic * lda + pc), lda, &mut a_pack);

                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_F64).min(NR_NEON_F64);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_F64 * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_F64).min(MR_NEON_F64);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_F64 * kc);
                        let c_tile =
                            c.add((ic + ir * MR_NEON_F64) * ldc + jc + jr * NR_NEON_F64);
                        if m_tile == MR_NEON_F64 && n_tile == NR_NEON_F64 {
                            kernel_4x8_neon_f64(
                                kc,
                                alpha,
                                a_strip,
                                b_strip,
                                eff_beta,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        } else {
                            kernel_edge_neon_f64(
                                m_tile,
                                n_tile,
                                kc,
                                alpha,
                                a_strip,
                                b_strip,
                                eff_beta,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        }
                    }
                }
                ic += mc;
            }

            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}

// ============================================================
// f32
// ============================================================

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_f32(
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
                    *c.add(i * ldc + j) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let p = c.add(i * ldc + j);
                    *p *= beta;
                }
            }
        }
        return;
    }
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_F32);
        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let eff_beta = if first_pc { beta } else { 1.0 };
            let mut b_pack = vec![0.0_f32; nr_strips * NR_NEON_F32 * kc];
            pack_b_neon_f32(kc, nc, b.add(pc * ldb + jc), ldb, &mut b_pack);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_F32);
                let mut a_pack = vec![0.0_f32; n_strips_m * MR_NEON_F32 * kc];
                pack_a_neon_f32(mc, kc, a.add(ic * lda + pc), lda, &mut a_pack);
                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_F32).min(NR_NEON_F32);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_F32 * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_F32).min(MR_NEON_F32);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_F32 * kc);
                        let c_tile =
                            c.add((ic + ir * MR_NEON_F32) * ldc + jc + jr * NR_NEON_F32);
                        if m_tile == MR_NEON_F32 && n_tile == NR_NEON_F32 {
                            kernel_4x16_neon_f32(
                                kc,
                                alpha,
                                a_strip,
                                b_strip,
                                eff_beta,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        } else {
                            kernel_edge_neon_f32(
                                m_tile,
                                n_tile,
                                kc,
                                alpha,
                                a_strip,
                                b_strip,
                                eff_beta,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        }
                    }
                }
                ic += mc;
            }
            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}

// ============================================================
// Complex<f64>
// ============================================================

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_c64(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let p_re = c.add(i * ldc * 2 + j * 2);
                let p_im = p_re.add(1);
                if beta_re == 0.0 && beta_im == 0.0 {
                    *p_re = 0.0;
                    *p_im = 0.0;
                } else if !(beta_re == 1.0 && beta_im == 0.0) {
                    let cr = *p_re;
                    let ci = *p_im;
                    *p_re = beta_re * cr - beta_im * ci;
                    *p_im = beta_re * ci + beta_im * cr;
                }
            }
        }
        return;
    }
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_C64);
        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let (eff_beta_re, eff_beta_im) = if first_pc {
                (beta_re, beta_im)
            } else {
                (1.0, 0.0)
            };
            let mut b_pack = vec![0.0_f64; nr_strips * NR_NEON_C64 * 2 * kc];
            pack_b_neon_c64(kc, nc, b.add(pc * ldb * 2 + jc * 2), ldb, &mut b_pack);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_C64);
                let mut a_pack = vec![0.0_f64; n_strips_m * MR_NEON_C64 * 2 * kc];
                pack_a_neon_c64(mc, kc, a.add(ic * lda * 2 + pc * 2), lda, &mut a_pack);
                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_C64).min(NR_NEON_C64);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_C64 * 2 * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_C64).min(MR_NEON_C64);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_C64 * 2 * kc);
                        let c_tile =
                            c.add(((ic + ir * MR_NEON_C64) * ldc + jc + jr * NR_NEON_C64) * 2);
                        if m_tile == MR_NEON_C64 && n_tile == NR_NEON_C64 {
                            kernel_4x2_neon_c64(
                                kc,
                                alpha_re,
                                alpha_im,
                                a_strip,
                                b_strip,
                                eff_beta_re,
                                eff_beta_im,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        } else {
                            kernel_edge_neon_c64(
                                m_tile,
                                n_tile,
                                kc,
                                alpha_re,
                                alpha_im,
                                a_strip,
                                b_strip,
                                eff_beta_re,
                                eff_beta_im,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        }
                    }
                }
                ic += mc;
            }
            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}

// ============================================================
// Complex<f32>
// ============================================================

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_c32(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let p_re = c.add(i * ldc * 2 + j * 2);
                let p_im = p_re.add(1);
                if beta_re == 0.0 && beta_im == 0.0 {
                    *p_re = 0.0;
                    *p_im = 0.0;
                } else if !(beta_re == 1.0 && beta_im == 0.0) {
                    let cr = *p_re;
                    let ci = *p_im;
                    *p_re = beta_re * cr - beta_im * ci;
                    *p_im = beta_re * ci + beta_im * cr;
                }
            }
        }
        return;
    }
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_C32);
        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let (eff_beta_re, eff_beta_im) = if first_pc {
                (beta_re, beta_im)
            } else {
                (1.0, 0.0)
            };
            let mut b_pack = vec![0.0_f32; nr_strips * NR_NEON_C32 * 2 * kc];
            pack_b_neon_c32(kc, nc, b.add(pc * ldb * 2 + jc * 2), ldb, &mut b_pack);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_C32);
                let mut a_pack = vec![0.0_f32; n_strips_m * MR_NEON_C32 * 2 * kc];
                pack_a_neon_c32(mc, kc, a.add(ic * lda * 2 + pc * 2), lda, &mut a_pack);
                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_C32).min(NR_NEON_C32);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_C32 * 2 * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_C32).min(MR_NEON_C32);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_C32 * 2 * kc);
                        let c_tile =
                            c.add(((ic + ir * MR_NEON_C32) * ldc + jc + jr * NR_NEON_C32) * 2);
                        if m_tile == MR_NEON_C32 && n_tile == NR_NEON_C32 {
                            kernel_4x4_neon_c32(
                                kc,
                                alpha_re,
                                alpha_im,
                                a_strip,
                                b_strip,
                                eff_beta_re,
                                eff_beta_im,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        } else {
                            kernel_edge_neon_c32(
                                m_tile,
                                n_tile,
                                kc,
                                alpha_re,
                                alpha_im,
                                a_strip,
                                b_strip,
                                eff_beta_re,
                                eff_beta_im,
                                c_tile,
                                ldc as isize,
                                1,
                            );
                        }
                    }
                }
                ic += mc;
            }
            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}

// ============================================================
// i16
// ============================================================

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_i16(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_I16);
        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let eff_acc = accumulate || !first_pc;
            let mut b_pack = vec![0_i16; nr_strips * NR_NEON_I16 * kc];
            pack_b_neon_i16(kc, nc, b.add(pc * ldb + jc), ldb, &mut b_pack);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_I16);
                let mut a_pack = vec![0_i16; n_strips_m * MR_NEON_I16 * kc];
                pack_a_neon_i16(mc, kc, a.add(ic * lda + pc), lda, &mut a_pack);
                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_I16).min(NR_NEON_I16);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_I16 * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_I16).min(MR_NEON_I16);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_I16 * kc);
                        let c_tile =
                            c.add((ic + ir * MR_NEON_I16) * ldc + jc + jr * NR_NEON_I16);
                        if m_tile == MR_NEON_I16 && n_tile == NR_NEON_I16 {
                            kernel_4x8_neon_i16(
                                kc,
                                a_strip,
                                b_strip,
                                c_tile,
                                ldc as isize,
                                1,
                                eff_acc,
                            );
                        } else {
                            kernel_edge_neon_i16(
                                m_tile,
                                n_tile,
                                kc,
                                a_strip,
                                b_strip,
                                c_tile,
                                ldc as isize,
                                1,
                                eff_acc,
                            );
                        }
                    }
                }
                ic += mc;
            }
            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}

// ============================================================
// i8 (u8 × i8 → i32)
// ============================================================

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_i8(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_I8);
        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let eff_acc = accumulate || !first_pc;
            let mut b_pack = vec![0_i8; nr_strips * NR_NEON_I8 * kc];
            pack_b_neon_i8(kc, nc, b.add(pc * ldb + jc), ldb, &mut b_pack);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_I8);
                let mut a_pack = vec![0_u8; n_strips_m * MR_NEON_I8 * kc];
                pack_a_neon_u8(mc, kc, a.add(ic * lda + pc), lda, &mut a_pack);
                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_I8).min(NR_NEON_I8);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_I8 * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_I8).min(MR_NEON_I8);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_I8 * kc);
                        let c_tile =
                            c.add((ic + ir * MR_NEON_I8) * ldc + jc + jr * NR_NEON_I8);
                        if m_tile == MR_NEON_I8 && n_tile == NR_NEON_I8 {
                            kernel_4x8_neon_i8(
                                kc,
                                a_strip,
                                b_strip,
                                c_tile,
                                ldc as isize,
                                1,
                                eff_acc,
                            );
                        } else {
                            kernel_edge_neon_i8(
                                m_tile,
                                n_tile,
                                kc,
                                a_strip,
                                b_strip,
                                c_tile,
                                ldc as isize,
                                1,
                                eff_acc,
                            );
                        }
                    }
                }
                ic += mc;
            }
            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}

// ============================================================
// i8s (i8 × i8 → i32)
// ============================================================

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gemm_neon_i8s(
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_NEON_I8S);
        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let eff_acc = accumulate || !first_pc;
            let mut b_pack = vec![0_i8; nr_strips * NR_NEON_I8S * kc];
            pack_b_neon_i8s(kc, nc, b.add(pc * ldb + jc), ldb, &mut b_pack);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                let n_strips_m = mc.div_ceil(MR_NEON_I8S);
                let mut a_pack = vec![0_i8; n_strips_m * MR_NEON_I8S * kc];
                pack_a_neon_i8s(mc, kc, a.add(ic * lda + pc), lda, &mut a_pack);
                for jr in 0..nr_strips {
                    let n_tile = (nc - jr * NR_NEON_I8S).min(NR_NEON_I8S);
                    let b_strip = b_pack.as_ptr().add(jr * NR_NEON_I8S * kc);
                    for ir in 0..n_strips_m {
                        let m_tile = (mc - ir * MR_NEON_I8S).min(MR_NEON_I8S);
                        let a_strip = a_pack.as_ptr().add(ir * MR_NEON_I8S * kc);
                        let c_tile =
                            c.add((ic + ir * MR_NEON_I8S) * ldc + jc + jr * NR_NEON_I8S);
                        if m_tile == MR_NEON_I8S && n_tile == NR_NEON_I8S {
                            kernel_4x8_neon_i8_signed(
                                kc,
                                a_strip,
                                b_strip,
                                c_tile,
                                ldc as isize,
                                1,
                                eff_acc,
                            );
                        } else {
                            kernel_edge_neon_i8_signed(
                                m_tile,
                                n_tile,
                                kc,
                                a_strip,
                                b_strip,
                                c_tile,
                                ldc as isize,
                                1,
                                eff_acc,
                            );
                        }
                    }
                }
                ic += mc;
            }
            pc += kc;
            first_pc = false;
        }
        jc += nc;
    }
}
