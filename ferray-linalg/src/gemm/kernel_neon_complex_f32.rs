// AArch64 NEON CGEMM (Complex<f32>) micro-kernel.
//
// Tile shape: MR=4, NR=4. Each 128-bit vreg holds 2 Complex<f32> (4 f32),
// so a 4-complex-wide row is 2 vregs. With the swap-accumulator pattern,
// each row needs 2 (cols × 2 vregs) acc_re plus 2 acc_swap = 4 vregs.
// Total accumulators: 4 rows × 4 = 16 vregs. Plus 2 vregs for B and 1
// for A broadcasts → ~19 of 32 used.
//
// Complex multiply identity is identical to ZGEMM; the swap is built via
// `vrev64q_f32` which reverses each 64-bit half (swaps adjacent (re, im)
// pairs within each lane-pair).
//
// Per K iter: 2 v-loads of B (8 floats), 4 broadcasts of A (one per row),
// 16 FMAs (4 rows × 2 col-vregs × 2 acc_re/acc_swap). 16/2 = 8 cycles
// per K-iter; 64 cplx-FLOP / 8 = 8 cplx-FLOP/cycle = 32 real FLOP/cycle
// = NEON-FMA peak for f32.

use core::arch::aarch64::{
    float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vfmsq_f32, vld1q_f32, vmulq_f32, vrev64q_f32,
    vst1q_f32,
};

pub const MR_NEON_C32: usize = 4;
pub const NR_NEON_C32: usize = 4;

/// 4×4 NEON CGEMM micro-kernel.
///
/// SAFETY: caller guarantees aarch64 NEON, `packed_a >= kc * MR * 2 floats`,
/// `packed_b >= kc * NR * 2 floats`, `c` addresses an (m_tile, n_tile)
/// sub-tile.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x4_neon_c32(
    kc: usize,
    alpha_re: f32,
    alpha_im: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    // 4 rows × 2 col-vregs × 2 (acc_re/acc_sw) = 16 vregs.
    let mut acc_re_00 = vdupq_n_f32(0.0);
    let mut acc_re_01 = vdupq_n_f32(0.0);
    let mut acc_sw_00 = vdupq_n_f32(0.0);
    let mut acc_sw_01 = vdupq_n_f32(0.0);
    let mut acc_re_10 = vdupq_n_f32(0.0);
    let mut acc_re_11 = vdupq_n_f32(0.0);
    let mut acc_sw_10 = vdupq_n_f32(0.0);
    let mut acc_sw_11 = vdupq_n_f32(0.0);
    let mut acc_re_20 = vdupq_n_f32(0.0);
    let mut acc_re_21 = vdupq_n_f32(0.0);
    let mut acc_sw_20 = vdupq_n_f32(0.0);
    let mut acc_sw_21 = vdupq_n_f32(0.0);
    let mut acc_re_30 = vdupq_n_f32(0.0);
    let mut acc_re_31 = vdupq_n_f32(0.0);
    let mut acc_sw_30 = vdupq_n_f32(0.0);
    let mut acc_sw_31 = vdupq_n_f32(0.0);

    let mut a = packed_a;
    let mut b = packed_b;

    for _ in 0..kc {
        // 2 vregs of B = NR=4 complex = 8 floats.
        // b0 = [c0.re, c0.im, c1.re, c1.im],  b1 = [c2.re, c2.im, c3.re, c3.im].
        let b0 = vld1q_f32(b);
        let b1 = vld1q_f32(b.add(4));
        // vrev64q_f32 reverses lanes within each 64-bit half:
        //   b0_sw = [c0.im, c0.re, c1.im, c1.re]  ✓
        let b0_sw = vrev64q_f32(b0);
        let b1_sw = vrev64q_f32(b1);

        // Row 0: A[0,k] = (a0.re, a0.im).
        let a0re = vdupq_n_f32(*a);
        let a0im = vdupq_n_f32(*a.add(1));
        acc_re_00 = vfmaq_f32(acc_re_00, a0re, b0);
        acc_re_01 = vfmaq_f32(acc_re_01, a0re, b1);
        acc_sw_00 = vfmaq_f32(acc_sw_00, a0im, b0_sw);
        acc_sw_01 = vfmaq_f32(acc_sw_01, a0im, b1_sw);

        // Row 1.
        let a1re = vdupq_n_f32(*a.add(2));
        let a1im = vdupq_n_f32(*a.add(3));
        acc_re_10 = vfmaq_f32(acc_re_10, a1re, b0);
        acc_re_11 = vfmaq_f32(acc_re_11, a1re, b1);
        acc_sw_10 = vfmaq_f32(acc_sw_10, a1im, b0_sw);
        acc_sw_11 = vfmaq_f32(acc_sw_11, a1im, b1_sw);

        // Row 2.
        let a2re = vdupq_n_f32(*a.add(4));
        let a2im = vdupq_n_f32(*a.add(5));
        acc_re_20 = vfmaq_f32(acc_re_20, a2re, b0);
        acc_re_21 = vfmaq_f32(acc_re_21, a2re, b1);
        acc_sw_20 = vfmaq_f32(acc_sw_20, a2im, b0_sw);
        acc_sw_21 = vfmaq_f32(acc_sw_21, a2im, b1_sw);

        // Row 3.
        let a3re = vdupq_n_f32(*a.add(6));
        let a3im = vdupq_n_f32(*a.add(7));
        acc_re_30 = vfmaq_f32(acc_re_30, a3re, b0);
        acc_re_31 = vfmaq_f32(acc_re_31, a3re, b1);
        acc_sw_30 = vfmaq_f32(acc_sw_30, a3im, b0_sw);
        acc_sw_31 = vfmaq_f32(acc_sw_31, a3im, b1_sw);

        a = a.add(MR_NEON_C32 * 2);
        b = b.add(NR_NEON_C32 * 2);
    }

    // Combine via vfmsq_f32 with [+1, -1, +1, -1] sign pattern.
    let sign: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
    let sign_v = vld1q_f32(sign.as_ptr());
    let c00 = vfmsq_f32(acc_re_00, acc_sw_00, sign_v);
    let c01 = vfmsq_f32(acc_re_01, acc_sw_01, sign_v);
    let c10 = vfmsq_f32(acc_re_10, acc_sw_10, sign_v);
    let c11 = vfmsq_f32(acc_re_11, acc_sw_11, sign_v);
    let c20 = vfmsq_f32(acc_re_20, acc_sw_20, sign_v);
    let c21 = vfmsq_f32(acc_re_21, acc_sw_21, sign_v);
    let c30 = vfmsq_f32(acc_re_30, acc_sw_30, sign_v);
    let c31 = vfmsq_f32(acc_re_31, acc_sw_31, sign_v);

    // Apply complex alpha.
    let c00 = scale_alpha(c00, alpha_re, alpha_im);
    let c01 = scale_alpha(c01, alpha_re, alpha_im);
    let c10 = scale_alpha(c10, alpha_re, alpha_im);
    let c11 = scale_alpha(c11, alpha_re, alpha_im);
    let c20 = scale_alpha(c20, alpha_re, alpha_im);
    let c21 = scale_alpha(c21, alpha_re, alpha_im);
    let c30 = scale_alpha(c30, alpha_re, alpha_im);
    let c31 = scale_alpha(c31, alpha_re, alpha_im);

    if c_cs == 1 && c_rs > 0 {
        let ldc_floats = (c_rs as usize) * 2;
        store_row_4c(c, 0, ldc_floats, c00, c01, beta_re, beta_im);
        store_row_4c(c, 1, ldc_floats, c10, c11, beta_re, beta_im);
        store_row_4c(c, 2, ldc_floats, c20, c21, beta_re, beta_im);
        store_row_4c(c, 3, ldc_floats, c30, c31, beta_re, beta_im);
    } else {
        store_strided_row_4c(c, c_rs, c_cs, 0, c00, c01, beta_re, beta_im);
        store_strided_row_4c(c, c_rs, c_cs, 1, c10, c11, beta_re, beta_im);
        store_strided_row_4c(c, c_rs, c_cs, 2, c20, c21, beta_re, beta_im);
        store_strided_row_4c(c, c_rs, c_cs, 3, c30, c31, beta_re, beta_im);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn scale_alpha(c: float32x4_t, alpha_re: f32, alpha_im: f32) -> float32x4_t {
    if alpha_re == 1.0 && alpha_im == 0.0 {
        return c;
    }
    let alpha_re_v = vdupq_n_f32(alpha_re);
    let alpha_im_v = vdupq_n_f32(alpha_im);
    let c_swap = vrev64q_f32(c);
    let term_re = vmulq_f32(alpha_re_v, c);
    let term_sw = vmulq_f32(alpha_im_v, c_swap);
    let sign: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
    let sign_v = vld1q_f32(sign.as_ptr());
    vfmsq_f32(term_re, term_sw, sign_v)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_row_4c(
    c: *mut f32,
    row: usize,
    ldc_floats: usize,
    new0: float32x4_t,
    new1: float32x4_t,
    beta_re: f32,
    beta_im: f32,
) {
    let p = c.add(row * ldc_floats);
    if beta_re == 0.0 && beta_im == 0.0 {
        vst1q_f32(p, new0);
        vst1q_f32(p.add(4), new1);
        return;
    }
    if beta_re == 1.0 && beta_im == 0.0 {
        let cur0 = vld1q_f32(p);
        let cur1 = vld1q_f32(p.add(4));
        vst1q_f32(p, vaddq_f32(cur0, new0));
        vst1q_f32(p.add(4), vaddq_f32(cur1, new1));
        return;
    }
    let cur0 = vld1q_f32(p);
    let cur1 = vld1q_f32(p.add(4));
    let scaled0 = beta_complex_scale(cur0, beta_re, beta_im);
    let scaled1 = beta_complex_scale(cur1, beta_re, beta_im);
    vst1q_f32(p, vaddq_f32(scaled0, new0));
    vst1q_f32(p.add(4), vaddq_f32(scaled1, new1));
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn beta_complex_scale(v: float32x4_t, beta_re: f32, beta_im: f32) -> float32x4_t {
    let beta_re_v = vdupq_n_f32(beta_re);
    let beta_im_v = vdupq_n_f32(beta_im);
    let v_swap = vrev64q_f32(v);
    let term_re = vmulq_f32(beta_re_v, v);
    let term_sw = vmulq_f32(beta_im_v, v_swap);
    let sign: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
    let sign_v = vld1q_f32(sign.as_ptr());
    vfmsq_f32(term_re, term_sw, sign_v)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_strided_row_4c(
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    new0: float32x4_t,
    new1: float32x4_t,
    beta_re: f32,
    beta_im: f32,
) {
    let mut buf = [0.0_f32; 8];
    vst1q_f32(buf.as_mut_ptr(), new0);
    vst1q_f32(buf.as_mut_ptr().add(4), new1);
    let row_off = (row as isize) * c_rs;
    for j in 0..NR_NEON_C32 {
        let p = c.offset(row_off * 2 + (j as isize) * c_cs * 2);
        let new_re = buf[j * 2];
        let new_im = buf[j * 2 + 1];
        if beta_re == 0.0 && beta_im == 0.0 {
            *p = new_re;
            *p.add(1) = new_im;
        } else if beta_re == 1.0 && beta_im == 0.0 {
            *p += new_re;
            *p.add(1) += new_im;
        } else {
            let cur_re = *p;
            let cur_im = *p.add(1);
            let scaled_re = beta_re * cur_re - beta_im * cur_im;
            let scaled_im = beta_re * cur_im + beta_im * cur_re;
            *p = scaled_re + new_re;
            *p.add(1) = scaled_im + new_im;
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_neon_c32(
    m: usize,
    n: usize,
    kc: usize,
    alpha_re: f32,
    alpha_im: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    debug_assert!(m <= MR_NEON_C32);
    debug_assert!(n <= NR_NEON_C32);
    let mut tile_re = [[0.0_f32; NR_NEON_C32]; MR_NEON_C32];
    let mut tile_im = [[0.0_f32; NR_NEON_C32]; MR_NEON_C32];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let a_re = *a.add(i * 2);
            let a_im = *a.add(i * 2 + 1);
            for j in 0..n {
                let b_re = *b.add(j * 2);
                let b_im = *b.add(j * 2 + 1);
                tile_re[i][j] += a_re * b_re - a_im * b_im;
                tile_im[i][j] += a_re * b_im + a_im * b_re;
            }
        }
        a = a.add(MR_NEON_C32 * 2);
        b = b.add(NR_NEON_C32 * 2);
    }
    for i in 0..m {
        for j in 0..n {
            let p_re = c.offset((i as isize) * c_rs * 2 + (j as isize) * c_cs * 2);
            let p_im = p_re.add(1);
            let v_re = alpha_re * tile_re[i][j] - alpha_im * tile_im[i][j];
            let v_im = alpha_re * tile_im[i][j] + alpha_im * tile_re[i][j];
            if beta_re == 0.0 && beta_im == 0.0 {
                *p_re = v_re;
                *p_im = v_im;
            } else if beta_re == 1.0 && beta_im == 0.0 {
                *p_re += v_re;
                *p_im += v_im;
            } else {
                let cur_re = *p_re;
                let cur_im = *p_im;
                *p_re = beta_re * cur_re - beta_im * cur_im + v_re;
                *p_im = beta_re * cur_im + beta_im * cur_re + v_im;
            }
        }
    }
}
