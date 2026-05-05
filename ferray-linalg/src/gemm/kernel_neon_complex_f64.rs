// AArch64 NEON ZGEMM (Complex<f64>) micro-kernel.
//
// Tile shape: MR=4, NR=2 (matches the AVX2 ZGEMM tile, since each NEON
// vreg holds exactly 1 Complex<f64> = 2 doubles, the same as 1/4 of an
// AVX2 ymm or 1 SSE-128 vreg).
//   - Result tile: 4 rows × 2 complex cols = 8 complex = 16 doubles.
//   - In v-regs: each row needs 2 complex / 1 cplx-per-vreg = 2 v-regs
//     for the "additive accumulator" and 2 for the "swap accumulator".
//     Total accumulators: 4 rows × 4 v-regs = 16 v-regs.
//   - Plus 2 v-regs for B (b_v, b_swap).
//   - Plus 2 v-regs for A broadcasts (a_re, a_im for one row at a time).
//   - Live: ~22 of 32.
//
// Complex multiply identity:
//   (a_re + i*a_im) * (b_re + i*b_im)
//     = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
//
// Per K iter, per row i:
//   acc_re[i].lane = sum(a_re[i] * b.lane)               // partial of real a×b
//   acc_swap[i].lane = sum(a_im[i] * b_swap.lane)        // partial with swapped b
// Combine at end:
//   c[i].re = acc_re[i].lane0 - acc_swap[i].lane0   (since b_swap.lane0 = b.im)
//   c[i].im = acc_re[i].lane1 + acc_swap[i].lane1   (since b_swap.lane1 = b.re)
//
// The "lane0 = sub, lane1 = add" pattern is realised via vfmsq_f64 with
// a constant `[+1.0, -1.0]` factor: `result = acc_re - acc_swap * sign`.
//
// Per K iter: 1 v-load of B (2 doubles per col × 2 cols = 4 doubles, 2
// vregs), 1 vextq for swap, 8 broadcasts (2 per row × 4 rows), 16 FMAs
// (4 rows × 2 cplx-cols × 2 acc_re/acc_swap). That's 16 FMAs per K =
// 8 cycles per K-iter (2 FMA pipes), giving 32 complex-FLOPs / 8 cycles
// = 4 cplx-FLOP/cycle = 16 real-FLOP/cycle = NEON-FMA peak for f64.

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vdupq_n_f64, vextq_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vmulq_f64,
    vst1q_f64,
};

pub const MR_NEON_C64: usize = 4;
pub const NR_NEON_C64: usize = 2;

/// 4×2 NEON ZGEMM micro-kernel. Treats `Complex<f64>` arrays as `f64`
/// slices of double the length: each complex element is 2 contiguous
/// f64s `[re, im]`.
///
/// `kc` is the depth in complex elements. `c_rs`/`c_cs` are row/column
/// strides of C **in complex elements**.
///
/// SAFETY: caller guarantees aarch64 NEON availability, `packed_a >=
/// kc * MR_NEON_C64 * 2 doubles`, `packed_b >= kc * NR_NEON_C64 * 2
/// doubles`, and `c` addresses an (m_tile, n_tile) sub-tile.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x2_neon_c64(
    kc: usize,
    alpha_re: f64,
    alpha_im: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    c_rs: isize, // row stride in COMPLEX elements
    c_cs: isize, // col stride in COMPLEX elements
) {
    // 4 rows × 2 cplx-cols × 2 (acc_re/acc_swap) = 16 vregs.
    let mut acc_re_00 = vdupq_n_f64(0.0); // row 0, col 0, additive
    let mut acc_re_01 = vdupq_n_f64(0.0); // row 0, col 1, additive
    let mut acc_sw_00 = vdupq_n_f64(0.0); // row 0, col 0, swap
    let mut acc_sw_01 = vdupq_n_f64(0.0);
    let mut acc_re_10 = vdupq_n_f64(0.0);
    let mut acc_re_11 = vdupq_n_f64(0.0);
    let mut acc_sw_10 = vdupq_n_f64(0.0);
    let mut acc_sw_11 = vdupq_n_f64(0.0);
    let mut acc_re_20 = vdupq_n_f64(0.0);
    let mut acc_re_21 = vdupq_n_f64(0.0);
    let mut acc_sw_20 = vdupq_n_f64(0.0);
    let mut acc_sw_21 = vdupq_n_f64(0.0);
    let mut acc_re_30 = vdupq_n_f64(0.0);
    let mut acc_re_31 = vdupq_n_f64(0.0);
    let mut acc_sw_30 = vdupq_n_f64(0.0);
    let mut acc_sw_31 = vdupq_n_f64(0.0);

    let mut a = packed_a;
    let mut b = packed_b;

    for _ in 0..kc {
        // Load B row: 2 complex = 4 doubles in 2 vregs.
        // b0 = [b.col0.re, b.col0.im],  b1 = [b.col1.re, b.col1.im].
        let b0 = vld1q_f64(b);
        let b1 = vld1q_f64(b.add(2));
        // b_swap0 = [b.col0.im, b.col0.re],  b_swap1 = [b.col1.im, b.col1.re].
        // vextq_f64::<1> rotates by 1 lane.
        let b0_sw = vextq_f64::<1>(b0, b0);
        let b1_sw = vextq_f64::<1>(b1, b1);

        // Row 0: A[0,k] = (a0.re, a0.im) at packed_a[0..2].
        let a0re = vdupq_n_f64(*a);
        let a0im = vdupq_n_f64(*a.add(1));
        acc_re_00 = vfmaq_f64(acc_re_00, a0re, b0);
        acc_re_01 = vfmaq_f64(acc_re_01, a0re, b1);
        acc_sw_00 = vfmaq_f64(acc_sw_00, a0im, b0_sw);
        acc_sw_01 = vfmaq_f64(acc_sw_01, a0im, b1_sw);

        // Row 1.
        let a1re = vdupq_n_f64(*a.add(2));
        let a1im = vdupq_n_f64(*a.add(3));
        acc_re_10 = vfmaq_f64(acc_re_10, a1re, b0);
        acc_re_11 = vfmaq_f64(acc_re_11, a1re, b1);
        acc_sw_10 = vfmaq_f64(acc_sw_10, a1im, b0_sw);
        acc_sw_11 = vfmaq_f64(acc_sw_11, a1im, b1_sw);

        // Row 2.
        let a2re = vdupq_n_f64(*a.add(4));
        let a2im = vdupq_n_f64(*a.add(5));
        acc_re_20 = vfmaq_f64(acc_re_20, a2re, b0);
        acc_re_21 = vfmaq_f64(acc_re_21, a2re, b1);
        acc_sw_20 = vfmaq_f64(acc_sw_20, a2im, b0_sw);
        acc_sw_21 = vfmaq_f64(acc_sw_21, a2im, b1_sw);

        // Row 3.
        let a3re = vdupq_n_f64(*a.add(6));
        let a3im = vdupq_n_f64(*a.add(7));
        acc_re_30 = vfmaq_f64(acc_re_30, a3re, b0);
        acc_re_31 = vfmaq_f64(acc_re_31, a3re, b1);
        acc_sw_30 = vfmaq_f64(acc_sw_30, a3im, b0_sw);
        acc_sw_31 = vfmaq_f64(acc_sw_31, a3im, b1_sw);

        a = a.add(MR_NEON_C64 * 2);
        b = b.add(NR_NEON_C64 * 2);
    }

    // Combine: c[i, j].re = acc_re.lane0 - acc_sw.lane0
    //          c[i, j].im = acc_re.lane1 + acc_sw.lane1
    // Encoded as: result = vfmsq(acc_re, acc_sw, [+1, -1]).
    //   lane0: acc_re.0 - acc_sw.0 * 1  =  acc_re.0 - acc_sw.0   ✓
    //   lane1: acc_re.1 - acc_sw.1 * -1 =  acc_re.1 + acc_sw.1   ✓
    let sign: [f64; 2] = [1.0, -1.0];
    let sign_v = vld1q_f64(sign.as_ptr());
    let c00 = vfmsq_f64(acc_re_00, acc_sw_00, sign_v);
    let c01 = vfmsq_f64(acc_re_01, acc_sw_01, sign_v);
    let c10 = vfmsq_f64(acc_re_10, acc_sw_10, sign_v);
    let c11 = vfmsq_f64(acc_re_11, acc_sw_11, sign_v);
    let c20 = vfmsq_f64(acc_re_20, acc_sw_20, sign_v);
    let c21 = vfmsq_f64(acc_re_21, acc_sw_21, sign_v);
    let c30 = vfmsq_f64(acc_re_30, acc_sw_30, sign_v);
    let c31 = vfmsq_f64(acc_re_31, acc_sw_31, sign_v);

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
        let ldc_doubles = (c_rs as usize) * 2;
        store_row_2c(c, 0, ldc_doubles, c00, c01, beta_re, beta_im);
        store_row_2c(c, 1, ldc_doubles, c10, c11, beta_re, beta_im);
        store_row_2c(c, 2, ldc_doubles, c20, c21, beta_re, beta_im);
        store_row_2c(c, 3, ldc_doubles, c30, c31, beta_re, beta_im);
    } else {
        store_strided_row_2c(c, c_rs, c_cs, 0, c00, c01, beta_re, beta_im);
        store_strided_row_2c(c, c_rs, c_cs, 1, c10, c11, beta_re, beta_im);
        store_strided_row_2c(c, c_rs, c_cs, 2, c20, c21, beta_re, beta_im);
        store_strided_row_2c(c, c_rs, c_cs, 3, c30, c31, beta_re, beta_im);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn scale_alpha(c: float64x2_t, alpha_re: f64, alpha_im: f64) -> float64x2_t {
    if alpha_re == 1.0 && alpha_im == 0.0 {
        return c;
    }
    // result = alpha * c, where alpha = (alpha_re + i*alpha_im).
    //   result.re = alpha_re * c.re - alpha_im * c.im
    //   result.im = alpha_re * c.im + alpha_im * c.re
    let alpha_re_v = vdupq_n_f64(alpha_re);
    let alpha_im_v = vdupq_n_f64(alpha_im);
    let c_swap = vextq_f64::<1>(c, c);
    // term_re = alpha_re * c   = [alpha_re*c.re,  alpha_re*c.im]
    let term_re = vmulq_f64(alpha_re_v, c);
    // term_sw = alpha_im * c_swap = [alpha_im*c.im, alpha_im*c.re]
    let term_sw = vmulq_f64(alpha_im_v, c_swap);
    // result.lane0 = term_re.0 - term_sw.0,  result.lane1 = term_re.1 + term_sw.1
    let sign: [f64; 2] = [1.0, -1.0];
    let sign_v = vld1q_f64(sign.as_ptr());
    vfmsq_f64(term_re, term_sw, sign_v)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_row_2c(
    c: *mut f64,
    row: usize,
    ldc_doubles: usize,
    new0: float64x2_t,
    new1: float64x2_t,
    beta_re: f64,
    beta_im: f64,
) {
    let p = c.add(row * ldc_doubles);
    if beta_re == 0.0 && beta_im == 0.0 {
        vst1q_f64(p, new0);
        vst1q_f64(p.add(2), new1);
        return;
    }
    if beta_re == 1.0 && beta_im == 0.0 {
        let cur0 = vld1q_f64(p);
        let cur1 = vld1q_f64(p.add(2));
        vst1q_f64(p, vaddq_f64(cur0, new0));
        vst1q_f64(p.add(2), vaddq_f64(cur1, new1));
        return;
    }
    // General complex beta * cur + new.
    let cur0 = vld1q_f64(p);
    let cur1 = vld1q_f64(p.add(2));
    let scaled0 = beta_complex_scale(cur0, beta_re, beta_im);
    let scaled1 = beta_complex_scale(cur1, beta_re, beta_im);
    vst1q_f64(p, vaddq_f64(scaled0, new0));
    vst1q_f64(p.add(2), vaddq_f64(scaled1, new1));
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn beta_complex_scale(v: float64x2_t, beta_re: f64, beta_im: f64) -> float64x2_t {
    // out.re = beta_re * v.re - beta_im * v.im
    // out.im = beta_re * v.im + beta_im * v.re
    // Encoded as: term_re = beta_re * v          = [beta_re*v.re, beta_re*v.im]
    //             term_sw = beta_im * v_swap     = [beta_im*v.im, beta_im*v.re]
    // out = term_re - term_sw * [+1, -1]    (vfmsq with sign)
    //   lane0 = term_re.0 - term_sw.0 * 1  = beta_re*v.re - beta_im*v.im   ✓
    //   lane1 = term_re.1 - term_sw.1 * -1 = beta_re*v.im + beta_im*v.re   ✓
    let beta_re_v = vdupq_n_f64(beta_re);
    let beta_im_v = vdupq_n_f64(beta_im);
    let v_swap = vextq_f64::<1>(v, v);
    let term_re = vmulq_f64(beta_re_v, v);
    let term_sw = vmulq_f64(beta_im_v, v_swap);
    let sign: [f64; 2] = [1.0, -1.0];
    let sign_v = vld1q_f64(sign.as_ptr());
    vfmsq_f64(term_re, term_sw, sign_v)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_strided_row_2c(
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    new0: float64x2_t,
    new1: float64x2_t,
    beta_re: f64,
    beta_im: f64,
) {
    let mut buf = [0.0_f64; 4];
    vst1q_f64(buf.as_mut_ptr(), new0);
    vst1q_f64(buf.as_mut_ptr().add(2), new1);
    let row_off = (row as isize) * c_rs;
    for j in 0..NR_NEON_C64 {
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

/// Edge-case kernel for tail tiles where m < MR_NEON_C64 or n < NR_NEON_C64.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_neon_c64(
    m: usize,
    n: usize,
    kc: usize,
    alpha_re: f64,
    alpha_im: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    debug_assert!(m <= MR_NEON_C64);
    debug_assert!(n <= NR_NEON_C64);
    let mut tile_re = [[0.0_f64; NR_NEON_C64]; MR_NEON_C64];
    let mut tile_im = [[0.0_f64; NR_NEON_C64]; MR_NEON_C64];
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
        a = a.add(MR_NEON_C64 * 2);
        b = b.add(NR_NEON_C64 * 2);
    }
    for i in 0..m {
        for j in 0..n {
            let p_re = c.offset((i as isize) * c_rs * 2 + (j as isize) * c_cs * 2);
            let p_im = p_re.add(1);
            // Apply complex alpha.
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
