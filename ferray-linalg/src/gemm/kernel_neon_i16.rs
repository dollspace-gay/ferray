// AArch64 NEON i16 × i16 → i32 GEMM micro-kernel.
//
// Tile: MR=4, NR=8 (matches the AVX2 i16 tile).
//   - Each NEON 128-bit vreg holds 8 i16 or 4 i32. So one row's 8 i32
//     accumulators occupy 2 vregs (low 4 + high 4).
//   - Total accumulators: 4 rows × 2 vregs = 8 vregs.
//   - Plus 1 vreg for current B row (8 i16) and a few for A broadcasts.
//   - Live: ~12 of 32. Plenty of headroom (NEON has 32 vregs).
//
// Per K iter: 1 v-load of B (8 i16), 4 broadcasts of A (one per row,
// 8 i16 each), 8 widening multiply-accumulates (vmlal_s16 + vmlal_high_s16
// per row × 4 rows). vmlal_s16 = 4 i16*i16 → 4 i32 += products.
//
// Each output cell needs one i32 accumulator, so KB (the K block depth
// per inner-loop iter) is just 1 — unlike AVX2's KB=2 which is forced
// by the vpmaddwd pairwise reduction. NEON's vmlal_s16 already gives
// per-element products, so we don't need pairwise pre-reduction.
//
// Packing layout:
//   packed_a[k * MR + i] = A[row + i, depth + k]  (MR-tall column strips)
//   packed_b[k * NR + j] = B[depth + k, col + j]  (NR-wide row strips)

use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddq_s32, vdupq_n_s16, vdupq_n_s32, vget_low_s16, vld1q_s16, vld1q_s32,
    vmlal_high_s16, vmlal_s16, vst1q_s32,
};

pub const MR_NEON_I16: usize = 4;
pub const NR_NEON_I16: usize = 8;

/// 4×8 NEON i16 GEMM micro-kernel. Computes C += A @ B for
/// row-major i16 matrices into i32 accumulator.
///
/// SAFETY: caller guarantees aarch64 NEON, `packed_a >= kc * MR_NEON_I16
/// i16s`, `packed_b >= kc * NR_NEON_I16 i16s`, `c` addresses an
/// (m_tile, n_tile) sub-tile of i32.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x8_neon_i16(
    kc: usize,
    packed_a: *const i16,
    packed_b: *const i16,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    // 4 rows × 2 vregs (lo, hi) of i32.
    let mut acc0_lo = vdupq_n_s32(0);
    let mut acc0_hi = vdupq_n_s32(0);
    let mut acc1_lo = vdupq_n_s32(0);
    let mut acc1_hi = vdupq_n_s32(0);
    let mut acc2_lo = vdupq_n_s32(0);
    let mut acc2_hi = vdupq_n_s32(0);
    let mut acc3_lo = vdupq_n_s32(0);
    let mut acc3_hi = vdupq_n_s32(0);

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        let b_v: int16x8_t = vld1q_s16(b);
        // Each broadcast is 8 i16 = same value in all 8 lanes.
        let a0 = vdupq_n_s16(*a);
        let a1 = vdupq_n_s16(*a.add(1));
        let a2 = vdupq_n_s16(*a.add(2));
        let a3 = vdupq_n_s16(*a.add(3));

        // For row i: acc_lo += low_half(a_i) * low_half(b)
        //            acc_hi += high_half(a_i) * high_half(b)
        // vmlal_s16(c, a, b) = c + a*b widened to i32; takes 4-lane i16
        // operands (vget_low_s16 / vget_high_s16 splits an 8-lane vreg).
        acc0_lo = vmlal_s16(acc0_lo, vget_low_s16(a0), vget_low_s16(b_v));
        acc0_hi = vmlal_high_s16(acc0_hi, a0, b_v);
        acc1_lo = vmlal_s16(acc1_lo, vget_low_s16(a1), vget_low_s16(b_v));
        acc1_hi = vmlal_high_s16(acc1_hi, a1, b_v);
        acc2_lo = vmlal_s16(acc2_lo, vget_low_s16(a2), vget_low_s16(b_v));
        acc2_hi = vmlal_high_s16(acc2_hi, a2, b_v);
        acc3_lo = vmlal_s16(acc3_lo, vget_low_s16(a3), vget_low_s16(b_v));
        acc3_hi = vmlal_high_s16(acc3_hi, a3, b_v);

        a = a.add(MR_NEON_I16);
        b = b.add(NR_NEON_I16);
    }

    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_8i32(c, 0, ldc, acc0_lo, acc0_hi, accumulate);
        store_row_8i32(c, 1, ldc, acc1_lo, acc1_hi, accumulate);
        store_row_8i32(c, 2, ldc, acc2_lo, acc2_hi, accumulate);
        store_row_8i32(c, 3, ldc, acc3_lo, acc3_hi, accumulate);
    } else {
        store_strided_row_8i32(c, c_rs, c_cs, 0, acc0_lo, acc0_hi, accumulate);
        store_strided_row_8i32(c, c_rs, c_cs, 1, acc1_lo, acc1_hi, accumulate);
        store_strided_row_8i32(c, c_rs, c_cs, 2, acc2_lo, acc2_hi, accumulate);
        store_strided_row_8i32(c, c_rs, c_cs, 3, acc3_lo, acc3_hi, accumulate);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_row_8i32(
    c: *mut i32,
    row: usize,
    ldc: usize,
    lo: int32x4_t,
    hi: int32x4_t,
    accumulate: bool,
) {
    let p = c.add(row * ldc);
    if accumulate {
        let cur_lo = vld1q_s32(p);
        let cur_hi = vld1q_s32(p.add(4));
        vst1q_s32(p, vaddq_s32(cur_lo, lo));
        vst1q_s32(p.add(4), vaddq_s32(cur_hi, hi));
    } else {
        vst1q_s32(p, lo);
        vst1q_s32(p.add(4), hi);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_strided_row_8i32(
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    lo: int32x4_t,
    hi: int32x4_t,
    accumulate: bool,
) {
    let mut buf = [0_i32; 8];
    vst1q_s32(buf.as_mut_ptr(), lo);
    vst1q_s32(buf.as_mut_ptr().add(4), hi);
    let row_off = (row as isize) * c_rs;
    for j in 0..8 {
        let p = c.offset(row_off + (j as isize) * c_cs);
        if accumulate {
            *p = (*p).wrapping_add(buf[j]);
        } else {
            *p = buf[j];
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_neon_i16(
    m: usize,
    n: usize,
    kc: usize,
    packed_a: *const i16,
    packed_b: *const i16,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(m <= MR_NEON_I16);
    debug_assert!(n <= NR_NEON_I16);
    let mut tile = [[0_i32; NR_NEON_I16]; MR_NEON_I16];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i) as i32;
            for j in 0..n {
                tile[i][j] = tile[i][j].wrapping_add(ai.wrapping_mul(*b.add(j) as i32));
            }
        }
        a = a.add(MR_NEON_I16);
        b = b.add(NR_NEON_I16);
    }
    for i in 0..m {
        for j in 0..n {
            let p = c.offset((i as isize) * c_rs + (j as isize) * c_cs);
            if accumulate {
                *p = (*p).wrapping_add(tile[i][j]);
            } else {
                *p = tile[i][j];
            }
        }
    }
}
