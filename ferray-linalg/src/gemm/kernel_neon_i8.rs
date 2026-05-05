// AArch64 NEON quantized i8 GEMM micro-kernel — u8 × i8 → i32, the
// canonical quantized ML inference pattern (activations u8 after
// ReLU+quant, weights i8).
//
// NEON design: ARMv8 base NEON has no native mixed-sign u8×i8→i32 dot
// product. The Armv8.4-A `dotprod` extension adds `vusdotq_s32` for
// exactly this case, but it requires runtime feature detection and
// compile-time `-C target-feature=+dotprod`. For a stable portable
// kernel we widen both inputs to i16 first, then use `vmlal_s16` /
// `vmlal_high_s16` widening multiply-accumulate to i32.
//
// (A `dotprod`-enhanced fast path is a follow-up; tracking via a
// future i8mm-aware dispatcher.)
//
// Tile: MR=4, NR=8 (matches AVX2 i8 tile).
//   - Each row's 8 i32 accumulators: 2 vregs (low 4 + high 4).
//   - Total accumulators: 4 rows × 2 = 8 vregs.
//   - Plus 1 vreg for the widened B row (8 i16) and 1 for A broadcast.
//   - Live: ~10 of 32.
//
// Per K iter: 1 load + widen of B (8 i8 → 8 i16), 4 broadcasts of A
// (one per row, widened to 8 i16), 8 widening multiply-accumulates
// (vmlal_s16 + vmlal_high_s16 per row × 4 rows).
//
// Packing layouts:
//   packed_a[k * MR + i] = A[row + i, depth + k]   (u8, MR-tall column strips)
//   packed_b[k * NR + j] = B[depth + k, col + j]   (i8, NR-wide row strips)
// KB=1 (single K per inner iter); the kernel takes the K-dim entirely
// in the outer loop. (AVX2 uses KB=4 because vpmaddubsw + vpmaddwd
// reduces 4 K-iters at once; NEON's element-wise vmlal_s16 handles
// one K at a time naturally.)

use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddq_s32, vdupq_n_s16, vdupq_n_s32, vget_low_s16, vld1_s8, vld1q_s32,
    vmlal_high_s16, vmlal_s16, vmovl_s8, vst1q_s32,
};

pub const MR_NEON_I8: usize = 4;
pub const NR_NEON_I8: usize = 8;

/// 4×8 NEON quantized i8 micro-kernel. Computes C += A @ B (u8 × i8 → i32).
///
/// SAFETY: caller guarantees aarch64 NEON, `packed_a >= kc * MR_NEON_I8 u8s`,
/// `packed_b >= kc * NR_NEON_I8 i8s`, `c` addresses an (m_tile, n_tile)
/// sub-tile of i32.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x8_neon_i8(
    kc: usize,
    packed_a: *const u8,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
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
        // Load 8 i8 of B[k, 0..8] then sign-widen to 8 i16.
        let b_i8 = vld1_s8(b);
        let b_i16: int16x8_t = vmovl_s8(b_i8);

        // Each A entry is u8 ∈ [0, 255]; convert to i16 (zero-extend),
        // broadcast to 8 lanes.
        let a0 = vdupq_n_s16(*a as i16);
        let a1 = vdupq_n_s16(*a.add(1) as i16);
        let a2 = vdupq_n_s16(*a.add(2) as i16);
        let a3 = vdupq_n_s16(*a.add(3) as i16);

        // i16 × i16 widening multiply-accumulate to i32.
        // vmlal_s16(c, a, b) = c + (i32) a * (i32) b, element-wise on 4 lanes.
        acc0_lo = vmlal_s16(acc0_lo, vget_low_s16(a0), vget_low_s16(b_i16));
        acc0_hi = vmlal_high_s16(acc0_hi, a0, b_i16);
        acc1_lo = vmlal_s16(acc1_lo, vget_low_s16(a1), vget_low_s16(b_i16));
        acc1_hi = vmlal_high_s16(acc1_hi, a1, b_i16);
        acc2_lo = vmlal_s16(acc2_lo, vget_low_s16(a2), vget_low_s16(b_i16));
        acc2_hi = vmlal_high_s16(acc2_hi, a2, b_i16);
        acc3_lo = vmlal_s16(acc3_lo, vget_low_s16(a3), vget_low_s16(b_i16));
        acc3_hi = vmlal_high_s16(acc3_hi, a3, b_i16);

        a = a.add(MR_NEON_I8);
        b = b.add(NR_NEON_I8);
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
pub unsafe fn kernel_edge_neon_i8(
    m: usize,
    n: usize,
    kc: usize,
    packed_a: *const u8,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(m <= MR_NEON_I8);
    debug_assert!(n <= NR_NEON_I8);
    let mut tile = [[0_i32; NR_NEON_I8]; MR_NEON_I8];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i) as i32; // u8 → i32 (zero-extend)
            for j in 0..n {
                let bj = *b.add(j) as i32; // i8 → i32 (sign-extend)
                tile[i][j] = tile[i][j].wrapping_add(ai.wrapping_mul(bj));
            }
        }
        a = a.add(MR_NEON_I8);
        b = b.add(NR_NEON_I8);
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
