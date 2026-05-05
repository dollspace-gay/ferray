// AArch64 NEON SGEMM micro-kernel — design parallel to the AVX2 4×16
// SGEMM kernel.
//
// Tile shape: MR=4, NR=16.
//   - Result tile: 4 rows × 16 cols of f32 = 64 floats.
//   - In v-regs: each row needs 16 floats / 4 lanes = 4 v-regs.
//     Total accumulators: 4 rows × 4 v-regs = 16 v-regs.
//   - Plus 4 v-regs for the current B row.
//   - Plus 1 v-reg for A broadcasts (4 floats = 1 vreg, lanes addressed).
//   - Total live v-regs: ~21 of 32.
//
// Per K iteration: 4 v-loads of B (16 floats), 1 v-load of A (4 floats),
// 16 FMAs (4 rows × 4 col-vregs).
// Cortex-A78 / Neoverse-N1 throughput: 16 FMAs / 2 = 8 cycles per K
// → 128 FLOPs / 8 cycles = 16 FLOPs/cycle = NEON-FMA peak for f32.

use core::arch::aarch64::{
    float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vfmaq_laneq_f32, vld1q_f32, vmulq_f32,
    vst1q_f32,
};

pub const MR_NEON_F32: usize = 4;
pub const NR_NEON_F32: usize = 16;

/// 4×16 NEON SGEMM micro-kernel. Computes the 4-row × 16-col tile of C
/// from kc depth of packed A and B.
///
/// SAFETY: caller guarantees aarch64 NEON availability (mandatory on
/// ARMv8), `packed_a >= kc * MR floats`, `packed_b >= kc * NR floats`,
/// and `c` addresses an (m_tile, n_tile) sub-tile of the destination.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x16_neon_f32(
    kc: usize,
    alpha: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    let mut c00 = vdupq_n_f32(0.0);
    let mut c01 = vdupq_n_f32(0.0);
    let mut c02 = vdupq_n_f32(0.0);
    let mut c03 = vdupq_n_f32(0.0);
    let mut c10 = vdupq_n_f32(0.0);
    let mut c11 = vdupq_n_f32(0.0);
    let mut c12 = vdupq_n_f32(0.0);
    let mut c13 = vdupq_n_f32(0.0);
    let mut c20 = vdupq_n_f32(0.0);
    let mut c21 = vdupq_n_f32(0.0);
    let mut c22 = vdupq_n_f32(0.0);
    let mut c23 = vdupq_n_f32(0.0);
    let mut c30 = vdupq_n_f32(0.0);
    let mut c31 = vdupq_n_f32(0.0);
    let mut c32 = vdupq_n_f32(0.0);
    let mut c33 = vdupq_n_f32(0.0);

    let mut a = packed_a;
    let mut b = packed_b;

    for _ in 0..kc {
        // 16 floats of B in 4 vregs.
        let b0 = vld1q_f32(b);
        let b1 = vld1q_f32(b.add(4));
        let b2 = vld1q_f32(b.add(8));
        let b3 = vld1q_f32(b.add(12));

        // 4 floats of A (one column of the strip) in one vreg.
        let av = vld1q_f32(a);

        // Row 0: lane 0 of av is A[0,k].
        c00 = vfmaq_laneq_f32::<0>(c00, b0, av);
        c01 = vfmaq_laneq_f32::<0>(c01, b1, av);
        c02 = vfmaq_laneq_f32::<0>(c02, b2, av);
        c03 = vfmaq_laneq_f32::<0>(c03, b3, av);

        // Row 1: lane 1.
        c10 = vfmaq_laneq_f32::<1>(c10, b0, av);
        c11 = vfmaq_laneq_f32::<1>(c11, b1, av);
        c12 = vfmaq_laneq_f32::<1>(c12, b2, av);
        c13 = vfmaq_laneq_f32::<1>(c13, b3, av);

        // Row 2: lane 2.
        c20 = vfmaq_laneq_f32::<2>(c20, b0, av);
        c21 = vfmaq_laneq_f32::<2>(c21, b1, av);
        c22 = vfmaq_laneq_f32::<2>(c22, b2, av);
        c23 = vfmaq_laneq_f32::<2>(c23, b3, av);

        // Row 3: lane 3.
        c30 = vfmaq_laneq_f32::<3>(c30, b0, av);
        c31 = vfmaq_laneq_f32::<3>(c31, b1, av);
        c32 = vfmaq_laneq_f32::<3>(c32, b2, av);
        c33 = vfmaq_laneq_f32::<3>(c33, b3, av);

        a = a.add(MR_NEON_F32);
        b = b.add(NR_NEON_F32);
    }

    if alpha != 1.0 {
        let alpha_v = vdupq_n_f32(alpha);
        c00 = vmulq_f32(c00, alpha_v);
        c01 = vmulq_f32(c01, alpha_v);
        c02 = vmulq_f32(c02, alpha_v);
        c03 = vmulq_f32(c03, alpha_v);
        c10 = vmulq_f32(c10, alpha_v);
        c11 = vmulq_f32(c11, alpha_v);
        c12 = vmulq_f32(c12, alpha_v);
        c13 = vmulq_f32(c13, alpha_v);
        c20 = vmulq_f32(c20, alpha_v);
        c21 = vmulq_f32(c21, alpha_v);
        c22 = vmulq_f32(c22, alpha_v);
        c23 = vmulq_f32(c23, alpha_v);
        c30 = vmulq_f32(c30, alpha_v);
        c31 = vmulq_f32(c31, alpha_v);
        c32 = vmulq_f32(c32, alpha_v);
        c33 = vmulq_f32(c33, alpha_v);
    }

    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_16(c, 0, ldc, c00, c01, c02, c03, beta);
        store_row_16(c, 1, ldc, c10, c11, c12, c13, beta);
        store_row_16(c, 2, ldc, c20, c21, c22, c23, beta);
        store_row_16(c, 3, ldc, c30, c31, c32, c33, beta);
    } else {
        store_strided_row_16(c, c_rs, c_cs, 0, c00, c01, c02, c03, beta);
        store_strided_row_16(c, c_rs, c_cs, 1, c10, c11, c12, c13, beta);
        store_strided_row_16(c, c_rs, c_cs, 2, c20, c21, c22, c23, beta);
        store_strided_row_16(c, c_rs, c_cs, 3, c30, c31, c32, c33, beta);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_row_16(
    c: *mut f32,
    row: usize,
    ldc: usize,
    v0: float32x4_t,
    v1: float32x4_t,
    v2: float32x4_t,
    v3: float32x4_t,
    beta: f32,
) {
    let p = c.add(row * ldc);
    if beta == 0.0 {
        vst1q_f32(p, v0);
        vst1q_f32(p.add(4), v1);
        vst1q_f32(p.add(8), v2);
        vst1q_f32(p.add(12), v3);
    } else if beta == 1.0 {
        let cur0 = vld1q_f32(p);
        let cur1 = vld1q_f32(p.add(4));
        let cur2 = vld1q_f32(p.add(8));
        let cur3 = vld1q_f32(p.add(12));
        vst1q_f32(p, vaddq_f32(cur0, v0));
        vst1q_f32(p.add(4), vaddq_f32(cur1, v1));
        vst1q_f32(p.add(8), vaddq_f32(cur2, v2));
        vst1q_f32(p.add(12), vaddq_f32(cur3, v3));
    } else {
        let beta_v = vdupq_n_f32(beta);
        let cur0 = vld1q_f32(p);
        let cur1 = vld1q_f32(p.add(4));
        let cur2 = vld1q_f32(p.add(8));
        let cur3 = vld1q_f32(p.add(12));
        vst1q_f32(p, vfmaq_f32(v0, cur0, beta_v));
        vst1q_f32(p.add(4), vfmaq_f32(v1, cur1, beta_v));
        vst1q_f32(p.add(8), vfmaq_f32(v2, cur2, beta_v));
        vst1q_f32(p.add(12), vfmaq_f32(v3, cur3, beta_v));
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_strided_row_16(
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    v0: float32x4_t,
    v1: float32x4_t,
    v2: float32x4_t,
    v3: float32x4_t,
    beta: f32,
) {
    let mut buf = [0.0_f32; 16];
    vst1q_f32(buf.as_mut_ptr(), v0);
    vst1q_f32(buf.as_mut_ptr().add(4), v1);
    vst1q_f32(buf.as_mut_ptr().add(8), v2);
    vst1q_f32(buf.as_mut_ptr().add(12), v3);
    let row_off = (row as isize) * c_rs;
    for j in 0..16 {
        let p = c.offset(row_off + (j as isize) * c_cs);
        let v = buf[j];
        if beta == 0.0 {
            *p = v;
        } else if beta == 1.0 {
            *p += v;
        } else {
            *p = beta * *p + v;
        }
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_neon_f32(
    m: usize,
    n: usize,
    kc: usize,
    alpha: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    debug_assert!(m <= MR_NEON_F32);
    debug_assert!(n <= NR_NEON_F32);
    let mut tile = [[0.0_f32; NR_NEON_F32]; MR_NEON_F32];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i);
            for j in 0..n {
                tile[i][j] = ai.mul_add(*b.add(j), tile[i][j]);
            }
        }
        a = a.add(MR_NEON_F32);
        b = b.add(NR_NEON_F32);
    }
    for i in 0..m {
        for j in 0..n {
            let p = c.offset((i as isize) * c_rs + (j as isize) * c_cs);
            let v = alpha * tile[i][j];
            if beta == 0.0 {
                *p = v;
            } else if beta == 1.0 {
                *p += v;
            } else {
                *p = beta * *p + v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] += acc;
            }
        }
    }

    #[test]
    fn kernel_4x16_neon_matches_naive() {
        let kc = 32;
        let a: Vec<f32> = (0..MR_NEON_F32 * kc).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let b: Vec<f32> = (0..kc * NR_NEON_F32).map(|i| (i as f32) * 0.013 + 0.5).collect();
        let mut c_ours = vec![0.0_f32; MR_NEON_F32 * NR_NEON_F32];
        let mut c_ref = vec![0.0_f32; MR_NEON_F32 * NR_NEON_F32];

        // pack A: kc-major, MR-minor.
        let mut pa = vec![0.0_f32; MR_NEON_F32 * kc];
        for i in 0..MR_NEON_F32 {
            for k in 0..kc {
                pa[k * MR_NEON_F32 + i] = a[i * kc + k];
            }
        }
        // pack B: kc-major, NR-minor (already row-major).
        let pb = b.clone();

        unsafe {
            kernel_4x16_neon_f32(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                NR_NEON_F32 as isize,
                1,
            );
        }
        naive_gemm(MR_NEON_F32, NR_NEON_F32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_NEON_F32 * NR_NEON_F32 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(diff < 1e-3, "mismatch at {i}: ours={} ref={} diff={}", c_ours[i], c_ref[i], diff);
        }
    }
}
