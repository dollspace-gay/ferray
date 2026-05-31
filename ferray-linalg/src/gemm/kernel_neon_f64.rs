// AArch64 NEON DGEMM micro-kernel — design parallel to the AVX2 4×8
// kernel but tuned for the ARMv8 NEON register file:
//   - 32 v-registers, each 128-bit (vs x86's 16 ymm at 256-bit)
//   - vfmaq_f64: 2 fused multiply-accumulates per instruction (2 lanes)
//   - vfmaq_laneq_f64: indexed FMA, fan-out 1 broadcast lane to 2 dest lanes
//
// Tile shape choice: MR=4, NR=8.
//   - Result tile: 4 rows × 8 cols of f64 = 32 doubles.
//   - In v-regs: each row needs 8 doubles / 2 lanes = 4 v-regs.
//     Total accumulators: 4 rows × 4 v-regs = 16 v-regs.
//   - Plus 4 v-regs for the current B row (loaded once per K).
//   - Plus 1-2 v-regs for A broadcasts (we use vfmaq_laneq_f64 which
//     reads a lane from a vreg — A is loaded as 4 doubles in 2 vregs).
//   - Total live v-regs: ~22 of 32. Comfortable headroom; the compiler
//     spills only on edge stores.
//
// Per K iteration: 4 v-loads of B (8 doubles → 4 vregs of 2 lanes each),
// 2 v-loads of A (4 doubles → 2 vregs), 16 FMAs (4 rows × 4 col-vregs).
// Cortex-A78 / A76 / Neoverse-N1: 2 FP FMA pipes, latency 4, throughput 2/cycle.
// 16 FMAs / 2 = 8 cycles per K iter. 64 FLOPs / 8 cycles = 8 FLOPs/cycle =
// NEON-FMA peak for f64 (4 ops × 2 pipes / cycle).
//
// Packing layout (matches AVX2's pack semantics):
//   packed_a[k * MR + i] = A[row + i, depth + k]   (MR-tall column strips)
//   packed_b[k * NR + j] = B[depth + k, col + j]   (NR-wide row strips)

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vdupq_n_f64, vfmaq_f64, vfmaq_laneq_f64, vld1q_f64, vmulq_f64,
    vst1q_f64,
};

pub const MR_NEON_F64: usize = 4;
pub const NR_NEON_F64: usize = 8;

/// 4×8 NEON DGEMM micro-kernel. Computes the 4-row × 8-col tile of C from
/// kc depth of packed A and B.
///
/// `c_rs` is row stride of C in doubles, `c_cs` is column stride. For
/// row-major C with leading dim ldc, pass c_rs=ldc, c_cs=1 — the fast
/// vector-store path. Other layouts go through scalar scatter.
///
/// SAFETY: caller guarantees aarch64 NEON availability (mandatory in
/// ARMv8), `packed_a >= kc * MR doubles`, `packed_b >= kc * NR doubles`,
/// and `c` addresses an (m_tile, n_tile) sub-tile of the destination.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x8_neon_f64(
    kc: usize,
    alpha: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    // INIT — zero all 16 accumulators (4 rows × 4 vregs each).
    let mut c00 = vdupq_n_f64(0.0); // row 0, cols 0..2
    let mut c01 = vdupq_n_f64(0.0); // row 0, cols 2..4
    let mut c02 = vdupq_n_f64(0.0); // row 0, cols 4..6
    let mut c03 = vdupq_n_f64(0.0); // row 0, cols 6..8
    let mut c10 = vdupq_n_f64(0.0);
    let mut c11 = vdupq_n_f64(0.0);
    let mut c12 = vdupq_n_f64(0.0);
    let mut c13 = vdupq_n_f64(0.0);
    let mut c20 = vdupq_n_f64(0.0);
    let mut c21 = vdupq_n_f64(0.0);
    let mut c22 = vdupq_n_f64(0.0);
    let mut c23 = vdupq_n_f64(0.0);
    let mut c30 = vdupq_n_f64(0.0);
    let mut c31 = vdupq_n_f64(0.0);
    let mut c32 = vdupq_n_f64(0.0);
    let mut c33 = vdupq_n_f64(0.0);

    let mut a = packed_a;
    let mut b = packed_b;

    for _ in 0..kc {
        // Load 8 doubles of B[k, 0..8] = 4 vregs.
        let b0 = vld1q_f64(b);
        let b1 = vld1q_f64(b.add(2));
        let b2 = vld1q_f64(b.add(4));
        let b3 = vld1q_f64(b.add(6));

        // Load 4 doubles of A[0..4, k] = 2 vregs.
        // a01 = [A[0,k], A[1,k]],  a23 = [A[2,k], A[3,k]].
        let a01 = vld1q_f64(a);
        let a23 = vld1q_f64(a.add(2));

        // Row 0: broadcast A[0,k] (lane 0 of a01), FMA into the 4 cols of row 0.
        c00 = vfmaq_laneq_f64::<0>(c00, b0, a01);
        c01 = vfmaq_laneq_f64::<0>(c01, b1, a01);
        c02 = vfmaq_laneq_f64::<0>(c02, b2, a01);
        c03 = vfmaq_laneq_f64::<0>(c03, b3, a01);

        // Row 1: broadcast A[1,k] (lane 1 of a01).
        c10 = vfmaq_laneq_f64::<1>(c10, b0, a01);
        c11 = vfmaq_laneq_f64::<1>(c11, b1, a01);
        c12 = vfmaq_laneq_f64::<1>(c12, b2, a01);
        c13 = vfmaq_laneq_f64::<1>(c13, b3, a01);

        // Row 2: broadcast A[2,k] (lane 0 of a23).
        c20 = vfmaq_laneq_f64::<0>(c20, b0, a23);
        c21 = vfmaq_laneq_f64::<0>(c21, b1, a23);
        c22 = vfmaq_laneq_f64::<0>(c22, b2, a23);
        c23 = vfmaq_laneq_f64::<0>(c23, b3, a23);

        // Row 3: broadcast A[3,k] (lane 1 of a23).
        c30 = vfmaq_laneq_f64::<1>(c30, b0, a23);
        c31 = vfmaq_laneq_f64::<1>(c31, b1, a23);
        c32 = vfmaq_laneq_f64::<1>(c32, b2, a23);
        c33 = vfmaq_laneq_f64::<1>(c33, b3, a23);

        a = a.add(MR_NEON_F64);
        b = b.add(NR_NEON_F64);
    }

    // Apply alpha (skip when alpha == 1.0, the common case).
    if alpha != 1.0 {
        let alpha_v = vdupq_n_f64(alpha);
        c00 = vmulq_f64(c00, alpha_v);
        c01 = vmulq_f64(c01, alpha_v);
        c02 = vmulq_f64(c02, alpha_v);
        c03 = vmulq_f64(c03, alpha_v);
        c10 = vmulq_f64(c10, alpha_v);
        c11 = vmulq_f64(c11, alpha_v);
        c12 = vmulq_f64(c12, alpha_v);
        c13 = vmulq_f64(c13, alpha_v);
        c20 = vmulq_f64(c20, alpha_v);
        c21 = vmulq_f64(c21, alpha_v);
        c22 = vmulq_f64(c22, alpha_v);
        c23 = vmulq_f64(c23, alpha_v);
        c30 = vmulq_f64(c30, alpha_v);
        c31 = vmulq_f64(c31, alpha_v);
        c32 = vmulq_f64(c32, alpha_v);
        c33 = vmulq_f64(c33, alpha_v);
    }

    // Store. Row-major fast path: 8 cols per row = 4 vector stores per row.
    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_8(c, 0, ldc, c00, c01, c02, c03, beta);
        store_row_8(c, 1, ldc, c10, c11, c12, c13, beta);
        store_row_8(c, 2, ldc, c20, c21, c22, c23, beta);
        store_row_8(c, 3, ldc, c30, c31, c32, c33, beta);
    } else {
        store_strided_row_8(c, c_rs, c_cs, 0, c00, c01, c02, c03, beta);
        store_strided_row_8(c, c_rs, c_cs, 1, c10, c11, c12, c13, beta);
        store_strided_row_8(c, c_rs, c_cs, 2, c20, c21, c22, c23, beta);
        store_strided_row_8(c, c_rs, c_cs, 3, c30, c31, c32, c33, beta);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_row_8(
    c: *mut f64,
    row: usize,
    ldc: usize,
    v0: float64x2_t,
    v1: float64x2_t,
    v2: float64x2_t,
    v3: float64x2_t,
    beta: f64,
) {
    let p = c.add(row * ldc);
    if beta == 0.0 {
        vst1q_f64(p, v0);
        vst1q_f64(p.add(2), v1);
        vst1q_f64(p.add(4), v2);
        vst1q_f64(p.add(6), v3);
    } else if beta == 1.0 {
        let cur0 = vld1q_f64(p);
        let cur1 = vld1q_f64(p.add(2));
        let cur2 = vld1q_f64(p.add(4));
        let cur3 = vld1q_f64(p.add(6));
        vst1q_f64(p, vaddq_f64(cur0, v0));
        vst1q_f64(p.add(2), vaddq_f64(cur1, v1));
        vst1q_f64(p.add(4), vaddq_f64(cur2, v2));
        vst1q_f64(p.add(6), vaddq_f64(cur3, v3));
    } else {
        let beta_v = vdupq_n_f64(beta);
        let cur0 = vld1q_f64(p);
        let cur1 = vld1q_f64(p.add(2));
        let cur2 = vld1q_f64(p.add(4));
        let cur3 = vld1q_f64(p.add(6));
        vst1q_f64(p, vfmaq_f64(v0, cur0, beta_v));
        vst1q_f64(p.add(2), vfmaq_f64(v1, cur1, beta_v));
        vst1q_f64(p.add(4), vfmaq_f64(v2, cur2, beta_v));
        vst1q_f64(p.add(6), vfmaq_f64(v3, cur3, beta_v));
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn store_strided_row_8(
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    v0: float64x2_t,
    v1: float64x2_t,
    v2: float64x2_t,
    v3: float64x2_t,
    beta: f64,
) {
    let mut buf = [0.0_f64; 8];
    vst1q_f64(buf.as_mut_ptr(), v0);
    vst1q_f64(buf.as_mut_ptr().add(2), v1);
    vst1q_f64(buf.as_mut_ptr().add(4), v2);
    vst1q_f64(buf.as_mut_ptr().add(6), v3);
    let row_off = (row as isize) * c_rs;
    for j in 0..8 {
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

/// Edge-case kernel for tail tiles where m < MR or n < NR. Slow scalar
/// path used at right and bottom edges of the matrix.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_neon_f64(
    m: usize,
    n: usize,
    kc: usize,
    alpha: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    debug_assert!(m <= MR_NEON_F64);
    debug_assert!(n <= NR_NEON_F64);
    let mut tile = [[0.0_f64; NR_NEON_F64]; MR_NEON_F64];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i);
            for j in 0..n {
                tile[i][j] = ai.mul_add(*b.add(j), tile[i][j]);
            }
        }
        a = a.add(MR_NEON_F64);
        b = b.add(NR_NEON_F64);
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

    fn naive_gemm(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
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

    fn pack_a(a: &[f64], kc: usize) -> Vec<f64> {
        let mut packed = vec![0.0_f64; MR_NEON_F64 * kc];
        for i in 0..MR_NEON_F64 {
            for k in 0..kc {
                packed[k * MR_NEON_F64 + i] = a[i * kc + k];
            }
        }
        packed
    }

    fn pack_b(b: &[f64], kc: usize) -> Vec<f64> {
        let mut packed = vec![0.0_f64; kc * NR_NEON_F64];
        for k in 0..kc {
            for j in 0..NR_NEON_F64 {
                packed[k * NR_NEON_F64 + j] = b[k * NR_NEON_F64 + j];
            }
        }
        packed
    }

    #[test]
    fn kernel_4x8_neon_matches_naive() {
        let kc = 32;
        let a: Vec<f64> = (0..MR_NEON_F64 * kc)
            .map(|i| (i as f64) * 0.01 - 1.0)
            .collect();
        let b: Vec<f64> = (0..kc * NR_NEON_F64)
            .map(|i| (i as f64) * 0.013 + 0.5)
            .collect();
        let mut c_ours = vec![0.0_f64; MR_NEON_F64 * NR_NEON_F64];
        let mut c_ref = vec![0.0_f64; MR_NEON_F64 * NR_NEON_F64];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_neon_f64(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                NR_NEON_F64 as isize,
                1,
            );
        }
        naive_gemm(MR_NEON_F64, NR_NEON_F64, kc, &a, &b, &mut c_ref);

        for i in 0..MR_NEON_F64 * NR_NEON_F64 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-9,
                "mismatch at {i}: ours={} ref={} diff={}",
                c_ours[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn kernel_4x8_neon_beta_one_accumulates() {
        let kc = 16;
        let a: Vec<f64> = (0..MR_NEON_F64 * kc).map(|i| (i as f64) * 0.02).collect();
        let b: Vec<f64> = (0..kc * NR_NEON_F64).map(|i| (i as f64) * 0.03).collect();
        let mut c_ours = vec![1.5_f64; MR_NEON_F64 * NR_NEON_F64];
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_neon_f64(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                1.0,
                c_ours.as_mut_ptr(),
                NR_NEON_F64 as isize,
                1,
            );
        }
        naive_gemm(MR_NEON_F64, NR_NEON_F64, kc, &a, &b, &mut c_ref);

        for i in 0..MR_NEON_F64 * NR_NEON_F64 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-9,
                "mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }
}
