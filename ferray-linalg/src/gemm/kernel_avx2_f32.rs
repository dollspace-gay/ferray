// AVX2+FMA SGEMM micro-kernel — design parallel to the DGEMM 4x8
// kernel translated from OpenBLAS dgemm_kernel_4x8_haswell.S, but with
// f32 lanes (8 per ymm vs 4 for f64) and a wider register tile.
//
// OpenBLAS's reference SGEMM Haswell kernel (sgemm_kernel_8x4_haswell_2.c)
// uses a "double-broadcast" trick: vmovsldup/vmovshdup split A into
// even/odd lanes, then a vbroadcastsd of TWO B floats (as one f64) lets
// each ymm pack hold 4 row-pairs * 2 col-pairs = 8 interleaved partial
// products. This matches OpenBLAS's column-major contract but requires
// SAVE-time un-shuffles. For row-major C output (ferray's contract) the
// row-oriented layout we use for DGEMM 4x8 maps more cleanly: each ymm
// accumulator is one row's 8 contiguous columns, two ymm cover one
// 16-wide tile row, no transpose at SAVE.
//
// Register usage (16 ymm available, we use 11):
//   ymm0       broadcast scratch (A[i, k])
//   ymm1, ymm2 current B row, split: ymm1 = B[k, 0..8], ymm2 = B[k, 8..16]
//   ymm4..ymm7   accumulators for cols 0..8: ymm{4+i} = C[row i, cols 0..8]
//   ymm8..ymm11  accumulators for cols 8..16: ymm{8+i} = C[row i, cols 8..16]
//
// Per K iteration: 2 loads (B row, 16 floats), 4 broadcasts (A column,
// 4 floats), 8 FMAs. Haswell has 2 FMA units → 8 FMAs / 2 = 4 cycles per
// K-iter, yielding 128 FLOPs / 4 cycles = 32 FLOPs/cycle = AVX2-FMA peak
// for f32 (twice the f64 peak — 8 lanes per ymm vs 4).
//
// Packing layout (parallel to f64):
//   packed_a[k * MR + i] = A[row + i, depth + k]    (MR-tall column strips)
//   packed_b[k * NR + j] = B[depth + k, col + j]    (NR-wide row strips)

use core::arch::x86_64::{
    __m256, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_broadcast_ss, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

pub const MR_F32: usize = 4;
pub const NR_F32: usize = 16;

// Prefetch distances scaled from OpenBLAS's DGEMM constants. Same byte
// distances (A_PR1=512, B_PR1=160) but for f32 elements:
//   A: 512 bytes / 4 B = 128 floats / MR=4 = 32 K-iters ahead
//   B: 160 bytes / 4 B = 40 floats / NR=16 = 2.5 K-iters (rounded)
const A_PF_FLOATS: usize = 512 / 4;
const B_PF_FLOATS: usize = 160 / 4;

/// 4x16 SGEMM micro-kernel. Computes the 4-row by 16-col tile of C from
/// kc depth of packed A and B.
///
/// SAFETY: caller guarantees AVX2+FMA availability, `packed_a >= kc * MR_F32 floats`,
/// `packed_b >= kc * NR_F32 floats`, and `c` addresses an (m_tile, n_tile) sub-tile.
///
/// `c_rs` is row stride of C in floats, `c_cs` is column stride. For
/// row-major C with leading dim ldc, pass c_rs=ldc, c_cs=1 — that hits
/// the fast vector-store path. Other layouts go through scalar scatter.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn kernel_4x16(
    kc: usize,
    alpha: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    // INIT — zero all 8 accumulators.
    let mut c00 = _mm256_setzero_ps(); // row 0, cols 0..8
    let mut c01 = _mm256_setzero_ps(); // row 0, cols 8..16
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        // A_PR1 / B_PR1 prefetches.
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_FLOATS).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_FLOATS).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_FLOATS + 8).cast());

        // Load 16 floats of B[k, 0..16] into two ymm.
        let b_lo = _mm256_loadu_ps(b);
        let b_hi = _mm256_loadu_ps(b.add(8));

        // Row 0: broadcast A[0,k], FMA into both ymm halves of row 0.
        let a0 = _mm256_broadcast_ss(&*a);
        c00 = _mm256_fmadd_ps(a0, b_lo, c00);
        c01 = _mm256_fmadd_ps(a0, b_hi, c01);

        // Row 1.
        let a1 = _mm256_broadcast_ss(&*a.add(1));
        c10 = _mm256_fmadd_ps(a1, b_lo, c10);
        c11 = _mm256_fmadd_ps(a1, b_hi, c11);

        // Row 2.
        let a2 = _mm256_broadcast_ss(&*a.add(2));
        c20 = _mm256_fmadd_ps(a2, b_lo, c20);
        c21 = _mm256_fmadd_ps(a2, b_hi, c21);

        // Row 3.
        let a3 = _mm256_broadcast_ss(&*a.add(3));
        c30 = _mm256_fmadd_ps(a3, b_lo, c30);
        c31 = _mm256_fmadd_ps(a3, b_hi, c31);

        a = a.add(MR_F32);
        b = b.add(NR_F32);
    }

    // SAVE — apply alpha (skip when alpha == 1.0).
    if alpha != 1.0 {
        let alpha_v = _mm256_set1_ps(alpha);
        c00 = _mm256_mul_ps(c00, alpha_v);
        c01 = _mm256_mul_ps(c01, alpha_v);
        c10 = _mm256_mul_ps(c10, alpha_v);
        c11 = _mm256_mul_ps(c11, alpha_v);
        c20 = _mm256_mul_ps(c20, alpha_v);
        c21 = _mm256_mul_ps(c21, alpha_v);
        c30 = _mm256_mul_ps(c30, alpha_v);
        c31 = _mm256_mul_ps(c31, alpha_v);
    }

    // Store. Row-major fast path: each ymm is one row's 8 contiguous
    // columns, so two `vmovups` per row writes the full 16-col tile.
    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_16(c, 0, ldc, c00, c01, beta);
        store_row_16(c, 1, ldc, c10, c11, beta);
        store_row_16(c, 2, ldc, c20, c21, beta);
        store_row_16(c, 3, ldc, c30, c31, beta);
    } else {
        store_strided_row_16(c, c_rs, c_cs, 0, c00, c01, beta);
        store_strided_row_16(c, c_rs, c_cs, 1, c10, c11, beta);
        store_strided_row_16(c, c_rs, c_cs, 2, c20, c21, beta);
        store_strided_row_16(c, c_rs, c_cs, 3, c30, c31, beta);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_row_16(c: *mut f32, row: usize, ldc: usize, lo: __m256, hi: __m256, beta: f32) {
    let p = c.add(row * ldc);
    if beta == 0.0 {
        _mm256_storeu_ps(p, lo);
        _mm256_storeu_ps(p.add(8), hi);
    } else if beta == 1.0 {
        let cur_lo = _mm256_loadu_ps(p);
        let cur_hi = _mm256_loadu_ps(p.add(8));
        _mm256_storeu_ps(p, _mm256_add_ps(cur_lo, lo));
        _mm256_storeu_ps(p.add(8), _mm256_add_ps(cur_hi, hi));
    } else {
        let beta_v = _mm256_set1_ps(beta);
        let cur_lo = _mm256_loadu_ps(p);
        let cur_hi = _mm256_loadu_ps(p.add(8));
        _mm256_storeu_ps(p, _mm256_fmadd_ps(cur_lo, beta_v, lo));
        _mm256_storeu_ps(p.add(8), _mm256_fmadd_ps(cur_hi, beta_v, hi));
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_strided_row_16(
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    lo: __m256,
    hi: __m256,
    beta: f32,
) {
    let mut buf = [0.0_f32; 16];
    _mm256_storeu_ps(buf.as_mut_ptr(), lo);
    _mm256_storeu_ps(buf.as_mut_ptr().add(8), hi);
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

/// Edge-case kernel for tail tiles where m < MR_F32 or n < NR_F32. Slow
/// scalar path — runs only on `<= MR_F32-1 + NR_F32-1` row/column edges.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_f32(
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
    debug_assert!(m <= MR_F32);
    debug_assert!(n <= NR_F32);
    let mut tile = [[0.0_f32; NR_F32]; MR_F32];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i);
            for j in 0..n {
                tile[i][j] = ai.mul_add(*b.add(j), tile[i][j]);
            }
        }
        a = a.add(MR_F32);
        b = b.add(NR_F32);
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
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] += acc;
            }
        }
    }

    fn pack_a(a: &[f32], kc: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; MR_F32 * kc];
        for i in 0..MR_F32 {
            for k in 0..kc {
                packed[k * MR_F32 + i] = a[i * kc + k];
            }
        }
        packed
    }

    fn pack_b(b: &[f32], kc: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; kc * NR_F32];
        for k in 0..kc {
            for j in 0..NR_F32 {
                packed[k * NR_F32 + j] = b[k * NR_F32 + j];
            }
        }
        packed
    }

    #[test]
    fn kernel_4x16_matches_naive() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 32;
        let a: Vec<f32> = (0..MR_F32 * kc).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let b: Vec<f32> = (0..kc * NR_F32).map(|i| (i as f32) * 0.013 + 0.5).collect();
        let mut c_ours = vec![0.0_f32; MR_F32 * NR_F32];
        let mut c_ref = vec![0.0_f32; MR_F32 * NR_F32];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                NR_F32 as isize,
                1,
            );
        }
        naive_gemm(MR_F32, NR_F32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_F32 * NR_F32 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            // f32 ulp ~6e-8 at unit scale; allow generous tol for accumulation
            assert!(
                diff < 1e-3,
                "mismatch at {i}: ours={} ref={} diff={}",
                c_ours[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn kernel_4x16_beta_one_accumulates() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 16;
        let a: Vec<f32> = (0..MR_F32 * kc).map(|i| (i as f32) * 0.02).collect();
        let b: Vec<f32> = (0..kc * NR_F32).map(|i| (i as f32) * 0.03).collect();
        let mut c_ours = vec![1.5_f32; MR_F32 * NR_F32];
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                1.0,
                c_ours.as_mut_ptr(),
                NR_F32 as isize,
                1,
            );
        }
        naive_gemm(MR_F32, NR_F32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_F32 * NR_F32 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-3,
                "mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_edge_handles_partial() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 8;
        // 3 rows x 13 cols (both partial)
        let m = 3;
        let n = 13;
        let a: Vec<f32> = (0..MR_F32 * kc).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..kc * NR_F32).map(|i| (i as f32) * 0.1).collect();
        let mut c_ours = vec![0.0_f32; m * n];
        let mut c_ref = vec![0.0_f32; m * n];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_edge_f32(
                m,
                n,
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        // Reference: naive m x n GEMM using only the first m rows of A and
        // the first n cols of B (mimicking the edge-tile partial coverage).
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..kc {
                    acc += a[i * kc + p] * b[p * NR_F32 + j];
                }
                c_ref[i * n + j] = acc;
            }
        }
        for i in 0..m * n {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-3,
                "mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }
}
