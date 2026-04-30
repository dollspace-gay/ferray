// AVX2+FMA DGEMM micro-kernel — translated from OpenBLAS's
// `kernel/x86_64/dgemm_kernel_4x8_haswell.S` (BSD-3 licensed). Register
// tile is 4 rows x 8 cols. With BROADCASTKERNEL semantics each ymm
// accumulator holds one row's 4 contiguous columns, so for row-major C
// output we drop OpenBLAS's column-major transpose in SAVE and store
// rows directly.
//
// Register usage (16 ymm available, we use 11):
//   ymm0       broadcast scratch (current A[i, k])
//   ymm1, ymm2 current B row, split: ymm1 = B[k, 0..4], ymm2 = B[k, 4..8]
//   ymm4..ymm7   accumulators for cols 0..4: ymm{4+i} = C[row i, cols 0..4]
//   ymm8..ymm11  accumulators for cols 4..8: ymm{8+i} = C[row i, cols 4..8]
//
// Per K iteration: 2 loads (B row, 8 doubles), 4 broadcasts (A column, 4
// doubles), 8 FMAs. Haswell has 2 FMA units (1 cycle latency-hidden each)
// → 8 FMAs / 2 = 4 cycles per K iter, yielding 64 FLOPs / 4 cycles =
// 16 FLOPs/cycle, matching AVX2-FMA peak.
//
// Packing layout matches OpenBLAS's `gemm_ncopy_4` and `dgemm_ncopy_8`
// semantics (with row-major adapted via `pack.rs` rather than ncopy/itcopy):
//   packed_a[k * MR + i] = A[row + i, depth + k]   (MR-tall column strips)
//   packed_b[k * NR + j] = B[depth + k, col + j]   (NR-wide row strips)
//
// Source: OpenBLAS commit history, `kernel/x86_64/dgemm_kernel_4x8_haswell.S`,
// macros INIT4x8, KERNEL4x8_SUB, KERNEL4x8_M1, KERNEL4x8_M2, SAVE4x8.

use core::arch::x86_64::{
    __m256d, _MM_HINT_T0, _mm_prefetch, _mm256_add_pd, _mm256_broadcast_sd, _mm256_fmadd_pd,
    _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
};

pub const MR: usize = 4;
pub const NR: usize = 8;

// Prefetch distances borrowed from OpenBLAS (line 108-109 of source):
//   #define A_PR1 512
//   #define B_PR1 160
// 512 bytes ahead in A = 64 doubles = 16 K-iters at MR=4.
// 160 bytes ahead in B = 20 doubles = 2.5 K-iters at NR=8 (rounded down).
const A_PF_DOUBLES: usize = 512 / 8;
const B_PF_DOUBLES: usize = 160 / 8;

/// 4x8 micro-kernel. Computes the 4-row by 8-col tile of C from kc depth
/// of packed A and B. SAFETY: caller guarantees AVX2+FMA availability,
/// `packed_a >= kc * MR doubles`, `packed_b >= kc * NR doubles`, and `c`
/// addresses a (m_tile, n_tile) sub-tile of the destination.
///
/// `c_rs` is row stride of C in doubles, `c_cs` is column stride. For
/// row-major C with leading dim ldc, pass c_rs=ldc, c_cs=1 — that hits
/// the fast vector-store path. Other layouts go through scalar scatter.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn kernel_8x6(
    kc: usize,
    alpha: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    // INIT4x8 — zero all 8 accumulators.
    let mut c00 = _mm256_setzero_pd(); // row 0, cols 0..4
    let mut c01 = _mm256_setzero_pd(); // row 0, cols 4..8
    let mut c10 = _mm256_setzero_pd(); // row 1, cols 0..4
    let mut c11 = _mm256_setzero_pd(); // row 1, cols 4..8
    let mut c20 = _mm256_setzero_pd(); // row 2, cols 0..4
    let mut c21 = _mm256_setzero_pd(); // row 2, cols 4..8
    let mut c30 = _mm256_setzero_pd(); // row 3, cols 0..4
    let mut c31 = _mm256_setzero_pd(); // row 3, cols 4..8

    // Main K loop, hand-pipelined as M1/M2 pairs (KERNEL4x8_M1 +
    // KERNEL4x8_M2 in OpenBLAS source). Within each pair the B-loads for
    // iter K+1 are interleaved with the FMAs of iter K, hiding load
    // latency behind FMA throughput. Two-iter unrolling lets the compiler
    // schedule loads against the dual-FMA ports without us having to spell
    // out the order (which was OpenBLAS's reason for splitting M1/M2 in
    // assembly — they couldn't rely on a scheduler).
    let mut a = packed_a;
    let mut b = packed_b;

    // Pre-load B for iteration 0 outside the loop body (matches
    // KERNEL4x8_I in OpenBLAS — the "initial" macro that primes the
    // pipeline before the M1/M2 sequence).
    let mut b_lo = _mm256_loadu_pd(b);
    let mut b_hi = _mm256_loadu_pd(b.add(4));

    let kc_pairs = kc / 2;
    let kc_tail = kc % 2;

    for _ in 0..kc_pairs {
        // ----- M1: process current K, pre-load next K's B at the end -----
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_DOUBLES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_DOUBLES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_DOUBLES + 8).cast());

        let a0 = _mm256_broadcast_sd(&*a);
        c00 = _mm256_fmadd_pd(a0, b_lo, c00);
        c01 = _mm256_fmadd_pd(a0, b_hi, c01);

        let a1 = _mm256_broadcast_sd(&*a.add(1));
        c10 = _mm256_fmadd_pd(a1, b_lo, c10);
        c11 = _mm256_fmadd_pd(a1, b_hi, c11);

        let a2 = _mm256_broadcast_sd(&*a.add(2));
        c20 = _mm256_fmadd_pd(a2, b_lo, c20);
        c21 = _mm256_fmadd_pd(a2, b_hi, c21);

        let a3 = _mm256_broadcast_sd(&*a.add(3));
        c30 = _mm256_fmadd_pd(a3, b_lo, c30);
        c31 = _mm256_fmadd_pd(a3, b_hi, c31);

        // Pre-load B for the NEXT K iteration (the M2 below will use it).
        // Mirrors OpenBLAS lines 889-891: vmovups -12*SIZE(BO), %ymm1 and
        // vmovups -8*SIZE(BO), %ymm2 issued at the tail of M1.
        b_lo = _mm256_loadu_pd(b.add(NR));
        b_hi = _mm256_loadu_pd(b.add(NR + 4));

        // ----- M2: process K+1, pre-load K+2's B at the end -----
        let a0_2 = _mm256_broadcast_sd(&*a.add(MR));
        c00 = _mm256_fmadd_pd(a0_2, b_lo, c00);
        c01 = _mm256_fmadd_pd(a0_2, b_hi, c01);

        let a1_2 = _mm256_broadcast_sd(&*a.add(MR + 1));
        c10 = _mm256_fmadd_pd(a1_2, b_lo, c10);
        c11 = _mm256_fmadd_pd(a1_2, b_hi, c11);

        let a2_2 = _mm256_broadcast_sd(&*a.add(MR + 2));
        c20 = _mm256_fmadd_pd(a2_2, b_lo, c20);
        c21 = _mm256_fmadd_pd(a2_2, b_hi, c21);

        let a3_2 = _mm256_broadcast_sd(&*a.add(MR + 3));
        c30 = _mm256_fmadd_pd(a3_2, b_lo, c30);
        c31 = _mm256_fmadd_pd(a3_2, b_hi, c31);

        // Advance pointers by 2 K iters and pre-load for next M1.
        a = a.add(2 * MR);
        b = b.add(2 * NR);
        b_lo = _mm256_loadu_pd(b);
        b_hi = _mm256_loadu_pd(b.add(4));
    }

    // Odd-K tail (only one iteration left to consume the pre-loaded B).
    if kc_tail == 1 {
        let a0 = _mm256_broadcast_sd(&*a);
        c00 = _mm256_fmadd_pd(a0, b_lo, c00);
        c01 = _mm256_fmadd_pd(a0, b_hi, c01);

        let a1 = _mm256_broadcast_sd(&*a.add(1));
        c10 = _mm256_fmadd_pd(a1, b_lo, c10);
        c11 = _mm256_fmadd_pd(a1, b_hi, c11);

        let a2 = _mm256_broadcast_sd(&*a.add(2));
        c20 = _mm256_fmadd_pd(a2, b_lo, c20);
        c21 = _mm256_fmadd_pd(a2, b_hi, c21);

        let a3 = _mm256_broadcast_sd(&*a.add(3));
        c30 = _mm256_fmadd_pd(a3, b_lo, c30);
        c31 = _mm256_fmadd_pd(a3, b_hi, c31);
    }

    // SAVE4x8 — apply alpha. Skip when alpha == 1.0 (the common case from
    // the BLAS contract).
    if alpha != 1.0 {
        let alpha_v = _mm256_set1_pd(alpha);
        c00 = _mm256_mul_pd(c00, alpha_v);
        c01 = _mm256_mul_pd(c01, alpha_v);
        c10 = _mm256_mul_pd(c10, alpha_v);
        c11 = _mm256_mul_pd(c11, alpha_v);
        c20 = _mm256_mul_pd(c20, alpha_v);
        c21 = _mm256_mul_pd(c21, alpha_v);
        c30 = _mm256_mul_pd(c30, alpha_v);
        c31 = _mm256_mul_pd(c31, alpha_v);
    }

    // Store. Row-major fast path: each ymm is one row's 4 contiguous
    // columns, so two `vmovups` per row writes the full 8-col tile. With
    // beta=1.0 we add the existing C row first (same as OpenBLAS's
    // `vaddpd (CO1), %ymm4, %ymm4` then `vmovups %ymm4, (CO1)`).
    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_8(c, 0, ldc, c00, c01, beta);
        store_row_8(c, 1, ldc, c10, c11, beta);
        store_row_8(c, 2, ldc, c20, c21, beta);
        store_row_8(c, 3, ldc, c30, c31, beta);
    } else {
        store_strided_row_8(c, c_rs, c_cs, 0, c00, c01, beta);
        store_strided_row_8(c, c_rs, c_cs, 1, c10, c11, beta);
        store_strided_row_8(c, c_rs, c_cs, 2, c20, c21, beta);
        store_strided_row_8(c, c_rs, c_cs, 3, c30, c31, beta);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_row_8(c: *mut f64, row: usize, ldc: usize, lo: __m256d, hi: __m256d, beta: f64) {
    let p = c.add(row * ldc);
    if beta == 0.0 {
        _mm256_storeu_pd(p, lo);
        _mm256_storeu_pd(p.add(4), hi);
    } else if beta == 1.0 {
        // OpenBLAS pattern: vaddpd (mem), acc, acc; vmovups acc, (mem).
        let cur_lo = _mm256_loadu_pd(p);
        let cur_hi = _mm256_loadu_pd(p.add(4));
        _mm256_storeu_pd(p, _mm256_add_pd(cur_lo, lo));
        _mm256_storeu_pd(p.add(4), _mm256_add_pd(cur_hi, hi));
    } else {
        let beta_v = _mm256_set1_pd(beta);
        let cur_lo = _mm256_loadu_pd(p);
        let cur_hi = _mm256_loadu_pd(p.add(4));
        _mm256_storeu_pd(p, _mm256_fmadd_pd(cur_lo, beta_v, lo));
        _mm256_storeu_pd(p.add(4), _mm256_fmadd_pd(cur_hi, beta_v, hi));
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_strided_row_8(
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    lo: __m256d,
    hi: __m256d,
    beta: f64,
) {
    let mut buf = [0.0_f64; 8];
    _mm256_storeu_pd(buf.as_mut_ptr(), lo);
    _mm256_storeu_pd(buf.as_mut_ptr().add(4), hi);
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
/// path used at right and bottom edges of the matrix; only runs O(M+N)
/// extra work, not O(M*N), so the slowdown is negligible.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge(
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
    debug_assert!(m <= MR);
    debug_assert!(n <= NR);
    let mut tile = [[0.0_f64; NR]; MR];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i);
            for j in 0..n {
                tile[i][j] = ai.mul_add(*b.add(j), tile[i][j]);
            }
        }
        a = a.add(MR);
        b = b.add(NR);
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
        let mut packed = vec![0.0_f64; MR * kc];
        for i in 0..MR {
            for k in 0..kc {
                packed[k * MR + i] = a[i * kc + k];
            }
        }
        packed
    }

    fn pack_b(b: &[f64], kc: usize) -> Vec<f64> {
        let mut packed = vec![0.0_f64; kc * NR];
        for k in 0..kc {
            for j in 0..NR {
                packed[k * NR + j] = b[k * NR + j];
            }
        }
        packed
    }

    #[test]
    fn kernel_4x8_matches_naive() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 32;
        let a: Vec<f64> = (0..MR * kc).map(|i| (i as f64) * 0.01 - 1.0).collect();
        let b: Vec<f64> = (0..kc * NR).map(|i| (i as f64) * 0.013 + 0.5).collect();
        let mut c_ours = vec![0.0_f64; MR * NR];
        let mut c_ref = vec![0.0_f64; MR * NR];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_8x6(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                NR as isize,
                1,
            );
        }
        naive_gemm(MR, NR, kc, &a, &b, &mut c_ref);

        for i in 0..MR * NR {
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
    fn kernel_4x8_beta_one_accumulates() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 16;
        let a: Vec<f64> = (0..MR * kc).map(|i| (i as f64) * 0.02).collect();
        let b: Vec<f64> = (0..kc * NR).map(|i| (i as f64) * 0.03).collect();
        let mut c_ours = vec![1.5_f64; MR * NR];
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_8x6(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                1.0,
                c_ours.as_mut_ptr(),
                NR as isize,
                1,
            );
        }
        naive_gemm(MR, NR, kc, &a, &b, &mut c_ref);

        for i in 0..MR * NR {
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
