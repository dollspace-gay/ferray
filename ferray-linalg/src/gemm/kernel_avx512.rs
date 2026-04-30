// AVX-512F DGEMM micro-kernel — translated from OpenBLAS's
// `kernel/x86_64/dgemm_kernel_16x2_skylakex.c` (BSD-3 licensed). OpenBLAS
// uses a 16-rows x 12-cols column-oriented register tile that needs
// SAVE-time vunpcklpd/vunpckhpd shuffles to land in column-major C; for
// our row-major contract, the row-oriented 4-rows x 16-cols layout
// (parallel to the AVX2 4x8 DGEMM kernel) avoids those shuffles.
//
// Register usage (32 zmm available on AVX-512, we use 11):
//   zmm0       broadcast scratch (current A[i, k])
//   zmm1, zmm2 current B row, split: zmm1 = B[k, 0..8], zmm2 = B[k, 8..16]
//   zmm4..zmm7   accumulators for cols 0..8: zmm{4+i} = C[row i, cols 0..8]
//   zmm8..zmm11  accumulators for cols 8..16: zmm{8+i} = C[row i, cols 8..16]
//
// Per K iteration: 2 zmm loads (B row, 16 doubles), 4 broadcasts (A
// column, 4 doubles), 8 FMAs. Each AVX-512 FMA processes 8 doubles vs
// AVX2's 4, so peak FLOPs/cycle doubles. With 2 FMA units (port 0 + port 5
// fused on Skylake-X, single 512-bit FMA unit on consumer Skylake-SP),
// 8 FMAs / 2 ports = 4 cycles per K-iter; 128 FLOPs / 4 cycles =
// 32 FLOPs/cycle = AVX-512 DGEMM peak.
//
// On consumer CPUs that fuse port 0+1 into a single 512-bit FMA path
// (most non-server Skylake-X and downscaled SKUs), throughput halves but
// per-iteration density doubles, so total perf still beats AVX2.
//
// Packing layout matches the AVX2 DGEMM kernel:
//   packed_a[k * MR_AVX512 + i] = A[row + i, depth + k]
//   packed_b[k * NR_AVX512 + j] = B[depth + k, col + j]

use core::arch::x86_64::{
    __m512d, _MM_HINT_T0, _mm_prefetch, _mm512_add_pd, _mm512_fmadd_pd, _mm512_loadu_pd,
    _mm512_mul_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd,
};

pub const MR_AVX512: usize = 4;
pub const NR_AVX512: usize = 16;

// Prefetch distances. OpenBLAS uses 384-byte and 448-byte A prefetches
// (lines `prefetcht0 384(%0); prefetcht0 448(%0)`). At MR_AVX512=4 doubles
// per K-iter (32 bytes), 384 bytes = 12 K-iters ahead, 448 bytes = 14.
const A_PF_DOUBLES: usize = 384 / 8;
const B_PF_DOUBLES: usize = 160 / 8;

/// 4x16 AVX-512 DGEMM micro-kernel.
///
/// SAFETY: caller guarantees AVX-512F+FMA availability, `packed_a >=
/// kc * MR_AVX512 doubles`, `packed_b >= kc * NR_AVX512 doubles`, and
/// `c` addresses an (m_tile, n_tile) sub-tile.
///
/// `c_rs`/`c_cs` are row/column strides of C in doubles. Row-major C
/// with `c_cs == 1` hits the fast vector-store path.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn kernel_4x16_avx512(
    kc: usize,
    alpha: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    // INIT — zero all 8 accumulators.
    let mut c00 = _mm512_setzero_pd(); // row 0, cols 0..8
    let mut c01 = _mm512_setzero_pd(); // row 0, cols 8..16
    let mut c10 = _mm512_setzero_pd();
    let mut c11 = _mm512_setzero_pd();
    let mut c20 = _mm512_setzero_pd();
    let mut c21 = _mm512_setzero_pd();
    let mut c30 = _mm512_setzero_pd();
    let mut c31 = _mm512_setzero_pd();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_DOUBLES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_DOUBLES).cast());

        // Load 16 doubles of B[k, 0..16] into two zmm.
        let b_lo = _mm512_loadu_pd(b);
        let b_hi = _mm512_loadu_pd(b.add(8));

        // Row 0: broadcast A[0,k], FMA into both zmm halves of row 0.
        let a0 = _mm512_set1_pd(*a);
        c00 = _mm512_fmadd_pd(a0, b_lo, c00);
        c01 = _mm512_fmadd_pd(a0, b_hi, c01);

        // Row 1.
        let a1 = _mm512_set1_pd(*a.add(1));
        c10 = _mm512_fmadd_pd(a1, b_lo, c10);
        c11 = _mm512_fmadd_pd(a1, b_hi, c11);

        // Row 2.
        let a2 = _mm512_set1_pd(*a.add(2));
        c20 = _mm512_fmadd_pd(a2, b_lo, c20);
        c21 = _mm512_fmadd_pd(a2, b_hi, c21);

        // Row 3.
        let a3 = _mm512_set1_pd(*a.add(3));
        c30 = _mm512_fmadd_pd(a3, b_lo, c30);
        c31 = _mm512_fmadd_pd(a3, b_hi, c31);

        a = a.add(MR_AVX512);
        b = b.add(NR_AVX512);
    }

    // SAVE — apply alpha (skip when alpha == 1.0).
    if alpha != 1.0 {
        let alpha_v = _mm512_set1_pd(alpha);
        c00 = _mm512_mul_pd(c00, alpha_v);
        c01 = _mm512_mul_pd(c01, alpha_v);
        c10 = _mm512_mul_pd(c10, alpha_v);
        c11 = _mm512_mul_pd(c11, alpha_v);
        c20 = _mm512_mul_pd(c20, alpha_v);
        c21 = _mm512_mul_pd(c21, alpha_v);
        c30 = _mm512_mul_pd(c30, alpha_v);
        c31 = _mm512_mul_pd(c31, alpha_v);
    }

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
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_row_16(c: *mut f64, row: usize, ldc: usize, lo: __m512d, hi: __m512d, beta: f64) {
    let p = c.add(row * ldc);
    if beta == 0.0 {
        _mm512_storeu_pd(p, lo);
        _mm512_storeu_pd(p.add(8), hi);
    } else if beta == 1.0 {
        let cur_lo = _mm512_loadu_pd(p);
        let cur_hi = _mm512_loadu_pd(p.add(8));
        _mm512_storeu_pd(p, _mm512_add_pd(cur_lo, lo));
        _mm512_storeu_pd(p.add(8), _mm512_add_pd(cur_hi, hi));
    } else {
        let beta_v = _mm512_set1_pd(beta);
        let cur_lo = _mm512_loadu_pd(p);
        let cur_hi = _mm512_loadu_pd(p.add(8));
        _mm512_storeu_pd(p, _mm512_fmadd_pd(cur_lo, beta_v, lo));
        _mm512_storeu_pd(p.add(8), _mm512_fmadd_pd(cur_hi, beta_v, hi));
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_strided_row_16(
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    lo: __m512d,
    hi: __m512d,
    beta: f64,
) {
    let mut buf = [0.0_f64; 16];
    _mm512_storeu_pd(buf.as_mut_ptr(), lo);
    _mm512_storeu_pd(buf.as_mut_ptr().add(8), hi);
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

/// Edge-case kernel for tail tiles where m < MR_AVX512 or n < NR_AVX512.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_avx512(
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
    debug_assert!(m <= MR_AVX512);
    debug_assert!(n <= NR_AVX512);
    let mut tile = [[0.0_f64; NR_AVX512]; MR_AVX512];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i);
            for j in 0..n {
                tile[i][j] = ai.mul_add(*b.add(j), tile[i][j]);
            }
        }
        a = a.add(MR_AVX512);
        b = b.add(NR_AVX512);
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
        let mut packed = vec![0.0_f64; MR_AVX512 * kc];
        for i in 0..MR_AVX512 {
            for k in 0..kc {
                packed[k * MR_AVX512 + i] = a[i * kc + k];
            }
        }
        packed
    }

    fn pack_b(b: &[f64], kc: usize) -> Vec<f64> {
        let mut packed = vec![0.0_f64; kc * NR_AVX512];
        for k in 0..kc {
            for j in 0..NR_AVX512 {
                packed[k * NR_AVX512 + j] = b[k * NR_AVX512 + j];
            }
        }
        packed
    }

    /// Test only runs when the host CPU supports AVX-512F. On AVX2-only
    /// machines (e.g., consumer Raptor Lake) this is a no-op.
    #[test]
    fn kernel_4x16_avx512_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 32;
        let a: Vec<f64> = (0..MR_AVX512 * kc)
            .map(|i| (i as f64) * 0.01 - 1.0)
            .collect();
        let b: Vec<f64> = (0..kc * NR_AVX512)
            .map(|i| (i as f64) * 0.013 + 0.5)
            .collect();
        let mut c_ours = vec![0.0_f64; MR_AVX512 * NR_AVX512];
        let mut c_ref = vec![0.0_f64; MR_AVX512 * NR_AVX512];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16_avx512(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                NR_AVX512 as isize,
                1,
            );
        }
        naive_gemm(MR_AVX512, NR_AVX512, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512 * NR_AVX512 {
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
    fn kernel_4x16_avx512_beta_one_accumulates() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 16;
        let a: Vec<f64> = (0..MR_AVX512 * kc).map(|i| (i as f64) * 0.02).collect();
        let b: Vec<f64> = (0..kc * NR_AVX512).map(|i| (i as f64) * 0.03).collect();
        let mut c_ours = vec![1.5_f64; MR_AVX512 * NR_AVX512];
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16_avx512(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                1.0,
                c_ours.as_mut_ptr(),
                NR_AVX512 as isize,
                1,
            );
        }
        naive_gemm(MR_AVX512, NR_AVX512, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512 * NR_AVX512 {
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
