// AVX-512F SGEMM micro-kernel — translated from OpenBLAS's
// `kernel/x86_64/sgemm_kernel_16x4_skylakex_3.c` (BSD-3 licensed).
// OpenBLAS uses a 16-rows x N-cols column-oriented register tile with
// the "double-broadcast" trick (vmovsldup/vmovshdup of A, vbroadcastsd
// of two-float B pairs) yielding interleaved partial products that get
// un-shuffled at SAVE for column-major C output. For our row-major
// contract, the row-oriented 4-rows x 32-cols layout (parallel to the
// AVX2 4x16 SGEMM and AVX-512 4x16 DGEMM kernels) avoids the SAVE
// shuffles entirely.
//
// Register usage (32 zmm available on AVX-512, we use 11):
//   zmm0       broadcast scratch (current A[i, k])
//   zmm1, zmm2 current B row, split: zmm1 = B[k, 0..16], zmm2 = B[k, 16..32]
//   zmm4..zmm7   accumulators for cols 0..16: zmm{4+i} = C[row i, cols 0..16]
//   zmm8..zmm11  accumulators for cols 16..32: zmm{8+i} = C[row i, cols 16..32]
//
// Per K iteration: 2 zmm loads (B row, 32 floats), 4 broadcasts (A
// column, 4 floats), 8 FMAs. Each AVX-512 FMA processes 16 floats vs
// AVX2's 8, so peak FLOPs/cycle doubles vs the AVX2 SGEMM. With 2 FMA
// units, 8 FMAs / 2 = 4 cycles per K-iter, 256 FLOPs / 4 cycles =
// 64 FLOPs/cycle = AVX-512 SGEMM peak (twice the DGEMM AVX-512 peak,
// twice the SGEMM AVX2 peak).
//
// Packing layout matches the AVX2 SGEMM kernel for A (MR=4 unchanged)
// and uses a wider NR=32 strip for B (vs AVX2's NR_F32=16).

use core::arch::x86_64::{
    __m512, _MM_HINT_T0, _mm_prefetch, _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
    _mm512_mul_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
};

pub const MR_AVX512_F32: usize = 4;
pub const NR_AVX512_F32: usize = 32;

const A_PF_FLOATS: usize = 512 / 4;
const B_PF_FLOATS: usize = 160 / 4;

/// 4x32 AVX-512 SGEMM micro-kernel.
///
/// SAFETY: caller guarantees AVX-512F availability, `packed_a >=
/// kc * MR_AVX512_F32 floats`, `packed_b >= kc * NR_AVX512_F32 floats`,
/// and `c` addresses a valid (m_tile, n_tile) sub-tile.
///
/// `c_rs`/`c_cs` are row/column strides of C in floats. Row-major C
/// with `c_cs == 1` hits the fast vector-store path.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn kernel_4x32_avx512(
    kc: usize,
    alpha: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    let mut c00 = _mm512_setzero_ps(); // row 0, cols 0..16
    let mut c01 = _mm512_setzero_ps(); // row 0, cols 16..32
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();
    let mut c20 = _mm512_setzero_ps();
    let mut c21 = _mm512_setzero_ps();
    let mut c30 = _mm512_setzero_ps();
    let mut c31 = _mm512_setzero_ps();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_FLOATS).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_FLOATS).cast());

        // Load 32 floats of B[k, 0..32] into two zmm.
        let b_lo = _mm512_loadu_ps(b);
        let b_hi = _mm512_loadu_ps(b.add(16));

        // Row 0: broadcast A[0,k], FMA into both zmm halves of row 0.
        let a0 = _mm512_set1_ps(*a);
        c00 = _mm512_fmadd_ps(a0, b_lo, c00);
        c01 = _mm512_fmadd_ps(a0, b_hi, c01);

        // Row 1.
        let a1 = _mm512_set1_ps(*a.add(1));
        c10 = _mm512_fmadd_ps(a1, b_lo, c10);
        c11 = _mm512_fmadd_ps(a1, b_hi, c11);

        // Row 2.
        let a2 = _mm512_set1_ps(*a.add(2));
        c20 = _mm512_fmadd_ps(a2, b_lo, c20);
        c21 = _mm512_fmadd_ps(a2, b_hi, c21);

        // Row 3.
        let a3 = _mm512_set1_ps(*a.add(3));
        c30 = _mm512_fmadd_ps(a3, b_lo, c30);
        c31 = _mm512_fmadd_ps(a3, b_hi, c31);

        a = a.add(MR_AVX512_F32);
        b = b.add(NR_AVX512_F32);
    }

    if alpha != 1.0 {
        let alpha_v = _mm512_set1_ps(alpha);
        c00 = _mm512_mul_ps(c00, alpha_v);
        c01 = _mm512_mul_ps(c01, alpha_v);
        c10 = _mm512_mul_ps(c10, alpha_v);
        c11 = _mm512_mul_ps(c11, alpha_v);
        c20 = _mm512_mul_ps(c20, alpha_v);
        c21 = _mm512_mul_ps(c21, alpha_v);
        c30 = _mm512_mul_ps(c30, alpha_v);
        c31 = _mm512_mul_ps(c31, alpha_v);
    }

    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_32(c, 0, ldc, c00, c01, beta);
        store_row_32(c, 1, ldc, c10, c11, beta);
        store_row_32(c, 2, ldc, c20, c21, beta);
        store_row_32(c, 3, ldc, c30, c31, beta);
    } else {
        store_strided_row_32(c, c_rs, c_cs, 0, c00, c01, beta);
        store_strided_row_32(c, c_rs, c_cs, 1, c10, c11, beta);
        store_strided_row_32(c, c_rs, c_cs, 2, c20, c21, beta);
        store_strided_row_32(c, c_rs, c_cs, 3, c30, c31, beta);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_row_32(c: *mut f32, row: usize, ldc: usize, lo: __m512, hi: __m512, beta: f32) {
    let p = c.add(row * ldc);
    if beta == 0.0 {
        _mm512_storeu_ps(p, lo);
        _mm512_storeu_ps(p.add(16), hi);
    } else if beta == 1.0 {
        let cur_lo = _mm512_loadu_ps(p);
        let cur_hi = _mm512_loadu_ps(p.add(16));
        _mm512_storeu_ps(p, _mm512_add_ps(cur_lo, lo));
        _mm512_storeu_ps(p.add(16), _mm512_add_ps(cur_hi, hi));
    } else {
        let beta_v = _mm512_set1_ps(beta);
        let cur_lo = _mm512_loadu_ps(p);
        let cur_hi = _mm512_loadu_ps(p.add(16));
        _mm512_storeu_ps(p, _mm512_fmadd_ps(cur_lo, beta_v, lo));
        _mm512_storeu_ps(p.add(16), _mm512_fmadd_ps(cur_hi, beta_v, hi));
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_strided_row_32(
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    lo: __m512,
    hi: __m512,
    beta: f32,
) {
    let mut buf = [0.0_f32; 32];
    _mm512_storeu_ps(buf.as_mut_ptr(), lo);
    _mm512_storeu_ps(buf.as_mut_ptr().add(16), hi);
    let row_off = (row as isize) * c_rs;
    for j in 0..32 {
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

/// Edge-case kernel for tail tiles where m < MR_AVX512_F32 or
/// n < NR_AVX512_F32.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_avx512_f32(
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
    debug_assert!(m <= MR_AVX512_F32);
    debug_assert!(n <= NR_AVX512_F32);
    let mut tile = [[0.0_f32; NR_AVX512_F32]; MR_AVX512_F32];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ai = *a.add(i);
            for j in 0..n {
                tile[i][j] = ai.mul_add(*b.add(j), tile[i][j]);
            }
        }
        a = a.add(MR_AVX512_F32);
        b = b.add(NR_AVX512_F32);
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
        let mut packed = vec![0.0_f32; MR_AVX512_F32 * kc];
        for i in 0..MR_AVX512_F32 {
            for k in 0..kc {
                packed[k * MR_AVX512_F32 + i] = a[i * kc + k];
            }
        }
        packed
    }

    fn pack_b(b: &[f32], kc: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; kc * NR_AVX512_F32];
        for k in 0..kc {
            for j in 0..NR_AVX512_F32 {
                packed[k * NR_AVX512_F32 + j] = b[k * NR_AVX512_F32 + j];
            }
        }
        packed
    }

    /// Skips on AVX2-only CPUs (e.g., consumer Raptor Lake).
    #[test]
    fn kernel_4x32_avx512_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 32;
        let a: Vec<f32> = (0..MR_AVX512_F32 * kc)
            .map(|i| (i as f32) * 0.01 - 1.0)
            .collect();
        let b: Vec<f32> = (0..kc * NR_AVX512_F32)
            .map(|i| (i as f32) * 0.013 + 0.5)
            .collect();
        let mut c_ours = vec![0.0_f32; MR_AVX512_F32 * NR_AVX512_F32];
        let mut c_ref = vec![0.0_f32; MR_AVX512_F32 * NR_AVX512_F32];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x32_avx512(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                c_ours.as_mut_ptr(),
                NR_AVX512_F32 as isize,
                1,
            );
        }
        naive_gemm(MR_AVX512_F32, NR_AVX512_F32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_F32 * NR_AVX512_F32 {
            let diff = (c_ours[i] - c_ref[i]).abs();
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
    fn kernel_4x32_avx512_beta_one_accumulates() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 16;
        let a: Vec<f32> = (0..MR_AVX512_F32 * kc).map(|i| (i as f32) * 0.02).collect();
        let b: Vec<f32> = (0..kc * NR_AVX512_F32).map(|i| (i as f32) * 0.03).collect();
        let mut c_ours = vec![1.5_f32; MR_AVX512_F32 * NR_AVX512_F32];
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x32_avx512(
                kc,
                1.0,
                pa.as_ptr(),
                pb.as_ptr(),
                1.0,
                c_ours.as_mut_ptr(),
                NR_AVX512_F32 as isize,
                1,
            );
        }
        naive_gemm(MR_AVX512_F32, NR_AVX512_F32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_F32 * NR_AVX512_F32 {
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
