// AVX2 quantized i8 GEMM micro-kernel — u8 × i8 → i32, the canonical
// quantized ML inference pattern (activations are u8 after ReLU+quant,
// weights are i8). Uses the classical VPMADDUBSW + VPMADDWD chain that
// every i8 BLAS-style implementation (FBGEMM, oneDNN, QNNPACK) builds
// around.
//
// Each AVX2 ymm holds 32 u8 / 32 i8 / 8 i32. The kernel processes a
// 4 K-iter block at a time because:
//   - VPMADDUBSW(ymm_a_u8, ymm_b_i8) takes 32 (u8, i8) pairs and emits
//     16 i16 outputs, each = a[i]*b[i] + a[i+1]*b[i+1] (adjacent-pair sum)
//   - VPMADDWD(ymm_i16, ymm_one_i16) takes 16 i16 and emits 8 i32, each
//     = i16[2j] + i16[2j+1] (adjacent-pair sum)
//   - Together: 32 (u8, i8) pairs reduce to 8 i32 sums of 4 products each
//
// Register tile: 4 rows × 8 cols of i32 output. 4 ymm accumulators, one
// per row. Per K-block (4 K-iters):
//   - 1 ymm load of B (32 bytes = 4 K × 8 cols, packed)
//   - 4 broadcasts of A (one per row, each broadcasts a 4-byte K-quad)
//   - 4 × {vpmaddubsw, vpmaddwd, vpaddd} chains
//
// Per K iteration: ~3 ymm ops per row × 4 rows / 4 = 3 ymm ops/K. Each
// vpmaddubsw is 32 mul-adds and vpmaddwd is 8 mul-adds; total ~88 i8
// ops/K iter for the 4×8 tile = 32 mul-adds (one per output cell) at
// 1 op/cycle throughput → ~32 ops/cycle = 4× FP32 SIMD peak (i8 is
// where x86 quantized inference traditionally beats float).
//
// Packing layouts (must match the kernel's expected order):
//   packed_a: for each MR-row strip, 4-K-block-major:
//     [a[0,k+0..3], a[1,k+0..3], a[2,k+0..3], a[3,k+0..3], <next k-block>...]
//     16 u8 per k-block per strip.
//   packed_b: NR-wide col strips, 4-K-block-major:
//     [b[k+0..3, 0], b[k+0..3, 1], ..., b[k+0..3, 7], <next k-block>...]
//     32 i8 per k-block per strip.
//
// kc must be a multiple of 4 (caller pads K-tail with zeros via the
// pack routines).

use core::arch::x86_64::{
    __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_storeu_si256,
};

pub const MR_I8: usize = 4;
pub const NR_I8: usize = 8;
/// Inner K-block size — VPMADDUBSW+VPMADDWD reduces 4 K iters into the
/// accumulators in one shot.
pub const KB_I8: usize = 4;

const A_PF_BYTES: usize = 512;
const B_PF_BYTES: usize = 512;

/// 4×8 i8 GEMM micro-kernel. Computes C += A @ B where A is u8, B is i8,
/// C is i32. `kc` is the depth in u8/i8 elements (must be a multiple of 4).
///
/// SAFETY: caller guarantees AVX2 availability, `packed_a >= kc * MR_I8 u8s`,
/// `packed_b >= kc * NR_I8 i8s`, and `c` addresses an (m_tile, n_tile)
/// sub-tile of i32. `c_rs` / `c_cs` are row/column strides in i32 elements.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
pub unsafe fn kernel_4x8_i8(
    kc: usize,
    packed_a: *const u8,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(kc % KB_I8 == 0, "kc must be a multiple of KB_I8={KB_I8}");

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    let one_i16 = _mm256_set1_epi16(1);

    let n_blocks = kc / KB_I8;
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..n_blocks {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_BYTES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_BYTES).cast());

        // Load 32 i8 of B = 4 K × 8 cols, packed col-major within the
        // 4-K block (each col occupies 4 contiguous bytes, then the next
        // col, etc.).
        let b_v = _mm256_loadu_si256(b as *const __m256i);

        // Row 0: broadcast A[0, k..k+4] (4 u8 = i32-sized quad).
        // The 4 bytes are at packed_a[0..4] within the current k-block.
        let a0 = _mm256_set1_epi32(*(a as *const i32));
        let p0 = _mm256_maddubs_epi16(a0, b_v); // 16 i16 partial sums
        let p0i32 = _mm256_madd_epi16(p0, one_i16); // 8 i32 partial sums
        acc0 = _mm256_add_epi32(acc0, p0i32);

        // Row 1: 4 bytes ahead in the k-block (one MR-row stride).
        let a1 = _mm256_set1_epi32(*(a.add(KB_I8) as *const i32));
        let p1 = _mm256_maddubs_epi16(a1, b_v);
        let p1i32 = _mm256_madd_epi16(p1, one_i16);
        acc1 = _mm256_add_epi32(acc1, p1i32);

        // Row 2.
        let a2 = _mm256_set1_epi32(*(a.add(2 * KB_I8) as *const i32));
        let p2 = _mm256_maddubs_epi16(a2, b_v);
        let p2i32 = _mm256_madd_epi16(p2, one_i16);
        acc2 = _mm256_add_epi32(acc2, p2i32);

        // Row 3.
        let a3 = _mm256_set1_epi32(*(a.add(3 * KB_I8) as *const i32));
        let p3 = _mm256_maddubs_epi16(a3, b_v);
        let p3i32 = _mm256_madd_epi16(p3, one_i16);
        acc3 = _mm256_add_epi32(acc3, p3i32);

        a = a.add(MR_I8 * KB_I8);
        b = b.add(NR_I8 * KB_I8);
    }

    // Store. Row-major fast path: each ymm holds one row's 8 i32 cols.
    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_8i32(c, 0, ldc, acc0, accumulate);
        store_row_8i32(c, 1, ldc, acc1, accumulate);
        store_row_8i32(c, 2, ldc, acc2, accumulate);
        store_row_8i32(c, 3, ldc, acc3, accumulate);
    } else {
        store_strided_row_8i32(c, c_rs, c_cs, 0, acc0, accumulate);
        store_strided_row_8i32(c, c_rs, c_cs, 1, acc1, accumulate);
        store_strided_row_8i32(c, c_rs, c_cs, 2, acc2, accumulate);
        store_strided_row_8i32(c, c_rs, c_cs, 3, acc3, accumulate);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_row_8i32(c: *mut i32, row: usize, ldc: usize, val: __m256i, accumulate: bool) {
    let p = c.add(row * ldc) as *mut __m256i;
    if accumulate {
        let cur = _mm256_loadu_si256(p);
        _mm256_storeu_si256(p, _mm256_add_epi32(cur, val));
    } else {
        _mm256_storeu_si256(p, val);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_strided_row_8i32(
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    val: __m256i,
    accumulate: bool,
) {
    let mut buf = [0_i32; 8];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, val);
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

/// Edge-case kernel for tail tiles where m < MR_I8 or n < NR_I8.
/// Scalar fallback that handles arbitrary kc (no 4-multiple requirement).
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_i8(
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
    debug_assert!(m <= MR_I8);
    debug_assert!(n <= NR_I8);
    debug_assert!(kc % KB_I8 == 0);
    let mut tile = [[0_i32; NR_I8]; MR_I8];
    let mut a = packed_a;
    let mut b = packed_b;
    let n_blocks = kc / KB_I8;
    for _ in 0..n_blocks {
        for i in 0..m {
            for kk in 0..KB_I8 {
                let av = *a.add(i * KB_I8 + kk) as i32;
                for j in 0..n {
                    let bv = *b.add(j * KB_I8 + kk) as i32;
                    tile[i][j] = tile[i][j].wrapping_add(av * bv);
                }
            }
        }
        a = a.add(MR_I8 * KB_I8);
        b = b.add(NR_I8 * KB_I8);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_i8gemm(m: usize, n: usize, k: usize, a: &[u8], b: &[i8], c: &mut [i32]) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0_i32;
                for p in 0..k {
                    acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
                }
                c[i * n + j] = c[i * n + j].wrapping_add(acc);
            }
        }
    }

    /// Pack A as the kernel expects: for each k-block of KB_I8=4 K-iters,
    /// store 4 K-iters × MR_I8=4 rows interleaved as
    ///   [a[0,k..k+4], a[1,k..k+4], a[2,k..k+4], a[3,k..k+4]]
    /// i.e., per row: 4 contiguous u8s for the K-block.
    fn pack_a(a: &[u8], kc: usize) -> Vec<u8> {
        let mut packed = vec![0_u8; MR_I8 * kc];
        let n_blocks = kc / KB_I8;
        for kb in 0..n_blocks {
            for i in 0..MR_I8 {
                for kk in 0..KB_I8 {
                    packed[kb * MR_I8 * KB_I8 + i * KB_I8 + kk] = a[i * kc + kb * KB_I8 + kk];
                }
            }
        }
        packed
    }

    /// Pack B: for each k-block, 4 K-iters × NR_I8=8 cols, ordered as
    ///   [b[k..k+4, 0], b[k..k+4, 1], ..., b[k..k+4, 7]]
    /// (col-major within each k-block, K contiguous within each col).
    fn pack_b(b: &[i8], kc: usize) -> Vec<i8> {
        let mut packed = vec![0_i8; NR_I8 * kc];
        let n_blocks = kc / KB_I8;
        for kb in 0..n_blocks {
            for j in 0..NR_I8 {
                for kk in 0..KB_I8 {
                    packed[kb * NR_I8 * KB_I8 + j * KB_I8 + kk] = b[(kb * KB_I8 + kk) * NR_I8 + j];
                }
            }
        }
        packed
    }

    #[test]
    fn kernel_4x8_i8_matches_naive() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 16;
        // Use small values to avoid overflow when accumulating products.
        // u8 max=255, i8 max=127, abs(prod) max = 255*127 = ~32k. Over 16
        // K iters that's still <2^20, fine in i32.
        let a: Vec<u8> = (0..MR_I8 * kc)
            .map(|i| ((i * 7 + 3) as u8) & 0x7f)
            .collect();
        let b: Vec<i8> = (0..kc * NR_I8)
            .map(|i| (((i as i32) * 11 - 50) % 100) as i8)
            .collect();
        let mut c_ours = vec![0_i32; MR_I8 * NR_I8];
        let mut c_ref = vec![0_i32; MR_I8 * NR_I8];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_i8(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_I8 as isize,
                1,
                false, // beta = 0 / accumulate = false
            );
        }
        naive_i8gemm(MR_I8, NR_I8, kc, &a, &b, &mut c_ref);

        for i in 0..MR_I8 * NR_I8 {
            assert_eq!(
                c_ours[i], c_ref[i],
                "mismatch at idx {i}: ours={} ref={}",
                c_ours[i], c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_4x8_i8_accumulate() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 8;
        let a: Vec<u8> = (0..MR_I8 * kc).map(|i| ((i * 5) as u8) & 0x3f).collect();
        let b: Vec<i8> = (0..kc * NR_I8)
            .map(|i| ((i as i32 - 10) % 50) as i8)
            .collect();
        // Pre-fill C with non-zero values so accumulate=true matters.
        let mut c_ours: Vec<i32> = (0..MR_I8 * NR_I8).map(|i| i as i32 * 100).collect();
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_i8(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_I8 as isize,
                1,
                true, // accumulate
            );
        }
        naive_i8gemm(MR_I8, NR_I8, kc, &a, &b, &mut c_ref);

        for i in 0..MR_I8 * NR_I8 {
            assert_eq!(
                c_ours[i], c_ref[i],
                "idx {i}: ours={} ref={}",
                c_ours[i], c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_edge_i8_handles_partial() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 8;
        let m = 3;
        let n = 5;
        let a: Vec<u8> = (0..MR_I8 * kc).map(|i| ((i * 3) as u8) & 0x1f).collect();
        let b: Vec<i8> = (0..kc * NR_I8)
            .map(|i| ((i as i32) % 30 - 15) as i8)
            .collect();
        let mut c_ours = vec![0_i32; m * n];
        let mut c_ref = vec![0_i32; m * n];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_edge_i8(
                m,
                n,
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                n as isize,
                1,
                false,
            );
        }
        // Reference: naive m × n × kc using only first m rows of A and
        // first n cols of B.
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0_i32;
                for p in 0..kc {
                    acc = acc.wrapping_add((a[i * kc + p] as i32) * (b[p * NR_I8 + j] as i32));
                }
                c_ref[i * n + j] = acc;
            }
        }
        for i in 0..m * n {
            assert_eq!(
                c_ours[i], c_ref[i],
                "edge mismatch at {i}: ours={} ref={}",
                c_ours[i], c_ref[i]
            );
        }
    }
}
