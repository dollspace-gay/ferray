// AVX-512 quantized i16 GEMM micro-kernel — i16 × i16 → i32. Uses
// VPMADDWD on zmm registers: each VPMADDWD takes 32 i16 inputs (2K × 16
// cols) and produces 16 i32 outputs in one instruction. Doubles the
// throughput of the AVX2 4×8 i16 kernel.
//
// Register tile: 4 rows × 16 cols of i32 output. 4 zmm accumulators,
// one per row.
//
// Per K-block (2 K iters):
//   - 1 zmm load of B (32 i16 = 2 K × 16 cols, packed col-major within
//     the 2-K block)
//   - 4× VPBROADCASTD of A's 4-byte K-pair (= 2 i16) → 16 i32 lanes per zmm
//   - 4× {VPMADDWD, VPADDD}
//
// AVX-512 VPMADDWD lives in the AVX-512BW feature subset (not pure F),
// so the kernel target-features both `avx512f` and `avx512bw`. Both
// have shipped together on every CPU that supports AVX-512 since
// Skylake-X (the F-only "Knights Mill" ones excluded).
//
// Cfg-gated on the `avx512` feature.
//
// Packing layouts (must match the kernel's expected order):
//   packed_a (i16): per MR-row strip, 2-K-block-major:
//     [a[0, k+0..2], a[1, ..], a[2, ..], a[3, ..], <next>...]
//     8 i16 per k-block per strip.
//   packed_b (i16): per NR-wide col strip, 2-K-block-major:
//     [b[k+0..2, 0], b[k+0..2, 1], ..., b[k+0..2, 15], <next>...]
//     32 i16 per k-block per strip.
//
// kc must be a multiple of 2 (caller pads with zeros via the pack routines).

use core::arch::x86_64::{
    __m512i, _MM_HINT_T0, _mm_prefetch, _mm512_add_epi32, _mm512_loadu_si512, _mm512_madd_epi16,
    _mm512_set1_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
};

pub const MR_AVX512_I16: usize = 4;
/// AVX-512 zmm holds 32 i16, VPMADDWD reduces them to 16 i32 → NR=16.
pub const NR_AVX512_I16: usize = 16;
/// Same K-block size as the AVX2 i16 kernel: VPMADDWD reduces 2 K iters
/// into the i32 accumulators.
pub use super::kernel_avx2_i16::KB_I16;

const A_PF_BYTES: usize = 512;
const B_PF_BYTES: usize = 512;

/// 4×16 AVX-512 i16 GEMM micro-kernel. Computes C += A @ B where A and
/// B are i16, C is i32. `kc` is the depth in i16 elements (must be a
/// multiple of 2).
///
/// SAFETY: caller guarantees AVX-512F + AVX-512BW availability,
/// `packed_a >= kc * MR_AVX512_I16 i16s`, `packed_b >= kc *
/// NR_AVX512_I16 i16s`. `c_rs` / `c_cs` are row/column strides in i32
/// elements.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn kernel_4x16_i16_avx512(
    kc: usize,
    packed_a: *const i16,
    packed_b: *const i16,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(kc % KB_I16 == 0);

    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    let n_blocks = kc / KB_I16;
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..n_blocks {
        _mm_prefetch::<_MM_HINT_T0>(a.cast::<u8>().add(A_PF_BYTES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.cast::<u8>().add(B_PF_BYTES).cast());

        // Load 32 i16 of B = 2 K × 16 cols.
        let b_v = _mm512_loadu_si512(b as *const __m512i);

        // Row 0: broadcast A[0, k..k+2] (2 i16 = 4 bytes = i32-sized pair).
        let a0 = _mm512_set1_epi32(*(a as *const i32));
        let p0 = _mm512_madd_epi16(a0, b_v);
        acc0 = _mm512_add_epi32(acc0, p0);

        // Row 1.
        let a1 = _mm512_set1_epi32(*(a.add(KB_I16) as *const i32));
        let p1 = _mm512_madd_epi16(a1, b_v);
        acc1 = _mm512_add_epi32(acc1, p1);

        // Row 2.
        let a2 = _mm512_set1_epi32(*(a.add(2 * KB_I16) as *const i32));
        let p2 = _mm512_madd_epi16(a2, b_v);
        acc2 = _mm512_add_epi32(acc2, p2);

        // Row 3.
        let a3 = _mm512_set1_epi32(*(a.add(3 * KB_I16) as *const i32));
        let p3 = _mm512_madd_epi16(a3, b_v);
        acc3 = _mm512_add_epi32(acc3, p3);

        a = a.add(MR_AVX512_I16 * KB_I16);
        b = b.add(NR_AVX512_I16 * KB_I16);
    }

    if c_cs == 1 && c_rs > 0 {
        let ldc = c_rs as usize;
        store_row_16i32(c, 0, ldc, acc0, accumulate);
        store_row_16i32(c, 1, ldc, acc1, accumulate);
        store_row_16i32(c, 2, ldc, acc2, accumulate);
        store_row_16i32(c, 3, ldc, acc3, accumulate);
    } else {
        store_strided_row_16i32(c, c_rs, c_cs, 0, acc0, accumulate);
        store_strided_row_16i32(c, c_rs, c_cs, 1, acc1, accumulate);
        store_strided_row_16i32(c, c_rs, c_cs, 2, acc2, accumulate);
        store_strided_row_16i32(c, c_rs, c_cs, 3, acc3, accumulate);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_row_16i32(c: *mut i32, row: usize, ldc: usize, val: __m512i, accumulate: bool) {
    let p = c.add(row * ldc) as *mut __m512i;
    if accumulate {
        let cur = _mm512_loadu_si512(p as *const __m512i);
        _mm512_storeu_si512(p, _mm512_add_epi32(cur, val));
    } else {
        _mm512_storeu_si512(p, val);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_strided_row_16i32(
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    val: __m512i,
    accumulate: bool,
) {
    let mut buf = [0_i32; 16];
    _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, val);
    let row_off = (row as isize) * c_rs;
    for j in 0..16 {
        let p = c.offset(row_off + (j as isize) * c_cs);
        if accumulate {
            *p = (*p).wrapping_add(buf[j]);
        } else {
            *p = buf[j];
        }
    }
}

/// Edge-case kernel for tail tiles. Scalar fallback.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_i16_avx512(
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
    debug_assert!(m <= MR_AVX512_I16);
    debug_assert!(n <= NR_AVX512_I16);
    debug_assert!(kc % KB_I16 == 0);
    let mut tile = [[0_i32; NR_AVX512_I16]; MR_AVX512_I16];
    let mut a = packed_a;
    let mut b = packed_b;
    let n_blocks = kc / KB_I16;
    for _ in 0..n_blocks {
        for i in 0..m {
            for kk in 0..KB_I16 {
                let av = *a.add(i * KB_I16 + kk) as i32;
                for j in 0..n {
                    let bv = *b.add(j * KB_I16 + kk) as i32;
                    tile[i][j] = tile[i][j].wrapping_add(av * bv);
                }
            }
        }
        a = a.add(MR_AVX512_I16 * KB_I16);
        b = b.add(NR_AVX512_I16 * KB_I16);
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

    fn naive_i16gemm(m: usize, n: usize, k: usize, a: &[i16], b: &[i16], c: &mut [i32]) {
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

    fn pack_a(a: &[i16], kc: usize) -> Vec<i16> {
        let mut packed = vec![0_i16; MR_AVX512_I16 * kc];
        let n_blocks = kc / KB_I16;
        for kb in 0..n_blocks {
            for i in 0..MR_AVX512_I16 {
                for kk in 0..KB_I16 {
                    packed[kb * MR_AVX512_I16 * KB_I16 + i * KB_I16 + kk] =
                        a[i * kc + kb * KB_I16 + kk];
                }
            }
        }
        packed
    }

    fn pack_b(b: &[i16], kc: usize) -> Vec<i16> {
        let mut packed = vec![0_i16; NR_AVX512_I16 * kc];
        let n_blocks = kc / KB_I16;
        for kb in 0..n_blocks {
            for j in 0..NR_AVX512_I16 {
                for kk in 0..KB_I16 {
                    packed[kb * NR_AVX512_I16 * KB_I16 + j * KB_I16 + kk] =
                        b[(kb * KB_I16 + kk) * NR_AVX512_I16 + j];
                }
            }
        }
        packed
    }

    /// Skips on hosts without AVX-512BW (e.g., consumer Raptor Lake).
    #[test]
    fn kernel_4x16_i16_avx512_matches_naive() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let kc = 16;
        let a: Vec<i16> = (0..MR_AVX512_I16 * kc)
            .map(|i| ((i as i32) * 13 - 200) as i16)
            .collect();
        let b: Vec<i16> = (0..kc * NR_AVX512_I16)
            .map(|i| ((i as i32) * 17 - 300) as i16)
            .collect();
        let mut c_ours = vec![0_i32; MR_AVX512_I16 * NR_AVX512_I16];
        let mut c_ref = vec![0_i32; MR_AVX512_I16 * NR_AVX512_I16];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16_i16_avx512(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_AVX512_I16 as isize,
                1,
                false,
            );
        }
        naive_i16gemm(MR_AVX512_I16, NR_AVX512_I16, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_I16 * NR_AVX512_I16 {
            assert_eq!(c_ours[i], c_ref[i], "AVX-512 i16 mismatch at idx {i}");
        }
    }

    #[test]
    fn kernel_4x16_i16_avx512_accumulate() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let kc = 8;
        let a: Vec<i16> = (0..MR_AVX512_I16 * kc)
            .map(|i| -((i + 1) as i16) * 7)
            .collect();
        let b: Vec<i16> = (0..kc * NR_AVX512_I16)
            .map(|i| ((i + 5) as i16) * 11)
            .collect();
        let mut c_ours: Vec<i32> = (0..MR_AVX512_I16 * NR_AVX512_I16)
            .map(|i| i as i32 * 100)
            .collect();
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16_i16_avx512(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_AVX512_I16 as isize,
                1,
                true,
            );
        }
        naive_i16gemm(MR_AVX512_I16, NR_AVX512_I16, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_I16 * NR_AVX512_I16 {
            assert_eq!(c_ours[i], c_ref[i], "accumulate mismatch at {i}");
        }
    }

    #[test]
    fn kernel_edge_i16_avx512_partial() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }
        let kc = 6;
        let m = 3;
        let n = 9;
        let a: Vec<i16> = (0..MR_AVX512_I16 * kc)
            .map(|i| (i as i16) * 7 - 30)
            .collect();
        let b: Vec<i16> = (0..kc * NR_AVX512_I16)
            .map(|i| (i as i16) * 11 - 50)
            .collect();
        let mut c_ours = vec![0_i32; m * n];
        let mut c_ref = vec![0_i32; m * n];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_edge_i16_avx512(
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
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0_i32;
                for p in 0..kc {
                    acc = acc
                        .wrapping_add((a[i * kc + p] as i32) * (b[p * NR_AVX512_I16 + j] as i32));
                }
                c_ref[i * n + j] = acc;
            }
        }
        for i in 0..m * n {
            assert_eq!(c_ours[i], c_ref[i], "edge mismatch at {i}");
        }
    }
}
