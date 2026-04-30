// AVX-512 VNNI quantized i8 GEMM micro-kernel — same shape as the
// AVX-VNNI 4x8 i8 kernel but uses zmm registers and `vpdpbusd` on the
// 512-bit form. The intrinsic is `_mm512_dpbusd_epi32`; it's gated on
// the AVX-512 VNNI feature flag (`avx512vnni`), distinct from the
// 256-bit-only AVX-VNNI flag (`avxvnni`). AVX-512 VNNI is available on
// Cascade Lake +, Ice Lake server +, Sapphire Rapids +, Zen4 +.
//
// Each zmm holds 64 u8 / 64 i8 / 16 i32. The kernel processes a 4 K-iter
// block per inner iteration: one VPDPBUSD takes 64 (u8, i8) pairs and
// emits 16 i32 sums of 4 products each. So per K-block per row, one
// vpdpbusd reduces 4 K iters of 16 cols into the row's accumulator.
//
// Register tile: 4 rows × 16 cols of i32 output. 4 zmm accumulators,
// one per row. Per K-block (4 K iters):
//   - 1 zmm B-load (64 i8 = 4 K × 16 cols, packed)
//   - 4 broadcasts of A (4-byte K-quads, one per row)
//   - 4 vpdpbusd into accumulators
//
// Per K iter: ~2.25 zmm ops (9/4) per row × 4 rows = ~9 zmm ops/K-block,
// 4 K iters = ~2.25 zmm ops/K iter. Each vpdpbusd is 64 mul-adds, so
// per K-block (4 K) we get 4 × 64 = 256 mul-adds for 64 output cells =
// 4 mul-adds/output/K — i.e. 1 mul-add per output per K iter, the
// theoretical i8 GEMM roof.
//
// Cfg-gated on the existing `avx512` feature.

use core::arch::x86_64::{
    __m512i, _MM_HINT_T0, _mm_prefetch, _mm512_add_epi32, _mm512_dpbusd_epi32, _mm512_loadu_si512,
    _mm512_set1_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
};

pub use super::kernel_avx2_i8::KB_I8;

pub const MR_AVX512_I8: usize = 4;
/// AVX-512 zmm holds 16 i32 → NR=16 cols per ymm-sized strip (vs NR=8
/// for ymm).
pub const NR_AVX512_I8: usize = 16;

const A_PF_BYTES: usize = 512;
const B_PF_BYTES: usize = 512;

/// 4x16 AVX-512 VNNI i8 GEMM micro-kernel. `kc` must be a multiple of 4.
///
/// SAFETY: caller guarantees both `avx512f` AND `avx512vnni` are
/// available at runtime. Buffer-length contract mirrors `kernel_4x8_i8`
/// (just doubled for the wider register tile).
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f,avx512vnni")]
pub unsafe fn kernel_4x16_i8_avx512(
    kc: usize,
    packed_a: *const u8,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(kc % KB_I8 == 0);

    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    let n_blocks = kc / KB_I8;
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..n_blocks {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_BYTES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_BYTES).cast());

        // Load 64 i8 of B = 4 K × 16 cols, packed col-major within the
        // 4-K block.
        let b_v = _mm512_loadu_si512(b as *const __m512i);

        let a0 = _mm512_set1_epi32(*(a as *const i32));
        acc0 = _mm512_dpbusd_epi32(acc0, a0, b_v);

        let a1 = _mm512_set1_epi32(*(a.add(KB_I8) as *const i32));
        acc1 = _mm512_dpbusd_epi32(acc1, a1, b_v);

        let a2 = _mm512_set1_epi32(*(a.add(2 * KB_I8) as *const i32));
        acc2 = _mm512_dpbusd_epi32(acc2, a2, b_v);

        let a3 = _mm512_set1_epi32(*(a.add(3 * KB_I8) as *const i32));
        acc3 = _mm512_dpbusd_epi32(acc3, a3, b_v);

        a = a.add(MR_AVX512_I8 * KB_I8);
        b = b.add(NR_AVX512_I8 * KB_I8);
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

/// Edge-case kernel for tail tiles. Slow scalar path, used only on
/// row/col edges where the full register tile doesn't apply.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_i8_avx512(
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
    debug_assert!(m <= MR_AVX512_I8);
    debug_assert!(n <= NR_AVX512_I8);
    debug_assert!(kc % KB_I8 == 0);
    let mut tile = [[0_i32; NR_AVX512_I8]; MR_AVX512_I8];
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
        a = a.add(MR_AVX512_I8 * KB_I8);
        b = b.add(NR_AVX512_I8 * KB_I8);
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

    fn pack_a(a: &[u8], kc: usize) -> Vec<u8> {
        let mut packed = vec![0_u8; MR_AVX512_I8 * kc];
        let n_blocks = kc / KB_I8;
        for kb in 0..n_blocks {
            for i in 0..MR_AVX512_I8 {
                for kk in 0..KB_I8 {
                    packed[kb * MR_AVX512_I8 * KB_I8 + i * KB_I8 + kk] =
                        a[i * kc + kb * KB_I8 + kk];
                }
            }
        }
        packed
    }

    fn pack_b(b: &[i8], kc: usize) -> Vec<i8> {
        let mut packed = vec![0_i8; NR_AVX512_I8 * kc];
        let n_blocks = kc / KB_I8;
        for kb in 0..n_blocks {
            for j in 0..NR_AVX512_I8 {
                for kk in 0..KB_I8 {
                    packed[kb * NR_AVX512_I8 * KB_I8 + j * KB_I8 + kk] =
                        b[(kb * KB_I8 + kk) * NR_AVX512_I8 + j];
                }
            }
        }
        packed
    }

    /// Skips on hosts without AVX-512 VNNI (e.g., consumer Raptor Lake).
    #[test]
    fn kernel_4x16_i8_avx512_matches_naive() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vnni") {
            return;
        }
        let kc = 16;
        let a: Vec<u8> = (0..MR_AVX512_I8 * kc)
            .map(|i| ((i * 7 + 3) as u8) & 0x7f)
            .collect();
        let b: Vec<i8> = (0..kc * NR_AVX512_I8)
            .map(|i| (((i as i32) * 11 - 50) % 100) as i8)
            .collect();
        let mut c_ours = vec![0_i32; MR_AVX512_I8 * NR_AVX512_I8];
        let mut c_ref = vec![0_i32; MR_AVX512_I8 * NR_AVX512_I8];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16_i8_avx512(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_AVX512_I8 as isize,
                1,
                false,
            );
        }
        naive_i8gemm(MR_AVX512_I8, NR_AVX512_I8, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_I8 * NR_AVX512_I8 {
            assert_eq!(
                c_ours[i], c_ref[i],
                "AVX-512 VNNI mismatch at idx {i}: ours={} ref={}",
                c_ours[i], c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_4x16_i8_avx512_accumulate() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vnni") {
            return;
        }
        let kc = 8;
        let a: Vec<u8> = (0..MR_AVX512_I8 * kc)
            .map(|i| ((i * 5) as u8) & 0x3f)
            .collect();
        let b: Vec<i8> = (0..kc * NR_AVX512_I8)
            .map(|i| ((i as i32 - 10) % 50) as i8)
            .collect();
        let mut c_ours: Vec<i32> = (0..MR_AVX512_I8 * NR_AVX512_I8)
            .map(|i| i as i32 * 100)
            .collect();
        let mut c_ref = c_ours.clone();

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x16_i8_avx512(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_AVX512_I8 as isize,
                1,
                true,
            );
        }
        naive_i8gemm(MR_AVX512_I8, NR_AVX512_I8, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_I8 * NR_AVX512_I8 {
            assert_eq!(
                c_ours[i], c_ref[i],
                "idx {i}: ours={} ref={}",
                c_ours[i], c_ref[i]
            );
        }
    }
}
