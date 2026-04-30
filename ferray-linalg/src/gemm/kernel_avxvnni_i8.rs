// AVX-VNNI quantized i8 GEMM micro-kernel — same 4×8 register tile and
// packing layout as `kernel_avx2_i8.rs`, but uses the single-instruction
// VPDPBUSD path. AVX-VNNI is the 256-bit-only VNNI extension introduced
// on Alder Lake / Tiger Lake / Sapphire Rapids / Zen4 (so it's available
// on consumer Intel chips that DON'T have AVX-512). The intrinsic is
// `_mm256_dpbusd_avx_epi32`; the `_avx_` infix distinguishes it from
// the AVX-512 form (`_mm256_dpbusd_epi32`) which gates on AVX-512 VNNI
// rather than the standalone AVX-VNNI flag.
//
// Compared to the VPMADDUBSW + VPMADDWD chain in the AVX2 baseline,
// VPDPBUSD does the (u8 × i8) → i32 4-way dot-product accumulate in a
// single instruction, halving the inner-loop instruction count. Each
// VPDPBUSD takes 32 (u8, i8) pairs and accumulates 8 i32 sums of 4
// products each — same output shape as the chain but 1 instruction.
//
// Per K-block (4 K iters):
//   - 1 ymm load of B (32 bytes)
//   - 4 broadcasts of A (one per row, 4-byte K-quad each)
//   - 4 vpdpbusd into accumulators
// vs the VPMADDUBSW chain's 4 × {vpmaddubsw, vpmaddwd, vpaddd}.

use core::arch::x86_64::{
    __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_dpbusd_avx_epi32, _mm256_loadu_si256,
    _mm256_set1_epi32, _mm256_setzero_si256, _mm256_storeu_si256,
};

pub use super::kernel_avx2_i8::{KB_I8, MR_I8, NR_I8};

const A_PF_BYTES: usize = 512;
const B_PF_BYTES: usize = 512;

/// 4×8 i8 GEMM micro-kernel using AVX-VNNI VPDPBUSD. Same packing
/// contract as `kernel_4x8_i8` (AVX2 baseline). `kc` must be a multiple
/// of `KB_I8 = 4`.
///
/// SAFETY: caller guarantees AVX-VNNI availability via
/// `is_x86_feature_detected!("avxvnni")`. Buffers as in `kernel_4x8_i8`.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn kernel_4x8_i8_vnni(
    kc: usize,
    packed_a: *const u8,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(kc % KB_I8 == 0);

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    let n_blocks = kc / KB_I8;
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..n_blocks {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_BYTES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_BYTES).cast());

        let b_v = _mm256_loadu_si256(b as *const __m256i);

        // Each row: broadcast 4-byte K-quad, dpbusd into accumulator.
        let a0 = _mm256_set1_epi32(*(a as *const i32));
        acc0 = _mm256_dpbusd_avx_epi32(acc0, a0, b_v);

        let a1 = _mm256_set1_epi32(*(a.add(KB_I8) as *const i32));
        acc1 = _mm256_dpbusd_avx_epi32(acc1, a1, b_v);

        let a2 = _mm256_set1_epi32(*(a.add(2 * KB_I8) as *const i32));
        acc2 = _mm256_dpbusd_avx_epi32(acc2, a2, b_v);

        let a3 = _mm256_set1_epi32(*(a.add(3 * KB_I8) as *const i32));
        acc3 = _mm256_dpbusd_avx_epi32(acc3, a3, b_v);

        a = a.add(MR_I8 * KB_I8);
        b = b.add(NR_I8 * KB_I8);
    }

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
    use core::arch::x86_64::_mm256_add_epi32;
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

    /// Skips on CPUs without AVX-VNNI (older than Alder Lake / Zen4).
    #[test]
    fn kernel_4x8_i8_vnni_matches_naive() {
        if !is_x86_feature_detected!("avxvnni") {
            return;
        }
        let kc = 16;
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
            kernel_4x8_i8_vnni(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_I8 as isize,
                1,
                false,
            );
        }
        naive_i8gemm(MR_I8, NR_I8, kc, &a, &b, &mut c_ref);

        for i in 0..MR_I8 * NR_I8 {
            assert_eq!(
                c_ours[i], c_ref[i],
                "VNNI mismatch at idx {i}: ours={} ref={}",
                c_ours[i], c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_4x8_i8_vnni_matches_avx2_baseline() {
        if !is_x86_feature_detected!("avxvnni") || !is_x86_feature_detected!("avx2") {
            return;
        }
        // Cross-check VNNI vs the AVX2 chain: same inputs, same outputs.
        let kc = 24;
        let a: Vec<u8> = (0..MR_I8 * kc)
            .map(|i| ((i * 13 + 1) as u8) & 0x7f)
            .collect();
        let b: Vec<i8> = (0..kc * NR_I8)
            .map(|i| ((i as i32) * 9 - 100) as i8)
            .collect();
        let mut c_vnni = vec![0_i32; MR_I8 * NR_I8];
        let mut c_avx2 = vec![0_i32; MR_I8 * NR_I8];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_i8_vnni(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_vnni.as_mut_ptr(),
                NR_I8 as isize,
                1,
                false,
            );
            super::super::kernel_avx2_i8::kernel_4x8_i8(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_avx2.as_mut_ptr(),
                NR_I8 as isize,
                1,
                false,
            );
        }

        for i in 0..MR_I8 * NR_I8 {
            assert_eq!(
                c_vnni[i], c_avx2[i],
                "VNNI vs AVX2 mismatch at {i}: vnni={} avx2={}",
                c_vnni[i], c_avx2[i]
            );
        }
    }
}
