// AVX2 quantized i8 signed-signed GEMM micro-kernel — i8 × i8 → i32.
// Used for symmetric quantization schemes where both operands are signed
// (vs the gemm_i8 / vpmaddubsw path which assumes u8 × i8).
//
// Strategy: VPMADDUBSW only works on (u8, i8) pairs, so for (i8, i8) we
// can't use the saturating-i16 chain. Instead we sign-extend B's i8 lane
// to i16 in-register via VPMOVSXBW (16 i8 → 16 i16, single insn) and
// keep A pre-widened to i16 in the pack buffer, so per-iter A loads
// match the i16 kernel's free 4-byte broadcast.
//
// Why pre-widen A but not B:
//   - A is broadcast per-row per-K-block — 4 i32 broadcasts per K-block.
//     Intel's 4-byte broadcast (VPBROADCASTD) is 1 cycle, throughput 1/c.
//     Doing scalar 2-byte widening per row before broadcast would add
//     ~5 dependent uops per row (4 rows × ~5 uops = 20 uops/K-block) and
//     measurably regresses vs just paying the one-time pack-side widen.
//   - B is loaded once per K-block (1 ymm load), so a single VPMOVSXBW
//     (3-cycle latency, 1/cycle throughput on port 5) is essentially
//     free, and B as i8 in the pack halves L2 footprint.
//
// Register tile: 4 rows × 8 cols i32 output. 4 ymm accumulators.
//
// Per K-block (2 K iters):
//   - 1 xmm load of B (16 i8 = 2 K × 8 cols) + 1 VPMOVSXBW → 16 i16 ymm
//   - 4× VPBROADCASTD of A's 4-byte K-pair (already i16) → 16 i16 ymm
//   - 4× {VPMADDWD, VPADDD}
//
// Packing layouts (must match the kernel's expected order):
//   packed_a (i16, pre-widened from i8):
//     for each MR-row strip, 2-K-block-major:
//       [a[0, k+0..2], a[1, k+0..2], a[2, k+0..2], a[3, k+0..2], <next>...]
//       8 i16 (= 16 bytes) per k-block per strip.
//   packed_b (i8, kernel widens via VPMOVSXBW):
//     NR-wide col strips, 2-K-block-major:
//       [b[k+0..2, 0], b[k+0..2, 1], ..., b[k+0..2, 7], <next>...]
//       16 i8 (= 16 bytes) per k-block per strip.
//
// kc must be a multiple of 2 (caller pads with zeros via the pack routines).

use core::arch::x86_64::{
    __m128i, __m256i, _MM_HINT_T0, _mm_loadu_si128, _mm_prefetch, _mm256_add_epi32,
    _mm256_cvtepi8_epi16, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_storeu_si256,
};

pub const MR_I8S: usize = 4;
pub const NR_I8S: usize = 8;
/// K-block size — vpmaddwd reduces 2 K iters into the accumulators.
pub const KB_I8S: usize = 2;

const A_PF_BYTES: usize = 512;
const B_PF_BYTES: usize = 512;

/// 4×8 i8 signed-signed GEMM micro-kernel. Computes C += A @ B where
/// A is i16 (pre-widened from i8 by the pack routine) and B is i8,
/// C is i32. `kc` is the depth in i8 elements (must be a multiple of 2).
///
/// SAFETY: caller guarantees AVX2 availability, `packed_a >= kc * MR_I8S
/// i16s` (= 2x bytes), `packed_b >= kc * NR_I8S i8s`. `c_rs` / `c_cs`
/// are row/column strides in i32 elements.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
pub unsafe fn kernel_4x8_i8_signed(
    kc: usize,
    packed_a: *const i16,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(kc % KB_I8S == 0, "kc must be a multiple of KB_I8S={KB_I8S}");

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    let n_blocks = kc / KB_I8S;
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..n_blocks {
        _mm_prefetch::<_MM_HINT_T0>(a.cast::<u8>().add(A_PF_BYTES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.cast::<u8>().add(B_PF_BYTES).cast());

        // Load 16 i8 of B = 2 K × 8 cols, sign-extend to 16 i16.
        let b_i8 = _mm_loadu_si128(b as *const __m128i);
        let b_v = _mm256_cvtepi8_epi16(b_i8);

        // Row 0: broadcast A[0, k..k+2] (2 i16 = 4 bytes = i32-sized pair).
        let a0 = _mm256_set1_epi32(*(a as *const i32));
        let p0 = _mm256_madd_epi16(a0, b_v);
        acc0 = _mm256_add_epi32(acc0, p0);

        // Row 1.
        let a1 = _mm256_set1_epi32(*(a.add(KB_I8S) as *const i32));
        let p1 = _mm256_madd_epi16(a1, b_v);
        acc1 = _mm256_add_epi32(acc1, p1);

        // Row 2.
        let a2 = _mm256_set1_epi32(*(a.add(2 * KB_I8S) as *const i32));
        let p2 = _mm256_madd_epi16(a2, b_v);
        acc2 = _mm256_add_epi32(acc2, p2);

        // Row 3.
        let a3 = _mm256_set1_epi32(*(a.add(3 * KB_I8S) as *const i32));
        let p3 = _mm256_madd_epi16(a3, b_v);
        acc3 = _mm256_add_epi32(acc3, p3);

        a = a.add(MR_I8S * KB_I8S);
        b = b.add(NR_I8S * KB_I8S);
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

/// Edge-case kernel for tail tiles where m < MR_I8S or n < NR_I8S.
/// Scalar fallback. A is i16-widened; B is i8.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_i8_signed(
    m: usize,
    n: usize,
    kc: usize,
    packed_a: *const i16,
    packed_b: *const i8,
    c: *mut i32,
    c_rs: isize,
    c_cs: isize,
    accumulate: bool,
) {
    debug_assert!(m <= MR_I8S);
    debug_assert!(n <= NR_I8S);
    debug_assert!(kc % KB_I8S == 0);
    let mut tile = [[0_i32; NR_I8S]; MR_I8S];
    let mut a = packed_a;
    let mut b = packed_b;
    let n_blocks = kc / KB_I8S;
    for _ in 0..n_blocks {
        for i in 0..m {
            for kk in 0..KB_I8S {
                let av = *a.add(i * KB_I8S + kk) as i32;
                for j in 0..n {
                    let bv = *b.add(j * KB_I8S + kk) as i32;
                    tile[i][j] = tile[i][j].wrapping_add(av * bv);
                }
            }
        }
        a = a.add(MR_I8S * KB_I8S);
        b = b.add(NR_I8S * KB_I8S);
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

    fn naive_i8s_gemm(m: usize, n: usize, k: usize, a: &[i8], b: &[i8], c: &mut [i32]) {
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

    /// Pack A as i16 with sign-extension from i8 source.
    fn pack_a(a: &[i8], kc: usize) -> Vec<i16> {
        let mut packed = vec![0_i16; MR_I8S * kc];
        let n_blocks = kc / KB_I8S;
        for kb in 0..n_blocks {
            for i in 0..MR_I8S {
                for kk in 0..KB_I8S {
                    packed[kb * MR_I8S * KB_I8S + i * KB_I8S + kk] =
                        a[i * kc + kb * KB_I8S + kk] as i16;
                }
            }
        }
        packed
    }

    fn pack_b(b: &[i8], kc: usize) -> Vec<i8> {
        let mut packed = vec![0_i8; NR_I8S * kc];
        let n_blocks = kc / KB_I8S;
        for kb in 0..n_blocks {
            for j in 0..NR_I8S {
                for kk in 0..KB_I8S {
                    packed[kb * NR_I8S * KB_I8S + j * KB_I8S + kk] =
                        b[(kb * KB_I8S + kk) * NR_I8S + j];
                }
            }
        }
        packed
    }

    #[test]
    fn kernel_4x8_i8_signed_matches_naive() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 16;
        let a: Vec<i8> = (0..MR_I8S * kc)
            .map(|i| ((i as i32) * 7 - 60) as i8)
            .collect();
        let b: Vec<i8> = (0..kc * NR_I8S)
            .map(|i| ((i as i32) * 11 - 80) as i8)
            .collect();
        let mut c_ours = vec![0_i32; MR_I8S * NR_I8S];
        let mut c_ref = vec![0_i32; MR_I8S * NR_I8S];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_i8_signed(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_I8S as isize,
                1,
                false,
            );
        }
        naive_i8s_gemm(MR_I8S, NR_I8S, kc, &a, &b, &mut c_ref);

        for i in 0..MR_I8S * NR_I8S {
            assert_eq!(c_ours[i], c_ref[i], "i8s mismatch at idx {i}");
        }
    }

    #[test]
    fn kernel_4x8_i8_signed_negative_inputs() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 8;
        let a: Vec<i8> = (0..MR_I8S * kc)
            .map(|i| {
                if i % 2 == 0 {
                    -((i % 100) as i8)
                } else {
                    (i % 100) as i8
                }
            })
            .collect();
        let b: Vec<i8> = (0..kc * NR_I8S)
            .map(|i| {
                if i % 3 == 0 {
                    -((i % 80) as i8)
                } else {
                    (i % 80) as i8
                }
            })
            .collect();
        let mut c_ours = vec![0_i32; MR_I8S * NR_I8S];
        let mut c_ref = vec![0_i32; MR_I8S * NR_I8S];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_i8_signed(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_I8S as isize,
                1,
                false,
            );
        }
        naive_i8s_gemm(MR_I8S, NR_I8S, kc, &a, &b, &mut c_ref);

        for i in 0..MR_I8S * NR_I8S {
            assert_eq!(c_ours[i], c_ref[i], "negative-input mismatch at {i}");
        }
    }

    #[test]
    fn kernel_4x8_i8_signed_extreme_values() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 6;
        let a: Vec<i8> = (0..MR_I8S * kc)
            .map(|i| match i % 5 {
                0 => i8::MIN,
                1 => i8::MAX,
                2 => -64,
                3 => 0,
                _ => 33,
            })
            .collect();
        let b: Vec<i8> = (0..kc * NR_I8S)
            .map(|i| match i % 4 {
                0 => i8::MAX,
                1 => i8::MIN,
                2 => -1,
                _ => 1,
            })
            .collect();
        let mut c_ours = vec![0_i32; MR_I8S * NR_I8S];
        let mut c_ref = vec![0_i32; MR_I8S * NR_I8S];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_i8_signed(
                kc,
                pa.as_ptr(),
                pb.as_ptr(),
                c_ours.as_mut_ptr(),
                NR_I8S as isize,
                1,
                false,
            );
        }
        naive_i8s_gemm(MR_I8S, NR_I8S, kc, &a, &b, &mut c_ref);

        for i in 0..MR_I8S * NR_I8S {
            assert_eq!(c_ours[i], c_ref[i], "extreme-input mismatch at {i}");
        }
    }

    #[test]
    fn kernel_edge_i8_signed_handles_partial() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kc = 6;
        let m = 3;
        let n = 5;
        // Build varied i8 inputs without tripping debug overflow:
        // (i * mult) wraps mod 256, then we subtract a fixed offset
        // and re-cast. The naive reference kernel sees the same
        // bit pattern, so the comparison stays valid.
        let a: Vec<i8> = (0..MR_I8S * kc)
            .map(|i| (i as i8).wrapping_mul(3).wrapping_sub(20))
            .collect();
        let b: Vec<i8> = (0..kc * NR_I8S)
            .map(|i| (i as i8).wrapping_mul(5).wrapping_sub(40))
            .collect();
        let mut c_ours = vec![0_i32; m * n];
        let mut c_ref = vec![0_i32; m * n];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_edge_i8_signed(
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
                    acc = acc.wrapping_add((a[i * kc + p] as i32) * (b[p * NR_I8S + j] as i32));
                }
                c_ref[i * n + j] = acc;
            }
        }
        for i in 0..m * n {
            assert_eq!(c_ours[i], c_ref[i], "edge mismatch at {i}");
        }
    }
}
