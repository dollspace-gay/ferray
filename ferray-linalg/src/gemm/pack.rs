// Packing routines for the AVX2 8x6 micro-kernel.
//
// Two transformations:
//
// pack_a: copy a row-major (MC, KC) panel of A into an MR-tall column-major
// strip layout. For each MR-row strip of the panel, the packed buffer holds
// `kc` columns of MR contiguous doubles. The micro-kernel reads the A
// column at depth k as `packed_a[k * MR ..][..MR]`.
//
// pack_b: copy a row-major (KC, NC) panel of B into an NR-wide row-major
// strip layout. For each NR-col strip of the panel, the packed buffer holds
// `kc` rows of NR contiguous doubles. The micro-kernel reads the B row at
// depth k as `packed_b[k * NR ..][..NR]`.
//
// Tail strips (m < MR or n < NR) are zero-padded so the kernel can run a
// full-width body and the spurious tail elements multiply with zero.

use super::kernel_avx2::{MR, NR};

/// Pack a (mc, kc) panel of A into MR-tall strips.
/// `a` is the source row-major matrix with leading dimension `lda` doubles.
/// `packed` must hold `mc.next_multiple_of(MR) * kc` doubles.
///
/// Layout: for each MR-strip s in 0..ceil(mc/MR), the strip occupies
///   packed[s*MR*kc ..][..MR*kc]
/// stored kc-major: packed[s*MR*kc + k*MR + i] = A[s*MR + i, k] (zero if
/// s*MR + i >= mc).
pub fn pack_a(mc: usize, kc: usize, a: *const f64, lda: usize, packed: &mut [f64]) {
    let n_strips = mc.div_ceil(MR);
    debug_assert!(packed.len() >= n_strips * MR * kc);

    for s in 0..n_strips {
        let row_base = s * MR;
        let strip = &mut packed[s * MR * kc..s * MR * kc + MR * kc];
        let m_in_strip = (mc - row_base).min(MR);

        if m_in_strip == MR {
            // Full strip — fast path.
            for k in 0..kc {
                for i in 0..MR {
                    // SAFETY: caller guarantees a[..mc*lda] is valid.
                    strip[k * MR + i] = unsafe { *a.add((row_base + i) * lda + k) };
                }
            }
        } else {
            // Partial strip — pad with zeros.
            for k in 0..kc {
                for i in 0..MR {
                    strip[k * MR + i] = if i < m_in_strip {
                        unsafe { *a.add((row_base + i) * lda + k) }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

/// Pack a (kc, nc) panel of B into NR-wide strips.
/// `b` is the source row-major matrix with leading dimension `ldb` doubles.
/// `packed` must hold `kc * nc.next_multiple_of(NR)` doubles.
///
/// Layout: for each NR-strip s in 0..ceil(nc/NR), the strip occupies
///   packed[s*NR*kc ..][..NR*kc]
/// stored kc-major: packed[s*NR*kc + k*NR + j] = B[k, s*NR + j] (zero if
/// s*NR + j >= nc).
pub fn pack_b(kc: usize, nc: usize, b: *const f64, ldb: usize, packed: &mut [f64]) {
    let n_strips = nc.div_ceil(NR);
    debug_assert!(packed.len() >= n_strips * NR * kc);

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;

    for s in 0..n_strips {
        let col_base = s * NR;
        let strip = &mut packed[s * NR * kc..s * NR * kc + NR * kc];
        let n_in_strip = (nc - col_base).min(NR);

        if n_in_strip == NR {
            // Full strip: NR=8 contiguous doubles per row. Two unaligned
            // 256-bit loads + stores per row replace 8 scalar copies.
            #[cfg(target_arch = "x86_64")]
            if use_simd {
                // SAFETY: AVX2 detected at runtime; bounds checked above.
                unsafe { pack_b_strip_avx2(kc, b, ldb, col_base, strip) };
                continue;
            }
            for k in 0..kc {
                for j in 0..NR {
                    strip[k * NR + j] = unsafe { *b.add(k * ldb + col_base + j) };
                }
            }
        } else {
            for k in 0..kc {
                for j in 0..NR {
                    strip[k * NR + j] = if j < n_in_strip {
                        unsafe { *b.add(k * ldb + col_base + j) }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pack_b_strip_avx2(
    kc: usize,
    b: *const f64,
    ldb: usize,
    col_base: usize,
    strip: &mut [f64],
) {
    use core::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};
    let strip_ptr = strip.as_mut_ptr();
    for k in 0..kc {
        let src = b.add(k * ldb + col_base);
        let dst = strip_ptr.add(k * NR);
        let lo = _mm256_loadu_pd(src);
        let hi = _mm256_loadu_pd(src.add(4));
        _mm256_storeu_pd(dst, lo);
        _mm256_storeu_pd(dst.add(4), hi);
    }
}

// ---------------------------------------------------------------------
// f64 NR=16 B-pack — sibling of `pack_b` for the AVX-512 DGEMM kernel,
// which uses an NR=16 strip (vs NR=8 for the AVX2 DGEMM). Same f64
// element type, just doubled strip width. Behind `avx512` feature.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512::NR_AVX512;

/// Pack a (kc, nc) panel of B into NR_AVX512=16-wide strips. Layout
/// matches `pack_b` but with double the strip width.
#[cfg(feature = "avx512")]
pub fn pack_b_f64_nr16(kc: usize, nc: usize, b: *const f64, ldb: usize, packed: &mut [f64]) {
    let n_strips = nc.div_ceil(NR_AVX512);
    debug_assert!(packed.len() >= n_strips * NR_AVX512 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_AVX512;
        let strip = &mut packed[s * NR_AVX512 * kc..s * NR_AVX512 * kc + NR_AVX512 * kc];
        let n_in_strip = (nc - col_base).min(NR_AVX512);

        if n_in_strip == NR_AVX512 {
            for k in 0..kc {
                for j in 0..NR_AVX512 {
                    strip[k * NR_AVX512 + j] = unsafe { *b.add(k * ldb + col_base + j) };
                }
            }
        } else {
            for k in 0..kc {
                for j in 0..NR_AVX512 {
                    strip[k * NR_AVX512 + j] = if j < n_in_strip {
                        unsafe { *b.add(k * ldb + col_base + j) }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Complex<f64> (ZGEMM) packing routines — same kc-major MR-minor layout
// as the real DGEMM packs, but each "element" is a Complex<f64> = 2 f64s
// laid out [re, im]. The kernel reads the buffer as *const f64 and indexes
// by (k * MR_C64 * 2 + i * 2 + 0/1) for the real/imag of row i at depth k.
// ---------------------------------------------------------------------

use super::kernel_avx2_complex_f64::{MR_C64, NR_C64};

/// Pack a (mc, kc) panel of complex A into MR_C64-tall strips.
/// `a` points at the source row-major matrix as `*const f64` (each
/// Complex<f64> is 2 contiguous f64s). `lda` is the row stride **in
/// complex elements**.
pub fn pack_a_c64(mc: usize, kc: usize, a: *const f64, lda: usize, packed: &mut [f64]) {
    let n_strips = mc.div_ceil(MR_C64);
    debug_assert!(packed.len() >= n_strips * MR_C64 * 2 * kc);

    for s in 0..n_strips {
        let row_base = s * MR_C64;
        let strip_start = s * MR_C64 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + MR_C64 * 2 * kc];
        let m_in_strip = (mc - row_base).min(MR_C64);

        if m_in_strip == MR_C64 {
            for k in 0..kc {
                for i in 0..MR_C64 {
                    let src_off = (row_base + i) * lda * 2 + k * 2;
                    strip[k * MR_C64 * 2 + i * 2] = unsafe { *a.add(src_off) };
                    strip[k * MR_C64 * 2 + i * 2 + 1] = unsafe { *a.add(src_off + 1) };
                }
            }
        } else {
            for k in 0..kc {
                for i in 0..MR_C64 {
                    if i < m_in_strip {
                        let src_off = (row_base + i) * lda * 2 + k * 2;
                        strip[k * MR_C64 * 2 + i * 2] = unsafe { *a.add(src_off) };
                        strip[k * MR_C64 * 2 + i * 2 + 1] = unsafe { *a.add(src_off + 1) };
                    } else {
                        strip[k * MR_C64 * 2 + i * 2] = 0.0;
                        strip[k * MR_C64 * 2 + i * 2 + 1] = 0.0;
                    }
                }
            }
        }
    }
}

/// Pack a (kc, nc) panel of complex B into NR_C64-wide strips.
/// `b` points at the source row-major matrix as `*const f64`. `ldb` is
/// the row stride **in complex elements**.
pub fn pack_b_c64(kc: usize, nc: usize, b: *const f64, ldb: usize, packed: &mut [f64]) {
    let n_strips = nc.div_ceil(NR_C64);
    debug_assert!(packed.len() >= n_strips * NR_C64 * 2 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_C64;
        let strip_start = s * NR_C64 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + NR_C64 * 2 * kc];
        let n_in_strip = (nc - col_base).min(NR_C64);

        for k in 0..kc {
            for j in 0..NR_C64 {
                if j < n_in_strip {
                    let src_off = k * ldb * 2 + (col_base + j) * 2;
                    strip[k * NR_C64 * 2 + j * 2] = unsafe { *b.add(src_off) };
                    strip[k * NR_C64 * 2 + j * 2 + 1] = unsafe { *b.add(src_off + 1) };
                } else {
                    strip[k * NR_C64 * 2 + j * 2] = 0.0;
                    strip[k * NR_C64 * 2 + j * 2 + 1] = 0.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// i8 GEMM packing routines (u8 × i8 → i32, the canonical quantized ML
// GEMM pattern). Both packs use a 4-K-block layout: 4 K-iterations are
// stored adjacent within each MR-tall (A) or NR-wide (B) strip, so the
// kernel's vpmaddubsw + vpmaddwd chain (or single vpdpbusd on AVX-VNNI)
// can read exactly 4 K's worth of contiguous bytes per row/col per
// k-block.
//
// Both packs zero-pad K up to a multiple of `KB_I8 = 4` and zero-pad
// trailing rows/cols when the strip is partial. The kernel is then
// always called with a kc that's a multiple of 4 even if the user's
// k is not.
// ---------------------------------------------------------------------

use super::kernel_avx2_i8::{KB_I8, MR_I8, NR_I8};

/// Round k up to the next multiple of `KB_I8 = 4`. The caller must
/// allocate pack buffers based on this rounded depth.
pub fn pack_kc_i8(k: usize) -> usize {
    k.div_ceil(KB_I8) * KB_I8
}

/// Pack a (mc, k) panel of u8 A into MR_I8-tall strips with the kernel's
/// 4-K-block layout. `lda` is the row stride **in u8 elements**.
/// `packed.len() >= n_strips * MR_I8 * pack_kc_i8(k)`.
pub fn pack_a_u8(mc: usize, k: usize, a: *const u8, lda: usize, packed: &mut [u8]) {
    let kc = pack_kc_i8(k);
    let n_blocks = kc / KB_I8;
    let n_strips = mc.div_ceil(MR_I8);
    debug_assert!(packed.len() >= n_strips * MR_I8 * kc);

    for s in 0..n_strips {
        let row_base = s * MR_I8;
        let m_in_strip = (mc - row_base).min(MR_I8);
        let strip_start = s * MR_I8 * kc;
        for kb in 0..n_blocks {
            for i in 0..MR_I8 {
                for kk in 0..KB_I8 {
                    let k_idx = kb * KB_I8 + kk;
                    let dst = strip_start + kb * MR_I8 * KB_I8 + i * KB_I8 + kk;
                    let v = if i < m_in_strip && k_idx < k {
                        unsafe { *a.add((row_base + i) * lda + k_idx) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

/// Pack a (k, nc) panel of i8 B into NR_I8-wide strips with the
/// 4-K-block layout. `ldb` is the row stride **in i8 elements**.
/// `packed.len() >= n_strips * NR_I8 * pack_kc_i8(k)`.
pub fn pack_b_i8(k: usize, nc: usize, b: *const i8, ldb: usize, packed: &mut [i8]) {
    let kc = pack_kc_i8(k);
    let n_blocks = kc / KB_I8;
    let n_strips = nc.div_ceil(NR_I8);
    debug_assert!(packed.len() >= n_strips * NR_I8 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_I8;
        let n_in_strip = (nc - col_base).min(NR_I8);
        let strip_start = s * NR_I8 * kc;
        for kb in 0..n_blocks {
            for j in 0..NR_I8 {
                for kk in 0..KB_I8 {
                    let k_idx = kb * KB_I8 + kk;
                    let dst = strip_start + kb * NR_I8 * KB_I8 + j * KB_I8 + kk;
                    let v = if j < n_in_strip && k_idx < k {
                        unsafe { *b.add(k_idx * ldb + (col_base + j)) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// i16 GEMM packing routines (i16 × i16 → i32). Same kc-major MR/NR-minor
// layout as i8 GEMM but with KB=2 (vpmaddwd reduces 2 K-iters at a time
// vs vpmaddubsw's 4) and i16 elements (2 bytes each).
// ---------------------------------------------------------------------

use super::kernel_avx2_i16::{KB_I16, MR_I16, NR_I16};

/// Round k up to the next multiple of `KB_I16 = 2`.
pub fn pack_kc_i16(k: usize) -> usize {
    k.div_ceil(KB_I16) * KB_I16
}

pub fn pack_a_i16(mc: usize, k: usize, a: *const i16, lda: usize, packed: &mut [i16]) {
    let kc = pack_kc_i16(k);
    let n_blocks = kc / KB_I16;
    let n_strips = mc.div_ceil(MR_I16);
    debug_assert!(packed.len() >= n_strips * MR_I16 * kc);

    for s in 0..n_strips {
        let row_base = s * MR_I16;
        let m_in_strip = (mc - row_base).min(MR_I16);
        let strip_start = s * MR_I16 * kc;
        for kb in 0..n_blocks {
            for i in 0..MR_I16 {
                for kk in 0..KB_I16 {
                    let k_idx = kb * KB_I16 + kk;
                    let dst = strip_start + kb * MR_I16 * KB_I16 + i * KB_I16 + kk;
                    let v = if i < m_in_strip && k_idx < k {
                        unsafe { *a.add((row_base + i) * lda + k_idx) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

pub fn pack_b_i16(k: usize, nc: usize, b: *const i16, ldb: usize, packed: &mut [i16]) {
    let kc = pack_kc_i16(k);
    let n_blocks = kc / KB_I16;
    let n_strips = nc.div_ceil(NR_I16);
    debug_assert!(packed.len() >= n_strips * NR_I16 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_I16;
        let n_in_strip = (nc - col_base).min(NR_I16);
        let strip_start = s * NR_I16 * kc;
        for kb in 0..n_blocks {
            for j in 0..NR_I16 {
                for kk in 0..KB_I16 {
                    let k_idx = kb * KB_I16 + kk;
                    let dst = strip_start + kb * NR_I16 * KB_I16 + j * KB_I16 + kk;
                    let v = if j < n_in_strip && k_idx < k {
                        unsafe { *b.add(k_idx * ldb + (col_base + j)) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// i8 signed-signed GEMM packing routines (i8 × i8 → i32). Uses the
// dedicated i8s kernel layout: MR=4, NR=8, KB=2 (matches the i16 kernel,
// since vpmaddwd reduces 2 K-pairs into 1 i32 after vpmovsxbw widens B).
// ---------------------------------------------------------------------

use super::kernel_avx2_i8s::{KB_I8S, MR_I8S, NR_I8S};

/// Round k up to the next multiple of `KB_I8S = 2`.
pub fn pack_kc_i8s(k: usize) -> usize {
    k.div_ceil(KB_I8S) * KB_I8S
}

/// Pack a (mc, k) panel of signed i8 A into MR_I8S-tall strips with the
/// kernel's 2-K-block layout, widening to i16 inline so the kernel's
/// per-iter A broadcast can reuse the i16 kernel's free 4-byte
/// VPBROADCASTD pattern. `lda` is the row stride in i8 elements.
/// `packed.len() >= n_strips * MR_I8S * pack_kc_i8s(k)`.
pub fn pack_a_i8s(mc: usize, k: usize, a: *const i8, lda: usize, packed: &mut [i16]) {
    let kc = pack_kc_i8s(k);
    let n_blocks = kc / KB_I8S;
    let n_strips = mc.div_ceil(MR_I8S);
    debug_assert!(packed.len() >= n_strips * MR_I8S * kc);

    for s in 0..n_strips {
        let row_base = s * MR_I8S;
        let m_in_strip = (mc - row_base).min(MR_I8S);
        let strip_start = s * MR_I8S * kc;
        for kb in 0..n_blocks {
            for i in 0..MR_I8S {
                for kk in 0..KB_I8S {
                    let k_idx = kb * KB_I8S + kk;
                    let dst = strip_start + kb * MR_I8S * KB_I8S + i * KB_I8S + kk;
                    let v = if i < m_in_strip && k_idx < k {
                        unsafe { *a.add((row_base + i) * lda + k_idx) as i16 }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

/// Pack a (k, nc) panel of signed i8 B into NR_I8S-wide strips with the
/// 2-K-block layout. `ldb` is the row stride in i8 elements.
/// `packed.len() >= n_strips * NR_I8S * pack_kc_i8s(k)`.
pub fn pack_b_i8s(k: usize, nc: usize, b: *const i8, ldb: usize, packed: &mut [i8]) {
    let kc = pack_kc_i8s(k);
    let n_blocks = kc / KB_I8S;
    let n_strips = nc.div_ceil(NR_I8S);
    debug_assert!(packed.len() >= n_strips * NR_I8S * kc);

    for s in 0..n_strips {
        let col_base = s * NR_I8S;
        let n_in_strip = (nc - col_base).min(NR_I8S);
        let strip_start = s * NR_I8S * kc;
        for kb in 0..n_blocks {
            for j in 0..NR_I8S {
                for kk in 0..KB_I8S {
                    let k_idx = kb * KB_I8S + kk;
                    let dst = strip_start + kb * NR_I8S * KB_I8S + j * KB_I8S + kk;
                    let v = if j < n_in_strip && k_idx < k {
                        unsafe { *b.add(k_idx * ldb + (col_base + j)) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// i8 NR=16 B-pack — sibling of `pack_b_i8` for the AVX-512 VNNI i8 GEMM
// kernel (NR=16 strip vs NR=8 for AVX2/AVX-VNNI). Behind `avx512`
// feature.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512_vnni_i8::NR_AVX512_I8;

#[cfg(feature = "avx512")]
pub fn pack_b_i8_nr16(k: usize, nc: usize, b: *const i8, ldb: usize, packed: &mut [i8]) {
    let kc = pack_kc_i8(k);
    let n_blocks = kc / KB_I8;
    let n_strips = nc.div_ceil(NR_AVX512_I8);
    debug_assert!(packed.len() >= n_strips * NR_AVX512_I8 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_AVX512_I8;
        let n_in_strip = (nc - col_base).min(NR_AVX512_I8);
        let strip_start = s * NR_AVX512_I8 * kc;
        for kb in 0..n_blocks {
            for j in 0..NR_AVX512_I8 {
                for kk in 0..KB_I8 {
                    let k_idx = kb * KB_I8 + kk;
                    let dst = strip_start + kb * NR_AVX512_I8 * KB_I8 + j * KB_I8 + kk;
                    let v = if j < n_in_strip && k_idx < k {
                        unsafe { *b.add(k_idx * ldb + (col_base + j)) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// i16 NR=16 packs — siblings of `pack_a_i16`/`pack_b_i16` for the
// AVX-512 i16 GEMM kernel (NR=16 strip vs NR=8 for AVX2). Behind the
// `avx512` feature.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512_i16::{MR_AVX512_I16, NR_AVX512_I16};

#[cfg(feature = "avx512")]
pub fn pack_a_i16_mr4_avx512(mc: usize, k: usize, a: *const i16, lda: usize, packed: &mut [i16]) {
    let kc = pack_kc_i16(k);
    let n_blocks = kc / KB_I16;
    let n_strips = mc.div_ceil(MR_AVX512_I16);
    debug_assert!(packed.len() >= n_strips * MR_AVX512_I16 * kc);

    for s in 0..n_strips {
        let row_base = s * MR_AVX512_I16;
        let m_in_strip = (mc - row_base).min(MR_AVX512_I16);
        let strip_start = s * MR_AVX512_I16 * kc;
        for kb in 0..n_blocks {
            for i in 0..MR_AVX512_I16 {
                for kk in 0..KB_I16 {
                    let k_idx = kb * KB_I16 + kk;
                    let dst = strip_start + kb * MR_AVX512_I16 * KB_I16 + i * KB_I16 + kk;
                    let v = if i < m_in_strip && k_idx < k {
                        unsafe { *a.add((row_base + i) * lda + k_idx) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

#[cfg(feature = "avx512")]
pub fn pack_b_i16_nr16(k: usize, nc: usize, b: *const i16, ldb: usize, packed: &mut [i16]) {
    let kc = pack_kc_i16(k);
    let n_blocks = kc / KB_I16;
    let n_strips = nc.div_ceil(NR_AVX512_I16);
    debug_assert!(packed.len() >= n_strips * NR_AVX512_I16 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_AVX512_I16;
        let n_in_strip = (nc - col_base).min(NR_AVX512_I16);
        let strip_start = s * NR_AVX512_I16 * kc;
        for kb in 0..n_blocks {
            for j in 0..NR_AVX512_I16 {
                for kk in 0..KB_I16 {
                    let k_idx = kb * KB_I16 + kk;
                    let dst = strip_start + kb * NR_AVX512_I16 * KB_I16 + j * KB_I16 + kk;
                    let v = if j < n_in_strip && k_idx < k {
                        unsafe { *b.add(k_idx * ldb + (col_base + j)) }
                    } else {
                        0
                    };
                    packed[dst] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Complex<f64> NR=4 B-pack — sibling of `pack_b_c64` for the AVX-512
// ZGEMM kernel, which uses an NR=4 strip (vs NR=2 for AVX2 ZGEMM).
// Behind `avx512` feature.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512_complex_f64::NR_AVX512_C64;

#[cfg(feature = "avx512")]
pub fn pack_b_c64_nr4(kc: usize, nc: usize, b: *const f64, ldb: usize, packed: &mut [f64]) {
    let n_strips = nc.div_ceil(NR_AVX512_C64);
    debug_assert!(packed.len() >= n_strips * NR_AVX512_C64 * 2 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_AVX512_C64;
        let strip_start = s * NR_AVX512_C64 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + NR_AVX512_C64 * 2 * kc];
        let n_in_strip = (nc - col_base).min(NR_AVX512_C64);

        for k in 0..kc {
            for j in 0..NR_AVX512_C64 {
                if j < n_in_strip {
                    let src_off = k * ldb * 2 + (col_base + j) * 2;
                    strip[k * NR_AVX512_C64 * 2 + j * 2] = unsafe { *b.add(src_off) };
                    strip[k * NR_AVX512_C64 * 2 + j * 2 + 1] = unsafe { *b.add(src_off + 1) };
                } else {
                    strip[k * NR_AVX512_C64 * 2 + j * 2] = 0.0;
                    strip[k * NR_AVX512_C64 * 2 + j * 2 + 1] = 0.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Complex<f32> NR=8 B-pack — sibling of `pack_b_c32` for the AVX-512
// CGEMM kernel, which uses an NR=8 strip (vs NR=4 for AVX2 CGEMM).
// Behind `avx512` feature.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512_complex_f32::NR_AVX512_C32;

#[cfg(feature = "avx512")]
pub fn pack_b_c32_nr8(kc: usize, nc: usize, b: *const f32, ldb: usize, packed: &mut [f32]) {
    let n_strips = nc.div_ceil(NR_AVX512_C32);
    debug_assert!(packed.len() >= n_strips * NR_AVX512_C32 * 2 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_AVX512_C32;
        let strip_start = s * NR_AVX512_C32 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + NR_AVX512_C32 * 2 * kc];
        let n_in_strip = (nc - col_base).min(NR_AVX512_C32);

        for k in 0..kc {
            for j in 0..NR_AVX512_C32 {
                if j < n_in_strip {
                    let src_off = k * ldb * 2 + (col_base + j) * 2;
                    strip[k * NR_AVX512_C32 * 2 + j * 2] = unsafe { *b.add(src_off) };
                    strip[k * NR_AVX512_C32 * 2 + j * 2 + 1] = unsafe { *b.add(src_off + 1) };
                } else {
                    strip[k * NR_AVX512_C32 * 2 + j * 2] = 0.0;
                    strip[k * NR_AVX512_C32 * 2 + j * 2 + 1] = 0.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Complex<f32> (CGEMM) packing routines — same kc-major MR-minor layout
// as the real packs, but each "element" is a Complex<f32> = 2 f32s
// `[re, im]`. The kernel reads the buffer as *const f32 and indexes by
// (k * MR_C32 * 2 + i * 2 + 0/1).
// ---------------------------------------------------------------------

use super::kernel_avx2_complex_f32::{MR_C32, NR_C32};

/// Pack a (mc, kc) panel of Complex<f32> A into MR_C32-tall strips.
/// `lda` is the row stride **in complex elements**.
pub fn pack_a_c32(mc: usize, kc: usize, a: *const f32, lda: usize, packed: &mut [f32]) {
    let n_strips = mc.div_ceil(MR_C32);
    debug_assert!(packed.len() >= n_strips * MR_C32 * 2 * kc);

    for s in 0..n_strips {
        let row_base = s * MR_C32;
        let strip_start = s * MR_C32 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + MR_C32 * 2 * kc];
        let m_in_strip = (mc - row_base).min(MR_C32);

        if m_in_strip == MR_C32 {
            for k in 0..kc {
                for i in 0..MR_C32 {
                    let src_off = (row_base + i) * lda * 2 + k * 2;
                    strip[k * MR_C32 * 2 + i * 2] = unsafe { *a.add(src_off) };
                    strip[k * MR_C32 * 2 + i * 2 + 1] = unsafe { *a.add(src_off + 1) };
                }
            }
        } else {
            for k in 0..kc {
                for i in 0..MR_C32 {
                    if i < m_in_strip {
                        let src_off = (row_base + i) * lda * 2 + k * 2;
                        strip[k * MR_C32 * 2 + i * 2] = unsafe { *a.add(src_off) };
                        strip[k * MR_C32 * 2 + i * 2 + 1] = unsafe { *a.add(src_off + 1) };
                    } else {
                        strip[k * MR_C32 * 2 + i * 2] = 0.0;
                        strip[k * MR_C32 * 2 + i * 2 + 1] = 0.0;
                    }
                }
            }
        }
    }
}

/// Pack a (kc, nc) panel of Complex<f32> B into NR_C32-wide strips.
/// `ldb` is the row stride **in complex elements**.
pub fn pack_b_c32(kc: usize, nc: usize, b: *const f32, ldb: usize, packed: &mut [f32]) {
    let n_strips = nc.div_ceil(NR_C32);
    debug_assert!(packed.len() >= n_strips * NR_C32 * 2 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_C32;
        let strip_start = s * NR_C32 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + NR_C32 * 2 * kc];
        let n_in_strip = (nc - col_base).min(NR_C32);

        for k in 0..kc {
            for j in 0..NR_C32 {
                if j < n_in_strip {
                    let src_off = k * ldb * 2 + (col_base + j) * 2;
                    strip[k * NR_C32 * 2 + j * 2] = unsafe { *b.add(src_off) };
                    strip[k * NR_C32 * 2 + j * 2 + 1] = unsafe { *b.add(src_off + 1) };
                } else {
                    strip[k * NR_C32 * 2 + j * 2] = 0.0;
                    strip[k * NR_C32 * 2 + j * 2 + 1] = 0.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// f32 (SGEMM) packing routines — same layout as f64, with MR_F32=4 tall
// A-strips and NR_F32=16 wide B-strips.
// ---------------------------------------------------------------------

use super::kernel_avx2_f32::{MR_F32, NR_F32};

/// Pack a (mc, kc) panel of A into MR_F32-tall strips. Layout matches
/// `pack_a` for f64 but with f32 elements.
pub fn pack_a_f32(mc: usize, kc: usize, a: *const f32, lda: usize, packed: &mut [f32]) {
    let n_strips = mc.div_ceil(MR_F32);
    debug_assert!(packed.len() >= n_strips * MR_F32 * kc);

    for s in 0..n_strips {
        let row_base = s * MR_F32;
        let strip = &mut packed[s * MR_F32 * kc..s * MR_F32 * kc + MR_F32 * kc];
        let m_in_strip = (mc - row_base).min(MR_F32);

        if m_in_strip == MR_F32 {
            for k in 0..kc {
                for i in 0..MR_F32 {
                    strip[k * MR_F32 + i] = unsafe { *a.add((row_base + i) * lda + k) };
                }
            }
        } else {
            for k in 0..kc {
                for i in 0..MR_F32 {
                    strip[k * MR_F32 + i] = if i < m_in_strip {
                        unsafe { *a.add((row_base + i) * lda + k) }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

/// Pack a (kc, nc) panel of B into NR_F32-wide strips. Layout matches
/// `pack_b` for f64 but with f32 elements. Full-strip path is SIMD-
/// vectorised (16 floats per row → two `vmovups` ymm).
pub fn pack_b_f32(kc: usize, nc: usize, b: *const f32, ldb: usize, packed: &mut [f32]) {
    let n_strips = nc.div_ceil(NR_F32);
    debug_assert!(packed.len() >= n_strips * NR_F32 * kc);

    #[cfg(target_arch = "x86_64")]
    let use_simd = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;

    for s in 0..n_strips {
        let col_base = s * NR_F32;
        let strip = &mut packed[s * NR_F32 * kc..s * NR_F32 * kc + NR_F32 * kc];
        let n_in_strip = (nc - col_base).min(NR_F32);

        if n_in_strip == NR_F32 {
            #[cfg(target_arch = "x86_64")]
            if use_simd {
                unsafe { pack_b_strip_avx2_f32(kc, b, ldb, col_base, strip) };
                continue;
            }
            for k in 0..kc {
                for j in 0..NR_F32 {
                    strip[k * NR_F32 + j] = unsafe { *b.add(k * ldb + col_base + j) };
                }
            }
        } else {
            for k in 0..kc {
                for j in 0..NR_F32 {
                    strip[k * NR_F32 + j] = if j < n_in_strip {
                        unsafe { *b.add(k * ldb + col_base + j) }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pack_b_strip_avx2_f32(
    kc: usize,
    b: *const f32,
    ldb: usize,
    col_base: usize,
    strip: &mut [f32],
) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
    let strip_ptr = strip.as_mut_ptr();
    for k in 0..kc {
        let src = b.add(k * ldb + col_base);
        let dst = strip_ptr.add(k * NR_F32);
        let lo = _mm256_loadu_ps(src);
        let hi = _mm256_loadu_ps(src.add(8));
        _mm256_storeu_ps(dst, lo);
        _mm256_storeu_ps(dst.add(8), hi);
    }
}

// ---------------------------------------------------------------------
// f32 NR=32 B-pack — sibling of `pack_b_f32` for the AVX-512 SGEMM
// kernel, which uses an NR=32 strip (vs NR=16 for the AVX2 SGEMM).
// Behind `avx512` feature.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512_f32::NR_AVX512_F32;

/// Pack a (kc, nc) panel of B into NR_AVX512_F32=32-wide strips.
/// Mirrors `pack_b_f32` but with double the strip width.
#[cfg(feature = "avx512")]
pub fn pack_b_f32_nr32(kc: usize, nc: usize, b: *const f32, ldb: usize, packed: &mut [f32]) {
    let n_strips = nc.div_ceil(NR_AVX512_F32);
    debug_assert!(packed.len() >= n_strips * NR_AVX512_F32 * kc);

    for s in 0..n_strips {
        let col_base = s * NR_AVX512_F32;
        let strip =
            &mut packed[s * NR_AVX512_F32 * kc..s * NR_AVX512_F32 * kc + NR_AVX512_F32 * kc];
        let n_in_strip = (nc - col_base).min(NR_AVX512_F32);

        if n_in_strip == NR_AVX512_F32 {
            for k in 0..kc {
                for j in 0..NR_AVX512_F32 {
                    strip[k * NR_AVX512_F32 + j] = unsafe { *b.add(k * ldb + col_base + j) };
                }
            }
        } else {
            for k in 0..kc {
                for j in 0..NR_AVX512_F32 {
                    strip[k * NR_AVX512_F32 + j] = if j < n_in_strip {
                        unsafe { *b.add(k * ldb + col_base + j) }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_a_round_trip_full() {
        let mc: usize = 16;
        let kc: usize = 8;
        let a: Vec<f64> = (0..mc * kc).map(|i| i as f64).collect();
        let mut packed = vec![0.0_f64; mc.div_ceil(MR) * MR * kc];
        pack_a(mc, kc, a.as_ptr(), kc, &mut packed);

        for s in 0..mc.div_ceil(MR) {
            for k in 0..kc {
                for i in 0..MR {
                    let row = s * MR + i;
                    let want = if row < mc { a[row * kc + k] } else { 0.0 };
                    let got = packed[s * MR * kc + k * MR + i];
                    assert_eq!(got, want, "s={s} k={k} i={i}");
                }
            }
        }
    }

    #[test]
    fn pack_a_pads_partial_strip() {
        let mc = 3; // less than MR=4 → single partial strip
        let kc = 4;
        let a: Vec<f64> = (0..mc * kc).map(|i| (i + 1) as f64).collect();
        let mut packed = vec![999.0_f64; MR * kc];
        pack_a(mc, kc, a.as_ptr(), kc, &mut packed);

        for k in 0..kc {
            for i in 0..MR {
                let want = if i < mc { a[i * kc + k] } else { 0.0 };
                assert_eq!(packed[k * MR + i], want, "k={k} i={i}");
            }
        }
    }

    #[test]
    fn pack_b_round_trip_full() {
        let kc: usize = 4;
        let nc: usize = 18;
        let b: Vec<f64> = (0..kc * nc).map(|i| (i as f64) * 0.5).collect();
        let mut packed = vec![0.0_f64; nc.div_ceil(NR) * NR * kc];
        pack_b(kc, nc, b.as_ptr(), nc, &mut packed);

        for s in 0..nc.div_ceil(NR) {
            for k in 0..kc {
                for j in 0..NR {
                    let col = s * NR + j;
                    let want = if col < nc { b[k * nc + col] } else { 0.0 };
                    let got = packed[s * NR * kc + k * NR + j];
                    assert_eq!(got, want, "s={s} k={k} j={j}");
                }
            }
        }
    }

    #[test]
    fn pack_b_pads_partial_strip() {
        let kc = 3;
        let nc = 4; // less than NR=6
        let b: Vec<f64> = (0..kc * nc).map(|i| (i + 1) as f64 * 0.7).collect();
        let mut packed = vec![999.0_f64; NR * kc];
        pack_b(kc, nc, b.as_ptr(), nc, &mut packed);

        for k in 0..kc {
            for j in 0..NR {
                let want = if j < nc { b[k * nc + j] } else { 0.0 };
                assert_eq!(packed[k * NR + j], want, "k={k} j={j}");
            }
        }
    }
}
