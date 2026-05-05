// Packing routines for the NEON micro-kernels.
//
// The packing layout matches the AVX2 packs (kc-major MR-/NR-minor strips
// with zero padding for partial tiles) — only the SIMD intrinsics and tile
// shapes differ. Since pack code is bandwidth-bound rather than ALU-bound,
// scalar (auto-vectorisable) packing is sufficient: rustc's NEON
// auto-vectoriser turns the inner copy loops into vld1q/vst1q on
// aarch64 already.

use super::kernel_neon_complex_f32::{MR_NEON_C32, NR_NEON_C32};
use super::kernel_neon_complex_f64::{MR_NEON_C64, NR_NEON_C64};
use super::kernel_neon_f32::{MR_NEON_F32, NR_NEON_F32};
use super::kernel_neon_f64::{MR_NEON_F64, NR_NEON_F64};
use super::kernel_neon_i8::{MR_NEON_I8, NR_NEON_I8};
use super::kernel_neon_i8s::{MR_NEON_I8S, NR_NEON_I8S};
use super::kernel_neon_i16::{MR_NEON_I16, NR_NEON_I16};

// ---------------- f64 ----------------

pub fn pack_a_neon_f64(mc: usize, kc: usize, a: *const f64, lda: usize, packed: &mut [f64]) {
    let n_strips = mc.div_ceil(MR_NEON_F64);
    debug_assert!(packed.len() >= n_strips * MR_NEON_F64 * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_F64;
        let strip = &mut packed[s * MR_NEON_F64 * kc..s * MR_NEON_F64 * kc + MR_NEON_F64 * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_F64);
        for k in 0..kc {
            for i in 0..MR_NEON_F64 {
                strip[k * MR_NEON_F64 + i] = if i < m_in_strip {
                    unsafe { *a.add((row_base + i) * lda + k) }
                } else {
                    0.0
                };
            }
        }
    }
}

pub fn pack_b_neon_f64(kc: usize, nc: usize, b: *const f64, ldb: usize, packed: &mut [f64]) {
    let n_strips = nc.div_ceil(NR_NEON_F64);
    debug_assert!(packed.len() >= n_strips * NR_NEON_F64 * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_F64;
        let strip = &mut packed[s * NR_NEON_F64 * kc..s * NR_NEON_F64 * kc + NR_NEON_F64 * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_F64);
        for k in 0..kc {
            for j in 0..NR_NEON_F64 {
                strip[k * NR_NEON_F64 + j] = if j < n_in_strip {
                    unsafe { *b.add(k * ldb + col_base + j) }
                } else {
                    0.0
                };
            }
        }
    }
}

// ---------------- f32 ----------------

pub fn pack_a_neon_f32(mc: usize, kc: usize, a: *const f32, lda: usize, packed: &mut [f32]) {
    let n_strips = mc.div_ceil(MR_NEON_F32);
    debug_assert!(packed.len() >= n_strips * MR_NEON_F32 * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_F32;
        let strip = &mut packed[s * MR_NEON_F32 * kc..s * MR_NEON_F32 * kc + MR_NEON_F32 * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_F32);
        for k in 0..kc {
            for i in 0..MR_NEON_F32 {
                strip[k * MR_NEON_F32 + i] = if i < m_in_strip {
                    unsafe { *a.add((row_base + i) * lda + k) }
                } else {
                    0.0
                };
            }
        }
    }
}

pub fn pack_b_neon_f32(kc: usize, nc: usize, b: *const f32, ldb: usize, packed: &mut [f32]) {
    let n_strips = nc.div_ceil(NR_NEON_F32);
    debug_assert!(packed.len() >= n_strips * NR_NEON_F32 * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_F32;
        let strip = &mut packed[s * NR_NEON_F32 * kc..s * NR_NEON_F32 * kc + NR_NEON_F32 * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_F32);
        for k in 0..kc {
            for j in 0..NR_NEON_F32 {
                strip[k * NR_NEON_F32 + j] = if j < n_in_strip {
                    unsafe { *b.add(k * ldb + col_base + j) }
                } else {
                    0.0
                };
            }
        }
    }
}

// ---------------- Complex<f64> ----------------
// Each complex element occupies 2 doubles [re, im]. lda/ldb are row
// strides in COMPLEX elements (matching the public API).

pub fn pack_a_neon_c64(mc: usize, kc: usize, a: *const f64, lda: usize, packed: &mut [f64]) {
    let n_strips = mc.div_ceil(MR_NEON_C64);
    debug_assert!(packed.len() >= n_strips * MR_NEON_C64 * 2 * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_C64;
        let strip_start = s * MR_NEON_C64 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + MR_NEON_C64 * 2 * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_C64);
        for k in 0..kc {
            for i in 0..MR_NEON_C64 {
                let dst_off = k * MR_NEON_C64 * 2 + i * 2;
                if i < m_in_strip {
                    let src_off = (row_base + i) * lda * 2 + k * 2;
                    strip[dst_off] = unsafe { *a.add(src_off) };
                    strip[dst_off + 1] = unsafe { *a.add(src_off + 1) };
                } else {
                    strip[dst_off] = 0.0;
                    strip[dst_off + 1] = 0.0;
                }
            }
        }
    }
}

pub fn pack_b_neon_c64(kc: usize, nc: usize, b: *const f64, ldb: usize, packed: &mut [f64]) {
    let n_strips = nc.div_ceil(NR_NEON_C64);
    debug_assert!(packed.len() >= n_strips * NR_NEON_C64 * 2 * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_C64;
        let strip_start = s * NR_NEON_C64 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + NR_NEON_C64 * 2 * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_C64);
        for k in 0..kc {
            for j in 0..NR_NEON_C64 {
                let dst_off = k * NR_NEON_C64 * 2 + j * 2;
                if j < n_in_strip {
                    let src_off = k * ldb * 2 + (col_base + j) * 2;
                    strip[dst_off] = unsafe { *b.add(src_off) };
                    strip[dst_off + 1] = unsafe { *b.add(src_off + 1) };
                } else {
                    strip[dst_off] = 0.0;
                    strip[dst_off + 1] = 0.0;
                }
            }
        }
    }
}

// ---------------- Complex<f32> ----------------

pub fn pack_a_neon_c32(mc: usize, kc: usize, a: *const f32, lda: usize, packed: &mut [f32]) {
    let n_strips = mc.div_ceil(MR_NEON_C32);
    debug_assert!(packed.len() >= n_strips * MR_NEON_C32 * 2 * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_C32;
        let strip_start = s * MR_NEON_C32 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + MR_NEON_C32 * 2 * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_C32);
        for k in 0..kc {
            for i in 0..MR_NEON_C32 {
                let dst_off = k * MR_NEON_C32 * 2 + i * 2;
                if i < m_in_strip {
                    let src_off = (row_base + i) * lda * 2 + k * 2;
                    strip[dst_off] = unsafe { *a.add(src_off) };
                    strip[dst_off + 1] = unsafe { *a.add(src_off + 1) };
                } else {
                    strip[dst_off] = 0.0;
                    strip[dst_off + 1] = 0.0;
                }
            }
        }
    }
}

pub fn pack_b_neon_c32(kc: usize, nc: usize, b: *const f32, ldb: usize, packed: &mut [f32]) {
    let n_strips = nc.div_ceil(NR_NEON_C32);
    debug_assert!(packed.len() >= n_strips * NR_NEON_C32 * 2 * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_C32;
        let strip_start = s * NR_NEON_C32 * 2 * kc;
        let strip = &mut packed[strip_start..strip_start + NR_NEON_C32 * 2 * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_C32);
        for k in 0..kc {
            for j in 0..NR_NEON_C32 {
                let dst_off = k * NR_NEON_C32 * 2 + j * 2;
                if j < n_in_strip {
                    let src_off = k * ldb * 2 + (col_base + j) * 2;
                    strip[dst_off] = unsafe { *b.add(src_off) };
                    strip[dst_off + 1] = unsafe { *b.add(src_off + 1) };
                } else {
                    strip[dst_off] = 0.0;
                    strip[dst_off + 1] = 0.0;
                }
            }
        }
    }
}

// ---------------- i16 ----------------

pub fn pack_a_neon_i16(mc: usize, kc: usize, a: *const i16, lda: usize, packed: &mut [i16]) {
    let n_strips = mc.div_ceil(MR_NEON_I16);
    debug_assert!(packed.len() >= n_strips * MR_NEON_I16 * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_I16;
        let strip = &mut packed[s * MR_NEON_I16 * kc..s * MR_NEON_I16 * kc + MR_NEON_I16 * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_I16);
        for k in 0..kc {
            for i in 0..MR_NEON_I16 {
                strip[k * MR_NEON_I16 + i] = if i < m_in_strip {
                    unsafe { *a.add((row_base + i) * lda + k) }
                } else {
                    0
                };
            }
        }
    }
}

pub fn pack_b_neon_i16(kc: usize, nc: usize, b: *const i16, ldb: usize, packed: &mut [i16]) {
    let n_strips = nc.div_ceil(NR_NEON_I16);
    debug_assert!(packed.len() >= n_strips * NR_NEON_I16 * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_I16;
        let strip = &mut packed[s * NR_NEON_I16 * kc..s * NR_NEON_I16 * kc + NR_NEON_I16 * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_I16);
        for k in 0..kc {
            for j in 0..NR_NEON_I16 {
                strip[k * NR_NEON_I16 + j] = if j < n_in_strip {
                    unsafe { *b.add(k * ldb + col_base + j) }
                } else {
                    0
                };
            }
        }
    }
}

// ---------------- i8 (u8 × i8) ----------------

pub fn pack_a_neon_u8(mc: usize, kc: usize, a: *const u8, lda: usize, packed: &mut [u8]) {
    let n_strips = mc.div_ceil(MR_NEON_I8);
    debug_assert!(packed.len() >= n_strips * MR_NEON_I8 * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_I8;
        let strip = &mut packed[s * MR_NEON_I8 * kc..s * MR_NEON_I8 * kc + MR_NEON_I8 * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_I8);
        for k in 0..kc {
            for i in 0..MR_NEON_I8 {
                strip[k * MR_NEON_I8 + i] = if i < m_in_strip {
                    unsafe { *a.add((row_base + i) * lda + k) }
                } else {
                    0
                };
            }
        }
    }
}

pub fn pack_b_neon_i8(kc: usize, nc: usize, b: *const i8, ldb: usize, packed: &mut [i8]) {
    let n_strips = nc.div_ceil(NR_NEON_I8);
    debug_assert!(packed.len() >= n_strips * NR_NEON_I8 * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_I8;
        let strip = &mut packed[s * NR_NEON_I8 * kc..s * NR_NEON_I8 * kc + NR_NEON_I8 * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_I8);
        for k in 0..kc {
            for j in 0..NR_NEON_I8 {
                strip[k * NR_NEON_I8 + j] = if j < n_in_strip {
                    unsafe { *b.add(k * ldb + col_base + j) }
                } else {
                    0
                };
            }
        }
    }
}

// ---------------- i8s (i8 × i8 → i32) ----------------

pub fn pack_a_neon_i8s(mc: usize, kc: usize, a: *const i8, lda: usize, packed: &mut [i8]) {
    let n_strips = mc.div_ceil(MR_NEON_I8S);
    debug_assert!(packed.len() >= n_strips * MR_NEON_I8S * kc);
    for s in 0..n_strips {
        let row_base = s * MR_NEON_I8S;
        let strip = &mut packed[s * MR_NEON_I8S * kc..s * MR_NEON_I8S * kc + MR_NEON_I8S * kc];
        let m_in_strip = (mc - row_base).min(MR_NEON_I8S);
        for k in 0..kc {
            for i in 0..MR_NEON_I8S {
                strip[k * MR_NEON_I8S + i] = if i < m_in_strip {
                    unsafe { *a.add((row_base + i) * lda + k) }
                } else {
                    0
                };
            }
        }
    }
}

pub fn pack_b_neon_i8s(kc: usize, nc: usize, b: *const i8, ldb: usize, packed: &mut [i8]) {
    let n_strips = nc.div_ceil(NR_NEON_I8S);
    debug_assert!(packed.len() >= n_strips * NR_NEON_I8S * kc);
    for s in 0..n_strips {
        let col_base = s * NR_NEON_I8S;
        let strip = &mut packed[s * NR_NEON_I8S * kc..s * NR_NEON_I8S * kc + NR_NEON_I8S * kc];
        let n_in_strip = (nc - col_base).min(NR_NEON_I8S);
        for k in 0..kc {
            for j in 0..NR_NEON_I8S {
                strip[k * NR_NEON_I8S + j] = if j < n_in_strip {
                    unsafe { *b.add(k * ldb + col_base + j) }
                } else {
                    0
                };
            }
        }
    }
}
