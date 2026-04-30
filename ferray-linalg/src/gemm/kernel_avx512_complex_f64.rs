// AVX-512F ZGEMM (Complex<f64>) micro-kernel — translated from
// `kernel/x86_64/zgemm_kernel_4x2_skylakex.c` lineage in OpenBLAS,
// adapted to ferray's row-major contract using the same row-oriented
// expanded-accumulator pattern as our AVX2 ZGEMM 4x2 kernel.
//
// AVX-512F doubles the f64 lane count vs AVX2 (8 vs 4 per register),
// so a single zmm holds 4 Complex<f64> instead of 2. We use the same
// register tile shape as AVX2 in row count (4 rows) but double the
// column count (4 complex cols, all in one zmm per row) for a wider
// 4×4-complex tile.
//
// Notable AVX-512 difference: there is no native `vaddsubpd` (the
// instruction was not extended to 512-bit width). At SAVE we emulate
// via `_mm512_fmaddsub_pd(ones, acc_re, acc_im)`:
//   result[even] = 1*acc_re - acc_im  (real part of complex product)
//   result[odd]  = 1*acc_re + acc_im  (imag part)
//
// Register usage (32 zmm available, we use 11):
//   zmm0       broadcast scratch (current A[i, k].re)
//   zmm1       broadcast scratch (current A[i, k].im)
//   zmm2       current B row (8 doubles = NR complex)
//   zmm3       B-swap (vpermilpd $0x55 of zmm2)
//   zmm4..zmm7   acc_re for rows 0..3
//   zmm8..zmm11  acc_im for rows 0..3
//
// Per K iteration: 1 zmm B-load, 1 vpermilpd, 8 broadcasts of A, 8 FMAs.
// Each FMA is 4 complex multiplies = 16 real FLOPs, so 8 FMAs = 128 real
// FLOPs / 4 cycles = 32 FLOPs/cycle = AVX-512 ZGEMM peak (2x AVX2 ZGEMM).

use core::arch::x86_64::{
    __m512d, _MM_HINT_T0, _mm_prefetch, _mm512_fmadd_pd, _mm512_fmaddsub_pd, _mm512_loadu_pd,
    _mm512_permute_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd,
};

pub const MR_AVX512_C64: usize = 4;
pub const NR_AVX512_C64: usize = 4;

const A_PF_DOUBLES: usize = 512 / 8;
const B_PF_DOUBLES: usize = 320 / 8;

/// 4×4 AVX-512 ZGEMM micro-kernel.
///
/// SAFETY: caller guarantees AVX-512F availability; `packed_a >=
/// kc * MR_AVX512_C64 * 2 doubles`, `packed_b >= kc * NR_AVX512_C64 * 2`.
/// `c_rs`/`c_cs` are row/column strides **in complex elements**.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn kernel_4x4_c64_avx512(
    kc: usize,
    alpha_re: f64,
    alpha_im: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    let mut acc_re_0 = _mm512_setzero_pd();
    let mut acc_re_1 = _mm512_setzero_pd();
    let mut acc_re_2 = _mm512_setzero_pd();
    let mut acc_re_3 = _mm512_setzero_pd();
    let mut acc_im_0 = _mm512_setzero_pd();
    let mut acc_im_1 = _mm512_setzero_pd();
    let mut acc_im_2 = _mm512_setzero_pd();
    let mut acc_im_3 = _mm512_setzero_pd();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_DOUBLES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_DOUBLES).cast());

        // Load B[k, 0..4] = 4 complex = 8 doubles.
        let b_v = _mm512_loadu_pd(b);
        // Swap each adjacent f64 pair within 128-bit subregions.
        // imm = 0b01010101 = 0x55: every output lane selects the
        // OTHER lane within its 64-bit pair.
        let b_swap = _mm512_permute_pd::<0b01010101>(b_v);

        // Row 0: A[0, k] = packed_a[0..2] = (a0_re, a0_im).
        let a0_re = _mm512_set1_pd(*a);
        let a0_im = _mm512_set1_pd(*a.add(1));
        acc_re_0 = _mm512_fmadd_pd(a0_re, b_v, acc_re_0);
        acc_im_0 = _mm512_fmadd_pd(a0_im, b_swap, acc_im_0);

        // Row 1.
        let a1_re = _mm512_set1_pd(*a.add(2));
        let a1_im = _mm512_set1_pd(*a.add(3));
        acc_re_1 = _mm512_fmadd_pd(a1_re, b_v, acc_re_1);
        acc_im_1 = _mm512_fmadd_pd(a1_im, b_swap, acc_im_1);

        // Row 2.
        let a2_re = _mm512_set1_pd(*a.add(4));
        let a2_im = _mm512_set1_pd(*a.add(5));
        acc_re_2 = _mm512_fmadd_pd(a2_re, b_v, acc_re_2);
        acc_im_2 = _mm512_fmadd_pd(a2_im, b_swap, acc_im_2);

        // Row 3.
        let a3_re = _mm512_set1_pd(*a.add(6));
        let a3_im = _mm512_set1_pd(*a.add(7));
        acc_re_3 = _mm512_fmadd_pd(a3_re, b_v, acc_re_3);
        acc_im_3 = _mm512_fmadd_pd(a3_im, b_swap, acc_im_3);

        a = a.add(MR_AVX512_C64 * 2);
        b = b.add(NR_AVX512_C64 * 2);
    }

    // Combine acc_re and acc_im: result = vaddsub(acc_re, acc_im).
    // AVX-512 has no addsubpd, so we emulate via fmaddsub(ones, acc_re, acc_im):
    //   result[even] = 1*acc_re - acc_im
    //   result[odd]  = 1*acc_re + acc_im
    let ones = _mm512_set1_pd(1.0);
    let c0 = _mm512_fmaddsub_pd(ones, acc_re_0, acc_im_0);
    let c1 = _mm512_fmaddsub_pd(ones, acc_re_1, acc_im_1);
    let c2 = _mm512_fmaddsub_pd(ones, acc_re_2, acc_im_2);
    let c3 = _mm512_fmaddsub_pd(ones, acc_re_3, acc_im_3);

    let c0 = scale_alpha_c64_avx512(c0, alpha_re, alpha_im);
    let c1 = scale_alpha_c64_avx512(c1, alpha_re, alpha_im);
    let c2 = scale_alpha_c64_avx512(c2, alpha_re, alpha_im);
    let c3 = scale_alpha_c64_avx512(c3, alpha_re, alpha_im);

    if c_cs == 1 && c_rs > 0 {
        let ldc_doubles = (c_rs as usize) * 2;
        store_row_4c64(c, 0, ldc_doubles, c0, beta_re, beta_im);
        store_row_4c64(c, 1, ldc_doubles, c1, beta_re, beta_im);
        store_row_4c64(c, 2, ldc_doubles, c2, beta_re, beta_im);
        store_row_4c64(c, 3, ldc_doubles, c3, beta_re, beta_im);
    } else {
        store_strided_row_4c64(c, c_rs, c_cs, 0, c0, beta_re, beta_im);
        store_strided_row_4c64(c, c_rs, c_cs, 1, c1, beta_re, beta_im);
        store_strided_row_4c64(c, c_rs, c_cs, 2, c2, beta_re, beta_im);
        store_strided_row_4c64(c, c_rs, c_cs, 3, c3, beta_re, beta_im);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn scale_alpha_c64_avx512(c: __m512d, alpha_re: f64, alpha_im: f64) -> __m512d {
    if alpha_re == 1.0 && alpha_im == 0.0 {
        return c;
    }
    let c_swap = _mm512_permute_pd::<0b01010101>(c);
    let alpha_re_v = _mm512_set1_pd(alpha_re);
    let alpha_im_v = _mm512_set1_pd(alpha_im);
    let term_re = _mm512_fmadd_pd(alpha_re_v, c, _mm512_setzero_pd());
    let term_im = _mm512_fmadd_pd(alpha_im_v, c_swap, _mm512_setzero_pd());
    _mm512_fmaddsub_pd(_mm512_set1_pd(1.0), term_re, term_im)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_row_4c64(
    c: *mut f64,
    row: usize,
    ldc_doubles: usize,
    new_val: __m512d,
    beta_re: f64,
    beta_im: f64,
) {
    let p = c.add(row * ldc_doubles);
    if beta_re == 0.0 && beta_im == 0.0 {
        _mm512_storeu_pd(p, new_val);
        return;
    }
    let cur = _mm512_loadu_pd(p);
    if beta_re == 1.0 && beta_im == 0.0 {
        // result = cur + new_val
        _mm512_storeu_pd(p, _mm512_fmadd_pd(_mm512_set1_pd(1.0), cur, new_val));
        return;
    }
    let cur_swap = _mm512_permute_pd::<0b01010101>(cur);
    let beta_re_v = _mm512_set1_pd(beta_re);
    let beta_im_v = _mm512_set1_pd(beta_im);
    let term_re = _mm512_fmadd_pd(beta_re_v, cur, _mm512_setzero_pd());
    let term_im = _mm512_fmadd_pd(beta_im_v, cur_swap, _mm512_setzero_pd());
    let scaled_cur = _mm512_fmaddsub_pd(_mm512_set1_pd(1.0), term_re, term_im);
    let result = _mm512_fmadd_pd(_mm512_set1_pd(1.0), scaled_cur, new_val);
    _mm512_storeu_pd(p, result);
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_strided_row_4c64(
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    new_val: __m512d,
    beta_re: f64,
    beta_im: f64,
) {
    let mut buf = [0.0_f64; 8];
    _mm512_storeu_pd(buf.as_mut_ptr(), new_val);
    let row_off_doubles = (row as isize) * c_rs * 2;
    for j in 0..NR_AVX512_C64 {
        let col_off_doubles = (j as isize) * c_cs * 2;
        let p_re = c.offset(row_off_doubles + col_off_doubles);
        let p_im = p_re.add(1);
        let v_re = buf[j * 2];
        let v_im = buf[j * 2 + 1];
        if beta_re == 0.0 && beta_im == 0.0 {
            *p_re = v_re;
            *p_im = v_im;
        } else {
            let cur_re = *p_re;
            let cur_im = *p_im;
            let scaled_re = beta_re * cur_re - beta_im * cur_im;
            let scaled_im = beta_re * cur_im + beta_im * cur_re;
            *p_re = scaled_re + v_re;
            *p_im = scaled_im + v_im;
        }
    }
}

/// Edge-case AVX-512 ZGEMM kernel for tail tiles (m < MR or n < NR).
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_c64_avx512(
    m: usize,
    n: usize,
    kc: usize,
    alpha_re: f64,
    alpha_im: f64,
    packed_a: *const f64,
    packed_b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    c_rs: isize,
    c_cs: isize,
) {
    debug_assert!(m <= MR_AVX512_C64);
    debug_assert!(n <= NR_AVX512_C64);
    let mut tile_re = [[0.0_f64; NR_AVX512_C64]; MR_AVX512_C64];
    let mut tile_im = [[0.0_f64; NR_AVX512_C64]; MR_AVX512_C64];
    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        for i in 0..m {
            let ar = *a.add(i * 2);
            let ai = *a.add(i * 2 + 1);
            for j in 0..n {
                let br = *b.add(j * 2);
                let bi = *b.add(j * 2 + 1);
                tile_re[i][j] += ar * br - ai * bi;
                tile_im[i][j] += ar * bi + ai * br;
            }
        }
        a = a.add(MR_AVX512_C64 * 2);
        b = b.add(NR_AVX512_C64 * 2);
    }
    for i in 0..m {
        for j in 0..n {
            let alpha_x_re = alpha_re * tile_re[i][j] - alpha_im * tile_im[i][j];
            let alpha_x_im = alpha_re * tile_im[i][j] + alpha_im * tile_re[i][j];
            let off = (i as isize) * c_rs * 2 + (j as isize) * c_cs * 2;
            let p_re = c.offset(off);
            let p_im = p_re.add(1);
            if beta_re == 0.0 && beta_im == 0.0 {
                *p_re = alpha_x_re;
                *p_im = alpha_x_im;
            } else {
                let cr = *p_re;
                let ci = *p_im;
                *p_re = beta_re * cr - beta_im * ci + alpha_x_re;
                *p_im = beta_re * ci + beta_im * cr + alpha_x_im;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_zgemm(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        for i in 0..m {
            for j in 0..n {
                let mut acc_re = 0.0;
                let mut acc_im = 0.0;
                for p in 0..k {
                    let ar = a[i * k * 2 + p * 2];
                    let ai = a[i * k * 2 + p * 2 + 1];
                    let br = b[p * n * 2 + j * 2];
                    let bi = b[p * n * 2 + j * 2 + 1];
                    acc_re += ar * br - ai * bi;
                    acc_im += ar * bi + ai * br;
                }
                c[i * n * 2 + j * 2] += acc_re;
                c[i * n * 2 + j * 2 + 1] += acc_im;
            }
        }
    }

    fn pack_a(a: &[f64], kc: usize) -> Vec<f64> {
        let mut packed = vec![0.0_f64; MR_AVX512_C64 * 2 * kc];
        for i in 0..MR_AVX512_C64 {
            for k in 0..kc {
                packed[k * MR_AVX512_C64 * 2 + i * 2] = a[i * kc * 2 + k * 2];
                packed[k * MR_AVX512_C64 * 2 + i * 2 + 1] = a[i * kc * 2 + k * 2 + 1];
            }
        }
        packed
    }

    fn pack_b(b: &[f64], kc: usize) -> Vec<f64> {
        let mut packed = vec![0.0_f64; kc * NR_AVX512_C64 * 2];
        for k in 0..kc {
            for j in 0..NR_AVX512_C64 {
                packed[k * NR_AVX512_C64 * 2 + j * 2] = b[k * NR_AVX512_C64 * 2 + j * 2];
                packed[k * NR_AVX512_C64 * 2 + j * 2 + 1] = b[k * NR_AVX512_C64 * 2 + j * 2 + 1];
            }
        }
        packed
    }

    /// Skips on AVX2-only CPUs.
    #[test]
    fn kernel_4x4_c64_avx512_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 16;
        let a: Vec<f64> = (0..MR_AVX512_C64 * kc * 2)
            .map(|i| (i as f64) * 0.013 - 0.7)
            .collect();
        let b: Vec<f64> = (0..kc * NR_AVX512_C64 * 2)
            .map(|i| (i as f64) * 0.011 + 0.3)
            .collect();
        let mut c_ours = vec![0.0_f64; MR_AVX512_C64 * NR_AVX512_C64 * 2];
        let mut c_ref = vec![0.0_f64; MR_AVX512_C64 * NR_AVX512_C64 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x4_c64_avx512(
                kc,
                1.0,
                0.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_AVX512_C64 as isize,
                1,
            );
        }
        naive_zgemm(MR_AVX512_C64, NR_AVX512_C64, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_C64 * NR_AVX512_C64 * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-9,
                "mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_4x4_c64_avx512_alpha_complex() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 8;
        let a: Vec<f64> = (0..MR_AVX512_C64 * kc * 2)
            .map(|i| (i as f64) * 0.02)
            .collect();
        let b: Vec<f64> = (0..kc * NR_AVX512_C64 * 2)
            .map(|i| (i as f64) * 0.03)
            .collect();
        let mut c_ours = vec![0.0_f64; MR_AVX512_C64 * NR_AVX512_C64 * 2];
        let mut c_ref = vec![0.0_f64; MR_AVX512_C64 * NR_AVX512_C64 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x4_c64_avx512(
                kc,
                0.5,
                0.7,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_AVX512_C64 as isize,
                1,
            );
        }
        naive_zgemm(MR_AVX512_C64, NR_AVX512_C64, kc, &a, &b, &mut c_ref);
        for i in 0..MR_AVX512_C64 * NR_AVX512_C64 {
            let cr = c_ref[i * 2];
            let ci = c_ref[i * 2 + 1];
            c_ref[i * 2] = 0.5 * cr - 0.7 * ci;
            c_ref[i * 2 + 1] = 0.5 * ci + 0.7 * cr;
        }
        for i in 0..MR_AVX512_C64 * NR_AVX512_C64 * 2 {
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
