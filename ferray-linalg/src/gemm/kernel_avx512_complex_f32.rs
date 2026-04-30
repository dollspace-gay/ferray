// AVX-512F CGEMM (Complex<f32>) micro-kernel — translated from
// `kernel/x86_64/cgemm_kernel_8x2_skylakex.c` lineage in OpenBLAS,
// adapted to ferray's row-major contract. Mirrors our AVX2 CGEMM 4x4
// kernel design but with AVX-512's wider lane count (16 f32 vs 8 per
// register), so we double the column count to 8 complex per ymm... err
// per zmm.
//
// Each zmm holds 16 f32 = 8 Complex<f32>. Register tile: 4 rows × 8
// complex cols, 8 zmm accumulators (4 acc_re + 4 acc_im).
//
// Per K iteration: 1 zmm B-load (16 f32 = 8 complex), 1 vpermilps to
// produce B_swap (swap each adjacent (re, im) pair within 64-bit
// subregions: imm = 0b10110001), 8 broadcasts of A (re + im for 4
// rows), 8 FMAs.
//
// Each FMA processes 16 f32 = 8 complex multiplies = 32 real FLOPs,
// so 8 FMAs = 256 real FLOPs / 4 cycles = 64 FLOPs/cycle = AVX-512
// CGEMM peak (2x AVX-512 ZGEMM, 2x AVX2 CGEMM).
//
// AVX-512 has no native vaddsubps; we emulate via _mm512_fmaddsub_ps
// at SAVE — same pattern as the AVX-512 ZGEMM kernel.

use core::arch::x86_64::{
    __m512, _MM_HINT_T0, _mm_prefetch, _mm512_fmadd_ps, _mm512_fmaddsub_ps, _mm512_loadu_ps,
    _mm512_permute_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps,
};

pub const MR_AVX512_C32: usize = 4;
pub const NR_AVX512_C32: usize = 8;

const A_PF_FLOATS: usize = 512 / 4;
const B_PF_FLOATS: usize = 320 / 4;

/// 4x8 AVX-512 CGEMM micro-kernel.
///
/// SAFETY: caller guarantees AVX-512F availability; `packed_a >=
/// kc * MR_AVX512_C32 * 2 floats`, `packed_b >= kc * NR_AVX512_C32 * 2`.
/// `c_rs`/`c_cs` are row/column strides **in complex elements**.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn kernel_4x8_c32_avx512(
    kc: usize,
    alpha_re: f32,
    alpha_im: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    let mut acc_re_0 = _mm512_setzero_ps();
    let mut acc_re_1 = _mm512_setzero_ps();
    let mut acc_re_2 = _mm512_setzero_ps();
    let mut acc_re_3 = _mm512_setzero_ps();
    let mut acc_im_0 = _mm512_setzero_ps();
    let mut acc_im_1 = _mm512_setzero_ps();
    let mut acc_im_2 = _mm512_setzero_ps();
    let mut acc_im_3 = _mm512_setzero_ps();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_FLOATS).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_FLOATS).cast());

        // Load B[k, 0..8] = 8 complex = 16 f32.
        let b_v = _mm512_loadu_ps(b);
        // Swap each adjacent (re, im) pair within 64-bit subregions.
        // For _mm512_permute_ps: imm 0b10110001 = MM_SHUFFLE(2,3,0,1)
        // selects: out[0]=in[1], out[1]=in[0], out[2]=in[3], out[3]=in[2]
        // applied per 128-bit subregion, so each (re, im) pair swaps.
        let b_swap = _mm512_permute_ps::<0b10110001>(b_v);

        // Row 0.
        let a0_re = _mm512_set1_ps(*a);
        let a0_im = _mm512_set1_ps(*a.add(1));
        acc_re_0 = _mm512_fmadd_ps(a0_re, b_v, acc_re_0);
        acc_im_0 = _mm512_fmadd_ps(a0_im, b_swap, acc_im_0);

        // Row 1.
        let a1_re = _mm512_set1_ps(*a.add(2));
        let a1_im = _mm512_set1_ps(*a.add(3));
        acc_re_1 = _mm512_fmadd_ps(a1_re, b_v, acc_re_1);
        acc_im_1 = _mm512_fmadd_ps(a1_im, b_swap, acc_im_1);

        // Row 2.
        let a2_re = _mm512_set1_ps(*a.add(4));
        let a2_im = _mm512_set1_ps(*a.add(5));
        acc_re_2 = _mm512_fmadd_ps(a2_re, b_v, acc_re_2);
        acc_im_2 = _mm512_fmadd_ps(a2_im, b_swap, acc_im_2);

        // Row 3.
        let a3_re = _mm512_set1_ps(*a.add(6));
        let a3_im = _mm512_set1_ps(*a.add(7));
        acc_re_3 = _mm512_fmadd_ps(a3_re, b_v, acc_re_3);
        acc_im_3 = _mm512_fmadd_ps(a3_im, b_swap, acc_im_3);

        a = a.add(MR_AVX512_C32 * 2);
        b = b.add(NR_AVX512_C32 * 2);
    }

    // Combine via fmaddsub(ones, acc_re, acc_im) — emulates vaddsubps
    // which doesn't exist in AVX-512.
    let ones = _mm512_set1_ps(1.0);
    let c0 = _mm512_fmaddsub_ps(ones, acc_re_0, acc_im_0);
    let c1 = _mm512_fmaddsub_ps(ones, acc_re_1, acc_im_1);
    let c2 = _mm512_fmaddsub_ps(ones, acc_re_2, acc_im_2);
    let c3 = _mm512_fmaddsub_ps(ones, acc_re_3, acc_im_3);

    let c0 = scale_alpha_c32_avx512(c0, alpha_re, alpha_im);
    let c1 = scale_alpha_c32_avx512(c1, alpha_re, alpha_im);
    let c2 = scale_alpha_c32_avx512(c2, alpha_re, alpha_im);
    let c3 = scale_alpha_c32_avx512(c3, alpha_re, alpha_im);

    if c_cs == 1 && c_rs > 0 {
        let ldc_floats = (c_rs as usize) * 2;
        store_row_8c32(c, 0, ldc_floats, c0, beta_re, beta_im);
        store_row_8c32(c, 1, ldc_floats, c1, beta_re, beta_im);
        store_row_8c32(c, 2, ldc_floats, c2, beta_re, beta_im);
        store_row_8c32(c, 3, ldc_floats, c3, beta_re, beta_im);
    } else {
        store_strided_row_8c32(c, c_rs, c_cs, 0, c0, beta_re, beta_im);
        store_strided_row_8c32(c, c_rs, c_cs, 1, c1, beta_re, beta_im);
        store_strided_row_8c32(c, c_rs, c_cs, 2, c2, beta_re, beta_im);
        store_strided_row_8c32(c, c_rs, c_cs, 3, c3, beta_re, beta_im);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn scale_alpha_c32_avx512(c: __m512, alpha_re: f32, alpha_im: f32) -> __m512 {
    if alpha_re == 1.0 && alpha_im == 0.0 {
        return c;
    }
    let c_swap = _mm512_permute_ps::<0b10110001>(c);
    let alpha_re_v = _mm512_set1_ps(alpha_re);
    let alpha_im_v = _mm512_set1_ps(alpha_im);
    let term_re = _mm512_fmadd_ps(alpha_re_v, c, _mm512_setzero_ps());
    let term_im = _mm512_fmadd_ps(alpha_im_v, c_swap, _mm512_setzero_ps());
    _mm512_fmaddsub_ps(_mm512_set1_ps(1.0), term_re, term_im)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_row_8c32(
    c: *mut f32,
    row: usize,
    ldc_floats: usize,
    new_val: __m512,
    beta_re: f32,
    beta_im: f32,
) {
    let p = c.add(row * ldc_floats);
    if beta_re == 0.0 && beta_im == 0.0 {
        _mm512_storeu_ps(p, new_val);
        return;
    }
    let cur = _mm512_loadu_ps(p);
    if beta_re == 1.0 && beta_im == 0.0 {
        _mm512_storeu_ps(p, _mm512_fmadd_ps(_mm512_set1_ps(1.0), cur, new_val));
        return;
    }
    let cur_swap = _mm512_permute_ps::<0b10110001>(cur);
    let beta_re_v = _mm512_set1_ps(beta_re);
    let beta_im_v = _mm512_set1_ps(beta_im);
    let term_re = _mm512_fmadd_ps(beta_re_v, cur, _mm512_setzero_ps());
    let term_im = _mm512_fmadd_ps(beta_im_v, cur_swap, _mm512_setzero_ps());
    let scaled_cur = _mm512_fmaddsub_ps(_mm512_set1_ps(1.0), term_re, term_im);
    let result = _mm512_fmadd_ps(_mm512_set1_ps(1.0), scaled_cur, new_val);
    _mm512_storeu_ps(p, result);
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_strided_row_8c32(
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
    row: usize,
    new_val: __m512,
    beta_re: f32,
    beta_im: f32,
) {
    let mut buf = [0.0_f32; 16];
    _mm512_storeu_ps(buf.as_mut_ptr(), new_val);
    let row_off_floats = (row as isize) * c_rs * 2;
    for j in 0..NR_AVX512_C32 {
        let col_off_floats = (j as isize) * c_cs * 2;
        let p_re = c.offset(row_off_floats + col_off_floats);
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

/// Edge-case AVX-512 CGEMM kernel.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_c32_avx512(
    m: usize,
    n: usize,
    kc: usize,
    alpha_re: f32,
    alpha_im: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    c_rs: isize,
    c_cs: isize,
) {
    debug_assert!(m <= MR_AVX512_C32);
    debug_assert!(n <= NR_AVX512_C32);
    let mut tile_re = [[0.0_f32; NR_AVX512_C32]; MR_AVX512_C32];
    let mut tile_im = [[0.0_f32; NR_AVX512_C32]; MR_AVX512_C32];
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
        a = a.add(MR_AVX512_C32 * 2);
        b = b.add(NR_AVX512_C32 * 2);
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

    fn naive_cgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..m {
            for j in 0..n {
                let mut acc_re = 0.0_f32;
                let mut acc_im = 0.0_f32;
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

    fn pack_a(a: &[f32], kc: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; MR_AVX512_C32 * 2 * kc];
        for i in 0..MR_AVX512_C32 {
            for k in 0..kc {
                packed[k * MR_AVX512_C32 * 2 + i * 2] = a[i * kc * 2 + k * 2];
                packed[k * MR_AVX512_C32 * 2 + i * 2 + 1] = a[i * kc * 2 + k * 2 + 1];
            }
        }
        packed
    }

    fn pack_b(b: &[f32], kc: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; kc * NR_AVX512_C32 * 2];
        for k in 0..kc {
            for j in 0..NR_AVX512_C32 {
                packed[k * NR_AVX512_C32 * 2 + j * 2] = b[k * NR_AVX512_C32 * 2 + j * 2];
                packed[k * NR_AVX512_C32 * 2 + j * 2 + 1] = b[k * NR_AVX512_C32 * 2 + j * 2 + 1];
            }
        }
        packed
    }

    /// Skips on AVX2-only CPUs.
    #[test]
    fn kernel_4x8_c32_avx512_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 16;
        let a: Vec<f32> = (0..MR_AVX512_C32 * kc * 2)
            .map(|i| (i as f32) * 0.013 - 0.7)
            .collect();
        let b: Vec<f32> = (0..kc * NR_AVX512_C32 * 2)
            .map(|i| (i as f32) * 0.011 + 0.3)
            .collect();
        let mut c_ours = vec![0.0_f32; MR_AVX512_C32 * NR_AVX512_C32 * 2];
        let mut c_ref = vec![0.0_f32; MR_AVX512_C32 * NR_AVX512_C32 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_c32_avx512(
                kc,
                1.0,
                0.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_AVX512_C32 as isize,
                1,
            );
        }
        naive_cgemm(MR_AVX512_C32, NR_AVX512_C32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_AVX512_C32 * NR_AVX512_C32 * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-3,
                "mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_4x8_c32_avx512_alpha_complex() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let kc = 8;
        let a: Vec<f32> = (0..MR_AVX512_C32 * kc * 2)
            .map(|i| (i as f32) * 0.02)
            .collect();
        let b: Vec<f32> = (0..kc * NR_AVX512_C32 * 2)
            .map(|i| (i as f32) * 0.03)
            .collect();
        let mut c_ours = vec![0.0_f32; MR_AVX512_C32 * NR_AVX512_C32 * 2];
        let mut c_ref = vec![0.0_f32; MR_AVX512_C32 * NR_AVX512_C32 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x8_c32_avx512(
                kc,
                0.5,
                0.7,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_AVX512_C32 as isize,
                1,
            );
        }
        naive_cgemm(MR_AVX512_C32, NR_AVX512_C32, kc, &a, &b, &mut c_ref);
        for i in 0..MR_AVX512_C32 * NR_AVX512_C32 {
            let cr = c_ref[i * 2];
            let ci = c_ref[i * 2 + 1];
            c_ref[i * 2] = 0.5 * cr - 0.7 * ci;
            c_ref[i * 2 + 1] = 0.5 * ci + 0.7 * cr;
        }
        for i in 0..MR_AVX512_C32 * NR_AVX512_C32 * 2 {
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
