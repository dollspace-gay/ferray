// AVX2+FMA CGEMM (Complex<f32>) micro-kernel — translated from OpenBLAS's
// `kernel/x86_64/cgemm_kernel_8x2_haswell.c` (BSD-3 licensed).
//
// OpenBLAS uses an 8x2 column-oriented register tile with the "expanded
// accumulators" pattern (separate acc_re / acc_im that combine via
// vaddsubps at SAVE) for small N, plus a "contracted" vfmaddsub231ps
// path for larger N. For ferray's row-major contract we use the same
// expanded-accumulators pattern as our ZGEMM 4x2 kernel but with f32
// lanes (8 per ymm vs 4 for f64), and a wider register tile of 4 rows
// × 4 complex cols (each ymm holds 4 complex = 8 f32).
//
// Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
//                 = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
//
// Per K iteration we maintain:
//   acc_re[i] += a_re_bcast * B          (partial a_re lanes)
//   acc_im[i] += a_im_bcast * B_swap     (partial a_im lanes, with swapped pairs)
//
// where B_swap = vpermilps(0b10110001)(B) = swap each adjacent (re, im)
// pair within every 128-bit lane. After the K loop, vaddsubps combines:
//   c[i] = vaddsubps(acc_re[i], acc_im[i])
//   - even lanes (real slot) = acc_re - acc_im = sum(a_re*b.re) - sum(a_im*b.im) ← real part
//   - odd lanes (imag slot)  = acc_re + acc_im = sum(a_re*b.im) + sum(a_im*b.re) ← imag part
//
// Register usage (16 ymm available, we use 11):
//   ymm0       broadcast scratch (current A[i, k].re)
//   ymm1       broadcast scratch (current A[i, k].im)
//   ymm2       current B row (8 f32 = NR_C32 complex)
//   ymm3       B-swap (vpermilps 0b10110001 of ymm2)
//   ymm4..ymm7   acc_re for rows 0..3
//   ymm8..ymm11  acc_im for rows 0..3
//
// Per K iteration: 1 ymm B-load, 1 vpermilps, 8 broadcasts of A
// (re + im for 4 rows), 8 FMAs. Each FMA processes 8 f32 = 4 complex,
// yielding 16 real FLOPs per FMA × 8 FMAs = 128 real FLOPs / 4 cycles =
// 32 FLOPs/cycle = AVX2-FMA peak for f32 (twice ZGEMM peak).

use core::arch::x86_64::{
    __m256, _MM_HINT_T0, _mm_prefetch, _mm256_addsub_ps, _mm256_broadcast_ss, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_permute_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

/// Number of complex rows in the register tile.
pub const MR_C32: usize = 4;
/// Number of complex cols (4 complex = 8 f32 = 1 ymm).
pub const NR_C32: usize = 4;

const A_PF_FLOATS: usize = 512 / 4;
const B_PF_FLOATS: usize = 160 / 4;

/// 4x4 CGEMM micro-kernel. Treats `Complex<f32>` arrays as `f32` slices
/// of double the length: each complex element is 2 contiguous f32s
/// `[re, im]` (matches `num_complex::Complex<f32>` repr(C)).
///
/// `kc` is the depth in complex elements. `c_rs`/`c_cs` are row/column
/// strides of C **in complex elements**.
///
/// SAFETY: caller guarantees AVX2+FMA availability, `packed_a >=
/// kc * MR_C32 * 2 floats`, `packed_b >= kc * NR_C32 * 2 floats`, and `c`
/// addresses an (m_tile, n_tile) sub-tile.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn kernel_4x4_c32(
    kc: usize,
    alpha_re: f32,
    alpha_im: f32,
    packed_a: *const f32,
    packed_b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    c_rs: isize, // row stride in COMPLEX elements
    c_cs: isize, // col stride in COMPLEX elements
) {
    // INIT — zero all 8 accumulators (4 acc_re + 4 acc_im).
    let mut acc_re_0 = _mm256_setzero_ps();
    let mut acc_re_1 = _mm256_setzero_ps();
    let mut acc_re_2 = _mm256_setzero_ps();
    let mut acc_re_3 = _mm256_setzero_ps();
    let mut acc_im_0 = _mm256_setzero_ps();
    let mut acc_im_1 = _mm256_setzero_ps();
    let mut acc_im_2 = _mm256_setzero_ps();
    let mut acc_im_3 = _mm256_setzero_ps();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_FLOATS).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_FLOATS).cast());

        // Load B[k, 0..4] = 4 complex = 8 f32.
        let b_v = _mm256_loadu_ps(b);
        // B_swap = swap adjacent (re, im) pairs within each 128-bit lane.
        // imm = 0b10110001 = MM_SHUFFLE(2,3,0,1).
        let b_swap = _mm256_permute_ps::<0b10110001>(b_v);

        // Row 0.
        let a0_re = _mm256_broadcast_ss(&*a);
        let a0_im = _mm256_broadcast_ss(&*a.add(1));
        acc_re_0 = _mm256_fmadd_ps(a0_re, b_v, acc_re_0);
        acc_im_0 = _mm256_fmadd_ps(a0_im, b_swap, acc_im_0);

        // Row 1.
        let a1_re = _mm256_broadcast_ss(&*a.add(2));
        let a1_im = _mm256_broadcast_ss(&*a.add(3));
        acc_re_1 = _mm256_fmadd_ps(a1_re, b_v, acc_re_1);
        acc_im_1 = _mm256_fmadd_ps(a1_im, b_swap, acc_im_1);

        // Row 2.
        let a2_re = _mm256_broadcast_ss(&*a.add(4));
        let a2_im = _mm256_broadcast_ss(&*a.add(5));
        acc_re_2 = _mm256_fmadd_ps(a2_re, b_v, acc_re_2);
        acc_im_2 = _mm256_fmadd_ps(a2_im, b_swap, acc_im_2);

        // Row 3.
        let a3_re = _mm256_broadcast_ss(&*a.add(6));
        let a3_im = _mm256_broadcast_ss(&*a.add(7));
        acc_re_3 = _mm256_fmadd_ps(a3_re, b_v, acc_re_3);
        acc_im_3 = _mm256_fmadd_ps(a3_im, b_swap, acc_im_3);

        a = a.add(MR_C32 * 2);
        b = b.add(NR_C32 * 2);
    }

    // Combine accumulators via vaddsubps.
    let c0 = _mm256_addsub_ps(acc_re_0, acc_im_0);
    let c1 = _mm256_addsub_ps(acc_re_1, acc_im_1);
    let c2 = _mm256_addsub_ps(acc_re_2, acc_im_2);
    let c3 = _mm256_addsub_ps(acc_re_3, acc_im_3);

    // Apply complex alpha.
    let c0 = scale_alpha_c32(c0, alpha_re, alpha_im);
    let c1 = scale_alpha_c32(c1, alpha_re, alpha_im);
    let c2 = scale_alpha_c32(c2, alpha_re, alpha_im);
    let c3 = scale_alpha_c32(c3, alpha_re, alpha_im);

    // Store. Row-major fast path: each ymm holds one row's 4 complex
    // (8 contiguous f32). With c_cs == 1 (complex), we use a single
    // _mm256_storeu_ps per row.
    if c_cs == 1 && c_rs > 0 {
        let ldc_floats = (c_rs as usize) * 2;
        store_row_4c32(c, 0, ldc_floats, c0, beta_re, beta_im);
        store_row_4c32(c, 1, ldc_floats, c1, beta_re, beta_im);
        store_row_4c32(c, 2, ldc_floats, c2, beta_re, beta_im);
        store_row_4c32(c, 3, ldc_floats, c3, beta_re, beta_im);
    } else {
        store_strided_row_4c32(c, c_rs, c_cs, 0, c0, beta_re, beta_im);
        store_strided_row_4c32(c, c_rs, c_cs, 1, c1, beta_re, beta_im);
        store_strided_row_4c32(c, c_rs, c_cs, 2, c2, beta_re, beta_im);
        store_strided_row_4c32(c, c_rs, c_cs, 3, c3, beta_re, beta_im);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn scale_alpha_c32(c: __m256, alpha_re: f32, alpha_im: f32) -> __m256 {
    if alpha_re == 1.0 && alpha_im == 0.0 {
        return c;
    }
    let c_swap = _mm256_permute_ps::<0b10110001>(c);
    let alpha_re_v = _mm256_set1_ps(alpha_re);
    let alpha_im_v = _mm256_set1_ps(alpha_im);
    let term_re = _mm256_fmadd_ps(alpha_re_v, c, _mm256_setzero_ps());
    let term_im = _mm256_fmadd_ps(alpha_im_v, c_swap, _mm256_setzero_ps());
    _mm256_addsub_ps(term_re, term_im)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_row_4c32(
    c: *mut f32,
    row: usize,
    ldc_floats: usize,
    new_val: __m256,
    beta_re: f32,
    beta_im: f32,
) {
    let p = c.add(row * ldc_floats);
    if beta_re == 0.0 && beta_im == 0.0 {
        _mm256_storeu_ps(p, new_val);
        return;
    }
    let cur = _mm256_loadu_ps(p);
    if beta_re == 1.0 && beta_im == 0.0 {
        _mm256_storeu_ps(p, _mm256_fmadd_ps(_mm256_set1_ps(1.0), cur, new_val));
        return;
    }
    // beta * cur + new_val (complex)
    let cur_swap = _mm256_permute_ps::<0b10110001>(cur);
    let beta_re_v = _mm256_set1_ps(beta_re);
    let beta_im_v = _mm256_set1_ps(beta_im);
    let term_re = _mm256_fmadd_ps(beta_re_v, cur, _mm256_setzero_ps());
    let term_im = _mm256_fmadd_ps(beta_im_v, cur_swap, _mm256_setzero_ps());
    let scaled_cur = _mm256_addsub_ps(term_re, term_im);
    let result = _mm256_fmadd_ps(_mm256_set1_ps(1.0), scaled_cur, new_val);
    _mm256_storeu_ps(p, result);
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_strided_row_4c32(
    c: *mut f32,
    c_rs: isize, // complex
    c_cs: isize, // complex
    row: usize,
    new_val: __m256,
    beta_re: f32,
    beta_im: f32,
) {
    let mut buf = [0.0_f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), new_val);
    let row_off_floats = (row as isize) * c_rs * 2;
    for j in 0..NR_C32 {
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

/// Edge-case CGEMM kernel for tail tiles where m < MR_C32 or n < NR_C32.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_c32(
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
    debug_assert!(m <= MR_C32);
    debug_assert!(n <= NR_C32);
    let mut tile_re = [[0.0_f32; NR_C32]; MR_C32];
    let mut tile_im = [[0.0_f32; NR_C32]; MR_C32];
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
        a = a.add(MR_C32 * 2);
        b = b.add(NR_C32 * 2);
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
        let mut packed = vec![0.0_f32; MR_C32 * 2 * kc];
        for i in 0..MR_C32 {
            for k in 0..kc {
                packed[k * MR_C32 * 2 + i * 2] = a[i * kc * 2 + k * 2];
                packed[k * MR_C32 * 2 + i * 2 + 1] = a[i * kc * 2 + k * 2 + 1];
            }
        }
        packed
    }

    fn pack_b(b: &[f32], kc: usize) -> Vec<f32> {
        let mut packed = vec![0.0_f32; kc * NR_C32 * 2];
        for k in 0..kc {
            for j in 0..NR_C32 {
                packed[k * NR_C32 * 2 + j * 2] = b[k * NR_C32 * 2 + j * 2];
                packed[k * NR_C32 * 2 + j * 2 + 1] = b[k * NR_C32 * 2 + j * 2 + 1];
            }
        }
        packed
    }

    #[test]
    fn kernel_4x4_c32_matches_naive() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 16;
        let a: Vec<f32> = (0..MR_C32 * kc * 2)
            .map(|i| (i as f32) * 0.013 - 0.7)
            .collect();
        let b: Vec<f32> = (0..kc * NR_C32 * 2)
            .map(|i| (i as f32) * 0.011 + 0.3)
            .collect();
        let mut c_ours = vec![0.0_f32; MR_C32 * NR_C32 * 2];
        let mut c_ref = vec![0.0_f32; MR_C32 * NR_C32 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x4_c32(
                kc,
                1.0,
                0.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_C32 as isize,
                1,
            );
        }
        naive_cgemm(MR_C32, NR_C32, kc, &a, &b, &mut c_ref);

        for i in 0..MR_C32 * NR_C32 * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-4,
                "mismatch at idx {i}: ours={} ref={} diff={}",
                c_ours[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn kernel_4x4_c32_alpha_complex() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 8;
        let a: Vec<f32> = (0..MR_C32 * kc * 2).map(|i| (i as f32) * 0.02).collect();
        let b: Vec<f32> = (0..kc * NR_C32 * 2).map(|i| (i as f32) * 0.03).collect();
        let mut c_ours = vec![0.0_f32; MR_C32 * NR_C32 * 2];
        let mut c_ref = vec![0.0_f32; MR_C32 * NR_C32 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x4_c32(
                kc,
                0.5,
                0.7,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_C32 as isize,
                1,
            );
        }
        naive_cgemm(MR_C32, NR_C32, kc, &a, &b, &mut c_ref);
        for i in 0..MR_C32 * NR_C32 {
            let cr = c_ref[i * 2];
            let ci = c_ref[i * 2 + 1];
            c_ref[i * 2] = 0.5 * cr - 0.7 * ci;
            c_ref[i * 2 + 1] = 0.5 * ci + 0.7 * cr;
        }

        for i in 0..MR_C32 * NR_C32 * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-3,
                "mismatch at idx {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_edge_c32_handles_partial() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 6;
        let m = 3;
        let n = 2;
        let a: Vec<f32> = (0..MR_C32 * kc * 2).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..kc * NR_C32 * 2).map(|i| (i as f32) * 0.07).collect();
        let mut c_ours = vec![0.0_f32; m * n * 2];
        let mut c_ref = vec![0.0_f32; m * n * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_edge_c32(
                m,
                n,
                kc,
                1.0,
                0.0,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        for i in 0..m {
            for j in 0..n {
                let mut ar_acc = 0.0_f32;
                let mut ai_acc = 0.0_f32;
                for p in 0..kc {
                    let ar = a[i * kc * 2 + p * 2];
                    let ai = a[i * kc * 2 + p * 2 + 1];
                    let br = b[p * NR_C32 * 2 + j * 2];
                    let bi = b[p * NR_C32 * 2 + j * 2 + 1];
                    ar_acc += ar * br - ai * bi;
                    ai_acc += ar * bi + ai * br;
                }
                c_ref[i * n * 2 + j * 2] = ar_acc;
                c_ref[i * n * 2 + j * 2 + 1] = ai_acc;
            }
        }
        for i in 0..m * n * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-4,
                "edge mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }
}
