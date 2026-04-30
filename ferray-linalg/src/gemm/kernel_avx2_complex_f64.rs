// AVX2+FMA ZGEMM (Complex<f64>) micro-kernel — translated from OpenBLAS's
// `kernel/x86_64/zgemm_kernel_4x2_haswell.c` (BSD-3 licensed).
//
// Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
//                  = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
//
// We use the OpenBLAS "expanded accumulators" pattern (m=2 / m=1 / small N
// variant) which keeps the cross-product additions linear:
//
//   For each K iteration, A[i, k] = (a_re, a_im), B[k, 0..NR_C64] is a
//   row of NR_C64 complex numbers laid out [b0_re, b0_im, b1_re, b1_im, ...]
//   in memory. We compute:
//
//     ymm_b      = load(B)                         // [b0.re, b0.im, b1.re, b1.im]
//     ymm_b_swap = vpermilpd($5)(ymm_b)            // [b0.im, b0.re, b1.im, b1.re]
//     a_re_bcast = vbroadcastsd(A[i, k].re)
//     a_im_bcast = vbroadcastsd(A[i, k].im)
//     acc_re[i] += a_re_bcast * ymm_b              // partial sums of a_re * b lanes
//     acc_im[i] += a_im_bcast * ymm_b_swap         // partial sums of a_im * b_swap lanes
//
//   After the K loop, combine:
//     c[i] = vaddsubpd(acc_re[i], acc_im[i])
//   which yields, per even-indexed lane (real slot):
//       c[i].re_lane = acc_re[i].re_lane - acc_im[i].re_lane
//                    = sum(a_re * b.re) - sum(a_im * b.im)  ← correct real part
//   and per odd-indexed lane (imag slot):
//       c[i].im_lane = acc_re[i].im_lane + acc_im[i].im_lane
//                    = sum(a_re * b.im) + sum(a_im * b.re)  ← correct imag part
//
// This kernel handles the non-conjugated NN case (most common). Conjugate
// variants (NC/CN/CC) are a follow-up — they swap the FMA polarity (use
// vfnmadd / vaddsub permutations).
//
// Register usage (16 ymm available, we use 13):
//   ymm0       broadcast scratch (current A[i, k].re)
//   ymm1       broadcast scratch (current A[i, k].im)
//   ymm2       current B row (4 doubles = NR_C64 complex)
//   ymm3       B-swap (vpermilpd $5 of ymm2)
//   ymm4..ymm7   acc_re for rows 0..3 (each holds partial a_re*B sums for the row)
//   ymm8..ymm11  acc_im for rows 0..3 (each holds partial a_im*B_swap sums)
//
// Per K iteration: 1 ymm load (B row, 4 doubles = 2 complex), 1 vpermilpd
// for B_swap, 4 broadcasts of A (2 per row × 2 rows... wait 4 rows = 8
// broadcasts), 8 FMAs (4 acc_re + 4 acc_im).
//
// A complex multiply costs 4 real FMAs (vs 1 for real GEMM), so ZGEMM
// peak FLOPs/cycle is half DGEMM peak. With 8 FMAs / 2 ports = 4 cycles
// per K-iter and 8 FMAs * 4 lanes = 32 FLOPs of complex work
// (= 64 real FLOPs since each complex FMA is 4 real ops). 64/4 = 16 real
// FLOPs/cycle = AVX2-FMA peak. ✓

use core::arch::x86_64::{
    __m256d, _MM_HINT_T0, _mm_prefetch, _mm256_addsub_pd, _mm256_broadcast_sd, _mm256_fmadd_pd,
    _mm256_loadu_pd, _mm256_permute_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
};

/// Number of complex rows in the register tile.
pub const MR_C64: usize = 4;
/// Number of complex cols in the register tile (2 complex = 4 doubles = 1 ymm).
pub const NR_C64: usize = 2;

const A_PF_DOUBLES: usize = 512 / 8;
const B_PF_DOUBLES: usize = 160 / 8;

/// 4×2 ZGEMM micro-kernel. Treats `Complex<f64>` arrays as `f64` slices
/// of double the length: each complex element is 2 contiguous f64s
/// `[re, im]` (matches `num_complex::Complex<f64>` repr(C) layout).
///
/// `kc` is the depth in complex elements. `c_rs`/`c_cs` are row/column
/// strides of C **in complex elements** (each complex = 2 f64s in memory,
/// so we multiply by 2 internally to get f64 strides).
///
/// SAFETY: caller guarantees AVX2+FMA availability, `packed_a >=
/// kc * MR_C64 complex elements` (= 2x doubles), `packed_b >= kc *
/// NR_C64 complex elements`, and `c` addresses an (m_tile, n_tile)
/// sub-tile.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn kernel_4x2_c64(
    kc: usize,
    alpha_re: f64,
    alpha_im: f64,
    packed_a: *const f64, // points at start of complex strip; reads 2 doubles per complex
    packed_b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64, // points at C[row, col] as f64 (complex = 2 doubles)
    c_rs: isize, // row stride in COMPLEX elements
    c_cs: isize, // col stride in COMPLEX elements
) {
    // INIT — zero all 8 accumulators (4 acc_re + 4 acc_im).
    let mut acc_re_0 = _mm256_setzero_pd();
    let mut acc_re_1 = _mm256_setzero_pd();
    let mut acc_re_2 = _mm256_setzero_pd();
    let mut acc_re_3 = _mm256_setzero_pd();
    let mut acc_im_0 = _mm256_setzero_pd();
    let mut acc_im_1 = _mm256_setzero_pd();
    let mut acc_im_2 = _mm256_setzero_pd();
    let mut acc_im_3 = _mm256_setzero_pd();

    let mut a = packed_a;
    let mut b = packed_b;
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(a.add(A_PF_DOUBLES).cast());
        _mm_prefetch::<_MM_HINT_T0>(b.add(B_PF_DOUBLES).cast());

        // Load B[k, 0..2] = 2 complex = 4 doubles = [b0.re, b0.im, b1.re, b1.im].
        let b_v = _mm256_loadu_pd(b);
        // B_swap = swap each adjacent f64 pair: [b0.im, b0.re, b1.im, b1.re].
        let b_swap = _mm256_permute_pd::<0b0101>(b_v);

        // Row 0: A[0, k] = packed_a[0..2] = (a0_re, a0_im).
        let a0_re = _mm256_broadcast_sd(&*a);
        let a0_im = _mm256_broadcast_sd(&*a.add(1));
        acc_re_0 = _mm256_fmadd_pd(a0_re, b_v, acc_re_0);
        acc_im_0 = _mm256_fmadd_pd(a0_im, b_swap, acc_im_0);

        // Row 1: A[1, k] = packed_a[2..4].
        let a1_re = _mm256_broadcast_sd(&*a.add(2));
        let a1_im = _mm256_broadcast_sd(&*a.add(3));
        acc_re_1 = _mm256_fmadd_pd(a1_re, b_v, acc_re_1);
        acc_im_1 = _mm256_fmadd_pd(a1_im, b_swap, acc_im_1);

        // Row 2.
        let a2_re = _mm256_broadcast_sd(&*a.add(4));
        let a2_im = _mm256_broadcast_sd(&*a.add(5));
        acc_re_2 = _mm256_fmadd_pd(a2_re, b_v, acc_re_2);
        acc_im_2 = _mm256_fmadd_pd(a2_im, b_swap, acc_im_2);

        // Row 3.
        let a3_re = _mm256_broadcast_sd(&*a.add(6));
        let a3_im = _mm256_broadcast_sd(&*a.add(7));
        acc_re_3 = _mm256_fmadd_pd(a3_re, b_v, acc_re_3);
        acc_im_3 = _mm256_fmadd_pd(a3_im, b_swap, acc_im_3);

        a = a.add(MR_C64 * 2);
        b = b.add(NR_C64 * 2);
    }

    // Combine accumulators via vaddsubpd — yields complex products.
    let c0 = _mm256_addsub_pd(acc_re_0, acc_im_0);
    let c1 = _mm256_addsub_pd(acc_re_1, acc_im_1);
    let c2 = _mm256_addsub_pd(acc_re_2, acc_im_2);
    let c3 = _mm256_addsub_pd(acc_re_3, acc_im_3);

    // Apply alpha (complex scalar). result = alpha * c, where alpha = (alpha_re + i*alpha_im):
    //   result.re_lane = alpha_re * c.re - alpha_im * c.im
    //   result.im_lane = alpha_re * c.im + alpha_im * c.re
    // Same broadcast/swap/addsub pattern as the inner kernel.
    let c0 = scale_alpha(c0, alpha_re, alpha_im);
    let c1 = scale_alpha(c1, alpha_re, alpha_im);
    let c2 = scale_alpha(c2, alpha_re, alpha_im);
    let c3 = scale_alpha(c3, alpha_re, alpha_im);

    // Store. Row-major fast path: each ymm is one row's 2 complex
    // (4 contiguous doubles). With c_cs == 1 (in COMPLEX elements; i.e.
    // doubles are contiguous) we use a single _mm256_storeu_pd per row.
    if c_cs == 1 && c_rs > 0 {
        let ldc_doubles = (c_rs as usize) * 2;
        store_row_2c(c, 0, ldc_doubles, c0, beta_re, beta_im);
        store_row_2c(c, 1, ldc_doubles, c1, beta_re, beta_im);
        store_row_2c(c, 2, ldc_doubles, c2, beta_re, beta_im);
        store_row_2c(c, 3, ldc_doubles, c3, beta_re, beta_im);
    } else {
        store_strided_row_2c(c, c_rs, c_cs, 0, c0, beta_re, beta_im);
        store_strided_row_2c(c, c_rs, c_cs, 1, c1, beta_re, beta_im);
        store_strided_row_2c(c, c_rs, c_cs, 2, c2, beta_re, beta_im);
        store_strided_row_2c(c, c_rs, c_cs, 3, c3, beta_re, beta_im);
    }
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn scale_alpha(c: __m256d, alpha_re: f64, alpha_im: f64) -> __m256d {
    if alpha_re == 1.0 && alpha_im == 0.0 {
        return c;
    }
    // Same complex-multiply trick: c_new = alpha * c
    let c_swap = _mm256_permute_pd::<0b0101>(c);
    let alpha_re_v = _mm256_set1_pd(alpha_re);
    let alpha_im_v = _mm256_set1_pd(alpha_im);
    let term_re = _mm256_fmadd_pd(alpha_re_v, c, _mm256_setzero_pd());
    let term_im = _mm256_fmadd_pd(alpha_im_v, c_swap, _mm256_setzero_pd());
    _mm256_addsub_pd(term_re, term_im)
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_row_2c(
    c: *mut f64,
    row: usize,
    ldc_doubles: usize,
    new_val: __m256d,
    beta_re: f64,
    beta_im: f64,
) {
    let p = c.add(row * ldc_doubles);
    if beta_re == 0.0 && beta_im == 0.0 {
        _mm256_storeu_pd(p, new_val);
        return;
    }
    let cur = _mm256_loadu_pd(p);
    if beta_re == 1.0 && beta_im == 0.0 {
        // result = new_val + cur
        _mm256_storeu_pd(p, _mm256_fmadd_pd(_mm256_set1_pd(1.0), cur, new_val));
        return;
    }
    // General complex beta * cur + new_val.
    let cur_swap = _mm256_permute_pd::<0b0101>(cur);
    let beta_re_v = _mm256_set1_pd(beta_re);
    let beta_im_v = _mm256_set1_pd(beta_im);
    let term_re = _mm256_fmadd_pd(beta_re_v, cur, _mm256_setzero_pd());
    let term_im = _mm256_fmadd_pd(beta_im_v, cur_swap, _mm256_setzero_pd());
    let scaled_cur = _mm256_addsub_pd(term_re, term_im);
    let result = _mm256_fmadd_pd(_mm256_set1_pd(1.0), scaled_cur, new_val);
    _mm256_storeu_pd(p, result);
}

#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_strided_row_2c(
    c: *mut f64,
    c_rs: isize, // complex
    c_cs: isize, // complex
    row: usize,
    new_val: __m256d,
    beta_re: f64,
    beta_im: f64,
) {
    let mut buf = [0.0_f64; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), new_val);
    let row_off_doubles = (row as isize) * c_rs * 2;
    for j in 0..NR_C64 {
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

/// Edge-case ZGEMM kernel for tail tiles where m < MR_C64 or n < NR_C64.
/// Slow scalar complex path.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_edge_c64(
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
    debug_assert!(m <= MR_C64);
    debug_assert!(n <= NR_C64);
    let mut tile_re = [[0.0_f64; NR_C64]; MR_C64];
    let mut tile_im = [[0.0_f64; NR_C64]; MR_C64];
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
        a = a.add(MR_C64 * 2);
        b = b.add(NR_C64 * 2);
    }
    for i in 0..m {
        for j in 0..n {
            // alpha * (tile.re + i*tile.im)
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
        // a is row-major (m, k) of complex f64 (2 doubles per complex).
        // b is row-major (k, n).
        // c is row-major (m, n). c += a @ b.
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
        // a is row-major (MR_C64, kc) complex; pack as kc-major MR_C64-minor.
        // Output: packed[k * (MR_C64*2) + i*2 + 0/1] = A[i, k].re/im.
        let mut packed = vec![0.0_f64; MR_C64 * 2 * kc];
        for i in 0..MR_C64 {
            for k in 0..kc {
                packed[k * MR_C64 * 2 + i * 2] = a[i * kc * 2 + k * 2];
                packed[k * MR_C64 * 2 + i * 2 + 1] = a[i * kc * 2 + k * 2 + 1];
            }
        }
        packed
    }

    fn pack_b(b: &[f64], kc: usize) -> Vec<f64> {
        // b is row-major (kc, NR_C64) complex; already in NR_C64-strip format.
        let mut packed = vec![0.0_f64; kc * NR_C64 * 2];
        for k in 0..kc {
            for j in 0..NR_C64 {
                packed[k * NR_C64 * 2 + j * 2] = b[k * NR_C64 * 2 + j * 2];
                packed[k * NR_C64 * 2 + j * 2 + 1] = b[k * NR_C64 * 2 + j * 2 + 1];
            }
        }
        packed
    }

    #[test]
    fn kernel_4x2_c64_matches_naive() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 16;
        let a: Vec<f64> = (0..MR_C64 * kc * 2)
            .map(|i| (i as f64) * 0.013 - 0.7)
            .collect();
        let b: Vec<f64> = (0..kc * NR_C64 * 2)
            .map(|i| (i as f64) * 0.011 + 0.3)
            .collect();
        let mut c_ours = vec![0.0_f64; MR_C64 * NR_C64 * 2];
        let mut c_ref = vec![0.0_f64; MR_C64 * NR_C64 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_4x2_c64(
                kc,
                1.0,
                0.0, // alpha = 1
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0, // beta = 0
                c_ours.as_mut_ptr(),
                NR_C64 as isize, // row stride = NR_C64 complex
                1,
            );
        }
        naive_zgemm(MR_C64, NR_C64, kc, &a, &b, &mut c_ref);

        for i in 0..MR_C64 * NR_C64 * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-9,
                "mismatch at idx {i}: ours={} ref={} diff={}",
                c_ours[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn kernel_4x2_c64_alpha_complex() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 8;
        let a: Vec<f64> = (0..MR_C64 * kc * 2).map(|i| (i as f64) * 0.02).collect();
        let b: Vec<f64> = (0..kc * NR_C64 * 2).map(|i| (i as f64) * 0.03).collect();
        let mut c_ours = vec![0.0_f64; MR_C64 * NR_C64 * 2];
        let mut c_ref = vec![0.0_f64; MR_C64 * NR_C64 * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        // alpha = 0.5 + 0.7i
        unsafe {
            kernel_4x2_c64(
                kc,
                0.5,
                0.7,
                pa.as_ptr(),
                pb.as_ptr(),
                0.0,
                0.0,
                c_ours.as_mut_ptr(),
                NR_C64 as isize,
                1,
            );
        }
        naive_zgemm(MR_C64, NR_C64, kc, &a, &b, &mut c_ref);
        // Apply alpha post-hoc to ref
        for i in 0..MR_C64 * NR_C64 {
            let cr = c_ref[i * 2];
            let ci = c_ref[i * 2 + 1];
            c_ref[i * 2] = 0.5 * cr - 0.7 * ci;
            c_ref[i * 2 + 1] = 0.5 * ci + 0.7 * cr;
        }

        for i in 0..MR_C64 * NR_C64 * 2 {
            let diff = (c_ours[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-9,
                "mismatch at idx {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn kernel_edge_c64_handles_partial() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let kc = 6;
        let m = 3;
        let n = 1;
        let a: Vec<f64> = (0..MR_C64 * kc * 2).map(|i| (i as f64) * 0.05).collect();
        let b: Vec<f64> = (0..kc * NR_C64 * 2).map(|i| (i as f64) * 0.07).collect();
        let mut c_ours = vec![0.0_f64; m * n * 2];
        let mut c_ref = vec![0.0_f64; m * n * 2];

        let pa = pack_a(&a, kc);
        let pb = pack_b(&b, kc);

        unsafe {
            kernel_edge_c64(
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
        // Naive ref using only first m rows of A (kc deep) × first n cols of B.
        for i in 0..m {
            for j in 0..n {
                let mut ar_acc = 0.0;
                let mut ai_acc = 0.0;
                for p in 0..kc {
                    let ar = a[i * kc * 2 + p * 2];
                    let ai = a[i * kc * 2 + p * 2 + 1];
                    let br = b[p * NR_C64 * 2 + j * 2];
                    let bi = b[p * NR_C64 * 2 + j * 2 + 1];
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
                diff < 1e-9,
                "edge mismatch at {i}: ours={} ref={}",
                c_ours[i],
                c_ref[i]
            );
        }
    }
}
