// ferray-linalg::gemm — hand-tuned Rust DGEMM micro-kernels.
//
// Why this exists: faer (and its underlying `private-gemm-x86`) gives
// portable Rust GEMM but lags hand-tuned BLAS by 4-9x on AVX2-only CPUs at
// N in [128, 512] (#653, #654). This module ports the OpenBLAS-style
// register-blocked algorithm to stable Rust intrinsics, targeting the
// Haswell+ family (AVX2 + FMA, 256-bit YMM registers).
//
// Currently shipped:
//   - DGEMM (f64) on AVX2+FMA, 4x8 register tile (translated from
//     OpenBLAS dgemm_kernel_4x8_haswell.S)
//   - SGEMM (f32) on AVX2+FMA, 4x16 register tile (parallel to the
//     DGEMM design; OpenBLAS's reference SGEMM uses a different
//     interleaved layout that doesn't map cleanly to row-major output)
//   - AVX-512F DGEMM (f64) on opt-in `avx512` feature, 4x16 register
//     tile (translated from dgemm_kernel_16x2_skylakex.c). Off by
//     default because AVX-512 intrinsics in core::arch::x86_64
//     stabilised in Rust 1.89.
//
// Roadmap (in priority order):
//   - AVX-512 SGEMM (Skylake-X / Sapphire Rapids / Zen4)
//   - Zen3/4 retune (different L1/L2 sizes)

// GEMM functions have many parameters by mathematical necessity (m, n, k,
// alpha, A, lda, B, ldb, beta, C, ldc) — the BLAS contract.
#![allow(clippy::too_many_arguments)]
// When the `openblas` feature is on, all gemm_* entries dispatch to the
// system OpenBLAS instead of the hand-tuned kernels. The kernels and
// their packing/driver helpers are still compiled (so the AVX-512 and
// AVX-VNNI feature combinations stay healthy under CI), but unused.
#![cfg_attr(feature = "openblas", allow(dead_code))]
// Each gemm_* entry has a `#[cfg(feature = "openblas")] { ... return ...; }`
// dispatch block. The explicit return is needed to short-circuit the
// `#[cfg(not(feature = "openblas"))]` fallback when the feature is on;
// clippy can't see across cfg branches and flags it as needless.
#![cfg_attr(feature = "openblas", allow(clippy::needless_return))]

#[cfg(test)]
mod integer_tests;

mod kernel_avx2;
mod kernel_avx2_complex_f32;
mod kernel_avx2_complex_f64;
mod kernel_avx2_f32;
mod kernel_avx2_i16;
mod kernel_avx2_i8;
mod kernel_avx2_i8s;
#[cfg(feature = "avx512")]
mod kernel_avx512;
#[cfg(feature = "avx512")]
mod kernel_avx512_complex_f32;
#[cfg(feature = "avx512")]
mod kernel_avx512_complex_f64;
#[cfg(feature = "avx512")]
mod kernel_avx512_f32;
#[cfg(feature = "avx512")]
mod kernel_avx512_i16;
#[cfg(feature = "avx512")]
mod kernel_avx512_vnni_i8;
#[cfg(feature = "avxvnni")]
mod kernel_avxvnni_i8;
#[cfg(feature = "openblas")]
mod openblas_backend;
mod outer;
mod pack;

/// Returns true iff the running CPU supports our hand-tuned kernel.
#[inline]
pub fn cpu_supports_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns true iff the running CPU supports AVX-512F (and the
/// `avx512` feature is compiled in). When this returns true, the
/// AVX-512 4x16 DGEMM kernel is preferred over the AVX2 4x8 path.
#[inline]
pub fn cpu_supports_avx512f() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        false
    }
}

/// Returns true iff the running CPU supports AVX-512 VNNI (and the
/// `avx512` feature is compiled in). When this returns true, the
/// `gemm_i8` dispatch uses the zmm-wide single-instruction `vpdpbusd`
/// path. Available on Cascade Lake +, Ice Lake server +, Sapphire
/// Rapids +, Zen4 +.
#[inline]
pub fn cpu_supports_avx512_vnni() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        false
    }
}

/// Returns true iff the running CPU supports AVX-512BW (byte/word) and
/// the `avx512` feature is compiled in. When this returns true,
/// `gemm_i16` dispatches to the zmm-wide 4×16 kernel that uses
/// `vpmaddwd` on 32 i16 inputs per instruction (vs 16 for AVX2).
/// Available on every Skylake-X-class AVX-512 CPU (the F-only
/// "Knights Mill" parts excluded).
#[inline]
pub fn cpu_supports_avx512_bw() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        false
    }
}

/// Compute C += alpha * A @ B (row-major, leading dim = k for A, n for B/C),
/// using the hand-tuned AVX2 kernel. Caller must verify CPU support via
/// `cpu_supports_avx2_fma()`. Returns `false` and does nothing if the
/// platform doesn't support AVX2+FMA at runtime.
///
/// `alpha = 1.0`, `beta = 0.0` is the common case for plain `C = A @ B`.
#[inline]
pub fn gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) -> bool {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

    // OpenBLAS backend takes priority over hand-tuned kernels when its
    // feature is enabled. The host's BLAS does its own CPU dispatch.
    #[cfg(feature = "openblas")]
    {
        // SAFETY: caller upholds the buffer-length invariants.
        unsafe {
            openblas_backend::gemm_f64_openblas(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
            );
        }
        return true;
    }

    #[cfg(not(feature = "openblas"))]
    {
        if !cpu_supports_avx2_fma() {
            return false;
        }

        // Prefer the AVX-512 path when both feature `avx512` is on and the
        // CPU advertises AVX-512F. The kernel processes 8 doubles per zmm
        // (vs 4 per ymm for AVX2) so peak FLOPs/cycle doubles.
        #[cfg(feature = "avx512")]
        if cpu_supports_avx512f() {
            // SAFETY: AVX-512F detected at runtime.
            unsafe {
                outer::gemm_avx512_f64(
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    k,
                    b.as_ptr(),
                    n,
                    beta,
                    c.as_mut_ptr(),
                    n,
                );
            }
            return true;
        }

        // SAFETY: cpu_supports_avx2_fma() returned true, so AVX2+FMA are present.
        // Slice lengths are checked by debug_assert above; in release we trust
        // the caller (this path is internal to ferray-linalg).
        unsafe {
            outer::gemm_avx2_f64(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                beta,
                c.as_mut_ptr(),
                n,
            );
        }
        true
    }
}

/// i16 quantized GEMM: C = A @ B + (optional accumulator) for row-major
/// `i16 × i16 → i32` matrices. Higher-precision quantization compared
/// to `gemm_i8` (16-bit operands instead of 8-bit) at half the
/// throughput. Uses AVX2 `vpmaddwd` on the 4×8 register tile by
/// default. When the `avx512` feature is enabled and the host supports
/// AVX-512BW, dispatches to the 4×16 zmm tile (32 i16 per VPMADDWD,
/// double the throughput).
///
/// Returns `false` if AVX2 is not available.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k` i16s
/// - `b` points to at least `k * n` i16s
/// - `c` points to at least `m * n` i32s, writable
/// - The buffers don't overlap
#[inline]
pub unsafe fn gemm_i16(
    m: usize,
    n: usize,
    k: usize,
    a: *const i16,
    b: *const i16,
    c: *mut i32,
    accumulate: bool,
) -> bool {
    if !cpu_supports_avx2_fma() {
        return false;
    }
    // Prefer AVX-512BW i16 path when feature is on AND host advertises it.
    #[cfg(feature = "avx512")]
    if cpu_supports_avx512_bw() {
        // SAFETY: AVX-512F + AVX-512BW detected at runtime.
        unsafe {
            outer::gemm_avx512_i16(m, n, k, a, k, b, n, c, n, accumulate);
        }
        return true;
    }
    // SAFETY: cpu check passed.
    unsafe {
        outer::gemm_avx2_i16(m, n, k, a, k, b, n, c, n, accumulate);
    }
    true
}

/// i8 (signed-signed) GEMM: C = A @ B + (optional accumulator) for
/// row-major `i8 × i8 → i32` matrices. Used for symmetric quantization
/// schemes where both operands are signed (vs gemm_i8's u8 × i8).
///
/// Implemented by a dedicated AVX2 4×8 / KB=2 kernel that does
/// `vpmovsxbw` on B inline (16 i8 → 16 i16 in one ymm) and packs A's
/// 2-byte K-pair into a broadcastable i32 with manual sign-extension,
/// then `vpmaddwd` reduces 2 K-pairs into the i32 accumulator. This
/// avoids the O(m·k + k·n) widening pre-pass that the previous wrapper
/// implementation performed and halves the input bandwidth.
///
/// Returns `false` if AVX2 is not available.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k` i8s
/// - `b` points to at least `k * n` i8s
/// - `c` points to at least `m * n` i32s, writable
/// - The buffers don't overlap
pub unsafe fn gemm_i8_signed(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    b: *const i8,
    c: *mut i32,
    accumulate: bool,
) -> bool {
    if !cpu_supports_avx2_fma() {
        return false;
    }
    // SAFETY: cpu check passed; caller upholds the buffer-length contract.
    unsafe {
        outer::gemm_avx2_i8s(m, n, k, a, k, b, n, c, n, accumulate);
    }
    true
}

/// Quantized matmul with zero-points and scales — implements the full
/// ONNX `QLinearMatMul` / oneDNN `s8s8s32` quantized GEMM formula:
///
/// ```text
/// c_quant[i, j] = scale_a * scale_b * sum_k((a[i, k] - a_zp) * (b[k, j] - b_zp))
/// ```
///
/// Expanded form (mathematically equivalent, computationally cheaper):
///
/// ```text
/// raw[i, j] = sum_k(a*b)
///           - a_zp * sum_k(b[k, j])
///           - b_zp * sum_k(a[i, k])
///           + k * a_zp * b_zp
///
/// c[i, j] = scale_a * scale_b * raw[i, j]   (output is f32)
/// ```
///
/// The dominant `sum_k(a*b)` term goes through `gemm_i8` (the AVX-VNNI/
/// AVX-512 fast paths kick in when available). The two row/column sum
/// terms are O(m·k + k·n) — small relative to the O(m·n·k) matmul.
///
/// `accumulate=false`: c is overwritten. `accumulate=true`: f32 results
/// are added to existing c.
///
/// Returns `false` if AVX2 is not available.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k` u8s
/// - `b` points to at least `k * n` i8s
/// - `c` points to at least `m * n` f32s, writable
/// - The buffers don't overlap
pub unsafe fn quantized_matmul(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    a_zp: u8,
    scale_a: f32,
    b: *const i8,
    b_zp: i8,
    scale_b: f32,
    c: *mut f32,
    accumulate: bool,
) -> bool {
    if !cpu_supports_avx2_fma() {
        return false;
    }
    if m == 0 || n == 0 {
        return true;
    }
    // 1. raw_ab[i, j] = sum_k(a[i,k] * b[k,j]) via gemm_i8.
    let mut raw: Vec<i32> = vec![0; m * n];
    let ok = unsafe { gemm_i8(m, n, k, a, b, raw.as_mut_ptr(), false) };
    if !ok {
        return false;
    }
    if k == 0 {
        // Just write zeros (or leave c alone if accumulate).
        if !accumulate {
            unsafe {
                for i in 0..m * n {
                    *c.add(i) = 0.0;
                }
            }
        }
        return true;
    }

    // 2. Row sums of A: row_sum_a[i] = sum_k(a[i, k]) (as i32).
    let mut row_sum_a: Vec<i32> = vec![0; m];
    unsafe {
        for i in 0..m {
            let mut s = 0_i32;
            let row = a.add(i * k);
            for p in 0..k {
                s = s.wrapping_add(*row.add(p) as i32);
            }
            row_sum_a[i] = s;
        }
    }

    // 3. Col sums of B: col_sum_b[j] = sum_k(b[k, j]) (as i32).
    let mut col_sum_b: Vec<i32> = vec![0; n];
    unsafe {
        for p in 0..k {
            let row = b.add(p * n);
            for j in 0..n {
                col_sum_b[j] = col_sum_b[j].wrapping_add(*row.add(j) as i32);
            }
        }
    }

    // 4. Compose: out[i, j] = scale_a * scale_b * (raw - a_zp*col_sum_b
    //                            - b_zp*row_sum_a + k*a_zp*b_zp).
    let azp = a_zp as i32;
    let bzp = b_zp as i32;
    let k_az_bz = (k as i32).wrapping_mul(azp).wrapping_mul(bzp);
    let scale = scale_a * scale_b;
    unsafe {
        for i in 0..m {
            let rsa = row_sum_a[i];
            for j in 0..n {
                let csb = col_sum_b[j];
                let raw_ij = raw[i * n + j];
                let corrected = raw_ij
                    .wrapping_sub(azp.wrapping_mul(csb))
                    .wrapping_sub(bzp.wrapping_mul(rsa))
                    .wrapping_add(k_az_bz);
                let out = scale * (corrected as f32);
                let p = c.add(i * n + j);
                if accumulate {
                    *p += out;
                } else {
                    *p = out;
                }
            }
        }
    }
    true
}

/// i8 quantized GEMM: C = A @ B + (optional accumulator) for row-major
/// `u8 × i8 → i32` matrices. This is the canonical activations × weights
/// → accumulator pattern used by ONNX QLinearMatMul, PyTorch quantized
/// ops, and TFLite int8 inference.
///
/// `a`: row-major u8 (m × k); `b`: row-major i8 (k × n); `c`: row-major
/// i32 (m × n). When `accumulate` is true, products are added to the
/// existing C; otherwise C is overwritten.
///
/// On hosts that support AVX-VNNI (Alder Lake +, Tiger Lake +, Sapphire
/// Rapids +, Zen4 +) the kernel uses single-instruction `vpdpbusd` for
/// 4-way (u8, i8) → i32 dot-product accumulate. Otherwise it falls back
/// to the portable AVX2 `vpmaddubsw + vpmaddwd` chain.
///
/// Returns `false` if AVX2 is not available.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k` u8s
/// - `b` points to at least `k * n` i8s
/// - `c` points to at least `m * n` i32s, writable
/// - The buffers don't overlap
#[inline]
pub unsafe fn gemm_i8(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    b: *const i8,
    c: *mut i32,
    accumulate: bool,
) -> bool {
    if !cpu_supports_avx2_fma() {
        return false;
    }
    // Prefer AVX-512 VNNI when the `avx512` feature is on AND the host
    // CPU advertises both AVX-512F and AVX-512 VNNI. The kernel uses
    // zmm-wide vpdpbusd: 64 (u8, i8) products per FMA vs 32 for AVX-VNNI.
    #[cfg(feature = "avx512")]
    if cpu_supports_avx512_vnni() {
        // SAFETY: AVX-512F + AVX-512 VNNI detected at runtime.
        unsafe {
            outer::gemm_avx512_vnni_i8(m, n, k, a, k, b, n, c, n, accumulate);
        }
        return true;
    }
    // SAFETY: cpu check passed; caller upholds the buffer-length invariants.
    unsafe {
        outer::gemm_avx2_i8(
            m, n, k, a, k, // lda = k for tightly packed row-major
            b, n, // ldb = n
            c, n, // ldc = n
            accumulate,
        );
    }
    true
}

/// CGEMM (Complex<f32>) entry — computes C = alpha * A @ B + beta * C
/// for row-major Complex<f32> matrices using the hand-tuned AVX2 4x4
/// CGEMM kernel. `a`, `b`, `c` are passed as `*const f32` / `*mut f32`
/// where each Complex<f32> element is two contiguous f32s `[re, im]`.
/// `lda`, `ldb`, `ldc` are row strides **in complex elements**.
///
/// Returns `false` if AVX2+FMA is not available at runtime.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k * 2` f32s
/// - `b` points to at least `k * n * 2` f32s
/// - `c` points to at least `m * n * 2` f32s, writable
/// - The buffers don't overlap.
#[inline]
pub unsafe fn gemm_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    b: *const f32,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
) -> bool {
    #[cfg(feature = "openblas")]
    {
        unsafe {
            openblas_backend::gemm_c32_openblas(
                m, n, k, alpha_re, alpha_im, a, b, beta_re, beta_im, c,
            );
        }
        return true;
    }
    #[cfg(not(feature = "openblas"))]
    {
        if !cpu_supports_avx2_fma() {
            return false;
        }
        // Prefer AVX-512 CGEMM when feature `avx512` is on AND CPU has avx512f.
        // Each FMA processes 16 f32 = 8 complex (2x AVX2 lanes), doubling peak.
        #[cfg(feature = "avx512")]
        if cpu_supports_avx512f() {
            unsafe {
                outer::gemm_avx512_c32(
                    m, n, k, alpha_re, alpha_im, a, k, b, n, beta_re, beta_im, c, n,
                );
            }
            return true;
        }
        // SAFETY: cpu check passed; caller upholds buffer-length invariants.
        unsafe {
            outer::gemm_avx2_c32(
                m, n, k, alpha_re, alpha_im, a, k, b, n, beta_re, beta_im, c, n,
            );
        }
        true
    }
}

/// ZGEMM (Complex<f64>) entry — computes C = alpha * A @ B + beta * C
/// for row-major Complex<f64> matrices using the hand-tuned AVX2 4x2
/// ZGEMM kernel. `a`, `b`, `c` are passed as `*const f64` / `*mut f64`
/// where each Complex<f64> element is two contiguous f64s `[re, im]`.
/// `lda`, `ldb`, `ldc` are row strides **in complex elements**.
/// `alpha = (1, 0)`, `beta = (0, 0)` is the common case for plain
/// `C = A @ B`.
///
/// Returns `false` and does nothing if AVX2+FMA is not available at
/// runtime.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k * 2` f64s
/// - `b` points to at least `k * n * 2` f64s
/// - `c` points to at least `m * n * 2` f64s, writable
/// - The buffers don't overlap.
#[inline]
pub unsafe fn gemm_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    b: *const f64,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
) -> bool {
    #[cfg(feature = "openblas")]
    {
        unsafe {
            openblas_backend::gemm_c64_openblas(
                m, n, k, alpha_re, alpha_im, a, b, beta_re, beta_im, c,
            );
        }
        return true;
    }
    #[cfg(not(feature = "openblas"))]
    {
        if !cpu_supports_avx2_fma() {
            return false;
        }
        // Prefer AVX-512 ZGEMM when feature `avx512` is on AND CPU has avx512f.
        // Each FMA processes 8 doubles = 4 complex (2x AVX2 lanes).
        #[cfg(feature = "avx512")]
        if cpu_supports_avx512f() {
            unsafe {
                outer::gemm_avx512_c64(
                    m, n, k, alpha_re, alpha_im, a, k, b, n, beta_re, beta_im, c, n,
                );
            }
            return true;
        }
        // SAFETY: cpu check passed; caller upholds the buffer-length invariants.
        unsafe {
            outer::gemm_avx2_c64(
                m, n, k, alpha_re, alpha_im, a, k, // lda = k for tightly packed row-major
                b, n, // ldb = n
                beta_re, beta_im, c, n, // ldc = n
            );
        }
        true
    }
}

/// Mixed-precision GEMM: row-major `bf16 × bf16 → f32`. C is accumulated
/// in f32 for accuracy (bf16 has ~3 decimal digits of mantissa precision,
/// not enough for direct accumulation in any non-trivial K).
///
/// Implementation: pre-widens both inputs from bf16 → f32 via the
/// AVX2-accelerated `HalfFloatSliceExt::convert_to_f32_slice` path
/// (uses F16C-equivalent SIMD conversion when available), then dispatches
/// to the hand-tuned SGEMM kernel. The widen overhead is O(m·k + k·n);
/// the matmul is O(m·n·k) — negligible amortized cost at moderate K.
///
/// `accumulate=false`: C is overwritten. `accumulate=true`: results are
/// added to existing C.
///
/// Returns `false` if AVX2+FMA is not available.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k` bf16s
/// - `b` points to at least `k * n` bf16s
/// - `c` points to at least `m * n` f32s, writable
/// - The buffers don't overlap
#[cfg(feature = "bf16")]
pub unsafe fn gemm_bf16_f32(
    m: usize,
    n: usize,
    k: usize,
    a: *const half::bf16,
    b: *const half::bf16,
    c: *mut f32,
    accumulate: bool,
) -> bool {
    use half::slice::HalfFloatSliceExt;
    if !cpu_supports_avx2_fma() {
        return false;
    }
    if m == 0 || n == 0 {
        return true;
    }
    // Widen bf16 → f32 in fresh buffers via the SIMD-accelerated half
    // crate path. SAFETY: caller's buffer-length contract.
    let a_slice = unsafe { core::slice::from_raw_parts(a, m * k) };
    let b_slice = unsafe { core::slice::from_raw_parts(b, k * n) };
    let mut a_f32 = vec![0.0_f32; m * k];
    let mut b_f32 = vec![0.0_f32; k * n];
    a_slice.convert_to_f32_slice(&mut a_f32);
    b_slice.convert_to_f32_slice(&mut b_f32);

    let c_slice = unsafe { core::slice::from_raw_parts_mut(c, m * n) };
    let beta = if accumulate { 1.0 } else { 0.0 };
    gemm_f32(m, n, k, 1.0, &a_f32, &b_f32, beta, c_slice)
}

/// Mixed-precision GEMM: row-major `f16 × f16 → f32` (IEEE 754 binary16).
/// C is accumulated in f32 for accuracy (f16 has ~10 mantissa bits, not
/// enough for direct accumulation past trivial K).
///
/// Implementation: pre-widens both inputs from f16 → f32 via the
/// hardware-accelerated `HalfFloatSliceExt::convert_to_f32_slice` path
/// (uses F16C `vcvtph2ps` when available), then dispatches to the
/// hand-tuned SGEMM kernel.
///
/// `accumulate=false`: C is overwritten. `accumulate=true`: results are
/// added to existing C.
///
/// Returns `false` if AVX2+FMA is not available.
///
/// # Safety
///
/// Caller must ensure:
/// - `a` points to at least `m * k` f16s
/// - `b` points to at least `k * n` f16s
/// - `c` points to at least `m * n` f32s, writable
/// - The buffers don't overlap
#[cfg(feature = "f16")]
pub unsafe fn gemm_f16_f32(
    m: usize,
    n: usize,
    k: usize,
    a: *const half::f16,
    b: *const half::f16,
    c: *mut f32,
    accumulate: bool,
) -> bool {
    use half::slice::HalfFloatSliceExt;
    if !cpu_supports_avx2_fma() {
        return false;
    }
    if m == 0 || n == 0 {
        return true;
    }
    let a_slice = unsafe { core::slice::from_raw_parts(a, m * k) };
    let b_slice = unsafe { core::slice::from_raw_parts(b, k * n) };
    let mut a_f32 = vec![0.0_f32; m * k];
    let mut b_f32 = vec![0.0_f32; k * n];
    a_slice.convert_to_f32_slice(&mut a_f32);
    b_slice.convert_to_f32_slice(&mut b_f32);

    let c_slice = unsafe { core::slice::from_raw_parts_mut(c, m * n) };
    let beta = if accumulate { 1.0 } else { 0.0 };
    gemm_f32(m, n, k, 1.0, &a_f32, &b_f32, beta, c_slice)
}

/// f32 sibling of [`gemm_f64`]. Computes C += alpha * A @ B for row-major
/// f32 matrices using the hand-tuned AVX2 4x16 SGEMM kernel.
///
/// `alpha = 1.0`, `beta = 0.0` is the common case for plain `C = A @ B`.
#[inline]
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> bool {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

    #[cfg(feature = "openblas")]
    {
        // SAFETY: caller upholds the buffer-length invariants.
        unsafe {
            openblas_backend::gemm_f32_openblas(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
            );
        }
        return true;
    }

    #[cfg(not(feature = "openblas"))]
    {
        if !cpu_supports_avx2_fma() {
            return false;
        }

        // Prefer AVX-512 SGEMM when feature `avx512` is on AND the CPU
        // advertises AVX-512F. The kernel processes 16 floats per zmm vs
        // 8 per ymm for AVX2, doubling peak FLOPs/cycle.
        #[cfg(feature = "avx512")]
        if cpu_supports_avx512f() {
            // SAFETY: AVX-512F detected at runtime.
            unsafe {
                outer::gemm_avx512_f32(
                    m,
                    n,
                    k,
                    alpha,
                    a.as_ptr(),
                    k,
                    b.as_ptr(),
                    n,
                    beta,
                    c.as_mut_ptr(),
                    n,
                );
            }
            return true;
        }

        // SAFETY: cpu_supports_avx2_fma() returned true, so AVX2+FMA are present.
        unsafe {
            outer::gemm_avx2_f32(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                beta,
                c.as_mut_ptr(),
                n,
            );
        }
        true
    }
}
