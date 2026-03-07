// ferray-ufunc: SIMD dispatch via pulp (REQ-17, REQ-18, REQ-20)
//
// Provides runtime CPU feature detection and dispatch for SIMD-accelerated
// elementwise operations. Uses `pulp::Arch` for portable dispatch across
// SSE2, AVX2, AVX-512 on x86_64 and NEON on aarch64.

use pulp::Arch;

/// Check if SIMD is forcibly disabled via `FERRAY_FORCE_SCALAR=1`.
///
/// The env var is read once and cached for the lifetime of the process.
#[inline]
pub fn force_scalar() -> bool {
    use std::sync::LazyLock;
    static CACHED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("FERRAY_FORCE_SCALAR")
            .ok()
            .is_some_and(|v| v == "1")
    });
    *CACHED
}

/// Apply a unary SIMD kernel over contiguous `f32` slices, falling back to
/// scalar when SIMD is disabled.
#[inline]
pub fn dispatch_unary_f32(input: &[f32], output: &mut [f32], scalar_fn: fn(f32) -> f32) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = scalar_fn(i);
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(UnaryF32Op {
            input,
            output,
            scalar_fn,
        });
    }
}

/// Apply a unary SIMD kernel over contiguous `f64` slices.
#[inline]
pub fn dispatch_unary_f64(input: &[f64], output: &mut [f64], scalar_fn: fn(f64) -> f64) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = scalar_fn(i);
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(UnaryF64Op {
            input,
            output,
            scalar_fn,
        });
    }
}

/// Apply a binary SIMD kernel over contiguous `f32` slices.
#[inline]
pub fn dispatch_binary_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    scalar_fn: fn(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    if force_scalar() {
        for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = scalar_fn(ai, bi);
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(BinaryF32Op {
            a,
            b,
            output,
            scalar_fn,
        });
    }
}

/// Apply a binary SIMD kernel over contiguous `f64` slices.
#[inline]
pub fn dispatch_binary_f64(
    a: &[f64],
    b: &[f64],
    output: &mut [f64],
    scalar_fn: fn(f64, f64) -> f64,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    if force_scalar() {
        for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = scalar_fn(ai, bi);
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(BinaryF64Op {
            a,
            b,
            output,
            scalar_fn,
        });
    }
}

/// Apply a unary operation on `f16` slices via f32 promotion.
///
/// Each input `f16` is promoted to `f32`, the scalar function is applied,
/// and the result is converted back to `f16`.
#[cfg(feature = "f16")]
#[inline]
pub fn dispatch_unary_f16(
    input: &[half::f16],
    output: &mut [half::f16],
    scalar_fn: fn(f32) -> f32,
) {
    debug_assert_eq!(input.len(), output.len());
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = half::f16::from_f32(scalar_fn(i.to_f32()));
    }
}

/// Apply a binary operation on `f16` slices via f32 promotion.
///
/// Each pair of input `f16` values is promoted to `f32`, the scalar function
/// is applied, and the result is converted back to `f16`.
#[cfg(feature = "f16")]
#[inline]
pub fn dispatch_binary_f16(
    a: &[half::f16],
    b: &[half::f16],
    output: &mut [half::f16],
    scalar_fn: fn(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = half::f16::from_f32(scalar_fn(ai.to_f32(), bi.to_f32()));
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated operations that use actual hardware SIMD intrinsics
// ---------------------------------------------------------------------------

/// SIMD sqrt for f64 slices using hardware `vsqrtpd` / equivalent.
#[inline]
pub fn simd_sqrt_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = i.sqrt();
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(SqrtF64Op { input, output });
    }
}

/// SIMD sqrt for f32 slices using hardware `vsqrtps` / equivalent.
#[inline]
pub fn simd_sqrt_f32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = i.sqrt();
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(SqrtF32Op { input, output });
    }
}

/// SIMD abs for f64 slices.
#[inline]
pub fn simd_abs_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = i.abs();
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(AbsF64Op { input, output });
    }
}

/// SIMD neg for f64 slices.
#[inline]
pub fn simd_neg_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = -i;
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(NegF64Op { input, output });
    }
}

/// SIMD square (x*x) for f64 slices.
#[inline]
pub fn simd_square_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = i * i;
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(SquareF64Op { input, output });
    }
}

/// SIMD reciprocal (1/x) for f64 slices.
#[inline]
pub fn simd_reciprocal_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    if force_scalar() {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = 1.0 / i;
        }
    } else {
        let arch = Arch::new();
        arch.dispatch(ReciprocalF64Op { input, output });
    }
}

// ---------------------------------------------------------------------------
// WithSimd implementations for pulp dispatch
// ---------------------------------------------------------------------------

struct UnaryF32Op<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    scalar_fn: fn(f32) -> f32,
}

impl pulp::WithSimd for UnaryF32Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> Self::Output {
        let f = self.scalar_fn;
        for (o, &i) in self.output.iter_mut().zip(self.input.iter()) {
            *o = f(i);
        }
    }
}

struct UnaryF64Op<'a> {
    input: &'a [f64],
    output: &'a mut [f64],
    scalar_fn: fn(f64) -> f64,
}

impl pulp::WithSimd for UnaryF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> Self::Output {
        let f = self.scalar_fn;
        for (o, &i) in self.output.iter_mut().zip(self.input.iter()) {
            *o = f(i);
        }
    }
}

struct BinaryF32Op<'a> {
    a: &'a [f32],
    b: &'a [f32],
    output: &'a mut [f32],
    scalar_fn: fn(f32, f32) -> f32,
}

impl pulp::WithSimd for BinaryF32Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> Self::Output {
        let f = self.scalar_fn;
        for ((o, &ai), &bi) in self.output.iter_mut().zip(self.a.iter()).zip(self.b.iter()) {
            *o = f(ai, bi);
        }
    }
}

struct BinaryF64Op<'a> {
    a: &'a [f64],
    b: &'a [f64],
    output: &'a mut [f64],
    scalar_fn: fn(f64, f64) -> f64,
}

impl pulp::WithSimd for BinaryF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> Self::Output {
        let f = self.scalar_fn;
        for ((o, &ai), &bi) in self.output.iter_mut().zip(self.a.iter()).zip(self.b.iter()) {
            *o = f(ai, bi);
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD intrinsic implementations (actual hardware SIMD, not scalar fallback)
// ---------------------------------------------------------------------------

struct SqrtF64Op<'a> {
    input: &'a [f64],
    output: &'a mut [f64],
}

impl pulp::WithSimd for SqrtF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let n = self.input.len();
        let lane_count = size_of::<S::f64s>() / size_of::<f64>();
        let stride = lane_count * 4;
        let unrolled_end = n - (n % stride);
        let simd_end = n - (n % lane_count);

        // 4-wide unroll to hide sqrt's ~12-cycle latency
        let mut i = 0;
        while i < unrolled_end {
            let v0 = simd.partial_load_f64s(&self.input[i..i + lane_count]);
            let v1 = simd.partial_load_f64s(&self.input[i + lane_count..i + lane_count * 2]);
            let v2 = simd.partial_load_f64s(&self.input[i + lane_count * 2..i + lane_count * 3]);
            let v3 = simd.partial_load_f64s(&self.input[i + lane_count * 3..i + stride]);
            let r0 = simd.sqrt_f64s(v0);
            let r1 = simd.sqrt_f64s(v1);
            let r2 = simd.sqrt_f64s(v2);
            let r3 = simd.sqrt_f64s(v3);
            simd.partial_store_f64s(&mut self.output[i..i + lane_count], r0);
            simd.partial_store_f64s(&mut self.output[i + lane_count..i + lane_count * 2], r1);
            simd.partial_store_f64s(&mut self.output[i + lane_count * 2..i + lane_count * 3], r2);
            simd.partial_store_f64s(&mut self.output[i + lane_count * 3..i + stride], r3);
            i += stride;
        }
        while i < simd_end {
            let v = simd.partial_load_f64s(&self.input[i..i + lane_count]);
            let r = simd.sqrt_f64s(v);
            simd.partial_store_f64s(&mut self.output[i..i + lane_count], r);
            i += lane_count;
        }
        for j in simd_end..n {
            self.output[j] = self.input[j].sqrt();
        }
    }
}

struct SqrtF32Op<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl pulp::WithSimd for SqrtF32Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let n = self.input.len();
        let lane_count = size_of::<S::f32s>() / size_of::<f32>();
        let simd_end = n - (n % lane_count);

        for i in (0..simd_end).step_by(lane_count) {
            let v = simd.partial_load_f32s(&self.input[i..i + lane_count]);
            let r = simd.sqrt_f32s(v);
            simd.partial_store_f32s(&mut self.output[i..i + lane_count], r);
        }
        for i in simd_end..n {
            self.output[i] = self.input[i].sqrt();
        }
    }
}

struct AbsF64Op<'a> {
    input: &'a [f64],
    output: &'a mut [f64],
}

impl pulp::WithSimd for AbsF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let n = self.input.len();
        let lane_count = size_of::<S::f64s>() / size_of::<f64>();
        let simd_end = n - (n % lane_count);

        for i in (0..simd_end).step_by(lane_count) {
            let v = simd.partial_load_f64s(&self.input[i..i + lane_count]);
            let r = simd.abs_f64s(v);
            simd.partial_store_f64s(&mut self.output[i..i + lane_count], r);
        }
        for i in simd_end..n {
            self.output[i] = self.input[i].abs();
        }
    }
}

struct NegF64Op<'a> {
    input: &'a [f64],
    output: &'a mut [f64],
}

impl pulp::WithSimd for NegF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let n = self.input.len();
        let lane_count = size_of::<S::f64s>() / size_of::<f64>();
        let simd_end = n - (n % lane_count);

        for i in (0..simd_end).step_by(lane_count) {
            let v = simd.partial_load_f64s(&self.input[i..i + lane_count]);
            let r = simd.neg_f64s(v);
            simd.partial_store_f64s(&mut self.output[i..i + lane_count], r);
        }
        for i in simd_end..n {
            self.output[i] = -self.input[i];
        }
    }
}

struct SquareF64Op<'a> {
    input: &'a [f64],
    output: &'a mut [f64],
}

impl pulp::WithSimd for SquareF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let n = self.input.len();
        let lane_count = size_of::<S::f64s>() / size_of::<f64>();
        let simd_end = n - (n % lane_count);

        for i in (0..simd_end).step_by(lane_count) {
            let v = simd.partial_load_f64s(&self.input[i..i + lane_count]);
            let r = simd.mul_f64s(v, v);
            simd.partial_store_f64s(&mut self.output[i..i + lane_count], r);
        }
        for i in simd_end..n {
            self.output[i] = self.input[i] * self.input[i];
        }
    }
}

struct ReciprocalF64Op<'a> {
    input: &'a [f64],
    output: &'a mut [f64],
}

impl pulp::WithSimd for ReciprocalF64Op<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let n = self.input.len();
        let lane_count = size_of::<S::f64s>() / size_of::<f64>();
        let simd_end = n - (n % lane_count);
        let one = simd.splat_f64s(1.0);

        for i in (0..simd_end).step_by(lane_count) {
            let v = simd.partial_load_f64s(&self.input[i..i + lane_count]);
            let r = simd.div_f64s(one, v);
            simd.partial_store_f64s(&mut self.output[i..i + lane_count], r);
        }
        for i in simd_end..n {
            self.output[i] = 1.0 / self.input[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_unary_f32_works() {
        // Tests the dispatch path (SIMD or scalar depending on platform).
        // To test the forced-scalar path, run with FERRAY_FORCE_SCALAR=1.
        let input = [1.0f32, 4.0, 9.0, 16.0];
        let mut output = [0.0f32; 4];
        dispatch_unary_f32(&input, &mut output, f32::sqrt);
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dispatch_unary_f64_simd() {
        let input = [1.0f64, 4.0, 9.0, 16.0];
        let mut output = [0.0f64; 4];
        dispatch_unary_f64(&input, &mut output, f64::sqrt);
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dispatch_binary_f32_works() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        dispatch_binary_f32(&a, &b, &mut out, |x, y| x + y);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn dispatch_binary_f64_works() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        let mut out = [0.0f64; 3];
        dispatch_binary_f64(&a, &b, &mut out, |x, y| x * y);
        assert_eq!(out, [4.0, 10.0, 18.0]);
    }

    #[test]
    fn force_scalar_env() {
        // force_scalar() is cached via LazyLock for performance.
        // In normal test runs, FERRAY_FORCE_SCALAR is not set,
        // so force_scalar() returns false. We verify that here.
        // To test the FERRAY_FORCE_SCALAR=1 path, run tests with
        // the env var set: FERRAY_FORCE_SCALAR=1 cargo test
        assert!(!force_scalar());
    }

    #[cfg(feature = "f16")]
    #[test]
    fn dispatch_unary_f16_works() {
        let input = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(9.0),
            half::f16::from_f32(16.0),
        ];
        let mut output = [half::f16::ZERO; 4];
        super::dispatch_unary_f16(&input, &mut output, f32::sqrt);
        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (o, &e) in output.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 0.01);
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn dispatch_binary_f16_works() {
        let a = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ];
        let b = [
            half::f16::from_f32(4.0),
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
        ];
        let mut out = [half::f16::ZERO; 3];
        super::dispatch_binary_f16(&a, &b, &mut out, |x, y| x + y);
        let expected = [5.0f32, 7.0, 9.0];
        for (o, &e) in out.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 0.01);
        }
    }
}
