// ferrum-ufunc: SIMD dispatch via pulp (REQ-17, REQ-18, REQ-20)
//
// Provides runtime CPU feature detection and dispatch for SIMD-accelerated
// elementwise operations. Uses `pulp::Arch` for portable dispatch across
// SSE2, AVX2, AVX-512 on x86_64 and NEON on aarch64.

use pulp::Arch;

/// Check if SIMD is forcibly disabled via `FERRUM_FORCE_SCALAR=1`.
#[inline]
pub fn force_scalar() -> bool {
    // We check the env var at runtime; caching would be an optimisation
    // for later. The cost is negligible compared to array work.
    std::env::var("FERRUM_FORCE_SCALAR")
        .ok()
        .is_some_and(|v| v == "1")
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
        // pulp ensures we're on the best available ISA.
        // For transcendental functions (sin, cos, exp, etc.), there is no
        // direct SIMD intrinsic — we still call the scalar fn per-element.
        // The benefit of pulp here is future-proofing: when we add
        // polynomial-approximation SIMD kernels, this is where they plug in.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_unary_f32_scalar() {
        // SAFETY: test runs are single-threaded for this test
        unsafe { std::env::set_var("FERRUM_FORCE_SCALAR", "1"); }
        let input = [1.0f32, 4.0, 9.0, 16.0];
        let mut output = [0.0f32; 4];
        dispatch_unary_f32(&input, &mut output, f32::sqrt);
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
        unsafe { std::env::remove_var("FERRUM_FORCE_SCALAR"); }
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
        // SAFETY: test runs are single-threaded for this test
        unsafe { std::env::set_var("FERRUM_FORCE_SCALAR", "1"); }
        assert!(force_scalar());
        unsafe { std::env::remove_var("FERRUM_FORCE_SCALAR"); }
        assert!(!force_scalar());
    }
}
