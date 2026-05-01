//! Integer SIMD kernels for i32 elementwise arithmetic (#384).
//!
//! Following the same shape as the f32/f64 binary kernels: a tight
//! scalar loop with linear memory access pattern that LLVM's
//! auto-vectorizer turns into target-appropriate SIMD instructions
//! (paddd/psubd/pmulld on x86 SSE2+, vpaddd on AVX2, vector add on
//! NEON). Pulp's `Simd` trait does not expose cross-platform integer
//! arithmetic (it only has per-backend integer methods), so we stay
//! at the auto-vectorizer for portable integer SIMD coverage —
//! matching numpy's loops_arithmetic.dispatch.c.src approach where
//! the SIMD codegen is compiler-driven rather than intrinsics.
//!
//! Wrapping semantics match Rust's signed-integer overflow convention
//! and numpy's two's-complement wraparound.

/// Elementwise i32 add (wrapping on overflow).
#[inline]
pub fn add_i32(a: &[i32], b: &[i32], output: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai.wrapping_add(bi);
    }
}

/// Elementwise i32 subtract (wrapping on overflow).
#[inline]
pub fn sub_i32(a: &[i32], b: &[i32], output: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai.wrapping_sub(bi);
    }
}

/// Elementwise i32 multiply (wrapping on overflow).
#[inline]
pub fn mul_i32(a: &[i32], b: &[i32], output: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai.wrapping_mul(bi);
    }
}

/// Elementwise i32 bitwise AND.
#[inline]
pub fn bitand_i32(a: &[i32], b: &[i32], output: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai & bi;
    }
}

/// Elementwise i32 bitwise OR.
#[inline]
pub fn bitor_i32(a: &[i32], b: &[i32], output: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai | bi;
    }
}

/// Elementwise i32 bitwise XOR.
#[inline]
pub fn bitxor_i32(a: &[i32], b: &[i32], output: &mut [i32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai ^ bi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_i32_works() {
        let a: Vec<i32> = (0..32).collect();
        let b: Vec<i32> = (100..132).collect();
        let mut out = vec![0i32; 32];
        add_i32(&a, &b, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], a[i] + b[i]);
        }
    }

    #[test]
    fn add_i32_handles_partial_tail() {
        let a: Vec<i32> = (1..=9).collect();
        let b: Vec<i32> = (10..=18).collect();
        let mut out = vec![0i32; 9];
        add_i32(&a, &b, &mut out);
        assert_eq!(out, vec![11, 13, 15, 17, 19, 21, 23, 25, 27]);
    }

    #[test]
    fn sub_i32_works() {
        let a: Vec<i32> = (10..18).collect();
        let b: Vec<i32> = (0..8).collect();
        let mut out = vec![0i32; 8];
        sub_i32(&a, &b, &mut out);
        assert_eq!(out, vec![10; 8]);
    }

    #[test]
    fn mul_i32_works() {
        let a: Vec<i32> = (1..=8).collect();
        let b: Vec<i32> = vec![2; 8];
        let mut out = vec![0i32; 8];
        mul_i32(&a, &b, &mut out);
        assert_eq!(out, vec![2, 4, 6, 8, 10, 12, 14, 16]);
    }

    #[test]
    fn add_i32_wraps_on_overflow() {
        let a = vec![i32::MAX; 4];
        let b = vec![1i32; 4];
        let mut out = vec![0i32; 4];
        add_i32(&a, &b, &mut out);
        assert_eq!(out, vec![i32::MIN; 4]);
    }

    #[test]
    fn bitand_bitor_bitxor_i32() {
        let a = vec![0b1100i32, 0b1010, 0b1111];
        let b = vec![0b1010i32, 0b0101, 0b0000];
        let mut and_out = vec![0i32; 3];
        let mut or_out = vec![0i32; 3];
        let mut xor_out = vec![0i32; 3];
        bitand_i32(&a, &b, &mut and_out);
        bitor_i32(&a, &b, &mut or_out);
        bitxor_i32(&a, &b, &mut xor_out);
        assert_eq!(and_out, vec![0b1000, 0b0000, 0b0000]);
        assert_eq!(or_out, vec![0b1110, 0b1111, 0b1111]);
        assert_eq!(xor_out, vec![0b0110, 0b1111, 0b1111]);
    }
}
