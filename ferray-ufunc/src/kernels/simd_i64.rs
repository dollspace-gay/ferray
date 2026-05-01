//! Integer SIMD kernels for i64 elementwise arithmetic (#384).
//!
//! Same shape as `simd_i32`: tight scalar loops with linear access
//! pattern that LLVM auto-vectorizes to target-appropriate i64 SIMD
//! (paddq/psubq on SSE2+, vpaddq on AVX2, vector add on NEON).
//! Pulp's cross-platform Simd trait does not expose generic integer
//! arithmetic, so we stay at the auto-vectorizer for portable
//! integer SIMD coverage.

#[inline]
pub fn add_i64(a: &[i64], b: &[i64], output: &mut [i64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai.wrapping_add(bi);
    }
}

#[inline]
pub fn sub_i64(a: &[i64], b: &[i64], output: &mut [i64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai.wrapping_sub(bi);
    }
}

#[inline]
pub fn mul_i64(a: &[i64], b: &[i64], output: &mut [i64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai.wrapping_mul(bi);
    }
}

#[inline]
pub fn bitand_i64(a: &[i64], b: &[i64], output: &mut [i64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai & bi;
    }
}

#[inline]
pub fn bitor_i64(a: &[i64], b: &[i64], output: &mut [i64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = ai | bi;
    }
}

#[inline]
pub fn bitxor_i64(a: &[i64], b: &[i64], output: &mut [i64]) {
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
    fn add_i64_works() {
        let a: Vec<i64> = (0..16).collect();
        let b: Vec<i64> = (100..116).collect();
        let mut out = vec![0i64; 16];
        add_i64(&a, &b, &mut out);
        for i in 0..16 {
            assert_eq!(out[i], a[i] + b[i]);
        }
    }

    #[test]
    fn add_i64_handles_partial_tail() {
        let a: Vec<i64> = (1..=5).collect();
        let b: Vec<i64> = (10..=14).collect();
        let mut out = vec![0i64; 5];
        add_i64(&a, &b, &mut out);
        assert_eq!(out, vec![11, 13, 15, 17, 19]);
    }

    #[test]
    fn sub_i64_works() {
        let a = vec![100i64, 200, 300, 400];
        let b = vec![1i64, 2, 3, 4];
        let mut out = vec![0i64; 4];
        sub_i64(&a, &b, &mut out);
        assert_eq!(out, vec![99, 198, 297, 396]);
    }

    #[test]
    fn mul_i64_works() {
        let a = vec![1i64, 2, 3, 4];
        let b = vec![10i64; 4];
        let mut out = vec![0i64; 4];
        mul_i64(&a, &b, &mut out);
        assert_eq!(out, vec![10, 20, 30, 40]);
    }

    #[test]
    fn add_i64_wraps_on_overflow() {
        let a = vec![i64::MAX; 4];
        let b = vec![1i64; 4];
        let mut out = vec![0i64; 4];
        add_i64(&a, &b, &mut out);
        assert_eq!(out, vec![i64::MIN; 4]);
    }

    #[test]
    fn bitand_bitor_bitxor_i64() {
        let a = vec![0b1100i64, 0b1010, 0b1111];
        let b = vec![0b1010i64, 0b0101, 0b0000];
        let mut and_out = vec![0i64; 3];
        let mut or_out = vec![0i64; 3];
        let mut xor_out = vec![0i64; 3];
        bitand_i64(&a, &b, &mut and_out);
        bitor_i64(&a, &b, &mut or_out);
        bitxor_i64(&a, &b, &mut xor_out);
        assert_eq!(and_out, vec![0b1000, 0b0000, 0b0000]);
        assert_eq!(or_out, vec![0b1110, 0b1111, 0b1111]);
        assert_eq!(xor_out, vec![0b0110, 0b1111, 0b1111]);
    }
}
