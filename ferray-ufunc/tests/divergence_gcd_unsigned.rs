//! Divergence regression: `gcd_int` / `lcm_int` must serve UNSIGNED integer
//! dtypes and be MIN-safe (no panic on `i*::MIN`), matching numpy.
//!
//! numpy registers `gcd`/`lcm` for ALL integer dtypes incl. unsigned
//! (`numpy/_core/code_generators/generate_umath.py:1156` gcd, `:1163` lcm —
//! `TD(ints)` covers uint8/16/32/64) and returns the NON-NEGATIVE gcd. ferray's
//! kernels were `num_traits::Signed`-bounded (used only for `.abs()`); the
//! `GcdAbs` abstraction (`arithmetic.rs`) replaces that bound so the same
//! Euclidean kernel serves unsigned widths and wraps `i*::MIN` instead of
//! panicking. Expected values are from live numpy 2.4.x (R-CHAR-3).

#[cfg(test)]
mod tests {
    use ferray_core::Array;
    use ferray_core::dimension::Ix1;
    use ferray_ufunc::{gcd_int, lcm_int};

    #[test]
    fn gcd_unsigned_u8() {
        // np.gcd(np.uint8(12), np.uint8(8)) == 4; np.gcd(np.uint8(8),uint8(6))==2.
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![12, 8]).unwrap();
        let b = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![8, 6]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4u8, 2]);
    }

    #[test]
    fn gcd_unsigned_u32() {
        // np.gcd(np.uint32(36), np.uint32(24)) == 12; gcd(0,7)==7.
        let a = Array::<u32, Ix1>::from_vec(Ix1::new([2]), vec![36, 0]).unwrap();
        let b = Array::<u32, Ix1>::from_vec(Ix1::new([2]), vec![24, 7]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[12u32, 7]);
    }

    #[test]
    fn lcm_unsigned_u8() {
        // np.lcm(np.uint8(4), np.uint8(6)) == 12; lcm(0,5)==0.
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![4, 0]).unwrap();
        let b = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![6, 5]).unwrap();
        let r = lcm_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[12u8, 0]);
    }

    #[test]
    fn lcm_unsigned_u32() {
        // np.lcm(np.uint32(6), np.uint32(8)) == 24.
        let a = Array::<u32, Ix1>::from_vec(Ix1::new([1]), vec![6]).unwrap();
        let b = Array::<u32, Ix1>::from_vec(Ix1::new([1]), vec![8]).unwrap();
        let r = lcm_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[24u32]);
    }

    #[test]
    fn gcd_signed_negative_regression() {
        // np.gcd(np.int32(-12), np.int32(8)) == 4: NON-NEGATIVE gcd regardless
        // of operand signs.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![-12, 15]).unwrap();
        let b = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![8, -25]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4, 5]);
    }

    #[test]
    fn lcm_signed_negative_regression() {
        // np.lcm(np.int32(-4), np.int32(6)) == 12 (non-negative).
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([1]), vec![-4]).unwrap();
        let b = Array::<i32, Ix1>::from_vec(Ix1::new([1]), vec![6]).unwrap();
        let r = lcm_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[12]);
    }

    #[test]
    fn gcd_i8_min_no_panic() {
        // `GcdAbs` uses `wrapping_abs`, so `gcd_int(i8::MIN, 0)` returns the
        // WRAPPED abs (`i8::MIN`) WITHOUT panicking — matching numpy, whose
        // integer gcd loop wraps two's-complement `|i8::MIN|`:
        //   np.gcd(np.int8(-128), np.int8(0)) == -128 (live numpy 2.4.x).
        // The old `num_traits::Signed::abs()` would panic here in debug builds.
        let a = Array::<i8, Ix1>::from_vec(Ix1::new([1]), vec![i8::MIN]).unwrap();
        let b = Array::<i8, Ix1>::from_vec(Ix1::new([1]), vec![0]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[i8::MIN]);
    }
}
