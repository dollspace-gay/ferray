// ferrum-ufunc: Scalar fallback kernels
//
// These are used when SIMD is disabled or when the array is non-contiguous.

/// Apply a unary function elementwise over slices.
#[inline]
pub fn unary_map<T: Copy>(input: &[T], output: &mut [T], f: impl Fn(T) -> T) {
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = f(i);
    }
}

/// Apply a binary function elementwise over slices.
#[inline]
pub fn binary_map<T: Copy>(a: &[T], b: &[T], output: &mut [T], f: impl Fn(T, T) -> T) {
    for ((o, &ai), &bi) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = f(ai, bi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_unary() {
        let input = [1.0f64, 4.0, 9.0];
        let mut output = [0.0f64; 3];
        unary_map(&input, &mut output, f64::sqrt);
        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn scalar_binary() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        let mut out = [0.0f64; 3];
        binary_map(&a, &b, &mut out, |x, y| x + y);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }
}
