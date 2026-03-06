// ferrum-ufunc: Convolution
//
// convolve with modes: Full, Same, Valid

use ferrum_core::dimension::Ix1;
use ferrum_core::dtype::Element;
use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::Array;

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolveMode {
    /// Full convolution output (length = N + M - 1).
    Full,
    /// Output has length max(N, M).
    Same,
    /// Output only where signals fully overlap (length = max(N, M) - min(N, M) + 1).
    Valid,
}

/// Discrete, linear convolution of two 1-D arrays.
///
/// Computes `convolve(a, v, mode)` following NumPy semantics.
pub fn convolve<T>(
    a: &Array<T, Ix1>,
    v: &Array<T, Ix1>,
    mode: ConvolveMode,
) -> FerrumResult<Array<T, Ix1>>
where
    T: Element + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
{
    let a_data: Vec<T> = a.iter().copied().collect();
    let v_data: Vec<T> = v.iter().copied().collect();
    let n = a_data.len();
    let m = v_data.len();

    if n == 0 || m == 0 {
        return Err(FerrumError::invalid_value(
            "convolve: input arrays must be non-empty",
        ));
    }

    // Full convolution
    let full_len = n + m - 1;
    let mut full = vec![<T as Element>::zero(); full_len];

    for i in 0..n {
        for j in 0..m {
            full[i + j] = full[i + j] + a_data[i] * v_data[j];
        }
    }

    match mode {
        ConvolveMode::Full => Array::from_vec(Ix1::new([full_len]), full),
        ConvolveMode::Same => {
            let out_len = n.max(m);
            let start = (full_len - out_len) / 2;
            let result = full[start..start + out_len].to_vec();
            Array::from_vec(Ix1::new([out_len]), result)
        }
        ConvolveMode::Valid => {
            let out_len = if n >= m { n - m + 1 } else { m - n + 1 };
            let start = m.min(n) - 1;
            let result = full[start..start + out_len].to_vec();
            Array::from_vec(Ix1::new([out_len]), result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_convolve_full() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let v = arr1(vec![0.0, 1.0, 0.5]);
        let r = convolve(&a, &v, ConvolveMode::Full).unwrap();
        let s = r.as_slice().unwrap();
        // [0*1, 1*1+0*2, 0.5*1+1*2+0*3, 0.5*2+1*3, 0.5*3]
        // = [0, 1, 2.5, 4, 1.5]
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.0).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
        assert!((s[2] - 2.5).abs() < 1e-12);
        assert!((s[3] - 4.0).abs() < 1e-12);
        assert!((s[4] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_same() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let v = arr1(vec![0.0, 1.0, 0.5]);
        let r = convolve(&a, &v, ConvolveMode::Same).unwrap();
        assert_eq!(r.size(), 3);
        let s = r.as_slice().unwrap();
        // Full = [0, 1, 2.5, 4, 1.5], same takes middle 3 = [1, 2.5, 4]
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.5).abs() < 1e-12);
        assert!((s[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_valid() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let v = arr1(vec![0.0, 1.0, 0.5]);
        let r = convolve(&a, &v, ConvolveMode::Valid).unwrap();
        assert_eq!(r.size(), 1);
        let s = r.as_slice().unwrap();
        assert!((s[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_simple() {
        let a = arr1(vec![1.0, 1.0, 1.0]);
        let v = arr1(vec![1.0, 1.0, 1.0]);
        let r = convolve(&a, &v, ConvolveMode::Full).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.0).abs() < 1e-12);
        assert!((s[2] - 3.0).abs() < 1e-12);
        assert!((s[3] - 2.0).abs() < 1e-12);
        assert!((s[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_i32() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let v = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![1, 1]).unwrap();
        let r = convolve(&a, &v, ConvolveMode::Full).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1, 3, 5, 3]);
    }
}
