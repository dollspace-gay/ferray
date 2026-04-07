// Property-based tests for ferray-ufunc
//
// Tests mathematical invariants of universal functions using proptest.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

use ferray_ufunc::{
    absolute, add, arcsin, cos, divide, exp, log, multiply, negative, power, sign, sin, sqrt,
    square, subtract,
};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn scalar_arr(v: f64) -> Array<f64, Ix1> {
    arr1(vec![v])
}

fn const_arr(val: f64, n: usize) -> Array<f64, Ix1> {
    arr1(vec![val; n])
}

fn approx_eq_arr(a: &Array<f64, Ix1>, b: &Array<f64, Ix1>, tol: f64) -> bool {
    if a.size() != b.size() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. Inverse pair: arcsin(sin(x)) ~= x for x in [-pi/2, pi/2]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_arcsin_sin_inverse(x in -1.0f64..1.0) {
        let input = scalar_arr(x);
        let s = sin(&input).unwrap();
        let back = arcsin(&s).unwrap();
        let diff = (back.iter().next().unwrap() - x).abs();
        prop_assert!(diff < 1e-10, "arcsin(sin({})) = {}, diff = {}", x, back.iter().next().unwrap(), diff);
    }

    // -----------------------------------------------------------------------
    // 2. Inverse pair: exp(log(x)) ~= x for x > 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_exp_log_inverse(x in 0.01f64..1000.0) {
        let input = scalar_arr(x);
        let l = log(&input).unwrap();
        let back = exp(&l).unwrap();
        let result = *back.iter().next().unwrap();
        let diff = (result - x).abs() / x.max(1.0);
        prop_assert!(diff < 1e-10, "exp(log({})) = {}, rel_diff = {}", x, result, diff);
    }

    // -----------------------------------------------------------------------
    // 3. Inverse pair: sqrt(x)^2 ~= x for x >= 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sqrt_square_inverse(x in 0.0f64..1000.0) {
        let input = scalar_arr(x);
        let s = sqrt(&input).unwrap();
        let back = square(&s).unwrap();
        let result = *back.iter().next().unwrap();
        let diff = (result - x).abs();
        prop_assert!(diff < 1e-8, "sqrt({})^2 = {}, diff = {}", x, result, diff);
    }

    // -----------------------------------------------------------------------
    // 4. Identity element: add(a, 0) == a
    // -----------------------------------------------------------------------
    #[test]
    fn prop_add_identity(data in proptest::collection::vec(-100.0f64..100.0, 1..=20)) {
        let a = arr1(data.clone());
        let z = const_arr(0.0, data.len());
        let result = add(&a, &z).unwrap();
        prop_assert!(approx_eq_arr(&result, &a, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 5. Identity element: multiply(a, 1) == a
    // -----------------------------------------------------------------------
    #[test]
    fn prop_multiply_identity(data in proptest::collection::vec(-100.0f64..100.0, 1..=20)) {
        let a = arr1(data.clone());
        let one = const_arr(1.0, data.len());
        let result = multiply(&a, &one).unwrap();
        prop_assert!(approx_eq_arr(&result, &a, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 6. Identity element: power(a, 1) == a
    // -----------------------------------------------------------------------
    #[test]
    fn prop_power_identity(x in 0.1f64..100.0) {
        let a = scalar_arr(x);
        let one = scalar_arr(1.0);
        let result = power(&a, &one).unwrap();
        let r = *result.iter().next().unwrap();
        prop_assert!((r - x).abs() < 1e-10, "power({}, 1) = {}", x, r);
    }

    // -----------------------------------------------------------------------
    // 7. Commutativity: add(a, b) == add(b, a)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_add_commutative(
        a_data in proptest::collection::vec(-100.0f64..100.0, 10),
        b_data in proptest::collection::vec(-100.0f64..100.0, 10),
    ) {
        let a = arr1(a_data);
        let b = arr1(b_data);
        let ab = add(&a, &b).unwrap();
        let ba = add(&b, &a).unwrap();
        prop_assert!(approx_eq_arr(&ab, &ba, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 8. Commutativity: multiply(a, b) == multiply(b, a)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_multiply_commutative(
        a_data in proptest::collection::vec(-100.0f64..100.0, 10),
        b_data in proptest::collection::vec(-100.0f64..100.0, 10),
    ) {
        let a = arr1(a_data);
        let b = arr1(b_data);
        let ab = multiply(&a, &b).unwrap();
        let ba = multiply(&b, &a).unwrap();
        prop_assert!(approx_eq_arr(&ab, &ba, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 9. Associativity: add(add(a, b), c) ~= add(a, add(b, c))
    // -----------------------------------------------------------------------
    #[test]
    fn prop_add_associative(
        a_data in proptest::collection::vec(-50.0f64..50.0, 10),
        b_data in proptest::collection::vec(-50.0f64..50.0, 10),
        c_data in proptest::collection::vec(-50.0f64..50.0, 10),
    ) {
        let a = arr1(a_data);
        let b = arr1(b_data);
        let c = arr1(c_data);
        let ab = add(&a, &b).unwrap();
        let ab_c = add(&ab, &c).unwrap();
        let bc = add(&b, &c).unwrap();
        let a_bc = add(&a, &bc).unwrap();
        prop_assert!(approx_eq_arr(&ab_c, &a_bc, 1e-10));
    }

    // -----------------------------------------------------------------------
    // 10. Sign properties: absolute(x) >= 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_absolute_nonnegative(data in proptest::collection::vec(-100.0f64..100.0, 1..=20)) {
        let a = arr1(data);
        let abs_a = absolute(&a).unwrap();
        for &v in abs_a.iter() {
            prop_assert!(v >= 0.0, "absolute value {} should be >= 0", v);
        }
    }

    // -----------------------------------------------------------------------
    // 11. Sign properties: absolute(-x) == absolute(x)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_absolute_symmetric(data in proptest::collection::vec(-100.0f64..100.0, 1..=20)) {
        let a = arr1(data);
        let neg_a = negative(&a).unwrap();
        let abs_a = absolute(&a).unwrap();
        let abs_neg_a = absolute(&neg_a).unwrap();
        prop_assert!(approx_eq_arr(&abs_a, &abs_neg_a, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 12. Range properties: sin(x) in [-1, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sin_range(x in -1000.0f64..1000.0) {
        let input = scalar_arr(x);
        let s = sin(&input).unwrap();
        let val = *s.iter().next().unwrap();
        prop_assert!((-1.0..=1.0).contains(&val), "sin({}) = {} out of [-1, 1]", x, val);
    }

    // -----------------------------------------------------------------------
    // 13. Range properties: exp(x) > 0 for finite x
    // -----------------------------------------------------------------------
    #[test]
    fn prop_exp_positive(x in -500.0f64..500.0) {
        let input = scalar_arr(x);
        let e = exp(&input).unwrap();
        let val = *e.iter().next().unwrap();
        // exp can underflow to 0 for very negative x; >= covers both cases.
        prop_assert!(val >= 0.0, "exp({}) = {} should be >= 0", x, val);
    }

    // -----------------------------------------------------------------------
    // 14. Monotonicity: for sorted input, exp output is sorted
    // -----------------------------------------------------------------------
    #[test]
    fn prop_exp_monotone(mut data in proptest::collection::vec(-10.0f64..10.0, 2..=20)) {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let a = arr1(data);
        let e = exp(&a).unwrap();
        let vals: Vec<f64> = e.iter().copied().collect();
        for i in 1..vals.len() {
            prop_assert!(vals[i] >= vals[i - 1], "exp not monotone: {} < {}", vals[i], vals[i - 1]);
        }
    }

    // -----------------------------------------------------------------------
    // 15. Monotonicity: for sorted positive input, log output is sorted
    // -----------------------------------------------------------------------
    #[test]
    fn prop_log_monotone(mut data in proptest::collection::vec(0.01f64..100.0, 2..=20)) {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let a = arr1(data);
        let l = log(&a).unwrap();
        let vals: Vec<f64> = l.iter().copied().collect();
        for i in 1..vals.len() {
            prop_assert!(vals[i] >= vals[i - 1], "log not monotone: {} < {}", vals[i], vals[i - 1]);
        }
    }

    // -----------------------------------------------------------------------
    // 16. sign(x) * absolute(x) == x for finite non-NaN
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sign_times_abs(data in proptest::collection::vec(-100.0f64..100.0, 1..=20)) {
        let a = arr1(data);
        let s = sign(&a).unwrap();
        let abs_a = absolute(&a).unwrap();
        let reconstructed = multiply(&s, &abs_a).unwrap();
        prop_assert!(approx_eq_arr(&reconstructed, &a, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 17. subtract(a, a) == zeros
    // -----------------------------------------------------------------------
    #[test]
    fn prop_subtract_self_is_zero(data in proptest::collection::vec(-100.0f64..100.0, 1..=20)) {
        let n = data.len();
        let a = arr1(data);
        let result = subtract(&a, &a).unwrap();
        let z = const_arr(0.0, n);
        prop_assert!(approx_eq_arr(&result, &z, 1e-15));
    }

    // -----------------------------------------------------------------------
    // 18. divide(a, a) ~= ones (for non-zero a)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_divide_self_is_one(data in proptest::collection::vec(1.0f64..100.0, 1..=20)) {
        let n = data.len();
        let a = arr1(data);
        let result = divide(&a, &a).unwrap();
        let o = const_arr(1.0, n);
        prop_assert!(approx_eq_arr(&result, &o, 1e-14));
    }

    // -----------------------------------------------------------------------
    // 19. cos(x)^2 + sin(x)^2 ~= 1 (Pythagorean identity)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_pythagorean_identity(x in -100.0f64..100.0) {
        let input = scalar_arr(x);
        let s = sin(&input).unwrap();
        let c = cos(&input).unwrap();
        let s2 = square(&s).unwrap();
        let c2 = square(&c).unwrap();
        let sum = add(&s2, &c2).unwrap();
        let val = *sum.iter().next().unwrap();
        prop_assert!((val - 1.0).abs() < 1e-10, "sin^2 + cos^2 = {} != 1 for x={}", val, x);
    }
}
