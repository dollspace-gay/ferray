//! Divergence pin: `ferray_window::kaiser` rejects a `beta` value that
//! NumPy's `numpy.kaiser` evaluates to a finite, correct window.
//!
//! ## Upstream contract (live numpy 2.4.4 oracle, R-CHAR-3)
//!
//! NumPy's `kaiser(M, beta)` computes `i0(beta * sqrt(...)) / i0(beta)`
//! (`numpy/lib/_function_base_impl.py:3735`). The normaliser `i0(beta)`
//! uses the Cephes Chebyshev expansion `_i0_2(x) = exp(x) * chbevl(...) /
//! sqrt(x)` (`_function_base_impl.py:3536`). For `x` up to ~709.78
//! (`ln(f64::MAX)`), `exp(x)` is still finite, so `i0(beta)` stays finite
//! and the whole window is finite and well-defined.
//!
//! Concretely, `np.kaiser(5, 709.0)` returns (oracle, verified live):
//!
//! ```text
//!   [8.11986409099987017e-307, 6.00464311823984035e-42, 1.0,
//!    6.00464311823984035e-42, 8.11986409099987017e-307]
//! ```
//!
//! all finite; `i0(709.0) == 1.2315477067016538e+306` (finite).
//!
//! ferray's `kaiser` (ferray-window/src/windows/mod.rs:192-198) hard-codes
//! `BETA_OVERFLOW_THRESHOLD = 708.0` and returns `Err(InvalidValue)` for
//! every `|beta| > 708.0`. So `kaiser(5, 709.0)` is rejected even though
//! NumPy produces a perfectly good finite window. The conservative
//! threshold sacrifices a full `[708, 709.78]` band of valid betas that
//! upstream accepts.
//!
//! This test calls the target `kaiser(5, 709.0)` and asserts the NumPy
//! oracle values. It FAILS today because the target returns `Err`.
//!
//! Tracking: #1087 (`-l blocker`). Left un-`#[ignore]`d: numpy accepts
//! a valid input that ferray rejects, so this is a release-blocker and the
//! failing test IS the block.

use ferray_window::kaiser;

/// Divergence: target's `kaiser` diverges from
/// `numpy.kaiser` (`numpy/lib/_function_base_impl.py:3735`,
/// i0 normaliser `:3536`) for `M=5, beta=709.0`.
/// Upstream returns a finite window
/// `[8.11986e-307, 6.00464e-42, 1.0, 6.00464e-42, 8.11986e-307]`;
/// target returns `Err(InvalidValue)` because of an overly conservative
/// `BETA_OVERFLOW_THRESHOLD = 708.0` (mod.rs:192).
#[test]
fn divergence_kaiser_beta_709_rejected() {
    // NumPy oracle values for np.kaiser(5, 709.0) (numpy 2.4.4, R-CHAR-3).
    // i0(709.0) = 1.2315477067016538e+306 is finite in f64, so every
    // sample is finite.
    let expected: [f64; 5] = [
        8.119_864_090_999_87e-307,
        6.004_643_118_239_84e-42,
        1.0,
        6.004_643_118_239_84e-42,
        8.119_864_090_999_87e-307,
    ];

    let got = kaiser(5, 709.0)
        .expect("upstream numpy.kaiser(5, 709.0) is finite; target must not reject beta=709");

    let s = got.as_slice().unwrap();
    assert_eq!(s.len(), 5, "kaiser(5, 709.0) must have length 5");
    for (i, (&a, &e)) in s.iter().zip(expected.iter()).enumerate() {
        // Tight relative tolerance: ferray and numpy share the identical
        // Cephes i0 coefficients, so once the value is computed it should
        // match to a few ULP. Use relative tolerance to cover the wide
        // dynamic range (1.0 down to 8e-307).
        let denom = e.abs().max(f64::MIN_POSITIVE);
        let rel = (a - e).abs() / denom;
        assert!(
            rel <= 1e-12,
            "kaiser(5, 709.0)[{i}]: got {a:e}, numpy {e:e} (rel diff {rel:e})"
        );
    }
}
