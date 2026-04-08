// ferray-ufunc: Shared test helpers.
//
// Lives under `#[cfg(test)]` so it never contributes to release
// builds. Before this module existed, every `#[cfg(test)]` submodule
// grew a private `arr1` helper (see issue #148 — roughly a dozen
// near-identical 3-line copies). Put any new single-file test helpers
// here rather than redefining them per-module.

#![cfg(test)]

use ferray_core::Array;
use ferray_core::dimension::Ix1;

/// Build a 1-D `f64` test array from a slice-ish input.
///
/// Accepts anything that can be copied into a `Vec<f64>`, so callers
/// can use either `arr1(&[1.0, 2.0])` or `arr1(vec![1.0, 2.0])`.
pub(crate) fn arr1<I: Into<Vec<f64>>>(data: I) -> Array<f64, Ix1> {
    let data = data.into();
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}
