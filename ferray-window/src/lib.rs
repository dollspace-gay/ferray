//! Window functions and functional programming utilities for ferray.
//!
//! This crate provides NumPy-equivalent window functions for signal processing
//! and spectral analysis (`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`),
//! along with functional programming utilities (`vectorize`, `piecewise`,
//! `apply_along_axis`, `apply_over_axes`).
//!
//! All window functions return `Array1<f64>` and match `NumPy`'s output to
//! high precision.

// Window functions evaluate analytic formulae over `usize`-indexed grids
// and produce `f64` tap weights; the integer-to-float lift is intrinsic
// to the math, and oracle tests compare endpoint values against analytic
// closed forms via exact equality.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms,
    clippy::suboptimal_flops
)]

pub mod functional;
pub mod windows;

// Re-export window functions at crate root for convenience
pub use windows::{
    bartlett, bartlett_f32, blackman, blackman_f32, bohman, boxcar, cosine, exponential, flattop,
    gaussian, general_cosine, general_hamming, hamming, hamming_f32, hanning, hanning_f32, kaiser,
    kaiser_f32, lanczos, nuttall, parzen, taylor, triang, tukey,
};

// Re-export functional utilities at crate root. `vectorize_nd` was
// removed as a redundant alias for `vectorize` — vectorize itself is
// generic over dimension (#293).
pub use functional::{apply_along_axis, apply_over_axes, piecewise, sum_axis_keepdims, vectorize};
