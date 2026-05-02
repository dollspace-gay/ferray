// ferray-fft: Complete numpy.fft parity with plan caching
//
//! FFT operations for the ferray numeric computing library.
//!
//! This crate provides the full `numpy.fft` surface:
//!
//! - **Complex FFTs**: [`fft`], [`ifft`], [`fft2`], [`ifft2`], [`fftn`], [`ifftn`]
//! - **Real FFTs**: [`rfft`], [`irfft`], [`rfft2`], [`irfft2`], [`rfftn`], [`irfftn`]
//! - **Hermitian FFTs**: [`hfft`], [`ihfft`]
//! - **Frequency utilities**: [`fftfreq`], [`rfftfreq`]
//! - **Shift utilities**: [`fftshift`], [`ifftshift`]
//! - **Plan caching**: [`FftPlan`] for efficient repeated transforms
//! - **Normalization**: [`FftNorm`] enum matching `NumPy`'s `norm` parameter
//!
//! ## Precision
//!
//! All FFT functions are generic over [`FftFloat`], implemented for both
//! `f32` and `f64`. The same call works with either precision:
//!
//! ```ignore
//! let spectrum_f64 = ferray_fft::fft(&complex_f64_array, None, None, FftNorm::Backward)?;
//! let spectrum_f32 = ferray_fft::fft(&complex_f32_array, None, None, FftNorm::Backward)?;
//! ```
//!
//! Plan caches are per-precision — using only f64 means the f32 plan
//! cache is never constructed.
//!
//! ## Real-input convenience
//!
//! Real arrays can be transformed with the full complex FFT via
//! [`fft_real`] / [`ifft_real`] / [`fft_real2`] / [`fft_realn`], which
//! auto-promote real inputs to complex before calling the standard
//! complex FFT. For the half-spectrum form use [`rfft`] / [`rfftn`]
//! instead.
//!
//! Internally powered by [`rustfft`](https://crates.io/crates/rustfft)
//! and [`realfft`](https://crates.io/crates/realfft) with automatic plan
//! caching for repeated transforms of the same size.

// FFT kernels divide by `usize`-typed lengths cast to the float precision
// in use, and truncate `f64` shift offsets back into `usize` array indices.
// Roundtrip and Hermitian-symmetry tests assert exact float equality
// against analytic reference values. These are part of the algebra, not
// precision bugs.
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

mod axes;
pub mod complex;
pub mod float;
pub mod freq;
pub mod hermitian;
mod nd;
pub mod norm;
pub mod plan;
pub mod real;
pub mod shift;

// Re-export public API at crate root for ergonomic access.

// Normalization
pub use norm::FftNorm;

// Plan caching
pub use plan::{FftPlan, FftPlanND};

// Complex FFTs (REQ-1..REQ-4)
pub use complex::{fft, fft2, fftn, ifft, ifft2, ifftn};

// Pre-allocated `_into` variants of the 1-D complex/real FFTs (#429).
pub use complex::{fft_into, ifft_into};

// Real-input convenience wrappers — auto-promote real arrays to complex.
// For the Hermitian-folded half-spectrum form use `rfft`/`rfftn` instead.
pub use complex::{fft_real, fft_real2, fft_realn, ifft_real};

// Real FFTs (REQ-5..REQ-7)
pub use real::{irfft, irfft2, irfftn, rfft, rfft2, rfftn};

// Pre-allocated `_into` variants of the 1-D real FFTs (#429).
pub use real::{irfft_into, rfft_into};

// Hermitian FFTs (REQ-8)
pub use hermitian::{hfft, hfft2, hfftn, ihfft, ihfft2, ihfftn};

// Frequency utilities (REQ-9, REQ-10)
pub use freq::{fftfreq, rfftfreq};

// Shift utilities (REQ-11)
pub use shift::{fftshift, ifftshift};
