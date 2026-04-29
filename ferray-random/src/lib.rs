// ferray-random: NumPy-compatible random number generation for Rust
//
//! # ferray-random
//!
//! Implements `NumPy`'s modern `Generator`/`BitGenerator` model with pluggable
//! pseudo-random number generators, 30+ continuous and discrete distributions,
//! permutation/sampling operations, and deterministic parallel generation.
//!
//! ## Quick Start
//!
//! ```
//! use ferray_random::{default_rng_seeded, Generator};
//!
//! let mut rng = default_rng_seeded(42);
//!
//! // Uniform [0, 1)
//! let values = rng.random(100).unwrap();
//!
//! // Standard normal
//! let normals = rng.standard_normal(100).unwrap();
//!
//! // Integers in [0, 10)
//! let ints = rng.integers(0, 10, 100).unwrap();
//! ```
//!
//! ## `BitGenerators`
//!
//! Three `BitGenerators` are provided:
//! - [`Xoshiro256StarStar`](bitgen::Xoshiro256StarStar) — default, fast, supports jump-ahead
//! - [`Pcg64`](bitgen::Pcg64) — PCG family, good statistical properties
//! - [`Philox`](bitgen::Philox) — counter-based, supports stream IDs for parallel generation
//!
//! ## Determinism
//!
//! All generation is deterministic given the same seed and shape. Parallel
//! generation via [`standard_normal_parallel`](Generator::standard_normal_parallel)
//! produces output identical to sequential generation with the same seed.

// RNG kernels lift `u64`/`usize` bit streams into `f32`/`f64` samples and
// truncate `f64` results back into integer bins (Bernoulli/binomial,
// Poisson tail, choice). Reproducibility tests assert exact float
// equality against fixed seeded outputs. These int<->float crossings and
// exact comparisons are the core of every distribution.
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
    clippy::suboptimal_flops,
    clippy::while_float
)]

pub mod bitgen;
pub mod distributions;
pub mod generator;
pub mod parallel;
pub mod permutations;
pub mod shape;

pub use bitgen::{BitGenerator, Pcg64, Philox, Xoshiro256StarStar};
pub use generator::{Generator, default_rng, default_rng_seeded};
pub use shape::IntoShape;
