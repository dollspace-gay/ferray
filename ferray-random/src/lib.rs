// ferray-random: NumPy-compatible random number generation for Rust
//
//! # ferray-random
//!
//! Implements NumPy's modern `Generator`/`BitGenerator` model with pluggable
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
//! ## BitGenerators
//!
//! Three BitGenerators are provided:
//! - [`Xoshiro256StarStar`](bitgen::Xoshiro256StarStar) — default, fast, supports jump-ahead
//! - [`Pcg64`](bitgen::Pcg64) — PCG family, good statistical properties
//! - [`Philox`](bitgen::Philox) — counter-based, supports stream IDs for parallel generation
//!
//! ## Determinism
//!
//! All generation is deterministic given the same seed and shape. Parallel
//! generation via [`standard_normal_parallel`](Generator::standard_normal_parallel)
//! produces output identical to sequential generation with the same seed.

pub mod bitgen;
pub mod distributions;
pub mod generator;
pub mod parallel;
pub mod permutations;
pub mod shape;

pub use bitgen::{BitGenerator, Pcg64, Philox, Xoshiro256StarStar};
pub use generator::{Generator, default_rng, default_rng_seeded};
pub use shape::IntoShape;
