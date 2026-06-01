# ferray-random

NumPy-style random number generation and distributions for Rust.

Part of the [ferray](../README.md) workspace — a Rust-native, NumPy-equivalent scientific computing library.

## Overview

`ferray-random` implements NumPy's modern `Generator`/`BitGenerator` model on top
of the ferray array types:

- **Bit generators** — pluggable PRNG cores implementing the `BitGenerator` trait:
  [`Xoshiro256StarStar`] (the default), [`Pcg64`], [`Pcg64Dxsm`], [`Philox`],
  [`Sfc64`], and [`MT19937`]. Each supports full state serialization
  (`state_bytes` / `set_state_bytes`).
- **`Generator`** — the user-facing, stateful (`&mut self`) RNG. Construct it with
  `default_rng()` (OS-seeded via `getrandom`) or `default_rng_seeded(seed)`
  (deterministic). Independent child generators are produced with `Generator::spawn`.
- **`SeedSequence`** — the NumPy `SeedSequence` equivalent: mixes arbitrary entropy
  into high-quality seed words and spawns reproducible child sequences for parallel work.
- **Distributions** — 30+ continuous and discrete samplers exposed as `Generator`
  methods, including `random`, `uniform`, `integers`, `standard_normal`, `normal`,
  `standard_exponential`, `exponential`, `standard_gamma`, `gamma`, `beta`,
  `chisquare`, `noncentral_chisquare`, `f`, `noncentral_f`, `standard_t` /
  `student_t`, `standard_cauchy`, `lognormal`, `laplace`, `logistic`, `gumbel`,
  `pareto`, `power`, `rayleigh`, `weibull`, `wald`, `triangular`, `vonmises`,
  `binomial`, `poisson`, `geometric`, `negative_binomial`, `hypergeometric`,
  `multinomial`, `dirichlet`, `multivariate_normal`, and `zipf` / `logseries`.
- **Permutations & sampling** — `shuffle`, `permutation`, `permuted`, and `choice`
  (with or without replacement, optionally weighted), plus their N-D `*_dyn` variants.
- **Parallel generation** — `standard_normal_parallel` fans work across Rayon threads
  using jump-ahead, producing output identical to the sequential path for the same seed.

The legacy `numpy.random.RandomState` API is **not** provided; this crate targets the
modern `Generator` interface only.

## NumPy correspondence

| NumPy | ferray-random |
|-------|---------------|
| `np.random.default_rng()` | `default_rng()` |
| `np.random.default_rng(seed)` | `default_rng_seeded(seed)` |
| `np.random.Generator` | `Generator<B>` |
| `np.random.SeedSequence` | `SeedSequence` |
| `np.random.PCG64` / `PCG64DXSM` | `Pcg64` / `Pcg64Dxsm` |
| `np.random.Philox` | `Philox` |
| `np.random.SFC64` | `Sfc64` |
| `np.random.MT19937` | `MT19937` |
| `rng.random` / `uniform` / `integers` | `random` / `uniform` / `integers` |
| `rng.standard_normal` / `normal` | `standard_normal` / `normal` |
| `rng.standard_exponential` / `exponential` | `standard_exponential` / `exponential` |
| `rng.standard_gamma` / `gamma` / `beta` | `standard_gamma` / `gamma` / `beta` |
| `rng.binomial` / `poisson` / `geometric` | `binomial` / `poisson` / `geometric` |
| `rng.choice` / `shuffle` / `permutation` | `choice` / `shuffle` / `permutation` |

## Reproducibility note

Output is **deterministic given the same seed, bit generator, and call sequence** —
`default_rng_seeded(42)` always yields the same stream. The crate does **not**
reproduce NumPy's byte-exact sample stream: ferray's bit-generator state machines and
distribution kernels differ from NumPy's reference, and per-sample equality with NumPy
is explicitly out of scope. Conformance is validated against **distributional moments**
(sample mean and variance over 10,000 draws), not raw samples. Treat the streams as
statistically equivalent to NumPy, not bit-for-bit identical.

## Feature flags

This crate currently exposes **no Cargo features**; it builds with its default
dependencies only. Functionality is selected through the API (choice of
`BitGenerator`), not through feature gates.

## Example

```rust
use ferray_random::{default_rng_seeded, BitGenerator, Generator, Pcg64};

// Deterministic, default bit generator (Xoshiro256**).
let mut rng = default_rng_seeded(42);

let normals = rng.standard_normal(1000).unwrap();   // ~N(0, 1)
let scaled = rng.normal(5.0, 2.0, 1000).unwrap();   // ~N(5, 2)
let ints = rng.integers(0, 10, 100).unwrap();       // uniform in [0, 10)

// Explicitly choose a different bit generator.
let mut pcg = Generator::new(Pcg64::seed_from_u64(7));
let u = pcg.random(64).unwrap();                    // uniform [0, 1)
```

## MSRV & edition

- Edition: 2024
- MSRV: 1.88 (stable)
- License: MIT OR Apache-2.0

[`Xoshiro256StarStar`]: bitgen::Xoshiro256StarStar
[`Pcg64`]: bitgen::Pcg64
[`Pcg64Dxsm`]: bitgen::Pcg64Dxsm
[`Philox`]: bitgen::Philox
[`Sfc64`]: bitgen::Sfc64
[`MT19937`]: bitgen::MT19937
