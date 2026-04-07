# ferray-random

Random number generation and distributions for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **Generator API**: `Generator` type with pluggable `BitGenerator` backends
- **BitGenerators**: Xoshiro256** (default), PCG64, Philox
- **30+ distributions**: Normal, Uniform, Exponential, Poisson, Binomial, Gamma, Beta, Chi-squared, Student-t, Laplace, Weibull, and more
- **Permutations**: `shuffle`, `permutation`, `choice` (with/without replacement, weighted)
- **Parallel generation**: `standard_normal_parallel` with Rayon + jump-ahead
- **Deterministic**: All output is reproducible given the same seed

## Usage

```rust
use ferray_random::{default_rng_seeded, Generator};

let mut rng = default_rng_seeded(42);

// Uniform [0, 1)
let samples = rng.random(1000).unwrap();

// Standard normal
let normals = rng.standard_normal(1000).unwrap();

// Integers in [0, 10)
let ints = rng.integers(0, 10, 100).unwrap();
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `random` feature.

## License

MIT OR Apache-2.0
