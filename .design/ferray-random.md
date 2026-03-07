# Feature: ferray-random — NumPy's modern Generator API with full distribution coverage

## Summary
Implements NumPy's modern `numpy.random` Generator/BitGenerator model (not the legacy module-level functions). Provides pluggable BitGenerators (PCG64, Philox, Xoshiro256**), 30+ continuous and discrete distributions, permutation/sampling operations, and deterministic parallel generation via jump-ahead or stream splitting. The most commonly needed and most scattered part of the current Rust numeric ecosystem, unified under a single coherent API.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Dimension, Element, FerrayError)
- **Downstream**: ferray (re-export)
- **External crates**: `rayon` (parallel generation), `rand_core` 0.6 (RNG trait compatibility)
- **Phase**: 2 — Submodules

## Requirements

### Generator and BitGenerator (Section 10.1)
- REQ-1: `random::default_rng()` and `random::default_rng_seeded(seed)` — convenience constructors (default BitGenerator: Xoshiro256**)
- REQ-2: `random::Generator::new(bit_generator)` — explicit BitGenerator selection
- REQ-3: Implement BitGenerators: `Pcg64`, `Philox`, `Xoshiro256StarStar` — all implement a common `BitGenerator` trait
- REQ-4: `Generator` takes `&mut self` — it is stateful and NOT `Sync`. Thread-safety is handled via spawning.

### Parallel Generation
- REQ-5: `rng.standard_normal_parallel(shape)` — parallel generation using per-thread generators derived via `BitGenerator::jump()` (Xoshiro256**) or stream IDs (Philox). Result is deterministic given root seed and output size regardless of thread count.
- REQ-6: `rng.spawn(n)` — manual splitting into n independent child generators
- REQ-7: BitGenerators without jump-ahead (Pcg64) fall back to sequential generation with a documented warning

### Distributions — Continuous (Section 10.2)
- REQ-8: `random(size)` (Uniform [0,1)), `uniform(low, high, size)`, `standard_normal(size)`, `normal(loc, scale, size)`, `standard_exponential(size)`, `exponential(scale, size)`, `gamma(shape, scale, size)`, `beta(a, b, size)`, `chisquare(df, size)`, `f(dfnum, dfden, size)`, `student_t(df, size)`, `laplace(loc, scale, size)`, `logistic(loc, scale, size)`, `lognormal(mean, sigma, size)`, `rayleigh(scale, size)`, `weibull(a, size)`, `pareto(a, size)`, `gumbel(loc, scale, size)`, `power(a, size)`, `triangular(left, mode, right, size)`, `vonmises(mu, kappa, size)`, `wald(mean, scale, size)`, `standard_cauchy(size)`, `standard_gamma(shape, size)`

### Distributions — Discrete (Section 10.3)
- REQ-9: `integers(low, high, size)`, `binomial(n, p, size)`, `negative_binomial(n, p, size)`, `poisson(lam, size)`, `geometric(p, size)`, `hypergeometric(ngood, nbad, nsample, size)`, `logseries(p, size)`, `multinomial(n, pvals, size)`, `multivariate_normal(mean, cov, size)`, `dirichlet(alpha, size)`

### Permutations and Sampling (Section 10.4)
- REQ-10: `shuffle(&mut a)` (in-place), `permutation(&a)` (returns copy), `permuted(&a, axis)` (along axis), `choice(&a, size, replace, p)` (with optional probability weights)

### Determinism
- REQ-11: All generation methods are deterministic given the same seed and shape. Parallel generation uses a fixed assignment of index ranges to stream IDs, not dynamic work-stealing.

## Acceptance Criteria
- [ ] AC-1: `default_rng_seeded(42).standard_normal((1000,))` produces the same values on every invocation
- [ ] AC-2: All continuous distributions produce samples whose empirical statistics (mean, variance) match theoretical values within 3 sigma on 100k samples (statistical test)
- [ ] AC-3: `standard_normal_parallel((1_000_000,))` produces the same output as `standard_normal((1_000_000,))` with the same seed (deterministic parallel)
- [ ] AC-4: `spawn(4)` produces 4 generators whose output streams do not overlap for 2^64 samples each
- [ ] AC-5: `choice(&a, 5, replace=false)` never returns duplicate elements
- [ ] AC-6: `multinomial` and `dirichlet` produce correctly shaped output arrays matching NumPy
- [ ] AC-7: `shuffle` modifies in-place; `permutation` returns a new array; neither corrupts data
- [ ] AC-8: `cargo test -p ferray-random` passes. `cargo clippy -p ferray-random -- -D warnings` clean.

## Architecture

### Crate Layout
```
ferray-random/
  Cargo.toml
  src/
    lib.rs
    generator.rs              # Generator struct, default_rng, default_rng_seeded
    bitgen/
      mod.rs                  # BitGenerator trait
      pcg64.rs
      philox.rs
      xoshiro256.rs
    distributions/
      mod.rs
      uniform.rs              # random, uniform, integers
      normal.rs               # standard_normal, normal, lognormal
      exponential.rs          # standard_exponential, exponential
      gamma.rs                # gamma, beta, chisquare, f, student_t
      misc_continuous.rs      # laplace, logistic, rayleigh, weibull, pareto, gumbel, etc.
      discrete.rs             # binomial, poisson, geometric, hypergeometric, etc.
      multivariate.rs         # multinomial, multivariate_normal, dirichlet
    permutations.rs           # shuffle, permutation, permuted, choice
    parallel.rs               # Parallel generation via jump-ahead / stream splitting
```

### BitGenerator Trait
```rust
pub trait BitGenerator: Send {
    fn next_u64(&mut self) -> u64;
    fn seed_from_u64(seed: u64) -> Self where Self: Sized;
    fn jump(&mut self) -> Option<()>;  // None if jump not supported
    fn stream(seed: u64, stream_id: u64) -> Option<Self> where Self: Sized;  // None if streams not supported
}
```

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Legacy `np.random.*` module-level functions (deprecated in NumPy, not implemented)
- Cryptographically secure RNG (use `rand::OsRng` directly)
