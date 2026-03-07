# Feature: Phase 5 — Formal Verification, Oracle Testing, and Final Quality Gate

## Summary
The final phase: comprehensive verification that ferray produces correct results. Generates NumPy oracle fixtures for every public function, runs property-based tests for algorithmic invariants, verifies SIMD-vs-scalar equivalence, executes statistical equivalence benchmarks against NumPy, runs fuzz campaigns for robustness, and applies formal verification (Prusti annotations) to core primitives. This phase produces no new features — it proves that existing features are correct. Nothing ships until this phase passes.

## Dependencies
- **Upstream**: ALL phases (1-4) must be complete. The full ferray workspace must compile and pass basic tests.
- **External tooling**: Python 3.12+ with NumPy (for fixture generation), `cargo-fuzz` with libFuzzer, Prusti (Rust formal verifier), `criterion` (benchmarks)
- **Phase**: 5 — Verification (final gate before 1.0)

## Requirements

### Layer 1 — NumPy Oracle Fixtures (adapted from ferrolearn Section 20.1)
- REQ-1: A Python script (`scripts/generate_fixtures.py`) generates golden fixtures by running NumPy on curated inputs. Fixtures are JSON files under `fixtures/` committed to the repository.
- REQ-2: Fixture coverage per function: standard case, non-default parameters (3+ configs), edge cases (empty arrays, single-element, NaN/Inf inputs, very large values 1e6+, very small values 1e-7), all supported dtypes (f32, f64, Complex64, Complex128, relevant integer types), shapes from 0-D to 5-D
- REQ-3: Rust oracle tests load fixtures and compare ferray output using ULP-based tolerance:
  - Elementwise ufunc outputs: 1 ULP for correctly-rounded, 4 ULP max for approximations
  - Transcendental functions (sin, exp, log): 4 ULP budget, documented per function
  - Linear algebra outputs: 4 ULP for well-conditioned inputs, 10 ULP for iterative solvers
  - Integer and boolean outputs: exact match
- REQ-4: Every public function in every ferray subcrate must have at least one oracle test

### Layer 2 — Property-Based Testing for Algorithmic Invariants
- REQ-5: Every public function must have a minimum of 8 property-based tests using `proptest` with `ProptestConfig::with_cases(256)`. Properties are derived from mathematical definitions, not reference implementations.
- REQ-6: Required invariant categories:
  - Ufunc properties: `f(f_inverse(x)) == x` for inverse pairs (sin/arcsin, exp/log), broadcasting shape correctness, type promotion correctness
  - Linalg properties: decomposition reconstruction (A == Q*R, A == U*S*Vt), orthonormality of eigenvectors, det(A*B) == det(A)*det(B)
  - Stats properties: `mean(constant_array) == constant`, `var(constant) == 0`, `sort` is idempotent, `unique` output is sorted
  - FFT properties: `ifft(fft(x)) == x` (round-trip), Parseval's theorem (energy conservation)
  - Random properties: determinism with same seed, output shape matches requested shape, distribution moments within tolerance

### Layer 3 — SIMD vs Scalar Verification
- REQ-7: The full oracle fixture test suite must pass under `FERRAY_FORCE_SCALAR=1` (scalar code path)
- REQ-8: Results of SIMD and scalar paths must be bit-identical for all operations on the same input
- REQ-9: Benchmark both paths with `criterion` to verify SIMD achieves at least 2x throughput on contiguous inputs (on hardware with AVX2+)

### Layer 4 — Statistical Equivalence Benchmarking (adapted from ferrolearn Section 20.3)
- REQ-10: A Python harness (`benchmarks/statistical_equivalence.py`) runs both NumPy and ferray on identical inputs and applies Welch's t-test to verify no statistically significant regression (one-sided, alpha=0.05)
- REQ-11: Benchmark suite covers: all ufunc families (trig, exp/log, arithmetic) on arrays of size 1k, 100k, 10M; linalg operations on matrices 10x10 to 1000x1000; FFT on signals of length 64 to 1M; random distribution moments on 100k samples
- REQ-12: No function may produce a statistically significantly worse result than NumPy on any benchmark input

### Layer 5 — Fuzz Testing for Numerical Robustness (adapted from ferrolearn Section 20.5)
- REQ-13: Every public function that accepts array input must have a fuzz target in `fuzz/fuzz_targets/`
- REQ-14: The contract: ferray either returns `Ok(valid_result)` or `Err(FerrayError)` — never panics, never hangs, never propagates NaN from non-NaN input
- REQ-15: Fuzz corpus must be run for a minimum of 24 CPU-hours before release
- REQ-16: Mandatory seed corpus inputs: all-zero, all-NaN, single-row, single-column, n_samples < n_features, f64::MAX, f64::MIN_POSITIVE, duplicate rows, perfectly collinear features
- REQ-17: Any panic discovered by fuzzing is a P0 bug that blocks release

### Layer 6 — Formal Verification of Core Primitives (adapted from ferrolearn Section 20.6)
- REQ-18: `FerrayError` variants: Prusti pre/postconditions on all public functions that return `Result` verifying that error conditions are correctly detected (e.g., shape mismatch → ShapeMismatch error)
- REQ-19: Broadcasting: formally verify that `broadcast_shapes` output satisfies NumPy's broadcasting rules as logical postconditions
- REQ-20: Index bounds: formally verify that all indexing operations either return valid data or `Err(IndexOutOfBounds)` — no out-of-bounds access is possible
- REQ-21: Type promotion: formally verify that `promoted_type!(A, B)` satisfies the promotion table (smallest type without precision loss)
- REQ-22: Reduction correctness: formally verify that `sum(axis=None)` returns the mathematical sum of all elements (modulo floating-point associativity)
- REQ-23: NdArray structural invariants: after any public constructor or mutation, `strides` and `shape` are consistent, `data.len() >= product(shape)`, and contiguity flags are correct

### Verification Summary Gate
- REQ-24: The following must ALL pass before 1.0 release:
  - All oracle fixtures pass (every function, every dtype, every edge case)
  - All property-based tests pass (8+ per function, 256 cases each)
  - SIMD == scalar bit-identity verified
  - Statistical equivalence: no significant regressions vs NumPy
  - Fuzz corpus: 24 CPU-hours with zero panics
  - Formal verification: all mandatory Prusti annotations verify
  - `cargo test --workspace --all-features` passes
  - `cargo clippy --workspace --all-features -- -D warnings` clean

## Acceptance Criteria
- [ ] AC-1: `scripts/generate_fixtures.py` runs successfully and produces fixtures for all public functions in all subcrates
- [ ] AC-2: `cargo test --workspace` passes with fixture tests covering every public function
- [ ] AC-3: `FERRAY_FORCE_SCALAR=1 cargo test --workspace` passes with zero failures
- [ ] AC-4: SIMD and scalar paths produce bit-identical output on the full fixture suite
- [ ] AC-5: `benchmarks/statistical_equivalence.py` reports zero statistically significant regressions
- [ ] AC-6: `cargo fuzz run` for 24 CPU-hours discovers zero panics
- [ ] AC-7: All Prusti annotations verify (broadcasting, indexing, type promotion, structural invariants)
- [ ] AC-8: A verification report document (`docs/verification-report.md`) is generated summarizing all results with pass/fail status, ULP budgets observed, and any known numerical differences documented

## Architecture

### Directory Structure
```
scripts/
  generate_fixtures.py        # NumPy fixture generator
  verify_simd_scalar.sh       # Runs tests with and without FERRAY_FORCE_SCALAR
fixtures/
  core/                       # ferray-core fixtures (creation, indexing, broadcasting)
  ufunc/                      # ferray-ufunc fixtures (all math functions)
  stats/                      # ferray-stats fixtures (reductions, histograms, sorting)
  linalg/                     # ferray-linalg fixtures (decompositions, solvers)
  fft/                        # ferray-fft fixtures
  random/                     # ferray-random fixtures (distribution moments)
  io/                         # ferray-io fixtures (.npy round-trip)
  polynomial/                 # ferray-polynomial fixtures
  strings/                    # ferray-strings fixtures
  ma/                         # ferray-ma fixtures
benchmarks/
  statistical_equivalence.py  # NumPy vs ferray statistical comparison
  ferray_bench/               # Rust benchmark binary for criterion
    Cargo.toml
    src/main.rs
fuzz/
  Cargo.toml
  fuzz_targets/
    core_creation.rs
    core_indexing.rs
    core_broadcasting.rs
    ufunc_trig.rs
    ufunc_arithmetic.rs
    linalg_decomp.rs
    linalg_solve.rs
    fft_forward_inverse.rs
    random_distributions.rs
    stats_reductions.rs
    # ... one per major function family
docs/
  verification-report.md      # Generated summary of all verification results
  numerical_differences.md    # Documented per-function ULP budgets and known differences
```

### Verification Agent Model
Phase 5 is best executed by multiple specialized agents:
- **Fixture Agent** (sonnet): generates `scripts/generate_fixtures.py` and all fixture JSON files
- **Oracle Test Agent** (sonnet): writes Rust oracle tests loading fixtures for each subcrate
- **Property Test Agent** (opus): writes proptest invariants requiring mathematical reasoning
- **SIMD Verification Agent** (sonnet): writes the SIMD/scalar comparison harness
- **Fuzz Agent** (sonnet): writes fuzz targets and seed corpus
- **Formal Verification Agent** (opus): writes Prusti annotations — requires deep type-system reasoning
- **Integration Agent** (opus): runs the full suite, generates the verification report, triages failures

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Performance optimization (Phase 5 verifies correctness, not speed — performance is a Phase 1-4 concern)
- New feature development
- Publishing to crates.io (human step after Phase 5 passes)
