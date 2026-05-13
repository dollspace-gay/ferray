# Contributing to ferray

Thanks for taking the time to contribute. This document captures the rules that
keep ferray's NumPy-parity claim honest. Read it before you add or change a
public function.

For project conventions (Rust edition, MSRV, import paths, error handling,
SIMD strategy, naming, dependency versions, and code-quality rules), see
[`CLAUDE.md`](CLAUDE.md).

## Adding a public function — the conformance contract

ferray enforces a **strict surface-coverage gate** on every crate. CI fails if any `pub fn` / `pub struct` / `pub trait` is added without one of:

1. **A conformance test reference** in `<crate>/tests/conformance_<category>.rs` that mentions the function's full path (canonical inner path + re-export path).
2. **An exclusion entry** in `<crate>/tests/conformance/_surface_exclusions.toml` with non-empty `reason` and `covered_by` fields.
3. **A divergence entry** in `<crate>/tests/conformance/_divergences.toml` (for cases where ferray intentionally returns a more mathematically correct value than numpy/scipy, with cited justification).

### Workflow when adding a new public function

1. Regenerate the surface inventory: `cargo run -p ferray-test-oracle --bin surface-inventory -- <crate>`.
2. The gate will fail; the error message lists each uncovered path.
3. Either:
   - Add a conformance test in `tests/conformance_<category>.rs` that loads a fixture from `fixtures/<crate>/<name>.json` (generate via `scripts/generate_fixtures.py`) and asserts against numpy/scipy output. Tolerances come from `docs/conformance-suites.md` Stage 1 table (1 ULP for arithmetic, 1e-12 rel for f64 transcendentals, etc.). DO NOT relax tolerance to silence failure — escalate as a `TOLERANCE_GAP` issue.
   - Add an exclusion entry with a cited `covered_by` test path (NOT generic "covered elsewhere").
   - Add a divergence entry with citation + tracking issue (for ferray-more-correct cases).
4. Re-run the gate; it should pass.
5. Commit.

See `docs/conformance-suites.md` for the full four-layer pattern and `docs/conformance-suites.md#per-finding-classification-taxonomy` for how to classify first-run failures (MATCH / BUG / DIVERGENCE / TOLERANCE_GAP / FIXTURE_BUG).

### Cascade-bug discipline

When a conformance test reveals a real numerical divergence:

1. **File a tracking issue** (`crosslink quick "..." -p high -l "bug,cascade,conformance"`).
2. **Mark the test** with `#[ignore = "...; tracking #N"]` so the suite stays green.
3. **DO NOT fix bugs inline** during the dispatch that surfaces them — separate dispatches own bug fixes.
4. Reference the umbrella issue #748 (the rollout) when filing if related.
