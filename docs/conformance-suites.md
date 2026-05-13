# Conformance suites: what they are, when they help, how to build one

A conformance suite is a third axis of testing — distinct from unit tests and
integration tests. Where unit tests prove that a function does what its body
says, and integration tests prove that subsystems compose, a conformance suite
proves that **what the public API claims it does matches what it actually does,
against an external ground truth**.

This document explains the pattern in the context of **ferray** — a pure-Rust
NumPy reimplementation targeting `import ferray as np` as a drop-in for
`import numpy as np`. The reference is **numpy 2.4.4** (pinned). The
pattern was first developed for ferrotorch (a PyTorch reimplementation); this
document is the ferray adaptation. Where ferray's reality differs from
ferrotorch's, the delta is called out explicitly.

> **Ferray-specific framing**: ferray's conformance contract is not strict
> parity. The project's stated goal is correctness-first: ferray matches NumPy
> output except where NumPy is numerically wrong, and in those cases ferray
> prefers mathematical correctness and documents the divergence. The
> conformance suite must honour this contract — it is not enough to simply
> assert bit-for-bit agreement. Documented, cited divergences are acceptable
> outcomes. Undocumented divergences are not.

---

## The problem conformance suites solve

The most common failure mode in a reimplementation project is "tests pass,
ship it" — where:

- Unit tests cover the parts of the code the author thought to test.
- Integration tests cover the workflows the author thought to exercise.
- Doctests show that the example in the docstring compiles and runs.
- The README says "NumPy parity for op X."

And then a downstream user runs `op X` with a real input and gets a result
that differs from numpy by 5 ULP, or silently exercises a scalar fallback
instead of a SIMD path that has a latent bug, or gets a wrong answer on an
edge case numpy handles cleanly. The author's tests did not catch it because
the author was checking *internal correctness*, not *contract fidelity*.

Conformance suites close this gap by encoding the contract — "this op behaves
the same as numpy for the same inputs" — as a mechanically checkable
assertion. numpy 2.4.4 is the ground truth; the suite proves the
implementation matches (or documents why it intentionally diverges).

The pattern is also the only mechanism that can guarantee **total coverage**:
every public function, not just the ones the author remembered to test.
Without a coverage gate, drift is silent and cumulative.

---

## What a conformance suite *is not*

| Pattern | What it proves | Conformance is different because |
|---|---|---|
| **Unit tests** | This function returns what its body computes | Conformance proves the function matches an *external* spec, not its own implementation |
| **Integration tests** | These subsystems compose correctly | Conformance is per-public-item; integration is workflow-level |
| **Property-based tests** | Algebraic invariants hold (`add(x, 0) == x`) | Conformance can use property tests, but the assertion is "matches reference" not "satisfies invariant" |
| **Doctests** | The documented example compiles and runs | Doctests are author-authored; conformance is reference-authored |
| **Snapshot tests** | Output matches a previous run's output | Conformance compares to an *external* reference, not a previous self-run |
| **Fuzzing** | No crashes on random inputs | Conformance is curated inputs with known reference outputs |
| **Benchmarks** | This is fast | Conformance is correctness-only |
| **SIMD scalar-parity tests** | FERRAY_FORCE_SCALAR=1 and non-scalar paths agree | SIMD parity is a subset of conformance; conformance also asserts the scalar path itself is correct |

You can have all of these and still ship a "numpy parity for op X" claim that
is not true. Conformance is the assertion that closes that specific gap.

---

## Ferray-specific deltas from the ferrotorch pattern

ferray has a different architecture from ferrotorch. The following deltas
apply throughout this document.

### No GPU / CUDA / MPS lane

ferray is **CPU-only**. There is no GPU backend axis. SIMD dispatch (SSE2,
AVX2, AVX-512, NEON) is handled transparently via the `pulp` crate. The
tolerance table has one set of values per dtype category, not per backend.
Every ferrotorch reference to "CPU tolerance vs GPU tolerance" reduces to a
single tolerance column in ferray.

The `FERRAY_FORCE_SCALAR=1` environment variable disables SIMD and falls back
to scalar loops. Conformance tests must pass under both `FERRAY_FORCE_SCALAR=0`
(default, SIMD enabled) and `FERRAY_FORCE_SCALAR=1` (scalar fallback). This is
the ferray equivalent of ferrotorch's CPU/GPU parity check.

### No autograd backward lane

ferray-autodiff exists but uses forward-mode automatic differentiation (dual
numbers), not PyTorch-style reverse-mode autograd. For the Rust API
conformance suite (this rollout), ferray-autodiff conformance is:

- **Forward pass**: compare `ferray_autodiff::eval(f, x)` against numpy's
  `f(x)` — standard fixture comparison.
- **Derivative**: compare `ferray_autodiff::grad(f, x)` against numpy's
  analytic derivative where numpy provides one (e.g., `numpy.gradient`,
  or the analytic formula for standard functions). Where numpy has no
  analytic derivative, compare against the finite-difference approximation
  with tolerance scaled for the method order.

There is no backward-pass conformance lane and no gradient tape, because
ferray-autodiff does not have a gradient tape.

### Reference is numpy 2.4.4, not PyTorch

All fixtures are generated by calling `numpy 2.4.4` functions. The version is
pinned in both the fixture metadata (see fixture schema section) and in the
`scripts/generate_fixtures.py` script. When numpy releases 2.5.x, fixtures
must be regenerated and reviewed before the pin is bumped.

### Two-surface architecture: Rust API now, Python wheel deferred

ferray exposes two API surfaces:

1. **Rust API** (`ferray::*`, `ferray_core::*`, etc.) — the crate-level public
   interface consumed by downstream Rust code. This is what the current
   rollout (issues #748 et seq.) covers.

2. **Python wheel** (`ferray-python` crate, `import ferray as np`) — the PyO3
   binding layer that translates Python calls to Rust. This is a separate
   conformance surface.

The Python wheel conformance suite is **deferred** to a future rollout. The
reason: `ferray-python` is currently pre-publication (its `Cargo.toml` is
commented out of the workspace manifest). Authoring a Python-surface
conformance suite before the wheel has a stable public API would require
either (a) re-running the suite every time the wheel API stabilises, or
(b) treating `ferray-python` as stable when it is not. Both paths waste effort.
The correct time to add Python-surface conformance is when `ferray-python` is
declared stable and added back to the workspace as a published crate.

The Python wheel conformance suite will have a different structure: it will
drive `import ferray as np` against `import numpy as np` in a Python test
harness, exercising the Python-level API rather than the Rust-level API. That
design is out of scope for this document.

### Divergence policy: correctness > parity

This is the most important ferray-specific delta.

ferray's contract with its users is: "we match numpy except where numpy is
numerically wrong." When a conformance test reveals that ferray and numpy
disagree, the disagreement must be classified into one of five outcomes (see
Classification Taxonomy below). The `DIVERGENCE` outcome — ferray is more
correct than numpy — is an acceptable first-class result, not a test failure.

Each accepted divergence is documented in a `_divergences.toml` file in the
relevant crate's conformance directory. The format is specified below. Adding
a divergence entry without a numerical citation from a published source is
not acceptable; vague claims of "mathematical correctness" are not enough.

---

## The four-layer architecture

A ferray conformance suite has four mechanical layers. They are independent
enough to build in any order, but the dependencies flow downward:

```
+----------------------------------+
|  Layer 4 — Strict coverage gate  |   ← CI fails if any pub item lacks
|                                  |     a test reference or exclusion entry
+----------------------------------+
              ↑ refers to
+----------------------------------+
|  Layer 3 — Conformance tests     |   ← Per-op test functions that
|                                  |     load fixtures and assert
+----------------------------------+
              ↑ loads
+----------------------------------+
|  Layer 2 — Reference fixtures    |   ← JSON files committed to the repo,
|                                  |     generated by generate_fixtures.py
|                                  |     which calls numpy 2.4.4
+----------------------------------+
              ↑ derives from
+----------------------------------+
|  Layer 1 — Surface inventory     |   ← List of every `pub` item the
|                                  |     crate exposes; syn 2 based
+----------------------------------+
```

### Layer 1 — Surface inventory

A list of every public item the crate exposes. The "denominator" for coverage.
Built by a `syn 2`-based parser (the stable toolchain choice, added as a leaf
crate dependency in `ferray-test-oracle`) that walks all source files and
collects every `pub fn`, `pub struct`, `pub trait`, and `pub method` with
module path and signature.

The inventory is committed to the repo at
`<crate>/tests/conformance/_surface.json`. PRs that change the public surface
produce a clean JSON diff that the reviewer can check against the test
coverage diff.

```jsonc
// Example: ferray-window/tests/conformance/_surface.json
{
  "items": [
    {
      "path": "ferray_window::bartlett",
      "kind": "fn",
      "signature": "fn bartlett(m: usize) -> FerrayResult<Array1<f64>>"
    },
    {
      "path": "ferray_window::kaiser",
      "kind": "fn",
      "signature": "fn kaiser(m: usize, beta: f64) -> FerrayResult<Array1<f64>>"
    },
    {
      "path": "ferray_window::taylor",
      "kind": "fn",
      "signature": "fn taylor(m: usize, nbar: usize, sll: f64, norm: bool) -> FerrayResult<Array1<f64>>"
    }
  ]
}
```

### Layer 2 — Reference fixtures

Generated offline by `scripts/generate_fixtures.py`, which imports
`numpy 2.4.4` and records `(input, expected_output)` pairs for each op.
Curated input sets per op cover normal cases plus edge cases (empty input,
NaN/Inf boundaries, broadcast shapes, scalar vs array, etc.).

Fixtures are committed under `fixtures/<subcrate>/<op>.json`. CI never
invokes numpy or Python; the JSON files are the source of truth.

#### Fixture metadata schema (version 2)

Existing fixtures (schema version 1) lack a `numpy_version` field and a
`fixture_schema_version` field. Stage 2 of the conformance rollout will
retrofit all ~80 existing fixtures with these fields. No value in any
`test_cases` entry changes; only the top-level metadata is added.

The version 2 schema is:

```json
{
  "numpy_version": "2.4.4",
  "fixture_schema_version": 2,
  "function": "numpy.fft.rfft",
  "ferray_function": "ferray_fft::rfft",
  "test_cases": [
    {
      "name": "real_8",
      "inputs": {
        "x": {
          "data": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
          "shape": [8],
          "dtype": "float64"
        }
      },
      "expected": {
        "data": [{"re": 0.0, "im": 0.0}, ...],
        "shape": [5],
        "dtype": "complex128"
      },
      "tolerance_ulps": 10
    }
  ]
}
```

The `numpy_version` and `fixture_schema_version` fields serve two purposes:
(1) they make it mechanically detectable when fixtures were generated under a
different numpy version, and (2) they allow `ferray-test-oracle` to warn when
a fixture was generated before the currently pinned numpy version.

Stage 2 adds a migration step to `generate_fixtures.py` that injects
`"numpy_version": "2.4.4"` and `"fixture_schema_version": 2` into every
existing fixture file without touching any `test_cases` content.

#### Fixture generation example

```python
# scripts/generate_fixtures.py (excerpt)
import numpy as np

# Pin is enforced at the top of the script:
assert np.__version__ == "2.4.4", f"Expected numpy 2.4.4, got {np.__version__}"

def make_fixture(np_func_name, ferray_func_name, test_cases):
    return {
        "numpy_version": "2.4.4",
        "fixture_schema_version": 2,
        "function": np_func_name,
        "ferray_function": ferray_func_name,
        "test_cases": test_cases,
    }
```

### Layer 3 — Conformance tests

Rust integration tests that load fixtures and assert. Tolerance is determined
by the per-op-category table (see Tolerance Table below). Tests live at
`<crate>/tests/conformance_<category>.rs`, mirroring the
`fixtures/<crate>/<category>.json` organization.

```rust
// ferray-window/tests/conformance_windows.rs
use ferray_test_oracle::{fixtures_dir, run_unary_f64_oracle};

#[test]
fn taylor_matches_numpy() {
    let path = fixtures_dir().join("window/taylor.json");
    run_unary_f64_oracle(&path, |arr| {
        // extract (m, nbar, sll, norm) from arr shape / metadata...
        ferray_window::taylor(m, nbar, sll, norm)
            .map(|w| w.into_dyn())
    });
}
```

The `ferray-test-oracle` crate (at `ferray-test-oracle/src/lib.rs`) provides:

- `load_fixture` / `fixtures_dir` — fixture loading with workspace-root
  discovery.
- `run_unary_f64_oracle`, `run_unary_f32_oracle`, `run_binary_f64_oracle`,
  `run_real_to_complex_oracle`, `run_complex_to_complex_oracle`,
  `run_reduction_f64_oracle`, `run_matrix_scalar_oracle` — generic test
  runners that iterate over test cases, construct ferray arrays from fixture
  data, call the function under test, and compare to the expected output using
  ULP comparison.
- `assert_f64_ulp`, `assert_f32_ulp`, `assert_f64_slice_ulp`,
  `assert_f32_slice_ulp`, `assert_complex_slice_ulp` — tolerance-aware
  comparison primitives.
- `MIN_ULP_TOLERANCE = 10` — the workspace-wide floor; fixture-level
  `tolerance_ulps` fields override upward where per-op envelopes are wider.

Conformance tests run under both default (SIMD-enabled) and
`FERRAY_FORCE_SCALAR=1` conditions in CI to verify that SIMD and scalar paths
agree with the numpy reference.

### Layer 4 — Strict coverage gate

A test file at `<crate>/tests/conformance_surface_coverage.rs` that
cross-references the surface inventory against the conformance test files and
the exclusion/divergence registries. The gate fails if any public item lacks a
test reference, an exclusion entry, or a divergence entry.

```rust
// ferray-window/tests/conformance_surface_coverage.rs
#[test]
fn every_public_item_has_a_conformance_reference() {
    let surface: Vec<Item> = load_surface_json();           // _surface.json
    let exclusions: Vec<Exclusion> = load_exclusions_toml(); // _surface_exclusions.toml
    let divergences: Vec<Divergence> = load_divergences_toml(); // _divergences.toml
    let test_text = read_all_conformance_test_files();

    let uncovered: Vec<_> = surface.iter()
        .filter(|item| !test_text.contains(&item.path))
        .filter(|item| !exclusions.iter().any(|e| e.path == item.path))
        .filter(|item| !divergences.iter().any(|d| d.ferray_path == item.path))
        .collect();

    assert!(
        uncovered.is_empty(),
        "{} public items lack a conformance reference: {:?}",
        uncovered.len(),
        uncovered
    );
}
```

The gate is **strict from day 1**: every new `pub fn` added to the project
must be referenced in a conformance test, or added to `_surface_exclusions.toml`
with a tracking issue, or documented in `_divergences.toml` with a citation.
Without this strictness, coverage drifts invisibly.

---

## File layout per crate

Each workspace crate that participates in the conformance suite has this
directory structure under its `tests/` directory:

```
<crate>/
  tests/
    conformance/
      _surface.json                  ← generated surface inventory (committed)
      _surface_exclusions.toml       ← manual exclusions with cited reasons
      _divergences.toml              ← documented divergences from numpy 2.4.4
    conformance_<category>.rs        ← per-op-category conformance tests
                                       (mirrors fixtures/<crate>/<category>.json)
    conformance_surface_coverage.rs  ← strict coverage gate
```

Conformance test `.rs` files MUST be at the `tests/` root with the
`conformance_` prefix (e.g., `tests/conformance_windows.rs`), NOT inside a
`tests/conformance/` subdirectory — cargo does not auto-compile files in
subdirectories of `tests/` as integration test binaries. Data files
(`_surface.json`, `_surface_exclusions.toml`, `_divergences.toml`) DO live
under `tests/conformance/` as a subdir; only the test binaries are flat.

The `<category>` names mirror the existing fixture organization:
`core`, `fft`, `linalg`, `io`, `ma`, `polynomial`, `random`, `stats`,
`strings`, `ufunc`, `window`, etc.

### `_surface_exclusions.toml` format

```toml
# Each [[exclusion]] block documents one public item that legitimately
# cannot be tested with a numpy reference fixture.

[[exclusion]]
path = "ferray_core::FerrayError"
kind = "struct"
reason = "Error type; has no numpy equivalent. Behaviour is tested via unit tests on error-returning functions."
tracking_issue = 748

[[exclusion]]
path = "ferray_core::Element"
kind = "trait"
reason = "Marker trait with no callable API surface. Covered implicitly by every numeric test."
tracking_issue = 748
```

Each exclusion must have:
- `path` — the fully qualified Rust path.
- `kind` — `fn`, `struct`, `trait`, `method`, or `type`.
- `reason` — a specific sentence explaining why no numpy fixture is applicable.
- `tracking_issue` — the issue number under which the exclusion was reviewed.

Generic reasons ("tested somewhere") are not acceptable.

### `_divergences.toml` format

scipy.signal.windows.taylor is the actual reference for the seed divergence example; an earlier draft used "numpy_function" which was misleading for non-numpy references.

```toml
# Each [[divergence]] block documents one case where ferray intentionally
# returns a more mathematically correct value than numpy 2.4.4.
#
# A divergence is NOT a bug. It is an explicit, cited policy decision that
# ferray's correctness > parity contract permits.

[[divergence]]
ferray_path = "ferray_window::taylor"
reference_function = "scipy.signal.windows.taylor"
reference_library = "scipy.signal"
reference_version = "1.13.0"
ferray_version_introduced = "0.3.7"
tracking_issue = 810

# Concrete example: taylor(M=16, nbar=4, sll=30.0, norm=True)
example_input = "m=16, nbar=4, sll=30.0, norm=true"
numpy_output  = "centre normalisation via secant midpoint of adjacent samples"
ferray_output = "centre normalisation via analytic peak W((M-1)/2) = 1 + 2·Σ F_m"
max_observed_delta = 1.5e-3

justification = """
For even M, scipy.signal.windows.taylor normalises the window by dividing by
the average of the two centre samples w[M/2 - 1] and w[M/2]. These samples
lie at ±0.5 / M in normalised frequency, not at the origin, so the average is
the secant approximation to the window's peak value, not the peak itself. The
Taylor window's cosine-sum formula W(xn) = 1 + 2·Σ_{m=1}^{nbar-1} F_m cos(π m xn)
evaluates exactly to 1 + 2·Σ F_m at xn = 0 (every cosine term collapses to 1).
ferray computes this closed-form analytic peak and divides by it, which is the
mathematically correct normalisation. The secant midpoint is an O((Δxn)²)
approximation that is exact only when the cosine sum is linear between
samples, which it is not for nbar ≥ 2. The error is correlated with nbar and
sll; for (M=16, nbar=4, sll=30) the residual is ~1.5e-3. Reference:
Carrara, Goodman, Majewski, "Spotlight Synthetic Aperture Radar", Artech
House 1995, ch. 10; scipy source code signals/windows.py `taylor()` function
confirmed with scipy 1.13.0 (uses secant midpoint for even M). ferray
validates to machine precision (max abs diff ≤ 5e-13) against scipy on
M ∈ {4, 8, 16, 32}, nbar ∈ {2, 4, 6}, sll ∈ {30, 50, 100} when scipy is
run with the analytic-peak normalisation patch applied.
"""

citation = "Carrara et al. 1995 ch.10; scipy/signal/windows.py confirmed scipy 1.13.0"
```

Each divergence entry requires:
- `ferray_path` — the ferray function that diverges.
- `reference_function` — the numpy / scipy function being compared.
- `reference_library` — one of "numpy", "scipy.signal", "scipy.special", etc. — disambiguates which reference library the divergence is measured against.
- `reference_version` — the reference library version under which the divergence was observed.
- `ferray_version_introduced` — the ferray release where ferray took the
  more-correct path (not the release where the divergence was documented).
- `tracking_issue` — the issue number where the divergence is recorded for
  posterity. This issue remains open indefinitely as a reference anchor.
- `example_input` — a concrete input that demonstrates the divergence.
- `numpy_output` / `ferray_output` — what each library returns for that input.
- `max_observed_delta` — the largest observed absolute difference.
- `justification` — a paragraph citing a published paper, libm source, GNU
  MPFR result, or equivalent. "Ferray is more correct" without citation is
  not acceptable.
- `citation` — a short reference string.

---

## Tolerance table

ferray has **no GPU backend**. The tolerance table has one set of values (f64
and f32) without a backend axis. Tolerances reflect CPU arithmetic under IEEE
754 with pure-Rust implementations (ndarray, faer, rustfft, pulp SIMD).

| Category | f64 tolerance | f32 tolerance | Reason |
|---|---|---|---|
| Indexing / slicing / shape ops | bit-exact | bit-exact | No arithmetic |
| Elementwise arithmetic | 1 ULP | 1 ULP | IEEE 754 deterministic on CPU |
| Elementwise transcendentals (sin/cos/exp/log) | 1e-12 rel | 1e-5 rel | libm approx-instruction looser on f32 |
| Reductions (sum/mean/std) | 1e-12 abs | 1e-6 abs | Tree-order reduction drift |
| Matmul / linalg | 1e-10 rel | 1e-4 rel | O(n³) error amplification |
| FFT | 1e-10 rel | 1e-5 rel | Bit-reversal + butterfly accumulation |
| Polynomial evaluation | 1e-10 rel | 1e-5 rel | Horner / Clenshaw accumulation |
| RNG | distribution moments only | distribution moments only | Different bit generators |
| Complex (c64, c128) | same as underlying f32/f64 | same as underlying f32/f64 | Real and imaginary tracked separately |
| Integer ops | bit-exact | bit-exact | No FP |

These values apply when the fixture's `tolerance_ulps` field is at or below
`MIN_ULP_TOLERANCE = 10`. For relative tolerances, the ferray-test-oracle
library's `assert_f64_ulp` / `assert_f32_ulp` functions enforce the ULP
comparison; per-op fixtures may set a higher `tolerance_ulps` value where
the inherent error envelope of the algorithm requires it (e.g., large
matmul, deep FFT).

The tolerance table lives in a shared helper
`tests/conformance/_tolerance.rs` imported by all conformance test files in
a crate.

**Escalation rule**: if a conformance test fails and the correct response
appears to be raising the tolerance, do not raise the tolerance unilaterally.
Escalate to the architect via the tracking issue. Tolerance increases are
architectural decisions, not local fixes.

---

## Per-finding classification taxonomy

When a Stage 4 sweep conformance test reveals a disagreement between ferray
and numpy 2.4.4, the result must be classified as one of five outcomes:

### `MATCH`
ferray's output is within the tolerance for this op category.
No action required. The test passes.

### `BUG`
ferray's output is outside tolerance and the deviation is not explainable by
a known mathematical advantage of ferray's algorithm. ferray is wrong.

Required actions:
1. File a tracking issue with: the function name, the failing input, ferray's
   output, numpy 2.4.4's output, and the observed delta.
2. Add `cascade_skip(#N)` to the test (where `N` is the issue number), so the
   suite can be completed without the failing test blocking progress.
3. Do NOT fix the bug inline. The Stage 4 sweep is audit work; bug fixes are
   a separate dispatch.

### `DIVERGENCE`
ferray's output is outside tolerance, but ferray is more mathematically
correct than numpy 2.4.4. This is an acceptable outcome under ferray's
correctness > parity contract.

Required actions:
1. Add an entry to `_divergences.toml` with all required fields (see format
   above), including a published citation for the numerical justification.
2. The conformance test asserts the divergence: it verifies that ferray
   produces the more-correct value, and contains a comment citing the
   `_divergences.toml` entry.
3. File (or reference an existing) tracking issue for posterity. The issue
   stays open as a reference anchor; closing it would lose the paper trail.

### `TOLERANCE_GAP`
The tolerance table is wrong for this op family: the test fails even though
ferray is behaving correctly per IEEE 754 and the algorithm's theoretical
error bound. Neither ferray nor the fixture is wrong; the tolerance row in
the table is too tight.

Required action: escalate to the architect by commenting on the issue.
Do not unilaterally raise the tolerance. A tolerance change affects every
test in that category and must be reviewed as an architectural decision.

### `FIXTURE_BUG`
The fixture itself is malformed: wrong dtype tag, serialisation error,
truncated data, or was generated under a numpy version other than 2.4.4.

Required actions:
1. Identify the malformed fixture file.
2. Regenerate it via `python3 scripts/generate_fixtures.py` (with
   numpy==2.4.4 active in the environment).
3. Commit the regenerated fixture and note the regeneration in the PR
   description.

---

## What subagents must NOT do during Stage 4 sweeps

Stage 4 dispatches subagents to sweep one crate at a time (serial order, one
crate per dispatch). Each subagent is responsible for authoring the
conformance files for exactly the crate named in its dispatch. The following
actions are forbidden:

- **Must not fix bugs they find.** The sweep is audit work. File the issue,
  add `cascade_skip(#N)`, and continue. Bug fixing is a different work mode.

- **Must not raise tolerances.** If a tolerance miss is not a bug and not a
  known divergence, escalate to the architect. Do not increase any value in
  the tolerance table or any fixture's `tolerance_ulps` field.

- **Must not add workspace-level dependencies.** Adding a dep to
  `Cargo.toml` at the workspace root affects all 16 crates. Subagents work
  in leaf-crate scope; workspace changes require architect sign-off.

- **Must not modify `CHANGELOG.md`.** The CHANGELOG is maintained at
  release time, not during sweep work.

- **Must not touch crates other than the one named in their dispatch.**
  Even if a bug found in crate A is clearly caused by code in crate B, the
  subagent files the issue and skips the test; it does not edit crate B.

- **Must not add exclusions without a cited reason and a tracking issue
  number.** Generic exclusions ("complex to test", "not sure") are not
  acceptable. Every exclusion entry must cite a specific reason and a
  specific issue number.

- **Must not classify a disagreement as `DIVERGENCE` without a published
  citation.** The divergence policy exists to honour mathematical
  correctness, not to paper over bugs. If the subagent cannot find a
  published citation, it files a `BUG`.

---

## What programs benefit

Not every project earns its keep with a conformance suite. The heuristic:
**does your project make a behavioral-parity claim?**

### Strong fit

- **Reimplementations of an existing library.** ferray (NumPy), ferrotorch
  (PyTorch), polars-rust (pandas), arrow-rs (Arrow C++). The reference
  library is the ground truth; conformance proves you match.

- **Cross-language ports.** A Python library reimplemented in Rust; a
  C++ library reimplemented in Go. The original is the spec.

- **Compatibility shims.** A library that claims "drop-in replacement for X"
  — every public item must behave identically, or divergences must be
  documented. ferray is the canonical example of this category: its pitch is
  `import ferray as np` as a drop-in for `import numpy as np`.

- **Numerical / scientific libraries.** Anything claiming "matches numpy" or
  "matches scipy" or "matches MATLAB." Numerical drift is invisible to unit
  tests; conformance catches it. ferray is a strong fit.

- **Standards implementations.** A library implementing a published spec
  (HTTP/3, MessagePack, JSON Schema). The spec's reference implementation
  is the ground truth.

### Weak fit

- **Greenfield libraries with no reference.** Nothing to be conformant
  against. Property-based testing with algebraic invariants serves better.

- **UI / interactive applications.** Behavior is human-perceived; hard to
  encode as fixture-vs-output assertion.

- **Pre-1.0 libraries with rapidly-evolving APIs.** The suite invests in the
  public surface; if the surface is in flux, the investment evaporates.

### Mixed fit

- **DSLs and macro-heavy code.** The "public surface" is the macros
  themselves; conformance applies to what the macros expand to.

- **Compilers.** Conformance against a published language spec works (C, Rust);
  against a competing compiler is fragile (their bugs are not your spec).

---

## What conformance suites produce

The visible output is "tests pass." The invisible output is the
**bug-finding rate**.

ferrotorch's conformance work surfaced ~30 latent bugs in code that already
had unit tests, doctests, and "PyTorch parity" claims. None would have
surfaced through any other testing pattern.

For ferray, the expected bug shapes are different because ferray is CPU-only
(no silent GPU detour pattern), but analogous failure modes exist:

1. **Silent SIMD fallback to wrong path**: a SIMD inner loop has a latent
   bug that the scalar path does not. `FERRAY_FORCE_SCALAR=1` passes; the
   default SIMD-enabled path fails. Unit tests miss this because they test
   the function, not the dispatch path.

2. **Dtype routing asymmetry**: the f64 path is correct; the f32 path calls
   a different internal function that has a latent bug. Unit tests cover the
   f64 path; conformance catches the f32 path because it exercises both.

3. **Undocumented edge cases**: a reduction on an empty array, a window
   function with M=0 or M=1, FFT of length 1, a matmul with a degenerate
   matrix. Unit tests cover the happy path; conformance covers the same
   edges numpy covers.

4. **Stub residue**: a public function that returns
   `Err(FerrayError::NotImplemented(...))` despite the documentation claiming
   it works. Conformance catches this because the test asserts a valid output,
   not an error.

5. **Numerical drift**: a polynomial approximation or SIMD reduction
   accumulates error beyond the tolerance for its category. Unit tests using
   self-comparison miss it; conformance catches it against numpy 2.4.4.

---

## How to set one up

### Step 0 — Pin the reference and verify

The reference is **numpy 2.4.4**. Before generating any fixtures, verify the
numpy version in the active environment:

```bash
python3 -c "import numpy as np; assert np.__version__ == '2.4.4', np.__version__"
```

The `scripts/generate_fixtures.py` script enforces this assertion at the top
of the file. Do not generate fixtures under any other version.

### Step 1 — Build the surface inventory tool

Add a binary in `ferray-test-oracle` that uses `syn 2` to walk a target
crate's `src/` directory and emit `tests/conformance/_surface.json`. The
tool:

- Uses a stable toolchain (not nightly `rustdoc --output-format json`).
- Collects `pub fn`, `pub struct`, `pub trait`, `pub enum`, and inherent
  `pub` methods.
- Emits a deterministic, sorted JSON file.
- Is run as part of a CI step that commits the result; diff is reviewed with
  the PR.

Commit the initial `_surface.json` for each crate. Subsequent changes to the
public surface produce a clean diff.

### Step 2 — Decide tolerance categories and write `_tolerance.rs`

The tolerance table (above) is the authoritative source. Write a shared
`tests/conformance/_tolerance.rs` in each participating crate that exposes
the per-category tolerance constants used by all conformance tests in that
crate.

### Step 3 — Proof-of-pattern crate: ferray-window

The Stage 3 proof-of-pattern crate is `ferray-window`. It is a good choice
because:

- It has a bounded public surface (~15 window functions).
- It has existing fixtures under `fixtures/` (if not, they are easy to add
  via `generate_fixtures.py`).
- It has the canonical divergence example (taylor, see `_divergences.toml`
  format above), which proves the divergence machinery works end to end.
- Its functions are pure (no I/O, no RNG), making the test harness simple.

Build all four layers for `ferray-window`, get them green under both
`FERRAY_FORCE_SCALAR=0` and `FERRAY_FORCE_SCALAR=1`. The result is the
template for all subsequent crates.

### Step 4 — Serial sweep of remaining crates

Stage 4 sweeps the remaining workspace crates one at a time (serial, not
parallel). Serial order avoids conflicts on shared files such as the
`fixtures/` directory and the workspace `Cargo.toml`. The order is:

1. ferray-core (shape ops, array creation — mostly bit-exact)
2. ferray-ufunc (elementwise transcendentals — highest bug-finding rate expected)
3. ferray-fft (complex arithmetic — known tolerance envelope)
4. ferray-linalg (matmul, solvers — widest tolerance, most complex dtype routing)
5. ferray-stats, ferray-polynomial, ferray-random (reductions, RNG, Horner)
6. ferray-ma, ferray-strings, ferray-io, ferray-stride-tricks (mixed)
7. ferray-autodiff, ferray-numpy-interop (forward-mode AD, interop layer)

Each stage-4 dispatch receives: the crate name, a reference to this
document, a reference to the Stage 3 proof-of-pattern output, and the
classification taxonomy above. The dispatch explicitly lists the forbidden
actions (see "What subagents must NOT do" above).

### Step 5 — Triage the bug cascade

Stage 4 will surface bugs. Do not fix them inline — that is a different work
mode. Each finding gets:

- A `cascade_skip(#N)` annotation in the test.
- A tracking issue with the input, ferray output, numpy output, and delta.

When the sweep is complete, the tracking issues form a structured queue of
real bugs to fix. Each fix dispatch flips a skip back to a live assertion.

### Step 6 — Make the gate strict in CI

Add `conformance_surface_coverage` to the CI test matrix for each crate.
Once it is in CI, no `pub fn` can be added without either a conformance test
reference or an exclusion entry with a tracking issue. The cost of keeping
coverage current is paid at PR time, not at "we should write more tests" time.

---

## Common objections

**"This is just unit testing with extra steps."**

No — unit testing proves the implementation matches its own logic. Conformance
proves the implementation matches numpy 2.4.4. They test orthogonal claims.

**"Tolerance comparisons are fragile."**

In two ways. (1) Tolerances drift across numpy versions. Mitigation: pin to
numpy 2.4.4, record the pin in fixture metadata (schema version 2), re-run
when bumping. (2) The tolerance table may be wrong for an op family —
escalate to the architect rather than widening the tolerance locally.

**"The fixture files are large."**

Each fixture is small but they accumulate. ferray currently has ~80 fixture
files. Trade-off: repo size vs running Python in CI. The consensus is to
commit fixtures and keep Python out of CI. Git LFS is an option if the
corpus grows past ~100 MB.

**"What about RNG ops? They can't be bit-exact."**

Compare distribution moments (mean, variance, range, histogram shape) instead
of raw values. The assertion is "samples have the same distribution," not
"samples have the same bits." ferray uses different bit generators than numpy,
so the raw values are expected to differ; only the statistical properties must
match.

**"What about ferray-autodiff? It uses dual numbers."**

The conformance assertion for ferray-autodiff is forward-pass comparison
against numpy's `f(x)`, plus derivative comparison against either numpy's
analytic derivative (where one exists) or a finite-difference approximation.
There is no backward-pass lane because ferray-autodiff does not have a
backward pass. This is explicitly not a gap — it is a consequence of the
forward-mode design.

**"Can't I just run numpy in CI?"**

You can, but: (a) Python-in-Rust-CI doubles build complexity and CI time,
(b) numpy updates silently change test outcomes if the version is not pinned,
(c) the fixture files are the auditable evidence that the test compared
against numpy 2.4.4, not "whatever numpy was installed when CI ran."
Committed fixtures avoid all three problems.

**"numpy 2.4.4 has bugs; should we match them?"**

No. ferray's contract is correctness > parity. When numpy is wrong, ferray
is correct and the disagreement is documented in `_divergences.toml` with a
citation. When ferray is wrong, ferray files a bug. The classification
taxonomy distinguishes these cases mechanically.

---

## Anti-patterns to avoid

- **Generic exclusion entries.** "Tested by something somewhere" is not a
  reason. Each exclusion needs a specific path, a specific reason, and a
  tracking issue number.

- **Tolerance weakening to silence a failing test.** The right response to
  a tolerance miss is: classify as `BUG`, `DIVERGENCE`, `TOLERANCE_GAP`, or
  `FIXTURE_BUG`; apply the action for that classification; do not raise the
  tolerance value.

- **Skipping the strict gate "until coverage is up."** The gate is what makes
  coverage stay up; without it, every PR adds a tiny bit of drift.

- **Phantom tests for type aliases / re-exports / marker traits.** These are
  usually covered transitively. Use exclusion entries with "implicit coverage"
  reasons rather than authoring tests that do not prove anything.

- **Same-library snapshot tests dressed as conformance.** If you are comparing
  ferray's output to its own previous output, that is a snapshot test, not
  conformance. Conformance requires an external reference (numpy 2.4.4).

- **Undocumented divergences.** If ferray disagrees with numpy 2.4.4 and
  the test passes because the tolerance is wide enough to cover the gap,
  that is a hidden divergence. Every divergence must be classified and
  either documented in `_divergences.toml` (if ferray is more correct) or
  filed as a bug (if ferray is wrong).

- **Generating fixtures under the wrong numpy version.** If a fixture is
  generated under numpy 2.3.x or 2.5.x, it is not a conformance fixture
  against numpy 2.4.4. The `numpy_version` field in schema version 2 makes
  this detectable.

---

## Maintenance over time

Once the suite is in place:

1. **numpy version bumps** (e.g., 2.4.4 → 2.5.0): re-run
   `scripts/generate_fixtures.py`, commit new fixtures, run the suite. New
   failures are either numpy changes ferray should track (file issue) or
   numpy bugs ferray already fixed (update `_divergences.toml` accordingly).

2. **New public items**: the strict gate catches them at PR time. PR authors
   must add a conformance test reference (or an exclusion, or a divergence
   entry) with their PR. The gate fails if they do not.

3. **Bug fixes**: when a tracked cascade bug is fixed, remove the
   `cascade_skip(#N)` and verify the test now passes under both SIMD-enabled
   and `FERRAY_FORCE_SCALAR=1` modes.

4. **Tolerance reviews**: if a tolerance is consistently passing with large
   margin or consistently failing without bugs, the tolerance row may be
   wrong. Audit the table periodically; changes require architect sign-off.

5. **Fixture audits**: fixtures grow over time. Periodically prune redundant
   cases (duplicate shapes, duplicate dtypes) to keep the corpus lean.

6. **Divergence reviews**: when the upstream numpy bug that caused a
   documented divergence is fixed in a new numpy release, the divergence
   may no longer apply. Review `_divergences.toml` entries at numpy version
   bumps and update or retire entries as appropriate.

The suite is a living artifact — not a one-time investment.

---

## When NOT to use this pattern

- **Throwaway code.** The investment is not recouped.

- **Code where "correct" is defined by humans, not a reference.** E.g., UI
  rendering, linting, autocompletion ranking. No external spec to be
  conformant against.

- **Code where the reference is itself the system under test.** No oracle.

- **Pre-1.0 libraries with rapidly-evolving APIs.** The suite invests in the
  public surface; if the surface is in flux, the investment evaporates.

For these cases, property-based testing (`proptest`), snapshot testing, or
human-curated regression tests serve better.

---

## Summary

Conformance suites prove that what ferray claims to do matches what
numpy 2.4.4 actually does — and, where ferray intentionally diverges for
mathematical correctness reasons, that the divergence is cited, documented,
and reviewed.

The four-layer architecture — surface inventory → reference fixtures →
conformance tests → strict coverage gate — is mechanical enough to build
incrementally across the 16-crate ferray workspace. The strict gate from day
1 is the lock-in: once it is in CI, coverage cannot drift.

**Ferray-specific departures from the generic pattern:**

- No GPU lane, no CUDA, no MPS, no backend axis in the tolerance table.
  ferray is CPU-only; SIMD dispatch is verified via `FERRAY_FORCE_SCALAR=1`.
- No autograd backward lane. ferray-autodiff is forward-mode; the
  conformance lane tests `eval(f, x)` and `grad(f, x)` against numpy
  analytic derivatives.
- Reference is numpy 2.4.4 (pinned), not PyTorch.
- Two-surface architecture: Rust API (this rollout) and Python wheel
  (deferred until ferray-python is stable and published).
- Divergence policy: `_divergences.toml` with published citations is the
  mechanism that turns "ferray is more correct than numpy" from a vague
  claim into auditable evidence.
- Classification taxonomy (`MATCH`, `BUG`, `DIVERGENCE`, `TOLERANCE_GAP`,
  `FIXTURE_BUG`) is the protocol that Stage 4 subagents follow when a
  conformance test surfaces a disagreement.

Build it once, maintain it incrementally, file the bugs it surfaces, fix them
as separate work. The bugs are the value.
