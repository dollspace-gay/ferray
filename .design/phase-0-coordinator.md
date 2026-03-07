# Feature: Phase 0 — Build Coordinator

## Summary
A single orchestration document that, when kicked off, autonomously builds the entire ferray library across five phases by spawning, monitoring, and merging the work of ~27 subagents. The coordinator reads the per-crate design docs in `.design/`, spawns agents in dependency-graph order, gates phase transitions on acceptance criteria, recovers stuck agents, and merges worktrees into a coherent codebase. The human's role reduces to approving phase gates and resolving any issues the coordinator escalates.

## Requirements

### Orchestration Core
- REQ-1: The coordinator reads `.design/ferray-*.md` and `.design/phase-{4,5}-*.md` as its authoritative task definitions — it does not invent work beyond what the design docs specify
- REQ-2: Agents are spawned using the `Agent` tool with `isolation: "worktree"` so each works on an isolated copy of the repo
- REQ-3: The coordinator maintains the dependency DAG (documented below) and never spawns an agent before its dependencies have been merged to the integration branch
- REQ-4: Phase transitions (1→2, 2→3, 3→4, 4→5) are gated on all acceptance criteria for the completing phase passing — verified by running `cargo test --workspace` and `cargo clippy --workspace` on the integration branch
- REQ-5: The coordinator uses crosslink issues to track every agent's assignment, status, and outcome
- REQ-6: When an agent completes, the coordinator merges its worktree branch into the `dev` integration branch, resolving conflicts if necessary
- REQ-7: When an agent is stuck (no progress for 3+ turns, or reports a blocker), the coordinator reads the agent's output, diagnoses the issue, and either provides guidance by resuming the agent or spawns a fixup agent
- REQ-8: After each phase completes, the coordinator runs the full test suite and records the result as a crosslink comment before proceeding
- REQ-9: The coordinator creates a `CLAUDE.md` project file at the start with conventions all subagents must follow
- REQ-10: On context compression, the coordinator recovers by re-reading crosslink issue state and the current phase design doc to reconstruct its position

### Agent Management
- REQ-11: Each spawned agent receives: (a) its design doc requirements, (b) file paths to create/modify, (c) acceptance criteria, (d) CLAUDE.md conventions, (e) instructions to commit and test before finishing
- REQ-12: Agents use `model: "sonnet"` for straightforward implementation and `model: "opus"` for architecturally complex tasks (see Model Selection table)
- REQ-13: Maximum 8 concurrent background agents
- REQ-14: Each agent's crosslink issue includes a `--kind result` comment documenting what was delivered

### Merge Strategy
- REQ-15: Integration branch `dev` is created from `main` at the start
- REQ-16: Merges happen sequentially with `cargo build --workspace` verification after each
- REQ-17: After all agents in a phase merge, full acceptance criteria check before next phase
- REQ-18: At the end of Phase 5, create a PR from `dev` to `main`

## Acceptance Criteria
- [ ] AC-1: Running `crosslink kickoff run "Build ferray" --doc .design/phase-0-coordinator.md` produces a working library with no further human intervention (beyond phase gate approvals)
- [ ] AC-2: All acceptance criteria across phases 1-5 pass on the final `dev` branch
- [ ] AC-3: Every agent's work is tracked via a crosslink issue with typed comments
- [ ] AC-4: No agent is left stuck for more than 10 minutes without coordinator intervention
- [ ] AC-5: `cargo test --workspace` passes after each phase gate
- [ ] AC-6: Context compression does not cause the coordinator to lose track of progress

## Architecture

### Dependency DAG

```
Phase 1: Foundation
  ┌─────────────────────────────────────────────────────┐
  │  Agent 1a: ferray-core-types (BLOCKING)             │
  │    → NdArray, Dimension, DType, Error, ownership    │
  │    → introspection, iterators, Display              │
  │    → Model: opus (architectural foundation)         │
  └──────────────────────┬──────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────────┐
  │ Agent 1b │   │ Agent 1c     │   │ Agent 1d     │
  │ core-    │   │ core-create  │   │ core-macros  │
  │ indexing │   │ + manipulate │   │ (sonnet)     │
  │ (opus)   │   │ (sonnet)     │   │              │
  └────┬─────┘   └──────┬───────┘   └──────┬───────┘
       └────────────────┼───────────────────┘
                        │ (merge core sub-agents)
         ┌──────────────┼───────────────┬───────────────┐
         ▼              ▼               ▼               ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
  │ Agent 2  │   │ Agent 3  │   │ Agent 4  │   │ Agent 5      │
  │ ufunc    │   │ stats    │   │ io       │   │ workspace +  │
  │ (opus)   │   │ (sonnet) │   │ (sonnet) │   │ reexport     │
  └────┬─────┘   └────┬─────┘   └────┬─────┘   │ (sonnet)     │
       │              │              │          └──────┬───────┘
       └──────────────┴──────────────┴─────────────────┘
                         │
                   PHASE 1 GATE
                   cargo test --workspace
                         │
Phase 2: Submodules      ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  ┌──────────┐
  │ Agent 6  │   │ Agent 7  │   │ Agent 8  │   │ Agent 9      │  │ Agent 9b │
  │ linalg   │   │ fft      │   │ random   │   │ polynomial   │  │ window   │
  │ (opus)   │   │ (sonnet) │   │ (sonnet) │   │ (sonnet)     │  │ (sonnet) │
  └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘  └────┬─────┘
       │              │              │                 │               │
       └──────────────┴──────────────┴─────────────────┴───────────────┘
                                     │
  ┌──────────────────────┐           │
  │ Agent 10: Phase 2    │◀──────────┘
  │ integration          │
  │ (opus)               │
  └──────────┬───────────┘
             │
       PHASE 2 GATE
             │
Phase 3: Completeness    ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
  │ Agent 11 │   │ Agent 12 │   │ Agent 13 │   │ Agent 14     │
  │ strings  │   │ ma       │   │ stride-  │   │ numpy-interop│
  │ (sonnet) │   │ (sonnet) │   │ tricks   │   │ (sonnet)     │
  │          │   │          │   │ (sonnet) │   │              │
  └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘
       │              │              │                 │
       └──────────────┴──────────────┴─────────────────┘
                         │
  ┌──────────────────────┤
  │ Agent 15: Phase 3    │
  │ integration          │
  │ (opus)               │
  └──────────┬───────────┘
             │
       PHASE 3 GATE
             │
Phase 4: Beyond NumPy    ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Agent 16 │   │ Agent 17 │   │ Agent 18 │
  │ f16      │   │ no_std   │   │ const    │
  │ support  │   │ core     │   │ generics │
  │ (sonnet) │   │ (sonnet) │   │ (opus)   │
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                PHASE 4 GATE
                      │
Phase 5: Verification  ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Agent 19 │   │ Agent 20 │   │ Agent 21 │   │ Agent 22 │
  │ fixture  │   │ oracle   │   │ property │   │ fuzz +   │
  │ gen      │   │ tests    │   │ tests    │   │ formal   │
  │ (sonnet) │   │ (sonnet) │   │ (opus)   │   │ (opus)   │
  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │              │
       │    (fixture gen must finish before oracle tests start)
       │              │              │              │
  ┌──────────────────────────────────────────────────────┐
  │ Agent 23: Verification integration + report (opus)   │
  └──────────────────────┬───────────────────────────────┘
                         │
                   PHASE 5 GATE
                   ALL verification passes
                         │
                    CREATE PR: dev → main
```

### Model Selection Per Agent

| Agent | Crate / Task | Model | Rationale |
|-------|-------------|-------|-----------|
| 1a | ferray-core-types | opus | Architectural foundation — NdArray, ownership, traits, iterators, Display |
| 1b | ferray-core-indexing | opus | Broadcasting + basic/advanced/extended indexing |
| 1c | ferray-core-creation-manipulation | sonnet | Array creation, shape manipulation, pad/tile/repeat, constants, finfo |
| 1d | ferray-core-macros | sonnet | FerrayRecord proc macro, s![] macro, promoted_type! macro |
| 2 | ferray-ufunc | opus | SIMD dispatch via pulp, 100+ function implementations, cumsum/diff/convolve/interp |
| 3 | ferray-stats | sonnet | Standard reduction algorithms, cumulative ops, well-defined math |
| 4 | ferray-io | sonnet | Binary format parsing, straightforward I/O |
| 5 | workspace + ferray reexport | sonnet | Cargo.toml scaffolding, re-exports, constants, pool cache |
| 6 | ferray-linalg | opus | einsum parser, faer bridge, batched operations, multi_dot, matrix_power |
| 7 | ferray-fft | sonnet | Wraps rustfft, plan caching |
| 8 | ferray-random | sonnet | Well-documented distribution implementations |
| 9 | ferray-polynomial | sonnet | Standard polynomial arithmetic |
| 9b | ferray-window | sonnet | Window functions + vectorize/piecewise/apply_along_axis |
| 10 | Phase 2 integration | opus | Cross-crate interface resolution |
| 11 | ferray-strings | sonnet | Straightforward string operations |
| 12 | ferray-ma | sonnet | Mask propagation logic, ufunc support, sorting, mask ops |
| 13 | ferray-stride-tricks | sonnet | Small crate, well-defined semantics |
| 14 | ferray-numpy-interop | sonnet | PyO3 + Arrow integration plumbing |
| 15 | Phase 3 integration | opus | Cross-crate resolution, full feature flag testing |
| 16 | f16 support | sonnet | Additions to existing crates |
| 17 | no_std core | sonnet | Conditional compilation |
| 18 | const generic shapes | opus | Advanced type-level programming |
| 19 | fixture generation | sonnet | Python script, JSON output |
| 20 | oracle tests | sonnet | Mechanical test writing from fixtures |
| 21 | property tests | opus | Mathematical invariant reasoning |
| 22 | fuzz + formal verification | opus | Prusti annotations, fuzz target design |
| 23 | verification integration | opus | Full-suite execution, report generation |
| fixup agents | (any) | opus | Debugging requires deep reasoning |

**Breakdown: 11 opus, 16 sonnet, fixups opus (27 agents total)**

**NOTE**: ferray-core is split into 4 sub-agents (1a-1d). Agent 1a must finish before 1b/1c/1d start. The coordinator merges all 4 before starting Phase 1 parallel agents (2-5).

### CLAUDE.md Project Conventions

Written by the coordinator at startup:

```markdown
# ferray — Project Conventions

## Rust Edition & MSRV
- Edition: 2024
- MSRV: 1.85 (stable)

## Import Paths
- Core types: `use ferray_core::{NdArray, Array1, Array2, ArrayD, ArrayView, Dimension}`
- Errors: `use ferray_core::FerrayError`
- Element trait: `use ferray_core::Element`
- Complex: `use num_complex::Complex`

## Error Handling
- All public functions return `Result<T, FerrayError>`
- Use `thiserror` 2.0 for derive
- Never panic in library code
- Every error variant carries diagnostic context

## Numeric Generics
- Element bound: `T: Element` (defined in ferray-core)
- Float-specific: `T: Element + Float` (uses num_traits::Float)
- Support f32, f64, Complex<f32>, Complex<f64>, and integer types

## SIMD Strategy
- Use `pulp` crate for runtime CPU dispatch (SSE2/AVX2/AVX-512/NEON)
- Scalar fallback controlled by `FERRAY_FORCE_SCALAR=1` env var
- All contiguous inner loops must have SIMD paths for f32, f64, i32, i64

## Testing Patterns
- Oracle fixtures: load JSON from `fixtures/`, compare with ULP tolerance
- Property tests: `proptest` with `ProptestConfig::with_cases(256)`
- Fuzz targets: one per public function family
- SIMD verification: run all tests with FERRAY_FORCE_SCALAR=1

## Naming Conventions
- Public array type: `NdArray<T, D>` (never expose ndarray types)
- Type aliases: Array1, Array2, Array3, ArrayD
- Module structure matches NumPy: linalg::, fft::, random::, etc.

## SIMD Strategy
- Use `pulp` crate for runtime CPU dispatch (SSE2/AVX2/AVX-512/NEON)
- Do NOT use `std::simd` — it is unstable. If you see examples using `std::simd::f64x4`, ignore them and use `pulp` instead.
- Scalar fallback controlled by `FERRAY_FORCE_SCALAR=1` env var
- All contiguous inner loops must have SIMD paths for f32, f64, i32, i64

## Crate Dependencies (use these exact versions)
ndarray = "0.17"
faer = "0.24"
rustfft = "6.4"
pulp = "0.22"
num-complex = "0.4"
num-traits = "0.2"
half = "2.4"
rayon = "1.11"
serde = { version = "1.0", features = ["derive"] }
thiserror = "2.0"
```

### Stuck-Agent Protocol

```
DETECT: Agent has been running >15 minutes with no new tool calls
         OR agent reports "I'm blocked" / "I can't figure out"
         OR agent's cargo test fails repeatedly (>3 attempts)

DIAGNOSE:
  1. Read the agent's full output transcript
  2. Identify the category:
     a. COMPILE_ERROR — missing import, wrong type, API mismatch
     b. TEST_FAILURE — logic bug, fixture mismatch, tolerance issue
     c. DESIGN_GAP — the design doc doesn't specify enough detail
     d. DEPENDENCY_CONFLICT — version incompatibility, missing feature flag
     e. SCOPE_CREEP — agent is doing more than assigned

RESPOND:
  a. COMPILE_ERROR → Resume agent with the exact fix
  b. TEST_FAILURE → Resume agent with diagnosis and suggested approach
  c. DESIGN_GAP → Coordinator makes the design decision, documents in crosslink,
     resumes agent with the decision
  d. DEPENDENCY_CONFLICT → Coordinator fixes Cargo.toml on dev, tells agent to pull
  e. SCOPE_CREEP → Resume agent with "Stop. Only implement {X}. Commit what you have."

ESCALATE: If unresolved after 2 attempts, create a crosslink issue tagged `blocker`,
  describe the problem, and move on. Return to it after the phase's other work completes.
```

### Context Compression Recovery

The coordinator's persistent state lives in three places:
1. **Crosslink issues** — every agent assignment, status, and outcome
2. **Design docs** — the authoritative task definitions
3. **Git branch state** — `dev` branch shows what's been merged

After context compression:
```bash
crosslink issues list --open
git log --oneline dev
cargo test --workspace 2>&1 | tail -20
```

## Open Questions

*None — all design decisions are resolved in the per-crate design docs.*

## Out of Scope
- The coordinator does not write algorithm/library code itself — it only orchestrates
- The coordinator does not modify design docs — if a design gap is found, it makes a decision and documents it in crosslink
- The coordinator does not handle deployment, CI setup, or publishing to crates.io
- The coordinator does not run the 24-hour fuzz campaign — it sets up targets, but the long run is human-initiated
