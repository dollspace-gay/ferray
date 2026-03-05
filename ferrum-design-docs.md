---
title: ferrum — Project Conventions
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

A single orchestration document that, when kicked off, autonomously builds the entire ferrum library across five phases by spawning, monitoring, and merging the work of ~20 subagents. The coordinator reads the per-crate design docs in `.design/`, spawns agents in dependency-graph order, gates phase transitions on acceptance criteria, recovers stuck agents, and merges worktrees into a coherent codebase. The human's role reduces to approving phase gates and resolving any issues the coordinator escalates.

### Requirements

- REQ-1: The coordinator reads `.design/ferrum-*.md` and `.design/phase-{4,5}-*.md` as its authoritative task definitions — it does not invent work beyond what the design docs specify
- REQ-2: Agents are spawned using the `Agent` tool with `isolation: "worktree"` so each works on an isolated copy of the repo
- REQ-3: The coordinator maintains the dependency DAG (documented below) and never spawns an agent before its dependencies have been merged to the integration branch
- REQ-4: Phase transitions (1→2, 2→3, 3→4, 4→5) are gated on all acceptance criteria for the completing phase passing — verified by running `cargo test --workspace` and `cargo clippy --workspace` on the integration branch
- REQ-5: The coordinator uses crosslink issues to track every agent's assignment, status, and outcome
- REQ-6: When an agent completes, the coordinator merges its worktree branch into the `dev` integration branch, resolving conflicts if necessary
- REQ-7: When an agent is stuck (no progress for 3+ turns, or reports a blocker), the coordinator reads the agent's output, diagnoses the issue, and either provides guidance by resuming the agent or spawns a fixup agent
- REQ-8: After each phase completes, the coordinator runs the full test suite and records the result as a crosslink comment before proceeding
- REQ-9: The coordinator creates a `CLAUDE.md` project file at the start with conventions all subagents must follow
- REQ-10: On context compression, the coordinator recovers by re-reading crosslink issue state and the current phase design doc to reconstruct its position ### Agent Management
- REQ-11: Each spawned agent receives: (a) its design doc requirements, (b) file paths to create/modify, (c) acceptance criteria, (d) CLAUDE.md conventions, (e) instructions to commit and test before finishing
- REQ-12: Agents use `model: "sonnet"` for straightforward implementation and `model: "opus"` for architecturally complex tasks (see Model Selection table)
- REQ-13: Maximum 8 concurrent background agents
- REQ-14: Each agent's crosslink issue includes a `--kind result` comment documenting what was delivered ### Merge Strategy
- REQ-15: Integration branch `dev` is created from `main` at the start
- REQ-16: Merges happen sequentially with `cargo build --workspace` verification after each
- REQ-17: After all agents in a phase merge, full acceptance criteria check before next phase
- REQ-18: At the end of Phase 5, create a PR from `dev` to `main`

### Acceptance Criteria

- [ ] AC-1: Running `crosslink kickoff run "Build ferrum" --doc .design/phase-0-coordinator.md` produces a working library with no further human intervention (beyond phase gate approvals)
- [ ] AC-2: All acceptance criteria across phases 1-5 pass on the final `dev` branch
- [ ] AC-3: Every agent's work is tracked via a crosslink issue with typed comments
- [ ] AC-4: No agent is left stuck for more than 10 minutes without coordinator intervention
- [ ] AC-5: `cargo test --workspace` passes after each phase gate
- [ ] AC-6: Context compression does not cause the coordinator to lose track of progress

### Out of Scope

- The coordinator does not write algorithm/library code itself — it only orchestrates
- The coordinator does not modify design docs — if a design gap is found, it makes a decision and documents it in crosslink
- The coordinator does not handle deployment, CI setup, or publishing to crates.io
- The coordinator does not run the 24-hour fuzz campaign — it sets up targets, but the long run is human-initiated

### rust edition & msrv

- Edition: 2024
- MSRV: 1.85 (stable)

### import paths

- Core types: `use ferrum_core::{NdArray, Array1, Array2, ArrayD, ArrayView, Dimension}`
- Errors: `use ferrum_core::FerrumError`
- Element trait: `use ferrum_core::Element`
- Complex: `use num_complex::Complex`

### error handling

- All public functions return `Result<T, FerrumError>`
- Use `thiserror` 2.0 for derive
- Never panic in library code
- Every error variant carries diagnostic context

### numeric generics

- Element bound: `T: Element` (defined in ferrum-core)
- Float-specific: `T: Element + Float` (uses num_traits::Float)
- Support f32, f64, Complex<f32>, Complex<f64>, and integer types

### simd strategy

- Use `pulp` crate for runtime CPU dispatch (SSE2/AVX2/AVX-512/NEON)
- Scalar fallback controlled by `FERRUM_FORCE_SCALAR=1` env var
- All contiguous inner loops must have SIMD paths for f32, f64, i32, i64

### testing patterns

- Oracle fixtures: load JSON from `fixtures/`, compare with ULP tolerance
- Property tests: `proptest` with `ProptestConfig::with_cases(256)`
- Fuzz targets: one per public function family
- SIMD verification: run all tests with FERRUM_FORCE_SCALAR=1

### naming conventions

- Public array type: `NdArray<T, D>` (never expose ndarray types)
- Type aliases: Array1, Array2, Array3, ArrayD
- Module structure matches NumPy: linalg::, fft::, random::, etc.

### crate dependencies (use these exact versions)

ndarray = "0.16"
faer = "0.24"
rustfft = "6.2"
pulp = "0.20"
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

