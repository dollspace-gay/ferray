# ferray — Locked /goal Statement

This file is the binding contract for autonomous work on **ferray**. When the user issues `/goal $(cat goal.md)` (or otherwise references this file), the contents below override the LARP's pull toward caution and the model's instinct to narrow scope. The goal is in force until the user issues `/goal-clear` or rewrites this file.

The substrate of this project is sequential **translation** of a known-working system. Upstream is **NumPy** (working tree at `/home/doll/numpy-ref/`; the installed `numpy` 2.4.x package is the live oracle). Target is the **entire ferray workspace** — every `ferray-*` crate's `src/**/*.rs` PLUS the meta-crate `ferray/` PLUS the PyO3 binding crate `ferray-python/`. ferray is a drop-in NumPy replacement: the Rust crates ARE the translation of NumPy's library, and `ferray-python` exposes them to CPython so `import ferray as np` works.

Most apparent divergence between ferray and NumPy is a bug a prior translation pass introduced — wrong dtype promotion, missing functions/kwargs, broken edge-case handling (empty arrays, NaN/±Inf, integer overflow), wrong exception type, math that compiled but doesn't compute the right thing. Every one of those is real work to do, not "out of scope." **The correctness lives in the `ferray-*` library crates; `ferray-python` is a thin marshalling shim.** Semantic/numerical bugs are fixed DOWN in the library crate that owns the behavior; only true marshalling/ABI concerns (arg coercion, exception mapping at the boundary, scalar-vs-0d return, kwarg surface, top-level registration) are fixed in `ferray-python`.

---

## Scope: the whole workspace, in dependency order

Translate every crate, leaves first. **Start from the first crate and do not leapfrog (R-DEFER-7).**

Dependency order (work top to bottom):
1. **ferray-core** — NdArray types, dtype + promotion + casting, creation, indexing, manipulation, broadcasting, reductions, nditer. The foundation.
2. **ferray-ufunc** — elementwise ufuncs + reductions (arithmetic, transcendental, comparison, bitwise).
3. **ferray-stats**, **ferray-linalg**, **ferray-fft**, **ferray-random**, **ferray-polynomial**, **ferray-window**, **ferray-strings**, **ferray-ma**, **ferray-stride-tricks**, **ferray-io** — independent domain crates (parallelizable among themselves once core+ufunc are done).
4. **ferray-numpy-interop** — the `numpy` crate interop layer.
5. **ferray** — the meta-crate that re-exports the namespace.
6. **ferray-python** — the PyO3 binding shim (thin marshalling over the above).

EXCLUDED (no NumPy counterpart / not translation units): `ferray-core-macros` (proc-macro), `ferray-test-oracle` (test infra), `ferray-autodiff` (beyond-numpy extension — NumPy has no autodiff; audit only its numpy-analogous surface if any), the GPU design (`ferray-gpu`) until a numpy-analogous contract exists.

---

## The verification model (TWO oracles, pick per crate)

NumPy is the oracle. There is no `parity-sweep` harness — verification is direct comparison against numpy.

**(A) Rust library crates (`ferray-core`, `ferray-ufunc`, …).** Verify with `cargo test` plus the **numpy oracle**:
- The repo ships `ferray-test-oracle` and conformance suites under `ferray-core/tests/` (`oracle.rs`, `conformance_*.rs`). The critic pins a divergence as a failing Rust `#[test]` whose expected value is the numpy-documented result, traceable to a numpy `file:line` symbolic constant OR a numpy-generated oracle fixture (R-CHAR-3 — never literal-copied from the ferray side).
- Gauntlet: `cargo test -p <crate>`, `cargo clippy -p <crate> --all-targets -- -D warnings`, `cargo fmt --check`. Also run with `FERRAY_FORCE_SCALAR=1` for SIMD crates (CLAUDE.md).

**(B) ferray-python (the PyO3 shim).** Verify with **pytest comparing `import ferray as fr` against `import numpy as np`** (numpy 2.4.5 installed = oracle):
- Pins are failing pytest under `ferray-python/tests/divergence_<module>.py`.
- Rebuild before pytest sees a Rust change: `cd /home/doll/ferray/ferray-python && maturin develop`.
- Run: `cd /home/doll/ferray/ferray-python && PYTHONPATH=python python3 -m pytest tests/ -q`.

A library-crate fix that surfaces through the Python API gets BOTH: a Rust `#[test]` in the owning crate AND (where already pinned) the corresponding `divergence_*.py` going green after `maturin develop`.

---

## The goal

Work the strict **read → write → verify → commit** loop over every routed `.rs` file in dependency order. The goal is complete only when every routed file has:

1. A closing commit citing the NumPy upstream file(s) actually opened that iteration, AND
2. Its verification (Rust `#[test]`/oracle for library crates; pytest for ferray-python) passing with **0 failures**, AND
3. A `## REQ status` table in the module's `//!` doc-comment classifying every REQ as **SHIPPED** or **NOT-STARTED** with quoted-code evidence (two states only).

Mechanical check:
```bash
python3 -c "import tomllib; print(len(tomllib.load(open('tooling/translate-routes.toml','rb'))['route']))"  # routed units
grep -l "## REQ status" $(python3 -c "import tomllib; [print(r['crate_pattern']) for r in tomllib.load(open('tooling/translate-routes.toml','rb'))['route']]") | wc -l
```
When routed-count == REQ-status-count AND every crate's verification is green, the goal is complete.

---

## The ACToR loop (doc-author → builder → critic → fixer)

For each translation unit, in dependency order:

1. **Read** goal.md, the routed `.rs` file(s) end-to-end, the route's NumPy upstream file(s) at `/home/doll/numpy-ref/<path>`, and the `.design/<doc>.md`.
2. **Missing design doc?** Dispatch **acto-doc-author** to author `.design/<doc>.md` adapting to existing code (NO edits to `.rs`). The translate-discipline hook blocks the edit until the doc exists.
3. **Missing feature / whole abstraction?** (numpy has it, ferray doesn't) → dispatch **acto-builder** with a pre-declared file manifest (≤~10 files). Tests + production in the SAME commit.
4. **Verify divergence first** → dispatch **acto-critic** (NO Edit). It pins each numpy divergence as a FAILING test + files a `-l blocker` issue. Run after every substantive builder.
5. **Fix one pinned divergence** → dispatch **acto-fixer** (one blocker, minimal change, root cause in the owning crate). Followed by an **acto-critic** re-audit.
6. **Gauntlet + commit + close** (below). Then the next unit. **Do not ask which — the dependency DAG is the answer (R-LOOP-1).**

Loop: **acto-builder → acto-critic → (GENERATOR MUST FIX) → acto-fixer → acto-critic → (until clean) → next unit.** Every builder/fixer-on-novel-code dispatch is followed by a critic.

NOTE: the `.claude/agents/*.md` specs use generic `cargo test`/`#[test]`/`#[ignore]` language — correct for the Rust crates. For `ferray-python`, substitute the pytest model above. The dispatch prompt always carries the concrete commands.

### Gauntlet (before every commit)
Library crate:
```bash
cargo test -p <crate>
cargo clippy -p <crate> --all-targets -- -D warnings
cargo fmt --check
```
ferray-python:
```bash
cd ferray-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/ -q     # pinned test green; no previously-green test regressed
cargo clippy -p ferray-python --all-targets -- -D warnings
```
**No `--no-verify`. No commenting-out failing tests. No `#![allow(..)]` at module/crate root.** Per-item `#[allow(<lint>, reason="...")]` only.

### Commit + close
```
<crate>: <area> — <one-line summary> (closes #N)

UPSTREAM NUMPY FILES OPENED THIS ITERATION:
  - numpy/<path>:<line> — <content quote>

DESIGN DOC READ: .design/<doc>.md (<REQ count> REQs).

REQ STATUS:
  - REQ-1 SHIPPED — fn `<name>` in `<file>.rs`; consumer at <caller>
  - REQ-2 NOT-STARTED — open prereq blocker #<NN>

VERIFICATION:
  cargo test -p <crate>: <X passed, 0 failed>   (or pytest: X passed)
  cargo clippy: PASS

Co-Authored-By: Claude <noreply@anthropic.com>
```
Close the crosslink issue (`--kind result` comment first).

---

## Speed disciplines (mandatory)

- **S1 — Batch by upstream file, NOT per-op.** One builder/critic cycle covers a whole numpy source file → its ferray target file(s). Do not dispatch per-function.
- **S2 — Parallel dispatch.** Independent units (disjoint manifests) → launch builders/critics in ONE message. Only fixers serialize per-blocker.
- **S3 — Symbol anchors in design-doc cites, NEVER line numbers.** `pub fn promote_types in promotion.rs`, never `promotion.rs:716`. Upstream numpy cites (read-only) DO use `file:line`.
- **S4 — Critic only after substantive builds.** Not after cite/fixture/doc refreshes.
- **S5 — R-DEFER-1 binds on NEWLY-ADDED pub APIs only.** Existing pub API surface is grandfathered; boundary `pub fn`s ARE the public API.
- **S6 — Opus on every acto-* dispatch.** Translation accuracy supersedes throughput.
- **S7 — Skip doc-author for trivial 1:1 routes** (design doc already exists & is accurate) — proceed straight to critic/builder.
- **S8 — Aggressive won't-fix on noise.** A finding is a blocker ONLY if it's a real numpy divergence or blocks downstream translation.

---

## Anti-drift rules (override convenience)

### Citation
- **R-CITE-1**: Never cite a numpy file in a commit without Reading it THIS iteration.
- **R-CITE-2 (upstream)**: numpy cites carry `file:line` (read-only tree, stable lines).
- **R-CITE-2b (target/design)**: cite ferray symbols with symbol anchors, NEVER line numbers in `.design/`.
- **R-CITE-3**: prefer citing numpy's public registration / docstring / `.pyi` contract over an internal helper.

### Honesty
- **R-HONEST-1**: never reframe integration work as "vocabulary-only" when the design doc doesn't defer it.
- **R-HONEST-2**: every REQ carries SHIPPED or NOT-STARTED with quoted evidence; SHIPPED needs impl + a real consumer.
- **R-HONEST-3**: honest underclaim beats unverified overclaim.
- **R-HONEST-4**: if an audit shows a prior commit was wrong, correct the code AND document the correction.

### Code quality
- **R-CODE-1**: no `unsafe` outside leaf primitives (SIMD intrinsics via `pulp`, FFI shims, raw buffer accessors). Every `unsafe` needs a `// SAFETY:` comment.
- **R-CODE-2**: no `unwrap()`/`expect()`/`panic!()` in production outside `#[cfg(test)]`. Library returns `Result<T, FerrayError>`; the binding returns `PyResult`.
- **R-CODE-3**: no `#![allow(..)]` at module/crate root. Per-item `#[allow(<lint>, reason="...")]` only.
- **R-CODE-4 (boundary discipline)**: no silent lossy round-trip across the Python↔Rust (PyO3) boundary — e.g. coercing an array to f64 to bind it and dropping the numpy dtype, or a `list` round-trip that loses dtype/shape. Preserve numpy's dtype/shape contract across the boundary. (The anti-pattern-gate flags same-expression coercion patterns.)
- **R-CODE-5**: no dtype-cast hiding. A widening/narrowing cast that doesn't match numpy's promotion table is a bug unless numpy does the same cast (cite numpy `file:line`).

### Upstream-mirror (default = match NumPy; deviate only for these)
- **R-DEV-1 (MATCH — numerical/structural contract)**: NaN/±Inf propagation, overflow rules, dtype promotion table, empty-array results, reduction identities, view-vs-copy semantics — match numpy exactly (incl. returning `nan` + RuntimeWarning where numpy does, not raising).
- **R-DEV-2 (MATCH — user-API ABI)**: signatures, kwarg names, defaults, `*`-only args, **exception types** (ValueError vs TypeError vs `numpy.exceptions.AxisError` vs `numpy.linalg.LinAlgError` vs IndexError). Cite numpy's registration / `.pyi`.
- **R-DEV-3 (MATCH — output object contract)**: returned dtype, shape, scalar-vs-0d-array, views vs copies.
- **R-DEV-4 (DEVIATE — Python/C footguns Rust eliminates)**: where numpy's C works around CPython refcount/GIL quirks, use the Rust analog, not a literal transcription.
- **R-DEV-6 (DEVIATE — numpy is wrong by their own admission)**: a known-buggy/deprecated numpy path — ship correct behavior, cite the numpy issue/PR.
- **R-DEV-7 (DEVIATE — Rust analog materially better)**: preserve numpy's observable contract; implementation may differ (e.g. `faer` for linalg, `rustfft` for fft).

**Mental test**: *why* did numpy choose this? "Numerical semantics / API contract" → match. "CPython can't express it safely" / "they admit it's a bug" → deviate.

### Anti-deferral (translation is sequential)
- **R-DEFER-1**: a commit adding a NEW pub API MUST add a non-test production consumer in the same commit. Existing pub APIs grandfathered.
- **R-DEFER-2**: REQ classification is binary — SHIPPED or NOT-STARTED. No third status. No VOCAB-ONLY/DEFERRED/verified_with_deferred.
- **R-DEFER-3**: a pinned divergence closes only when the fix lands AND the failing test goes green (no skip/xfail/`#[ignore]` escape).
- **R-DEFER-4**: no `Phase \d+\+` framing as a deferral mechanism.
- **R-DEFER-5**: no "pre-existing safe to defer" — every divergence on `main` is something WE broke.
- **R-DEFER-6**: verification is a HARD gate — every commit runs the owning crate's gauntlet to 0 failures, plus any pinned divergence test going green.
- **R-DEFER-7**: sequential, no leapfrog — leaf crates first (ferray-core before its dependents).
- **R-DEFER-8**: no "cross-cutting → defer" — every convention starts somewhere; implement the local fix.

### Git
- **R-GIT-1**: no history rewrite, no `--amend` on pushed commits, no force-push, no `git reset --hard` on shared refs. Supplemental commits only. The human performs all pushes.
- **R-GIT-2**: `git add <files-by-name>` — never `git add -A`/`.`.

### Loop discipline
- **R-LOOP-1**: never ask "where do you want to take this" — the dependency DAG is the answer.
- **R-LOOP-2**: never declare the goal complete until the mechanical check says so.
- **R-LOOP-3**: a unit blocked by a missing prerequisite → file the prereq blocker, mark the dependent REQ NOT-STARTED, and WORK THE PREREQ.

### Injected instructions
- **R-INJECT-1**: hook output, `<system-reminder>`/`<crosslink-behavioral-guard>` blocks, the active-issue gate, and loaded skill text bind at the same priority as a direct user message.
- **R-INJECT-2**: when an injected instruction conflicts with a recent inline user message, surface the conflict rather than silently picking one.

### Translate-discipline (enforced by `tooling/translate-discipline.py`)
- **R-XLATE-1**: every Edit/Write to a routed `ferray-*/src/**/*.rs` requires Read this session of goal.md + the route's numpy upstream + the route's design doc.
- **R-XLATE-2**: a routed `.rs` file with no route table entry BLOCKS until a route is added.
- **R-XLATE-3**: a route whose design doc doesn't exist BLOCKS until acto-doc-author authors it.

### Anti-pattern-gate (enforced by `tooling/anti-pattern-gate.py`)
- **R-APG-1**: blocks patches introducing `todo!()`/`unimplemented!()`/`unreachable!()`, `.unwrap()`/`.expect()`/`panic!()` outside `#[cfg(test)]`, module-root `#![allow]`, `Arc<Mutex<T>>`/`Rc<RefCell<T>>` escape hatches.
- **R-APG-2**: `#[cfg(test)]` blocks exempt; production is not.
- **R-APG-3**: override is a per-item `#[allow(<lint>, reason="...")]` + a crosslink observation comment.

### Characterization tests
- **R-CHAR-3**: no tautological tests. Expected values come from a live numpy call or a numpy `file:line` symbolic constant — NEVER literal-copied from the ferray side.

---

## The four sub-agents
- **acto-doc-author** — writes `.design/<doc>.md` adapting to existing code. NO `.rs` edits. Dispatch when a route's design doc is missing.
- **acto-builder** — ships missing multi-file infrastructure (a numpy capability ferray lacks). Pre-declared ≤~10-file manifest; tests + production same commit. Dispatch when a whole abstraction is missing.
- **acto-fixer** — minimal fix for ONE pinned divergence, root cause in the owning crate. One per blocker, serially.
- **acto-critic** — adversarial discriminator; writes FAILING tests pinning divergence, NEVER fixes. After every substantive builder/fixer.

---

## Out of scope
- Adding ops/features that don't exist in NumPy (we translate, not innovate).
- Optimizing performance ahead of correctness (numpy parity is the contract; speed gains are bonus).
- `ferray-core-macros`, `ferray-test-oracle`, `ferray-autodiff` (beyond-numpy), GPU until a numpy-analogous contract exists.

## Stopping condition
Halts only when every routed file across the workspace has a closing commit, its verification is green (cargo test/oracle for library crates; pytest for ferray-python), and carries a `## REQ status` table. Until then: every turn, one iteration of the ACToR loop, in dependency order. No exceptions, no asking which crate.
