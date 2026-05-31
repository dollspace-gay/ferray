# ferray (main array) — Complex-dtype support for `fr.array` + top-level ufuncs

<!--
tier: 3-component
status: draft
baseline-commit: 672a6aa5
upstream-paths:
  - /home/doll/numpy-ref/numpy/_core/code_generators/generate_umath.py
  - /home/doll/numpy-ref/numpy/_core/src/umath/loops_unary_complex.dispatch.c.src
  - /home/doll/numpy-ref/numpy/_core/fromnumeric.py
-->

## Summary

ferray-python is a thin marshalling shim: there is NO custom `fr.ndarray`
pyclass — `fr.ndarray` is numpy's own ndarray (re-exported in `lib.rs`), and
every top-level `fr.*` function accepts and returns a real `numpy.ndarray`.
Complex arrays therefore already EXIST end-to-end: `fr.zeros(3,
dtype='complex128')` and `fr.fft.fft(real)` produce usable `complex128`
ndarrays, and numpy's OWN operators/methods on those arrays (`z + z`, `z * z`,
`z.real`, `z.conj()`) work because they are numpy code, not ferray code.

The divergence is entirely in ferray's top-level **functions**, whose dtype
dispatch is real-only. Three failure classes (verified live, numpy 2.4.5):

1. **Input coercion rejects complex.** `fr.array([1+2j])` /
   `fr.asarray(np.array([1+2j]))` → `TypeError: unsupported dtype: "complex128"
   (supported: bool, int8/16/32/64, uint8/16/32/64, float32/64)`. The `array`
   copy path dispatches through `match_dtype_all!` (`conv.rs`), an 11-real-dtype
   macro with no complex arm.
2. **Silent imaginary-discard = DATA CORRUPTION (R-CODE-4).** `fr.sqrt(z)`
   returns `[1.+0.j]` (real part only, `ComplexWarning`) where numpy returns
   `[1.272+0.786j]`; `fr.exp`/`fr.log`/`fr.sin` likewise; `fr.mean(z)` returns
   `1.0` (drops imag) where numpy returns `(2+0.5j)`. The unary-float ufunc body
   coerces complex → `float32` (dropping imag) and then re-tags the result
   complex; `mean` coerces complex → `float64`.
3. **Raises on numeric ops.** `fr.abs(z)`/`fr.negative(z)` → `TypeError:
   unsupported dtype for numeric op`; `fr.add`/`subtract`/`multiply`/`divide`,
   `fr.sum`/`fr.prod`, and all six comparisons → the same.

This doc mirrors the proven ma-complex epic (#867; `.design/ferray-ma-complex.md`)
applied to the MAIN-array dispatch sites: thread complex arms through the
binding's dtype macros and reuse the ferray-ufunc / ferray-core complex
substrate, which (as in the ma case) is **almost entirely already shipped**.

## The pivotal finding (library readiness)

The library substrate is **MORE ready than the ma case was** — the
ma-complex epic (#868/#869/#873) already landed the complex library primitives
this main-array work composes. This is therefore **overwhelmingly binding work**
(thread complex arms through `conv.rs` / `ufunc.rs` / `stats.rs` dispatch
macros), with at most one small net-new library helper (a complex `negative`).

1. **`Complex<f32>`/`Complex<f64>` ARE `Element`.** `impl Element for
   Complex<f32>` / `impl Element for Complex<f64>` (`ferray-core/src/dtype/mod.rs`),
   with `DType::Complex32` / `DType::Complex64` threaded through promotion
   (`promotion.rs`: `Complex32 => 14`, `Complex64 => 15`) and casting
   (`casting.rs`). So `ArrayD<Complex<f64>>` is constructible — `fr.zeros(…,
   dtype='complex128')` already does so via `creation_dispatch!`'s `complex128`
   arm (RC-COMPLEX, SHIPPED in `creation.rs`).

2. **The complex transcendentals ALREADY EXIST in ferray-ufunc.**
   `ferray_ufunc::{sin_complex, cos_complex, tan_complex, sinh_complex,
   cosh_complex, tanh_complex, asin_complex, acos_complex, atan_complex,
   asinh_complex, acosh_complex, atanh_complex, exp_complex, ln_complex,
   sqrt_complex, log2_complex, log10_complex, expm1_complex, log1p_complex,
   power_complex}` are all `pub` re-exported at the crate root
   (`ferray-ufunc/src/lib.rs`), bounded `T: Element + Float, Complex<T>:
   Element` and delegating to `num_complex::Complex::*` (`ops/complex.rs`,
   `macro_rules! complex_unary`). So R-3 is **binding dispatch only** — no
   net-new library transcendental.

3. **Complex arithmetic ALREADY compiles.** `impl WrappingArith for
   Complex<f32>`/`Complex<f64>` (`impl_wrapping_arith_complex!`,
   `arithmetic.rs`) makes the generic `add_broadcast`/`subtract_broadcast`/
   `multiply_broadcast` (all bounded `T: Element + WrappingArith`) accept
   complex; `impl TrueDivide for Complex<…>` (`arithmetic.rs`) makes
   `divide_broadcast` (bounded `T: TrueDivide`) accept complex;
   `ferray_ufunc::power_complex` exists. Shipped under ma #869. So R-5
   arithmetic is **binding dispatch only**.

4. **Complex `equal`/`not_equal` ALREADY compile.** `pub fn equal_broadcast`
   bound `T: Element + PartialEq + Copy` (`comparison.rs`) — `Complex`
   satisfies it. So `fr.equal`/`fr.not_equal` over complex are binding dispatch
   only.

5. **Complex reduction accumulators ALREADY exist.** `impl ReduceAcc for
   num_complex::Complex<f32>` / `Complex<f64>` (`ferray-core/src/array/
   reductions.rs`, `Acc = Self`) — `Array::sum`/`Array::prod` fold complex
   today. So R-6 `sum`/`prod` are binding dispatch only.

6. **`abs(complex)` ALREADY exists and returns REAL.** `ferray_ufunc::abs`
   (`ops/complex.rs`, `pub fn abs<T,D>(input: &Array<Complex<T>, D>) ->
   FerrayResult<Array<T, D>>`, bounded `T: Element + Float`) returns the REAL
   magnitude array — exactly numpy's `abs(complex)->float64`. So R-4 `abs` is
   binding dispatch only.

**THE TWO GENUINE GAPS (both small):**

- **Lexicographic complex ordering.** `pub fn less_broadcast` / `greater_…` /
  `…_equal` are bounded `T: Element + PartialOrd + Copy` (`comparison.rs`), and
  `Complex` is NOT `PartialOrd`, so they will NOT compile for complex. numpy's
  `np.less([1+2j,3+2j],[1+5j,3+1j])` → `[True False]` — it compares
  **lexicographically (real, then imag)** and does NOT raise (verified live,
  numpy 2.4.5). This needs a lexicographic complex compare, identical in shape
  to ma #874's `cmp_complex_arm!`. (NOTE: this contradicts the task brief's
  claim that complex `<`/`>` raise — the live oracle computes them.)
- **Complex `negative`.** `ferray_ufunc::negative` is bounded `T: Element +
  Float` (`arithmetic.rs`) — `num_traits::Float` is not impl'd for `Complex`,
  so it won't take complex. Either a 2-line `negative_complex` helper
  (`-z` via `num_complex::Complex` `Neg`) or bind it as `subtract(0+0j, z)`
  reusing the existing complex `subtract_broadcast`.

**Magnitude verdict:** ~90% binding (thread complex arms through the
`conv.rs`/`ufunc.rs`/`stats.rs` dispatch macros + a complex unary/binary
dispatch branch), ~10% library (one tiny complex `negative`; lexicographic
compare can live as a binding helper exactly as ma did). NOT a multi-builder
epic on the scale of #867 — the library was already built by the ma epic; this
is the cheaper "wire the same primitives into the main-array dispatch sites"
follow-through. Smallest-highest-value first: **R-2 (stop the corruption) +
R-1 (accept complex input)** unblock the data-integrity hazard and the
construct path; R-3/R-4/R-5/R-6 then follow as dispatch threading.

## Requirements

- REQ-1 (R-1 input coercion): `fr.array([1+2j])` and
  `fr.asarray(np.array([1+2j]))` build a `complex128` (or `complex64`) ndarray
  instead of raising `TypeError` — the `array`/`asarray` copy path gains complex
  arms so complex input round-trips dtype + shape (R-CODE-4).
- REQ-2 (R-2 STOP the corruption): `fr.sqrt`/`exp`/`log`/`sin`/… and `fr.mean`
  over complex MUST NOT silently discard the imaginary part. Either compute the
  complex result (numpy's behavior) or raise — never emit a real-only result
  with `ComplexWarning`. (numpy computes; so the fix is to compute, satisfied by
  REQ-3/REQ-6.) This is the highest-priority REQ: it is an active R-CODE-4
  boundary violation.
- REQ-3 (R-3 complex transcendentals): `fr.sqrt`/`exp`/`log`/`log2`/`log10`/
  `expm1`/`log1p`/`sin`/`cos`/`tan`/`sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/
  `arctan`/`arcsinh`/`arccosh`/`arctanh`/`power` over complex compute the
  complex result via the existing `ferray_ufunc::*_complex` family.
- REQ-4 (R-4 abs/negative/positive): `fr.abs(complex)` returns the REAL
  magnitude (`float64` for c128, `float32` for c64) via `ferray_ufunc::abs`;
  `fr.negative(complex)` negates (complex result); `fr.positive(complex)` is the
  identity (already works).
- REQ-5 (R-5 arithmetic + comparison + raises): `fr.add`/`subtract`/`multiply`/
  `divide` over complex compute (via the WrappingArith/TrueDivide complex
  impls); `fr.equal`/`not_equal` compute; `fr.less`/`greater`/`less_equal`/
  `greater_equal` compute LEXICOGRAPHICALLY (real, then imag), matching numpy;
  `fr.floor_divide`/`remainder`/`mod`/`bitwise_*` RAISE `TypeError` matching
  numpy's "ufunc not supported".
- REQ-6 (R-6 reductions): `fr.sum`/`fr.prod` over complex yield a complex
  scalar (via the complex `ReduceAcc`); `fr.mean` over complex yields a complex
  scalar (`complex128`, matching numpy's c64→c128 mean promotion), NOT a real
  collapse.

## Acceptance criteria

- AC-1 (REQ-1): pytest `fr.array([1+2j]).dtype == "complex128"` and
  `np.asarray(fr.array([1+2j])) == np.array([1+2j])`;
  `fr.asarray(np.array([1+2j], dtype='complex64')).dtype == "complex64"`.
- AC-2 (REQ-2): pytest asserts NO `ComplexWarning` is raised by
  `fr.sqrt`/`exp`/`log`/`sin`/`mean` on a complex input, AND the result equals
  the numpy value (the R-CODE-4 imag-drop guard, mirroring ma's
  `test_complex_mean_keeps_imag_matches_numpy`).
- AC-3 (REQ-3): `fr.sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/… of a complex array
  equal `np.sqrt`/`np.exp`/… elementwise (oracle-generated, R-CHAR-3).
- AC-4 (REQ-4): `fr.abs([3+4j]) == [5.0]` with dtype `float64`;
  `fr.negative([1+2j]) == [-1-2j]`; `fr.positive([1+2j]) == [1+2j]`.
- AC-5 (REQ-5): `fr.add`/`*`/`-`/`divide` of complex equal numpy elementwise;
  `fr.equal`/`not_equal` equal numpy; `fr.less([1+2j,3+2j],[1+5j,3+1j]) ==
  [True, False]` (lexicographic); `fr.floor_divide(z,z)` raises `TypeError`;
  `fr.bitwise_and(z,z)` raises `TypeError`.
- AC-6 (REQ-6): `fr.sum([1+2j,3-1j]) == (4+1j)`; `fr.prod == (3+5j... )` equal
  to `np.prod`; `fr.mean([1+2j,3-1j]) == (2+0.5j)` with dtype `complex128`.

## REQ status table

REQ-1/REQ-2/REQ-3 are now **SHIPPED** (closed by #920/#921/#922 — the
input-coercion + unary-transcendental complex-dispatch build). REQ-4/REQ-5/REQ-6
remain **NOT-STARTED** (their binding dispatch sites are still real-only; the
divergences reproduce live, numpy 2.4.5). Two states only (R-DEFER-2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (input coercion) | SHIPPED | `complex_roundtrip_copy` + `is_complex_dtype` in `creation.rs` branch complex input ahead of `match_dtype_all!` and route it through `complex_pyarray_to_ferray`/`complex_ferray_to_pyarray`. Non-test production consumer: `pub fn array` / `pub fn asarray` / `pub fn copy` in `creation.rs` (registered top-level in `lib.rs`). conv.rs's `pyany_to_dynarray`/`dynarray_to_pyarray` also gained `Complex32`/`Complex64` arms for the file-IO `DynArray` path. numpy: `np.array([1+2j]).dtype == complex128` (numpy/_core/numeric.py def asarray). Pinned green: `tests/test_expansion_complex_array.py::test_array_python_complex_list_builds_complex128`, `::test_asarray_complex64_preserves_dtype`, `::test_copy_complex_round_trips`. |
| REQ-2 (STOP imag-discard) | SHIPPED | `bind_unary_float!` (`ufunc.rs`) now branches on `is_complex_dtype` BEFORE `unary_float_body!`'s `coerce_dtype(arr,"float32")`: ops with a complex loop compute complex; ops without one (`cbrt`/`fabs`/`degrees`/…) raise `reject_complex_unary` (numpy's `TypeError`) instead of dropping the imaginary part. Non-test consumer: `pub fn sqrt`/`exp`/`log`/`sin`/… in `ufunc.rs`. numpy computes complex (generate_umath.py:966-973 `sqrt` `'fdg' + cmplx`). Pinned green: `::test_transcendental_emits_no_complex_warning`, `::test_sqrt_complex_no_imag_discard_matches_numpy`. (`fr.mean`'s complex collapse stays under REQ-6/#925.) |
| REQ-3 (complex transcendentals) | SHIPPED | `complex_unary_dispatch!` (`ufunc.rs`) calls the matching `ferray_ufunc::*_complex` op (`sqrt_complex`/`exp_complex`/`ln_complex`/`sin_complex`/… 19 ops) for complex input, preserving width (c64→c64, c128→c128); `exp2` (no `exp2_complex` lib op) is composed via `exp2_complex_dispatch` (num_complex `Complex::exp2`). Non-test consumer: the 20 `bind_unary_float!(…, complex = …)` invocations + `pub fn exp2` in `ufunc.rs`. KNOWN residual: `arcsin`/`arccos`/`arctanh` pick the opposite branch-cut sign to numpy at a real argument `|re|>1, im==+0` (a num_complex vs C99 divergence in `ferray-ufunc/src/ops/complex.rs`, outside this binding's manifest — interior points match). Pinned green: `::test_transcendental_complex128_matches_numpy`, `::test_transcendental_complex64_preserves_width_matches_numpy`, `::test_sqrt_negative_real_complex_takes_complex_branch`; the residual is pinned in `::test_arc_branch_cut_real_axis_known_divergence`. |
| REQ-4 (abs/negative/positive) | NOT-STARTED | open prereq blocker #923. `fr.abs` is a Python-level alias of `absolute`; `bind_unary_numeric_split!(absolute, …)` (`ufunc.rs`) dispatches `match_dtype_float_or_int!`, real-only → `TypeError: unsupported dtype for numeric op`. `ferray_ufunc::abs(&Array<Complex<T>,D>) -> Array<T,D>` (REAL magnitude) EXISTS (`ops/complex.rs`). `fr.negative` likewise raises; `ferray_ufunc::negative` is `T: Float`-bounded (no Complex) — the one small library gap (a `negative_complex` helper, or bind via `subtract(0+0j, z)`). `fr.positive(complex)` already works (`positive_int_array`-style identity). Fix = complex arm in `absolute`/`negative` dispatch + the tiny complex negate. |
| REQ-5 (arithmetic/compare/raises) | NOT-STARTED | open prereq blocker #924. `fr.add`/`subtract`/`multiply`/`divide` use `binary_numeric_body!` → `match_dtype_numeric!` (`ufunc.rs`), real-only → `TypeError` though library `add_broadcast`/`subtract_broadcast`/`multiply_broadcast` (`T: WrappingArith`, Complex impl'd) + `divide_broadcast` (`T: TrueDivide`, Complex impl'd) ALREADY accept complex (#869). `fr.equal`/`not_equal` use `comparison_body!` → `match_dtype_numeric!` (real-only), though `equal_broadcast` is `PartialEq`-bound (Complex OK). `fr.less`/`greater`/`le`/`ge`: `less_broadcast` is `PartialOrd`-bound and Complex is NOT `PartialOrd` → library gap; numpy COMPUTES these LEXICOGRAPHICALLY (`np.less([1+2j,3+2j],[1+5j,3+1j])` → `[True False]`, live) so the fix is a lexicographic compare helper (mirror ma #874 `cmp_complex_arm!`), NOT a raise. `fr.floor_divide`/`remainder`/`mod`/`bitwise_*` must RAISE `TypeError` for complex (numpy: `ufunc 'floor_divide'/'bitwise_and' not supported`). Fix = complex arms in the arithmetic/comparison dispatch + lexicographic compare + raise guards. |
| REQ-6 (sum/prod/mean) | NOT-STARTED | open prereq blocker #925. `pub fn sum`/`pub fn prod` (`stats.rs`) use `match_dtype_numeric!`, real-only → `TypeError`, though `impl ReduceAcc for Complex<f32>`/`Complex<f64>` (`ferray-core/src/array/reductions.rs`, `Acc = Self`) ALREADY exists so `Array::sum`/`Array::prod` fold complex. `pub fn mean` uses `match_dtype_float!` → coerces complex→f64 DROPPING imag (`fr.mean(z)` → `1.0`, an R-CODE-4 corruption; numpy → `(2+0.5j)`, c64→c128). Fix = complex arms in `sum`/`prod`/`mean` dispatch (`stats.rs`); `mean` computes complex-sum/count → `complex128`, mirroring ma #873. |

## Architecture

### There is NO `fr.ndarray` pyclass — the "main array" is numpy's ndarray

`fr.ndarray` re-exports `numpy.ndarray` (`lib.rs`); no `add_class::<…>` for an
array type (only `Polynomial`, `PyMaskedArray`, the random generators, `PyDual`
are pyclasses). Every `fr.*` ufunc/creation/reduction is a `#[pyfunction]`
taking `&Bound<PyAny>` and returning a real `numpy.ndarray` (pushed across via
`IntoNumPy` / `into_pyarray`, or the `complex_ferray_to_pyarray` fft helper for
complex). Consequence: numpy's OWN `z+z`, `z*z`, `z.real`, `z.conj()`,
`z.sum()` (method form) all work on a complex ndarray because they are numpy,
not ferray — which is why the task's "conj/real/+/* work" observation holds
while the analogous `fr.add`/`fr.conj`-via-binding paths do not. ferray's job
is purely the top-level FUNCTION surface.

### Where complex must be threaded (binding)

The dispatch substrate is a dtype-string → `match` macro that monomorphizes one
generic body per real dtype. Adding complex means giving the complex dtypes a
path that dispatches to the complex-typed library ops (NOT the f32/f64 funnel):

1. **`conv.rs` — `match_dtype_all!`** is the 11-real-dtype macro behind
   `fr.array`/`asarray`/`copy`/`array_equal`/`logical_*`. For the
   construction/round-trip path (REQ-1) the `array` copy body must branch on a
   complex dtype and route through `complex_ferray_to_pyarray` (the egress
   helper `creation.rs`/`fft.rs` already use), since pyo3-numpy has no complex
   `NumpyElement`/`into_pyarray` — exactly as `creation_dispatch!`'s complex
   arms already do.

2. **`ufunc.rs` — `unary_float_body!` / `bind_unary_float!`** (sqrt/exp/log/
   sin/…): add a complex branch that, for a complex dtype, extracts the complex
   array (via the complex marshaller) and calls the matching
   `ferray_ufunc::*_complex` op, returning a complex ndarray — BEFORE the
   existing `coerce_dtype(arr, "float32")` step that currently corrupts.
   `bind_unary_numeric_split!` (`absolute`/`negative`): complex arm calling
   `ferray_ufunc::abs` (→ real) / a complex negate.

3. **`ufunc.rs` — `binary_numeric_body!` / `comparison_body!`**: complex arm
   over `add_broadcast`/`subtract_broadcast`/`multiply_broadcast`/
   `divide_broadcast` (already complex-capable) and `equal_broadcast`/
   `not_equal_broadcast`; a lexicographic complex compare for `less`/`greater`/
   `less_equal`/`greater_equal`; and explicit `TypeError` raises for
   `floor_divide`/`remainder`/`mod`/`bitwise_*` on complex.

4. **`stats.rs` — `sum`/`prod`/`mean`**: complex arms; `sum`/`prod` fold via
   the existing complex `ReduceAcc`; `mean` computes a complex sum / count →
   `complex128`.

### Library prereqs (ferray-ufunc / ferray-core) — almost all already shipped

- **Already present (no work):** `Element for Complex`
  (`ferray-core/src/dtype/mod.rs`); the `*_complex` transcendental family
  (`ferray-ufunc/src/ops/complex.rs`); `WrappingArith`/`TrueDivide for Complex`
  + `power_complex` (`arithmetic.rs`, #869); `equal_broadcast` (`PartialEq`);
  `ReduceAcc for Complex` (`reductions.rs`); `ferray_ufunc::abs` complex
  magnitude (`ops/complex.rs`).
- **Net-new (small):** a complex `negative` (`-z`) — `ferray_ufunc::negative`
  is `T: Float`-bounded. Either a `negative_complex` helper in `ops/complex.rs`
  or bind via `subtract_broadcast(0+0j, z)`. The lexicographic complex compare
  for `<`/`>`/`<=`/`>=` is most cleanly a binding helper (Complex is not
  `PartialOrd`), exactly mirroring ma #874's `cmp_complex_arm!`.

### numpy main-array complex semantics captured (verified live, numpy 2.4.5)

| op | numpy complex behavior | source |
|---|---|---|
| `np.array([1+2j])` | `complex128` ndarray | live |
| `sqrt`/`exp`/`log`/`sin`/… | complex compute (complex loop) | live: `sqrt(1+2j) = 1.272+0.786j`; `loops_unary_complex.dispatch.c.src` |
| `abs(complex)` | REAL magnitude (`float64`/`float32`) | live: `abs([3+4j]) → [5.0]`, dtype float64 |
| `negative(complex)` | complex negate | live: `-[1+2j,3-1j] → [-1-2j, -3+1j]` |
| `positive(complex)` | identity | live |
| `add`/`subtract`/`multiply`/`divide` | complex compute | live |
| `equal`/`not_equal` | bool, elementwise | live |
| `less`/`greater`/`<=`/`>=` | LEXICOGRAPHIC `(real, then imag)`, does NOT raise | live: `less([1+2j,3+2j],[1+5j,3+1j]) → [True False]` |
| `floor_divide`/`remainder`/`mod` | `TypeError` ('floor_divide' not supported) | live |
| `bitwise_and`/`or`/`xor` | `TypeError` ('bitwise_and' not supported) | live |
| `sum`/`prod` | complex scalar | live: `sum → (4+1j)`, `prod → (5+5j)` |
| `mean` | complex scalar (c64→c128) | live: `mean → (2+0.5j)`; `fromnumeric.py` mean |

## Incremental build plan (smallest-highest-value first)

Each group is a separate builder→critic cycle under epic #919, mirroring the
ma-complex sequencing (#868→#869→#870→#874→#872→#873). Because the library
primitives are already shipped, every group below is dominated by binding
threading.

1. **R-2 + R-1 first (STOP corruption + accept input).** Highest value: R-2
   eliminates the live R-CODE-4 data-corruption hazard (silent imag-drop in
   `sqrt`/`exp`/`log`/`sin`/`mean`); R-1 makes complex arrays constructible from
   Python so every other group is testable end-to-end. R-2's transcendental
   fixes are the same code as R-3, so land R-3's complex dispatch branch here.
   - Blockers: #921 (R-2), #920 (R-1), #922 (R-3 dispatch).
   - Manifest: `ferray-python/src/conv.rs`, `ferray-python/src/creation.rs`,
     `ferray-python/src/ufunc.rs`, `ferray-python/src/stats.rs`,
     `ferray-python/tests/divergence_complex.py` (extend).
2. **R-4 (abs/negative/positive).** Binding arms over `ferray_ufunc::abs` +
   the tiny complex negate.
   - Blocker: #923. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-ufunc/src/ops/complex.rs` (only if a `negative_complex` helper is
     chosen over `subtract`), `ferray-python/tests/divergence_complex.py`.
3. **R-5 (arithmetic + comparison + raises).** Complex arms in
   `binary_numeric_body!`/`comparison_body!` over the already-complex-capable
   broadcast ops; the lexicographic compare helper; the `TypeError` raise
   guards for floor_divide/remainder/mod/bitwise.
   - Blocker: #924. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-python/tests/divergence_complex.py`.
4. **R-6 (sum/prod/mean).** Complex arms in `stats.rs`; `mean` → `complex128`.
   - Blocker: #925. Manifest: `ferray-python/src/stats.rs`,
     `ferray-python/tests/divergence_complex.py`.

## Verification

The gauntlet for this module (each must be green for the corresponding REQ to
move to SHIPPED in a future build commit):

- `cd ferray-python && maturin develop`
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/divergence_complex.py -q`
  (pins the main-array complex divergences; expected values from the live numpy
  oracle, R-CHAR-3)
- `cargo test -p ferray-ufunc` (the existing `*_complex` transcendental unit
  tests in `ops/complex.rs`; any new `negative_complex` test)
- `cargo test -p ferray-core` (the `ReduceAcc for Complex` reduction path)
- `cargo clippy -p ferray-python --all-targets -- -D warnings`
- `cargo clippy -p ferray-ufunc --all-targets -- -D warnings`

Per-REQ pinned pytest (added by each group's builder/critic, oracle values from
live numpy, R-CHAR-3):
- R-1: `fr.array([1+2j]).dtype == "complex128"`; `np.asarray(...)` equals numpy.
- R-2: NO `ComplexWarning` from `sqrt`/`exp`/`log`/`sin`/`mean` on complex AND
  the result equals numpy (the imag-drop guard).
- R-3: `sqrt`/`exp`/`log`/`sin`/`cos`/… of complex equal numpy elementwise.
- R-4: `abs([3+4j]) == [5.0]` (float64); `negative([1+2j]) == [-1-2j]`.
- R-5: `add`/`*`/`-`/`divide` equal numpy; `equal`/`not_equal` equal numpy;
  `less`/`greater`/`<=`/`>=` equal numpy's LEXICOGRAPHIC result;
  `floor_divide`/`bitwise_and` raise `TypeError`.
- R-6: `sum`/`prod`/`mean` of complex equal numpy complex scalars (`mean` keeps
  its imaginary part; dtype `complex128`).
