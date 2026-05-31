# ferray (main array) — float16 / half dtype support for `fr.array` + top-level ufuncs/reductions

<!--
tier: 3-component
status: draft
baseline-commit: 0a558feb
upstream-paths:
  - /home/doll/numpy-ref/numpy/_core/code_generators/generate_umath.py
  - /home/doll/numpy-ref/numpy/_core/_methods.py
  - /home/doll/numpy-ref/numpy/_core/fromnumeric.py
-->

## Summary

`fr.ndarray` is numpy's own ndarray (re-exported in `lib.rs`); every top-level
`fr.*` function takes and returns a real `numpy.ndarray`. float16 arrays
therefore already exist end-to-end at the object level — the divergence is
entirely in ferray's top-level **functions**, whose dtype dispatch macros
enumerate only the 11 "real" dtypes (`bool`, `int8/16/32/64`, `uint8/16/32/64`,
`float32/64`) and have NO `float16` arm.

This mirrors the complex epic (#919, `.design/ferray-core-complex.md`) and the
datetime epic (#940, `.design/ferray-core-datetime.md`): thread one more dtype
through the binding's dispatch macros, reusing a library substrate that — even
more than those two cases — is **already shipped and already wired**. float16 is
the *simplest* of the three: it is a plain narrow float, and the binding already
has the exact strategy it needs — **compute in float32, narrow the result back
to float16 with numpy's own `astype`** — already proven on `zeros`/`sqrt`. No
new library kernel and no new marshalling seam are required.

Verified live (numpy 2.4.5, baseline `0a558feb`). NOTE: R-1 and R-4 below are
now **SHIPPED** (#952 / #954, `array`/`asarray`/`full` float16 coercion +
`sum`/`prod`/`mean`/`std`/`var`/`min`/`max`/`ptp`/`cumsum` float16 reductions);
the text records the original baseline divergence — see the REQ status table for
the shipped state:
- **R-1 input coercion PARTLY works.** `fr.zeros(3,dtype='float16')` /
  `fr.ones(3,dtype='float16')` → `float16` array (SHIPPED via the
  `creation_dispatch!` `float16` arm). BUT
  `fr.array([1.,2,3],dtype='float16')` → `TypeError: unsupported dtype:
  "float16" (supported: bool, int8/16/32/64, uint8/16/32/64, float32/64)`;
  `fr.asarray(np.array([1,2,3],dtype='float16'))` → same; `fr.full(3,2.0,
  dtype='float16')` → same. The `array`/`asarray`/`full` copy paths dispatch
  through `match_dtype_all!` / `full_impl`, neither of which has a `float16`
  arm.
- **R-2 ufuncs PARTLY work.** `fr.sqrt(float16_array)` → `array([...],
  dtype=float16)` (SHIPPED — the unary-float body computes in f32 and narrows
  back via `unary_promote_dtypes`). The whole transcendental family routes the
  same way.
- **R-3 arithmetic REJECTS.** `fr.add(f16,f16)` →
  `TypeError: unsupported dtype for numeric op: "float16"`; same for
  `subtract`/`multiply`/`divide`/`power`/`floor_divide`/`remainder`/`maximum`/
  `minimum`. The binary numeric bodies dispatch `match_dtype_numeric!`, an
  11-arm macro with no `float16` arm and no promote-narrow wrapper.
- **R-4 reductions DIVERGE.** `fr.sum(float16_array)` →
  `TypeError: unsupported dtype for numeric op: "float16"`;
  `fr.mean(float16_array)` → `array(2.)` **dtype `float64`** where numpy returns
  `float16` (a silent dtype-widening that violates numpy's output contract,
  R-CODE-4 / R-DEV-3); `std`/`var`/`prod`/`min`/`max` likewise reject or widen.
- **R-5 comparison / manipulation** — comparisons (`fr.less`/`equal`/…)
  dispatch `match_dtype_orderable!` (no f16 arm) and reject; data-move ops
  (reshape/where/sort/…) dispatch `match_dtype_all` / `match_dtype_all_complex`
  (no f16 arm).
- **R-6 astype** — `arr.astype('float16'|'float64'|'int32')` is numpy's own
  ndarray method (works object-level) and `fr.astype` delegates straight to it
  (SHIPPED).

## The pivotal finding (the f16 Element + the pyo3-numpy half feature)

**This is overwhelmingly BINDING work — the smallest of the three narrow-dtype
epics — AND the library is already built.** The genuinely novel question (does
ferray-core need an `Element for half::f16`, like complex needed the
`Complex<f32/f64>` Element impls?) is answered NO: ferray-core *already has the
complete f16 stack, feature-gated*. The binding never even needs it, because the
proven f32-compute / numpy-narrow strategy keeps `half::f16` out of the Rust
boundary entirely.

1. **`half::f16` IS already an `Element` in ferray-core (feature-gated).**
   `impl Element for half::f16 { fn dtype() -> DType { DType::F16 } }`
   (`ferray-core/src/dtype/mod.rs`, gated `#[cfg(feature = "f16")]`), with a
   `DType::F16` variant (`enum DType`, `#[cfg(feature = "f16")] F16`),
   `size_of → 2`, `align_of::<half::f16>()`, `is_float() → true`, and `Display →
   "float16"`. Promotion (`promotion.rs`) and casting (`casting.rs`) carry F16
   under the same gate. So `ArrayD<half::f16>` is constructible *if the feature
   is on*. **No net-new library Element type is needed — unlike complex.**

2. **`DynArray` ALREADY has the `F16` variant (feature-gated).**
   `DynArray::F16(Array<half::f16, IxDyn>)` (`ferray-core/src/dynarray.rs`,
   `#[cfg(feature = "f16")]`), with `dtype() → DType::F16`, a working
   `DynArray::zeros(DType::F16, …) → Self::F16(Array::zeros(dim)?)` arm, and the
   `impl_from_array_dyn!(half::f16, F16)` ingress. (Note: `DynArray::astype`
   *rejects* f16 with `"DynArray::astype does not yet support f16"` —
   `dynarray.rs` — but the binding does not use that path; astype is numpy's own
   method.) This is *ahead* of where complex started.

3. **ferray-ufunc ALREADY has extensive f16 support (feature-gated).** Ten
   modules under `ferray-ufunc/src/` carry `#[cfg(feature = "f16")]` f16 paths
   (`ops/floatintrinsic.rs`, `ops/trig.rs`, `promoted.rs`, `dispatch.rs`,
   `ops/special.rs`, `ops/arithmetic.rs`, `helpers.rs`, `ops/rounding.rs`,
   `ops/explog.rs`, `lib.rs`). The `PromoteFloat` machinery already maps
   `bool/int8/uint8 → float16` (`ferray-ufunc/tests/divergence_unary_promote.rs`
   contract table). So even native-half compute is available behind the gate.

4. **pyo3-numpy 0.28 HAS a `half` feature that impls `Element for half::f16`,
   but it is NOT enabled in the dependency tree.** The installed numpy crate is
   `numpy 0.28.0` (`Cargo.lock`), which carries
   `impl_element_scalar!(f16 => NPY_HALF)` under `#[cfg(feature = "half")]`
   (`~/.cargo/.../numpy-0.28.0/src/dtype.rs:607-608`) and an optional `half`
   dependency (`numpy-0.28.0/Cargo.toml [dependencies.half]`). **That `half`
   feature is NOT turned on:** `ferray-python/Cargo.toml` depends on `numpy =
   { workspace = true }` with no features, `ferray-numpy-interop`'s own `f16`
   feature only enables `["dep:half", "ferray-core/f16"]` (NOT
   `numpy/half`), and neither `ferray-python` nor `ferray-numpy-interop`
   forwards the `f16` feature at all. So today `PyReadonlyArrayDyn<half::f16>` /
   `half::f16::into_pyarray` are NOT available at the boundary — exactly the
   situation the creation `float16` arm documents ("the installed pyo3-numpy
   build has no `NumpyElement` for `half::f16`", `creation.rs`).

   **The decision this doc locks (R-DEV-7):** do NOT enable the pyo3-numpy
   `half` feature / the `ferray-core/f16` cascade. The proven, already-shipped
   strategy is to **compute in float32 inside ferray-core, then narrow the
   result to float16 with numpy's own `astype("float16")`** — keeping
   `half::f16` entirely out of the Rust↔Python boundary. This is exactly what
   `unary_promote_dtypes` (`conv.rs`) + the creation `float16` arm
   (`creation.rs`) already do, and it is *numpy's own internal contract* —
   numpy computes float16 ufuncs by upcasting to float32 and rounding the result
   back to float16 (see "compute contract" below), so f32-compute + round-to-f16
   is bit-for-bit numpy-correct, not an approximation.

## The f16 compute contract (promote-to-f32-round, NOT native half)

numpy float16 ufuncs and reductions compute in **float32** internally and round
the result back to **float16**: the result *dtype* is float16, but the
arithmetic is done at f32 width then narrowed. Verified live (numpy 2.4.5):

| op | numpy f16 behavior | source |
|---|---|---|
| `np.sqrt(np.float16(2))` | `float16` (`1.414`) | live |
| `np.float16(1)+np.float16(2)` | `float16` | live |
| `np.sum(f16[1,2,3,65505])` | `float16` (`6.55e+04`, overflows to inf-region exactly like numpy) | live |
| `np.mean(f16)` | `float16` | live |
| `np.std(f16)` / `np.var(f16)` | `float16` (computed via f32 `square`, then narrowed) | live; `_core/_methods.py:_var` |
| `np.prod(f16)` | `float16` | live |
| `np.min(f16)` / `np.max(f16)` | `float16` | live |
| `f16 + f16` | `float16` | live (NEP-50) |
| `f16 + f32` | `float32` | live (NEP-50) |
| `f16 + python-float` | `float16` (weak scalar) | live (NEP-50) |
| `f16 + int32-array` | `float64` | live (NEP-50 array-int promotion) |
| `arr.astype('float64')` | `float64` (numpy ndarray method) | live |

The contract: **float16 in → float16 out**, computed via f32 promotion + round.
The binding realizes this by (a) coercing the f16 input to f32, (b) running the
existing `T: Float` ferray-core/ferray-ufunc kernel at f32, (c) narrowing the
result with numpy's `astype("float16")` — which performs the exact
round-to-nearest-even f32→f16 numpy uses. NEP-50 promotion (`result_type`) is
delegated to numpy at the boundary, exactly as the binary bodies already
delegate it for the real dtypes.

## Requirements

- REQ-1 (R-1 input coercion): `fr.array([1.,2,3],dtype='float16')`,
  `fr.asarray(np.array(...,dtype='float16'))`, and `fr.full(shape, v,
  dtype='float16')` build a `float16` ndarray (preserving shape) instead of
  raising `TypeError`. (`fr.zeros`/`ones`/`empty`/`*_like(dtype='float16')` are
  already SHIPPED via the creation `float16` arm.)
- REQ-2 (R-2 ufunc dispatch): the unary transcendental family
  (`sqrt`/`exp`/`log`/`log2`/`log10`/`expm1`/`log1p`/`sin`/`cos`/`tan`/`sinh`/
  `cosh`/`tanh`/`arcsin`/…/`cbrt`/`rint`/…) over float16 returns a `float16`
  result via the f32-compute / f16-narrow path (already SHIPPED through
  `unary_promote_dtypes`).
- REQ-3 (R-3 arithmetic + NEP-50 promotion): `fr.add`/`subtract`/`multiply`/
  `divide`/`power`/`floor_divide`/`remainder`/`maximum`/`minimum` over float16
  compute (f32-compute, f16-narrow) and honor NEP-50 promotion (f16+f16→f16,
  f16+f32→f32, f16+python-float→f16, f16+int-array→f64).
- REQ-4 (R-4 reductions — STOP the silent widening): `fr.sum`/`fr.prod`/
  `fr.mean`/`fr.min`/`fr.max`/`fr.std`/`fr.var` over float16 return a `float16`
  result (numpy's contract), NOT the current `float64` widening
  (`fr.mean(f16)→array(2.)` dtype float64 is an active R-DEV-3 / R-CODE-4
  divergence).
- REQ-5 (R-5 comparison + manipulation + dtype-preservation): `fr.less`/
  `greater`/`less_equal`/`greater_equal`/`equal`/`not_equal` over float16
  compute bool; data-move / view ops (`reshape`/`transpose`/`concatenate`/
  `where`/`sort`/`flip`/`roll`/…) preserve the float16 dtype through to the
  result instead of rejecting it.
- REQ-6 (R-6 astype): `arr.astype('float16')`, `float16↔float32↔float64`, and
  `float16↔int` astype preserve numpy's dtype contract (numpy's own ndarray
  method; `fr.astype` delegates to it — already SHIPPED).

## Acceptance criteria

- AC-1 (REQ-1): pytest `fr.array([1.,2,3],dtype='float16').dtype == 'float16'`;
  `np.asarray(fr.asarray(np.array([1,2,3],dtype='float16')))` equals numpy;
  `fr.full(3,2.0,dtype='float16').dtype == 'float16'`.
- AC-2 (REQ-2): `fr.sqrt(np.array([2.0],dtype='float16'))` equals
  `np.sqrt(...)` with dtype `float16` (oracle-generated, R-CHAR-3).
- AC-3 (REQ-3): `fr.add`/`*`/`-`/`divide` of two float16 arrays equal numpy with
  dtype `float16`; `fr.add(f16,f32).dtype == 'float32'`;
  `fr.add(f16,1.0).dtype == 'float16'`; `fr.add(f16,int32_arr).dtype ==
  'float64'`.
- AC-4 (REQ-4): `fr.sum(f16)`/`fr.mean(f16)`/`fr.prod(f16)`/`fr.min(f16)`/
  `fr.max(f16)`/`fr.std(f16)`/`fr.var(f16)` equal numpy elementwise AND carry
  dtype `float16` (NOT `float64`).
- AC-5 (REQ-5): `fr.less(f16,f16)` equals numpy (bool);
  `fr.reshape(f16_arr,(…)).dtype == 'float16'`; `fr.where(mask,f16,f16).dtype ==
  'float16'`; `fr.sort(f16).dtype == 'float16'`.
- AC-6 (REQ-6): `fr.astype(f16_arr,'float64').dtype == 'float64'`;
  `np.asarray(fr.array(...,dtype='float16')).astype('int32')` equals numpy.

## REQ status table

`fr.zeros`/`ones`/`empty`/`*_like(dtype='float16')` (RC-F16, `creation.rs`),
the unary-float family (`unary_promote_dtypes`, `ufunc.rs`/`conv.rs`), and
`fr.astype` (`aliases.rs`) are already SHIPPED for float16. The
`array`/`asarray`/`full` coercion, all binary arithmetic, all reductions, and
the comparison/manipulation dispatch are real-only and reject or widen float16 —
divergences reproduce live at `0a558feb`. Two states only (R-DEFER-2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (input coercion) | SHIPPED | The `is_float16_dtype` branch in `pub fn array` (`creation.rs`) returns `arr.call_method0("copy")` on the numpy-parsed float16 buffer (numpy owns the f32→f16 round-to-nearest-even narrow via the `np.asarray(obj,'float16')` already run), and `is_float16_dtype` in `full_impl` (`creation.rs`) delegates `np.full(shape, fill, 'float16')` — both AHEAD of the real-only `match_dtype_all!`, mirroring the SHIPPED `creation_dispatch!` float16 arm. Non-test consumers: `pub fn array`/`pub fn asarray` (asarray delegates to array) + `pub fn full` (→ `full_impl`), all registered top-level in `lib.rs`. numpy (live 2.4.4): `np.array([1.,2,3],dtype='float16').dtype == float16`, `np.full(3,1.5,dtype='float16').dtype == float16`. Pinned green: `tests/test_expansion_float16.py::test_array_float16_dtype_supported`, `::test_asarray_of_numpy_float16_array`, `::test_full_float16_dtype_supported`, `::test_array_float16_is_a_copy_not_a_view`. (`fr.zeros`/`ones`/`empty`/`*_like(dtype='float16')` were already SHIPPED via the `creation_dispatch!` float16 arm.) |
| REQ-2 (ufunc dispatch) | SHIPPED | `macro_rules! unary_float_body` (`ufunc.rs`) calls `unary_promote_dtypes(py, &arr, in_dt)` (`pub fn unary_promote_dtypes` in `conv.rs`) which, for a `"float16"` input, returns `(compute="float32", out="float16")`; the body coerces the input to f32 (`coerce_dtype`), runs the existing `T: Float` ferray-ufunc kernel at f32, then `coerce_dtype(py, &result, "float16")` narrows back — numpy's own round-to-f16. Non-test consumers: `pub fn sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/… in `ufunc.rs` (the `bind_unary_float!`-generated `#[pyfunction]`s, registered top-level in `lib.rs`). numpy: `np.sqrt(np.float16(2))` → float16 (`generate_umath.py:966` sqrt float loop). Verification (live, this session): `fr.sqrt(np.array([2.0],dtype='float16'))` → `array([1.414], dtype=float16)`. |
| REQ-3 (arithmetic + NEP-50) | NOT-STARTED | open prereq blocker #953. `macro_rules! binary_numeric_body` / `binary_numeric_split_body` (`ufunc.rs`) dispatch `match_dtype_numeric!` (`conv.rs`, 10 real numeric arms, NO float16) on `binary_result_dtype(py,&a,&b)` — when both operands are float16 the common dtype is `"float16"` and the macro hits its `other =>` `TypeError: unsupported dtype for numeric op: "float16"`. Consumers `pub fn add`/`subtract`/`multiply`/`divide`/`power`/`floor_divide`/`remainder`/`maximum`/`minimum` (`ufunc.rs`) all reject f16 (live, numpy 2.4.5: `np.add(f16,f16).dtype == float16`, `f16+f32 → float32`, `f16+int → float64`). Fix = a promote-narrow wrapper on the numeric bodies (compute the f16 case at f32, narrow result via `astype`), mirroring `binary_float_promote_body`'s existing `unary_promote_dtypes` use; NEP-50 stays delegated to numpy's `result_type`. |
| REQ-4 (reductions — STOP widening) | SHIPPED | `fn f16_reduce` (`stats.rs`) delegates a float16 reduction to numpy's own top-level function on the ORIGINAL float16 array, so numpy owns the exact f32-compute + f16-narrow and the result carries numpy's float16 dtype (same detect-and-delegate pattern as the datetime reductions `crate::datetime::time_reduce`). The `is_float16_dtype` branch is wired into `pub fn sum`/`prod`/`min`/`max`/`ptp`/`cumsum` ahead of the real-only `match_dtype_numeric!`/`match_dtype_orderable!`, and into `pub fn mean`/`var`/`std` AHEAD of the `coerce_dtype(arr,"float64")` step — eliminating the silent widening (`fr.mean(f16)` was `float64`; now `float16`, R-DEV-3 / R-CODE-4). Non-test consumers: the `sum`/`prod`/`mean`/`var`/`std`/`min`/`max`/`ptp`/`cumsum` `#[pyfunction]`s in `stats.rs`, registered top-level in `lib.rs`. numpy (live 2.4.4): `np.sum/prod/mean/std/var/min/max/ptp/cumsum(f16).dtype == float16`; `np.sum([60000,60000]f16) == inf`. Pinned green: `tests/test_expansion_float16.py::test_reduction_float16_returns_float16` (9-op parametrize), `::test_mean_float16_not_widened_to_float64`, `::test_std_float16_not_widened_to_float64`, `::test_var_float16_not_widened_to_float64`, `::test_sum_float16_overflow_to_inf`, `::test_sum_float16_axis0`, `::test_var_float16_ddof1`. |
| REQ-5 (comparison + manipulation) | NOT-STARTED | open prereq blocker #955. Comparisons `pub fn less`/`greater`/`less_equal`/`greater_equal`/`equal`/`not_equal` (`ufunc.rs`) dispatch `match_dtype_orderable!` (`conv.rs`, no float16 arm) → `TypeError: unsupported dtype for ordering op: "float16"`. Data-move / view ops (`reshape`/`transpose`/`concatenate`/`stack`/`where`/`sort`/`flip`/`roll`/`repeat`/`tile`/…) dispatch `match_dtype_all!` / `match_dtype_all_complex!` (`conv.rs`, no float16 arm) → `TypeError: unsupported dtype`, so a float16 array does not round-trip its dtype through them. numpy 2.4.5 live: `np.less(f16,f16)` → bool; `np.reshape(f16,(…)).dtype == float16`. Fix = float16 arms in the comparison/manipulation macros: comparisons via the f32-promote path (bool output is width-independent), data-move ops via the same astype-narrow round-trip (or, simplest, view-as-f16 marshalling once the dtype set is extended). |
| REQ-6 (astype) | SHIPPED | `pub fn astype` (`aliases.rs`) delegates straight to numpy's own ndarray method: `arr.call_method("astype", (dtype,), Some(&kwargs))` with `copy=`. Since `fr.ndarray` IS `numpy.ndarray` (`lib.rs`), `float16↔float32↔float64` and `float16↔int` conversions are numpy's own narrowing/widening — no ferray dtype dispatch involved, so no float16 gap. Non-test consumer: `pub fn astype` registered top-level in `lib.rs` (`m.add_function(wrap_pyfunction!(aliases::astype, m)?)`). numpy: `np.array([1,2,3],dtype='float16').astype('float64').dtype == float64` (live, this session). Verification: `fr.astype(np.array([1.,2,3],dtype='float16'),'float64').dtype == 'float64'`. |

## Architecture

### There is NO `fr.ndarray` pyclass — the "main array" is numpy's ndarray

Same as the complex (#919) and datetime (#940) cases: `fr.ndarray` re-exports
`numpy.ndarray` (`lib.rs`); every `fr.*` ufunc/creation/reduction is a
`#[pyfunction]` taking `&Bound<PyAny>` and returning a real `numpy.ndarray`.
Consequence: numpy's OWN `a16 + a16`, `a16.sum()` (method form), `a16.astype(...)`
all work on a float16 ndarray because they are numpy, not ferray. ferray's job
is purely the top-level FUNCTION surface.

### f32-compute / f16-narrow is the transport — NOT the pyo3-numpy half feature

A float16 ndarray is 16-bit IEEE binary16. pyo3-numpy 0.28 *could* carry it
natively (its `half` feature impls `Element for half::f16`,
`numpy-0.28.0/src/dtype.rs:607`), but that feature is OFF in this tree
(`ferray-python/Cargo.toml` numpy dep has no features; `ferray-numpy-interop`'s
`f16` feature enables only `dep:half` + `ferray-core/f16`, not `numpy/half`).
The binding therefore keeps `half::f16` out of the boundary entirely and uses
the strategy already proven on `zeros`/`sqrt`:
- ingress: `np.asarray(obj, dtype='float32')` (or `coerce_dtype(py, &arr,
  "float32")`) lifts the f16 input to f32 so the existing `AsFerray`/
  `PyReadonlyArrayDyn<f32>` path applies.
- compute: the existing `T: Float`-bounded ferray-core / ferray-ufunc kernel
  runs at f32 (numpy itself computes f16 ufuncs at f32 width — see the compute
  contract — so this is bit-for-bit correct, not an approximation).
- egress: `result.call_method1("astype", ("float16",))` narrows the f32 result
  to float16 with numpy's round-to-nearest-even, preserving numpy's exact f16
  dtype + value contract (R-CODE-4 / R-DEV-3).

This is the SAME `unary_promote_dtypes` (`conv.rs`) + creation `float16`-arm
mechanism already in `main`; every group below reuses it. The
`DynMarshal`/`match_dtype_all_complex!` complex seam is NOT needed — float16
never crosses the boundary as `half::f16`.

### Where float16 must be threaded (binding)

1. **`creation.rs` `match_dtype_all!` path (`array`/`asarray`) + `full_impl`**
   (REQ-1): branch on a `"float16"` dtype ahead of the macro and route through
   `np.asarray(obj, dtype='float16')` (numpy owns the parse + f32→f16 narrow),
   mirroring the SHIPPED `creation_dispatch!` `float16` arm. The `*_like` /
   `zeros`/`ones`/`empty` paths are already done.
2. **`ufunc.rs` `binary_numeric_body!` / `binary_numeric_split_body!`** (REQ-3):
   wrap the float16 case in the promote-narrow pattern (compute at f32 via
   `unary_promote_dtypes`-style `(compute, out)`, narrow the result with
   `astype`), exactly as `binary_float_promote_body!` already does for the
   float-only ops; NEP-50 promotion stays in numpy's `result_type`
   (`binary_result_dtype`, `conv.rs`).
3. **`ufunc.rs` comparison bodies** (REQ-5 compares): a float16 arm computing
   the compare at f32 (bool output is width-independent, so no narrowing needed).
4. **`stats.rs` reductions** (REQ-4): float16 arms for
   `sum`/`prod`/`mean`/`min`/`max`/`std`/`var`/`cumsum`/`cumprod` that compute
   at f32 and narrow the *reduced* result to float16 — and stop coercing
   `mean`/`std`/`var` outputs to float64.
5. **`manipulation.rs` / `match_dtype_all` data-move ops** (REQ-5 manipulation):
   a float16 arm that round-trips the dtype (astype-narrow, or — if the team
   later elects to enable `numpy/half` — a native `half::f16` view arm; this doc
   locks the astype-narrow choice to avoid the feature cascade).

### Library prereqs (ferray-core / ferray-ufunc) — ALL already present

- **Already present (no work, even if unused by the chosen strategy):**
  `Element for half::f16` + `DType::F16` (`dtype/mod.rs`); `DynArray::F16` +
  `zeros`/`from`/`dtype` (`dynarray.rs`); the gated f16 paths across ten
  ferray-ufunc modules; the `PromoteFloat` `int → float16` machinery
  (`tests/divergence_unary_promote.rs`). All behind `#[cfg(feature = "f16")]`.
- **Net-new (library): NONE.** Unlike datetime (four timedelta numeric kernels)
  and complex (a complex `negative`), float16 needs no net-new library kernel:
  every op composes the existing `T: Float` f32 kernel + numpy's f16 narrow.

## Incremental build plan (smallest-highest-value first)

Each group is a separate builder→critic cycle under epic #951, mirroring the
complex (#920→…) and datetime (#941→…) sequencing. Because the transport is
already shipped (`unary_promote_dtypes` + the creation `float16` arm), every
group is pure binding threading reusing that one mechanism.

1. **R-4 + R-1 first (STOP the silent widening + accept input).** Highest value:
   R-4 eliminates the live `fr.mean(f16)→float64` widening (an R-DEV-3 output-
   contract divergence — numpy returns float16); R-1 makes float16 arrays
   constructible from `fr.array`/`fr.asarray`/`fr.full` so every other group is
   testable end-to-end.
   - Blockers: #954 (R-4), #952 (R-1).
   - Manifest: `ferray-python/src/stats.rs`, `ferray-python/src/creation.rs`,
     `ferray-python/src/conv.rs`, `ferray-python/tests/divergence_float16.py`
     (new).
2. **R-3 (arithmetic + NEP-50).** Promote-narrow wrapper on the binary numeric
   bodies.
   - Blocker: #953. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-python/tests/divergence_float16.py`.
3. **R-5 (comparison + manipulation + dtype-preservation).** float16 arms in the
   comparison + data-move dispatch macros.
   - Blocker: #955. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-python/src/manipulation.rs`, `ferray-python/src/conv.rs`,
     `ferray-python/tests/divergence_float16.py`.

(REQ-2 ufunc dispatch and REQ-6 astype are already SHIPPED — no builder.)

## Verification

The gauntlet for this module (each must be green for the corresponding REQ to
move to SHIPPED in a future build commit):

- `cd ferray-python && maturin develop`
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/divergence_float16.py -q`
  (pins the main-array float16 divergences; expected values from the live numpy
  oracle, R-CHAR-3)
- `cargo clippy -p ferray-python --all-targets -- -D warnings`

Per-REQ pinned pytest (added by each group's builder/critic, oracle values from
live numpy, R-CHAR-3):
- R-1: `fr.array([1.,2,3],dtype='float16').dtype == 'float16'`;
  `fr.asarray(np.array(...,'float16'))` equals numpy; `fr.full(3,2.0,
  dtype='float16').dtype == 'float16'`.
- R-2 (SHIPPED): `fr.sqrt(f16)` equals numpy, dtype float16.
- R-3: `fr.add`/`*`/`-`/`divide` of float16 equal numpy with dtype float16;
  `fr.add(f16,f32).dtype == 'float32'`; `fr.add(f16,int).dtype == 'float64'`.
- R-4: `fr.sum`/`mean`/`prod`/`min`/`max`/`std`/`var(f16)` equal numpy AND carry
  dtype float16 (NOT float64) — the silent-widening guard.
- R-5: `fr.less(f16,f16)` equals numpy; `fr.reshape`/`where`/`sort` preserve
  float16 dtype.
- R-6 (SHIPPED): `fr.astype(f16,'float64').dtype == 'float64'`.
