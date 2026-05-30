# ferray.ma — Complex-dtype MaskedArray support

<!--
tier: 3-component
status: draft
baseline-commit: afcadb63
upstream-paths:
  - /home/doll/numpy-ref/numpy/ma/core.py
-->

## Summary

`fr.ma.MaskedArray` (PyO3 `PyMaskedArray`) wraps `ferray_ma::MaskedArray<T,
IxDyn>` via a runtime-typed `DynMa` enum that, after #853, covers the **11 real
numpy dtypes only** (bool, i8/16/32/64, u8/16/32/64, f32/f64). `build_dynma`
(in `ma.rs`) raises `TypeError` on any complex / structured / datetime input.
numpy.ma fully supports `complex64`/`complex128` masked arrays — construction,
dtype, fill (`1e20+0j`), arithmetic, `==`/`!=`, `.real`/`.imag`/`.conjugate()`,
`abs()`, and `sum`/`prod`/`mean`/`min`/`max` — verified live (numpy 2.4.5):
`np.ma.array([1+2j,3-1j],mask=[0,1]).dtype` → `complex128`, while
`fr.ma.array([1+2j])` → `TypeError`. This doc scopes adding two complex
`DynMa` variants and threading them through every dispatch site, matching
numpy.ma's complex semantics (and its `TypeError` guards for ordering / `//` /
`%` / bitwise, which are undefined for complex).

This is the design contract for the gap pinned by
`test_KNOWN_complex_masked_array_supported_by_numpy`
(`tests/test_divergence_ma_dtype_audit.py`).

## The pivotal finding (library readiness)

The library substrate is **mostly ready**; the one genuine library gap is
**complex arithmetic in `ferray-ufunc`**.

1. **`Complex<f32>`/`Complex<f64>` ARE `Element`.** `impl Element for
   Complex<f32>` / `impl Element for Complex<f64>` in `ferray_core::dtype`
   (`mod.rs`), with `DType::Complex32` / `DType::Complex64`. So
   `ferray_ma::MaskedArray<Complex<f64>, IxDyn>` is constructible — the
   `MaskedArray<T: Element, D>` bound (`masked_array.rs`) imposes nothing
   beyond `Element`.

2. **Complex unary `real`/`imag`/`conj`/`conjugate`/`abs`/`angle` ALREADY
   EXIST** in `ferray_ufunc::ops::complex` (`complex.rs`), each bounded
   `T: Element + Float, Complex<T>: Element`. `abs` returns a **real**
   `Array<T, D>` (magnitude), matching numpy. These compute today.

3. **Complex `equal`/`not_equal` ALREADY work generically.** `pub fn equal` /
   `pub fn not_equal` (`comparison.rs`) are bounded `T: Element + PartialEq +
   Copy` — `Complex` satisfies all three.

4. **Complex `sum`/`prod` accumulators ALREADY exist.** `impl ReduceAcc for
   Complex<f32>` / `Complex<f64>` (`reductions.rs`, `Acc = Self` — "Complex
   stays itself — numpy never promotes a complex reduction"). The binding's
   `ma_sum_acc` / `ma_prod_acc` are keyed on `ReduceAcc`, so they monomorphize
   for complex once the binding-local `AccIdentity` trait gains complex arms.

5. **THE GAP — complex `add`/`subtract`/`multiply`/`divide`/`power` do NOT
   compile for Complex.** `pub fn add`/`subtract`/`multiply` (`arithmetic.rs`)
   are bounded `T: Element + WrappingArith`; `WrappingArith` is impl'd via
   `impl_wrapping_arith_int!(i8..u128)` and `impl_wrapping_arith_float!(f32,
   f64)` (plus `half::f16`) — **no `impl WrappingArith for Complex`**.
   `pub fn divide` is bounded `T: TrueDivide` (impl'd for f64/f32/ints only —
   **no Complex**). `pub fn power` is bounded `T: Element + Float` — `num_traits::Float`
   is **not** impl'd for `Complex`. So complex `+`,`-`,`*`,`/`,`**` are a real
   **ferray-ufunc library prerequisite**, NOT pure binding work.

**Magnitude verdict:** the work is **~70% binding (thread two variants), ~30%
library (one ferray-ufunc complex-arithmetic prereq).** Construction, dtype,
fill, real/imag/conj/abs, eq/ne, and sum/prod are binding-only — the library
already supports them. Only complex arithmetic and the complex-aware `mean` /
lexicographic `min`/`max` require touching library/binding compute helpers.

## Requirements

- REQ-1 (construction + dtype + filled): `fr.ma.array([1+2j,3-1j], mask=[0,1])`
  builds a complex128 masked array; `.dtype == "complex128"` (c64 for
  complex64); `.filled()` substitutes the complex default fill `1e20+0j`;
  `__array__` egresses the complex data verbatim.
- REQ-2 (`.real`/`.imag`/`.conjugate`/`.conj`/`abs`): return real masked
  arrays for `.real`/`.imag`/`abs` (magnitude), a complex masked array for
  `.conjugate`/`.conj`; mask propagated.
- REQ-3 (complex arithmetic `+`,`-`,`*`,`/`,`**`): compute over the complex
  variant via `ferray-ufunc` (needs the complex-arithmetic prereq); mask =
  operand union; `/` masks divisor-zero / non-finite per numpy.ma's domained op.
- REQ-4 (`==`/`!=` compute; `<`,`>`,`<=`,`>=` raise `TypeError`): complex
  `==`/`!=` compute (with the masked-position override); the four ordering ops
  raise `TypeError` like numpy (complex is not ordered).
- REQ-5 (`//`,`%`,bitwise raise `TypeError`): complex `//`,`%`,`&`,`|`,`^`,
  `<<`,`>>` raise `TypeError` matching numpy's "ufunc not supported" message.
- REQ-6 (complex reductions `sum`/`prod`/`mean`/`min`/`max`): `sum`/`prod`
  yield a complex scalar; `mean` yields a complex scalar (no real collapse);
  `min`/`max` use numpy's lexicographic complex ordering.

## Acceptance criteria

- AC-1 (REQ-1): pytest `fr.ma.array([1+2j,3-1j],mask=[0,1]).dtype == "complex128"`;
  `.filled() == np.ma.array([1+2j,3-1j],mask=[0,1]).filled()` (the masked slot
  becomes `1e20+0j`); `np.asarray(fr_ma) == np.asarray(np_ma)`.
- AC-2 (REQ-2): `fr.ma.array([3+4j],mask=[0]).real/.imag/abs()` equal the numpy
  values (`3.0`/`4.0`/`5.0`) and the result dtype is real (float64);
  `.conjugate()` equals `np.ma.array([3-4j])`.
- AC-3 (REQ-3): `(c_ma + c_ma)`, `* `, `- `, `/ 2`, `** 2` equal the numpy.ma
  results elementwise (data + mask) for masked complex inputs.
- AC-4 (REQ-4): `c_ma == c_ma` matches numpy.ma (masked-position override);
  `c_ma < c_ma` raises `TypeError`.
- AC-5 (REQ-5): `c_ma // 2`, `c_ma % 2`, `c_ma & 1` each raise `TypeError` with
  numpy's "not supported" wording.
- AC-6 (REQ-6): `c_ma.sum()`, `.prod()`, `.mean()`, `.min()`, `.max()` equal the
  numpy.ma complex scalars (the mean keeps its imaginary part).

## REQ status table

All six REQs are **NOT-STARTED**: `build_dynma` (`ma.rs`) raises `TypeError` for
every complex input today (`match dt.as_str() { ... other => Err(PyTypeError ...
"complex / structured are a follow-up") }`), so no complex masked array can be
constructed, and no downstream method can observe a complex variant.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (construct/dtype/filled) | NOT-STARTED | open prereq blocker #868. `build_dynma` in `ma.rs` rejects complex (`other => Err(PyTypeError::new_err("... complex / structured are a follow-up"))`); `enum DynMa` has 11 real variants only; `dtype_name`, `match_ma!`, `impl_wrap_dyn!`, `AccIdentity`, and the complex fill-default (`1e20+0j`) in `default_fill_value` (`ferray-ma/src/masked_array.rs`, complex falls to `T::zero()`) must all gain complex arms. Substrate ready: `Complex<f64>: Element` (`ferray_core::dtype::mod.rs`). |
| REQ-2 (real/imag/conj/abs) | NOT-STARTED | open prereq blocker #870. `ferray_ufunc::ops::complex` (`complex.rs`) ALREADY provides `pub fn real`/`imag`/`conj`/`conjugate`/`abs`/`angle` (bounded `T: Element + Float, Complex<T>: Element`); `carry_unary_mask` in `ma.rs` is generic over `T: Element + WrapDyn`. Binding-only: add `#[pymethods]`/free fns dispatching the new complex `DynMa` arm; blocked behind REQ-1's variants. |
| REQ-3 (complex `+`,`-`,`*`,`/`,`**`) | NOT-STARTED | open prereq blocker #869 (LIBRARY prereq). `pub fn add`/`subtract`/`multiply` (`arithmetic.rs`) bound `T: Element + WrappingArith`; `WrappingArith` impl'd only for `i8..u128`/`f32`/`f64`/`f16` (`impl_wrapping_arith_int!`/`impl_wrapping_arith_float!`) — NO Complex. `pub fn divide` bound `T: TrueDivide` (f64/f32/ints only). `pub fn power` bound `T: Element + Float` — Float not impl for Complex. Needs `WrappingArith`+`TrueDivide` for Complex + a complex `power` arm, then the complex arms in `compute_binary`'s match. |
| REQ-4 (`==`/`!=` compute; ordering raises) | NOT-STARTED | open prereq blocker #871. `pub fn equal`/`not_equal` (`comparison.rs`) bound `T: Element + PartialEq + Copy` → Complex OK (eq/ne compute once the complex arm is added to `compute_compare`'s `cmp_arm!` match). `pub fn less`/`less_equal`/`greater`/`greater_equal` bound `T: ... + PartialOrd` and Complex is NOT `PartialOrd` → `compare_op` must guard the four ordering ops to a `TypeError` for complex (numpy raises). |
| REQ-5 (`//`,`%`,bitwise raise) | NOT-STARTED | open prereq blocker #872. `compute_binary`'s `int_arm`/`float_arm` would compute `//`/`%` — for complex they must raise (numpy: `ufunc 'floor_divide'/'remainder' not supported`). `bitwise_op` in `ma.rs` already raises on a `float16|float32|float64` common dtype (`if matches!(result_dt.as_str(), "float16"|"float32"|"float64")`) — the guard must extend to `complex64`/`complex128`. |
| REQ-6 (sum/prod/mean/min/max) | NOT-STARTED | open prereq blocker #873. `ma_sum_acc`/`ma_prod_acc` keyed on `ReduceAcc` which IS impl'd for `Complex<f32>`/`Complex<f64>` (`reductions.rs`, `Acc=Self`) → sum/prod work once binding-local `AccIdentity` (`ma.rs`, currently i64/u64/f32/f64) gains complex arms. `mean`/`var`/`std`/`median` route through `to_f64_ma` (`ma.rs`) which collapses each element via `dyn_to_f64` → DROPS the imaginary part; `mean` needs a complex path. `ma_extremum` needs lexicographic complex compare (Complex not `PartialOrd`; numpy complex min/max is lexicographic, verified live 2.4.5). |

## Architecture

### Where complex must be threaded (binding, `ferray-python/src/ma.rs`)

The dispatch substrate is a runtime-typed enum + a macro that monomorphizes one
generic body per variant. Adding complex means extending each of these sites:

1. **`enum DynMa`** — add `Complex32(RustMa<Complex<f32>, IxDyn>)` and
   `Complex64(RustMa<Complex<f64>, IxDyn>)` (mirrors the 11 real variants).
2. **`match_ma!` macro** — add two arms binding `T = Complex<f32>` /
   `Complex<f64>`. Every `match_ma!`-bodied method (`shape`/`ndim`/`size`/
   `count`/`mask_bits`/`filled`/`compressed`/`data_pyarray`/`__repr__` …) then
   covers complex automatically — but ONLY if the body's ops are
   complex-defined (e.g. `filled`/`compressed` are; the float-only `to_f64_ma`
   funnel is not — see below).
3. **`DynMa::dtype_name`** — add `Complex32 => "complex64"`,
   `Complex64 => "complex128"` (numpy's `.dtype.name`).
4. **`impl_wrap_dyn!`** — add `Complex<f32> => Complex32, Complex<f64> =>
   Complex64` so `carry_unary_mask`/subscript-gather bodies re-tag the complex
   result (REQ-2).
5. **`build_dynma`** — add `"complex64"|"c8" => build!(Complex<f32>,
   Complex32)`, `"complex128"|"c16" => build!(Complex<f64>, Complex64)`
   (replacing the `other => TypeError` fallthrough for these two). This is the
   single gate that unblocks REQ-1; `dynma_from_input` flows through it
   unchanged.
6. **`AccIdentity`** (`ma.rs`) — add `Complex<f32> => (0,0), (1,0)` and
   `Complex<f64> => …`; `ReduceAcc::Acc` for complex is `Self`, so
   `ma_sum_acc`/`ma_prod_acc` then fold complex (REQ-6).
7. **`coerce_scalar<T>`** (`ma.rs`, the `filled(fill_value)` path) — works for
   any `T: NpElement + Copy`; needs `Complex` in `NpElement` (the interop
   boundary egress set) to round-trip a complex fill.

### Where complex must RAISE (matching numpy's "no loop" TypeError)

- **`compare_op`** — for a complex receiver, the four ordering ops
  (`Lt`/`Le`/`Gt`/`Ge`) must return `TypeError` (Complex is not `PartialOrd`,
  and numpy raises). `compute_compare`'s `cmp_arm!` macro must add complex arms
  ONLY for the `Eq`/`Ne` paths (`equal`/`not_equal` are `PartialEq`-bounded),
  with ordering arms guarded out before the macro is reached.
- **`compute_binary`** — complex `FloorDiv`/`Mod` must raise `TypeError`
  (numpy: `ufunc 'floor_divide'/'remainder' not supported`). Only
  `Add`/`Sub`/`Mul`/`TrueDiv`/`Pow` get a complex arm.
- **`bitwise_op`** — the existing float-only guard (`if matches!(result_dt
  .as_str(), "float16"|"float32"|"float64")`) must extend to
  `"complex64"|"complex128"` (numpy raises `ufunc 'bitwise_and' not
  supported`).

### Library prereq (ferray-ufunc / ferray-core)

- `ferray-ufunc/src/ops/arithmetic.rs`: add `impl WrappingArith for
  Complex<f32>`/`Complex<f64>` (`wadd`/`wsub`/`wmul` = native complex `+`/`-`/
  `*` — no "wrapping" needed for floats-of-complex), add `impl TrueDivide for
  Complex<...>` (`Output = Self`, native `/`), and a complex `power` path
  (`Complex::powc`). These unblock REQ-3. (The SIMD fast paths in `add`/
  `divide` are `TypeId`-gated to f32/f64 and fall through to the scalar
  `binary_elementwise_op`/`binary_map_op` for any other `T`, so complex needs
  no SIMD kernel.)
- `mean`/lexicographic `min`/`max` for complex live in the binding helpers
  (`to_f64_ma` cannot serve complex `mean`; `ma_extremum` needs a complex
  lexicographic compare). Tracked under #873.

### numpy.ma complex semantics captured (verified live, numpy 2.4.5)

| op | numpy.ma complex behavior | source |
|---|---|---|
| `.dtype` | `complex128` (c64→complex64) | live |
| `.fill_value` | `(1e+20+0j)` (both c64/c128) | `numpy/ma/core.py:163` default_filler `'c'` |
| `.real`/`.imag` | real masked array | live: `[1.0 --]` |
| `abs()` | real masked array (magnitude) | live: `abs([3+4j]) → [5.0]` |
| `.conjugate()` | complex masked array | live: `[(1-2j)]` |
| `+`,`-`,`*`,`/`,`**` | complex compute | live: `/2 → [(0.5+1j)]`, `**2 → [-3+4j]` |
| `//`,`%` | `TypeError` ('floor_divide'/'remainder' not supported) | live |
| `==`,`!=` | complex equality + masked-position override | live: `[True --]` |
| `<`,`>`,`<=`,`>=` | `TypeError` ('<' not supported) | live |
| `&`,`|`,`^`,`<<`,`>>` | `TypeError` ('bitwise_and' not supported) | live |
| `sum`/`prod`/`mean` | complex scalar | live: `sum→(1+11j)`, `prod→(2+4j)`, `mean→(2+3j)` |
| `min`/`max` | complex scalar (LEXICOGRAPHIC ordering) | live: `min→(1+2j)`, `max→(1+2j)` |

## Incremental build plan (smallest-first)

Each group is a separate builder→critic cycle under epic #850 / #867.

1. **R-1 (#868) — construction + dtype + filled.** Smallest first build: add the
   two `DynMa` variants + thread `match_ma!`/`dtype_name`/`impl_wrap_dyn!`/
   `build_dynma`/`AccIdentity` + complex fill-default in `ferray-ma`. Unblocks
   *constructing* and *egressing* a complex masked array — every other group
   depends on this.
   - Manifest: `ferray-python/src/ma.rs`, `ferray-ma/src/masked_array.rs`,
     `ferray-numpy-interop/src/numpy_conv.rs` (NpElement for Complex if absent),
     `ferray-python/tests/test_divergence_ma_dtype_audit.py`.
2. **R-3 (#869) — complex arithmetic.** The LIBRARY prereq. Land the ferray-ufunc
   `WrappingArith`/`TrueDivide`/`power` complex impls, then the complex arm in
   `compute_binary`. Done before R-2 because arithmetic exercises the variant
   most broadly and surfaces any threading miss in R-1.
   - Manifest: `ferray-ufunc/src/ops/arithmetic.rs`,
     `ferray-ufunc/tests/divergence_*.rs`, `ferray-python/src/ma.rs`,
     `ferray-python/tests/test_divergence_ma_*.py`.
3. **R-2 (#870) — real/imag/conjugate/abs.** Binding-only over the existing
   `ferray_ufunc::ops::complex` fns.
   - Manifest: `ferray-python/src/ma.rs`,
     `ferray-python/tests/test_divergence_ma_*.py`.
4. **R-4 (#871) — `==`/`!=` compute + ordering raises.** Complex `Eq`/`Ne` arm
   in `compute_compare`; ordering guard in `compare_op`.
   - Manifest: `ferray-python/src/ma.rs`,
     `ferray-python/tests/test_divergence_ma_*.py`.
5. **R-5 (#872) — `//`/`%`/bitwise raise.** Guards in `compute_binary` /
   `bitwise_op`.
   - Manifest: `ferray-python/src/ma.rs`,
     `ferray-python/tests/test_divergence_ma_*.py`.
6. **R-6 (#873) — complex reductions.** `AccIdentity` already lands in R-1;
   here add the complex `mean` path and lexicographic `min`/`max`.
   - Manifest: `ferray-python/src/ma.rs`,
     `ferray-python/tests/test_divergence_ma_*.py`.

## Verification

The gauntlet for this module (each must be green for the corresponding REQ to
move to SHIPPED in a future build commit):

- `cd ferray-python && maturin develop`
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/test_divergence_ma_dtype_audit.py -q`
  (pins `test_KNOWN_complex_masked_array_supported_by_numpy` — currently the
  driving divergence)
- `cargo test -p ferray-ufunc` (complex `WrappingArith`/`TrueDivide`/`power`
  unit + divergence tests for R-3)
- `cargo test -p ferray-ma` (complex `default_fill_value` arm for R-1)
- `cargo clippy -p ferray-python --all-targets -- -D warnings`
- `cargo clippy -p ferray-ufunc --all-targets -- -D warnings`

Per-REQ pinned pytest (to be added by the builder/critic of each group, expected
values from the live numpy.ma oracle, R-CHAR-3):
- R-1: `dtype == "complex128"`, `.filled()` equals numpy, `np.asarray` equals.
- R-2: `.real`/`.imag`/`abs()`/`.conjugate()` equal numpy.
- R-3: `+`/`-`/`*`/`/`/`**` data+mask equal numpy.ma.
- R-4: `==`/`!=` equal numpy.ma; `<` raises `TypeError`.
- R-5: `//`/`%`/`&` raise `TypeError`.
- R-6: `sum`/`prod`/`mean`/`min`/`max` equal numpy.ma complex scalars.
