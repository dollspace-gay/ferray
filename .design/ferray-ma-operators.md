# ferray.ma.MaskedArray — operator surface (arithmetic / comparison / bitwise / unary)

<!--
tier: 3-component
status: draft
baseline-commit: c85074dc4845e4568072382f6ba2eb46b852594b
upstream-paths:
  - numpy/ma/core.py
-->

## Summary

`ferray.ma.MaskedArray` (the PyO3 `PyMaskedArray` class in `ferray-python/src/ma.rs`,
now backed by the runtime-typed `DynMa` over 11 real dtypes after #853) exposes a
rich method/free-function surface but **no operator dunders at all**: every
arithmetic, comparison, bitwise, and unary operator on a `fr.ma.MaskedArray`
currently raises `TypeError`, and `a == 2` wrongly returns the Python `bool`
`False` (object-identity fallback) instead of a mask-aware bool array.

numpy.ma supports the FULL mask-aware operator suite by delegating each dunder to
its module-level `add`/`subtract`/`true_divide`/`floor_divide`/`power`/`_comparison`
helpers, which run on the `_MaskedBinaryOperation` / `_DomainedBinaryOperation` /
`_MaskedUnaryOperation` engine (`numpy/ma/core.py:1029`, `:1175`, `:955`).

This doc scopes the binding work to compose ferray-ufunc's existing compute ops
(arithmetic/comparison/bitwise are ALL present and generic over the dtype set) +
the ferray-ma library's mask-propagation engine (`fma::masked_binary`,
`fma::masked_unary`, `fma::divide_domain`, `fma::masked_binary_domain`) into the
`#[pymethods]` dunders on `PyMaskedArray`. No new ferray-ufunc compute primitive is
required for the core arithmetic/comparison/bitwise ops; the open library
prerequisites are narrower (dtype-generic mask-propagation entry points and integer
domain masking — see THE PIVOTAL FINDING).

## THE PIVOTAL FINDING

**ferray-ufunc already provides every compute op the masked operators need,
generic over the relevant dtype set. No ufunc-compute primitive is missing.**

Evidence (ferray-ufunc, baseline c85074dc):

- Arithmetic (R-A): `pub fn add`, `subtract`, `multiply` in `arithmetic.rs`
  (generic `T: Element + ...`); `pub fn divide` (= true_divide, integer→float64
  promotion, no div-by-zero panic: "Integer divide-by-zero returns `inf`/`nan` and
  never panics"); `pub fn floor_divide` (`T: Element + Float`) and
  `pub fn floor_divide_int` (integer dtypes); `pub fn power` / `pub fn power_int`;
  `pub fn remainder` / `pub fn mod_` / `pub fn remainder_int` / `pub fn mod_int`.
  Broadcasting variants: `pub fn add_broadcast`, `subtract_broadcast`,
  `multiply_broadcast`, `divide_broadcast` in `arithmetic.rs`.
- Comparison (R-B): `pub fn equal`, `not_equal`, `less`, `less_equal`, `greater`,
  `greater_equal` in `comparison.rs` (all `-> Array<bool, D>`), plus the
  `*_broadcast` variants.
- Bitwise (R-C): `pub fn bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`,
  `invert`, `left_shift`, `right_shift` in `bitwise.rs`; the `BitwiseOps` /
  `ShiftOps` traits are implemented for `i8..i64`, `u8..u64`, and `bool`
  (`impl_bitwise_ops!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, bool)`).
- Unary (R-D): `pub fn negative` / `negative_int`, `positive`, `absolute` /
  `absolute_int` / `fabs` in `arithmetic.rs`; bitwise invert via `invert` in
  `bitwise.rs`.

**Therefore the operator surface is BINDING + ferray-ma-library work, NOT
ferray-ufunc compute work.** The two genuine *library* prerequisites are:

1. **Dtype-generic mask propagation at the binding.** The binding's existing masked
   free-functions (`fma::masked_binary`, `divide`, etc.) are reached through
   `coerce_to_ma in ma.rs`, which collapses to `RustMa<f64, IxDyn>` (the float-only
   funnel). `fma::masked_binary` itself IS generic — its bound is
   `T: Element + Copy` (`ferray-ma/src/ufunc_support.rs`, `pub fn masked_binary`) —
   but `fma::divide_domain` is `T: Element + Float`
   (`ufunc_support.rs`, `pub fn divide_domain`). Preserving the operand's native
   dtype across `a + b` (R-CODE-4: int8 + int8 → int8, not f64) requires a
   `DynMa`-dispatching mask-propagation helper analogous to the `match_ma!` reduction
   helpers (`ma_sum_acc`/`ma_extremum` in `ma.rs`), not a new ufunc.
2. **Integer-dtype domain masking** for `//` and `%` (R-E). `fma::divide_domain` is
   float-only (`T: Element + Float`); integer floor-divide / mod by zero need either
   the `*_int` ufunc variants (`floor_divide_int`, `mod_int`) wrapped with an
   integer domain mask, or a `masked_binary_domain` instantiation over integer `T`.
   `fma::masked_binary_domain` is generic (`ufunc_support.rs:89`) but no integer
   domain wrapper exists yet.

## Requirements

- REQ-1 (R-A arithmetic): `__add__`/`__radd__`, `__sub__`/`__rsub__`,
  `__mul__`/`__rmul__`, `__truediv__`/`__rtruediv__`, `__floordiv__`/`__rfloordiv__`,
  `__mod__`/`__rmod__`, `__pow__`/`__rpow__` on `PyMaskedArray`, each returning a new
  `PyMaskedArray` whose data is the ferray-ufunc op result and whose mask is the
  union of operand masks.
- REQ-2 (R-B comparison): `__lt__`, `__le__`, `__gt__`, `__ge__` (and `__eq__`/`__ne__`,
  see REQ-7) returning a bool-dtype `PyMaskedArray`, masked where either operand is
  masked.
- REQ-3 (R-C bitwise): `__and__`/`__rand__`, `__or__`/`__ror__`, `__xor__`/`__rxor__`,
  `__invert__`, `__lshift__`/`__rlshift__`, `__rshift__`/`__rrshift__` on
  integer/bool-dtype masked arrays, preserving the integer dtype and propagating the
  mask.
- REQ-4 (R-D unary): `__neg__`, `__pos__`, `__abs__` returning a `PyMaskedArray` with
  the same mask and dtype as the input (`__invert__` is counted under R-C).
- REQ-5 (R-E domain masking): `__truediv__` masks positions where the denominator is
  exactly zero (numpy `_DomainSafeDivide`); `__floordiv__` and `__mod__` mask
  denominator-zero positions, for both float AND integer dtypes — the result mask is
  `union(operand masks) | domain_mask`.
- REQ-6 (R-F reflected + scalar operands): every binary operator accepts a Python
  scalar (`a + 1`, `a > 2`), a plain array/list, a numpy `ndarray`, a numpy.ma
  `MaskedArray`, or a `fr.ma.MaskedArray` as the other operand, coerced to the
  receiver's dtype (R-CODE-4); reflected dunders implement `1 + a`, `2 - a`, etc.
- REQ-7 (R-G `__eq__`/`__ne__` correctness): `a == x` / `a != x` return a bool
  `PyMaskedArray` (NOT Python `bool`), masked where either operand is masked, with the
  numpy.ma eq/ne special-case: where both are masked the underlying bool is "equal",
  where one is masked it is "unequal" (`numpy/ma/core.py:4245`
  `check = np.where(mask, compare(smask, omask), check)`).

## Acceptance criteria

- AC-1 (R-A): `a = fr.ma.array([1,2,3], mask=[0,1,0])`; `(a + 1).data` equals
  `np.ma.array([1,2,3], mask=[0,1,0]) + 1` `.data` and `(a+1).mask` equals its `.mask`
  for `+,-,*,/,//,%,**`; pytest in `tests/test_ma_operators.py`.
- AC-2 (R-A dtype): for `a` of int8, `(a + a).dtype == "int8"` and
  `(a / a).dtype == "float64"` (matches numpy.ma / NEP-50), no int→f64 collapse.
- AC-3 (R-B/R-G): `(a == 2)` is a `fr.ma.MaskedArray` of dtype bool with
  `(a==2).mask == a.mask` (the masked element masked) and unmasked bools equal to
  numpy.ma's `(np_a == 2)`; same for `!=,<,>,<=,>=`.
- AC-4 (R-C): for integer `a`, `(a & b)`, `(a | b)`, `(a ^ b)`, `(~a)`, `(a << n)`,
  `(a >> n)` match numpy.ma elementwise on data and mask.
- AC-5 (R-D): `(-a).data == -a.data`, `abs(a).data == np.abs(a.data)`, mask
  preserved; `(+a)` is a copy with the same data/mask.
- AC-6 (R-E): `fr.ma.array([1,2])/fr.ma.array([0,2])` masks element 0; same for
  `//` and `%`, for both float and integer dtypes; matches `np.ma` elementwise.
- AC-7 (R-F): `1 + a`, `a + 1`, `a + [10,20,30]`, `a + np.ma.array(...)`, and
  `a + fr.ma.array(...)` all succeed and equal numpy.ma.
- AC-8: no previously-green ma pytest regresses; `cargo clippy -p ferray-python
  --all-targets -- -D warnings` clean.

## REQ status table

Progress (binding `#[pymethods]` on `PyMaskedArray` in `ferray-python/src/ma.rs`):
REQ-4 (R-D unary: `__neg__`/`__pos__`/`__abs__`/`__invert__`), REQ-1 (R-A
arithmetic) + REQ-6 (R-F reflected/scalar: the seven binary dunders + reflected
forms via `binary_op`), and REQ-2 (R-B comparison) + REQ-7 (R-G eq/ne: the single
`__richcmp__` over the six comparison operators), and REQ-3 (R-C bitwise: the
ten `__and__`/`__or__`/`__xor__`/`__lshift__`/`__rshift__` dunders + reflected
forms via `bitwise_op`) are SHIPPED. REQ-5 (R-E integer-dtype domain masking)
remains NOT-STARTED with open prereq blocker #864. See the per-REQ table below.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (R-A arithmetic) | SHIPPED | `fn __add__`/`__sub__`/`__mul__`/`__truediv__`/`__floordiv__`/`__mod__`/`__pow__` (+ reflected) on `PyMaskedArray` in `ma.rs`, each delegating to `binary_op in ma.rs` → `compute_binary in ma.rs` over the `ferray_ufunc` ops `add`/`subtract`/`multiply`/`divide`/`floor_divide`/`floor_divide_int`/`mod_`/`mod_int`/`power`/`power_int`. Non-test production consumer: the dunders ARE invoked by CPython on every `a + b` (verified `test_expansion_ma_arith_ops.py`, 40 cases vs numpy.ma 2.4.5). Mask = `union(getmaskarray(a), getmaskarray(b))` via `compute_binary`; dtype via `numpy.result_type` (NEP-50). |
| REQ-2 (R-B comparison) | SHIPPED | `fn __richcmp__` (the single PyO3 rich-comparison entry) on `PyMaskedArray` in `ma.rs`, mapping `CompareOp::Lt`/`Le`/`Gt`/`Ge` to `CmpOp` and delegating to `compare_op in ma.rs` → `compute_compare in ma.rs` over the `ferray_ufunc` ops `less`/`less_equal`/`greater`/`greater_equal` in `comparison.rs` (`-> Array<bool, D>`). Result is a bool-dtype `DynMa::Bool` masked where either operand is masked (`union(getmaskarray(left), getmaskarray(right))`, broadcast); reflected `2 < a` arrives pre-oriented as `__richcmp__(a, 2, Gt)` (no manual swap). Non-test production consumer: `__richcmp__` is invoked by CPython on every `a > 2` / `a < b` (verified `test_expansion_ma_cmp_ops.py`, 42 cases vs numpy.ma — all six ops × {scalar, list, ndarray, numpy.ma, fr.ma} × {masked, ==masked, nomask, 2-D broadcast, 11 dtypes}). |
| REQ-3 (R-C bitwise) | SHIPPED | `fn __and__`/`__rand__`/`__or__`/`__ror__`/`__xor__`/`__rxor__`/`__lshift__`/`__rlshift__`/`__rshift__`/`__rrshift__` on `PyMaskedArray` in `ma.rs`, each delegating to `bitwise_op in ma.rs` → `compute_bitwise in ma.rs`. `&`/`\|`/`^` compose the `ferray_ufunc` ops `bitwise_and`/`bitwise_or`/`bitwise_xor` in `bitwise.rs` (`impl_bitwise_ops!` over `i8..i64,u8..u64,bool`) over the NEP-50 common variant; `<<`/`>>` delegate to numpy's `left_shift`/`right_shift` ufunc (the `ferray_ufunc` shift takes a `u32` count and would panic for numpy's negative / `>=` bit-width counts — R-CODE-2). Domain = INTEGER + BOOL only (a float operand raises `TypeError`, matching numpy's "no bitwise float loop"). Dtype = weak-scalar NEP-50 (`int8 & 1 → int8`, `bool & bool → bool`, `bool << 1 → int64`). Mask = `union(getmaskarray(a), getmaskarray(b))` (broadcast), non-domained: two nomask operands keep the `nomask` sentinel. Non-test production consumer: the dunders ARE invoked by CPython on every `a & b` / `a << n` / reflected `1 & a` (verified `test_expansion_ma_bitwise_ops.py`, 53 cases vs numpy.ma — 5 ops × {scalar, list, ndarray, numpy.ma, fr.ma} × {8 int dtypes, bool, float-raises, masked, nomask, mixed-dtype, negative/overflow shift, 2-D broadcast}). `~a` invert was already shipped in #861 (R-D unary, `__invert__`). |
| REQ-4 (R-D unary) | NOT-STARTED | open prereq blocker #861. No `__neg__`/`__pos__`/`__abs__` in `ma.rs`; live `-a`→`TypeError`, `abs(a)`→`TypeError`. Compute backend EXISTS: `pub fn negative`/`negative_int`/`positive`/`absolute`/`absolute_int` in `arithmetic.rs`. The mask-propagation primitive `fma::masked_unary` EXISTS (`ufunc_support.rs`, `T: Element + Copy`). |
| REQ-5 (R-E domain masking) | NOT-STARTED | open prereq blocker #864. No operator path exists. Float domain helper EXISTS: `pub fn divide_domain` in `ferray-ma/src/ufunc_support.rs` (`T: Element + Float`) and the generic `pub fn masked_binary_domain`. INTEGER `//`/`%` domain masking is the genuine library gap — `divide_domain` is float-only and no integer domain wrapper over `floor_divide_int`/`mod_int` exists. |
| REQ-6 (R-F reflected + scalar) | SHIPPED | `fn __radd__`/`__rsub__`/`__rmul__`/`__rtruediv__`/`__rfloordiv__`/`__rmod__`/`__rpow__` on `PyMaskedArray` in `ma.rs` (each `binary_op(.., reflected = true)` computing `other OP self`). Operand coercion: `binary_op in ma.rs` materializes `other` via `numpy.asarray` (scalar → its natural int64/float64, matching numpy.ma's `getdata`), reads a `fr.ma`/`numpy.ma` operand's mask, and promotes via `numpy.result_type` (no f64 funnel, R-CODE-4). Non-test production consumer: invoked by CPython on `1 + a` / `2 - a` (verified `test_expansion_ma_arith_ops.py::test_scalar_reflected` + `::test_array_operands_*`). |
| REQ-7 (R-G `__eq__`/`__ne__`) | SHIPPED | `fn __richcmp__` handles `CompareOp::Eq`/`Ne` (→ `CmpOp::Eq`/`Ne`) in `ma.rs`, so `a == 2` now returns a bool `MaskedArray` (NOT Python `False`). The `_comparison` eq/ne special-case is applied in `compute_compare in ma.rs`: at each masked position the underlying bool is `compare(smask, omask)` — `==` → `smask == omask` (equal iff both masked), `!=` → `smask != omask` (unequal iff one masked), matching `check = np.where(mask, compare(smask, omask), check)` (`numpy/ma/core.py:4245`). `a == np.ma.masked` masks every position (the 0-d masked singleton coerces to an all-True mask through the operand-union path); two nomask operands keep `mask is nomask`. Non-test production consumer: CPython invokes `__richcmp__` on every `a == x` / `a != x` (verified `test_expansion_ma_cmp_ops.py::test_scalar_eq_ne_masked_position_data`, `::test_eq_ne_both_masked_special_case`, `::test_compare_to_masked_singleton`, `::test_nomask_result_sentinel`). EDGE (flagged, divergence-pinned): `==`/`!=` against a truly-incomparable operand (e.g. an int array vs a string) returns `NotImplemented` (Python yields its identity result) rather than numpy.ma's all-`False` masked array, since the ferray-ufunc comparison ops require a unifiable common dtype. |

## Architecture

### How operators reach the masked engine (numpy.ma reference)

numpy.ma's dunders are thin delegators:

- `__add__` → `add(self, other)` (`numpy/ma/core.py:4313`), `__sub__` → `subtract`,
  `__mul__` → `multiply`, `__truediv__` → `true_divide` (`:4362`),
  `__floordiv__` → `floor_divide` (`:4378`), `__pow__` → `power` (`:4394`); reflected
  variants reverse operand order (`__radd__` → `add(other, self)`, `:4322`).
- `__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/`__ge__` → `self._comparison(other, op)`
  (`:4278`, `:4294`-`:4304`).
- Unary `-`/`abs`/`~` are inherited from `ndarray` and routed through
  `__array_ufunc__` onto `_MaskedUnaryOperation` (`numpy/ma/core.py:955`).

The mask-propagation engine, `_MaskedBinaryOperation.__call__`
(`numpy/ma/core.py:1062`): compute `result = self.f(getdata(a), getdata(b))` under
`errstate(divide='ignore', invalid='ignore')`, then
`m = logical_or(getmaskarray(a), getmaskarray(b))` (`:1079`-`:1083`), revert the data
to `da` where masked (`np.copyto(result, da, where=m)`, `:1096`), and tag the result
mask. The domained engine `_DomainedBinaryOperation` adds `m |= domain(da, db)`
(`:1220`-`:1221`) — `_DomainSafeDivide` masks `db == 0`. The comparison engine
`_comparison` (`:4193`): `mask = mask_or(smask, omask)` (`:4206`), `check =
compare(sdata, odata)` (`:4234`), and for eq/ne `check = np.where(mask, compare(smask,
omask), check)` (`:4245`) so masked-vs-masked compares equal.

### How ferray will mirror it (binding design)

`PyMaskedArray` already follows the `DynMa` + `match_ma!` shape for its reductions
(`ma_sum_acc`, `ma_extremum`, `scalar_pyobject` in `ma.rs`). The operator dunders add
`#[pymethods]` that mirror the `PyDual` dunder pattern in `autodiff.rs`
(`fn __add__(&self, other: &Bound<'_, PyAny>)`, `__radd__`, `__sub__`, `__rsub__`,
`__mul__`, `__neg__`, `__pow__`).

Per-operator flow (binary):

1. Coerce `other` to a same-dtype masked operand. Reuse `coerce_scalar in ma.rs` for
   scalars and the operand-coercion path used by `coerce_to_ma_impl in ma.rs` for
   array/ma operands — but routed through a `DynMa`-preserving variant (NOT the f64
   funnel `coerce_to_ma in ma.rs`, which would violate R-CODE-4 for integer data).
2. Compute the DATA via the matching ferray-ufunc op over the active `DynMa` variant
   (`match_ma!`): e.g. `ferray_ufunc::ops::arithmetic::add(a_data, b_data)` for `+`,
   `divide` for `/`, `bitwise_and` for `&`, `less` for `<`. NEP-50 promotion is
   already baked into those ops (`divide`→float64; wrapping integer arithmetic;
   `power_int`/`floor_divide_int`/`mod_int` for integer dtypes).
3. Propagate the MASK: `mask = getmaskarray(a) | getmaskarray(b)` — reuse
   `DynMa::mask_bits in ma.rs` (which calls `fma::getmaskarray`) + an elementwise
   OR (the existing `or_masks in ma.rs` helper, generalized to broadcast).
4. Apply the DOMAIN mask for `/`, `//`, `%`: OR in `(b_data == 0)`
   (`fma::masked_binary_domain` / `fma::divide_domain` already encode the float case;
   integer `//`/`%` need an integer domain wrapper — the BLOCK-E prerequisite).
5. Re-wrap as a `PyMaskedArray` via `from_dynma in ma.rs` (preserving dtype) or, for
   comparisons, as a `DynMa::Bool` result.

Unary (`-`, `+`, `abs`, `~`) flow: `match_ma!` over the variant, apply
`negative`/`positive`/`absolute`/`invert`, keep the mask via `fma::masked_unary`
(`ufunc_support.rs`, `T: Element + Copy`) or by carrying the mask buffer verbatim.

Key invariants:
- **Dtype preservation (R-CODE-4)**: `int8 ⊕ int8 → int8`; only `/` promotes to
  float64 (NEP-50, matching `ferray_ufunc::divide`'s `TrueDivide::Output`). Do NOT
  route the operator path through `to_f64_ma in ma.rs`.
- **Mask union (binary)**: result masked where EITHER operand masked.
- **Domain mask (R-E)**: additionally masked where the result is invalid (denominator
  zero), NOT inf/nan.
- **eq/ne special-case (R-G)**: masked-in-both → underlying bool `True` (equal);
  masked-in-one → `False`.

### File manifest (per group)

All production code lands in `ferray-python/src/ma.rs` (the dunders are
`#[pymethods]` on the existing `PyMaskedArray`; no new pyclass). Each group adds a
pytest file. The integer-domain prerequisite (R-E) is the only file outside
`ma.rs`/`ferray-python`:

- **R-D unary** (smallest, highest-value, first): `ferray-python/src/ma.rs`
  (`__neg__`/`__pos__`/`__abs__`/`__invert__`), `ferray-python/tests/test_ma_unary.py`.
- **R-A arithmetic + R-F reflected/scalar**: `ferray-python/src/ma.rs`
  (the 7 binary dunders + reflected + scalar coercion helper),
  `ferray-python/tests/test_ma_arithmetic.py`.
- **R-G eq/ne**: `ferray-python/src/ma.rs` (`__eq__`/`__ne__` with the masked
  special-case), `ferray-python/tests/test_ma_eq.py`.
- **R-B comparison**: `ferray-python/src/ma.rs`
  (`__lt__`/`__le__`/`__gt__`/`__ge__`), `ferray-python/tests/test_ma_compare.py`.
- **R-E domain masking**: `ferray-ma/src/ufunc_support.rs` (integer
  `floor_divide_domain`/`mod_domain` over `masked_binary_domain`),
  `ferray-python/src/ma.rs` (wire `/`,`//`,`%` to the domain helpers),
  `ferray-ma/tests/` + `ferray-python/tests/test_ma_domain.py`. **The only group
  with a ferray-ma library prerequisite.**
- **R-C bitwise**: `ferray-python/src/ma.rs`
  (`__and__`/`__or__`/`__xor__`/`__lshift__`/`__rshift__`),
  `ferray-python/tests/test_ma_bitwise.py`.

### Incremental build order

1. **R-D unary** — smallest, no library prereq, `fma::masked_unary` + ufunc
   `negative`/`absolute`/`invert` already exist. Highest value-per-line.
2. **R-A arithmetic + R-F reflected/scalar** — largest single value jump (`+,-,*,/,//,%,**`,
   scalars, reflected); all compute ops exist; needs the `DynMa`-preserving binary
   helper (BLOCK-A).
3. **R-G `__eq__`/`__ne__`** — fixes the worst correctness bug (`a==2` → `False`);
   small, depends on the comparison-coercion plumbing.
4. **R-B comparison** (`<,>,<=,>=`) — same plumbing as R-G.
5. **R-E domain masking** — the ONLY group needing a ferray-ma library extension
   (integer `//`/`%` domain wrapper); float `/` domain already in `divide_domain`.
6. **R-C bitwise** — independent, integer/bool only.

### Honest total magnitude

Medium. ~21 dunders (7 binary arithmetic + 7 reflected + 6 comparison + 6 bitwise +
3 unary + invert + 2 eq/ne) over the `DynMa` set, but each is a thin compose of an
EXISTING ferray-ufunc op + EXISTING ferray-ma mask helper. The bulk of the risk is
(a) the dtype-preserving operand-coercion path (avoiding the f64 funnel, R-CODE-4),
and (b) the single genuine library gap: integer-dtype `//`/`%` domain masking in
`ferray-ma/src/ufunc_support.rs`. No new ufunc compute primitive is needed.

## Verification

The gauntlet for this module is pytest parity vs numpy 2.4.5 (the ferray-python
verification model) plus the ferray-ma Rust gauntlet for the R-E prerequisite:

- `cd ferray-python && maturin develop` (rebuild before pytest sees a Rust change).
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/test_ma_unary.py tests/test_ma_arithmetic.py tests/test_ma_eq.py tests/test_ma_compare.py tests/test_ma_domain.py tests/test_ma_bitwise.py -q`
  — each green pins the corresponding REQ group; expected values come from live
  `numpy.ma` calls (R-CHAR-3), never literal-copied from ferray.
- `cargo test -p ferray-ma` — pins the integer domain-masking helper (R-E
  prerequisite) against numpy-documented div-by-zero masking.
- `cargo clippy -p ferray-python --all-targets -- -D warnings` and
  `cargo clippy -p ferray-ma --all-targets -- -D warnings`.

Until each pytest file above is green AND the ferray-ma R-E test passes, the
corresponding REQ stays NOT-STARTED (no operator dunder exists at baseline
c85074dc — all six commands currently have nothing to run).
