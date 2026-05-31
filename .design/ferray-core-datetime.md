# ferray (main array) — datetime64 / timedelta64 dtype support for `fr.array` + top-level ufuncs/reductions

<!--
tier: 3-component
status: draft
baseline-commit: 33cf1f54
upstream-paths:
  - /home/doll/numpy-ref/numpy/_core/src/multiarray/datetime.c
  - /home/doll/numpy-ref/numpy/_core/src/multiarray/datetime_busday.c
  - /home/doll/numpy-ref/numpy/_core/code_generators/generate_umath.py
  - /home/doll/numpy-ref/numpy/_core/fromnumeric.py
-->

## Summary

`fr.ndarray` is numpy's own ndarray (re-exported in `lib.rs`); every top-level
`fr.*` function takes and returns a real `numpy.ndarray`. datetime64/timedelta64
arrays therefore already exist end-to-end at the object level — the divergence
is entirely in ferray's top-level **functions**, whose dtype dispatch is
real-only and rejects (or silently mis-handles) the `"M"`/`"m"` dtype kinds.

This mirrors the complex epic #919/`.design/ferray-core-complex.md`: thread the
time dtypes through the binding's dispatch macros and reuse a library substrate
that — as with complex — is **almost entirely already shipped**. The difference
is the *result-dtype algebra*: datetime arithmetic is unit-aware and produces a
result that is datetime OR timedelta depending on the operand kinds
(`datetime - datetime = timedelta`, `datetime + timedelta = datetime`, …), with
NaT (`i64::MIN`) as the NaN-equivalent and finer-of-two unit promotion.

Verified live (numpy 2.4, baseline `33cf1f54`):
- **R-1 input coercion REJECTS:** `fr.array(['2020-01-01'], dtype='datetime64[D]')`
  / `fr.asarray(np_datetime)` / `fr.zeros(3, dtype='datetime64[D]')` /
  `fr.array([5], dtype='timedelta64[D]')` →
  `TypeError: unsupported dtype: "datetime64[D]" (supported: bool, int8/…, float32/64)`.
- **R-2 arithmetic PARTLY SHIPPED:** `fr.subtract(dt,dt)` → `timedelta64[D]`,
  `fr.add(dt,td)`/`fr.subtract(dt,td)`/`fr.add(td,td)` all WORK today (routed
  through `crate::datetime::add_time`/`subtract_time`, #831). BUT
  `fr.multiply(td, 2)`/`fr.divide(td,td)`/`fr.floor_divide`/`remainder` →
  `TypeError: unsupported dtype for numeric op`; the
  `datetime + datetime`/`datetime * scalar` RAISE branches surface a ferray-side
  message, not numpy's `UFuncTypeError`/`_UFuncBinaryResolutionError` text.
- **R-3 comparison/ordering REJECT:** `fr.less(dt,dt)` →
  `unsupported dtype for numeric op`; `fr.min`/`fr.max`/`fr.argmin`/`fr.sort` →
  `unsupported dtype for ordering op`.
- **R-4 reductions DIVERGE (data hazard):** `fr.mean(dt)` returns `array(18268.5)`
  (a raw float — DROPS the datetime contract) where **numpy RAISES**
  `UFuncTypeError: ufunc 'add' cannot use operands … <M8[D]> and <M8[D]>`;
  `fr.std(td)`/`fr.var(td)` return floats where **numpy RAISES**
  `TypeError: ufunc 'square' not supported`. `fr.sum(td)` → `unsupported dtype`
  where numpy → `timedelta64`.
- **R-5 arange datetime REJECTS:** `fr.arange('2020-01','2020-04',dtype='datetime64[M]')`
  → `TypeError: must be real number, not str`.
- **R-6 astype/NaT:** `fr.isnat` WORKS (#831); unit-conversion/`datetime↔int↔string`
  astype is numpy's own ndarray method (works at the object level — not a
  `fr.*` function gap).

## The pivotal finding (binding-vs-library split)

**This is overwhelmingly BINDING work, even more so than the complex epic.** The
ferray-core/ferray-ufunc library substrate is already built; almost nothing new
is needed in the library crates. Quoted evidence:

1. **`DateTime64`/`Timedelta64` ARE `Element`, and `DynArray` ALREADY has the
   variants** — datetime is *ahead* of where complex started.
   - `impl Element for DateTime64` / `impl Element for Timedelta64` return
     `DType::DateTime64(TimeUnit::Ns)` / `DType::Timedelta64(TimeUnit::Ns)`
     (`ferray-core/src/dtype/datetime.rs`, `impl Element for DateTime64`).
   - `DType` carries the unit: `DateTime64(TimeUnit)` / `Timedelta64(TimeUnit)`
     variants (`ferray-core/src/dtype/mod.rs`, `enum DType`), with `size_of`/
     `align_of` → 8/`align_of::<i64>` and `Display` → `datetime64[{u}]`.
   - `DynArray` has `DateTime64(Array<DateTime64, IxDyn>, TimeUnit)` and
     `Timedelta64(Array<Timedelta64, IxDyn>, TimeUnit)` variants (the int64 +
     unit representation the task hypothesised), with constructors
     `from_datetime64`/`from_timedelta64`, extractors `try_into_datetime64`/
     `try_into_timedelta64`, and a working `DynArray::zeros(DType::DateTime64(unit), …)`
     arm (`ferray-core/src/dynarray.rs`, `enum DynArray` + `pub fn zeros`). The
     `dispatch!` macro already binds both variants.

   **Answer to the pivotal question:** YES, datetime64/timedelta64 are
   representable as int64+unit in the existing `DynArray`, AND ferray-core
   already has the `DateTime64`/`Timedelta64` element types. **No net-new
   library element type is needed** — unlike complex, even the DynArray variants
   already exist. The binding-layer int64-view marshalling (#831) is the
   transport, and it is already implemented.

2. **The int64↔datetime marshalling helpers ALREADY EXIST (#831).** This is the
   transport every group below reuses (`ferray-python/src/conv.rs`):
   - `pub fn datetime64_unit(py, arr) -> (String, i64)` — reads
     `numpy.datetime_data(dtype)` (e.g. `("D", 1)`).
   - `pub fn as_int64_view(py, arr)` — zero-copy `ascontiguousarray(arr).view('int64')`
     so `PyReadonlyArrayDyn<i64>` can extract the ticks.
   - `pub fn int64_to_datetime64(py, ticks, unit_str)` /
     `pub fn int64_to_timedelta64(py, ticks, unit_str)` — rebuild via
     `int64_arr.view('datetime64[<unit>]')`.
   The doc-comment states the contract explicitly: *"a datetime64 ndarray is
   bridged to Rust through numpy's own zero-copy `.view('int64')`; the typed
   kernels in `ferray-ufunc::ops::datetime` run on the i64 ticks, and the result
   is reconstructed via `int64_arr.view('datetime64[<unit>]')`"*
   (`ferray-python/src/datetime.rs` module doc). `extract_ticks` in
   `datetime.rs` packages this into `(ArrayD<i64>, TimeUnit, unit_str)`.

3. **The unit-aware arithmetic kernels ALREADY EXIST in ferray-ufunc.** The
   result-dtype algebra is implemented:
   - `pub fn sub_datetime_promoted(a, unit_a, b, unit_b) -> (Array<Timedelta64,D>, TimeUnit)`
     — `datetime[ua] - datetime[ub] → timedelta[finer]`
     (`ferray-ufunc/src/ops/datetime.rs`).
   - `pub fn add_datetime_timedelta_promoted(…) -> (Array<DateTime64,D>, TimeUnit)`
     — `datetime + timedelta → datetime[finer]`.
   - `pub fn add_timedelta_promoted(…) -> (Array<Timedelta64,D>, TimeUnit)` —
     `timedelta ± timedelta → timedelta[finer]`.
   - `pub fn isnat_datetime`/`isnat_timedelta`. NaT propagation
     (`nat_propagate`) and finer-unit promotion (`TimeUnit::finer`/`scale_to`)
     are built into all three. The scalar operator overloads
     (`Sub<DateTime64> for DateTime64 → Timedelta64`, etc.) and array-level
     overloads also exist (`ferray-core/src/dtype/datetime.rs`, `mod array_ops`).

4. **The binding-side datetime arithmetic dispatch ALREADY EXISTS and is wired
   into `fr.add`/`fr.subtract`.** `ferray-python/src/datetime.rs`:
   - `pub fn is_time_op(py, a, b)` — sniffs `dtype.kind` ∈ `{"M","m"}`.
   - `pub fn add_time(py, a, b)` / `pub fn subtract_time(py, a, b)` — broadcast,
     extract ticks, route to the promoted ferray-ufunc kernels, reconstruct.
   - `pub fn add` / `pub fn subtract` in `ufunc.rs` branch on
     `crate::datetime::is_time_op(py, x1, x2)?` BEFORE the numeric path
     (`ufunc.rs`, `pub fn add` / `pub fn subtract`).

**The genuine gaps (all binding, mostly NEW dispatch arms over EXISTING kernels):**

- **R-1 input coercion (NEW binding arms).** `array`/`asarray`/`copy`'s
  `match_dtype_all!` (`conv.rs`) and the `creation_dispatch!` macro (`creation.rs`)
  have no `"M"`/`"m"` arms — they hit the `other =>` `TypeError`. Fix: branch on
  `dtype.kind` ahead of the macro and route through the existing
  `as_int64_view`/`int64_to_datetime64` transport (or, for value-list inputs like
  `fr.array(['2020-01-01'], dtype='datetime64[D]')`, delegate the parse to
  `np.asarray(obj, dtype)` then preserve the M/m dtype across the boundary —
  numpy owns the string→datetime64 parse). For `fr.zeros(3, dtype='datetime64[D]')`,
  `DynArray::zeros(DType::DateTime64(unit), …)` already produces the data;
  the binding needs only to push it across via `int64_to_datetime64`.
- **R-2 timedelta numeric arithmetic (NEW binding + small library).** numpy
  computes `timedelta * int/float → timedelta`, `timedelta / timedelta → float64`,
  `timedelta // timedelta → int64`, `timedelta % timedelta → timedelta`. ferray
  has NO kernel for these (only ±). Fix: a `multiply_time`/`divide_time` binding
  dispatch + small ferray-ufunc kernels (`mul_timedelta_scalar`,
  `divtrue_timedelta` → f64, `floordiv_timedelta` → i64, `mod_timedelta`). These
  are the ONLY net-new library kernels in the whole epic.
- **R-2 raise-text alignment (NEW binding).** `datetime + datetime` and
  `datetime * scalar` must raise numpy's exact text
  (`_UFuncBinaryResolutionError`/`UFuncTypeError` `"cannot use operands with
  types"`). Today the `(dt,dt)` add hits ferray's
  `"unsupported operand dtypes for datetime add"`; numpy raises a different
  message. Match the exception type + message (R-DEV-2).
- **R-3 comparison + ordering (NEW binding over int64-view).** `less`/`greater`/
  `equal`/etc. and `min`/`max`/`argmin`/`argmax`/`sort`/`searchsorted` compare by
  the **int64 tick value** (after promoting to a common finer unit). The cleanest
  binding is to view as int64, run the existing real int64 dispatch, and re-tag
  datetime results (min/max/sort → datetime; argmin/argmax/searchsorted → int64;
  comparisons → bool) — mirroring how `busday_offset` already does
  `int64_arr.view('datetime64[D]')`. NaT ordering must match numpy (NaT sorts
  last; comparisons with NaT → False, like the `PartialEq` impl).
- **R-4 reductions (NEW binding + STOP the divergence).** This carries an active
  divergence: `fr.mean(dt)` returns a raw float `array(18268.5)` and `fr.std(td)`
  returns `0.5`, but **numpy RAISES** for both. Per the live oracle:
  `mean(datetime)` → `UFuncTypeError` (add of two datetimes); `sum(datetime)` →
  raises; `std/var(timedelta)` → `TypeError ufunc 'square' not supported`. Only
  `sum(td)`/`mean(td)` → timedelta, and `ptp(dt)` → timedelta are computable.
  Fix: `stats.rs` gains `"M"`/`"m"` arms that compute the *computable*
  reductions over int64 ticks and re-tag, and **RAISE numpy's exact exception**
  for the non-computable ones (mean/sum/std/var of datetime; std/var of
  timedelta) instead of silently returning a float.
- **R-5 arange datetime (NEW binding).** `fr.arange('2020-01','2020-04',
  dtype='datetime64[M]')` and `arange(start_dt, stop_dt, step_td)` → datetime
  array. Cleanest binding: delegate the integer arange to numpy on the int64
  ticks (or to `np.arange` directly on the datetime dtype, preserving the
  contract across the boundary), since the start/stop are datetime64 strings
  numpy parses.
- **R-6 astype/NaT.** `arr.astype('datetime64[h]')`/`astype('int64')`/
  `astype('str')` is numpy's own ndarray method — it works at the object level and
  is NOT a `fr.*` function gap. NaT comparison/propagation is already correct in
  the `PartialEq` impls and `nat_propagate`. This group is mostly **already
  satisfied**; the only `fr.*`-function astype surface (`fr.astype` if exposed)
  would reuse the int64-view transport.

**Magnitude verdict:** ~85–90% binding (new dispatch arms in `conv.rs`/
`creation.rs`/`ufunc.rs`/`stats.rs` + new `multiply_time`/`divide_time`/arange/
ordering binding helpers reusing the int64-view transport), ~10–15% library (the
four small timedelta numeric kernels in `ferray-ufunc/src/ops/datetime.rs`:
`mul_timedelta_scalar`, true/floor div, mod). It IS a multi-builder epic on the
shape of complex #919 — but smaller, because the element types, DynArray
variants, transport, and ± arithmetic kernels are already shipped. Estimate
**5 builder→critic cycles** (one per REQ group R-1..R-5; R-6 is largely a
verification/no-op group). Smallest-highest-value first: **R-4 (stop the
silent mean/std float divergence — an active correctness hazard) + R-1 (accept
datetime input so every group is testable)**, then R-2/R-3/R-5.

## Requirements

- REQ-1 (R-1 input coercion): `fr.array(['2020-01-01'], dtype='datetime64[D]')`,
  `fr.asarray(np_datetime_array)`, `fr.array([5], dtype='timedelta64[D]')`,
  `fr.zeros/ones/empty/full(shape, dtype='datetime64[D]'|'timedelta64[D]')` build
  a datetime64/timedelta64 ndarray preserving unit + shape, instead of raising
  `TypeError: unsupported dtype`.
- REQ-2 (R-2 datetime arithmetic): the full unit-aware algebra computes —
  `datetime-datetime→timedelta`, `datetime±timedelta→datetime`,
  `timedelta±timedelta→timedelta` (the ± cases SHIPPED via `add_time`/
  `subtract_time`), plus `timedelta*int/float→timedelta`,
  `timedelta/timedelta→float64`, `timedelta//timedelta→int64`,
  `timedelta%timedelta→timedelta`; and `datetime+datetime`/`datetime*scalar`
  RAISE numpy's exact `UFuncTypeError`/`_UFuncBinaryResolutionError`. Mixed units
  promote to the finer unit; NaT propagates.
- REQ-3 (R-3 comparison + ordering): `fr.less`/`greater`/`less_equal`/
  `greater_equal`/`equal`/`not_equal` over datetime/timedelta compute bool by
  int64 tick value (common finer unit); `fr.min`/`max` → datetime/timedelta,
  `fr.argmin`/`argmax`/`searchsorted` → int64, `fr.sort` → datetime/timedelta,
  all ordered by the int64 value with NaT sorting last per numpy.
- REQ-4 (R-4 reductions — STOP the divergence): `fr.sum(timedelta)`/
  `fr.mean(timedelta)` → timedelta; `fr.ptp(datetime)` → timedelta; and
  `fr.mean(datetime)`/`fr.sum(datetime)`/`fr.std(datetime|timedelta)`/
  `fr.var(datetime|timedelta)` **RAISE numpy's exact exception** instead of the
  current silent float (`fr.mean(dt)→array(18268.5)`, `fr.std(td)→0.5` are active
  divergences — numpy raises).
- REQ-5 (R-5 arange datetime): `fr.arange('2020-01','2020-04',dtype='datetime64[M]')`
  and `fr.arange(start_datetime, stop_datetime, step_timedelta)` produce a
  datetime64 array of the requested unit instead of `TypeError: must be real
  number, not str`.
- REQ-6 (R-6 astype + NaT): unit conversion (`datetime64[D]→[h]`),
  `datetime↔int64`, `datetime↔string` astype preserve numpy's dtype contract
  (object-level numpy ndarray methods — verified to work today); NaT
  (`i64::MIN`) is the NaN-equivalent with `NaT != NaT` and propagation through
  every op (already correct in `DateTime64::PartialEq` / `nat_propagate`).

## Acceptance criteria

- AC-1 (REQ-1): pytest `fr.array(['2020-01-01'], dtype='datetime64[D]').dtype ==
  'datetime64[D]'`; `fr.asarray(np.array(['2020-01-01'],dtype='datetime64[D]'))`
  equals numpy; `fr.array([5], dtype='timedelta64[D]')` round-trips;
  `fr.zeros(3, dtype='datetime64[D]')` equals `np.zeros(3, dtype='datetime64[D]')`.
- AC-2 (REQ-2): `fr.subtract(dt,dt)`/`fr.add(dt,td)`/`fr.add(td,td)` equal numpy
  (SHIPPED); `fr.multiply(td,2)==np.multiply(td,2)` (timedelta64);
  `fr.divide(td,td)` → float64 equal numpy; `fr.floor_divide(td,td)` → int64;
  `fr.subtract(dt[D],dt[h])` → `timedelta64[h]` (finer unit);
  `fr.add(dt,dt)` raises `UFuncTypeError`; NaT in any operand → NaT.
- AC-3 (REQ-3): `fr.less(dt,dt)==np.less(dt,dt)`; `fr.min(dt)`/`fr.max(dt)` equal
  numpy (datetime64); `fr.argmin(dt)==np.argmin(dt)` (int64);
  `fr.sort(dt)==np.sort(dt)`; `fr.searchsorted` equals numpy.
- AC-4 (REQ-4): `fr.sum(td)==np.sum(td)` (timedelta64); `fr.mean(td)==np.mean(td)`;
  `fr.ptp(dt)==np.ptp(dt)` (timedelta64); `fr.mean(dt)` raises `UFuncTypeError`
  (NOT a float); `fr.std(td)`/`fr.var(td)` raise `TypeError` (NOT a float) — the
  imag-discard-analog guard.
- AC-5 (REQ-5): `fr.arange('2020-01','2020-04',dtype='datetime64[M]') ==
  np.arange('2020-01','2020-04',dtype='datetime64[M]')`;
  `fr.arange(d0,d1,td)==np.arange(d0,d1,td)`.
- AC-6 (REQ-6): `fr.isnat(np.array(['NaT'],dtype='datetime64[D]'))==[True]`
  (SHIPPED); `np.asarray(fr.array(...,dtype='datetime64[D]')).astype('datetime64[h]')`
  equals numpy's; NaT round-trips through `fr.array`/`fr.asarray`.

## REQ status table

All time-array `fr.*` *function* surfaces except `fr.add`/`fr.subtract`
(datetime ±, SHIPPED via #831) and `fr.isnat` (SHIPPED via #831) are real-only.
The divergences reproduce live at `33cf1f54`. Two states only (R-DEFER-2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (input coercion) | SHIPPED | #941. `fn datetime_roundtrip` + `fn is_time_dtype_name` (`datetime.rs`) round-trip an already-built numpy datetime64/timedelta64 array through the int64-view transport (`as_int64_view` → `ArrayD<i64>` → `int64_to_datetime64`/`int64_to_timedelta64`, #831), preserving unit+shape+NaT (R-CODE-4). Non-test consumers: the `array`/`asarray` `#[pyfunction]` (`creation.rs`, `pub fn array` — `M`/`m` branch ahead of `match_dtype_all!`) and `zeros`/`ones`/`empty`/`full`/`zeros_like`/`ones_like`/`empty_like`/`full_like` (`creation.rs` — `time_shape_create` branch ahead of `creation_dispatch!`), all registered top-level in `lib.rs`. numpy: `np.array(['2020-01-01'],dtype='datetime64[D]')`, `np.zeros(3,dtype='datetime64[s]')` (epoch), `np.ones(2,dtype='datetime64[D]')` (tick 1), `np.full(2,'2021-06-15',dtype='datetime64[D]')`. Verification: `tests/test_expansion_datetime_construct.py` (57 cases, units Y..ns, NaT, 2-D, real-unchanged), all live-derived (R-CHAR-3). |
| REQ-2 (datetime arithmetic) | SHIPPED | #942. The full unit-aware algebra now computes. The ± cases (SHIPPED via #831): `pub fn add`/`subtract` in `ufunc.rs` branch on `crate::datetime::is_time_op` → `add_time`/`subtract_time` (`datetime.rs`) → `add_datetime_timedelta_promoted`/`sub_datetime_promoted`/`add_timedelta_promoted` (`ops/datetime.rs`). The timedelta NUMERIC cases (this build): five NaT-propagating, finer-unit-promoted kernels in `ops/datetime.rs` — `mul_timedelta_scalar_i64`/`mul_timedelta_scalar_f64` (td·int exact / td·float trunc-toward-zero), `div_timedelta_scalar_i64`/`div_timedelta_scalar_f64` (td÷scalar trunc, 0→NaT), `truediv_timedelta` (td÷td→f64 ratio, NaT→nan), `floordiv_timedelta` (td//td→i64 FLOOR, NaT/0→0), `mod_timedelta` (td%td→td Python floor-mod, NaT/0→NaT) — mirroring numpy's `TIMEDELTA_*` loops (loops.c.src:902-1130) + the multiply/divide/floor_divide/remainder registrations (generate_umath.py:386-1046). Non-test consumers: `pub fn multiply`/`divide`/`floor_divide`/`remainder`/`mod_`/`true_divide` (`ufunc.rs`) branch on `crate::datetime::is_time_op` → `multiply_time`/`divide_time`/`floordiv_time`/`mod_time` (`datetime.rs`), all registered top-level in `lib.rs`. Raise-alignment: `dt+dt`/`dt*scalar`/`dt/x`/`td*td`/`int/td`/`td%int` now delegate to numpy on the original operands so numpy's EXACT `UFuncTypeError` family surfaces (`time_numeric_raise`, `datetime.rs`, R-DEV-2) instead of ferray text. numpy 2.4.5 live: `td(6,'D')*3==td(18,'D')`; `td(6,'D')/td(2,'D')==3.0`; `td(7,'D')//td(2,'D')==3`; `td(7,'D')%td(4,'D')==td(3,'D')`; `td(5,'D')*1.5==td(7,'D')` (trunc); `np.add(dt,dt)`→`UFuncTypeError`. Verification: `cargo test -p ferray-ufunc` (10 new kernel tests, R-CHAR-3) + `tests/test_expansion_datetime_arith.py` (47 cases — td×scalar/÷td→float/​//td→int/​%td, cross-unit, NaT, zero-divisor, dt/td-undefined-pair raises, real+complex unchanged; all live-derived). |
| REQ-3 (comparison + ordering) | NOT-STARTED | open prereq blocker #943. `fr.less(dt,dt)` → `unsupported dtype for numeric op`; `fr.min`/`max`/`argmin`/`sort` → `unsupported dtype for ordering op` (`stats.rs` dispatch is real-only). Fix = `"M"`/`"m"` binding arm viewing int64 ticks (common finer unit), running the existing int64 ordering dispatch, re-tagging (min/max/sort→datetime, argmin/argmax/searchsorted→int64, compares→bool), NaT-last per numpy. Reuses the `busday_offset`-style `.view('datetime64[D]')` re-tag. Live: `fr.min(dt)` → `unsupported dtype for ordering op`. |
| REQ-4 (reductions — STOP divergence) | SHIPPED | #944. The silent-float corruption is ELIMINATED: `pub fn time_reduce` + `enum TimeReduce` + `pub fn is_time_array` (`datetime.rs`) carry the compute-vs-raise contract over the int64-view transport (`extract_ticks` → NaT-aware int64 fold → `int64_to_datetime64`/`int64_to_timedelta64` re-tag, R-CODE-4), raising numpy's EXACT exception for the undefined `(kind, op)` pairs via `delegate_to_numpy` (R-DEV-2). **timedelta** computes `sum`/`mean` (int64 trunc-toward-zero)/`min`/`max`/`ptp`/`cumsum`/`argmin`/`argmax`; **datetime** computes `min`/`max`/`ptp`(→timedelta)/`argmin`/`argmax`; both RAISE for `std`/`var`/`prod` and datetime raises for `sum`/`mean`/`cumsum`. NaT (`i64::MIN`) propagates (sum/mean/min/max/ptp/cumsum → NaT; argmin/argmax → first NaT index). Non-test consumers: the `sum`/`mean`/`min`/`max`/`ptp`/`std`/`var`/`argmin`/`argmax`/`cumsum`/`prod` `#[pyfunction]`s (`stats.rs` — each branches on `crate::datetime::is_time_array` ahead of its real-only macro), all registered top-level in `lib.rs`. Live oracle: `np.mean(dt)`/`np.std(td)` raise (`UFuncTypeError`/`TypeError`); `np.sum(td)==timedelta64(15,'D')`; `np.mean(td)==timedelta64(5,'D')`; `np.ptp(dt)==timedelta64(4,'D')`. Verification: `tests/test_expansion_datetime_reduce.py` (33 cases — compute/raise/NaT/axis/real-unchanged, all live-derived, R-CHAR-3). |
| REQ-5 (arange datetime) | NOT-STARTED | open prereq blocker #945. `arange` (`creation.rs`, `pub fn arange`) coerces bounds to f64 (`arange_int`) → `TypeError: must be real number, not str` on a datetime64 string bound. Fix = a datetime branch delegating the datetime arange to numpy (which owns string→datetime64 parse) and preserving the M/m dtype across the boundary, or computing the int64 tick range and re-viewing. Live: `fr.arange('2020-01','2020-04',dtype='datetime64[M]')` → `TypeError`. numpy: `np.arange('2020-01','2020-04',dtype='datetime64[M]')==['2020-01','2020-02','2020-03']`. |
| REQ-6 (astype + NaT) | SHIPPED | `fr.isnat` is SHIPPED (`pub fn isnat` in `datetime.rs`, routed through `ferray_ufunc::isnat_datetime`/`isnat_timedelta`; registered in `lib.rs`). NaT semantics are correct: `DateTime64::PartialEq` / `Timedelta64::PartialEq` (`ferray-core/src/dtype/datetime.rs`) return `false` for any NaT operand (`NaT != NaT`), and `nat_propagate` (`ops/datetime.rs`) carries NaT through every kernel. Unit-conversion / `datetime↔int↔string` astype is numpy's own ndarray method on the returned ndarray (`fr.ndarray` IS `numpy.ndarray`, `lib.rs`) — verified live: `a.astype('datetime64[h]')`, `a.astype('int64')`, `a.astype('str')` all work at the object level (no `fr.*` function gap). Non-test consumer: `pub fn isnat` registered top-level in `lib.rs` (`m.add_function(wrap_pyfunction!(datetime::isnat, m)?)`). numpy: `np.isnat(np.datetime64('NaT'))==True` (generate_umath.py `isnat`). Verification: `fr.isnat(np.array(['NaT'],dtype='datetime64[D]'))==[True]` (live, this session). |

## Architecture

### There is NO `fr.ndarray` pyclass — the "main array" is numpy's ndarray

Same as the complex case: `fr.ndarray` re-exports `numpy.ndarray` (`lib.rs`).
Every `fr.*` ufunc/creation/reduction is a `#[pyfunction]` taking `&Bound<PyAny>`
and returning a real `numpy.ndarray`. Consequence: numpy's OWN `dt - dt`,
`dt.astype('datetime64[h]')`, `td * 2` (operator/method form) work on a
datetime64 ndarray because they are numpy, not ferray. ferray's job is the
top-level FUNCTION surface (`fr.array`, `fr.add`, `fr.min`, `fr.mean`, …).

### int64 + unit is the representation; the int64-view is the transport

A datetime64 ndarray is i64-backed (a count of `TimeUnit` ticks; NaT =
`i64::MIN`). pyo3-numpy has no native datetime64 `NumpyElement`, so the bridge
is numpy's own zero-copy `.view('int64')`:
- ingress: `datetime64_unit` (reads `numpy.datetime_data` → `(unit, count)`) +
  `as_int64_view` (→ `PyReadonlyArrayDyn<i64>`) → `extract_ticks` →
  `(ArrayD<i64>, TimeUnit, unit_str)` (`conv.rs`/`datetime.rs`, #831).
- egress: `int64_to_datetime64` / `int64_to_timedelta64`
  (`int64_arr.view('datetime64[<unit>]')`) (`conv.rs`, #831).
This preserves numpy's dtype + unit + shape across the boundary with no lossy
cast (R-CODE-4), and is the SAME transport `busday_offset` already uses
(`build_date_output` → `reshaped.view('datetime64[D]')`, `datetime.rs`).

### Where the time dtypes must be threaded (binding)

1. **`conv.rs` `match_dtype_all!` + `creation.rs` `creation_dispatch!`** (REQ-1):
   branch on `dtype.kind ∈ {"M","m"}` ahead of the real-dtype macro; for
   construction, route through the int64-view transport (or delegate the
   string→datetime64 parse to `np.asarray(obj, dtype)` and preserve the dtype).
   `fr.zeros(…, dtype='datetime64[D]')` composes `DynArray::zeros(DType::DateTime64(unit),…)`
   (already works) + `int64_to_datetime64`.
2. **`ufunc.rs` numeric/comparison dispatch** (REQ-2, REQ-3): the `add`/`subtract`
   datetime branch already exists; extend the SAME `is_time_op` gate to
   `multiply`/`divide`/`floor_divide`/`remainder` (timedelta numeric ops, via new
   `multiply_time`/`divide_time` + small `ops/datetime.rs` kernels) and to the six
   comparisons (int64-view compare → bool). Align `datetime+datetime`/
   `datetime*scalar` raise-text with numpy's `UFuncTypeError`.
3. **`stats.rs` reductions** (REQ-3 ordering, REQ-4 reductions): `"M"`/`"m"` arms
   in `min`/`max`/`argmin`/`argmax`/`ptp`/`sort`/`searchsorted` (int64-view ordering
   + re-tag) and `sum`/`mean`/`std`/`var` (compute the computable, RAISE numpy's
   exact exception for the rest — stops the silent-float divergence).
4. **`creation.rs` `arange`** (REQ-5): a datetime branch delegating to numpy's
   datetime arange and preserving the M/m dtype across the boundary.

### Library prereqs (ferray-core / ferray-ufunc) — almost all already shipped

- **Already present (no work):** `Element for DateTime64`/`Timedelta64` +
  `DType::DateTime64(TimeUnit)`/`Timedelta64(TimeUnit)` (`dtype/datetime.rs`,
  `dtype/mod.rs`); `DynArray::DateTime64`/`Timedelta64` variants + `zeros`/
  `from_*`/`try_into_*` (`dynarray.rs`); the int64-view transport (`conv.rs`,
  #831); the promoted ± kernels `sub_datetime_promoted`/
  `add_datetime_timedelta_promoted`/`add_timedelta_promoted` + `isnat_*`
  (`ops/datetime.rs`); NaT propagation (`nat_propagate`, `PartialEq`);
  finer-unit promotion (`TimeUnit::finer`/`scale_to`).
- **Net-new (small, library):** four timedelta numeric kernels in
  `ferray-ufunc/src/ops/datetime.rs` — `mul_timedelta_scalar` (timedelta·int/float
  → timedelta), `divtrue_timedelta` (timedelta/timedelta → f64),
  `floordiv_timedelta` (→ i64), `mod_timedelta` (→ timedelta), each with NaT
  propagation + unit promotion. Everything else is binding.

### numpy main-array datetime semantics captured (verified live, numpy 2.4)

| op | numpy behavior | source |
|---|---|---|
| `np.array(['2020-01-01'],dtype='datetime64[D]')` | `datetime64[D]` ndarray | live |
| `dt - dt` | `timedelta64[finer]` (`dt[D]-dt[h]→timedelta64[h]`) | live |
| `dt ± td` | `datetime64[finer]` | live |
| `td ± td` | `timedelta64[finer]` | live |
| `td * int/float` | `timedelta64` | live: `td*2→[6 8]` |
| `td / td` | `float64` | live: `[1. 1.]` |
| `td // td` | `int64` | live: `[1 1]` |
| `td % td` | `timedelta64` | live |
| `dt + dt` | `_UFuncBinaryResolutionError`/`UFuncTypeError` | live |
| `dt * scalar` | `UFuncTypeError` | live |
| `less(dt,dt)`/comparisons | `bool` by int64 value | live: `less→[False False]` |
| `min`/`max(dt)` | `datetime64` | live |
| `argmin`/`argmax(dt)` | `int64` | live |
| `sort(dt)` | `datetime64` (by int64) | live |
| `searchsorted` | `int64` | live: `[2 2]` |
| `sum(td)` | `timedelta64` | live: `timedelta64(7,'D')` |
| `mean(td)` | `timedelta64` | live: `timedelta64(3,'D')` |
| `ptp(dt)` | `timedelta64` | live: `timedelta64(5,'D')` |
| `mean(dt)`/`sum(dt)` | RAISES `UFuncTypeError` | live |
| `std`/`var(td\|dt)` | RAISES `TypeError 'square' not supported` | live |
| `arange('2020-01','2020-04',dtype='datetime64[M]')` | `datetime64[M]` array | live |
| `arange(d0,d1,td)` | `datetime64` array | live |
| `astype('datetime64[h]')`/`int64`/`str` | unit/int/string conversion | live (numpy ndarray method) |
| `isnat(NaT)` | `True`; `NaT != NaT` | live (SHIPPED) |

## Incremental build plan (smallest-highest-value first)

Each group is a separate builder→critic cycle under epic #940, mirroring the
complex sequencing (#920→#921→…). Because element types, DynArray variants,
transport, and ± kernels are already shipped, every group is dominated by binding
threading + a handful of small kernels.

1. **R-4 + R-1 first (STOP the divergence + accept input).** Highest value: R-4
   eliminates the live silent-float hazard (`fr.mean(dt)→18268.5`,
   `fr.std(td)→0.5` where numpy raises) — an R-CODE-4-class boundary violation
   (returning a value where numpy raises, silently dropping the datetime
   contract); R-1 makes datetime arrays constructible from Python so every other
   group is testable end-to-end.
   - Blockers: #944 (R-4), #941 (R-1).
   - Manifest: `ferray-python/src/stats.rs`, `ferray-python/src/conv.rs`,
     `ferray-python/src/creation.rs`, `ferray-python/src/datetime.rs` (extend),
     `ferray-python/tests/divergence_datetime.py` (new).
2. **R-2 (timedelta numeric arithmetic + raise alignment).** `multiply`/`divide`/
   `floor_divide`/`remainder` over timedelta + the four small `ops/datetime.rs`
   kernels; align `dt+dt`/`dt*scalar` raise-text.
   - Blocker: #942. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-python/src/datetime.rs`, `ferray-ufunc/src/ops/datetime.rs`,
     `ferray-python/tests/divergence_datetime.py`.
3. **R-3 (comparison + ordering).** `less`/`greater`/… + `min`/`max`/`argmin`/
   `argmax`/`sort`/`searchsorted` over the int64-view with re-tag + NaT-last.
   - Blocker: #943. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-python/src/stats.rs`, `ferray-python/src/datetime.rs`,
     `ferray-python/tests/divergence_datetime.py`.
4. **R-5 (arange datetime).** datetime branch in `arange`.
   - Blocker: #945. Manifest: `ferray-python/src/creation.rs`,
     `ferray-python/tests/divergence_datetime.py`.

(R-6 is SHIPPED — `isnat` + NaT semantics + object-level astype; no builder.)

## Verification

The gauntlet for this module (each must be green for the corresponding REQ to
move to SHIPPED in a future build commit):

- `cd ferray-python && maturin develop`
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/divergence_datetime.py -q`
  (pins the main-array datetime divergences; expected values from the live numpy
  oracle, R-CHAR-3)
- `cargo test -p ferray-ufunc` (the existing `ops/datetime.rs` kernel tests +
  any new timedelta numeric kernel tests)
- `cargo test -p ferray-core` (the `DateTime64`/`Timedelta64`/`DynArray` datetime
  path)
- `cargo clippy -p ferray-python --all-targets -- -D warnings`
- `cargo clippy -p ferray-ufunc --all-targets -- -D warnings`

Per-REQ pinned pytest (added by each group's builder/critic, oracle values from
live numpy, R-CHAR-3):
- R-1: `fr.array(['2020-01-01'],dtype='datetime64[D]').dtype == 'datetime64[D]'`;
  `fr.zeros(3,dtype='datetime64[D]')` equals numpy; timedelta round-trips.
- R-2: `fr.multiply(td,2)`/`fr.divide(td,td)`/`fr.floor_divide(td,td)` equal
  numpy; `fr.subtract(dt[D],dt[h])` → `timedelta64[h]`; `fr.add(dt,dt)` raises
  `UFuncTypeError`.
- R-3: `fr.less(dt,dt)`/`fr.min(dt)`/`fr.argmin(dt)`/`fr.sort(dt)` equal numpy.
- R-4: `fr.sum(td)`/`fr.mean(td)`/`fr.ptp(dt)` equal numpy; `fr.mean(dt)` RAISES
  `UFuncTypeError` (NOT a float); `fr.std(td)`/`fr.var(td)` RAISE `TypeError`
  (the silent-float guard).
- R-5: `fr.arange('2020-01','2020-04',dtype='datetime64[M]')` equals numpy.
- R-6 (SHIPPED): `fr.isnat(np.array(['NaT'],dtype='datetime64[D]'))==[True]`.
