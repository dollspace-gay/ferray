# ferray (main array) — string dtype (`<U` unicode / `<S` bytes) support for `fr.array` + manipulation + comparison

<!--
tier: 3-component
status: draft
baseline-commit: 895a1168
upstream-paths:
  - /home/doll/numpy-ref/numpy/_core/strings.py
  - /home/doll/numpy-ref/numpy/_core/src/multiarray/dtypemeta.c
  - /home/doll/numpy-ref/numpy/_core/numeric.py
  - /home/doll/numpy-ref/numpy/_core/fromnumeric.py
-->

## Summary

`fr.ndarray` is numpy's own ndarray (re-exported in `lib.rs`); every top-level
`fr.*` function takes and returns a real `numpy.ndarray`. Fixed-width string
arrays (numpy `<U` unicode / `<S` bytes flexible dtype) therefore already exist
end-to-end at the object level — the divergence is entirely in ferray's
top-level **functions**, whose dtype dispatch is real-only (11 numeric dtypes +
complex + datetime) and rejects the `"U"`/`"S"` dtype kinds.

This mirrors the datetime epic #940/`.design/ferray-core-datetime.md` and the
float16 epic #951: a **detect-and-delegate** dtype where the array object is
numpy's own and ferray's job is only to recognise the dtype, route around the
real-only macros, and let numpy own the semantics. Strings are even simpler than
datetime: there is **no arithmetic** to translate and **no int64-view transport**
— string arrays are not numeric, so they ride the boundary directly as numpy
ndarrays. The string COMPUTE surface (`upper`/`lower`/`find`/`replace`/`add`
concatenation / `multiply` repeat / predicates / split / …) is **already shipped**
in `fr.char` / `fr.strings` (`ferray-python/src/char.rs`, ferray-strings #911/#912).
This epic is the MAIN-ARRAY string dtype plumbing only: construction +
dtype-preserving manipulation passthrough + lexicographic comparison.

Verified live (numpy 2.4.5, baseline `895a1168`):
- **R-1 input coercion REJECTS:** `fr.array(['ab','cd'])` →
  `TypeError: unsupported dtype: "str64" (supported: bool, int8/…, float32/64)`;
  `fr.array(['ab'],dtype='U3')` → `"str96"`; `fr.zeros(2,dtype='U5')` →
  `"str160"`; `fr.array([b'ab'],dtype='S4')` → `"bytes32"`;
  `fr.full(2,'hi',dtype='U4')` → `TypeError: must be real number, not str`;
  `fr.asarray(np.array(['ab','cd']))` → `"str64"`.
- **R-2 manipulation passthrough REJECTS:** `fr.reshape(str_arr,(2,2))` →
  `"str32"`; `fr.concatenate([['a'],['bb']])` → `"str64"`;
  `fr.repeat`/`fr.tile`/`fr.flip`/`fr.roll`/`fr.stack` → `unsupported dtype`;
  `fr.where(mask, str, str)` → `TypeError: data type 'str32' not understood`;
  `fr.unique`/`fr.sort`/`fr.argsort` → `unsupported dtype for ordering op`.
- **R-3 comparison REJECTS:** `fr.less(['a','b'],['b','a'])` →
  `unsupported dtype for numeric op: "str32"`; `fr.equal`/`fr.not_equal`/
  `fr.greater`/… likewise.
- **R-4 astype WORKS (already SHIPPED):** `fr.astype(np.array([1,2,3]),'U3')` →
  `<U3 ['1','2','3']`; `fr.astype(np.array(['ab'],dtype='U3'),'S3')` →
  `|S3 [b'ab']`. `fr.char.*` operate on string arrays (`fr.char.upper(['ab','cd'])`
  → `<U2 ['AB','CD']`, live).

## The pivotal finding (binding-vs-library split): 100% binding, no library work

**This is entirely BINDING work — there is no ferray-core / ferray-library work,
and it is smaller than datetime.** Quoted evidence:

1. **ferray-core has NO String `Element` and NO `DynArray` string variant.**
   The `Element` trait is implemented only for the numeric types (`impl Element
   for bool`/`f32`/`f64`/`i8`…/`I256`/`Complex<f32>`/`Complex<f64>`/`f16`/`bf16`
   in `dtype/mod.rs`) plus `DateTime64`/`Timedelta64` (`impl Element for
   DateTime64`/`Timedelta64` in `dtype/datetime.rs`). There is **no** `impl
   Element` for any string type — fixed-width strings are a flexible
   (variable-itemsize) dtype, not a fixed numeric scalar, so they cannot be an
   `Element`. The `DynArray` enum (`enum DynArray` in `dynarray.rs`) has variants
   only for the numeric + complex + datetime element types — **no string
   variant.** The dtype *descriptors* `DType::FixedUnicode(usize)` /
   `DType::FixedAscii(usize)` / `DType::RawBytes(usize)` exist (`enum DType`,
   `dtype/mod.rs`, #741) with `size_of`/`alignment`/`Display` (`<U{n}`/`|S{n}`)
   support, BUT both `DynArray` construction sites **explicitly reject** them:
   `DynArray::from_dtype` and `pub fn zeros` (`dynarray.rs`) return
   `FerrayError::invalid_dtype("DynArray cannot represent fixed-width string /
   void dtype targets yet (#741)")` / `"DynArray::zeros doesn't support
   fixed-width string / void dtypes yet (#741)"`. So the library substrate is a
   dead-end for strings — and that is fine, because **strings never need to enter
   the Rust library.**

2. **The string array IS a numpy ndarray; ops delegate to numpy + the existing
   `fr.char`.** `fr.ndarray` re-exports `numpy.ndarray` (`lib.rs`), so a `<U`/`<S`
   array constructed by numpy is a first-class object that numpy's own
   methods/ufuncs operate on. The string COMPUTE surface is already a full
   re-implementation: `ferray-python/src/char.rs` binds the entire
   `numpy.char`/`numpy.strings` surface (`pub fn upper`/`lower`/`find`/`replace`/
   `add` (concatenation)/`multiply` (repeat)/`count`/`startswith`/`split`/`join`/
   predicates/…) over `ferray_strings::StringArray`, registered under both
   `fr.char.*` and `fr.strings.*`. `fr.char.upper(np.array(['ab','cd']))` →
   `<U2 ['AB','CD']` live. So the char-specific OPS are done; this epic is **only**
   the main-array dtype plumbing (`fr.array`/`fr.reshape`/`fr.less`/…).

3. **The detect-and-delegate seam already exists — copy the datetime pattern,
   minus the transport.** datetime threads `"M"`/`"m"` arms ahead of the
   real-only macros (`crate::datetime::is_time_dtype_name` in `creation.rs`
   `array`/`asarray`/`zeros`/…; `crate::datetime::is_time_array` +
   `delegate_manip` in `manipulation.rs` `reshape`/`transpose`/`flip`/…;
   `crate::datetime::is_time_compare` + `compare_time` in `ufunc.rs`
   `bind_comparison!`). For datetime the delegate marshals back through the
   int64-view transport (`datetime_roundtrip`) because datetime IS an `Element`.
   For strings there is **no transport** — the delegate returns numpy's string
   result directly (it never crosses into Rust). That makes every group a thinner
   version of its datetime counterpart.

**The pivotal question answered:** there is NO ferray-core String `Element` (and
none is needed) — strings are a flexible dtype that stays as a numpy ndarray. The
ONLY viable approach is detect-and-delegate to numpy + the already-shipped
`fr.char`. **No library/ferray-core work, no new kernels.** 100% binding.

### Detection predicate: key off `dtype.kind`, NOT `dtype.name`

Live-verified caveat (numpy 2.4.5): `np.dtype('U2').name == 'str64'`,
`np.dtype('U3').name == 'str96'`, `np.dtype('S4').name == 'bytes32'` — the `.name`
encodes the **bit width** and is unstable across widths. The stable discriminator
is `dtype.kind`: `'U'` for unicode, `'S'` for bytes (e.g.
`np.dtype('U3').kind == 'U'`, `np.dtype('S4').kind == 'S'`). The string detection
predicate must therefore sniff `numpy.asarray(obj).dtype.kind ∈ {'U','S'}`,
exactly as the datetime seam sniffs `kind ∈ {'M','m'}` (`time_kind` in
`datetime.rs`) — NOT the `.name` string the real-only macros match on.

## Requirements

- REQ-1 (R-1 input coercion): `fr.array(['ab','cd'])` (infer `<U2`),
  `fr.array(['ab'],dtype='U3')` (fixed `<U3`), `fr.array([b'ab'],dtype='S4')`
  (bytes `<S4`), `fr.asarray(np_string_array)`, and
  `fr.zeros/ones/empty/full/zeros_like/ones_like/empty_like/full_like(shape,
  dtype='U5'|'S4')` build a numpy unicode/bytes string ndarray preserving
  dtype + width + shape, instead of raising `TypeError: unsupported dtype`.
- REQ-2 (R-2 manipulation passthrough): `fr.reshape`/`ravel`/`transpose`/
  `swapaxes`/`moveaxis`/`rollaxis`/`flip`/`roll`/`repeat`/`tile`/`squeeze`/
  `expand_dims`/`concatenate`/`stack`/`where`/`unique`/`sort`/`argsort` over a
  `<U`/`<S` array preserve the string dtype (concatenate/stack width-promote to
  the max width per numpy `result_type`; argsort → int64; sort lexicographic by
  codepoint), instead of raising `TypeError: unsupported dtype`.
- REQ-3 (R-3 comparison): `fr.less`/`greater`/`less_equal`/`greater_equal`/
  `equal`/`not_equal` over `<U`/`<S` arrays compute bool lexicographically by
  codepoint (broadcasting operands), instead of raising
  `unsupported dtype for numeric op`.
- REQ-4 (R-4 astype): `fr.astype(arr, 'U3'|'S3')` round-trips numpy's width /
  `U↔S` / `number↔string` conversions wherever numpy allows it — this is numpy's
  own `arr.astype(dtype)` and is already SHIPPED.

## Acceptance criteria

- AC-1 (REQ-1): pytest `fr.array(['ab','cd']).dtype == np.array(['ab','cd']).dtype`
  (`<U2`); `fr.array(['ab'],dtype='U3').dtype == 'U3'`;
  `fr.array([b'ab'],dtype='S4').dtype == 'S4'`;
  `fr.zeros(2,dtype='U5')` equals `np.zeros(2,dtype='U5')`;
  `fr.full(2,'hi',dtype='U4')` equals numpy;
  `fr.asarray(np.array(['ab','cd']))` equals numpy.
- AC-2 (REQ-2): `fr.reshape(s,(2,2))`, `fr.concatenate([['a'],['bb']])`
  (`<U2`), `fr.sort(['c','a','b'])` (`['a','b','c']`), `fr.unique`,
  `fr.where(mask,s,t)`, `fr.repeat`/`tile`/`flip`/`roll`/`stack`/`argsort` each
  equal the corresponding `np.*` call (dtype + values).
- AC-3 (REQ-3): `fr.less(['a','b'],['b','a']) == np.less(...)` (`[True,False]`);
  `fr.equal`/`fr.not_equal`/`fr.greater`/`fr.greater_equal`/`fr.less_equal`
  equal numpy; broadcasting `fr.equal(['a','b'],'a')` equals numpy.
- AC-4 (REQ-4): `fr.astype(np.array([1,2,3]),'U3')` equals
  `np.array([1,2,3]).astype('U3')` (`<U3 ['1','2','3']`);
  `fr.astype(np.array(['ab'],dtype='U3'),'S3')` equals numpy (`|S3`).

## REQ status table

All string-array `fr.*` *function* surfaces except `fr.astype` (numpy's own
`arr.astype`, SHIPPED) and the `fr.char`/`fr.strings` COMPUTE surface (SHIPPED,
#911/#912) are real-only and reject `<U`/`<S`. The divergences reproduce live at
`895a1168`. Two states only (R-DEFER-2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (input coercion) | SHIPPED | `array`/`asarray` branch on `is_string_array` (sniffs `dtype.kind ∈ {U,S}`) ahead of `match_dtype_all!` (`creation.rs`, `pub fn array`) and return numpy's `<U`/`<S` ndarray via `arr.copy()`; `zeros`/`ones`/`empty`/`full`/`zeros_like`/`ones_like`/`empty_like`/`full_like` detect a string `dtype=` off the ORIGINAL object's `.kind` (`dtype_obj_is_string`, since the normalised `.name` `str160` does not round-trip) — or, for `*_like`, the source array's string dtype (`string_like_dtype`) — and delegate to numpy via `string_shape_create` (`np.zeros`/`np.full`(shape[,fill],dtype)), returning numpy's string ndarray unchanged (no transport — strings never enter the Rust library, R-CODE-4). `full` also infers `<U`/`<S` width from a string/bytes `fill_value` when no `dtype=` is given (`fill_is_string`). Consumer: the `array`/`asarray`/`zeros`/`ones`/`empty`/`full`/`*_like` `#[pyfunction]`s registered top-level in `lib.rs` `_ferray`. numpy (live 2.4.4): `np.array(['ab','cd'])`→`<U2`, `np.array(['a','abc'])`→`<U3`, `np.zeros(3,'U5')`→empty strings, `np.ones(2,'U5')`→`'1'`, `np.full(2,'hi','U4')`, `np.full(2,'hi')`→`<U2`, `np.full_like(<U2,'longfill')`→`'lo'`. Pinned green: `tests/test_expansion_string_construct.py` (38 cases — width inference, fixed `U3`/`S4`, bytes, zeros/ones/empty/full, `*_like`, 2-D, empty list, non-ASCII unicode; real/numeric/datetime/float16 unregressed). |
| REQ-2 (manipulation passthrough) | SHIPPED | #960. Shared detect-and-delegate seam in `manipulation.rs`: `is_string_array` (sniffs `dtype.kind ∈ {U,S}`, NOT the width-encoding `.name`), `list_first_is_string`, and `string_delegate`/`string_delegate_list` (call `np.<op>` and return the numpy string result DIRECTLY — no transport, since strings have no ferray `Element`/`DynArray` variant, R-CODE-4). Every data-move op gained a string branch AHEAD of its real-only `match_dtype_all_complex!`/ordering macro: `reshape`/`ravel`/`flatten`/`squeeze`/`expand_dims`/`transpose`/`swapaxes`/`moveaxis`/`rollaxis`/`flip`/`fliplr`/`flipud`/`rot90`/`roll`/`atleast_1d`/`atleast_2d`/`atleast_3d`/`tile`/`repeat`/`pad`/`delete`/`insert`/`append`/`resize`/`concatenate`/`stack`/`vstack`/`hstack`/`column_stack`/`dstack`/`block`/`split`/`array_split`/`vsplit`/`hsplit`/`dsplit` (`manipulation.rs`). Select/order ops route through `crate::manipulation::is_string_array` + `string_delegate`: `sort`/`argsort`/`searchsorted`/`unique`/`unique_extended`/`count_nonzero`/`partition`/`argpartition`/`lexsort`/`union1d`/`intersect1d`/`setdiff1d`/`setxor1d`/`isin`/`in1d`/`where` (`stats.rs`); `flatnonzero`/`nonzero`/`take`/`take_along_axis`/`compress`/`extract`/`argwhere` (`indexing.rs`). numpy owns the dtype passthrough, the concatenate/stack/insert width-promotion (`np.concatenate([['ab'],['cde']])` → `<U3`, live) and lexicographic sort (`np.sort(['c','a','b']) == ['a','b','c']`). Non-test consumer: the `reshape`/`sort`/`where`/`take`/… `#[pyfunction]`s registered top-level in `lib.rs` `_ferray`. Pinned green: `tests/test_expansion_string_manip.py` (59 cases — every data-move + select/order op vs the live numpy oracle, width-promotion, lexicographic sort, bytes `<S`, split lists; numeric/datetime/complex real paths unregressed). |
| REQ-3 (comparison) | NOT-STARTED | open prereq blocker #961. `less`/`greater`/`equal`/… are generated by `bind_comparison!` (`ufunc.rs`) which gates on `crate::datetime::is_time_compare` (e.g. `ufunc.rs` `pub fn less`/`equal` bodies) and `binary_involves_float16`, then falls to `comparison_body!`/`complex_eq_dispatch!` — all real-only, so `<U`/`<S` hits `match_dtype_numeric!`'s `unsupported dtype for numeric op`. No `is_string_compare` branch. Live: `fr.less(['a','b'],['b','a'])` → `unsupported dtype for numeric op: "str32"`. |
| REQ-4 (astype) | SHIPPED | `pub fn astype` in `aliases.rs` delegates fully to numpy: `as_ndarray(py,x)?` then `arr.call_method("astype",(dtype,),...)` — dtype-agnostic, so `<U`/`<S`/width/`U↔S`/`number↔string` casts are numpy's own. Non-test consumer: the `astype` `#[pyfunction]` registered top-level in `lib.rs`. numpy: `np.array([1,2,3]).astype('U3')==['1','2','3']`; `np.array(['ab'],dtype='U3').astype('S3')==[b'ab']` (numpy/_core/strings.py + ndarray.astype). Verification (live this session): `fr.astype(np.array([1,2,3]),'U3')` → `<U3 ['1','2','3']`; `fr.astype(np.array(['ab'],dtype='U3'),'S3')` → `|S3 [b'ab']`. |

(The `fr.char`/`fr.strings` string COMPUTE surface — `upper`/`lower`/`find`/
`replace`/`add`/`multiply`/predicates/split/… — is SHIPPED via #911/#912 in
`char.rs` over `ferray_strings::StringArray`, registered under `fr.char.*` and
`fr.strings.*` in `lib.rs`; this epic does not touch it. Verified live:
`fr.char.upper(np.array(['ab','cd']))` → `<U2 ['AB','CD']`.)

## Architecture

### There is NO `fr.ndarray` pyclass — the "main array" is numpy's ndarray

`fr.ndarray` re-exports `numpy.ndarray` (`lib.rs`). Every `fr.*` creation /
manipulation / comparison is a `#[pyfunction]` taking `&Bound<PyAny>` and
returning a real `numpy.ndarray`. Consequence: numpy's OWN `str_arr.astype('S3')`,
`str_a < str_b`, `np.sort(str_arr)` already work on a `<U`/`<S` ndarray because
they are numpy. ferray's job is the top-level FUNCTION surface (`fr.array`,
`fr.reshape`, `fr.less`, `fr.sort`, …).

### Strings stay as numpy ndarrays — no Rust transport

Unlike datetime (i64-backed, marshalled through `.view('int64')` into a
`DateTime64` `Element`), string arrays are a flexible (`itemsize`-parameterized)
dtype with no ferray `Element` and no `DynArray` variant
(`dynarray.rs` rejects `FixedUnicode`/`FixedAscii`/`RawBytes`, #741). They
therefore **never enter the Rust library** — every group detects the `'U'`/`'S'`
kind at the binding boundary and delegates to numpy's own op, returning the numpy
string ndarray (or bool, for comparisons) directly. This is strictly simpler than
the datetime delegate, which has to round-trip through `datetime_roundtrip`.

### Where the string kind must be threaded (binding only)

1. **`conv.rs` `match_dtype_all!` + `creation.rs` `creation_dispatch!`** (REQ-1):
   add an `is_string_array` / `dtype.kind ∈ {'U','S'}` branch ahead of the
   real-only macro in `array`/`asarray` and `zeros`/`ones`/`empty`/`full`/
   `*_like`. Construction delegates to numpy — `np.asarray(obj, dtype)` (owns the
   str-list → `<U`/`<S` width inference and the explicit `dtype='U3'`/`'S4'`
   parse) / `np.zeros(shape, dtype)` / `np.full(shape, fill, dtype)` — and the
   resulting numpy string ndarray is returned unchanged. Mirror
   `is_time_dtype_name`/`time_shape_create` (`datetime.rs`/`creation.rs`, #941),
   minus the int64-view round-trip.
2. **`manipulation.rs` + `stats.rs` data-move ops** (REQ-2): add an
   `is_string_array` branch ahead of each `match_dtype_all_complex!` / ordering
   macro in `reshape`/`ravel`/`transpose`/`swapaxes`/`moveaxis`/`rollaxis`/`flip`/
   `roll`/`repeat`/`tile`/`squeeze`/`expand_dims`/`concatenate`/`stack`/`where`/
   `unique`/`sort`/`argsort`, delegating to numpy's own op (which preserves the
   string dtype, width-promotes on concatenate/stack, and sorts lexicographically)
   and returning the numpy result directly. Mirror the datetime
   `is_time_array` + `delegate_manip`/`delegate_manip_list` seam (#948), minus the
   round-trip.
3. **`ufunc.rs` `bind_comparison!`** (REQ-3): add an `is_string_compare` branch
   (both operands `kind ∈ {'U','S'}`) ahead of the `is_time_compare` / float16 /
   real / complex paths in each of `less`/`greater`/`less_equal`/`greater_equal`/
   `equal`/`not_equal`, delegating to `np.<op>(x1, x2)` on the original operands
   (bool output, codepoint-lexicographic, broadcast). Mirror the datetime
   `is_time_compare` + `compare_time` seam (`ufunc.rs` ~`is_time_compare` branch,
   #943).

### Library prereqs (ferray-core) — none

There is **no ferray-core / ferray-ufunc / ferray-strings library work** in this
epic. The string COMPUTE library (`ferray-strings`) is already shipped and used
by `fr.char`; the main-array dtype is pure binding detect-and-delegate. No new
`Element`, no new `DynArray` variant, no new kernels. (The `#741` library
direction — a byte-buffer-backed `DynArray` string variant — is explicitly NOT
needed here and is out of scope: numpy owns the string buffer.)

### numpy main-array string semantics captured (verified live, numpy 2.4.5)

| op | numpy behavior | source |
|---|---|---|
| `np.array(['ab','cde'])` | `<U3` (width = longest) | live |
| `np.array(['ab'],dtype='U3')` | fixed `<U3` | live |
| `np.array([b'ab'],dtype='S4')` | `<S4` bytes | live |
| `np.zeros(2,dtype='U5')` | `<U5` of `''` | live |
| `np.dtype('U2').name`/`.kind` | `'str64'` / `'U'` (key off `.kind`) | live |
| `np.dtype('S4').name`/`.kind` | `'bytes32'` / `'S'` | live |
| `np.reshape(<U,(2,2))` | preserves `<U` | live |
| `np.concatenate([['a'],['bbb']])` | `<U3` (max width) | live |
| `np.sort(['c','a','b'])` | `['a','b','c']` lexicographic | live |
| `np.argsort(str)` | `int64` | live |
| `np.less(['a','b'],['b','a'])` | `bool [True,False]` | live |
| `np.equal(str,str)` | `bool` | live |
| `arr.astype('U3')` / `astype('S3')` | width / `U↔S` / num↔str cast | live (ndarray method; `fr.astype` SHIPPED) |
| `np.char.upper(['ab','cd'])` | `<U2 ['AB','CD']` | live (SHIPPED `fr.char`) |

## Incremental build plan (smallest-highest-value first)

Each group is a separate builder→critic cycle under epic #958, mirroring the
datetime sequencing (#941→#948→#943) and float16 (#951). Because there is no
library substrate and no transport, every group is thin binding threading +
pytest.

1. **R-1 (accept string input — unblocks every other group's tests). SHIPPED.**
   `array`/`asarray`/`zeros`/`ones`/`empty`/`full`/`*_like` gained a `kind ∈ {U,S}`
   branch delegating construction to numpy. Highest value: until strings are
   constructible from `fr.array`, no other group is testable end-to-end.
   - Blocker: **#959 CLOSED**. The string helpers (`is_string_array`,
     `dtype_obj_is_string`, `string_shape_create`, `string_like_dtype`,
     `fill_is_string`) live in `ferray-python/src/creation.rs` (no new module
     needed); pinned by `ferray-python/tests/test_expansion_string_construct.py`
     (38 cases). No `conv.rs` change was required (the kind-sniff is local to
     `creation.rs`).
2. **R-2 (manipulation passthrough).** `is_string_array` branch ahead of each
   real-only macro in `manipulation.rs`/`stats.rs`, delegating to numpy and
   returning the string result. Biggest risk lives here (width-promotion in
   `concatenate`/`stack`, lexicographic `sort`) — but numpy owns all of it.
   - Blocker: **#960**. Manifest: `ferray-python/src/manipulation.rs`,
     `ferray-python/src/stats.rs`, `ferray-python/src/string.rs` (the shared
     `is_string_array` helper), `ferray-python/tests/divergence_string.py`.
3. **R-3 (comparison).** `is_string_compare` branch in `bind_comparison!`
   delegating `np.<op>` → bool.
   - Blocker: **#961**. Manifest: `ferray-python/src/ufunc.rs`,
     `ferray-python/src/string.rs`, `ferray-python/tests/divergence_string.py`.

(R-4 astype is SHIPPED — numpy's own `arr.astype`; no builder.)

**Magnitude verdict:** 100% binding, **smaller than datetime** — no arithmetic
kernels, no result-dtype algebra, no int64-view transport, no library work. Three
builder→critic cycles (R-1, R-2, R-3) over the same detect-and-delegate seam the
datetime epic established; each is a thinner copy because the delegate returns
numpy's result directly. **2 open prereq blockers** (#960, #961); R-1 (#959) and
R-4 already shipped. Biggest risk: numpy's flexible-dtype edge cases —
**width-promotion** in `concatenate`/`stack` (output `<U{max}`), **bytes-vs-unicode**
(`<S` must not be coerced to `<U` or vice-versa — keep the original kind across
the boundary; mixed `U`/`S` operands follow numpy's own error), and
**lexicographic sort** by codepoint (numpy owns it; ferray must not reorder). All
three are mitigated by delegating to numpy on the original operands rather than
re-implementing.

## Verification

The gauntlet for this module (each must be green for the corresponding REQ to
move to SHIPPED in a future build commit):

- `cd ferray-python && maturin develop`
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/divergence_string.py -q`
  (pins the main-array string divergences; expected values from the live numpy
  oracle, R-CHAR-3)
- `cargo clippy -p ferray-python --all-targets -- -D warnings`
- `cargo fmt --check`

(No ferray-core / ferray-strings test changes — there is no library work.)

Per-REQ pinned pytest (added by each group's builder/critic, oracle values from
live numpy, R-CHAR-3):
- R-1: `fr.array(['ab','cd']).dtype == np.array(['ab','cd']).dtype` (`<U2`);
  `fr.array(['ab'],dtype='U3')`/`fr.array([b'ab'],dtype='S4')` round-trip;
  `fr.zeros(2,dtype='U5')`/`fr.full(2,'hi',dtype='U4')` equal numpy.
- R-2: `fr.reshape`/`fr.concatenate`/`fr.sort`/`fr.unique`/`fr.where`/`fr.repeat`/
  `fr.tile`/`fr.flip`/`fr.roll`/`fr.stack`/`fr.argsort` on string arrays equal
  numpy (dtype + values; concatenate width-promotes).
- R-3: `fr.less`/`fr.greater`/`fr.equal`/`fr.not_equal`/`fr.greater_equal`/
  `fr.less_equal` on string arrays equal numpy (bool); broadcasting
  `fr.equal(['a','b'],'a')` equals numpy.
- R-4 (SHIPPED): `fr.astype(np.array([1,2,3]),'U3')` equals numpy;
  `fr.astype(np.array(['ab'],dtype='U3'),'S3')` equals numpy.
