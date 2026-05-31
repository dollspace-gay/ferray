# Feature: ferray-ufunc — Universal functions and SIMD-accelerated elementwise operations

## Summary
Implements NumPy's universal function (ufunc) machinery: broadcast-aware, type-promoting elementwise operations with `reduce`, `accumulate`, and `outer` methods. Covers all math (trig, exp/log, rounding, arithmetic, floating-point intrinsics, complex, bitwise, comparison, logical) plus cumulative operations, differences, convolution, interpolation, and special functions. All contiguous inner loops are SIMD-accelerated via `pulp` with runtime CPU dispatch.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Element, Dimension, broadcasting, FerrayError)
- **Downstream**: ferray-stats (uses ufunc primitives), ferray-linalg (uses arithmetic), ferray (re-export)
- **External crates**: `pulp` (runtime SIMD dispatch, stable Rust), `num-complex` (complex arithmetic), `num-traits` (Float, Zero, One bounds)
- **Phase**: 1 — Core Array and Ufuncs

## Requirements

### Ufunc Trait System
- REQ-1: Define ufunc operations as generic free functions preserving input dimensionality: `fn sin<T: Element + Float, D: Dimension>(input: &NdArray<T, D>) -> Result<NdArray<T, D>, FerrayError>`. The function is generic over the dimension `D`, so calling `sin` on an `Array2<f64>` returns `Array2<f64>`, not `ArrayD<f64>`. Do NOT use a `Ufunc` trait with `IxDyn` — that loses compile-time dimensionality.
- REQ-2: Provide a `UnaryOp` and `BinaryOp` trait for the kernel dispatch layer (SIMD vs scalar selection), but these are internal — the public API is free functions.
- REQ-3: Each ufunc supports: direct call, `_reduce(axis)` (reduction along axis), `_accumulate(axis)` (running application), `_outer()` (all pairs) as separate free functions (e.g., `add_reduce`, `add_accumulate`, `multiply_outer`)
- REQ-4: All ufuncs support broadcasting — input arrays are broadcast before the elementwise kernel runs

### Math — Trigonometric (Section 7.2)
- REQ-5: Implement as free functions and array methods: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `hypot`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `degrees`, `radians`, `deg2rad`, `rad2deg`, `unwrap`

### Math — Exponential and Logarithmic (Section 7.3)
- REQ-6: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `logaddexp`, `logaddexp2`

### Math — Rounding (Section 7.4)
- REQ-7: `round` (banker's rounding matching NumPy), `floor`, `ceil`, `trunc`, `fix`, `rint`, `around`

### Math — Arithmetic (Section 7.5)
- REQ-8: `add`, `subtract`, `multiply`, `divide`, `true_divide`, `floor_divide`, `power`, `remainder`, `mod_`, `fmod`, `divmod`, `absolute`, `fabs`, `sign`, `negative`, `positive`, `reciprocal`, `sqrt`, `cbrt`, `square`, `heaviside`, `gcd`, `lcm`
- REQ-9: Operator overloads (`+`, `-`, `*`, `/`, `%`) on `NdArray` must delegate to the corresponding ufunc implementations

### Math — Sums, Products, and Differences
- REQ-8a: `cumsum(&a, axis)` — cumulative sum along axis. `cumprod(&a, axis)` — cumulative product along axis.
- REQ-8b: `nancumsum(&a, axis)` — NaN-aware cumulative sum. `nancumprod(&a, axis)` — NaN-aware cumulative product.
- REQ-8c: `diff(&a, n, axis)` — n-th discrete difference along axis. `ediff1d(&a, to_begin, to_end)` — differences between consecutive elements with optional prepend/append.
- REQ-8d: `gradient(&a, *varargs, axis, edge_order)` — numerical gradient via central/forward/backward differences.
- REQ-8e: `cross(&a, &b, axis)` — cross product of 3-element vectors.

### Math — Integration and Convolution
- REQ-8f: `trapezoid(&y, x, dx, axis)` — trapezoidal numerical integration (NumPy 2.0 name, was `trapz`).
- REQ-8g: `convolve(&a, &v, mode)` — 1D discrete convolution with modes Full, Same, Valid.

### Math — Interpolation
- REQ-8h: `interp(x, xp, fp, left, right, period)` — 1D linear interpolation matching `np.interp`.

### Math — Special Functions
- REQ-8i: `sinc(x)` — normalized sinc function (sin(πx)/(πx)). `i0(x)` — modified Bessel function of order 0.

### Math — Floating Point Intrinsics (Section 7.6)
- REQ-10: `isnan`, `isinf`, `isfinite`, `isneginf`, `isposinf`, `nan_to_num`, `nextafter`, `spacing`, `ldexp`, `frexp`, `signbit`, `copysign`, `float_power`, `fmax`, `fmin`, `maximum`, `minimum`, `clip`
- REQ-10a: `real_if_close(&a, tol)` — if imaginary part < tol, return real array. Convenience for post-FFT cleanup.

### Math — Complex (Section 7.7)
- REQ-11: `real`, `imag`, `conj`, `conjugate`, `angle`, `abs` (returns real magnitude)

### Bitwise (Section 7.8)
- REQ-12: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `invert`, `left_shift`, `right_shift`
- REQ-13: Operator overloads (`&`, `|`, `^`, `!`, `<<`, `>>`) on integer arrays delegate to bitwise ufuncs

### Comparison (Section 7.9)
- REQ-14: `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` — all return `NdArray<bool, D>`
- REQ-15: `array_equal`, `array_equiv`, `allclose`, `isclose` — return scalar bool

### Logical (Section 7.10)
- REQ-16: `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `all`, `any`

### SIMD Acceleration (Section 19)
- REQ-17: All unary and binary elementwise operations on contiguous arrays must use SIMD paths for `f32`, `f64`, `i32`, `i64`, `u8`, `u32` via the `pulp` crate
- REQ-18: Runtime CPU dispatch: select AVX-512 > AVX2 > SSE2 (x86_64) or NEON (aarch64) at startup. Scalar fallback for other architectures.
- REQ-19: Non-contiguous arrays fall through to scalar loops. Contiguity is checked at the start of each operation.
- REQ-20: A `FERRAY_FORCE_SCALAR=1` environment variable must disable SIMD dispatch for testing

### Parallelism (Section 20)
- REQ-21: Operations on large contiguous arrays are parallelized via Rayon. Thresholds are calibrated empirically, not hardcoded — use ~100k elements for memory-bound ops (arithmetic, comparisons), ~50k for compute-bound transcendentals (sin, exp, log). Expose threshold constants so they can be tuned.
- REQ-22: Parallelism uses a ferray-owned Rayon `ThreadPool`, not the global pool

### Unary-ufunc integer/bool input promotion (NumPy dtype-resolution parity)

NumPy accepts **integer and bool** input arrays across the entire unary
float-ufunc family and the rounding family, resolving the OUTPUT dtype via
its loop-registration order (`numpy/_core/code_generators/generate_umath.py`).
ferray's unary ops are all `T: Element + Float`-bounded (see
`pub fn exp in explog.rs`, `pub fn sqrt in arithmetic.rs`,
`pub fn sin in trig.rs`, `pub fn floor in rounding.rs`, all routing through
`unary_float_op` / `unary_float_op_compute` in `helpers.rs`), so integer/bool
input is **rejected at compile time** (`i64: num_traits::Float` is unsatisfied).
Two DISTINCT contracts must be mirrored — they resolve the output dtype in
opposite ways, so they are separate REQs.

- REQ-23: **Unary float-ufunc family — integer/bool input promotes to a
  FLOATING output.** For `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`,
  `log1p`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`,
  `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `fabs`, and `rint`
  (NOT floor/ceil/trunc/round — those are REQ-24), an integer or bool input
  array produces a float output whose kind is the **smallest float that
  safe-casts from the input kind**: `bool`/`int8`/`uint8` → **float16**,
  `int16`/`uint16` → **float32**, all of `int32`/`int64`/`uint32`/`uint64` →
  **float64**. These ops register only float (and `e`/`f`/`d`/`g`) loops with
  no `TD(bints)` — int safe-casts up to the resolved float loop. Cites:
  `exp` `generate_umath.py:910-915`
  (`TD('e', dispatch=...)`, `TD('fd', dispatch=[('loops_exponent_log','fd')])`),
  `log` `generate_umath.py:935-940`, `sqrt` `generate_umath.py:968-973`
  (`TD('e', ...)`, `TD(inexactvec, ...)`, `TD('fdg'+cmplx, f='sqrt')` — no
  bints), `cbrt` `generate_umath.py:976-981` (`TD(flts, ...)`),
  `sin` `generate_umath.py:865-873`, `cos` `generate_umath.py:855-863`,
  `fabs` `generate_umath.py:1003-1008` (`TD(flts, f='fabs')`),
  `rint` `generate_umath.py:1021-1027` (`TD('e', f='rint')` FIRST — **NO
  `TD(bints)`**, which is exactly why rint diverges from floor/ceil/trunc).
  Oracle: `np.sqrt(np.array([4,9])).tolist() == [2.0, 3.0]` with
  `.dtype == float64`; `np.exp(np.uint8([1])).dtype == float16`;
  `np.sqrt(np.array([True])).dtype == float16` with value `[1.0]`;
  `np.rint(np.array([1,2,4], dtype=np.int64)).dtype == float64`.

- REQ-24: **Rounding family `floor`/`ceil`/`trunc`/`round` (and aliases
  `around`/`fix`) — integer input is INPUT-INT-DTYPE IDENTITY (with a bool
  exception for round/around).** For an integer input array these ops return
  the input values **unchanged, in the input integer dtype** (NOT promoted to
  float) — `np.round(int64)→int64`, `np.floor(int32)→int32`. NumPy registers
  `TD(bints)` FIRST for floor/ceil/trunc, so the integer loop is selected and
  int stays int. **BOOL:** `floor`/`ceil`/`trunc`/`fix` keep bool as bool
  (their `TD(bints)` loops cover bool), BUT `round`/`around` on bool **promote
  to float16** — `round`/`around` dispatch to `ndarray.round` whose bool kernel
  has no `TD(bints)` (`generate_umath.py:1021`, same as the `rint` ufunc), so
  `np.round(np.array([True,False,True]))` is `float16 [1.0,0.0,1.0]`. Cites:
  `floor` `generate_umath.py:1011-1019` (`TD(bints)` is the first registered
  loop), `ceil` `generate_umath.py:983-991` (`TD(bints)` first), `trunc`
  `generate_umath.py:993-1001` (`TD(bints)` first); `rint`
  `generate_umath.py:1021-1027` (`TD('e', f='rint')` FIRST, NO `TD(bints)` —
  the loop `round`/`around` share on bool). This is the OPPOSITE of REQ-23's
  `rint`, which has no `TD(bints)` for ANY integer and promotes every integer
  to float. ferray additionally aliases `rint = round` in `rounding.rs`
  (`pub fn rint` delegates to `pub fn round`); that alias is correct for float
  input but diverges from NumPy on integer input (NumPy `rint(int)`→float per
  REQ-23, NumPy `round(int)`→int per REQ-24), so the int paths must be split.
  Oracle: `np.floor(np.array([1,2,4], dtype=np.int64)).tolist() == [1,2,4]`
  with `.dtype == int64`; `np.ceil(np.array([5], dtype=np.int32)).dtype ==
  int32`; `np.floor(np.array([True])).dtype == bool` (bool identity);
  `np.round(np.array([True,False,True])).dtype == float16` with values
  `[1.0,0.0,1.0]` (bool round/around promote, unlike floor/ceil/trunc/fix).

### Binary-ufunc integer/bool input promotion (NumPy dtype-resolution parity)

NumPy accepts **integer and bool** input arrays across the two-argument
float-ufunc family the same way it does for the unary family (REQ-23):
these ops register only float loops (`TD(flts)` — no `TD(bints)`), so an
integer/bool input **safe-casts UP** to the smallest float loop that can
hold its kind. ferray's binary float ops are all `T: Element + Float`-bounded
with NO binary `*_promote` sibling (the unary family has `sqrt_promote` etc.;
the binary float family has none), so integer input is **rejected at compile
time** (`i64: num_traits::Float` is unsatisfied — there is no callable entry
point, see `tests/divergence_binary_promote.rs` "MISSING COMPONENT" note and
the commented-out `ferray_ufunc::hypot(&Array<i64>, …)` that fails to compile).

- REQ-25: **Binary float-ufunc family — integer/bool input promotes to a
  FLOATING output.** For `hypot`, `arctan2`, `logaddexp`, `logaddexp2`,
  `copysign`, and `nextafter`, an integer or bool input pair produces a float
  output whose kind is the **smallest float that safe-casts from the input
  kind** — identical kind-mapping to REQ-23: `bool`/`int8`/`uint8` →
  **float16**, `int16`/`uint16` → **float32**, all of
  `int32`/`int64`/`uint32`/`uint64` → **float64**. These ops register only
  float loops with no `TD(bints)`, so each integer kind safe-casts up to the
  resolved float loop. For **same-width** integer pairs (the core contract)
  the result kind follows the single-input kind rule above; for **mixed-width**
  integer pairs NumPy first promotes the two integer inputs together (NEP-50
  `result_type`) and then to float (so `int16` paired with `int32` resolves
  via the wider `int32` → `float64`). Cites:
  `hypot` `generate_umath.py:1057-1063` (`TD(flts, f='hypot', astype={'e':'f'})`
  then `TD(P, …)` — no `TD(bints)`),
  `arctan2` `generate_umath.py:1030-1037` (`TD('e', f='atan2', astype={'e':'f'})`
  FIRST, then `TD('fd', …)`/`TD('g', …)`/`TD(P, …)` — no `TD(bints)`),
  `logaddexp` `generate_umath.py:710-715` (`TD(flts, f="logaddexp",
  astype={'e':'f'})` only),
  `logaddexp2` `generate_umath.py:716-721` (`TD(flts, f="logaddexp2",
  astype={'e':'f'})` only),
  `copysign` `generate_umath.py:1107-1113` (`TD(flts)` only),
  `nextafter` `generate_umath.py:1114-1119` (`TD(flts)` only).
  Oracle (live numpy 2.4.5): `np.hypot([3,4],[4,3]).tolist() == [5.0, 5.0]`
  with `.dtype == float64`; `np.hypot(np.int16([3]), np.int16([4])).dtype ==
  float32`; `np.arctan2([1,1],[1,1]).dtype == float64`;
  `np.copysign([1,2],[-1,1]).tolist() == [-1.0, 2.0]` with `.dtype == float64`;
  `np.nextafter(np.array([1]), np.array([2])).dtype == float64`;
  `np.logaddexp([0,0],[0,0]).dtype == float64`. This mirrors REQ-23 but for two
  same-shape operands: the mechanism is the existing `PromoteFloat` trait in
  `promoted.rs` (the smallest-safe-cast-float `Compute`/`Out` mapping) extended
  with a binary `*_promote` entry-point family that casts BOTH integer/bool
  inputs to the shared compute float, runs the EXISTING generic `T: Float`
  binary ufunc (`crate::hypot`/`crate::arctan2`/`crate::logaddexp`/
  `crate::logaddexp2`/`crate::copysign`/`crate::nextafter`) monomorphised at
  the compute float, and narrows (identity for f32/f64, f32→f16 for the f16
  kinds) — leaving the existing f32/f64 float kernels byte-identical.

## Acceptance Criteria
- [ ] AC-1: All 100+ ufunc functions compile and produce correct results for `f32`, `f64`, and `Complex<f64>` inputs verified against NumPy fixtures
- [ ] AC-2: `add_reduce(&a, axis=0)` computes correct column sums. `add_accumulate(&a, axis=0)` produces running sums matching `np.add.accumulate`.
- [ ] AC-3: `multiply_outer(&a, &b)` produces correct outer product matching `np.multiply.outer`
- [ ] AC-4: Ufuncs preserve dimensionality: `sin(&array2d)` returns the same `D` dimension type as the input, not `IxDyn`
- [ ] AC-5: Broadcasting works in all ufunc calls — `sin(array_3d)`, `add(array_2d, array_1d)` both produce correct shapes
- [ ] AC-6: SIMD paths activate on contiguous `f64` arrays (verified via benchmarks showing >2x throughput vs scalar on AVX2 hardware)
- [ ] AC-7: `FERRAY_FORCE_SCALAR=1 cargo test -p ferray-ufunc` passes — scalar fallback produces identical results
- [ ] AC-8: Operator overloads `a + b`, `a * b`, `a & b` produce results identical to `add(&a, &b)`, `multiply(&a, &b)`, `bitwise_and(&a, &b)`
- [ ] AC-9: `round()` uses banker's rounding matching NumPy (e.g., `round(0.5) == 0`, `round(1.5) == 2`)
- [ ] AC-10: `cargo test -p ferray-ufunc` passes. `cargo clippy -p ferray-ufunc -- -D warnings` clean.
- [ ] AC-11: `cumsum(&[1,2,3,4])` returns `[1,3,6,10]`. `diff(&[1,3,6,10], 1)` returns `[2,3,4]`. `convolve(&[1,2,3], &[0,1,0.5], Full)` matches NumPy output.
- [ ] AC-12: `interp(2.5, &[1.0, 2.0, 3.0], &[3.0, 2.0, 0.0])` returns `1.0` matching NumPy.
- [ ] AC-13: `sinc(0.0)` returns `1.0`. `i0(0.0)` returns `1.0`.

### Acceptance Criteria — unary integer/bool input promotion
- [x] AC-14: `sqrt_promote(int64 [1,2,4])` and `exp_promote(int64 [1,2,4])`
  produce a `float64` array equal to `np.sqrt`/`np.exp` of the same integers
  (REQ-23). Pinned by `divergence_sqrt_int_promotes_to_f64` /
  `divergence_exp_int_promotes_to_f64` in `tests/divergence_unary_promote.rs`.
- [x] AC-15: `sqrt_promote(bool [T,F,T])` and `sqrt_promote(uint8 ...)` produce
  a `float16` array (verified under `--features f16` by
  `divergence_sqrt_bool_promotes_to_f16`); `int16`/`uint16` input produces
  `float32` (`divergence_sqrt_int16_uint16_promotes_to_f32`); `int32`/`int64`/
  `uint32`/`uint64` input produces `float64`
  (`divergence_sqrt_int32_promotes_to_f64`) — REQ-23 smallest-safe-cast-float
  rule encoded in `PromoteFloat::Out`.
- [x] AC-16: `rint_promote(int64 [1,2,4])` produces `float64 [1.0,2.0,4.0]`
  (REQ-23 — rint has no `TD(bints)`). Pinned by
  `divergence_rint_int_promotes_to_f64`.
- [x] AC-17: `floor_int`/`ceil_int`/`trunc_int`/`round_int` on `int64 [1,2,4]`
  return the int64 identity `[1,2,4]` (REQ-24 — NOT float, INPUT dtype
  preserved). Pinned by `divergence_floor_int_is_identity_int64`,
  `divergence_round_family_int_identity`,
  `divergence_floor_preserves_input_int_dtype`.
- [x] AC-18: The two baselines
  (`baseline_sqrt_f64_matches_numpy`, `baseline_exp_f64_matches_numpy`) stay
  green — the float64-in/float64-out contract REQ-23 preserves (the
  `*_promote` path never touches the existing float kernels).

### Acceptance Criteria — binary integer/bool input promotion
- [x] AC-19: `hypot_promote(int64 [3,4], int64 [4,3])` produces a `float64`
  array equal to `np.hypot([3,4],[4,3]).tolist() == [5.0, 5.0]`, and
  `copysign_promote(int64 [1,2], int64 [-1,1])` equals `np.copysign([1,2],
  [-1,1]).tolist() == [-1.0, 2.0]` with `.dtype == float64` (REQ-25). Pinned by
  `divergence_hypot_int_promotes_to_f64` / `divergence_copysign_int_promotes_to_f64`
  in `tests/divergence_binary_promote.rs`; arctan2/logaddexp/logaddexp2/nextafter
  int→float64 pinned alongside.
- [x] AC-20: `hypot_promote(int16 [3], int16 [4])` produces a `float32` array
  and `bool`/`int8`/`uint8` input produces a `float16` array (under
  `--features f16`) — the REQ-23 smallest-safe-cast-float kind rule reused via
  `PromoteFloat::Out`, now applied to two operands (REQ-25). Pinned by
  `divergence_hypot_int16_promotes_to_f32` and `divergence_hypot_bool_u8_promotes_to_f16`.

## REQ status

This table tracks REQs that have a verifiable SHIPPED/NOT-STARTED state against
the live numpy 2.4.5 oracle (R-DEFER-2: two states only). REQ-23 and REQ-24
were authored from the acto-critic's confirmed compile-gate finding; the
remaining REQ-1..REQ-22 describe the already-shipped float ufunc surface.

| REQ | Status | Evidence |
|---|---|---|
| REQ-23 (unary float-ufunc int/bool→float promotion) | SHIPPED | The `PromoteFloat` trait + the `*_promote` family in `promoted.rs` (`pub fn sqrt_promote`/`exp_promote`/`rint_promote`/`sin_promote`/… — 24 entry points) accept integer/bool input and promote to the smallest-safe-cast float via `PromoteFloat::Out` (`impl_promote_float_same`: i16/u16→f32, i32/i64/u32/u64→f64; `impl_promote_float_f16` feature-gated: bool/i8/u8→half::f16). The shared `fn unary_promote_float in promoted.rs` casts int→compute-float, runs the EXISTING generic `T: Float` ufunc (`crate::sqrt`/`crate::exp`/…) monomorphised at the compute float, and narrows (identity for f32/f64, f32→f16 for the f16 kinds) — so existing f32/f64 float callers are byte-identical. Re-exported from `lib.rs` (`pub use promoted::{PromoteFloat, sqrt_promote, …}`). Non-test production consumer: `pub fn rint_promote in promoted.rs` IS the production callee for the rounding crate's `rint`-on-integer contract (the REQ-24 split routes int `rint` here), and every `*_promote` is the workspace's int-accepting unary-ufunc surface (the eventual ferray-python `bind_unary_float!` dispatch target). Verified: `divergence_sqrt_int_promotes_to_f64`, `divergence_exp_int_promotes_to_f64`, `divergence_rint_int_promotes_to_f64`, `divergence_sqrt_int16_uint16_promotes_to_f32`, `divergence_sqrt_int32_promotes_to_f64`, `divergence_sin_cos_int_promote_to_f64` (12 pass default features); `divergence_sqrt_bool_promotes_to_f16` (13 pass with `--features f16`); `cargo test -p ferray-ufunc --test divergence_unary_promote` and `FERRAY_FORCE_SCALAR=1` both green; clippy `-D warnings` clean. Cites: `generate_umath.py:907-983` (`sqrt`/`exp`/`log` start at `TD('e', …)` then `TD('fdg'+cmplx)`, NO `TD(bints)`), `rint` `:1021` (`TD('e', f='rint')` FIRST, NO `TD(bints)`). |
| REQ-24 (floor/ceil/trunc/round input-int-dtype identity; bool round/around→f16) | SHIPPED | `pub fn floor_int`/`ceil_int`/`trunc_int`/`round_int`/`fix_int`/`around_int in rounding.rs` accept `T: Element + Copy` (any int/bool) and return the input UNCHANGED in the INPUT-int-dtype via the shared `fn round_identity in rounding.rs` — `np.floor(int32)→int32`, `np.round(int64)→int64`, `np.floor(bool)→bool`. NumPy registers `TD(bints)` FIRST for floor/ceil/trunc (`generate_umath.py:1011`/`:983`/`:993`), so the integer loop is selected and int stays int. BOOL EXCEPTION: `floor`/`ceil`/`trunc`/`fix` keep bool, but `round`/`around` on bool PROMOTE to float16 (`round`/`around` dispatch to `ndarray.round`, no `TD(bints)`, same loop as `rint` `:1021`) — served by `pub fn round_bool`/`around_bool in rounding.rs` (feature `f16`), which route bool→f16 via `crate::rint_promote` so `np.round(bool [T,F,T])` is `float16 [1.0,0.0,1.0]`. The OPPOSITE of `rint` (no `TD(bints)` for ANY integer, promotes per REQ-23). The float `pub fn floor`/`ceil`/`trunc`/`round`/`rint` (`T: Float`) are untouched (f32/f64 byte-identical). Re-exported from `lib.rs` (`pub use ops::rounding::{floor_int, …, round_bool, around_bool}`). Non-test production consumer: `floor_int`/`ceil_int`/`trunc_int`/`round_int`/`round_bool` ARE the integer/bool-rounding public surface re-exported by `lib.rs` (the dtype arms the ferray-python rounding dispatch routes to, mirroring `absolute_int`/`negative_int`/`sign_int` siblings in `arithmetic.rs`). Verified: `divergence_floor_int_is_identity_int64`, `divergence_round_family_int_identity`, `divergence_floor_preserves_input_int_dtype`, `divergence_rint_vs_floor_int_contract_split`, plus `divergence_round_bool_promotes_to_f16_not_bool_identity`/`divergence_around_bool_promotes_to_f16_not_bool_identity` (`--features f16`) all green. |
| REQ-25 (binary float-ufunc int/bool→float promotion: hypot/arctan2/logaddexp/logaddexp2/copysign/nextafter) | SHIPPED | The binary `*_promote` family in `promoted.rs` (`pub fn hypot_promote`/`arctan2_promote`/`logaddexp_promote`/`logaddexp2_promote`/`copysign_promote`/`nextafter_promote`, generated by the `binary_promote_fn!` macro) accepts a same-type integer/bool input pair and promotes to the smallest-safe-cast float, reusing the REQ-23 `PromoteFloat` trait UNCHANGED (`PromoteFloat::Out`: i16/u16→f32, i32/i64/u32/u64→f64; bool/i8/u8→half::f16 feature-gated). The shared `fn binary_promote_float in promoted.rs` casts BOTH operands int→compute-float (via `fn cast_to_compute`), runs the EXISTING generic `T: Float` binary ufunc (`crate::hypot`/`crate::arctan2`/`crate::logaddexp`/`crate::logaddexp2`/`crate::copysign`/`crate::nextafter`) monomorphised at the compute float, and narrows (identity for f32/f64, f32→f16 for the f16 kinds) — so existing f32/f64 float callers are byte-identical (this module never touches their code path). Mixed-width integer pairs resolve via NEP-50 then to float — confirmed live `np.hypot(np.int16([3]), np.int32([4])).dtype == float64`; the same-`T` binary kernels (numpy's `hypot`/etc. require same-type operands) cover the same-width contract, mixed-width callers cast the narrower operand up via ferray-core's `Promoted` table first. Re-exported from `lib.rs` (`pub use promoted::{hypot_promote, arctan2_promote, copysign_promote, logaddexp_promote, logaddexp2_promote, nextafter_promote}`). Non-test production consumer: these six re-exports ARE the int-accepting binary-float-ufunc public surface (the ferray-python binary-float dispatch target, sibling to the unary `sqrt_promote`/`rint_promote` family and the `add_promoted` mixed-type arithmetic surface). NumPy promotes int→float for all six (`TD(flts)`-only, no `TD(bints)`): `hypot` `generate_umath.py:1057-1063`, `arctan2` `:1030-1037`, `logaddexp` `:710-715`, `logaddexp2` `:716-721`, `copysign` `:1107-1113`, `nextafter` `:1114-1119`. Verified: `divergence_hypot_int_promotes_to_f64`, `divergence_copysign_int_promotes_to_f64`, `divergence_arctan2_int_promotes_to_f64`, `divergence_logaddexp_int_promotes_to_f64`, `divergence_logaddexp2_int_promotes_to_f64`, `divergence_nextafter_int_promotes_to_f64`, `divergence_hypot_int16_promotes_to_f32` (default features); `divergence_hypot_bool_u8_promotes_to_f16` (`--features f16`); plus `baseline_hypot_f64_matches_numpy`/`baseline_copysign_f64_matches_numpy` (preserved float invariant) — 18 pass `--features f16`, `FERRAY_FORCE_SCALAR=1` green; `cargo test -p ferray-ufunc --features f16` 693 pass / 0 fail; clippy `--all-targets --features f16 -D warnings` clean. |
| REQ-27 (ferray-python `numpy.emath`/`numpy.lib.scimath` submodule: complex-aware sqrt/log/log2/log10/logn/power/arccos/arcsin/arctanh) | SHIPPED | The PyO3 binding in `ferray-python/src/emath.rs` ships all 9 `numpy.emath` functions (`_scimath_impl.py:23-26 __all__`). Each runs numpy's domain check at the boundary, then routes IN-domain real inputs to the existing real ferray ufunc bindings (`crate::ufunc::sqrt`/`log`/`log2`/`log10`/`arccos`/`arcsin`/`arctanh`/`power` — dtype preserved) and OUT-of-domain inputs to the ferray-ufunc complex kernels (`ferray_ufunc::sqrt_complex`/`ln_complex`/`log2_complex`/`log10_complex`/`acos_complex`/`asin_complex`/`atanh_complex` + `Complex::powc`). The three domain predicates mirror numpy's `_fix_*` helpers exactly: `fn any_real_lt_zero in emath.rs` = `_fix_real_lt_zero` (`any(isreal(x) & (x<0))`, `:120` — sqrt/log/log2/log10/logn/power-base); the power exponent fix `_fix_int_lt_zero` (`:148` — promote `p`→float64, NOT complex, so `power([2,4],-2)`→`[0.25,0.0625]` not int `[0,0]`); `fn any_real_abs_gt_1 in emath.rs` = `_fix_real_abs_gt_1` (`any(isreal(x) & (abs(x)>1))`, `:176` — arccos/arcsin/arctanh). `logn(n,x)` = `log(x)/log(n)` with both legs fixed independently (`:380-382`). BRANCH-CUT CORRECTION: `num_complex`'s `acos`/`asin`/`atanh` on the exact real axis match numpy for `x<-1` but are the conjugate for `x>1`; `fn dispatch_abs_gt_1 in emath.rs` selects numpy's `x+0.0j` branch per-element via `np.where(real(x)>0, conj(r), r)` on the real-cast path only (genuinely-complex inputs match num_complex unchanged, verified). Non-test production consumer: registered as the `fr.emath` submodule in `ferray-python/src/lib.rs` (`fn register_emath_module` — `m.add_function(wrap_pyfunction!(emath::sqrt/log/.../arctanh, &m))`, `sys.modules["ferray._ferray.emath"]`) and re-exported in `ferray-python/python/ferray/__init__.py` (`emath = _emath_module`, `sys.modules["ferray.emath"]`, `__all__`) — the user-facing `import ferray as fr; fr.emath.sqrt(-1)` surface. Verified: `tests/test_expansion_emath.py` 135/135 pass after `maturin develop`; full suite 1022 passed (was 887, +135, no regression); `clippy -p ferray-python --all-targets -D warnings` clean. Cites: `_scimath_impl.py:187` (sqrt), `:243` (log), `:387` (log2), `:293` (log10), `:380-382` (logn), `:441/489-491` (power + `_fix_int_lt_zero(p)`), `:496/537` (arccos), `:543/585` (arcsin), `:591/641` (arctanh), `:32-93` (`_tocomplex`→cdouble). |
| REQ-26 (ferray-python ufunc binding: int dispatch + NEP-50 mixed promotion + scalar-return + clip-int + missing top-level fns) | SHIPPED | The PyO3 binding in `ferray-python/src/ufunc.rs` routes integer/float input through the dtype-correct library paths instead of the old float-only `match_dtype_float!`. Impl: `macro unary_float_body` (promote-to-float ops `sin`/`exp`/`sqrt`/`fabs`/`rint`) coerces int→compute-float at the boundary (`fn unary_promote_dtypes in conv.rs` = `result_type(x, float16)`) and narrows the result to numpy's exact output dtype; `macro unary_numeric_split_body` + `bind_unary_numeric_split!` route int-identity ops (`negative`/`absolute`/`sign`/`floor`/`ceil`/`trunc`/`fix`/`round`) to the `*_int` library fns; `bind_unary_float_only_int_fallback!` handles `positive`/`square`/`reciprocal` on int via `multiply_broadcast`/`floor_divide_int`. `macro binary_numeric_body` promotes BOTH operands to `fn binary_result_dtype in conv.rs` (`result_type(a,b)`), so `add([1,2],[1.5,2.5])→[2.5,4.5] float64`; `binary_numeric_split_body`+`bind_binary_numeric_split!` route `power`/`maximum`/`minimum`/`floor_divide`/`remainder`/`mod_` to float-vs-int kernels; `binary_float_promote_body` covers float-only `hypot`/`arctan2`/`fmax`/`copysign`/`float_power`; `bind_binary_int_only!` makes `gcd`/`lcm` integer-only over signed AND unsigned widths (float → TypeError; see REQ-28). Scalar/0-d inputs return a numpy scalar via `fn all_scalar_inputs`/`fn scalarize in conv.rs` (`$OUT_SCALAR`). `fn clip` accepts int arrays, array-valued bounds, and `None` one-sided bounds by delegating to `maximum`/`minimum`. `out=` kwarg on the binary numeric ops writes back via `fn finish_with_out` (numpy `copyto`). Non-test production consumer: registered top-level in `ferray-python/src/lib.rs` (`m.add_function(wrap_pyfunction!(ufunc::true_divide/floor_divide/remainder/float_power, m))` + `m.add("mod", ...)`) and re-exported from `ferray-python/python/ferray/__init__.py` (the user-facing `import ferray as np` surface). Verified: `tests/divergence_ufunc.py` 38/38 pass after `maturin develop`; full suite 603 passed (≥568); `clippy -p ferray-python --all-targets -D warnings` clean. Cites: `generate_umath.py:404` (true_divide aliased to divide), `:405` (floor_divide int+float loops), `:422` (TrueDivisionTypeResolver), `:484` (power `TD(ints)`), `:661`/`:671` (maximum/minimum int loops), `:1003` (fabs float-only), `:1156`/`:1163` (gcd/lcm `TD(ints)`), `umath.py:44-59` (`mod`/`true_divide`/`floor_divide`/`remainder`/`float_power` in `__all__`). |
| REQ-28 (gcd/lcm support UNSIGNED integer dtypes; MIN-safe) | SHIPPED | numpy registers `gcd`/`lcm` for ALL integer dtypes incl. unsigned (`generate_umath.py:1156` gcd, `:1163` lcm — `TD(ints)` covers `uint8/16/32/64`) and returns the NON-NEGATIVE gcd. Previously ferray-ufunc's `gcd_int`/`lcm_int` were `num_traits::Signed`-bounded (used ONLY for `.abs()`), so unsigned dtypes had no kernel and the binding delegated them to numpy — and `Signed::abs()` on `i*::MIN` was a latent debug-panic. RESOLVED by a crate-internal `trait GcdAbs { fn gcd_abs(self) -> Self; }` (`ferray-ufunc/src/ops/arithmetic.rs:1140`): signed `i8/i16/i32/i64` via `wrapping_abs()` (MIN-safe — `i8::MIN.gcd_abs() == i8::MIN`, matching `np.gcd(np.int8(-128), np.int8(0)) == -128`), unsigned `u8/u16/u32/u64` via identity. `gcd_int` (`arithmetic.rs:1181`) and `lcm_int` (`arithmetic.rs:1204`) now bound `T: … + GcdAbs` instead of `Signed` and call `x.gcd_abs()`; the Euclidean body is otherwise unchanged. Non-test production consumer: the PyO3 binding `macro bind_binary_int_only!` (`ferray-python/src/ufunc.rs:3589`) dispatches the promoted common dtype's UNSIGNED widths (`"uint64"=>gcd_int::<u64>` … `"uint8"=>gcd_int::<u8>`, `ufunc.rs:3629-3632`) to these kernels, registered top-level in `ferray-python/src/lib.rs` (`wrap_pyfunction!(ufunc::gcd/ufunc::lcm)`) and re-exported as `fr.gcd`/`fr.lcm`. NEP-50 promotion is applied first (`binary_result_dtype` = `result_type`), so `gcd(uint8,uint32)->uint32`, `gcd(int8,uint8)->int16`; `gcd(uint64,int64)` promotes to float64 (no gcd loop) → `TypeError` (no float gcd added). Verified: `ferray-ufunc/tests/divergence_gcd_unsigned.rs` 7/7 (unsigned u8/u32 gcd+lcm, signed-negative regression, `i8::MIN` no-panic); `tests/divergence_ufunc.py::test_gcd_unsigned_supported`/`::test_lcm_unsigned_supported` GREEN; full pytest 5095 passed 0 failed; `clippy -p ferray-ufunc`/`-p ferray-python --all-targets -D warnings` clean. Cites: `generate_umath.py:1156`/`:1163` (gcd/lcm `TD(ints)` incl. unsigned). |

## Architecture

### Crate Layout
```
ferray-ufunc/
  Cargo.toml
  src/
    lib.rs                    # Public re-exports
    dispatch.rs               # SIMD dispatch: pulp-based runtime selection
    ops/
      mod.rs
      trig.rs                 # sin, cos, tan, arc*, hyp*, deg/rad
      explog.rs               # exp, log, expm1, log1p, logaddexp
      rounding.rs             # round, floor, ceil, trunc, fix, rint
      arithmetic.rs           # add, sub, mul, div, power, sqrt, etc.
      cumulative.rs           # cumsum, cumprod, nancumsum, nancumprod
      differences.rs          # diff, ediff1d, gradient
      products.rs             # cross
      integration.rs          # trapezoid
      convolution.rs          # convolve
      interpolation.rs        # interp
      special.rs              # sinc, i0
      floatintrinsic.rs       # isnan, isinf, nan_to_num, clip, real_if_close, etc.
      complex.rs              # real, imag, conj, angle, abs
      bitwise.rs              # and, or, xor, not, shift
      comparison.rs           # eq, ne, lt, le, gt, ge, allclose
      logical.rs              # and, or, xor, not, all, any
    kernels/
      mod.rs                  # Kernel registry
      simd_f32.rs             # SIMD kernels for f32 via pulp
      simd_f64.rs             # SIMD kernels for f64 via pulp
      simd_i32.rs             # SIMD kernels for i32
      scalar.rs               # Scalar fallback for all types
    operator_overloads.rs     # impl Add, Sub, Mul, etc. for NdArray
    parallel.rs               # Rayon threshold dispatch
```

### SIMD Strategy
The `pulp` crate provides `struct Arch` with runtime CPU detection. Each kernel is written as a `pulp::WithSimd` impl that receives the detected SIMD width and dispatches accordingly. The `FERRAY_FORCE_SCALAR` env var overrides detection to always select the scalar path. Do NOT use `std::simd` — it is unstable.

### Parallelism Model
Each ufunc checks `input.len()` against a threshold constant. Above threshold, the contiguous buffer is split into chunks and processed in parallel on ferray's Rayon pool. Below threshold, single-threaded SIMD. Thresholds are exposed as `ferray::config::PARALLEL_THRESHOLD_ELEMENTWISE` etc. and should be calibrated via benchmarks, not hardcoded.

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Reduction-only functions like `sum`, `mean`, `median` (ferray-stats)
- Linear algebra operations (ferray-linalg)
- Custom user-defined ufuncs (post-1.0)
- Window functions (`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`) — handled by ferray-window
