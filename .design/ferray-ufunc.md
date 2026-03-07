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
