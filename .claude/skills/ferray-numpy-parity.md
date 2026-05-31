---
description: Use this skill when verifying NumPy parity for ferray bindings, writing parity tests, or troubleshooting numerical/dtype discrepancies between ferray and NumPy. Triggers on prompts about "match numpy", "parity with numpy", "drop-in replacement", "numpy.X equivalent", "test against numpy", "why does ferray differ from numpy", or any dtype-promotion / shape-handling question. The complementary skill is `ferray-python-binding` (which covers the binding pattern itself).
---

# NumPy parity for ferray-python

The product goal is **100% NumPy API parity** so `import ferray as np`
is a true drop-in replacement. This skill captures the parity rules
and verification patterns that recur in every binding.

## Parity targets

| Layer | Goal |
|-------|------|
| Function name | Same as NumPy (e.g. `ferray.zeros`, not `ferray.create_zeros`) |
| Argument names | Same as NumPy (e.g. `dtype=`, `axis=`, `out=`) |
| Default values | Same as NumPy when feasible |
| Result dtype | Match NumPy promotion rules — see below |
| Result shape | Bit-identical to NumPy |
| Numerical values | Within `rtol=1e-10` for f64, `rtol=1e-6` for f32 |
| Exception types | `TypeError` for dtype, `ValueError` for shape/value, `IndexError` for OOB |

When ferray-Rust diverges from NumPy (e.g. ordering of `chebpts1`,
F-contiguous-as-clone in `asfortranarray`), the **binding** does the
adjustment, not the user. Document the deviation in the docstring so
the next maintainer doesn't re-introduce it.

## Dtype promotion rules

Match NumPy's behaviour:

| Operation | Default dtype |
|-----------|---------------|
| `zeros((3,4))` | `float64` |
| `arange(5)` | `int64` (no fractional input) |
| `arange(0, 1, 0.5)` | `float64` (any fractional input) |
| `array([1, 2, 3])` | `int64` |
| `array([1.0, 2.0])` | `float64` |
| `array([True, False])` | `bool` |
| `mean(int_array)` | `float64` (always promote integer to float) |
| `var/std/percentile(int_array)` | `float64` |
| `nansum/nanmean/nanvar(int_array)` | `float64` |
| `fft.fft(real_array)` | `complex128` (or `complex64` for f32 input) |
| `fft.rfft(int_array)` | promote real to `float64`, output `complex128` |
| Index-returning fns (`argmax`, `argsort`, `nonzero`) | `int64` (NumPy `intp` on 64-bit) |
| `bincount(unweighted)` | `int64` |
| `bincount(weighted)` | `float64` |

The way to enforce these in a binding is via `coerce_dtype(py, &arr, &target_dt)?` before extraction. Example:

```rust
let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
    dt.as_str().to_string()
} else {
    "float64".to_string()
};
let arr = coerce_dtype(py, &arr, &real_dt)?;
```

## Shape rules

| Input | Output |
|-------|--------|
| Scalar (0-D) | 0-D ndarray (NumPy returns numpy scalar; we return 0-D ndarray which is interchangeable for arithmetic + printing) |
| 1-D length n + reduction with `axis=None` | 0-D ndarray |
| N-D + reduction with `axis=k` | (N-1)-D ndarray (axis k removed) |
| N-D + reduction with `keepdims=True` | N-D ndarray (axis k size 1) |
| Broadcasting binary | shape per NumPy broadcasting rules — defer to numpy.broadcast_arrays |

For the reduction case ferray returns a 0-D ndarray and NumPy returns a numpy scalar. They're arithmetically interchangeable; if a test compares with `==` use `float(result)` or `result.item()` to drop to a Python scalar.

## Pytest patterns

Place tests in `ferray-python/tests/test_<area>.py`. The standard shape:

```python
import numpy as np
import pytest
import ferray


def test_<fn>_matches_numpy():
    src = np.random.default_rng(42).standard_normal(20)
    np.testing.assert_allclose(ferray.<fn>(src), np.<fn>(src), rtol=1e-10)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_<fn>_dtype_dispatch(dtype):
    src = np.array([1, 2, 3], dtype=dtype)
    out = ferray.<fn>(src)
    assert out.dtype == np.dtype(dtype)  # or whatever the parity result is


def test_<fn>_2d_axis():
    src = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(
        ferray.<fn>(src, axis=1),
        np.<fn>(src, axis=1),
    )


def test_<fn>_invalid_dtype_raises_typeerror():
    with pytest.raises(TypeError):
        ferray.<fn>(np.array(["a", "b"]))


def test_<fn>_shape_mismatch_raises_valueerror():
    with pytest.raises(ValueError):
        ferray.<fn>(np.zeros(5), np.zeros(3))  # broadcast incompatible
```

Run with:

```bash
cd ferray-python && source .venv/bin/activate
maturin develop --release   # rebuild wheel after Rust changes
pytest -q                    # full suite, expect 525+ to pass
pytest tests/test_<area>.py -q  # just one file while iterating
```

## Numerical tolerance

| dtype | `rtol` | `atol` |
|-------|--------|--------|
| `float64` exact ops (add/sub/mul) | `0` (exact) | `0` |
| `float64` transcendental (sin/cos/exp/log) | `1e-10` | `1e-12` |
| `float64` linalg (decomp/solve/inv) | `1e-10` | `1e-10` |
| `float32` exact | `0` | `1e-6` |
| `float32` transcendental | `1e-5` | `1e-6` |

Use `np.testing.assert_allclose(actual, desired, rtol=R, atol=A)`. For
exact comparisons (integer/bool/index ops): `np.testing.assert_array_equal`.

## Documenting deviations

When ferray differs from NumPy, document it in the binding's docstring:

```rust
/// `numpy.linalg.cholesky(a)`.
///
/// **Deviation**: ferray's input must be symmetric *positive definite*
/// (NumPy raises `LinAlgError` for non-PD; ferray returns
/// `FerrayError::SingularMatrix` mapped to `ValueError`).
```

And in the test, accept the deviation explicitly (don't fudge the test
to make it pass — make the deviation visible):

```python
def test_cholesky_rejects_non_pd():
    a = np.array([[1.0, 0.0], [0.0, -1.0]])  # not PD
    with pytest.raises(ValueError):  # would be LinAlgError in numpy
        ferray.linalg.cholesky(a)
```

## Common parity gotchas (history)

These have caused failing tests in the past — check them when adding new bindings:

- **NumPy 2.0 removed `np.in1d`** — use `np.isin` in the test reference. The binding can keep the legacy name (we expose `ferray.in1d`).
- **NumPy 2.0 vector products**: `np.matmul`, `np.vecdot`, `np.matvec`, `np.vecmat` are top-level. Bindings exist; new bindings should expose them at top-level too if they're top-level in NumPy.
- **`chebpts1`/`chebpts2`**: ferray-Rust returns descending order (math convention); NumPy returns ascending. The Python binding `.reverse()`s before returning. Test via `np.polynomial.chebyshev.chebpts1(n)` and expect ascending.
- **F-contiguous round-trip**: Was broken until #698 fixed both `IntoNumPy` and `asfortranarray`. Don't relax tests that check `arr.flags["F_CONTIGUOUS"]` without verifying the fix actually broke.
- **`asfortranarray` (#698)** was a no-op clone before; `flags().f_contiguous` was always wrong. The fix: transpose → `as_standard_layout` → transpose-back.
- **complex dtype dispatch** uses string `"complex64"`/`"complex128"`, not `"c64"`. The binding `coerce_to_complex` helper picks the right pair from the input dtype.
- **`numpy.lib.functional` (vectorize/apply_along_axis/piecewise) delegate to NumPy**. Don't try to bind ferray-Rust's `apply_along_axis` (it takes a Rust closure, not a Python callable; bridging through PyO3 adds GIL hops without speedup). The pattern lives in `python/ferray/lib/__init__.py`.
- **Datetime/timedelta arithmetic**: NumPy operator overloading covers it. Don't try to bind ferray's `DateTime64` round-trip separately; expose `isnat`/`add_datetime`/`sub_datetime` as `np.*` delegates in the Python facade (#706 pattern).

## When to defer

Some NumPy surface is too niche or too tangled to bind cleanly. File a
follow-up issue with `crosslink quick "..." -p low -l feature` and
move on. Acceptable defer reasons:

- The Rust kernel has a fundamentally different signature than NumPy (e.g. takes Rust closure, returns custom enum).
- Marshalling needs a substantial new helper crate (e.g. raw byte buffer protocol for `frombuffer` binary mode).
- The function is rarely used and well-served by `numpy.<fn>` if the user reaches for it.

When deferring, keep the description honest in the issue title: "bind X
in ferray-python" not "investigate Y" — the goal is concrete work
ready to pick up later.
