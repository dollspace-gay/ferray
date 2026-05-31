---
description: Use this skill when adding a new ferray-python binding (PyO3 wrapper around a ferray-core/ufunc/stats/linalg/fft/random/etc. function). Triggers on user prompts about "bind X to ferray.python", "expose X in ferray python", "add Y to ferray-python", "wrap Rust function Z in pyo3", or any time the conversation involves writing code in ferray-python/src/*.rs. Skip for pure Rust work in other crates.
---

# Adding a binding to ferray-python

This skill explains the established pattern. Every binding in
`ferray-python/src/{creation,manipulation,indexing,ufunc,stats,linalg,fft,random,window,polynomial,stride_tricks,char,ma,autodiff,complex}.rs`
follows this shape; copy from the closest existing one rather than
inventing.

## Crate architecture

```
ferray-python/
├── Cargo.toml          cdylib + extension-module always on, test=false
├── pyproject.toml      maturin backend, module-name = ferray._ferray
├── python/ferray/      Pure-Python facade — re-exports + sys.modules wiring
│   ├── __init__.py     Top-level surface: from ._ferray import zeros, ...
│   ├── lib/            ferray.lib package (functional + stride_tricks)
│   └── autodiff/       ferray.autodiff package (DualNumber + helpers)
├── src/
│   ├── lib.rs          Module entry, registers every fn/class
│   ├── conv.rs         Helpers + dispatch macros (match_dtype_*)
│   ├── creation.rs etc One file per NumPy namespace
└── tests/              pytest suite, the actual verification surface
```

`cargo test -p ferray-python` is intentionally a **no-op** — the lib has
`test = false` and no test targets. Pytest is the verification surface.
Always run:

```bash
cd ferray-python && source .venv/bin/activate
maturin develop --release
pytest -q
```

## Dispatch macros (the dtype-monomorphisation primitive)

`src/conv.rs` provides four macros. Each binds a local `type T = …`
inside each match arm so the body is written once and Rust compiles
each dtype as its own monomorphic call.

| Macro | Covers | Use for |
|-------|--------|---------|
| `match_dtype_all!` | bool + i8/16/32/64 + u8/16/32/64 + f32/64 (11 types) | Generic ops: copy, transpose, where, indexing |
| `match_dtype_numeric!` | Same as `_all` minus bool | Arithmetic that needs `Add`/`Sub`/`Mul`/`Div` bounds |
| `match_dtype_orderable!` | Same as `_numeric` | Sort/argmin/min — needs `PartialOrd` |
| `match_dtype_float!` | f32/64 only | Trig/exp/log/sqrt/etc. — `T: Float` bound |
| `match_dtype_bitwise!` | bool + integers (no float) | Bitwise ops |
| `match_dtype_int_only!` | Integers only | Shifts (shift amount must be int) |

Pattern:

```rust
let arr = as_ndarray(py, input)?;
let dt = dtype_name(&arr)?;
Ok(match_dtype_all!(dt.as_str(), T => {
    let view: PyReadonlyArrayDyn<T> = arr.extract()?;
    let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let r: ArrayD<T> = ferray_core::manipulation::transpose(&fa, axes.as_deref())
        .map_err(ferr_to_pyerr)?;
    r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
}))
```

## Marshalling helpers (`conv.rs`)

- **`as_ndarray(py, obj)`** — coerce any array-like (numpy ndarray, Python list, scalar) to `numpy.ndarray` via `numpy.asarray(obj)`. Always call this first when accepting an array input.
- **`dtype_name(arr)`** — read `.dtype.name` as a `String`. The result feeds the dispatch macros.
- **`coerce_dtype(py, obj, "float64")`** — call when binding a binary op that needs both inputs at the same dtype.
- **`extract_shape(obj)`** — accept Python int OR sequence as a `Vec<usize>`. Mirrors `numpy.zeros(5)` and `numpy.zeros((3, 4))`.
- **`ferr_to_pyerr(e)`** — map `FerrayError` to the right CPython exception type. `InvalidDtype` → `TypeError`, everything else → `ValueError`.

## ferray-numpy-interop traits

For round-trips, use the existing traits:

- **`AsFerray`** — `view.as_ferray()` extracts a numpy array view into a ferray `Array<T, D>`. Available for `T: NpElement` (the 11 standard types). Complex types are NOT covered — see fft.rs for the manual round-trip helpers (`complex_pyarray_to_ferray`, `complex_ferray_to_pyarray`).
- **`IntoNumPy`** — `arr.into_pyarray(py)` consumes a ferray array and returns a `Bound<PyArrayDyn<T>>` (or 1-D / 2-D variants). Call `.into_any()` to get a `Bound<PyAny>` for the function return.

The interop layer **preserves F-contiguous layout** (#698 fix) — if the input is F-contig, the round-trip stays F-contig. Don't second-guess this.

## #[pyfunction] signatures

Every binding is `pub fn name<'py>(...) -> PyResult<Bound<'py, PyAny>>` (or `PyResult<bool>` / `PyResult<usize>` / etc. when the result is a scalar).

```rust
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn squeeze<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    // ...
}
```

Keys:
- `'py` lifetime is the bound-to-Python lifetime; PyAny references must use it.
- Default kwargs go in `#[pyo3(signature = ...)]`.
- Don't accept `&str` for paths; use `String` to avoid borrow issues across the FFI boundary.

## Registration in `lib.rs`

Every `#[pyfunction]` must be registered. Top-level functions go in the `#[pymodule] fn _ferray` body:

```rust
m.add_function(wrap_pyfunction!(creation::zeros, m)?)?;
```

Submodule functions go in the relevant `register_X_module` helper — that function creates a `PyModule::new(py, "linalg")?`, registers the functions on it, calls `parent.add_submodule(&m)?`, **and** registers the submodule in `sys.modules` so `from ferray.linalg import inv` works:

```rust
py.import("sys")?
    .getattr("modules")?
    .set_item("ferray._ferray.linalg", &m)?;
```

The Python facade in `python/ferray/__init__.py` separately registers the user-facing path:

```python
_sys.modules["ferray.linalg"] = _linalg_module
```

Without that, `from ferray.linalg import inv` fails even though `ferray.linalg.inv` works.

## #[pyclass] for stateful types

Three concrete examples in the codebase:

1. **`Polynomial`/`Chebyshev`/...** in `polynomial.rs` — generated by the `poly_class!` macro, share an interface
2. **`MaskedArray`** in `ma.rs` — float64-only, single-class
3. **`DualNumber`** in `autodiff.rs` — value type with arithmetic dunders
4. **`Generator`** in `random.rs` — owns mutable RNG state

Standard form:

```rust
#[pyclass(name = "ClassName", module = "ferray.submodule", from_py_object)]
#[derive(Clone)]    // Required for `from_py_object`. Drop it (and use
                    // skip_from_py_object instead) when wrapped state
                    // shouldn't be cloned (e.g. RNG).
pub struct PyName {
    inner: RustType,
}
```

`from_py_object` is **required** by PyO3 0.28 to use the type as a method
argument (e.g. `__add__(&self, other: &Self)`). The auto-derived
behavior was removed; opt in explicitly.

If the wrapped Rust type is **not** `Clone` (e.g. `Generator<B>` owns
RNG state and cloning silently forks the stream), use:

```rust
#[pyclass(name = "Generator", module = "ferray.random", skip_from_py_object)]
pub struct PyGenerator { inner: Generator<...> }
```

Methods take `&self` or `&mut self` per usual; PyO3 enforces that
classes-without-FromPyObject can't be passed by-value to other functions.

## Verifying a new binding

1. **`cargo clippy -p ferray-python -- -D warnings`** — must be clean.
2. **`maturin develop --release`** — rebuild the wheel into the venv.
3. **`pytest -q`** — full suite must stay green.
4. **Add a parity test** — at least one call comparing to NumPy:
   ```python
   def test_my_fn_matches_numpy():
       src = np.random.default_rng(42).standard_normal((4, 5))
       np.testing.assert_allclose(ferray.my_fn(src), np.my_fn(src), rtol=1e-10)
   ```

## Common pitfalls

- **`#[pyclass]` Clone deprecation**: PyO3 0.28 deprecated automatic FromPyObject for #[pyclass: Clone] types. Use `from_py_object` explicitly. The compile error is verbose but the fix is one keyword.
- **`PyAnyMethods::downcast` is deprecated**: use `obj.cast::<PyList>()` instead of `obj.downcast::<PyList>()`.
- **`extension-module` and cargo test**: The crate has `test = false` so `cargo test` is a no-op. Don't try to add tests under `tests/*.rs` — they'll fail at link time. Use `tests/*.py` (pytest).
- **Submodule paths in sys.modules**: Without the `python/ferray/__init__.py` `_sys.modules["ferray.X"] = _X_module` lines, `from ferray.X import Y` fails. Always update both registrations when adding a submodule.
- **NumPy 2.0 deprecations**: `np.in1d` is gone (use `np.isin`), some dtype methods moved. When writing tests, compare to the NumPy 2.0 replacement.
- **Complex types bypass NpElement**: `Complex<f32>`/`Complex<f64>` aren't covered by `AsFerray`/`IntoNumPy`. Use `crate::fft::complex_pyarray_to_ferray<T>` and `crate::fft::complex_ferray_to_pyarray<T>` (already `pub(crate)`).
- **Macros emit one body per dtype**: An expensive computation in the body of `match_dtype_all!` becomes 11 monomorphic copies. Keep bodies tight; pull setup outside the macro.
- **`as_strided` semantic mismatch**: NumPy uses byte strides, ferray uses element strides. Don't expose ferray's `as_strided` directly — wrap NumPy's instead (see `python/ferray/lib/__init__.py`).

## Where things live

| Want to... | Edit |
|------------|------|
| Add a top-level NumPy function | `src/<area>.rs` + register in `lib.rs` `#[pymodule] fn _ferray` |
| Add to a submodule | `src/<area>.rs` + register in `register_<area>_module` |
| Add a class | `src/<area>.rs` (pyclass) + `m.add_class::<...>()` in the relevant register fn |
| Surface in ferray.* | `python/ferray/__init__.py` `from ._ferray import ...` + `__all__` + (if new submodule) `_sys.modules["ferray.X"]` |
| Add a parity test | `tests/test_<area>.py` (pytest, run with `pytest -q`) |
