# ferray-python

Python bindings for [ferray](https://github.com/dollspace-gay/ferray), a NumPy-equivalent library for Rust.

**Goal: 100% NumPy API parity.** `import ferray as np` should be a drop-in replacement for `import numpy as np` once the rollout is complete. The Rust core already mirrors the NumPy namespace function-for-function — the work here is binding it.

```python
import ferray as np

a = np.zeros((3, 4))            # numpy.ndarray, shape (3, 4), dtype float64
b = np.arange(0, 10)            # numpy.ndarray, [0..10), dtype int64
c = np.ones((2, 2), dtype="float32")
```

## Coverage roadmap

The full surface is bound in phases, each tracked as a child of the parent epic (issue #684):

| Phase | Issue | Surface | Status |
|------:|------:|---------|:------:|
| 0 | #685 | scaffold + `zeros`/`ones`/`arange` + `dtype` namespace | ✅ done |
| 1 | #686 | core creation, manipulation, indexing — main slice (~50 functions) | ✅ done |
| 1.x | #692-#697 | phase-1 long tail (from*, grids, pad/tile, split, take/choose) | pending |
| 2 | #687 | ufunc + stats — main slice (~95 functions) | ✅ done |
| 2.x | #699-#706 | phase-2 long tail (bitwise, complex, gcd/lcm, gradient, percentile, histogram, lexsort, datetime) | pending |
| 3 | #688 | linalg + fft + random submodules — main slice (~50 functions) | ✅ done |
| 3.x | #707-#713 | phase-3 long tail (hfft, complex/batched linalg, einsum, modern Generator, more distributions) | pending |
| 4 | #689 | window + polynomial + stride_tricks — main slice (~37 functions) | ✅ done |
| 4.1 | #717 | numpy.polynomial class API (Polynomial/Chebyshev/Hermite/HermiteE/Laguerre/Legendre) | ✅ done |
| 4.2 | #715 | numpy.char namespace (~22 vectorized string ops) | ✅ done |
| 4.3 | #716 | numpy.ma masked arrays (f64 main slice, 19 fns + MaskedArray pyclass) | ✅ done |
| 4.4 | #719 | numpy.lib functional (vectorize/apply_along_axis/apply_over_axes/piecewise) | ✅ done |
| 4.5 | #718 | ferray.autodiff DualNumber + derivative/gradient/jacobian | ✅ done |
| 4.x | #714 | as_strided binding (byte vs element strides translation) | pending |
|   | #690 | CI wheel matrix + PyPI publish | pending |
|   | #691 | Claude skills for the binding patterns | pending |
|   | #698 | ferray-numpy-interop: preserve F-contiguous layout (asfortranarray gap) | pending |

### Phase 1 surface (now bound)

**Creation**: `zeros`, `ones`, `empty`, `full`, `full_like`, `zeros_like`, `ones_like`, `eye`, `identity`, `tri`, `arange`, `linspace`, `logspace`, `geomspace`, `copy`, `array`, `asarray`, `asanyarray`, `ascontiguousarray`, `asfortranarray` *(see #698 for layout gap)*.

**Manipulation**: `reshape`, `ravel`, `flatten`, `squeeze`, `expand_dims`, `broadcast_to`, `transpose`, `swapaxes`, `moveaxis`, `rollaxis`, `flip`, `fliplr`, `flipud`, `rot90`, `roll`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `tril`, `triu`, `diag`, `diagflat`, `concatenate`, `stack`, `vstack`, `hstack`, `dstack`.

**Indexing**: `indices`, `diag_indices`, `tril_indices`, `triu_indices`, `mask_indices`, `ravel_multi_index`, `unravel_index`, `flatnonzero`, `nonzero`, `argwhere`.

### Phase 2 surface (now bound)

**Trig**: `sin`/`cos`/`tan`, `arcsin`/`arccos`/`arctan`/`arctan2`, `sinh`/`cosh`/`tanh`, `arcsinh`/`arccosh`/`arctanh`, `degrees`/`radians`/`deg2rad`/`rad2deg`, `hypot`.

**Exp/log**: `exp`/`exp2`/`expm1`, `log`/`log2`/`log10`/`log1p`, `logaddexp`/`logaddexp2`, `heaviside`.

**Arithmetic & roots**: `add`/`subtract`/`multiply`/`divide`, `power`, `sqrt`/`cbrt`/`square`/`reciprocal`, `negative`/`positive`/`absolute`/`abs`/`fabs`/`sign`, `clip`.

**Float intrinsics**: `floor`/`ceil`/`round`/`trunc`/`rint`/`fix`, `maximum`/`minimum`/`fmax`/`fmin`, `copysign`.

**Predicates**: `isnan`/`isinf`/`isfinite`/`isneginf`/`isposinf`/`signbit`.

**Comparison**: `equal`/`not_equal`/`less`/`less_equal`/`greater`/`greater_equal`.

**Logical**: `logical_and`/`logical_or`/`logical_xor`/`logical_not`.

**Reductions**: `sum`/`prod`/`mean`/`min`/`max`/`std`/`var`/`ptp`/`argmin`/`argmax`, with `axis=` and `ddof=` where applicable.

**NaN-aware**: `nansum`/`nanprod`/`nanmean`/`nanmin`/`nanmax`/`nanvar`/`nanstd`/`nanmedian`/`nanargmin`/`nanargmax`.

**Cumulative**: `cumsum`/`cumprod`/`diff`.

**Sort/search**: `sort`/`argsort`/`searchsorted`.

**Unique/set**: `unique`/`count_nonzero`/`union1d`/`intersect1d`/`setdiff1d`/`setxor1d`/`in1d`/`isin`.

**Conditional**: `where`.

### Phase 3 surface (now bound — submodules)

**`ferray.linalg`**: `norm`, `det`, `slogdet`, `matrix_rank`, `trace`, `cholesky`, `qr`, `lu`, `svd`, `eigh`, `eigvalsh`, `solve`, `inv`, `pinv`, `matrix_power`, `matmul`, `dot`, `vdot`, `inner`, `outer`, `kron`, `tensordot`.

**`ferray.fft`**: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `fftshift`, `ifftshift`, `fftfreq`, `rfftfreq`.

**`ferray.random`** (legacy module-level API, thread-local generator): `seed`, `random`, `rand`, `standard_normal`, `randn`, `normal`, `uniform`, `integers`, `randint`, `permutation`, `shuffle`, `choice`. Modern `default_rng()` Generator class deferred to #712.

### Phase 4 surface (now bound — submodules and top-level windows)

**Top-level windows** (matching NumPy): `hanning`, `hamming`, `blackman`, `bartlett`, `kaiser`. Plus `broadcast_arrays` and `broadcast_shapes`.

**`ferray.window`**: all 14 functions (`hanning`/`hamming`/`blackman`/`bartlett`/`kaiser` + SciPy extras `cosine`/`nuttall`/`parzen`/`gaussian`/`exponential`/`tukey`/`general_cosine`/`general_hamming`/`taylor`).

**`ferray.lib.stride_tricks`**: `broadcast_arrays`, `broadcast_shapes`, `sliding_window_view`. (`as_strided` is intentionally deferred to #714 — NumPy uses byte strides while ferray uses element strides; the translation deserves a careful binding.)

**`ferray.polynomial`** (function-style API): `polyvalfromroots`, `polyval2d`, `polyval3d`, `polygrid2d`, `polygrid3d`, `polyvander2d`, `polyvander3d`, `chebpts1`, `chebpts2`, `chebweight`, `chebgauss`, `leggauss`, `hermgauss`, `hermegauss`, `laggauss`, `poly2cheb`/`cheb2poly`, `poly2herm`/`herm2poly`, `poly2herme`/`herme2poly`, `poly2lag`/`lag2poly`, `poly2leg`/`leg2poly`. Class API: `Polynomial`, `Chebyshev`, `Hermite`, `HermiteE`, `Laguerre`, `Legendre` — full arithmetic dunders, `__call__`, `deriv`, `integ`, `roots`, `from_power_basis`, `fit`, `trim`/`truncate`.

### Phase 4 follow-ups (now bound)

**`ferray.char`**: `lower`, `upper`, `capitalize`, `title`, `swapcase`, `strip`/`lstrip`/`rstrip`, `count`, `find`, `startswith`, `endswith`, `str_len`, `replace`, `add`, `multiply`, `equal`/`not_equal`/`less`/`less_equal`/`greater`/`greater_equal`.

**`ferray.ma`** (f64 main slice): `MaskedArray` class with `data`/`mask`/`shape`/`ndim`/`size`/`dtype` getters, `count`/`sum`/`mean`/`min`/`max`/`var`/`std` reductions (full + axis variants), `filled`, `compressed`, `__array__` protocol. Constructors: `array`, `masked_array`, `masked_where`, `masked_invalid`, `masked_equal`/`_not_equal`/`_greater`/`_greater_equal`/`_less`/`_less_equal`, `masked_inside`, `masked_outside`. Helpers: `count_masked`, `is_masked`, `getmask`, `getdata`, `filled`, `compressed`.

**`ferray.lib`** (functional utilities): `vectorize`, `apply_along_axis`, `apply_over_axes`, `piecewise` — pure-Python wrappers around NumPy because they exist to apply user-provided callables and a Rust kernel can't speed that up. `ferray.lib.stride_tricks` (Rust-backed) sits inside the same package.

**`ferray.autodiff`** (forward-mode autodiff, ferray-specific): `DualNumber` class with `variable`/`constant` static constructors, full arithmetic dunder set (mixing with int/float), 16 elementary functions (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log2`, `log10`, `sqrt`, `abs`). High-level: `derivative(f, x)`, `gradient(f, point)`, `jacobian(f, point)`.

Each phase adds bindings until parity with the upstream NumPy version pinned in `pyproject.toml` is achieved. Anything ferray-core supports natively that NumPy lacks (e.g., autodiff dual numbers) is exposed as additional surface, never as a substitute for a NumPy-compatible function.

## Installing from PyPI

```bash
pip install ferray
```

Wheels are built for **Linux (x86_64 + aarch64)**, **macOS (Intel + Apple Silicon)**, and **Windows (x64)** across **Python 3.10–3.13** by the [`ferray-python-wheels` workflow](../.github/workflows/ferray-python-wheels.yml). Source distribution available too.

For maintainers cutting a release: see [RELEASING.md](RELEASING.md).

## Building from source

The crate is a `cdylib` driven by [maturin](https://www.maturin.rs/). From a Python venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy
cd ferray-python
maturin develop --release
python -c "import ferray; print(ferray.__version__)"
```

`maturin develop` builds the extension in-place and pip-installs it into the active venv. For a distributable wheel use `maturin build --release`.

The Rust crate stays out of the workspace's normal `cargo test` path because the `extension-module` pyo3 feature unlinks libpython. Local Rust tests should run with `cargo test -p ferray-python` (no feature flags); maturin enables the feature itself for wheel builds.

## Architecture

```
ferray-python/
├── Cargo.toml              cdylib + rlib, depends on the ferray umbrella
├── pyproject.toml          maturin build backend, project metadata
├── python/ferray/
│   └── __init__.py         Pure-Python facade re-exporting `_ferray`
├── src/
│   └── lib.rs              #[pymodule] fn _ferray — the actual bindings
└── tests/                  pytest tests run against the installed wheel
```

The compiled extension is `ferray._ferray`. The pure-Python `python/ferray/__init__.py` re-exports the symbols at the top level so users see `ferray.zeros`, not `ferray._ferray.zeros`.

## Adding a new binding

1. **Pick the ferray Rust function** you want to expose.
2. **Write a `#[pyfunction]`** in `src/lib.rs` (or a submodule file). Accept Python-friendly argument types (`&Bound<PyAny>`, `Vec<usize>`, `&str`, …) and return `PyResult<Bound<'py, PyAny>>`.
3. **Use the `creation_dispatch!` macro** for shape-creation functions (zeros / ones / empty / full / …). It dispatches a dtype string across the supported element types and forwards to the typed Rust function. To add more dtypes, edit the macro arms in one place and every binding picks them up.
4. **Convert errors** with `ferr_to_pyerr` so `FerrayError` variants land on the right CPython exception type (dtype mismatches → `TypeError`, others → `ValueError`).
5. **Convert ferray → numpy** with `IntoNumPy::into_pyarray`. Convert `numpy.ndarray` inputs the other way with `AsFerray::as_ferray`. Both traits are re-exported by `ferray-numpy-interop`.
6. **Register the function** by adding `m.add_function(wrap_pyfunction!(your_fn, m)?)?` in the `#[pymodule] fn _ferray` body.
7. **Re-export from Python** by adding the name to `python/ferray/__init__.py`.

For submodules (e.g., `ferray.linalg`), build a `PyModule::new(py, "linalg")?`, register functions on it, then `parent.add_submodule(&m)?` and also poke `sys.modules["ferray._ferray.linalg"] = m` so `from ferray.linalg import …` works.

## Memory semantics

Conversion in both directions currently **copies** the buffer. See `ferray-numpy-interop`'s docs for the full table; zero-copy is a tracked follow-up that requires sharing the raw buffer with Python's GC via a pinned refcount handshake.

## License

Dual-licensed under MIT and Apache-2.0 to match the rest of the workspace.
