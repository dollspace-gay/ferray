---
description: Use this skill when writing or modifying Rust code in any of the ferray workspace crates (ferray-core, ferray-ufunc, ferray-stats, ferray-linalg, ferray-fft, ferray-random, ferray-window, ferray-polynomial, ferray-strings, ferray-ma, ferray-stride-tricks, ferray-autodiff, ferray-numpy-interop, ferray-io, ferray). Triggers on prompts about adding/modifying Rust functions, fixing FerrayError variants, working with Element trait or dimension types, SIMD with pulp, or crate-level conventions. Skip for ferray-python (use `ferray-python-binding` skill instead).
---

# ferray Rust crate conventions

This skill captures the patterns that the ferray workspace expects of
new Rust code — the bare minimum a contributor needs to know before
adding to `ferray-core` or any of its sibling crates.

The authoritative document is `CLAUDE.md` in the workspace root; this
skill summarises the parts that come up most often when writing code.

## Edition + MSRV

- Edition: **2024**
- MSRV: **1.85** stable
- `cargo fmt` and `cargo clippy -- -D warnings` must both be clean before commit.

## Workspace shape

```
ferray (umbrella)
├── ferray-core               Array<T, D>, dim types, FerrayError, dtype
├── ferray-core-macros        promoted_type!, s! slice-info macros
├── ferray-ufunc              elementwise math, comparison, logical, reductions
├── ferray-stats              statistical reductions, histogram, sort, set ops
├── ferray-linalg             decomp, solve, norms, products, einsum, complex
├── ferray-fft                rustfft + realfft wrappers, shift/freq helpers
├── ferray-random             pluggable bitgen + ~30 distributions
├── ferray-polynomial         power + 5 orthogonal bases (Chebyshev/Hermite/...)
├── ferray-window             14 windows + functional utilities (vectorize/apply_along_axis)
├── ferray-strings            StringArray + 30+ vectorized string ops
├── ferray-ma                 MaskedArray + ~50 mask-aware ops
├── ferray-stride-tricks      broadcast_arrays/_shapes, sliding_window_view, as_strided
├── ferray-autodiff           DualNumber + forward-mode helpers
├── ferray-numpy-interop      AsFerray/IntoNumPy + Arrow/Polars bridges
├── ferray-io                 .npy/.npz/text/memmap I/O
└── ferray-test-oracle        oracle-fixture testing infrastructure
```

Add to existing crates rather than creating new ones unless you're
introducing a new top-level NumPy submodule.

## Import paths (canonical)

```rust
use ferray_core::{NdArray, Array1, Array2, ArrayD, ArrayView, Dimension};
use ferray_core::{FerrayError, FerrayResult, Element};
use ferray_core::dimension::{Axis, Ix1, Ix2, Ix3, IxDyn};
use num_complex::Complex;
```

Don't use `use ferray_core::array::owned::Array` — the type alias is
`NdArray<T, D>` (or `Array<T, D>` re-exported at crate root).

## Element trait + dtype matrix

`Element` (in `ferray-core`) is the bound for "anything that can live
in an `Array<T, D>`". The standard impls are:

| Type | `Element` | `NpElement` | Notes |
|------|:---------:|:-----------:|-------|
| `bool` | ✅ | ✅ | |
| `i8`/`i16`/`i32`/`i64` | ✅ | ✅ | |
| `u8`/`u16`/`u32`/`u64` | ✅ | ✅ | |
| `f32`/`f64` | ✅ | ✅ | |
| `Complex<f32>`/`Complex<f64>` | ✅ | ❌ | NpElement excludes complex; manual round-trip in interop |
| `f16`/`bf16` (half) | ✅ behind `f16`/`bf16` features | ✅ | Rare |
| `DateTime64`/`Timedelta64` | ✅ | ❌ | ferray-specific i64 wrappers |

When writing a generic function, choose the tightest bound that the
implementation actually needs:

| Bound | When |
|-------|------|
| `T: Element` | Elementwise copy, transpose, where, indexing |
| `T: Element + Copy` | Most things — Copy is cheap and avoids spurious clone overhead |
| `T: Element + Add<Output = T> + Copy` | Sum, cumsum, any binary numeric op |
| `T: Element + PartialOrd + Copy` | Min/max, sort, argmin |
| `T: Element + Float` | Trig, exp, log, sqrt, special functions |
| `T: LinalgFloat` | Anything in ferray-linalg — sealed to f32/f64 |
| `T: FftFloat` | Anything in ferray-fft — sealed to f32/f64 |

## Error handling

Every public function returns `FerrayResult<T>` (which is
`Result<T, FerrayError>`). Variants:

```rust
FerrayError::ShapeMismatch { reason: String }      // shape problem
FerrayError::InvalidDtype { from: DType, to: DType }  // dtype mismatch
FerrayError::InvalidValue(String)                  // generic bad input
FerrayError::IndexOutOfBounds { index, axis, size }  // OOB access
FerrayError::AxisOutOfBounds { axis, ndim }
FerrayError::DimensionMismatch { ... }
FerrayError::SingularMatrix                        // linalg
FerrayError::BroadcastFailure { shape_a, shape_b }
// ... see ferray-core/src/error.rs for the full list
```

Use the helper constructors:

```rust
return Err(FerrayError::shape_mismatch(format!("expected {expected:?}, got {actual:?}")));
return Err(FerrayError::invalid_value("step must be nonzero"));
return Err(FerrayError::axis_out_of_bounds(axis, ndim));
```

**Never `panic!`/`unwrap()`/`expect()` in library code** for input
validation — every error carries diagnostic context. Internal
unreachable-by-construction conditions can use `expect("doc-explanation")`.

## SIMD strategy

- Use the `pulp` crate for runtime CPU dispatch (SSE2/AVX2/AVX-512/NEON).
- Do **NOT** use `std::simd` (unstable). If you see examples using `std::simd::f64x4`, ignore them and use `pulp`.
- Scalar fallback is controlled by `FERRAY_FORCE_SCALAR=1` env var.
- All contiguous inner loops should have SIMD paths for f32, f64, i32, i64.
- Run all tests with `FERRAY_FORCE_SCALAR=1` to verify the scalar path works.

```rust
use pulp::Arch;

fn add_simd(a: &[f64], b: &[f64], out: &mut [f64]) {
    Arch::new().dispatch(|| {
        // Inside this closure, pulp gives you SIMD intrinsics that
        // the runtime arch supports.
        for ((a, b), out) in a.iter().zip(b).zip(out.iter_mut()) {
            *out = a + b;
        }
    });
}
```

For new ufuncs, follow the pattern in `ferray-ufunc/src/kernels/` —
there's a `scalar.rs` fallback alongside `simd_f32.rs` and `simd_f64.rs`.

## Naming conventions

- Public array type is **`NdArray<T, D>`** at the umbrella level. Inside `ferray-core` and most crates it's `Array<T, D>`. **Never** expose `ndarray::Array` — wrap in `Array::from_ndarray(...)` before returning.
- Type aliases: `Array1<T>`, `Array2<T>`, `Array3<T>`, `ArrayD<T>` (= `Array<T, IxDyn>`).
- Module structure mirrors NumPy: `linalg::`, `fft::`, `random::`, `polynomial::`, `ma::`, `strings::`, etc.
- Functions match NumPy names verbatim. Prefer `xy_indices` over `xy_indexing_array` even if more explicit; consistency with NumPy beats internal preference.

## Crate dependencies (pinned versions)

Always reference `[workspace.dependencies]` in the root `Cargo.toml`
rather than declaring direct versions in member crates:

```toml
ndarray = "0.17"
faer = "0.24"
rustfft = "6.4"
pulp = "0.22"
num-complex = "0.4"
num-traits = "0.2"
half = "2.7"
rayon = "1.12"
serde = { version = "1.0", features = ["derive"] }
thiserror = "2.0"
pyo3 = "0.28"
numpy = "0.28"
```

Use `workspace = true` in member `Cargo.toml`s:

```toml
[dependencies]
ferray-core = { workspace = true }
ndarray = { workspace = true }
```

## Testing patterns

Three test layers:

1. **Unit tests** (`#[cfg(test)] mod tests` inside each module): test the function in isolation.
2. **Oracle fixtures** (`fixtures/*.json`): load JSON oracle data and compare with ULP tolerance. Use `ferray-test-oracle` helpers.
3. **Property tests** (`proptest`): for invariants — `proptest! { #[test] fn fn_round_trips(...) }` with `ProptestConfig::with_cases(256)`.
4. **Fuzz targets** (`fuzz/`): one per public function family, exercising the full input space.

Always run with both SIMD and scalar paths:

```bash
cargo test -p <crate>
FERRAY_FORCE_SCALAR=1 cargo test -p <crate>
```

For numerical functions, compare against NumPy via the oracle fixtures
in `fixtures/`. Don't write hand-tuned constants — generate fixtures
with a Python script that exercises a known reference (numpy/scipy/etc.).

## Documentation

- All public items get doc comments. The first line is a one-sentence summary; following paragraphs explain.
- `# Errors` section enumerates every `FerrayError` variant the function can return, with conditions. Workspace convention: variants are documented on the type, not on every returning function — but if a function only returns a *subset*, listing them helps callers.
- `# Examples` blocks are doctests; they get run on `cargo test`. Use `///` example blocks for any non-trivial public API.

## What NOT to do

- **No stubs/TODOs/`unimplemented!()`** in committed code. CI will reject. If a function isn't done, file a crosslink issue and don't merge until it is.
- **Don't expose ndarray types** at public API boundaries. Always wrap in `Array<T, D>`.
- **Don't add `pub use ferray_core::*;`** from re-export points without auditing — this can pollute crate roots and break semver.
- **Don't add `#[allow(...)]` to suppress warnings**. Either fix the code or, if the lint is wrong for the case, add a targeted `#[allow]` on the specific item with a comment explaining why.
- **Don't use `std::simd`** — unstable. Use `pulp`.
- **Don't introduce new dependencies** without checking they're already in `[workspace.dependencies]`. The pinned versions are deliberate.

## Agent work protocol

When asked to do a Rust-side change:

1. Read the assigned design doc in `.design/` first if one exists.
2. Implement the requirements fully — no partial work.
3. Run `cargo test -p <crate>` and `cargo clippy -p <crate> -- -D warnings`.
4. Run `FERRAY_FORCE_SCALAR=1 cargo test -p <crate>` if the change touches a SIMD path.
5. Commit with a descriptive message tying the change to the crosslink issue.
