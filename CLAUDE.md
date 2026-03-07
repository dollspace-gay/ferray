# ferray — Project Conventions

## Rust Edition & MSRV
- Edition: 2024
- MSRV: 1.85 (stable)

## Import Paths
- Core types: `use ferray_core::{NdArray, Array1, Array2, ArrayD, ArrayView, Dimension}`
- Errors: `use ferray_core::FerrayError`
- Element trait: `use ferray_core::Element`
- Complex: `use num_complex::Complex`

## Error Handling
- All public functions return `Result<T, FerrayError>`
- Use `thiserror` 2.0 for derive
- Never panic in library code
- Every error variant carries diagnostic context

## Numeric Generics
- Element bound: `T: Element` (defined in ferray-core)
- Float-specific: `T: Element + Float` (uses num_traits::Float)
- Support f32, f64, Complex<f32>, Complex<f64>, and integer types

## SIMD Strategy
- Use `pulp` crate for runtime CPU dispatch (SSE2/AVX2/AVX-512/NEON)
- Do NOT use `std::simd` — it is unstable. If you see examples using `std::simd::f64x4`, ignore them and use `pulp` instead.
- Scalar fallback controlled by `FERRAY_FORCE_SCALAR=1` env var
- All contiguous inner loops must have SIMD paths for f32, f64, i32, i64

## Testing Patterns
- Oracle fixtures: load JSON from `fixtures/`, compare with ULP tolerance
- Property tests: `proptest` with `ProptestConfig::with_cases(256)`
- Fuzz targets: one per public function family
- SIMD verification: run all tests with FERRAY_FORCE_SCALAR=1

## Naming Conventions
- Public array type: `NdArray<T, D>` (never expose ndarray types)
- Type aliases: Array1, Array2, Array3, ArrayD
- Module structure matches NumPy: linalg::, fft::, random::, etc.

## Crate Dependencies (use these exact versions)
```toml
ndarray = "0.17"
faer = "0.24"
rustfft = "6.4"
pulp = "0.22"
num-complex = "0.4"
num-traits = "0.2"
half = "2.4"
rayon = "1.11"
serde = { version = "1.0", features = ["derive"] }
thiserror = "2.0"
```

## Code Quality Rules
- No stubs, TODOs, or `unimplemented!()` in committed code
- All public items have doc comments
- `cargo clippy -- -D warnings` must be clean
- `cargo fmt` before committing
- Read files before editing — never guess at contents

## Agent Work Protocol
- Read your assigned design doc in `.design/` first
- Implement all requirements — no partial implementations
- Run `cargo test -p <your-crate>` before finishing
- Run `cargo clippy -p <your-crate> -- -D warnings` before finishing
- Commit your work with a descriptive message
