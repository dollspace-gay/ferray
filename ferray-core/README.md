# ferray-core

N-dimensional array type and foundational primitives for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- `NdArray<T, D>` — owned, heap-allocated N-dimensional array (analogous to `numpy.ndarray`)
- Array views: `ArrayView`, `ArrayViewMut`, `ArcArray`, `CowArray`
- Type aliases: `Array1`, `Array2`, `Array3`, `ArrayD` (dynamic rank)
- Broadcasting (NumPy rules), basic/advanced/extended indexing, `s![]` macro
- Array creation: `zeros`, `ones`, `arange`, `linspace`, `eye`, `meshgrid`, `full`, etc.
- Shape manipulation: `reshape`, `transpose`, `concatenate`, `stack`, `split`, `pad`, `tile`
- `Element` trait for 17 dtypes: f16, f32, f64, Complex, i8-i128, u8-u128, bool
- `DType` runtime enum, `finfo`/`iinfo`, type promotion rules
- `FerrayError` hierarchy with diagnostic context — zero panics

## Usage

```rust
use ferray_core::prelude::*;

let a = Array2::<f64>::zeros(Ix2::new([3, 4]))?;
let b = Array1::<f64>::linspace(0.0, 1.0, 100)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate. Most users should depend on `ferray` directly.

## License

MIT OR Apache-2.0
