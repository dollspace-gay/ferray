# Design Document: `ferray` — A NumPy Equivalent for Rust

**Status:** Proposal  
**Version:** 0.1.0-draft  
**Target Audience:** Rust library authors, scientific computing engineers, numerical programmers  
**Companion document:** `rust-ml-design.md` (`ferrolearn`) — `ferray` is its foundational dependency

---

## 1. Motivation

NumPy is not a machine learning library. It is the substrate that everything else is built on. Its power comes from three things: a single, expressive N-dimensional array type; a complete and internally consistent mathematical function library operating on that type; and a unified, one-import-covers-everything experience that means a practitioner never has to think about which crate handles FFTs versus which one handles random number generation versus which one handles linear algebra.

The Rust ecosystem has fragments of this, but not the whole. `ndarray` provides the core array type and basic arithmetic. `faer` provides high-performance linear algebra. `rustfft` provides FFTs. `ndarray-linalg` wraps LAPACK (but requires the user to install system libraries). `ndarray-rand` provides random array generation. `ndarray-stats` provides statistics. These are all good crates, but they are not NumPy — they are a pile of parts that require the user to assemble them, manage version compatibility between them, and hold the mental model of which namespace to look in for which operation.

This separation of concerns causes friction for new users — where NumPy requires a single `import numpy`, using the equivalent Rust stack requires referring between documentation for several crates maintained by different sets of developers.

`ferray` closes this gap. It is a single crate providing the complete NumPy surface area, with Rust-native improvements where the language makes them possible. It does not replace `ndarray` or `faer` — it wraps and unifies them into a coherent whole that a practitioner can depend on without also depending on six other crates. `ferrolearn` depends on `ferray` as its array primitive.

---

## 2. Design Philosophy

`ferray` differs from `ferrolearn` in a critical way: NumPy is a *primitive*, not a toolkit. Where `ferrolearn` algorithms are composable units that sit on top of an array layer, `ferray` IS the array layer. Its design philosophy follows from this:

**Completeness over minimalism.** Partial coverage of NumPy is the current state of the art and it isn't good enough. `ferray` must cover all of NumPy's primary namespaces in a single dependency.

**One import.** `use ferray::prelude::*` should cover 95% of use cases, just as `import numpy as np` does in Python.

**No mandatory system dependencies.** The default build must compile on any Rust target without requiring OpenBLAS, LAPACK, or any system library. BLAS/LAPACK acceleration is opt-in via feature flags.

**Rust improvements over NumPy where possible.** The borrow checker, type generics, and const generics let `ferray` make guarantees at compile time that NumPy cannot — dimensionality mismatches, type errors, and mutability violations should be compiler errors, not runtime exceptions. Where Rust can prove correctness statically, it must.

**Numerical parity with NumPy.** The same six-layer correctness stack from `ferrolearn` applies here in full.

---

## 3. Relationship to Existing Crates

`ferray` is not a fork or replacement of any existing crate. It is a unification layer.

| Existing crate | Role in `ferray` |
|---|---|
| `ndarray` | Used internally as the storage engine for `ferray-core`. `ferray::Array` is **not** a newtype or re-export — it is an owned type with its own public API. Conversion traits (`From<ndarray::Array>`, `Into<ndarray::Array>`) are provided for interoperability, but `ndarray` does not appear in `ferray`'s public type signatures. This insulates users from ndarray version churn and avoids the diamond-dependency problem that would arise if ndarray were part of the public ABI. |
| `faer` | Powers all BLAS-level operations internally via the `ferray::linalg` module |
| `rustfft` | Powers `ferray::fft` internally |
| `ndarray-linalg` | Superseded by `ferray::linalg`, which wraps `faer` directly rather than LAPACK |
| `ndarray-rand` | Superseded by `ferray::random`, which implements NumPy's Generator/BitGenerator model |
| `ndarray-stats` | Superseded by `ferray`'s statistics methods on arrays |
| `num-complex` | `Complex<f32>` and `Complex<f64>` are the complex element types throughout |
| `half` | `f16` support (for ML workloads) |

Users who have existing code using `ndarray` can interoperate with `ferray` directly: `ferray::Array` implements `From<ndarray::Array>` and `Into<ndarray::Array>`. These conversions are zero-copy where memory layouts are compatible. Because `ndarray` is not part of the public API surface, users do not need to add `ndarray` as a direct dependency — `ferray` manages the version internally.

---

## 4. Core Array Type

### 4.1 `NdArray<T, D>`

The central type. Generic over element type `T` and dimensionality `D`.

```rust
pub struct NdArray<T, D: Dimension> {
    data: OwnedRepr<T>,   // or ViewRepr, MutViewRepr, ArcRepr, CowRepr
    shape: D,
    strides: D,
    layout: MemoryLayout,
}

pub enum MemoryLayout {
    C,           // Row-major (C-contiguous) — default
    Fortran,     // Column-major (Fortran-contiguous)
    Custom,      // Arbitrary strides (views, non-contiguous slices)
}
```

Type aliases mirror NumPy's naming conventions:

```rust
pub type Array1<T> = NdArray<T, Ix1>;
pub type Array2<T> = NdArray<T, Ix2>;
pub type Array3<T> = NdArray<T, Ix3>;
pub type ArrayD<T> = NdArray<T, IxDyn>;  // Dynamic rank, like np.ndarray

// Float aliases
pub type F32Array1 = Array1<f32>;
pub type F64Array2 = Array2<f64>;
// ... etc
```

### 4.2 Ownership Model

Unlike NumPy — which allows multiple arrays to mutably alias the same data, tracked only at runtime — `ferray` expresses ownership in the type system:

| Type alias | Ownership | NumPy equivalent |
|---|---|---|
| `Array<T, D>` | Owned, uniquely | `np.ndarray` with no views alive |
| `ArrayView<'a, T, D>` | Borrowed immutably | Read-only slice/view |
| `ArrayViewMut<'a, T, D>` | Borrowed mutably | `np.ndarray` view |
| `ArcArray<T, D>` | Ref-counted, copy-on-write | Shared ndarray |
| `CowArray<'a, T, D>` | Either owned or borrowed | — |

**`ArcArray` copy-on-write semantics:** `ArcArray` wraps the data buffer in an `Arc`. A mutation (any operation requiring `&mut` access to elements) triggers a clone of the buffer if the `Arc` reference count is greater than 1, then proceeds on the cloned buffer. Views (`ArrayView`) derived from an `ArcArray` hold a clone of the `Arc` — they keep the buffer alive and prevent CoW from being triggered on the *view*. If the originating `ArcArray` is mutated after a view is taken, the view continues to see the old data (it holds the old `Arc`). There is no silent invalidation. The invariant: a live `ArrayView<'a, ..>` derived from an `ArcArray` always observes the data that existed at the moment the view was created, regardless of subsequent mutations to the source.

This is a strict improvement over NumPy. Aliasing bugs — where two views of the same array are modified in conflicting ways — are a known source of subtle NumPy errors that `ferray` makes impossible at compile time.

### 4.3 Dtype System

NumPy's dtype system is one of its most powerful features and the hardest to replicate in Rust. `ferray` must support all NumPy numeric dtypes as Rust generic parameters:

**Floating point:** `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`  
**Signed integers:** `i8`, `i16`, `i32`, `i64`, `i128`  
**Unsigned integers:** `u8`, `u16`, `u32`, `u64`, `u128`  
**Boolean:** `bool`  
**String:** `StringArray` — a separate type backed by `Vec<String>` (see Section 10)

Structured dtypes (NumPy's `np.dtype([('x', np.float64), ('y', np.int32)])`) are supported via a derive macro:

```rust
#[derive(FerrayRecord)]
struct Point {
    x: f64,
    y: f64,
    label: i32,
}

let points: Array1<Point> = ferray::zeros(100);  // array-of-structs layout: [x0,y0,label0, x1,y1,label1, ...]
let xs: ArrayView1<f64> = points.field("x");     // zero-copy strided view of the x field
```

### 4.4 Broadcasting

`ferray` implements NumPy's full broadcasting rules, including the cases that `ndarray` currently cannot handle. Broadcasting is implemented generically over dimension type and does not require materializing the broadcast array unless the operation demands it.

Broadcasting rules (identical to NumPy):
1. If arrays have different numbers of dimensions, prepend `1`s to the shape of the smaller-dimensioned array
2. Arrays with size `1` along a dimension are stretched to match the other
3. Arrays whose sizes disagree and neither is `1` produce a `ShapeError`

```rust
let a: Array2<f64> = ferray::ones((4, 3));
let b: Array1<f64> = ferray::array([1.0, 2.0, 3.0]);
let c = &a + &b;  // Broadcasts b from (3,) to (4, 3) — works
let d: Array2<f64> = ferray::ones((4, 1));
let e = &a * &d;  // Broadcasts d from (4, 1) to (4, 3) — works

// Binary broadcasting between two non-trivial shapes:
let x: Array3<f32> = ferray::zeros((2, 1, 4));
let y: Array2<f32> = ferray::ones((3, 4));
let z = &x + &y;  // z.shape() == [2, 3, 4]
```

### 4.5 Indexing

`ferray` must support all NumPy indexing modes:

**Basic indexing** — integer and slice indexing, always returns a view:
```rust
let a = ferray::arange(0.0f64, 12.0, 1.0).reshape((3, 4));
let row = a.slice(s![1, ..]);        // row 1
let col = a.slice(s![.., 2]);        // column 2
let sub = a.slice(s![..2, 1..3]);    // submatrix
```

**Advanced (fancy) indexing** — integer array or boolean array indexing, always returns a copy:
```rust
let idx = ferray::array([0usize, 2]);
let rows = a.index_select(Axis(0), &idx);  // select rows 0 and 2

let mask: Array2<bool> = &a > &ferray::scalar(5.0);
let selected = a.boolean_index(&mask);     // 1D array of elements where mask is true
a.boolean_index_assign(&mask, 0.0);        // np.ndarray[mask] = 0
```

**`np.newaxis` / `None` indexing:**
```rust
let v: Array1<f64> = ferray::ones(5);
let col: Array2<f64> = v.insert_axis(Axis(1));  // shape (5,) → (5, 1)
```

---

## 5. Array Creation

All NumPy array creation routines must be present in the `ferray` top-level namespace:

### 5.1 From Data
```rust
ferray::array([1.0, 2.0, 3.0])               // np.array(...)
ferray::asarray(&existing_vec)               // np.asarray(...)
ferray::frombuffer(&bytes, dtype: f32)       // np.frombuffer(...)
ferray::fromiter(iter, count: 100)           // np.fromiter(...)
```

### 5.2 Ones, Zeros, Empty
```rust
ferray::zeros((3, 4))                        // np.zeros
ferray::ones((3, 4))                         // np.ones
ferray::full((3, 4), 7.5f64)                // np.full
ferray::zeros_like(&other)                  // np.zeros_like
ferray::ones_like(&other)
ferray::full_like(&other, 0.0)

// np.empty — uninitialized allocation.
// Returns Array<MaybeUninit<T>, D>; caller must initialize all elements
// before calling .assume_init() to obtain Array<T, D>.
// This forces explicit acknowledgment of uninitialized state rather than
// silently handing back a typed array with garbage values.
ferray::empty((3, 4))                        // → Array2<MaybeUninit<f64>>
ferray::empty_like(&other)                   // → Array<MaybeUninit<T>, D>
```

### 5.3 Ranges and Grids
```rust
ferray::arange(0.0, 10.0, 0.5)              // np.arange
ferray::linspace(0.0, 1.0, 50)             // np.linspace
ferray::logspace(0.0, 3.0, 10)             // np.logspace
ferray::geomspace(1.0, 1000.0, 4)          // np.geomspace
ferray::meshgrid(&[&x, &y])                // np.meshgrid
ferray::mgrid(s![0.0..1.0:0.1])            // np.mgrid
ferray::ogrid(s![0.0..1.0:0.1])            // np.ogrid
```

### 5.4 Identity and Diagonal
```rust
ferray::eye::<f64>(4)                       // np.eye(4)
ferray::identity::<f64>(4)                  // np.identity(4)
ferray::diag(&v, k: 0)                      // np.diag
ferray::diagflat(&v)                        // np.diagflat
ferray::tri::<f64>(3, 3, 0)                // np.tri
ferray::tril(&a, k: 0)                      // np.tril
ferray::triu(&a, k: 0)                      // np.triu
```

---

## 6. Array Manipulation

### 6.1 Shape Manipulation
```rust
a.reshape((2, 6))                           // a.reshape(...)
a.ravel()                                   // a.ravel() — contiguous 1D view
a.flatten()                                 // a.flatten() — owned copy
a.squeeze()                                 // remove length-1 axes
a.expand_dims(axis: 1)                      // np.expand_dims
a.broadcast_to((4, 3, 2))                   // np.broadcast_to — zero-copy
```

### 6.2 Joining and Splitting
```rust
ferray::concatenate(arrays: &[&dyn AnyArray], axis: 0)   // np.concatenate
ferray::stack(arrays: &[&dyn AnyArray], axis: 0)          // np.stack
ferray::vstack(arrays: &[&dyn AnyArray])                  // np.vstack
ferray::hstack(arrays: &[&dyn AnyArray])                  // np.hstack
ferray::dstack(arrays: &[&dyn AnyArray])                  // np.dstack
ferray::block(nested: &[&[&dyn AnyArray]])                // np.block

a.split(3, axis: 0)                                       // np.split
a.array_split(5, axis: 0)                                 // np.array_split
a.vsplit(3)
a.hsplit(3)
a.dsplit(3)
```

### 6.3 Transposing and Reordering
```rust
a.transpose()                               // a.T
a.swapaxes(0, 1)                            // np.swapaxes
a.moveaxis(source: 0, destination: 2)      // np.moveaxis
a.rollaxis(axis: 2, start: 0)              // np.rollaxis
a.flip(axis: 0)                             // np.flip
a.fliplr()
a.flipud()
a.rot90(k: 1)
a.roll(shift: 3, axis: 0)                  // np.roll
```

---

## 7. Mathematical Functions (ufuncs)

NumPy's universal function (ufunc) machinery is its core performance primitive — elementwise operations that broadcast, type-promote, and support output arrays. `ferray` must implement the complete ufunc surface.

### 7.1 The `Ufunc` Trait

The ufunc trait is generic over input and output element types. The `reduce`, `accumulate`, and `outer` methods use associated types so that, for example, a `u8` sum ufunc produces `u64` accumulations rather than attempting to operate on `f64`.

```rust
/// A universal function: a broadcast-aware, type-promoting elementwise operation.
pub trait Ufunc {
    /// The input element type.
    type In: Element;
    /// The output element type (may differ — e.g., comparison ufuncs produce bool).
    type Out: Element;

    fn call(&self, input: &NdArray<Self::In, IxDyn>) -> NdArray<Self::Out, IxDyn>;
    fn call_into(&self, input: &NdArray<Self::In, IxDyn>, out: &mut NdArray<Self::Out, IxDyn>);

    /// Reduce along an axis: e.g., `add.reduce(a, axis=0)` → cumulative sum.
    /// Output type may differ from `In` (e.g., integer sum accumulates to i64).
    fn reduce(&self, a: &NdArray<Self::In, IxDyn>, axis: usize)
        -> NdArray<Self::Out, IxDyn>;

    /// Accumulate along an axis: running application of the operation.
    fn accumulate(&self, a: &NdArray<Self::In, IxDyn>, axis: usize)
        -> NdArray<Self::Out, IxDyn>;

    /// Outer product: apply the operation to all pairs of elements.
    fn outer(
        &self,
        a: &NdArray<Self::In, IxDyn>,
        b: &NdArray<Self::In, IxDyn>,
    ) -> NdArray<Self::Out, IxDyn>;
}
```

Binary ufuncs (e.g., `add`, `multiply`) are expressed as `BinaryUfunc` with two input type parameters:

```rust
pub trait BinaryUfunc {
    type Lhs: Element;
    type Rhs: Element;
    type Out: Element;

    fn call(
        &self,
        lhs: &NdArray<Self::Lhs, IxDyn>,
        rhs: &NdArray<Self::Rhs, IxDyn>,
    ) -> NdArray<Self::Out, IxDyn>;
    // reduce, accumulate, outer follow the same pattern
}
```

Type promotion (e.g., `f32 + i64` → `f64`) is handled at the call site by the promotion rules defined in Section 16.1, not inside the trait itself.

### 7.2 Math — Trigonometric
All functions are available both as free functions and as array methods.

`sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `hypot`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `degrees`, `radians`, `deg2rad`, `rad2deg`, `unwrap`

### 7.3 Math — Exponential and Logarithmic
`exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `logaddexp`, `logaddexp2`

### 7.4 Math — Rounding
`round` (banker's rounding, matching NumPy), `floor`, `ceil`, `trunc`, `fix`, `rint`, `around`

### 7.5 Math — Arithmetic
`add`, `subtract`, `multiply`, `divide`, `true_divide`, `floor_divide`, `power`, `remainder`, `mod_`, `fmod`, `divmod`, `absolute`, `fabs`, `sign`, `negative`, `positive`, `reciprocal`, `sqrt`, `cbrt`, `square`, `heaviside`, `gcd`, `lcm`

### 7.6 Math — Floating Point Intrinsics
`isnan`, `isinf`, `isfinite`, `isneginf`, `isposinf`, `nan_to_num`, `nextafter`, `spacing`, `ldexp`, `frexp`, `signbit`, `copysign`, `float_power`, `fmax`, `fmin`, `maximum`, `minimum`, `clip`

### 7.7 Math — Complex
`real`, `imag`, `conj`, `conjugate`, `angle`, `abs` (returns real magnitude)

### 7.8 Bitwise
`bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `invert`, `left_shift`, `right_shift`

### 7.9 Comparison
`equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `array_equal`, `array_equiv`, `allclose`, `isclose`

### 7.10 Logical
`logical_and`, `logical_or`, `logical_xor`, `logical_not`, `all`, `any`

---

## 8. Linear Algebra (`ferray::linalg`)

Mirrors `numpy.linalg` completely. Internally powered by `faer` with an optional BLAS backend.

### 8.1 Matrix Products
```rust
linalg::dot(&a, &b)          // np.dot
linalg::vdot(&a, &b)         // np.vdot (complex conjugate)
linalg::inner(&a, &b)        // np.inner
linalg::outer(&a, &b)        // np.outer
linalg::matmul(&a, &b)       // np.matmul / @
linalg::tensordot(&a, &b, axes: &[(0,1)])   // np.tensordot
linalg::einsum("ij,jk->ik", &[&a, &b])     // np.einsum — full Einstein notation parser
linalg::kron(&a, &b)         // np.kron
```

The `einsum` implementation is a hard requirement. It is used pervasively in scientific computing, ML, physics simulations, and signal processing, and its absence is a significant gap in the current Rust ecosystem.

### 8.2 Decompositions
```rust
linalg::cholesky(&a)                         // np.linalg.cholesky → L
linalg::qr(&a, mode: QrMode::Reduced)       // np.linalg.qr → (Q, R)
linalg::svd(&a, full_matrices: false)       // np.linalg.svd → (U, S, Vt)
linalg::eig(&a)                              // np.linalg.eig → (eigenvalues, eigenvectors)
linalg::eigh(&a)                             // np.linalg.eigh — symmetric/Hermitian
linalg::eigvals(&a)                          // eigenvalues only
linalg::eigvalsh(&a)
linalg::lu(&a)                               // LU with partial pivoting → (P, L, U)
```

### 8.3 Solving and Inversion
```rust
linalg::solve(&a, &b)                        // np.linalg.solve — Ax = b
linalg::lstsq(&a, &b, rcond: None)          // np.linalg.lstsq — least squares
linalg::inv(&a)                              // np.linalg.inv
linalg::pinv(&a, rcond: None)               // np.linalg.pinv — Moore-Penrose pseudoinverse
```

### 8.4 Norms and Measures
```rust
linalg::norm(&a, ord: NormOrder::Fro)       // np.linalg.norm
linalg::cond(&a, p: None)                   // np.linalg.cond
linalg::det(&a)                              // np.linalg.det
linalg::slogdet(&a)                          // np.linalg.slogdet → (sign, logdet)
linalg::matrix_rank(&a, tol: None)          // np.linalg.matrix_rank
linalg::trace(&a)                            // np.trace
```

### 8.5 Stacked Array Operations

A key improvement over NumPy: all `linalg` functions that accept 2D arrays must also accept stacked arrays (3D+) and apply the operation along the last two axes, parallelized via Rayon. NumPy added this progressively and inconsistently; `ferray` requires it for all linalg functions from day one.

```rust
let batch: Array3<f64> = ferray::random::standard_normal((100, 4, 4));
let det: Array1<f64> = linalg::det(&batch);  // 100 determinants, parallelized
```

---

## 9. Fast Fourier Transform (`ferray::fft`)

Mirrors `numpy.fft` completely. Internally powered by `rustfft`.

```rust
fft::fft(&a, n: None, axis: -1, norm: FftNorm::Backward)     // np.fft.fft
fft::ifft(&a, n: None, axis: -1, norm: FftNorm::Backward)    // np.fft.ifft
fft::fft2(&a, s: None, axes: (-2, -1))                        // np.fft.fft2
fft::ifft2(&a, s: None, axes: (-2, -1))                       // np.fft.ifft2
fft::fftn(&a, s: None, axes: None)                            // np.fft.fftn
fft::ifftn(&a, s: None, axes: None)                           // np.fft.ifftn

// Real FFTs (input is real, output is Hermitian-symmetric — more efficient)
fft::rfft(&a, n: None, axis: -1)                              // np.fft.rfft
fft::irfft(&a, n: None, axis: -1)                             // np.fft.irfft
fft::rfft2(&a, s: None, axes: (-2, -1))
fft::irfft2(&a, s: None, axes: (-2, -1))
fft::rfftn(&a, s: None, axes: None)
fft::irfftn(&a, s: None, axes: None)

// Hermitian input
fft::hfft(&a, n: None, axis: -1)                              // np.fft.hfft
fft::ihfft(&a, n: None, axis: -1)                             // np.fft.ihfft

// Frequencies
fft::fftfreq(n: 128, d: 1.0f64)                               // np.fft.fftfreq
fft::rfftfreq(n: 128, d: 1.0f64)                              // np.fft.rfftfreq
fft::fftshift(&a, axes: None)                                  // np.fft.fftshift
fft::ifftshift(&a, axes: None)                                 // np.fft.ifftshift
```

**Plan caching:** `rustfft` uses planning to optimize FFTs for repeated calls with the same size. `ferray::fft` must expose a `FftPlan` type that caches the plan and can be reused:

```rust
let plan = fft::FftPlan::new(1024);
for signal in signals.iter() {
    let spectrum = plan.execute(signal);
}
```

---

## 10. Random Number Generation (`ferray::random`)

Mirrors `numpy.random`'s modern Generator API (not the legacy `np.random.*` module functions). This is one of the most commonly needed and most scattered parts of the current Rust ecosystem.

### 10.1 Generator and BitGenerator

```rust
// NumPy-style: create a Generator backed by a BitGenerator
let rng = random::default_rng();                     // np.random.default_rng()
let rng = random::default_rng_seeded(42u64);         // np.random.default_rng(42)

// Explicit BitGenerators
let rng = random::Generator::new(random::Pcg64::new(seed));
let rng = random::Generator::new(random::Philox::new(seed));
let rng = random::Generator::new(random::Xoshiro256StarStar::new(seed));
```

`Generator` methods take `&mut self` — the generator is stateful and not `Sync`. Generating large arrays in parallel (e.g., a 10M-element normal sample) uses per-thread generators derived from the root via jump-ahead or stream-splitting:

```rust
// Parallel generation: internally creates per-Rayon-thread generators via
// BitGenerator::jump() (Xoshiro256**) or stream IDs (Philox).
// The resulting array is identical regardless of thread count — deterministic
// given the same root seed and output size.
let samples = rng.standard_normal_parallel((1_000_000,));

// Manual splitting for user-controlled parallelism:
let child_rngs: Vec<Generator> = rng.spawn(num_threads);
```

The `jump()` operation advances the generator state by 2^128 steps, providing non-overlapping subsequences for each thread. Only BitGenerators that support jump-ahead (`Xoshiro256StarStar`, `Philox`) are eligible for parallel generation; `Pcg64` falls back to sequential generation with a warning.

### 10.2 Distributions — Continuous

```rust
rng.random(size: (3, 4))                         // Uniform [0, 1) — np.random.random
rng.uniform(low: 0.0, high: 1.0, size: (100,))  // np.random.uniform
rng.standard_normal(size: (100, 5))              // np.random.standard_normal
rng.normal(loc: 0.0, scale: 1.0, size: (100,))  // np.random.normal
rng.standard_exponential(size: (100,))
rng.exponential(scale: 1.0, size: (100,))
rng.gamma(shape: 2.0, scale: 1.0, size: (100,))
rng.beta(a: 2.0, b: 5.0, size: (100,))
rng.chisquare(df: 5.0, size: (100,))
rng.f(dfnum: 5.0, dfden: 2.0, size: (100,))
rng.student_t(df: 10.0, size: (100,))
rng.laplace(loc: 0.0, scale: 1.0, size: (100,))
rng.logistic(loc: 0.0, scale: 1.0, size: (100,))
rng.lognormal(mean: 0.0, sigma: 1.0, size: (100,))
rng.rayleigh(scale: 1.0, size: (100,))
rng.weibull(a: 5.0, size: (100,))
rng.pareto(a: 3.0, size: (100,))
rng.gumbel(loc: 0.0, scale: 1.0, size: (100,))
rng.power(a: 0.5, size: (100,))
rng.triangular(left: -3.0, mode: 0.0, right: 8.0, size: (100,))
rng.vonmises(mu: 0.0, kappa: 4.0, size: (100,))
rng.wald(mean: 3.0, scale: 2.0, size: (100,))
rng.standard_cauchy(size: (100,))
rng.standard_gamma(shape: 2.0, size: (100,))
```

### 10.3 Distributions — Discrete

```rust
rng.integers(low: 0, high: 10, size: (100,))    // np.random.integers
rng.binomial(n: 10, p: 0.5, size: (100,))
rng.negative_binomial(n: 5, p: 0.5, size: (100,))
rng.poisson(lam: 5.0, size: (100,))
rng.geometric(p: 0.5, size: (100,))
rng.hypergeometric(ngood: 5, nbad: 10, nsample: 5, size: (100,))
rng.logseries(p: 0.9, size: (100,))
rng.multinomial(n: 20, pvals: &[0.2, 0.3, 0.5], size: 100)
rng.multivariate_normal(&mean, &cov, size: 100)
rng.dirichlet(&alpha, size: 100)
```

### 10.4 Permutations and Sampling

```rust
rng.shuffle(&mut a)                              // np.random.shuffle — in-place
rng.permutation(&a)                              // np.random.permutation — returns copy
rng.permuted(&a, axis: 0)                        // np.random.permuted — along axis
rng.choice(&a, size: 10, replace: false, p: None) // np.random.choice
```

---

## 11. Polynomial Arithmetic (`ferray::polynomial`)

Mirrors `numpy.polynomial` completely. Provides polynomial representations with stable numerical behavior.

### 11.1 Polynomial Classes

Each polynomial class represents a different basis:

```rust
// Power basis (standard): p(x) = c[0] + c[1]*x + c[2]*x² + ...
let p = polynomial::Polynomial::new(&[1.0, -3.0, 2.0]);  // np.polynomial.polynomial.Polynomial

// Chebyshev (better conditioned for approximation)
let c = polynomial::Chebyshev::new(&coeffs);              // np.polynomial.chebyshev.Chebyshev

// Legendre
let l = polynomial::Legendre::new(&coeffs);               // np.polynomial.legendre.Legendre

// Laguerre
let lag = polynomial::Laguerre::new(&coeffs);

// Hermite (physicist's and probabilist's)
let h = polynomial::Hermite::new(&coeffs);
let he = polynomial::HermiteE::new(&coeffs);
```

### 11.2 Polynomial Operations

All polynomial classes implement a common `Poly` trait. Basis conversion uses a canonical pivot — every type converts through `Polynomial` (power basis) — avoiding the coherence hole that would arise if `convert<Q>` required each `Poly` to know about every other.

```rust
/// Conversion through the power basis as canonical pivot.
pub trait ToPowerBasis {
    fn to_power_basis(&self) -> Polynomial;
}

pub trait FromPowerBasis: Sized {
    fn from_power_basis(p: &Polynomial) -> Self;
}

// Blanket: any two Poly types that go through power basis can convert freely.
impl<P: ToPowerBasis, Q: FromPowerBasis> From<P> for Q {
    fn from(p: P) -> Q {
        Q::from_power_basis(&p.to_power_basis())
    }
}

pub trait Poly: Sized + ToPowerBasis + FromPowerBasis {
    fn eval(&self, x: f64) -> f64;                    // p(x)
    fn deriv(&self, m: usize) -> Self;                // p'(x), p''(x), ...
    fn integ(&self, m: usize, k: &[f64]) -> Self;    // ∫p(x)dx
    fn roots(&self) -> Array1<Complex<f64>>;          // companion matrix eigenvalues
    fn degree(&self) -> usize;
    fn trim(&self, tol: f64) -> Self;                 // remove trailing near-zero coefficients
    fn truncate(&self, size: usize) -> Self;

    // Arithmetic (all return the appropriate type)
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn pow(&self, n: usize) -> Self;
    fn divmod(&self, other: &Self) -> (Self, Self);

    // Fitting
    fn fit(x: &Array1<f64>, y: &Array1<f64>, deg: usize) -> Self;
    fn fit_weighted(x: &Array1<f64>, y: &Array1<f64>, deg: usize, w: &Array1<f64>) -> Self;
}
```

---

## 12. String Operations (`ferray::strings`)

Mirrors `numpy.strings` (NumPy 2.0+). Vectorized string operations on arrays of strings, operating elementwise with broadcasting.

```rust
let a: StringArray1 = ferray::strings::array(["hello", "world", "foo"]);
let b: StringArray1 = ferray::strings::array(["HELLO", "WORLD", "BAR"]);

strings::add(&a, &b)                    // elementwise concatenation
strings::multiply(&a, 3)               // "hellohellohello"
strings::upper(&a)                      // np.strings.upper
strings::lower(&b)                      // np.strings.lower
strings::capitalize(&a)
strings::title(&a)
strings::center(&a, width: 10, fillchar: ' ')
strings::ljust(&a, width: 10)
strings::rjust(&a, width: 10)
strings::zfill(&a, width: 8)
strings::strip(&a, chars: None)
strings::lstrip(&a, chars: None)
strings::rstrip(&a, chars: None)
strings::replace(&a, old: "l", new: "L", count: None)
strings::startswith(&a, prefix: "h")   // → Array1<bool>
strings::endswith(&a, suffix: "o")     // → Array1<bool>
strings::find(&a, sub: "ll")           // → Array1<i64>
strings::count(&a, sub: "l")           // → Array1<usize>
strings::split(&a, sep: " ")           // → Array1<Vec<String>>
                                        //   ragged output: each element is a Vec of the split parts.
                                        //   ferray does not have a jagged array type; Array1<Vec<String>>
                                        //   is the Rust-idiomatic representation for this case.
strings::join(sep: "-", sequence: &a)  // → Array1<String>

// Regex support
strings::match_(&a, pattern: r"\w+")   // → Array1<bool>
strings::extract(&a, pattern: r"(\w+)") // → StringArray
```

---

## 13. Statistics Functions

These are array methods and free functions in the top-level `ferray` namespace, not a submodule:

```rust
a.sum(axis: None)                              // np.sum
a.prod(axis: None)                             // np.prod
a.min(axis: None)                              // np.min
a.max(axis: None)                              // np.max
a.argmin(axis: None)                           // np.argmin
a.argmax(axis: None)                           // np.argmax
a.mean(axis: None)                             // np.mean
a.var(axis: None, ddof: 0)                    // np.var
a.std(axis: None, ddof: 0)                    // np.std
a.median(axis: None)                           // np.median
a.percentile(&a, q: 50.0, axis: None)         // np.percentile
a.quantile(&a, q: 0.5, axis: None)            // np.quantile

// NaN-aware variants (critical for real data)
a.nansum(axis: None)                           // np.nansum
a.nanprod(axis: None)
a.nanmin(axis: None)
a.nanmax(axis: None)
a.nanmean(axis: None)
a.nanvar(axis: None, ddof: 0)
a.nanstd(axis: None, ddof: 0)
a.nanmedian(axis: None)
a.nanpercentile(&a, q: 50.0, axis: None)

// Correlations and covariance
ferray::correlate(&a, &v, mode: CorrelateMode::Full)  // np.correlate
ferray::corrcoef(&x, rowvar: true)                    // np.corrcoef
ferray::cov(&m, rowvar: true, ddof: 1)                // np.cov

// Histograms
ferray::histogram(&a, bins: 10, range: None, density: false)   // np.histogram
ferray::histogram2d(&x, &y, bins: 10)                          // np.histogram2d
ferray::histogramdd(&sample, bins: 10)                         // np.histogramdd
ferray::bin_count(&x, weights: None, minlength: 0)             // np.bincount
ferray::digitize(&x, &bins, right: false)                      // np.digitize

// Sorting and searching
a.sort(axis: -1, kind: SortKind::Stable)               // np.sort
a.argsort(axis: -1)                                    // np.argsort
ferray::searchsorted(&a, &v, side: Side::Left)         // np.searchsorted
ferray::unique(&a, return_index: false, return_counts: false)   // np.unique
ferray::nonzero(&a)                                    // np.nonzero
ferray::where_(&condition, &x, &y)                     // np.where
ferray::count_nonzero(&a, axis: None)                  // np.count_nonzero
```

---

## 14. Set Operations

```rust
ferray::union1d(&a, &b)                        // np.union1d
ferray::intersect1d(&a, &b, assume_unique: false)  // np.intersect1d
ferray::setdiff1d(&a, &b, assume_unique: false)    // np.setdiff1d
ferray::setxor1d(&a, &b, assume_unique: false)     // np.setxor1d
ferray::in1d(&a, &test, assume_unique: false)      // np.in1d
ferray::isin(&element, &test_elements)             // np.isin
```

---

## 15. I/O (`ferray::io`)

Mirrors `numpy.lib.npyio` — serialization to `.npy` and `.npz` formats, and text I/O.

```rust
// Binary formats
io::save("array.npy", &a)?;                    // np.save
io::load::<f64, _>("array.npy")?               // np.load — caller specifies dtype at compile time
io::load_dynamic("array.npy")?                 // → Result<DynArray, FerrayError>
                                               //   DynArray is an enum over all supported dtypes;
                                               //   use when the file's dtype is not known ahead of time
io::savez("arrays.npz", &[("x", &x), ("y", &y)])?  // np.savez
io::savez_compressed("arrays.npz.gz", ...)?   // np.savez_compressed

// Text formats
io::savetxt("data.csv", &a, delimiter: b',', fmt: "%.6f")?   // np.savetxt
io::loadtxt::<f64>("data.csv", delimiter: b',', skiprows: 1)?  // np.loadtxt
io::genfromtxt("data.csv", delimiter: b',', filling_values: f64::NAN)?  // np.genfromtxt

// Memory mapping for large arrays
io::memmap::<f64>("large_array.npy", mode: MemmapMode::ReadOnly)?  // np.memmap
```

`DynArray` is defined as:

```rust
pub enum DynArray {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    Bool(ArrayD<bool>),
    Complex64(ArrayD<Complex<f32>>),
    Complex128(ArrayD<Complex<f64>>),
    // ... all supported dtypes
}
```

`io::load::<T>()` is a convenience that calls `load_dynamic` internally and returns `Err(FerrayError::InvalidDtype { .. })` if the file's dtype does not match `T`.

---

## 16. Type Casting and Type Inspection

### 16.1 Type Promotion Rules

`ferray` follows NumPy's promotion rules for mixed-type operations. The rule is: the output type is the smallest type that can represent all values of both input types without loss of precision.

| Lhs | Rhs | Result |
|---|---|---|
| `f32` | `f64` | `f64` |
| `f32` | `i64` | `f64` |
| `f64` | `i64` | `f64` |
| `i32` | `i64` | `i64` |
| `i32` | `f32` | `f64` (i32 doesn't fit in f32 without loss) |
| `u8` | `i16` | `i16` |
| `Complex<f32>` | `f64` | `Complex<f64>` |
| `bool` | `i32` | `i32` |

Mixed-type binary operations on arrays do **not** compile implicitly. The caller must either call `.astype()` to convert explicitly, or use `ferray::add_promoted(&a, &b)` which returns `NdArray<promoted_type!(A, B), _>`. The `promoted_type!` macro resolves at compile time where both types are known statically; for dynamic dtypes it resolves at runtime via `ferray::result_type`.

This is a deliberate difference from NumPy, where `np.array([1], dtype=np.int32) + np.array([1.0], dtype=np.float32)` silently promotes. Silent promotion is a footgun when the intent was to operate on same-type arrays — `ferray` surfaces it as a type error and provides an explicit escape hatch.

### 16.2 Cast and Inspect Functions

```rust
a.astype::<f32>()                              // a.astype(np.float32)
a.view::<u8>()                                 // reinterpret memory layout
ferray::result_type::<f32, i64>()             // np.result_type
ferray::can_cast::<f32, f64>(CastKind::Safe)  // np.can_cast
ferray::common_type(&[dtype1, dtype2])         // np.common_type
ferray::min_scalar_type(3.14f64)              // np.min_scalar_type
ferray::promote_types::<i32, f64>()           // np.promote_types

// Type checking predicates
ferray::issubdtype::<f32, FloatingPoint>()    // np.issubdtype
ferray::isrealobj(&a)                          // np.isrealobj
ferray::iscomplexobj(&a)                       // np.iscomplexobj
```

---

## 17. Stride Tricks and Views (`ferray::lib::stride_tricks`)

`as_strided` is marked `unsafe` because it can construct overlapping views of the same memory, which is unsound with Rust's aliasing rules if the resulting view is used mutably. This is an explicit exception to `ferray`'s general guarantee that aliasing bugs are impossible — it is the intentional escape hatch for operations (windowed convolution, Toeplitz construction, overlapping tiles) that require it. The safety contract is documented at the call site.

```rust
// SAFE: non-overlapping strides, equivalent to a reshape
let view = stride_tricks::as_strided(&a, shape: (3, 3), strides: (8, 8));

// UNSAFE: overlapping strides — caller must ensure no concurrent mutation
//         through any alias of the same memory region
let overlapping: ArrayView2<f64> = unsafe {
    stride_tricks::as_strided_unchecked(&a, shape: (3, 3), strides: (4, 4))
};

// SAFE variants (no unsafe required):
stride_tricks::sliding_window_view(&a, window_shape: (3,))      // np.lib.stride_tricks.sliding_window_view — read-only views only
stride_tricks::broadcast_to(&a, shape: (10, 3, 4))              // np.broadcast_to — always read-only
ferray::broadcast_arrays(&[&a, &b])                             // np.broadcast_arrays
ferray::broadcast_shapes(&[(3, 1), (1, 4)])                     // np.broadcast_shapes
```

---

## 18. Masked Arrays (`ferray::ma`)

Masked arrays allow representing missing or invalid data inline with the array — equivalent to `numpy.ma`.

```rust
let data = ferray::array([1.0, 2.0, f64::NAN, 4.0, 5.0]);
let mask = ferray::array([false, false, true, false, false]);
let ma = ma::MaskedArray::new(data, mask);

ma.mean()                    // ignores masked elements
ma.filled(fill_value: 0.0)  // replace masked with fill value
ma.compressed()             // 1D array of unmasked elements
ma.count()                  // number of valid (unmasked) elements
ma::masked_where(&condition, &a)   // mask where condition is true
ma::masked_invalid(&a)             // mask NaN and Inf
ma::masked_equal(&a, value: 0.0)  // mask where equal to value
ma::masked_greater(&a, 5.0)
ma::masked_less(&a, 0.0)
```

---

## 19. SIMD Acceleration

All inner loops that operate on contiguous memory must be SIMD-accelerated. This is not optional — it is a performance correctness requirement.

**Runtime CPU dispatch is required.** Compile-time SIMD width selection alone is insufficient: a binary compiled for the `x86_64` baseline target would not use AVX2 even on a machine that has it. `ferray` uses the `multiversion` crate (or equivalent `#[target_feature]` + runtime detection) to ship one binary that selects the widest available SIMD path at startup via `cpuid`:

- 512-bit (AVX-512) if available
- 256-bit (AVX2) if available
- 128-bit (SSE2 / NEON) as the baseline fallback
- Scalar for non-x86/aarch64 targets

Compile-time `target_feature` flags can still lock to a specific width for embedded or controlled-environment deployments where the target is fully known.

The SIMD implementation pattern:

```rust
// Internal implementation of vectorized addition
fn add_f64_contiguous(a: &[f64], b: &[f64], out: &mut [f64]) {
    use std::simd::f64x4;
    
    let chunks = a.len() / 4;
    let (a_chunks, a_tail) = a.split_at(chunks * 4);
    let (b_chunks, b_tail) = b.split_at(chunks * 4);
    let (out_chunks, out_tail) = out.split_at_mut(chunks * 4);

    // SIMD main loop: 4 doubles at a time
    for ((ac, bc), oc) in a_chunks.chunks_exact(4)
        .zip(b_chunks.chunks_exact(4))
        .zip(out_chunks.chunks_exact_mut(4))
    {
        let va = f64x4::from_slice(ac);
        let vb = f64x4::from_slice(bc);
        (va + vb).copy_to_slice(oc);
    }

    // Scalar tail
    for ((av, bv), ov) in a_tail.iter().zip(b_tail).zip(out_tail) {
        *ov = av + bv;
    }
}
```

**Requirements:**
- All unary and binary elementwise operations on contiguous arrays must use SIMD paths for `f32`, `f64`, `i32`, `i64`, `u8`, `u32`
- The SIMD width adapts at compile time to the target architecture: 128-bit (SSE2/NEON), 256-bit (AVX2), or 512-bit (AVX-512)
- Non-contiguous arrays fall through to scalar loops; contiguity is checked at the start of each operation
- `criterion` benchmarks must verify that SIMD paths achieve at least 80% of theoretical peak memory bandwidth on contiguous inputs

---

## 20. Parallelism

`ferray` uses Rayon for automatic parallelism on operations that are large enough to benefit. The threshold for parallel dispatch is calibrated so that small arrays do not incur threading overhead.

```rust
// Parallel operations are transparent to the user:
let large: Array2<f64> = ferray::random::standard_normal((10_000, 10_000));
let result = (&large * 2.0 + 1.0).mapv(f64::exp);  // auto-parallelized via Rayon

// Thread count control uses a ferray-owned thread pool, not Rayon's global pool.
// This avoids the "first initializer wins" problem when multiple crates share the
// Rayon global pool. ferray initializes its own rayon::ThreadPool at first use.
ferray::set_num_threads(4);

// Scoped serial execution via a per-call ThreadPool:
ferray::with_num_threads(1, || { /* serial execution — uses a 1-thread pool */ });
// Note: with_num_threads creates a temporary ThreadPool per call; avoid in hot paths.
```

Parallel dispatch thresholds are calibrated empirically per operation class and architecture during the benchmark suite, not hardcoded. The initial defaults (subject to tuning):
- Elementwise ops (memory-bandwidth bound): parallel above ~10M elements on typical 4-core hardware
- Transcendental ufuncs (`exp`, `log`, `sin` — compute-bound): parallel above ~1M elements
- Reductions (`sum`, `min`, `max`): parallel above ~10k elements with tree-reduce
- Matrix multiplication: always parallel via `faer`'s threading model

Thresholds are exposed as configurable constants (`ferray::config::PARALLEL_THRESHOLD_ELEMENTWISE`, etc.) so downstream users can tune for their hardware. All parallel code must be deterministic when floating-point associativity is not required; reductions may differ in operand ordering across parallel runs — this is documented explicitly.

---

## 21. Memory Layout and Interoperability

### 21.1 Buffer Protocol

`ferray` arrays must implement a safe buffer protocol allowing zero-copy interoperation with other memory-owning types:

```rust
pub trait AsRawBuffer {
    fn as_ptr(&self) -> *const u8;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn dtype(&self) -> DType;
    fn is_c_contiguous(&self) -> bool;
    fn is_f_contiguous(&self) -> bool;
}
```

### 21.2 PyO3 / NumPy Interop

A `ferray-numpy` companion crate (feature-flagged, not part of the main crate) provides zero-copy conversion between `ferray::Array` and NumPy arrays via PyO3:

```rust
#[pyo3::pyfunction]
fn process(py: Python, np_array: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let arr: ferray::Array2<f64> = np_array.as_ferray()?;  // zero-copy
    let result = arr.mapv(f64::sqrt);
    Ok(result.into_pyarray(py).into())
}
```

### 21.3 Arrow and Polars Interop

`ferray` arrays must be convertible to/from Apache Arrow `ArrayData` with zero copy where memory layouts are compatible:

```rust
let arrow_array: arrow::array::Float64Array = ferray_array.to_arrow()?;
let ferray_array: ferray::Array1<f64> = arrow_array.into_ferray()?;
```

---

## 22. Error Handling

All public functions return `Result<T, FerrayError>`. Panics are forbidden in library code.

```rust
#[non_exhaustive]
pub enum FerrayError {
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    BroadcastFailure { lhs: Vec<usize>, rhs: Vec<usize> },
    AxisOutOfBounds { axis: usize, ndim: usize },
    IndexOutOfBounds { index: usize, size: usize },
    SingularMatrix,
    ConvergenceFailure { iterations: usize },
    InvalidDtype { expected: DType, got: DType },
    NumericalInstability { context: String },
    IoError(std::io::Error),
    InvalidValue { context: String },
}
```

---

## 23. Correctness Verification

The full six-layer correctness stack from `ferrolearn` Section 20 applies to `ferray` in its entirety, with the following `ferray`-specific additions:

**Oracle fixture generation** must use NumPy's output as ground truth. For every function, fixtures must cover all supported dtypes (`f32`, `f64`, `Complex<f64>`, relevant integer types), all valid shapes from 0-D to 5-D, and all edge cases (empty arrays, single-element arrays, NaN/Inf inputs, very large and very small values).

**ufunc correctness** requires special attention: broadcasting must be verified to produce bit-identical results to NumPy for all valid broadcast combinations, not just the simple cases. A fixture generator must enumerate a representative set of broadcast shape pairs for each ufunc.

**SIMD path verification**: both the SIMD and scalar code paths must be tested against the same fixture set. A test flag `FERRAY_FORCE_SCALAR=1` must disable SIMD dispatch and allow running all fixture tests in scalar mode.

**Numerical parity tolerance** follows the same ULP budget as `ferrolearn`, with one addition: for transcendental functions (`sin`, `exp`, `log`, etc.) the budget is 1 ULP for correctly-rounded implementations (matching NumPy's target) and at most 4 ULPs for approximation-based implementations, with the budget documented per function.

---

## 24. Crate Structure

```
ferray/
├── ferray/                  # Main crate — one import covers everything
├── ferray-core/             # NdArray type, Dimension, DType, FerrayError
├── ferray-ufunc/            # Ufunc trait + all elementwise math
├── ferray-linalg/           # numpy.linalg — wraps faer
├── ferray-fft/              # numpy.fft — wraps rustfft  
├── ferray-random/           # numpy.random — Generator/BitGenerator model
├── ferray-polynomial/       # numpy.polynomial — all polynomial classes
├── ferray-strings/          # numpy.strings — vectorized string ops
├── ferray-stats/            # Statistical functions
├── ferray-io/               # .npy/.npz, loadtxt, savetxt
├── ferray-ma/               # Masked arrays
├── ferray-stride-tricks/    # as_strided, sliding_window_view
└── ferray-numpy/            # PyO3 interop (optional, behind feature flag)
```

Feature flags on the main crate:

| Feature | Default | Description |
|---|---|---|
| `full` | No | Enables all features |
| `blas` | No | Link system BLAS for matmul and linalg |
| `rayon` | Yes | Parallel execution |
| `simd` | Yes | SIMD-accelerated inner loops |
| `complex` | Yes | Complex number support |
| `f16` | No | Half-precision float support |
| `strings` | Yes | `ferray::strings` module |
| `ma` | Yes | `ferray::ma` masked array module |
| `io` | Yes | `.npy`/`.npz` file I/O |
| `serde` | No | Serialize/deserialize arrays via serde |
| `arrow` | No | Apache Arrow interop |
| `polars` | No | Polars DataFrame interop |
| `numpy` | No | PyO3 NumPy interop (requires pyo3) |
| `no_std` | No | Disable std for embedded targets (core numeric ops only) |

---

## 25. MSRV and Stability

**Minimum Supported Rust Version:** 1.78 (stable, May 2024) — required for stabilized `std::simd`.

The public API of `ferray` must reach 1.0 before `ferrolearn` can reach 1.0, since `ferrolearn` depends on `ferray` as a primitive. The API stability contract at 1.0:
- No breaking changes to public types, traits, or function signatures without a major version bump
- **Numerical output stability:** for a fixed compiler version, target architecture, and feature flag set, output does not change between patch versions. A global guarantee of ≤1 ULP across compiler and toolchain updates is not feasible — LLVM floating-point optimization passes and FMA legalization differ between LLVM versions, and locking the LLVM version across patch releases is impractical. The guarantee is scoped to a fixed build environment.
- Changes that alter numerical output under a fixed build environment (e.g., replacing an approximation algorithm) are breaking and require a minor version bump with a changelog entry documenting the maximum observed difference.

---

## 26. Roadmap

### Phase 1 — Core Array and Ufuncs (Months 1–4)
- `NdArray<T, D>` with full ownership model
- All dtypes: `f32`, `f64`, `Complex<f64>`, `i32`, `i64`, `u8`, `bool`
- Full broadcasting (all NumPy broadcast cases)
- Basic and advanced (fancy + boolean) indexing
- All array creation routines
- All array manipulation (reshape, concatenate, stack, transpose, etc.)
- Complete ufunc surface (all math, bitwise, comparison, logical)
- SIMD paths for `f32` and `f64` on x86_64 and aarch64
- `ferray::stats` (sum, mean, var, sort, unique, etc.)
- `.npy` / `.npz` I/O

### Phase 2 — Submodules (Months 5–8)
- `ferray::linalg` — full numpy.linalg parity including `einsum`
- `ferray::fft` — full numpy.fft parity with plan caching
- `ferray::random` — full Generator API with all distributions
- `ferray::polynomial` — all polynomial classes
- Rayon parallelism throughout
- Fixture test suite against NumPy for all P0 functions

### Phase 3 — Completeness (Months 9–12)
- `ferray::strings` — full numpy.strings
- `ferray::ma` — masked arrays
- `ferray::stride_tricks`
- Structured dtype support via derive macro
- Arrow and Polars interop
- `ferray-numpy` PyO3 companion crate
- Fuzz corpus (24 CPU-hours)
- Statistical equivalence benchmark suite

### Phase 4 — Beyond NumPy (Month 13+)
- `f16` support throughout (currently gated behind feature flag due to limited hardware support in stdsimd)
- `no_std` core for embedded targets
- Compile-time dimensionality checking via const generics (static shapes à la nalgebra, opt-in via `Array<T, Shape<3, 4>>` syntax)
- Automatic differentiation support via operator overloading (foundation for an autograd layer)

---

## 27. Prior Art and Key Differences

| Crate / Library | Relationship |
|---|---|
| `ndarray` | Used internally as the storage backend for `ferray-core`. Not part of the public API; insulates users from ndarray version churn |
| `faer` | Powers all `ferray::linalg` operations internally |
| `rustfft` | Powers `ferray::fft` internally |
| `nalgebra` | Complementary — fixed-size, compile-time-checked matrices; `ferray` covers dynamic |
| `ndarray-linalg` | Superseded by `ferray::linalg` for users who want a single dependency |
| `ndarray-rand` | Superseded by `ferray::random` |
| `ndarray-stats` | Superseded by `ferray` statistics methods |
| `num-complex` | `Complex<f32>` / `Complex<f64>` used as element types |
| `candle` | Deep learning framework that could depend on `ferray` as its array primitive |
| `polars` | DataFrame library; `ferray` arrays interop with Polars Series via Arrow |
| `ferrolearn` | Depends on `ferray` as its foundational array layer |

**Key improvements over NumPy itself:**

- Mutability aliasing bugs are impossible — the borrow checker enforces what NumPy's view system only documents (`as_strided` is explicitly `unsafe fn` with documented caller preconditions)
- `predict()` on unfitted models and similar logical errors are compile errors, not runtime exceptions (inherited from `ferrolearn`'s design)
- SIMD dispatch is more explicit and auditable than NumPy's internal C/Fortran dispatch, with runtime CPU feature detection for AVX2/AVX-512
- Uninitialized arrays (`empty`) return `Array<MaybeUninit<T>, D>`, forcing explicit initialization before use rather than silently handing back a typed array with garbage values
- Broadcasting failures are compile errors for statically-known shapes (Phase 4 const generics work)
- Mixed-type operations require explicit promotion rather than silent type coercion
- All parallel code is explicitly opt-in via a ferray-owned Rayon pool rather than implicitly controlled by environment variables (contrast: NumPy's `OMP_NUM_THREADS`)
- Zero-cost zero-copy views are enforced by the type system rather than tracked at runtime

---

