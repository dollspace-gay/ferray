---
title: ferrum-polynomial — Complete numpy.polynomial with all basis classes
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Implements `numpy.polynomial`: power, Chebyshev, Legendre, Laguerre, Hermite, and HermiteE polynomial classes with a common `Poly` trait. Supports evaluation, differentiation, integration, root-finding (via companion matrix eigenvalues), fitting, and arithmetic. Basis conversion uses power basis as a canonical pivot to avoid N^2 conversion implementations.

### Requirements

- REQ-1: `polynomial::Polynomial` — power basis: p(x) = c[0] + c[1]*x + c[2]*x^2 + ...
- REQ-2: `polynomial::Chebyshev` — Chebyshev basis (better conditioned for approximation)
- REQ-3: `polynomial::Legendre` — Legendre basis
- REQ-4: `polynomial::Laguerre` — Laguerre basis
- REQ-5: `polynomial::Hermite` — physicist's Hermite basis
- REQ-6: `polynomial::HermiteE` — probabilist's Hermite basis ### Poly Trait (Section 11.2)
- REQ-7: Common `Poly` trait with methods: `eval(x)`, `deriv(m)`, `integ(m, k)`, `roots()` (returns Array1<Complex<f64>>), `degree()`, `trim(tol)`, `truncate(size)`
- REQ-8: Arithmetic via trait: `add`, `sub`, `mul`, `pow(n)`, `divmod` returning (quotient, remainder)
- REQ-9: Fitting: `Poly::fit(x, y, deg)` and `Poly::fit_weighted(x, y, deg, w)` — least-squares polynomial fitting ### Basis Conversion
- REQ-10: `ToPowerBasis` and `FromPowerBasis` traits. All classes convert through power basis as canonical pivot.
- REQ-11: Provide a `.convert::<TargetType>()` method on the `Poly` trait for basis conversion. Do NOT use a blanket `impl<P: ToPowerBasis, Q: FromPowerBasis> From<P> for Q` — this conflicts with the standard library's blanket `impl<T> From<T> for T` and will not compile due to Rust's coherence rules. Instead, implement pairwise `From` impls for each (source, target) pair, or use the `.convert()` method as the primary API. ### Root Finding
- REQ-12: `roots()` computes roots via companion matrix eigenvalues (delegates to `ferrum-linalg::eigvals`). Returns all roots including complex ones.

### Acceptance Criteria

- [ ] AC-1: `Polynomial::new(&[1.0, -3.0, 2.0]).roots()` returns roots at x=1 and x=2
- [ ] AC-2: `Chebyshev::fit(x, y, 5).eval(x)` approximates y with residual matching NumPy's `Chebyshev.fit` to within 4 ULPs
- [ ] AC-3: Basis conversion round-trips: `Chebyshev -> Polynomial -> Chebyshev` produces original coefficients to within 4 ULPs
- [ ] AC-4: `deriv` and `integ` are inverse operations: `p.integ(1, &[0.0]).deriv(1)` recovers p
- [ ] AC-5: `divmod` satisfies: `a == q * b + r` where `(q, r) = a.divmod(&b)`
- [ ] AC-6: `cargo test -p ferrum-polynomial` passes. `cargo clippy` clean.

### Architecture

### Crate Layout
```
ferrum-polynomial/
  Cargo.toml
  src/
    lib.rs
    traits.rs                 # Poly, ToPowerBasis, FromPowerBasis traits
    power.rs                  # Polynomial (power basis)
    chebyshev.rs              # Chebyshev
    legendre.rs               # Legendre
    laguerre.rs               # Laguerre
    hermite.rs                # Hermite (physicist's)
    hermite_e.rs              # HermiteE (probabilist's)
    fitting.rs                # Least-squares fitting via Vandermonde/weighted normal equations
    companion.rs              # Companion matrix construction for root-finding
```

### Out of Scope

- Rational polynomials (post-1.0)
- Symbolic polynomial manipulation

### dependencies

- **Upstream**: `ferrum-core` (NdArray, Array1, Complex, FerrumError), `ferrum-linalg` (eigenvalue computation for root-finding via companion matrix)
- **Downstream**: ferrum (re-export)
- **External crates**: `num-complex`
- **Phase**: 2 — Submodules

