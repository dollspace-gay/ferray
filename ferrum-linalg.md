---
title: ferrum-linalg — Complete numpy.linalg parity powered by faer
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Implements the full `numpy.linalg` surface: matrix products (dot, matmul, einsum, tensordot, kron), decompositions (cholesky, QR, SVD, LU, eigen), solvers (solve, lstsq, inv, pinv), norms and measures (norm, cond, det, trace, matrix_rank). All functions support batched (stacked 3D+) arrays with automatic parallelization along batch dimensions. Internally powered by `faer` with an optional system BLAS backend via feature flag.

### Requirements

- REQ-1: `linalg::dot(&a, &b)` — generalized dot product matching `np.dot` semantics (1D: inner product, 2D: matmul, ND: sum over last axis of a and second-to-last of b)
- REQ-2: `linalg::vdot(&a, &b)` — flattened dot with complex conjugation
- REQ-3: `linalg::inner(&a, &b)`, `linalg::outer(&a, &b)` — inner and outer products
- REQ-4: `linalg::matmul(&a, &b)` — matrix multiplication matching `np.matmul` / `@` semantics including batch broadcasting
- REQ-5: `linalg::tensordot(&a, &b, axes)` — contraction over specified axis pairs
- REQ-6: `linalg::einsum(subscripts, &[arrays])` — full Einstein summation notation parser and executor. This is a hard requirement.
- REQ-7: `linalg::kron(&a, &b)` — Kronecker product
- REQ-7a: `linalg::multi_dot(&[arrays])` — optimized chain matrix multiplication using dynamic programming for optimal parenthesization. 10-100x faster than naive left-to-right chaining for long chains.
- REQ-7b: `linalg::vecdot(&a, &b, axis)` — vector dot product along specified axis (new in NumPy 2.0) ### Decompositions (Section 8.2)
- REQ-8: `linalg::cholesky(&a)` → lower triangular L where A = L L^T
- REQ-9: `linalg::qr(&a, mode)` with `QrMode::Reduced` and `QrMode::Complete` → (Q, R)
- REQ-10: `linalg::svd(&a, full_matrices)` → (U, S, Vt)
- REQ-11: `linalg::eig(&a)` → (eigenvalues, eigenvectors) for general square matrices
- REQ-12: `linalg::eigh(&a)` → symmetric/Hermitian eigendecomposition
- REQ-13: `linalg::eigvals(&a)`, `linalg::eigvalsh(&a)` — eigenvalues only
- REQ-14: `linalg::lu(&a)` → (P, L, U) with partial pivoting ### Solving and Inversion (Section 8.3)
- REQ-15: `linalg::solve(&a, &b)` — solve Ax = b for x
- REQ-16: `linalg::lstsq(&a, &b, rcond)` — least-squares solution
- REQ-17: `linalg::inv(&a)` — matrix inverse
- REQ-18: `linalg::pinv(&a, rcond)` — Moore-Penrose pseudoinverse
- REQ-18a: `linalg::matrix_power(&a, n)` — raise square matrix to integer power n (supports negative n via inversion). Used in Markov chains, graph theory.
- REQ-18b: `linalg::tensorsolve(&a, &b, axes)` — solve tensor equation a x = b for x
- REQ-18c: `linalg::tensorinv(&a, ind)` — compute inverse of N-dimensional array ### Norms and Measures (Section 8.4)
- REQ-19: `linalg::norm(&a, ord)` with `NormOrder` enum: `Fro`, `Nuc`, `Inf`, `NegInf`, `L1`, `L2`, `P(f64)`
- REQ-20: `linalg::cond(&a, p)` — condition number
- REQ-21: `linalg::det(&a)` and `linalg::slogdet(&a)` → (sign, logdet)
- REQ-22: `linalg::matrix_rank(&a, tol)` and `linalg::trace(&a)` ### Batched Operations (Section 8.5)
- REQ-23: ALL linalg functions that accept 2D arrays must also accept 3D+ stacked arrays. The operation applies along the last two axes, parallelized via Rayon.
- REQ-24: Example: `linalg::det(&batch_3d)` where batch_3d is (100, 4, 4) returns Array1 of 100 determinants ### Einsum Parser
- REQ-25: The einsum subscript parser must support: implicit output mode (`"ij,jk"`), explicit output mode (`"ij,jk->ik"`), trace (`"ii->i"`), batch dimensions (`"bij,bjk->bik"`), and broadcasting (`"...ij,...jk->...ik"`)
- REQ-26: Einsum optimization: for 2-operand cases, detect and dispatch to matmul/tensordot when possible rather than using the generic contraction loop

### Acceptance Criteria

- [ ] AC-1: `matmul` of two (100, 100) f64 matrices matches NumPy output to within 4 ULPs
- [ ] AC-2: All decompositions (cholesky, QR, SVD, LU, eig, eigh) produce results that reconstruct the original matrix within tolerance on fixture data
- [ ] AC-3: `einsum("ij,jk->ik", &[&a, &b])` produces the same result as `matmul(&a, &b)` for 2D inputs
- [ ] AC-4: `einsum("ii", &[&a])` computes the trace correctly
- [ ] AC-5: Batched `det` on (100, 4, 4) array returns 100 determinants matching per-matrix `det` calls
- [ ] AC-6: `solve(&a, &b)` where A is well-conditioned returns x such that `||Ax - b|| < 1e-10`
- [ ] AC-7: `lstsq` on an overdetermined system matches NumPy's residuals to within 4 ULPs
- [ ] AC-8: `SingularMatrix` error is returned for `inv()` and `det()` on singular input, not a panic
- [ ] AC-9: `cargo test -p ferrum-linalg` passes. `cargo clippy -p ferrum-linalg -- -D warnings` clean.
- [ ] AC-10: `multi_dot(&[&a, &b, &c])` produces same result as `matmul(&matmul(&a, &b), &c)` but selects optimal parenthesization
- [ ] AC-11: `matrix_power(&a, 3)` == `matmul(&matmul(&a, &a), &a)`. `matrix_power(&a, -1)` == `inv(&a)`.

### Architecture

### Crate Layout
```
ferrum-linalg/
  Cargo.toml
  src/
    lib.rs
    products/
      mod.rs                  # dot, vdot, inner, outer, matmul, kron
      tensordot.rs            # tensordot
      einsum/
        mod.rs                # Public einsum() function
        parser.rs             # Subscript string parser
        optimizer.rs          # Detect matmul/tensordot shortcuts
        contraction.rs        # Generic contraction loop
    decomp/
      mod.rs
      cholesky.rs             # Wraps faer::Cholesky
      qr.rs                   # Wraps faer::Qr
      svd.rs                  # Wraps faer::Svd
      lu.rs                   # Wraps faer::PartialPivLu
      eigen.rs                # eig, eigh, eigvals, eigvalsh via faer
    solve.rs                  # solve, lstsq, inv, pinv
    norms.rs                  # norm, cond, det, slogdet, matrix_rank, trace
    batch.rs                  # Batched dispatch: iterate over leading dims, parallelize
    faer_bridge.rs            # Conversion between ferrum NdArray and faer::Mat
```

### faer Integration
All heavy computation delegates to `faer`. The `faer_bridge` module converts between `NdArray<f64, Ix2>` and `faer::Mat<f64>` with zero-copy where memory layouts match (both C-contiguous). For non-contiguous arrays, a copy is made into a contiguous buffer before calling faer.

### Out of Scope

- Sparse matrix linear algebra (future ferrum-sparse crate)
- GPU-accelerated linear algebra (Phase 4)
- Iterative solvers (CG, GMRES — post-1.0)

### dependencies

- **Upstream**: `ferrum-core` (NdArray, Dimension, Element, FerrumError), `ferrum-ufunc` (arithmetic for einsum)
- **Downstream**: `ferrum-polynomial` (uses linalg for companion matrix eigenvalues / root finding), ferrum (re-export)
- **External crates**: `faer` 0.24 (all BLAS-level operations), `rayon` (batch parallelism)
- **Feature flags**: `blas` — link system BLAS/LAPACK for matmul and decompositions instead of faer's pure-Rust implementations
- **Phase**: 2 — Submodules

