# Feature: ferray-fft — Complete numpy.fft parity with plan caching

## Summary
Implements the full `numpy.fft` surface: 1D/2D/ND forward and inverse FFTs, real-input FFTs (rfft family), Hermitian FFTs, frequency generation (fftfreq, rfftfreq), and shift utilities. Internally powered by `rustfft`. Adds plan caching via `FftPlan` for repeated transforms of the same size.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Dimension, Element, FerrayError)
- **Downstream**: ferray (re-export)
- **External crates**: `rustfft` (FFT computation), `num-complex` (complex output), `rayon` (parallel multi-dimensional FFTs)
- **Phase**: 2 — Submodules

## Requirements

### Complex FFTs (Section 9)
- REQ-1: `fft::fft(&a, n, axis, norm)` — 1D FFT along specified axis, with `FftNorm::Backward` (default), `Forward`, `Ortho`
- REQ-2: `fft::ifft(&a, n, axis, norm)` — 1D inverse FFT
- REQ-3: `fft::fft2(&a, s, axes)` and `fft::ifft2` — 2D FFT/IFFT
- REQ-4: `fft::fftn(&a, s, axes)` and `fft::ifftn` — N-dimensional FFT/IFFT

### Real FFTs
- REQ-5: `fft::rfft(&a, n, axis)` — real-input FFT returning Hermitian-symmetric output (n/2+1 complex values)
- REQ-6: `fft::irfft(&a, n, axis)` — inverse of rfft, returning real output
- REQ-7: `fft::rfft2`, `fft::irfft2`, `fft::rfftn`, `fft::irfftn` — multi-dimensional real FFTs

### Hermitian FFTs
- REQ-8: `fft::hfft(&a, n, axis)` and `fft::ihfft` — Hermitian-symmetric input FFTs

### Frequency Utilities
- REQ-9: `fft::fftfreq(n, d)` — DFT sample frequencies
- REQ-10: `fft::rfftfreq(n, d)` — sample frequencies for rfft output
- REQ-11: `fft::fftshift(&a, axes)` and `fft::ifftshift` — shift zero-frequency to center

### Plan Caching
- REQ-12: `fft::FftPlan::new(size)` creates a reusable FFT plan. `plan.execute(&signal)` runs the cached plan. Plans are `Send + Sync` for use across threads.
- REQ-13: A global plan cache (thread-safe via `dashmap` or similar) automatically caches plans for repeated sizes. `fft::fft()` uses this cache transparently.

### Normalization
- REQ-14: `FftNorm::Backward` (no normalization on forward, 1/n on inverse), `Forward` (1/n on forward, none on inverse), `Ortho` (1/sqrt(n) both directions) — matching NumPy's `norm` parameter

## Acceptance Criteria
- [ ] AC-1: `fft(ifft(a))` round-trips to within 4 ULPs for complex f64 input
- [ ] AC-2: `fft` output matches NumPy's `np.fft.fft` on fixture data for 1D arrays of lengths 8, 64, 1024, 1023 (non-power-of-2)
- [ ] AC-3: `rfft` of a real signal returns n/2+1 complex values matching NumPy
- [ ] AC-4: `fftfreq(8, 1.0)` returns `[0, 1, 2, 3, -4, -3, -2, -1]` (scaled by 1/8)
- [ ] AC-5: `fftshift` / `ifftshift` round-trip correctly
- [ ] AC-6: `FftPlan` reuse produces identical results to non-cached `fft()` calls
- [ ] AC-7: All three normalization modes produce correct scaling factors
- [ ] AC-8: `cargo test -p ferray-fft` passes. `cargo clippy -p ferray-fft -- -D warnings` clean.

## Architecture

### Crate Layout
```
ferray-fft/
  Cargo.toml
  src/
    lib.rs
    complex.rs                # fft, ifft, fft2, ifft2, fftn, ifftn
    real.rs                   # rfft, irfft, rfft2, irfft2, rfftn, irfftn
    hermitian.rs              # hfft, ihfft
    freq.rs                   # fftfreq, rfftfreq
    shift.rs                  # fftshift, ifftshift
    plan.rs                   # FftPlan type, global plan cache
    norm.rs                   # FftNorm enum and scaling logic
    nd.rs                     # Multi-dimensional FFT via iterated 1D transforms along axes
```

### rustfft Integration
`rustfft::FftPlanner` is used to create plans. The global cache maps `(size, direction)` to `Arc<dyn rustfft::Fft<f64>>`. Multi-dimensional FFTs iterate 1D transforms along each requested axis, processing all lanes in parallel via Rayon.

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Short-time FFT / spectrogram (signal processing, post-1.0)
- GPU-accelerated FFT (Phase 4)
