# ferray-fft

`numpy.fft` for the [ferray](../README.md) workspace — a Rust-native, drop-in NumPy replacement.

Part of the [ferray](../README.md) workspace — backed by [rustfft](https://docs.rs/rustfft) (with [realfft](https://docs.rs/realfft) for the half-spectrum real transforms).

## Overview

- **1-D complex**: [`fft`], [`ifft`] (plus pre-allocated [`fft_into`] / [`ifft_into`]).
- **2-D complex**: [`fft2`], [`ifft2`].
- **N-D complex**: [`fftn`], [`ifftn`].
- **1-D real**: [`rfft`], [`irfft`] (Hermitian-folded `n/2 + 1` spectrum; [`rfft_into`] / [`irfft_into`]).
- **2-D / N-D real**: [`rfft2`], [`irfft2`], [`rfftn`], [`irfftn`].
- **Hermitian**: [`hfft`], [`ihfft`], plus [`hfft2`], [`ihfft2`], [`hfftn`], [`ihfftn`].
- **Real-input convenience**: [`fft_real`], [`ifft_real`], [`fft_real2`], [`fft_realn`] auto-promote a real array to complex and run the full complex FFT.
- **Frequency helpers**: [`fftfreq`], [`rfftfreq`].
- **Shift helpers**: [`fftshift`], [`ifftshift`].
- **Normalization**: the [`FftNorm`] enum — `Backward` (default, `1/n` on inverse), `Forward` (`1/n` on forward), `Ortho` (`1/sqrt(n)` both directions).
- **Plan caching**: [`FftPlan`] and [`FftPlanND`] reuse rustfft plans across repeated transforms of the same size.

All transforms are generic over `f32` and `f64` (via the `FftFloat` trait), and every public function returns `Result<_, FerrayError>` — no panics.

## NumPy correspondence

| NumPy | ferray-fft |
|-------|------------|
| `np.fft.fft` / `np.fft.ifft` | [`fft`] / [`ifft`] |
| `np.fft.fft2` / `np.fft.ifft2` | [`fft2`] / [`ifft2`] |
| `np.fft.fftn` / `np.fft.ifftn` | [`fftn`] / [`ifftn`] |
| `np.fft.rfft` / `np.fft.irfft` | [`rfft`] / [`irfft`] |
| `np.fft.rfft2` / `np.fft.irfft2` | [`rfft2`] / [`irfft2`] |
| `np.fft.rfftn` / `np.fft.irfftn` | [`rfftn`] / [`irfftn`] |
| `np.fft.hfft` / `np.fft.ihfft` | [`hfft`] / [`ihfft`] |
| `np.fft.fftfreq` / `np.fft.rfftfreq` | [`fftfreq`] / [`rfftfreq`] |
| `np.fft.fftshift` / `np.fft.ifftshift` | [`fftshift`] / [`ifftshift`] |
| `norm=` argument | [`FftNorm`] |

## Feature flags

This crate exposes no Cargo features — the full `numpy.fft` surface is always available. It is also re-exported through the umbrella [`ferray`](../README.md) crate under its `fft` module.

## Example

```rust
use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_fft::{rfft, irfft, fftfreq, FftNorm};

// A real signal.
let signal = Array::from_vec(Ix1::new([8]), vec![1.0_f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])?;

// Forward real FFT -> Hermitian-folded spectrum of length n/2 + 1 = 5.
let spectrum = rfft(&signal, None, None, FftNorm::Backward)?;

// Inverse real FFT recovers the original 8-sample signal.
let recovered = irfft(&spectrum, Some(8), None, FftNorm::Backward)?;

// Sample frequencies for n = 8, sample spacing d = 1.0.
let freqs = fftfreq(8, 1.0)?;
# Ok::<(), ferray_core::error::FerrayError>(())
```

## MSRV & edition

- Rust edition **2024**, MSRV **1.88**.
- Licensed under **MIT OR Apache-2.0**.
