# Feature: ferray-gpu — GPU-accelerated computing via CUDA, Vulkan, Metal, and WebGPU

## Summary
Adds optional GPU acceleration to ferray's highest-impact operations: matrix multiplication, FFT, elementwise ufuncs, and statistical reductions. Uses a hybrid architecture combining **CubeCL** (write-once cross-platform GPU kernels in Rust) with **cudarc** (direct access to NVIDIA's vendor-optimized cuBLAS, cuFFT, and cuSOLVER libraries). A `GpuArray<T, D>` type mirrors `NdArray<T, D>` with explicit host-device transfers and automatic size-based dispatch.

### Why GPU?
Current CPU benchmarks show ferray is 4-4.6x slower than NumPy on matmul (50x50-100x100) due to the BLAS gap, and transcendentals (sin, cos, exp, log) are 1.4-2.1x slower at scale due to CORE-MATH's correctly-rounded algorithms. GPU acceleration eliminates both gaps: cuBLAS matmul delivers 10-100x over CPU for large matrices, and GPU SIMD parallelism handles millions of elementwise ops in microseconds. For reference, CuPy (NumPy on GPU) typically achieves 10-1000x speedups over NumPy for arrays > 100K elements.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Dimension, Element, FerrayError), `ferray-ufunc` (CPU fallback paths), `ferray-linalg` (CPU fallback), `ferray-fft` (CPU fallback), `ferray-stats` (CPU fallback)
- **Downstream**: `ferray` (re-export under `ferray::gpu`)
- **External crates**:
  - `cubecl` ~0.5 (cross-platform GPU kernel compiler: CUDA, ROCm, Vulkan, Metal, WebGPU)
  - `cubecl-cuda` (CUDA runtime backend for CubeCL)
  - `cubecl-wgpu` (wgpu runtime backend for CubeCL — Vulkan/Metal/DX12/WebGPU)
  - `cudarc` ~0.19 (safe CUDA toolkit wrapper: driver, NVRTC, cuBLAS, cuFFT, cuSOLVER, cuRAND)
  - `wgpu` ~28.0 (cross-platform GPU API — used by CubeCL's wgpu backend)
- **Feature flags**:
  - `gpu-cuda` — NVIDIA CUDA backend via cudarc + CubeCL CUDA. Requires CUDA toolkit 12+ at runtime (dynamic loading, not at build time).
  - `gpu-wgpu` — Cross-platform backend via CubeCL wgpu (Vulkan/Metal/DX12/WebGPU). No vendor SDK required.
  - `gpu` — Enables both `gpu-cuda` and `gpu-wgpu`.
- **Phase**: 6 — GPU Acceleration (post-Phase 5)

## Ecosystem Assessment

### Backend Comparison Matrix

| Backend | f32 | f64 | Complex | Vendor Math Libs | Platforms | Maturity |
|---------|-----|-----|---------|-----------------|-----------|----------|
| **cudarc (CUDA)** | Yes | Yes | Yes | cuBLAS, cuFFT, cuSOLVER, cuRAND | NVIDIA Linux/Windows | Pre-alpha, active (v0.19) |
| **CubeCL CUDA** | Yes | Yes | No (manual) | No (custom kernels) | NVIDIA Linux/Windows | Active (v0.5), used by Burn |
| **CubeCL wgpu** | Yes | No* | No | No | All (Vulkan/Metal/DX12/WebGPU) | Active |
| **vulkano** | Yes | Yes** | No | No | Vulkan platforms | v0.35, pre-1.0 |
| **wgpu (raw)** | Yes | No* | No | No | All | v28.0, mature |
| **metal (objc2-metal)** | Yes | No*** | No | No | macOS/iOS | Active |

\* WebGPU spec has no f64 (gpuweb#2805 open). Vulkan backend can support f64 via `shaderFloat64` on desktop GPUs.
\** Via `shaderFloat64` Vulkan capability — supported on all modern NVIDIA/AMD desktop GPUs.
\*** Metal Shading Language does not support f64.

### Eliminated Options

| Option | Reason for Elimination |
|--------|----------------------|
| **rust-gpu** (Embark Studios) | Archived Oct 2025. Community fork (Rust-GPU org) not production-ready. Requires nightly Rust. |
| **metal-rs** | Deprecated in favor of objc2-metal. Use wgpu's Metal backend instead. |
| **ArrayFire-Rust** | Heavy C++ dependency, FFI overhead, licensing concerns. Against ferray's pure-Rust philosophy. |
| **OpenCL (ocl crate)** | Declining industry relevance. Semi-maintained. wgpu covers cross-platform better. |
| **Raw Vulkan** | Too low-level. vulkano or wgpu provide safe abstractions. |

### Key Design Decisions

1. **CubeCL for custom kernels** — Write ufunc and reduction kernels once in Rust (`#[cube]` macro), compile to CUDA/ROCm/Vulkan/Metal/WebGPU. Handles automatic vectorization, autotuning, and memory management.

2. **cudarc for vendor libraries** — cuBLAS (matmul), cuFFT (FFT), cuSOLVER (SVD/eigen/solve) are hand-tuned by NVIDIA engineers with decade+ of optimization. No point reimplementing these. cudarc wraps them safely with dynamic loading (no build-time CUDA dependency).

3. **f64 strategy** — CUDA: full f64 support. wgpu: f32 only (return `Err(GpuUnsupportedDtype)` for f64). Vulkan native: f64 via `shaderFloat64` when available.

4. **Hybrid dispatch** — For each operation, prefer vendor library (cuBLAS/cuFFT) when available on CUDA, fall back to CubeCL custom kernel, fall back to CPU.

## Requirements

### Core Types and Device Management (GPU-CORE)

- REQ-1: `GpuDevice` trait abstracting over CUDA and wgpu backends:
  ```rust
  pub trait GpuDevice: Send + Sync {
      fn name(&self) -> &str;
      fn backend(&self) -> GpuBackend;
      fn supports_f64(&self) -> bool;
      fn supports_complex(&self) -> bool;
      fn memory_total(&self) -> usize;
      fn memory_free(&self) -> usize;
  }
  ```

- REQ-2: `GpuBackend` enum: `Cuda`, `Vulkan`, `Metal`, `Dx12`, `WebGpu`

- REQ-3: `gpu::devices()` — enumerate available GPU devices. `gpu::best_device()` — auto-select highest-capability device (prefer CUDA > Vulkan > Metal > DX12 > WebGPU).

- REQ-4: `GpuArray<T, D>` — GPU-resident N-dimensional array:
  ```rust
  pub struct GpuArray<T: GpuElement, D: Dimension> {
      buffer: GpuBuffer<T>,     // device memory
      shape: D,                 // shape metadata (CPU-side)
      strides: Vec<usize>,      // strides (CPU-side)
      device: Arc<dyn GpuDevice>,
  }
  ```

- REQ-5: `GpuElement` trait — types that can live on GPU. Implement for: `f32`, `f64` (CUDA only), `i32`, `i64`, `u32`. Complex types via paired f32/f64 buffers.

- REQ-6: Host-device transfers:
  - `array.to_gpu(&device)` -> `GpuArray<T, D>` (copies data to GPU)
  - `gpu_array.to_host()` -> `NdArray<T, D>` (copies data back to CPU)
  - `GpuArray::from_slice(&device, shape, data)` — direct creation on GPU
  - `GpuArray::zeros(&device, shape)`, `ones()`, `full()` — allocate + fill on GPU (no host roundtrip)
  - All transfers are explicit. No implicit copying.

- REQ-7: `GpuArray` shape manipulation that stays on GPU (no data transfer): `reshape()`, `transpose()`, `expand_dims()`, `squeeze()`. These only modify CPU-side shape/stride metadata.

### Elementwise Ufuncs (GPU-UFUNC)

- REQ-8: GPU-accelerated unary ufuncs via CubeCL `#[cube]` kernels:
  - Transcendentals: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `tanh`, `sinh`, `cosh`
  - Inverse trig: `arcsin`, `arccos`, `arctan`
  - Other: `abs`, `neg`, `reciprocal`, `square`, `exp2`, `log2`, `log10`
  - All operate on `GpuArray` and return `GpuArray` (no host roundtrip).

- REQ-9: GPU-accelerated binary ufuncs: `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `atan2`

- REQ-10: Broadcasting on GPU — same NumPy broadcasting rules, implemented in kernel launch parameters (virtual stride expansion), not by materializing the broadcast array.

- REQ-11: Fused kernels — Hand-written CubeCL kernels for common multi-op patterns, avoiding intermediate GPU allocations:
  - `fma(a, x, b)` — `a * x + b`
  - `normalize(x)` — `(x - mean) / std`
  - `softmax(x)` — `exp(x - max) / sum(exp(x - max))`
  - `log_sum_exp(x)` — numerically stable `log(sum(exp(x)))`
  - `squared_diff(x, y)` — `(x - y)^2`
  - `exp_scale(x, a, b)` — `exp(a * x + b)`

### Statistical Reductions (GPU-STATS)

- REQ-12: GPU parallel reductions via CubeCL:
  - `sum`, `prod`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax`
  - Support `axis` parameter for reduction along specific dimensions.
  - Use warp-level reduction primitives on CUDA (shuffle instructions).

- REQ-13: Reduction accuracy: use Kahan compensated summation in GPU reduction kernels for f32 to match CPU pairwise summation accuracy. f64 reductions (CUDA) use standard parallel reduction.

### Linear Algebra (GPU-LINALG)

- REQ-14: `gpu::linalg::matmul(&a, &b)` — matrix multiplication:
  - CUDA: dispatch to cuBLAS `Dgemm`/`Sgemm` (f64/f32)
  - wgpu: CubeCL tiled matmul kernel (f32 only)
  - Batched matmul for 3D+ arrays (cuBLAS batched GEMM)

- REQ-15: `gpu::linalg::solve(&a, &b)` — solve Ax = b via cuSOLVER (CUDA backend)

- REQ-16: `gpu::linalg::svd(&a)`, `gpu::linalg::eig(&a)`, `gpu::linalg::cholesky(&a)` — decompositions via cuSOLVER (CUDA backend). wgpu backend falls back to CPU.

- REQ-17: `gpu::linalg::inv(&a)`, `gpu::linalg::det(&a)` — via cuSOLVER LU factorization

- REQ-18: `gpu::linalg::dot(&a, &b)`, `gpu::linalg::norm(&a)` — via cuBLAS dot/nrm2

### FFT (GPU-FFT)

- REQ-19: `gpu::fft::fft(&a, n)`, `gpu::fft::ifft()` — 1D FFT:
  - CUDA: dispatch to cuFFT `cufftExecZ2Z` (f64 complex) / `cufftExecC2C` (f32 complex)
  - wgpu: CubeCL Cooley-Tukey radix-2 kernel for power-of-2 sizes (f32 only)

- REQ-20: `gpu::fft::fft2()`, `gpu::fft::fftn()` — multi-dimensional FFT via cuFFT plans

- REQ-21: `gpu::fft::rfft()`, `gpu::fft::irfft()` — real-input FFT via cuFFT `R2C`/`C2R`

### Automatic Dispatch (GPU-AUTO)

- REQ-22: Size-based auto-dispatch: operations on `NdArray` can optionally auto-offload to GPU when array size exceeds a configurable threshold. Default thresholds:
  - ufuncs: 100K elements
  - matmul: 128x128
  - FFT: 64K elements
  - reductions: 1M elements
  These are conservative — below these sizes, PCIe transfer overhead exceeds GPU compute benefit.

- REQ-23: `gpu::set_auto_dispatch(bool)` — global toggle. `gpu::set_threshold(op, size)` — per-operation threshold. Disabled by default (opt-in).

- REQ-24: When auto-dispatch is enabled and no GPU is available, silently fall back to CPU. Never panic on missing GPU.

### Memory Management (GPU-MEM)

- REQ-25: GPU memory pool with sub-allocation to avoid per-operation `cudaMalloc` overhead. Reuse freed buffers via a size-bucketed free list (similar to CubeCL's built-in memory management).

- REQ-26: `GpuArray::clone()` — device-to-device copy (no host roundtrip). `GpuArray::to_device(&other_device)` — cross-device copy (via host staging if needed).

- REQ-27: Out-of-memory handling: return `FerrayError::GpuOutOfMemory` with device name and requested/available bytes. Never panic.

- REQ-28: `gpu::memory_stats(&device)` — report allocated, cached, peak usage.

### Pinned Memory and Async Transfers (GPU-ASYNC)

- REQ-31: Pinned (page-locked) host memory for high-throughput transfers:
  - `array.to_gpu_pinned(&device)` — allocates pinned host staging buffer, then DMA to device. Roughly doubles PCIe throughput vs pageable memory (~20 GB/s -> ~25 GB/s on PCIe 4.0).
  - `GpuArray::to_host_pinned()` — D2H via pinned buffer.
  - `PinnedBuffer<T>` — reusable pinned host allocation for pipeline workloads. Amortizes the cost of pinning across repeated transfers.

- REQ-32: Stream-ordered (asynchronous) execution on CUDA backend:
  - All kernel launches and transfers are submitted to a CUDA stream. Operations on the same stream execute in order but do not block the CPU.
  - `GpuArray` created from a kernel does not synchronize until `.to_host()` is called (or explicit `gpu::synchronize(&device)`).
  - Enables overlap: CPU can prepare the next batch while GPU computes the current one.
  - `gpu::Stream::new(&device)` — create additional streams for concurrent pipelines.
  - wgpu backend uses wgpu's queue submission model (similar semantics).

### Error Handling (GPU-ERR)

- REQ-29: New `FerrayError` variants:
  - `GpuNotAvailable` — no GPU device found
  - `GpuOutOfMemory { device, requested, available }`
  - `GpuKernelError { backend, message }` — kernel compilation/launch failure
  - `GpuUnsupportedDtype { dtype, backend }` — e.g., f64 on wgpu
  - `GpuTransferError { direction, message }` — host-device copy failure

- REQ-30: All GPU operations return `Result<T, FerrayError>`. Zero panics.

## Acceptance Criteria

- [ ] AC-1: `GpuArray::<f32, Ix2>::zeros(&cuda_device, shape)` allocates on GPU without host roundtrip
- [ ] AC-2: `array.to_gpu(&device).sin().exp().to_host()` produces results matching CPU `ufunc::exp(ufunc::sin(&array))` within 4 ULP for f32
- [ ] AC-3: `gpu::linalg::matmul` of two (1024, 1024) f64 matrices on CUDA matches CPU `linalg::matmul` within 8 ULPs and runs > 10x faster
- [ ] AC-4: `gpu::fft::fft` of 1M-element f64 array on CUDA matches CPU `fft::fft` within FFT ULP bounds and runs > 5x faster
- [ ] AC-5: `gpu::stats::var` of 1M f32 elements matches CPU `stats::var` within 64 ULP
- [ ] AC-6: Auto-dispatch with `gpu::set_auto_dispatch(true)` transparently offloads large matmuls to GPU
- [ ] AC-7: `GpuArray` operations with f64 on wgpu backend return `Err(GpuUnsupportedDtype)` (not panic)
- [ ] AC-8: Out-of-memory returns `Err(GpuOutOfMemory)` with diagnostic context
- [ ] AC-9: `gpu::devices()` returns empty vec on systems with no GPU. All CPU fallbacks work.
- [ ] AC-10: `cargo test -p ferray-gpu --features gpu-cuda` passes (requires CUDA). `cargo test -p ferray-gpu --features gpu-wgpu` passes (requires any GPU).
- [ ] AC-11: `cargo test -p ferray-gpu` (no features) compiles and all tests are `#[ignore]`-gated behind GPU availability
- [ ] AC-12: Pinned transfer (`to_gpu_pinned`) achieves measurably higher throughput than pageable (`to_gpu`) for 8MB+ arrays
- [ ] AC-13: Stream-ordered execution: `ga.sin().exp().cos()` enqueues 3 kernels without CPU blocking; only `.to_host()` synchronizes
- [ ] AC-14: Fused `normalize` kernel matches `(x - mean) / std` computed via separate ops within 2 ULP for f32

## Architecture

### Crate Layout
```
ferray-gpu/
  Cargo.toml
  src/
    lib.rs                    # Public API, feature gates, re-exports
    device.rs                 # GpuDevice trait, GpuBackend enum, device enumeration
    array.rs                  # GpuArray<T, D> type, host-device transfers
    element.rs                # GpuElement trait, type support matrix
    memory.rs                 # GPU memory pool, sub-allocator, stats
    error.rs                  # GPU-specific FerrayError variants
    auto_dispatch.rs          # Size-based auto-offload logic
    backends/
      mod.rs                  # Backend trait and dispatch
      cuda.rs                 # CUDA backend: cudarc device, streams, memory
      wgpu.rs                 # wgpu backend: device, queue, buffers
    kernels/
      mod.rs                  # Kernel registry and dispatch
      ufunc.rs                # CubeCL elementwise kernels (#[cube] sin, cos, exp, ...)
      reduction.rs            # CubeCL parallel reduction kernels (sum, mean, var, ...)
      fused.rs                # Hand-written fused kernels (fma, normalize, softmax, ...)
      matmul.rs               # CubeCL tiled matmul (wgpu fallback), cuBLAS dispatch (CUDA)
      fft.rs                  # cuFFT dispatch (CUDA), radix-2 CubeCL kernel (wgpu)
    linalg.rs                 # GPU linalg: matmul, solve, svd, eig, cholesky
    fft.rs                    # GPU FFT: fft, ifft, fft2, fftn, rfft
    stats.rs                  # GPU reductions: sum, mean, var, std
    ufunc.rs                  # GPU ufuncs: sin, cos, exp, log, ...
```

### Execution Model

Stream-ordered async execution — CPU does not block until `.to_host()`:

```
User code                  ferray-gpu                    Hardware
---------                  ----------                    --------

let a = array!(...);
let ga = a.to_gpu(&dev);  --> cudaMemcpyH2DAsync ------> GPU memory     } CPU returns
let gb = ga.sin();         --> CubeCL kernel enqueue ---> GPU stream     } immediately
let gc = gb.exp();         --> CubeCL kernel enqueue ---> GPU stream     } (async)
let c = gc.to_host();      --> cudaStreamSynchronize --> cudaMemcpyD2H  } blocks here

// Pipeline overlap: CPU prepares next batch while GPU computes current
let ga2 = a2.to_gpu(&dev);  // <-- this can overlap with gc's GPU execution
```

### Backend Priority and Fallback

```
Operation requested on GpuArray
        |
        +-- CUDA available?
        |   +-- Vendor library exists? (cuBLAS, cuFFT, cuSOLVER)
        |   |   +-- YES -> dispatch to vendor library (optimal)
        |   +-- NO -> dispatch to CubeCL CUDA kernel
        |
        +-- wgpu available?
        |   +-- f64 requested?
        |   |   +-- Vulkan + shaderFloat64 -> CubeCL Vulkan f64 kernel
        |   |   +-- No f64 support -> Err(GpuUnsupportedDtype)
        |   +-- f32 -> CubeCL wgpu kernel
        |
        +-- No GPU -> Err(GpuNotAvailable) or CPU fallback (if auto-dispatch)
```

### Integration with Existing Crates

ferray-gpu does NOT modify existing crates. It provides parallel GPU implementations:

```rust
// CPU path (existing, unchanged):
use ferray::linalg;
let c = linalg::matmul(&a, &b)?;

// GPU path (new, opt-in):
use ferray::gpu;
let dev = gpu::best_device()?;
let ga = a.to_gpu(&dev)?;
let gb = b.to_gpu(&dev)?;
let gc = gpu::linalg::matmul(&ga, &gb)?;
let c = gc.to_host()?;

// Auto-dispatch path (opt-in):
gpu::set_auto_dispatch(true);
let c = linalg::matmul(&a, &b)?;  // transparently uses GPU if large enough
```

### CubeCL Kernel Example (ufunc)

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn sin_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let idx = ABSOLUTE_POS;
    if idx < input.len() {
        output[idx] = F::sin(input[idx]);
    }
}
```

This single kernel compiles to CUDA PTX, SPIR-V (Vulkan), MSL (Metal), and WGSL (WebGPU) via CubeCL's JIT compiler.

### cudarc Vendor Library Example (matmul)

```rust
use cudarc::cublas::{CudaBlas, Gemm};

fn gpu_matmul_f64(
    blas: &CudaBlas,
    a: &CudaSlice<f64>,     // M x K, device memory
    b: &CudaSlice<f64>,     // K x N, device memory
    c: &mut CudaSlice<f64>, // M x N, device memory
    m: usize, n: usize, k: usize,
) -> Result<(), FerrayError> {
    unsafe {
        blas.gemm(
            GemmConfig {
                transa: Op::N, transb: Op::N,
                m, n, k, alpha: 1.0, beta: 0.0,
                lda: m, ldb: k, ldc: m,
            },
            a, b, c,
        ).map_err(|e| FerrayError::gpu_kernel_error("cuda", e))?;
    }
    Ok(())
}
```

### Performance Targets

Based on CuPy benchmarks and NVIDIA documentation:

| Operation | Size | CPU (current) | GPU Target | Expected Speedup |
|-----------|------|--------------|------------|-----------------|
| matmul | 1024x1024 | ~50ms (faer) | ~0.5ms (cuBLAS) | **100x** |
| matmul | 256x256 | ~2ms | ~0.1ms (cuBLAS) | **20x** |
| sin | 1M | 16ms | ~0.1ms (CubeCL) | **160x** |
| exp | 1M | 4.5ms | ~0.08ms (CubeCL) | **56x** |
| sum | 1M | 0.2ms | ~0.05ms (CubeCL) | **4x** |
| fft | 1M | ~5ms | ~0.3ms (cuFFT) | **17x** |
| var | 1M | 0.36ms | ~0.06ms (CubeCL) | **6x** |

Note: These exclude transfer time. For arrays already on GPU (typical in pipelines), these are realistic. Including H2D+D2H transfer (~1ms for 8MB over PCIe 4.0), GPU wins only for arrays > ~50K elements.

### Transfer Cost Model

PCIe 4.0 x16: ~25 GB/s theoretical, ~20 GB/s practical.

| Array Size | Elements (f64) | Transfer Time (H2D) | Breakeven vs CPU |
|-----------|---------------|--------------------|-----------------|
| 8 KB | 1K | 0.4 us | Never (CPU faster) |
| 80 KB | 10K | 4 us | Marginal |
| 800 KB | 100K | 40 us | Ufuncs, matmul |
| 8 MB | 1M | 400 us | All operations |
| 80 MB | 10M | 4 ms | Massive win |

This is why auto-dispatch thresholds default to 100K+ elements.

## Agent Splitting Guidance

This crate should be built in 3 sub-agents:

- **Agent 6a: gpu-core** (opus) — `GpuDevice`, `GpuArray`, `GpuElement`, memory pool, pinned memory, streams, error types, device enumeration, host-device transfers. Depends on: cudarc, cubecl-runtime. REQ-1 through REQ-7, REQ-25 through REQ-32.

- **Agent 6b: gpu-kernels** (opus) — CubeCL kernels for ufuncs, reductions, fused kernels, tiled matmul. cudarc dispatch for cuBLAS/cuFFT/cuSOLVER. REQ-8 through REQ-21. Depends on: Agent 6a.

- **Agent 6c: gpu-integration** (sonnet) — Auto-dispatch, integration with existing crate APIs, benchmarks, test harness with GPU availability gating. REQ-22 through REQ-24. Depends on: Agent 6a, 6b.

## Open Questions

### Q1: CubeCL vs raw kernel authoring
**Status**: RESOLVED — Use CubeCL for custom kernels (ufuncs, reductions, tiled matmul). Use cudarc for vendor libraries (cuBLAS, cuFFT, cuSOLVER). CubeCL's JIT, autotuning, and memory management justify the dependency. Burn's matmul benchmarks show CubeCL matching or exceeding cuBLAS on RTX 4080.

### Q2: f64 on non-CUDA backends
**Status**: RESOLVED — Return `Err(GpuUnsupportedDtype)` for f64 on wgpu/WebGPU. On Vulkan with `shaderFloat64`, allow f64. Document clearly that CUDA is required for full f64 GPU support.

### Q3: Complex number support
**Status**: RESOLVED — Use cuFFT (native complex transforms: CGEMM, ZGEMM, C2C, Z2Z) and cuBLAS (native complex BLAS) via cudarc for the operations where complex types matter for performance (FFT, matmul). Defer elementwise complex ufuncs on GPU (e.g., `sin(Complex<f64>)`) — users can `.to_host()` for those. Ship the 90% case first.

### Q4: Kernel fusion scope
**Status**: RESOLVED — CubeCL's fusion is tightly coupled to Burn's `Fusion` backend and not usable standalone. Instead, provide ~6 hand-written fused kernels for the patterns that actually matter in numerical computing:
  - `fma(a, x, b)` — `a * x + b` (single kernel, most common pattern)
  - `normalize(x)` — `(x - mean) / std`
  - `softmax(x)` — `exp(x - max) / sum(exp(x - max))`
  - `log_sum_exp(x)` — `log(sum(exp(x - max))) + max`
  - `squared_diff(x, y)` — `(x - y)^2` (for variance, MSE)
  - `exp_scale(x, a, b)` — `exp(a * x + b)` (logistic, Boltzmann)
Don't build a general fusion engine — that's a compiler project. These 6 cover the hot paths.

### Q5: Multi-GPU support
**Status**: DEFERRED — Single-GPU is the initial target. Multi-GPU (NCCL via cudarc, data parallelism) is post-1.0.

### Q6: Accuracy policy for GPU f32
**Status**: RESOLVED — Accept GPU hardware accuracy as-is and document the difference. GPU f32 transcendentals (sin, cos, exp) on modern NVIDIA are typically 1-2 ULP — very good, just not correctly-rounded. Running CORE-MATH on GPU would be absurdly slow because its Ziv rounding test uses data-dependent branching, the exact opposite of GPU architecture. Users who need < 0.5 ULP correctly-rounded results are already using f64 on CPU. Document as: "GPU f32 transcendentals use hardware-accelerated approximations. Typical accuracy: 1-2 ULP. For correctly-rounded results (< 0.5 ULP), use the CPU path with CORE-MATH."

## Out of Scope

- Multi-GPU / distributed computing (post-1.0)
- GPU-accelerated I/O (GPUDirect Storage)
- Training-specific ops (autograd tape on GPU, mixed-precision training)
- AMD ROCm backend (CubeCL supports it, but testing requires AMD hardware — community contribution)
- Tensor Core / matrix core dispatch (future optimization pass)
- Custom CUDA kernel hot-reloading (developer tool, not library feature)
- GPU random number generation (cuRAND integration — separate phase)
