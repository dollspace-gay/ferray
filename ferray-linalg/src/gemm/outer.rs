// Outer blocked GEMM driver — Goto-style 5-loop algorithm using the
// AVX2 4x8 micro-kernel translated from OpenBLAS dgemm_kernel_4x8_haswell.S.
//
// Loop nesting (outer to inner) for the SEQUENTIAL inner driver
// (`gemm_seq_avx2_f64`):
//   jc loop  : split N into NC-wide column panels
//   pc loop  : split K into KC-deep depth panels (pack B once per (jc, pc))
//   ic loop  : split M into MC-tall row panels (pack A once per (jc, pc, ic))
//   jr loop  : split NC into NR-wide column tiles within the panel
//   ir loop  : split MC into MR-tall row tiles within the panel
//     -> call kernel_8x6 on the (MR, NR) tile
//
// Parallelism (#657, OpenBLAS-style): the M dimension is split across
// rayon workers UPFRONT, before the loop nest. Each worker runs the
// sequential 5-loop on its own M-slab with its own thread-local A-pack
// and B-pack buffers. This eliminates the per-pc-iter join/barrier that
// the previous IC-parallel-inside-PC structure imposed (#655) and lets
// each thread keep both A and B packs in its own L2.
//
// Block sizes (matching OpenBLAS Haswell param.h sizes adapted for the
// 14700K's smaller per-P-core L2):
//   MC = 64   -> 64 * KC * 8B = 128 KB A panel
//   KC = 256  -> matches DGEMM_DEFAULT_Q from OpenBLAS Haswell param.h
//   NC = 512  -> 512 * KC * 8B = 1 MB B panel (fits in L2 per thread)
// 14700K Raptor Cove L2 is 1.25-2 MB per P-core. With per-thread B-pack
// of 1 MB plus per-thread A-pack of 128 KB, the working set sits inside
// L2. The previous NC=1024 (2 MB B-pack) spilled to L3 under multi-thread
// contention.

use super::kernel_avx2::{MR, NR, kernel_8x6, kernel_edge};
use super::kernel_avx2_complex_f32::{MR_C32, NR_C32, kernel_4x4_c32, kernel_edge_c32};
use super::kernel_avx2_complex_f64::{MR_C64, NR_C64, kernel_4x2_c64, kernel_edge_c64};
use super::kernel_avx2_f32::{MR_F32, NR_F32, kernel_4x16, kernel_edge_f32};
use super::kernel_avx2_i8::{MR_I8, NR_I8, kernel_4x8_i8, kernel_edge_i8};
use super::kernel_avx2_i8s::{MR_I8S, NR_I8S, kernel_4x8_i8_signed, kernel_edge_i8_signed};
use super::kernel_avx2_i16::{MR_I16, NR_I16, kernel_4x8_i16, kernel_edge_i16};
#[cfg(feature = "avx512")]
use super::kernel_avx512::{MR_AVX512, NR_AVX512, kernel_4x16_avx512, kernel_edge_avx512};
#[cfg(feature = "avx512")]
use super::kernel_avx512_complex_f32::{
    MR_AVX512_C32, NR_AVX512_C32, kernel_4x8_c32_avx512, kernel_edge_c32_avx512,
};
#[cfg(feature = "avx512")]
use super::kernel_avx512_complex_f64::{
    MR_AVX512_C64, NR_AVX512_C64, kernel_4x4_c64_avx512, kernel_edge_c64_avx512,
};
#[cfg(feature = "avx512")]
use super::kernel_avx512_f32::{
    MR_AVX512_F32, NR_AVX512_F32, kernel_4x32_avx512, kernel_edge_avx512_f32,
};
#[cfg(feature = "avx512")]
use super::kernel_avx512_vnni_i8::{
    MR_AVX512_I8, NR_AVX512_I8, kernel_4x16_i8_avx512, kernel_edge_i8_avx512,
};
#[cfg(feature = "avxvnni")]
use super::kernel_avxvnni_i8::kernel_4x8_i8_vnni;
use super::pack::{
    pack_a, pack_a_c32, pack_a_c64, pack_a_f32, pack_a_i8s, pack_a_i16, pack_a_u8, pack_b,
    pack_b_c32, pack_b_c64, pack_b_f32, pack_b_i8, pack_b_i8s, pack_b_i16, pack_kc_i8, pack_kc_i8s,
    pack_kc_i16,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::OnceLock;

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 512;

// Thread-local pack buffers. Each rayon worker (and the calling thread,
// for the sequential path) reuses the same Vec<f64> across calls,
// eliminating the per-IC-chunk `vec![0.0; ...]` allocation that
// previously fired on every panel pack.
//
// `A_PACK_TLS` holds an MR-strip pack of dimensions (MC, KC).
// `B_PACK_TLS` holds an NR-strip pack of dimensions (KC, NC).
//
// Sized to MAX(MC*KC) + a small overhead for tail padding.
thread_local! {
    static A_PACK_TLS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static A_PACK_TLS_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Run a closure with a thread-local A-pack scratch buffer of at least
/// `min_capacity` doubles. Tail padded with zeros so `pack_a`'s
/// partial-strip case sees zeroed gap rows.
fn with_a_pack<F: FnOnce(&mut [f64])>(min_capacity: usize, f: F) {
    A_PACK_TLS.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// Run a closure with a thread-local B-pack scratch buffer of at least
/// `min_capacity` doubles. Tail padded with zeros so `pack_b`'s
/// partial-strip case sees zeroed gap columns.
fn with_b_pack<F: FnOnce(&mut [f64])>(min_capacity: usize, f: F) {
    B_PACK_TLS.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// f32 sibling of `with_a_pack`. Separate TLS storage so f32 and f64
/// matmuls can co-exist on the same worker without churning a shared
/// buffer's capacity.
fn with_a_pack_f32<F: FnOnce(&mut [f32])>(min_capacity: usize, f: F) {
    A_PACK_TLS_F32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// f32 sibling of `with_b_pack`.
fn with_b_pack_f32<F: FnOnce(&mut [f32])>(min_capacity: usize, f: F) {
    B_PACK_TLS_F32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// Dedicated thread pool for the GEMM parallel branch. Capped at 8 to
/// avoid the heterogeneous-core penalty: on Intel hybrid (12th-14th gen)
/// the default `available_parallelism` count includes E-cores and
/// hyperthreading siblings, both of which regress DGEMM throughput
/// (measured #655: 1t=5040us, 4t=2096us, **8t=1768us**, 16t=2503us,
/// 28t=2412us at N=512).
fn matmul_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n = logical.min(8);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("ferray-gemm-{i}"))
            .build()
            .expect("ferray-gemm thread pool")
    })
}

/// Below this max-dim, run sequentially. Above it, parallelise via
/// M-slabs (each worker handles M/n_workers rows independently). Set to
/// 256 because each worker's per-thread pack_b is O(KC * NC) bytes ~=
/// 1 MB; that pack cost only amortises over kernel work once N >= 256
/// (kernel ops scale as M*N*K, pack as K*N).
const PARALLEL_MIN_DIM: usize = 256;

// Send-capable raw pointer wrappers. The outer driver verifies that each
// rayon worker accesses non-overlapping memory, so unsynchronised cross-
// thread access is sound. Field accessors are methods (not direct `.0`)
// so Rust 2021's disjoint-capture doesn't reach inside and try to capture
// the bare `*const T` (which is !Sync).
#[derive(Copy, Clone)]
struct SendConstPtr<T>(*const T);
#[derive(Copy, Clone)]
struct SendMutPtr<T>(*mut T);
impl<T> SendConstPtr<T> {
    #[inline]
    fn get(self) -> *const T {
        self.0
    }
}
impl<T> SendMutPtr<T> {
    #[inline]
    fn get(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Compute C = alpha * A @ B + beta * C for row-major matrices.
///
/// All inputs are row-major: A is (m, k), B is (k, n), C is (m, n).
/// Strides are in element counts (doubles), not bytes.
///
/// SAFETY: AVX2+FMA must be available; caller verifies via the dispatcher.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_avx2_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // C = beta * C
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let p = c.add(i * ldc + j);
                    *p *= beta;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR * 2;

    if parallel {
        // 2D grid parallelism (#659): split work into a `mw x nw` grid of
        // tiles, where mw * nw == workers and the factorisation balances
        // log(M / mw) ≈ log(N / nw). Each worker handles one (M, N) cell
        // of C, running the sequential 5-loop Goto on its sub-rectangle
        // with its own thread-local A-pack and B-pack buffers.
        //
        // The win over 1D M-slab (#657, the prior design): each worker's
        // pack_b is K * (N / nw) instead of K * N — at N=192 with grid
        // (4, 2) that's 50% less B-pack per worker. The cost: redundant
        // A-pack across N-cells in the same M-row (each worker packs its
        // own M-slab of A independently). Net pack-traffic reduction is
        // ~35% for square matrices (#659 measurements). Largest gain at
        // small/mid N where pack overhead dominates kernel work.
        //
        // For tall-skinny (M >> N) the picker selects (workers, 1),
        // recovering the 1D M-slab design exactly. For short-fat (N >> M)
        // it selects (1, workers), splitting on N alone.
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid(m, n, k, workers);

        // Round per-worker dimensions up to MR / NR multiples so the
        // bulk of kernel calls hit the full register tile.
        let m_per = (m.div_ceil(mw)).next_multiple_of(MR).max(MR);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR).max(NR);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| {
                // SAFETY: AVX2+FMA verified by caller. Each worker owns
                // a non-overlapping (M, N) sub-rectangle of C, and reads
                // a non-overlapping M-slab of A and N-slab of B (both
                // read-only across threads).
                unsafe {
                    let a_slab = a_send.get().add(chunk.m_start * lda);
                    let b_slab = b_send.get().add(chunk.n_start);
                    let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                    gemm_seq_avx2_f64(
                        chunk.m_size,
                        chunk.n_size,
                        k,
                        alpha,
                        a_slab,
                        lda,
                        b_slab,
                        ldb,
                        beta,
                        c_slab,
                        ldc,
                    );
                }
            });
        });
    } else {
        // Sequential path — fully serial 5-loop Goto.
        gemm_seq_avx2_f64(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/// One worker's 2D grid cell: a sub-rectangle of C at
/// `[m_start..m_start+m_size, n_start..n_start+n_size]`.
#[derive(Clone, Copy)]
struct GridChunk {
    m_start: usize,
    m_size: usize,
    n_start: usize,
    n_size: usize,
}

/// Choose a `(m_workers, n_workers)` factorisation of `workers`.
///
/// Default: 1D M-slab `(workers, 1)` — for square and tall-skinny shapes
/// it gives every worker its own non-overlapping range of A and C with
/// no redundant pack_a across the grid. Empirically (#659) this wins
/// over 2D on square matrices because 2D adds A-read traffic (workers
/// in the same M-row group both read the same A panel from L3/DRAM).
///
/// Switch to 2D `(mw, nw)` when EITHER:
///   - M is too small for 1D (`m / MR < workers`): some workers would
///     idle or take a sub-MR partial slab; 2D splits along N instead.
///   - 1D's per-worker B-pack would overflow per-P-core L2 (~1.25 MB):
///     each worker's B-pack is `min(KC, k) * n * 8 B`. If that exceeds
///     the L2 budget, splitting N reduces it proportionally.
///
/// Among 2D candidates, picks the factorisation minimising
/// `|log(per_m) - log(per_n)|`. Falls back to `(workers, 1)` for
/// degenerate inputs.
fn pick_grid(m: usize, n: usize, k: usize, workers: usize) -> (usize, usize) {
    if workers <= 1 || m == 0 || n == 0 {
        return (workers.max(1), 1);
    }

    let kc_eff = KC.min(k.max(1));
    // Conservative L2 budget per worker: leave room for A-pack + kernel
    // working set alongside B-pack. 14700K Raptor Cove L2 is 1.25-2 MB
    // per P-core; cap B-pack at ~1 MB (131k doubles).
    const L2_B_BUDGET_DOUBLES: usize = 128 * 1024;
    let per_worker_b_pack_1d = kc_eff * n;

    let mr_chunks_in_m = m / MR;

    // Use 1D M-slab when M is big enough AND 1D's per-worker B-pack fits.
    if mr_chunks_in_m >= workers && per_worker_b_pack_1d <= L2_B_BUDGET_DOUBLES {
        return (workers, 1);
    }

    // 2D fallback: pick factorisation minimising |log(per_m) - log(per_n)|.
    // Constrain mw <= max(mr_chunks_in_m, 1) so every M-row group has at
    // least one MR-tile of work.
    let mw_max = mr_chunks_in_m.max(1);
    let mut best = (1usize, workers);
    let mut best_score = f64::INFINITY;
    let mut mw = 1;
    while mw <= mw_max && mw <= workers {
        if workers % mw == 0 {
            let nw = workers / mw;
            let per_m = (m as f64) / (mw as f64);
            let per_n = (n as f64) / (nw as f64);
            let score = (per_m.ln() - per_n.ln()).abs();
            if score < best_score || (score == best_score && mw > best.0) {
                best = (mw, nw);
                best_score = score;
            }
        }
        mw += 1;
    }
    best
}

/// Sequential 5-loop blocked DGEMM. Used directly for small problems and
/// invoked per-thread from the M-slab parallel path. Uses the thread-local
/// A_PACK_TLS and B_PACK_TLS buffers so repeated calls (across pc/ic
/// iterations and across matmuls) reuse the same allocation.
///
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_seq_avx2_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR);
        let b_pack_size = nr_strips * NR * KC.min(k);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR * kc;
            let eff_beta = if first_pc { beta } else { 1.0 };

            // Pack B[pc..pc+kc, jc..jc+nc] into the thread-local B-pack.
            with_b_pack(b_pack_capacity.max(b_pack_size), |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b(kc, nc, b_panel_ptr, ldb, b_pack);

                // IC loop, sequential within this thread's slab.
                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR);
                    let a_pack_size = n_strips_m * MR * kc;

                    with_a_pack(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR).min(NR);
                            let b_strip = b_pack.as_ptr().add(jr * NR * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR).min(MR);
                                let a_strip = a_pack.as_ptr().add(ir * MR * kc);

                                let c_tile = c.add((ic + ir * MR) * ldc + jc + jr * NR);

                                if m_tile == MR && n_tile == NR {
                                    kernel_8x6(
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// SGEMM (f32) drivers — same M-slab + 2D grid structure as the f64 path,
// just with MR_F32=4, NR_F32=16, and f32 kernel/pack routines.
// ---------------------------------------------------------------------

/// Compute C = alpha * A @ B + beta * C for row-major f32 matrices.
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_avx2_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let p = c.add(i * ldc + j);
                    *p *= beta;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_F32 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid_f32(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_F32).max(MR_F32);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_F32).max(NR_F32);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx2_f32(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx2_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/// Sequential SGEMM 5-loop. Mirror of `gemm_seq_avx2_f64` with f32 types
/// and MR_F32 / NR_F32 register tile dimensions.
///
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_seq_avx2_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_F32);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR_F32 * kc;
            let eff_beta = if first_pc { beta } else { 1.0 };

            with_b_pack_f32(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_f32(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_F32);
                    let a_pack_size = n_strips_m * MR_F32 * kc;

                    with_a_pack_f32(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_f32(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_F32).min(NR_F32);
                            let b_strip = b_pack.as_ptr().add(jr * NR_F32 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_F32).min(MR_F32);
                                let a_strip = a_pack.as_ptr().add(ir * MR_F32 * kc);

                                let c_tile = c.add((ic + ir * MR_F32) * ldc + jc + jr * NR_F32);

                                if m_tile == MR_F32 && n_tile == NR_F32 {
                                    kernel_4x16(
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_f32(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

/// f32 sibling of `pick_grid`. Same heuristic, but uses MR_F32 for the
/// row-alignment budget and accounts for f32 element size in the L2
/// pack-fit threshold (1 MB / 4 B = 262k floats vs 128k doubles).
fn pick_grid_f32(m: usize, n: usize, k: usize, workers: usize) -> (usize, usize) {
    if workers <= 1 || m == 0 || n == 0 {
        return (workers.max(1), 1);
    }

    let kc_eff = KC.min(k.max(1));
    // L2 budget: ~1 MB per worker for B-pack. 1 MB = 262144 f32s.
    const L2_B_BUDGET_F32S: usize = 256 * 1024;
    let per_worker_b_pack_1d = kc_eff * n;

    let mr_chunks_in_m = m / MR_F32;

    if mr_chunks_in_m >= workers && per_worker_b_pack_1d <= L2_B_BUDGET_F32S {
        return (workers, 1);
    }

    let mw_max = mr_chunks_in_m.max(1);
    let mut best = (1usize, workers);
    let mut best_score = f64::INFINITY;
    let mut mw = 1;
    while mw <= mw_max && mw <= workers {
        if workers % mw == 0 {
            let nw = workers / mw;
            let per_m = (m as f64) / (mw as f64);
            let per_n = (n as f64) / (nw as f64);
            let score = (per_m.ln() - per_n.ln()).abs();
            if score < best_score || (score == best_score && mw > best.0) {
                best = (mw, nw);
                best_score = score;
            }
        }
        mw += 1;
    }
    best
}

// ---------------------------------------------------------------------
// AVX-512 DGEMM drivers — same M-slab / 2D-grid structure as the AVX2
// f64 path, just with the AVX-512 4x16 kernel and a NR_AVX512=16 B-pack.
// Cfg-gated on the `avx512` feature; module is absent when feature is
// off, keeping MSRV at 1.85 for default builds.
// ---------------------------------------------------------------------

// f64 AVX-512 thread-local B-pack (sibling of `B_PACK_TLS`). Separate
// because NR=16 means the buffer width differs from the AVX2 path's
// NR=8 — packing into the same buffer would churn capacity if a
// process ran both kernels back-to-back.
#[cfg(feature = "avx512")]
thread_local! {
    static B_PACK_TLS_AVX512: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "avx512")]
fn with_b_pack_avx512<F: FnOnce(&mut [f64])>(min_capacity: usize, f: F) {
    B_PACK_TLS_AVX512.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// AVX-512 DGEMM entry — parallel M-slab dispatch + 2D grid fallback.
/// SAFETY: AVX-512F must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_avx512_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let p = c.add(i * ldc + j);
                    *p *= beta;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_AVX512 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        // Reuse the f64 grid picker — MR_AVX512 == MR (both 4), so the
        // m/MR ratio is the same.
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_AVX512).max(MR_AVX512);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_AVX512).max(NR_AVX512);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx512_f64(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx512_f64(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/// Sequential AVX-512 5-loop DGEMM. Mirror of `gemm_seq_avx2_f64` but
/// using the 4x16 AVX-512 kernel. MR_AVX512 == MR == 4 so we share the
/// AVX2 f64 `pack_a`; `pack_b_f64_nr16` provides the wider B-pack.
///
/// SAFETY: AVX-512F must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_seq_avx512_f64(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    use super::pack::pack_b_f64_nr16;
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_AVX512);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR_AVX512 * kc;
            let eff_beta = if first_pc { beta } else { 1.0 };

            with_b_pack_avx512(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_f64_nr16(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    // MR_AVX512 == MR, so reuse the AVX2 A-pack and
                    // its TLS buffer.
                    let n_strips_m = mc.div_ceil(MR_AVX512);
                    let a_pack_size = n_strips_m * MR_AVX512 * kc;

                    with_a_pack(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_AVX512).min(NR_AVX512);
                            let b_strip = b_pack.as_ptr().add(jr * NR_AVX512 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_AVX512).min(MR_AVX512);
                                let a_strip = a_pack.as_ptr().add(ir * MR_AVX512 * kc);

                                let c_tile =
                                    c.add((ic + ir * MR_AVX512) * ldc + jc + jr * NR_AVX512);

                                if m_tile == MR_AVX512 && n_tile == NR_AVX512 {
                                    kernel_4x16_avx512(
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_avx512(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// AVX-512 SGEMM drivers — same M-slab / 2D-grid structure as the AVX2
// f32 path, just with the AVX-512 4x32 kernel and an NR_AVX512_F32=32
// B-pack. Cfg-gated on `avx512`.
// ---------------------------------------------------------------------

// f32 AVX-512 thread-local B-pack. Separate from B_PACK_TLS_F32 because
// NR=32 means double the buffer width vs the AVX2 SGEMM path's NR=16.
#[cfg(feature = "avx512")]
thread_local! {
    static B_PACK_TLS_AVX512_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "avx512")]
fn with_b_pack_avx512_f32<F: FnOnce(&mut [f32])>(min_capacity: usize, f: F) {
    B_PACK_TLS_AVX512_F32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// AVX-512 SGEMM entry — parallel M-slab dispatch + 2D grid fallback.
/// SAFETY: AVX-512F must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_avx512_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let p = c.add(i * ldc + j);
                    *p *= beta;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_AVX512_F32 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        // Reuse the f32 grid picker — MR_AVX512_F32 == MR_F32 == 4.
        let (mw, nw) = pick_grid_f32(m, n, k, workers);

        let m_per = (m.div_ceil(mw))
            .next_multiple_of(MR_AVX512_F32)
            .max(MR_AVX512_F32);
        let n_per = (n.div_ceil(nw))
            .next_multiple_of(NR_AVX512_F32)
            .max(NR_AVX512_F32);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx512_f32(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx512_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/// Sequential AVX-512 5-loop SGEMM. Mirror of `gemm_seq_avx2_f32` but
/// using the 4x32 AVX-512 kernel. MR_AVX512_F32 == MR_F32 == 4, so we
/// reuse the AVX2 f32 `pack_a_f32`; `pack_b_f32_nr32` provides the
/// wider B-pack.
///
/// SAFETY: AVX-512F must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_seq_avx512_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    use super::pack::pack_b_f32_nr32;
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_AVX512_F32);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR_AVX512_F32 * kc;
            let eff_beta = if first_pc { beta } else { 1.0 };

            with_b_pack_avx512_f32(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_f32_nr32(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_AVX512_F32);
                    let a_pack_size = n_strips_m * MR_AVX512_F32 * kc;

                    with_a_pack_f32(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_f32(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_AVX512_F32).min(NR_AVX512_F32);
                            let b_strip = b_pack.as_ptr().add(jr * NR_AVX512_F32 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_AVX512_F32).min(MR_AVX512_F32);
                                let a_strip = a_pack.as_ptr().add(ir * MR_AVX512_F32 * kc);

                                let c_tile = c
                                    .add((ic + ir * MR_AVX512_F32) * ldc + jc + jr * NR_AVX512_F32);

                                if m_tile == MR_AVX512_F32 && n_tile == NR_AVX512_F32 {
                                    kernel_4x32_avx512(
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_avx512_f32(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha,
                                        a_strip,
                                        b_strip,
                                        eff_beta,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// ZGEMM (Complex<f64>) drivers — same M-slab + 2D-grid structure as the
// real f64 path, but the kernel processes 1 ymm per row (NR_C64=2 complex
// fits in 4 doubles) and the pack/data layout doubles each element to
// 2 contiguous doubles for [re, im].
// ---------------------------------------------------------------------

// Thread-local complex pack buffers. Same idea as f64 — keep per-worker
// scratch allocated across calls.
thread_local! {
    static A_PACK_TLS_C64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_C64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

fn with_a_pack_c64<F: FnOnce(&mut [f64])>(min_capacity: usize, f: F) {
    A_PACK_TLS_C64.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

fn with_b_pack_c64<F: FnOnce(&mut [f64])>(min_capacity: usize, f: F) {
    B_PACK_TLS_C64.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// ZGEMM entry — parallel M-slab dispatch + 2D-grid fallback.
///
/// All inputs are row-major Complex<f64> as `*const f64` / `*mut f64`
/// where each complex element is two contiguous f64s `[re, im]`.
/// `lda`, `ldb`, `ldc` are row strides **in complex elements**.
/// `alpha` and `beta` are passed as separate (re, im) doubles.
///
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_avx2_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // C = beta * C — scalar fallback (rare).
        for i in 0..m {
            for j in 0..n {
                let p_re = c.add(i * ldc * 2 + j * 2);
                let p_im = p_re.add(1);
                if beta_re == 0.0 && beta_im == 0.0 {
                    *p_re = 0.0;
                    *p_im = 0.0;
                } else if !(beta_re == 1.0 && beta_im == 0.0) {
                    let cr = *p_re;
                    let ci = *p_im;
                    *p_re = beta_re * cr - beta_im * ci;
                    *p_im = beta_re * ci + beta_im * cr;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_C64 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        // Reuse the f64 grid picker — MR_C64 == 4 == MR.
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_C64).max(MR_C64);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_C64).max(NR_C64);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                // Each complex element is 2 doubles, so multiply offsets by 2.
                let a_slab = a_send.get().add(chunk.m_start * lda * 2);
                let b_slab = b_send.get().add(chunk.n_start * 2);
                let c_slab = c_send
                    .get()
                    .add(chunk.m_start * ldc * 2 + chunk.n_start * 2);
                gemm_seq_avx2_c64(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha_re,
                    alpha_im,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta_re,
                    beta_im,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx2_c64(
            m, n, k, alpha_re, alpha_im, a, lda, b, ldb, beta_re, beta_im, c, ldc,
        );
    }
}

/// Sequential 5-loop ZGEMM. Mirror of `gemm_seq_avx2_f64`.
///
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_seq_avx2_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_C64);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            // Each complex pack element is 2 doubles.
            let b_pack_capacity = nr_strips * NR_C64 * 2 * kc;
            // For pc>0 we accumulate, so eff_alpha = alpha but eff_beta = 1.
            let (eff_beta_re, eff_beta_im) = if first_pc {
                (beta_re, beta_im)
            } else {
                (1.0, 0.0)
            };

            with_b_pack_c64(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb * 2 + jc * 2);
                pack_b_c64(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_C64);
                    let a_pack_size = n_strips_m * MR_C64 * 2 * kc;

                    with_a_pack_c64(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda * 2 + pc * 2);
                        pack_a_c64(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_C64).min(NR_C64);
                            let b_strip = b_pack.as_ptr().add(jr * NR_C64 * 2 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_C64).min(MR_C64);
                                let a_strip = a_pack.as_ptr().add(ir * MR_C64 * 2 * kc);
                                let c_tile =
                                    c.add(((ic + ir * MR_C64) * ldc + jc + jr * NR_C64) * 2);

                                if m_tile == MR_C64 && n_tile == NR_C64 {
                                    kernel_4x2_c64(
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_c64(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// CGEMM (Complex<f32>) drivers — same M-slab + 2D-grid structure as the
// ZGEMM path, but with f32 lanes (8 per ymm vs 4 for f64) yielding a
// wider register tile (4 rows × 4 complex cols vs ZGEMM's 4×2).
// ---------------------------------------------------------------------

thread_local! {
    static A_PACK_TLS_C32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_C32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn with_a_pack_c32<F: FnOnce(&mut [f32])>(min_capacity: usize, f: F) {
    A_PACK_TLS_C32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

fn with_b_pack_c32<F: FnOnce(&mut [f32])>(min_capacity: usize, f: F) {
    B_PACK_TLS_C32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// CGEMM entry — parallel M-slab dispatch + 2D-grid fallback.
///
/// All inputs are row-major Complex<f32> as `*const f32` / `*mut f32`
/// where each complex element is two contiguous f32s `[re, im]`.
/// `lda`, `ldb`, `ldc` are row strides **in complex elements**.
///
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_avx2_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let p_re = c.add(i * ldc * 2 + j * 2);
                let p_im = p_re.add(1);
                if beta_re == 0.0 && beta_im == 0.0 {
                    *p_re = 0.0;
                    *p_im = 0.0;
                } else if !(beta_re == 1.0 && beta_im == 0.0) {
                    let cr = *p_re;
                    let ci = *p_im;
                    *p_re = beta_re * cr - beta_im * ci;
                    *p_im = beta_re * ci + beta_im * cr;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_C32 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        // MR_C32 == MR_F32 == 4, so reuse the f32 grid picker (it uses
        // MR_F32 internally for the m/MR_F32 chunk count).
        let (mw, nw) = pick_grid_f32(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_C32).max(MR_C32);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_C32).max(NR_C32);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                // Each complex = 2 f32s, so multiply offsets by 2.
                let a_slab = a_send.get().add(chunk.m_start * lda * 2);
                let b_slab = b_send.get().add(chunk.n_start * 2);
                let c_slab = c_send
                    .get()
                    .add(chunk.m_start * ldc * 2 + chunk.n_start * 2);
                gemm_seq_avx2_c32(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha_re,
                    alpha_im,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta_re,
                    beta_im,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx2_c32(
            m, n, k, alpha_re, alpha_im, a, lda, b, ldb, beta_re, beta_im, c, ldc,
        );
    }
}

/// Sequential 5-loop CGEMM. Mirror of `gemm_seq_avx2_c64` with f32 lanes.
///
/// SAFETY: AVX2+FMA must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_seq_avx2_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_C32);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR_C32 * 2 * kc;
            let (eff_beta_re, eff_beta_im) = if first_pc {
                (beta_re, beta_im)
            } else {
                (1.0, 0.0)
            };

            with_b_pack_c32(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb * 2 + jc * 2);
                pack_b_c32(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_C32);
                    let a_pack_size = n_strips_m * MR_C32 * 2 * kc;

                    with_a_pack_c32(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda * 2 + pc * 2);
                        pack_a_c32(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_C32).min(NR_C32);
                            let b_strip = b_pack.as_ptr().add(jr * NR_C32 * 2 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_C32).min(MR_C32);
                                let a_strip = a_pack.as_ptr().add(ir * MR_C32 * 2 * kc);
                                let c_tile =
                                    c.add(((ic + ir * MR_C32) * ldc + jc + jr * NR_C32) * 2);

                                if m_tile == MR_C32 && n_tile == NR_C32 {
                                    kernel_4x4_c32(
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_c32(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// AVX-512 ZGEMM / CGEMM drivers — same M-slab + 2D-grid structure as
// the AVX2 complex paths, with wider register tiles (4x4 for ZGEMM,
// 4x8 for CGEMM) reflecting the doubled f64/f32 lane count of zmm.
// Cfg-gated on the `avx512` feature.
// ---------------------------------------------------------------------

// f64 complex thread-local B-pack for AVX-512 ZGEMM (NR=4 vs AVX2's 2).
#[cfg(feature = "avx512")]
thread_local! {
    static B_PACK_TLS_AVX512_C64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_AVX512_C32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "avx512")]
fn with_b_pack_avx512_c64<F: FnOnce(&mut [f64])>(min_capacity: usize, f: F) {
    B_PACK_TLS_AVX512_C64.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

#[cfg(feature = "avx512")]
fn with_b_pack_avx512_c32<F: FnOnce(&mut [f32])>(min_capacity: usize, f: F) {
    B_PACK_TLS_AVX512_C32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0.0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// AVX-512 ZGEMM entry — parallel M-slab dispatch + 2D-grid fallback.
/// SAFETY: AVX-512F must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_avx512_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let p_re = c.add(i * ldc * 2 + j * 2);
                let p_im = p_re.add(1);
                if beta_re == 0.0 && beta_im == 0.0 {
                    *p_re = 0.0;
                    *p_im = 0.0;
                } else if !(beta_re == 1.0 && beta_im == 0.0) {
                    let cr = *p_re;
                    let ci = *p_im;
                    *p_re = beta_re * cr - beta_im * ci;
                    *p_im = beta_re * ci + beta_im * cr;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_AVX512_C64 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        // MR_AVX512_C64 == 4 == MR, reuse f64 grid picker.
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw))
            .next_multiple_of(MR_AVX512_C64)
            .max(MR_AVX512_C64);
        let n_per = (n.div_ceil(nw))
            .next_multiple_of(NR_AVX512_C64)
            .max(NR_AVX512_C64);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda * 2);
                let b_slab = b_send.get().add(chunk.n_start * 2);
                let c_slab = c_send
                    .get()
                    .add(chunk.m_start * ldc * 2 + chunk.n_start * 2);
                gemm_seq_avx512_c64(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha_re,
                    alpha_im,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta_re,
                    beta_im,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx512_c64(
            m, n, k, alpha_re, alpha_im, a, lda, b, ldb, beta_re, beta_im, c, ldc,
        );
    }
}

/// Sequential AVX-512 ZGEMM 5-loop driver. SAFETY: AVX-512F required.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_seq_avx512_c64(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f64,
    alpha_im: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta_re: f64,
    beta_im: f64,
    c: *mut f64,
    ldc: usize,
) {
    use super::pack::pack_b_c64_nr4;
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_AVX512_C64);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR_AVX512_C64 * 2 * kc;
            let (eff_beta_re, eff_beta_im) = if first_pc {
                (beta_re, beta_im)
            } else {
                (1.0, 0.0)
            };

            with_b_pack_avx512_c64(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb * 2 + jc * 2);
                pack_b_c64_nr4(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_AVX512_C64);
                    let a_pack_size = n_strips_m * MR_AVX512_C64 * 2 * kc;

                    // MR_AVX512_C64 == MR_C64 == 4, so reuse the AVX2
                    // pack_a_c64 + its TLS buffer.
                    with_a_pack_c64(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda * 2 + pc * 2);
                        pack_a_c64(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_AVX512_C64).min(NR_AVX512_C64);
                            let b_strip = b_pack.as_ptr().add(jr * NR_AVX512_C64 * 2 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_AVX512_C64).min(MR_AVX512_C64);
                                let a_strip = a_pack.as_ptr().add(ir * MR_AVX512_C64 * 2 * kc);
                                let c_tile = c.add(
                                    ((ic + ir * MR_AVX512_C64) * ldc + jc + jr * NR_AVX512_C64) * 2,
                                );

                                if m_tile == MR_AVX512_C64 && n_tile == NR_AVX512_C64 {
                                    kernel_4x4_c64_avx512(
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_c64_avx512(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

/// AVX-512 CGEMM entry — parallel M-slab dispatch + 2D-grid fallback.
/// SAFETY: AVX-512F must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_avx512_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let p_re = c.add(i * ldc * 2 + j * 2);
                let p_im = p_re.add(1);
                if beta_re == 0.0 && beta_im == 0.0 {
                    *p_re = 0.0;
                    *p_im = 0.0;
                } else if !(beta_re == 1.0 && beta_im == 0.0) {
                    let cr = *p_re;
                    let ci = *p_im;
                    *p_re = beta_re * cr - beta_im * ci;
                    *p_im = beta_re * ci + beta_im * cr;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_AVX512_C32 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid_f32(m, n, k, workers);

        let m_per = (m.div_ceil(mw))
            .next_multiple_of(MR_AVX512_C32)
            .max(MR_AVX512_C32);
        let n_per = (n.div_ceil(nw))
            .next_multiple_of(NR_AVX512_C32)
            .max(NR_AVX512_C32);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda * 2);
                let b_slab = b_send.get().add(chunk.n_start * 2);
                let c_slab = c_send
                    .get()
                    .add(chunk.m_start * ldc * 2 + chunk.n_start * 2);
                gemm_seq_avx512_c32(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    alpha_re,
                    alpha_im,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    beta_re,
                    beta_im,
                    c_slab,
                    ldc,
                );
            });
        });
    } else {
        gemm_seq_avx512_c32(
            m, n, k, alpha_re, alpha_im, a, lda, b, ldb, beta_re, beta_im, c, ldc,
        );
    }
}

/// Sequential AVX-512 CGEMM 5-loop driver. SAFETY: AVX-512F required.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_seq_avx512_c32(
    m: usize,
    n: usize,
    k: usize,
    alpha_re: f32,
    alpha_im: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta_re: f32,
    beta_im: f32,
    c: *mut f32,
    ldc: usize,
) {
    use super::pack::pack_b_c32_nr8;
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_AVX512_C32);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc = (k - pc).min(KC);
            let b_pack_capacity = nr_strips * NR_AVX512_C32 * 2 * kc;
            let (eff_beta_re, eff_beta_im) = if first_pc {
                (beta_re, beta_im)
            } else {
                (1.0, 0.0)
            };

            with_b_pack_avx512_c32(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb * 2 + jc * 2);
                pack_b_c32_nr8(kc, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_AVX512_C32);
                    let a_pack_size = n_strips_m * MR_AVX512_C32 * 2 * kc;

                    // Reuse AVX2 pack_a_c32 (MR same).
                    with_a_pack_c32(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda * 2 + pc * 2);
                        pack_a_c32(mc, kc, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_AVX512_C32).min(NR_AVX512_C32);
                            let b_strip = b_pack.as_ptr().add(jr * NR_AVX512_C32 * 2 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_AVX512_C32).min(MR_AVX512_C32);
                                let a_strip = a_pack.as_ptr().add(ir * MR_AVX512_C32 * 2 * kc);
                                let c_tile = c.add(
                                    ((ic + ir * MR_AVX512_C32) * ldc + jc + jr * NR_AVX512_C32) * 2,
                                );

                                if m_tile == MR_AVX512_C32 && n_tile == NR_AVX512_C32 {
                                    kernel_4x8_c32_avx512(
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                } else {
                                    kernel_edge_c32_avx512(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        alpha_re,
                                        alpha_im,
                                        a_strip,
                                        b_strip,
                                        eff_beta_re,
                                        eff_beta_im,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// i8 quantized GEMM (u8 × i8 → i32) drivers — same M-slab + 2D-grid
// structure as the FP paths, with a 4-K-block inner loop driven by
// vpmaddubsw+vpmaddwd (AVX2 baseline) or vpdpbusd (AVX-VNNI fast path).
// ---------------------------------------------------------------------

thread_local! {
    static A_PACK_TLS_I8: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_I8: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

fn with_a_pack_i8<F: FnOnce(&mut [u8])>(min_capacity: usize, f: F) {
    A_PACK_TLS_I8.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

fn with_b_pack_i8<F: FnOnce(&mut [i8])>(min_capacity: usize, f: F) {
    B_PACK_TLS_I8.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// Returns true iff the running CPU supports AVX-VNNI (Alder Lake +,
/// Tiger Lake +, Sapphire Rapids +, Zen4 +) **and** the `avxvnni` cargo
/// feature is enabled. When true, the i8 GEMM dispatch uses the
/// single-instruction `vpdpbusd` fast path.
#[inline]
fn cpu_supports_avxvnni() -> bool {
    #[cfg(feature = "avxvnni")]
    {
        is_x86_feature_detected!("avxvnni")
    }
    #[cfg(not(feature = "avxvnni"))]
    {
        false
    }
}

/// i8 GEMM entry — parallel M-slab dispatch + 2D-grid fallback.
///
/// `a` is row-major u8 (m × k), `b` is row-major i8 (k × n), `c` is
/// row-major i32 (m × n). `lda`, `ldb`, `ldc` are row strides in their
/// respective element types. The kernel internally pads K up to a
/// multiple of 4 with zeros via the pack routines.
///
/// SAFETY: AVX2 must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
pub unsafe fn gemm_avx2_i8(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_I8 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_I8).max(MR_I8);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_I8).max(NR_I8);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx2_i8(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    c_slab,
                    ldc,
                    accumulate,
                );
            });
        });
    } else {
        gemm_seq_avx2_i8(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
    }
}

/// Sequential 5-loop i8 GEMM driver. Picks the AVX-VNNI fast path when
/// the host CPU supports it; falls back to the AVX2 vpmaddubsw chain
/// otherwise.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn gemm_seq_avx2_i8(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let use_vnni = cpu_supports_avxvnni();

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_I8);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc_logical = (k - pc).min(KC);
            // Round kc up to next KB_I8 multiple — pack pads with zeros.
            let kc = pack_kc_i8(kc_logical);
            let b_pack_capacity = nr_strips * NR_I8 * kc;
            let acc_for_call = accumulate || !first_pc;

            with_b_pack_i8(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_i8(kc_logical, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_I8);
                    let a_pack_size = n_strips_m * MR_I8 * kc;

                    with_a_pack_i8(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_u8(mc, kc_logical, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_I8).min(NR_I8);
                            let b_strip = b_pack.as_ptr().add(jr * NR_I8 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_I8).min(MR_I8);
                                let a_strip = a_pack.as_ptr().add(ir * MR_I8 * kc);
                                let c_tile = c.add((ic + ir * MR_I8) * ldc + jc + jr * NR_I8);

                                if m_tile == MR_I8 && n_tile == NR_I8 {
                                    #[cfg(feature = "avxvnni")]
                                    if use_vnni {
                                        kernel_4x8_i8_vnni(
                                            kc,
                                            a_strip,
                                            b_strip,
                                            c_tile,
                                            ldc as isize,
                                            1,
                                            acc_for_call,
                                        );
                                    } else {
                                        kernel_4x8_i8(
                                            kc,
                                            a_strip,
                                            b_strip,
                                            c_tile,
                                            ldc as isize,
                                            1,
                                            acc_for_call,
                                        );
                                    }
                                    #[cfg(not(feature = "avxvnni"))]
                                    {
                                        let _ = use_vnni;
                                        kernel_4x8_i8(
                                            kc,
                                            a_strip,
                                            b_strip,
                                            c_tile,
                                            ldc as isize,
                                            1,
                                            acc_for_call,
                                        );
                                    }
                                } else {
                                    kernel_edge_i8(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc_logical;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// AVX2 i16 GEMM (i16 × i16 → i32) drivers — same structure as the i8
// path but with KB=2 (vpmaddwd reduces 2 K-iters at a time vs
// vpmaddubsw's 4) and i16 inputs.
// ---------------------------------------------------------------------

thread_local! {
    static A_PACK_TLS_I16: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_I16: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
}

fn with_a_pack_i16<F: FnOnce(&mut [i16])>(min_capacity: usize, f: F) {
    A_PACK_TLS_I16.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

fn with_b_pack_i16<F: FnOnce(&mut [i16])>(min_capacity: usize, f: F) {
    B_PACK_TLS_I16.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// i16 GEMM entry — parallel M-slab dispatch + 2D-grid fallback.
/// SAFETY: AVX2 must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
pub unsafe fn gemm_avx2_i16(
    m: usize,
    n: usize,
    k: usize,
    a: *const i16,
    lda: usize,
    b: *const i16,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_I16 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_I16).max(MR_I16);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_I16).max(NR_I16);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx2_i16(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    c_slab,
                    ldc,
                    accumulate,
                );
            });
        });
    } else {
        gemm_seq_avx2_i16(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
    }
}

/// Sequential 5-loop i16 GEMM driver. SAFETY: AVX2 required.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn gemm_seq_avx2_i16(
    m: usize,
    n: usize,
    k: usize,
    a: *const i16,
    lda: usize,
    b: *const i16,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_I16);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc_logical = (k - pc).min(KC);
            let kc = pack_kc_i16(kc_logical);
            let b_pack_capacity = nr_strips * NR_I16 * kc;
            let acc_for_call = accumulate || !first_pc;

            with_b_pack_i16(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_i16(kc_logical, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_I16);
                    let a_pack_size = n_strips_m * MR_I16 * kc;

                    with_a_pack_i16(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_i16(mc, kc_logical, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_I16).min(NR_I16);
                            let b_strip = b_pack.as_ptr().add(jr * NR_I16 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_I16).min(MR_I16);
                                let a_strip = a_pack.as_ptr().add(ir * MR_I16 * kc);
                                let c_tile = c.add((ic + ir * MR_I16) * ldc + jc + jr * NR_I16);

                                if m_tile == MR_I16 && n_tile == NR_I16 {
                                    kernel_4x8_i16(
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                } else {
                                    kernel_edge_i16(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc_logical;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// AVX2 i8 signed-signed GEMM (i8 × i8 → i32) drivers — uses the
// dedicated 4×8 / KB=2 kernel that does VPMOVSXBW + VPMADDWD inline,
// halving input bandwidth vs the sign-extending wrapper around the i16
// kernel.
// ---------------------------------------------------------------------

thread_local! {
    // A pack is i16 (widened from i8 during pack to enable the kernel's
    // free 4-byte VPBROADCASTD per row). B pack stays i8.
    static A_PACK_TLS_I8S: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_I8S: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

fn with_a_pack_i8s<F: FnOnce(&mut [i16])>(min_capacity: usize, f: F) {
    A_PACK_TLS_I8S.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

fn with_b_pack_i8s<F: FnOnce(&mut [i8])>(min_capacity: usize, f: F) {
    B_PACK_TLS_I8S.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// i8 signed-signed GEMM entry — parallel M-slab dispatch + 2D-grid
/// fallback. SAFETY: AVX2 must be available; caller verifies.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
pub unsafe fn gemm_avx2_i8s(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_I8S * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw)).next_multiple_of(MR_I8S).max(MR_I8S);
        let n_per = (n.div_ceil(nw)).next_multiple_of(NR_I8S).max(NR_I8S);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx2_i8s(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    c_slab,
                    ldc,
                    accumulate,
                );
            });
        });
    } else {
        gemm_seq_avx2_i8s(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
    }
}

/// Sequential 5-loop i8 signed-signed GEMM driver. SAFETY: AVX2 required.
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn gemm_seq_avx2_i8s(
    m: usize,
    n: usize,
    k: usize,
    a: *const i8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_I8S);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc_logical = (k - pc).min(KC);
            let kc = pack_kc_i8s(kc_logical);
            let b_pack_capacity = nr_strips * NR_I8S * kc;
            let acc_for_call = accumulate || !first_pc;

            with_b_pack_i8s(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_i8s(kc_logical, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_I8S);
                    let a_pack_size = n_strips_m * MR_I8S * kc;

                    with_a_pack_i8s(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_i8s(mc, kc_logical, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_I8S).min(NR_I8S);
                            let b_strip = b_pack.as_ptr().add(jr * NR_I8S * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_I8S).min(MR_I8S);
                                let a_strip = a_pack.as_ptr().add(ir * MR_I8S * kc);
                                let c_tile = c.add((ic + ir * MR_I8S) * ldc + jc + jr * NR_I8S);

                                if m_tile == MR_I8S && n_tile == NR_I8S {
                                    kernel_4x8_i8_signed(
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                } else {
                                    kernel_edge_i8_signed(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc_logical;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// AVX-512 i16 GEMM driver — same structure as the AVX2 i16 path but
// with the 4×16 zmm register tile (NR=16 vs AVX2's NR=8). Cfg-gated on
// `avx512`; the kernel additionally requires runtime avx512bw detection.
// ---------------------------------------------------------------------

#[cfg(feature = "avx512")]
use super::kernel_avx512_i16::{
    MR_AVX512_I16, NR_AVX512_I16, kernel_4x16_i16_avx512, kernel_edge_i16_avx512,
};
#[cfg(feature = "avx512")]
use super::pack::{pack_a_i16_mr4_avx512, pack_b_i16_nr16};

#[cfg(feature = "avx512")]
thread_local! {
    static A_PACK_TLS_AVX512_I16: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
    static B_PACK_TLS_AVX512_I16: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "avx512")]
fn with_a_pack_avx512_i16<F: FnOnce(&mut [i16])>(min_capacity: usize, f: F) {
    A_PACK_TLS_AVX512_I16.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

#[cfg(feature = "avx512")]
fn with_b_pack_avx512_i16<F: FnOnce(&mut [i16])>(min_capacity: usize, f: F) {
    B_PACK_TLS_AVX512_I16.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < min_capacity {
            buf.resize(min_capacity, 0);
        }
        f(&mut buf[..min_capacity]);
    });
}

/// AVX-512 i16 GEMM entry — parallel M-slab dispatch + 2D-grid fallback.
/// SAFETY: AVX-512F + AVX-512BW must be available at runtime; caller
/// verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn gemm_avx512_i16(
    m: usize,
    n: usize,
    k: usize,
    a: *const i16,
    lda: usize,
    b: *const i16,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_AVX512_I16 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw))
            .next_multiple_of(MR_AVX512_I16)
            .max(MR_AVX512_I16);
        let n_per = (n.div_ceil(nw))
            .next_multiple_of(NR_AVX512_I16)
            .max(NR_AVX512_I16);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx512_i16(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    c_slab,
                    ldc,
                    accumulate,
                );
            });
        });
    } else {
        gemm_seq_avx512_i16(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
    }
}

#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn gemm_seq_avx512_i16(
    m: usize,
    n: usize,
    k: usize,
    a: *const i16,
    lda: usize,
    b: *const i16,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_AVX512_I16);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc_logical = (k - pc).min(KC);
            let kc = pack_kc_i16(kc_logical);
            let b_pack_capacity = nr_strips * NR_AVX512_I16 * kc;
            let acc_for_call = accumulate || !first_pc;

            with_b_pack_avx512_i16(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_i16_nr16(kc_logical, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_AVX512_I16);
                    let a_pack_size = n_strips_m * MR_AVX512_I16 * kc;

                    with_a_pack_avx512_i16(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_i16_mr4_avx512(mc, kc_logical, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_AVX512_I16).min(NR_AVX512_I16);
                            let b_strip = b_pack.as_ptr().add(jr * NR_AVX512_I16 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_AVX512_I16).min(MR_AVX512_I16);
                                let a_strip = a_pack.as_ptr().add(ir * MR_AVX512_I16 * kc);
                                let c_tile = c
                                    .add((ic + ir * MR_AVX512_I16) * ldc + jc + jr * NR_AVX512_I16);

                                if m_tile == MR_AVX512_I16 && n_tile == NR_AVX512_I16 {
                                    kernel_4x16_i16_avx512(
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                } else {
                                    kernel_edge_i16_avx512(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc_logical;
            first_pc = false;
        }

        jc += nc;
    }
}

// ---------------------------------------------------------------------
// AVX-512 VNNI i8 GEMM driver — same structure as the AVX2 i8 path
// (M-slab + 2D grid, 4-K-block inner loop) but with the AVX-512 4x16
// register tile (NR=16 vs AVX2's NR=8). Cfg-gated on `avx512`; the
// kernel additionally requires runtime avx512vnni detection.
// ---------------------------------------------------------------------

/// AVX-512 VNNI i8 GEMM entry — parallel M-slab dispatch + 2D-grid
/// fallback. SAFETY: AVX-512F + AVX-512 VNNI must be available at
/// runtime; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f,avx512vnni")]
pub unsafe fn gemm_avx512_vnni_i8(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                for j in 0..n {
                    *c.add(i * ldc + j) = 0;
                }
            }
        }
        return;
    }

    let parallel = m.max(n).max(k) >= PARALLEL_MIN_DIM && m >= MR_AVX512_I8 * 2;

    if parallel {
        let workers = matmul_pool().current_num_threads();
        let (mw, nw) = pick_grid(m, n, k, workers);

        let m_per = (m.div_ceil(mw))
            .next_multiple_of(MR_AVX512_I8)
            .max(MR_AVX512_I8);
        let n_per = (n.div_ceil(nw))
            .next_multiple_of(NR_AVX512_I8)
            .max(NR_AVX512_I8);

        let mut chunks: Vec<GridChunk> = Vec::with_capacity(mw * nw);
        let mut m_start = 0;
        while m_start < m {
            let m_size = (m - m_start).min(m_per);
            let mut n_start = 0;
            while n_start < n {
                let n_size = (n - n_start).min(n_per);
                chunks.push(GridChunk {
                    m_start,
                    m_size,
                    n_start,
                    n_size,
                });
                n_start += n_size;
            }
            m_start += m_size;
        }

        let a_send = SendConstPtr(a);
        let b_send = SendConstPtr(b);
        let c_send = SendMutPtr(c);

        matmul_pool().install(|| {
            chunks.par_iter().for_each(|chunk| unsafe {
                let a_slab = a_send.get().add(chunk.m_start * lda);
                let b_slab = b_send.get().add(chunk.n_start);
                let c_slab = c_send.get().add(chunk.m_start * ldc + chunk.n_start);
                gemm_seq_avx512_vnni_i8(
                    chunk.m_size,
                    chunk.n_size,
                    k,
                    a_slab,
                    lda,
                    b_slab,
                    ldb,
                    c_slab,
                    ldc,
                    accumulate,
                );
            });
        });
    } else {
        gemm_seq_avx512_vnni_i8(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
    }
}

/// Sequential 5-loop AVX-512 VNNI i8 GEMM. Mirror of `gemm_seq_avx2_i8`
/// but using the 4x16 zmm kernel and NR=16 B-pack.
///
/// SAFETY: AVX-512F + AVX-512 VNNI must be available; caller verifies.
#[cfg(feature = "avx512")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn gemm_seq_avx512_vnni_i8(
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    lda: usize,
    b: *const i8,
    ldb: usize,
    c: *mut i32,
    ldc: usize,
    accumulate: bool,
) {
    use super::pack::pack_b_i8_nr16;
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let nr_strips = nc.div_ceil(NR_AVX512_I8);

        let mut pc = 0;
        let mut first_pc = true;
        while pc < k {
            let kc_logical = (k - pc).min(KC);
            let kc = pack_kc_i8(kc_logical);
            let b_pack_capacity = nr_strips * NR_AVX512_I8 * kc;
            let acc_for_call = accumulate || !first_pc;

            with_b_pack_i8(b_pack_capacity, |b_pack| {
                let b_panel_ptr = b.add(pc * ldb + jc);
                pack_b_i8_nr16(kc_logical, nc, b_panel_ptr, ldb, b_pack);

                let mut ic = 0;
                while ic < m {
                    let mc = (m - ic).min(MC);
                    let n_strips_m = mc.div_ceil(MR_AVX512_I8);
                    let a_pack_size = n_strips_m * MR_AVX512_I8 * kc;

                    // MR_AVX512_I8 == MR_I8 == 4, reuse the AVX2 pack.
                    with_a_pack_i8(a_pack_size, |a_pack| {
                        let a_panel_ptr = a.add(ic * lda + pc);
                        pack_a_u8(mc, kc_logical, a_panel_ptr, lda, a_pack);

                        for jr in 0..nr_strips {
                            let n_tile = (nc - jr * NR_AVX512_I8).min(NR_AVX512_I8);
                            let b_strip = b_pack.as_ptr().add(jr * NR_AVX512_I8 * kc);

                            for ir in 0..n_strips_m {
                                let m_tile = (mc - ir * MR_AVX512_I8).min(MR_AVX512_I8);
                                let a_strip = a_pack.as_ptr().add(ir * MR_AVX512_I8 * kc);
                                let c_tile =
                                    c.add((ic + ir * MR_AVX512_I8) * ldc + jc + jr * NR_AVX512_I8);

                                if m_tile == MR_AVX512_I8 && n_tile == NR_AVX512_I8 {
                                    kernel_4x16_i8_avx512(
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                } else {
                                    kernel_edge_i8_avx512(
                                        m_tile,
                                        n_tile,
                                        kc,
                                        a_strip,
                                        b_strip,
                                        c_tile,
                                        ldc as isize,
                                        1,
                                        acc_for_call,
                                    );
                                }
                            }
                        }
                    });

                    ic += mc;
                }
            });

            pc += kc_logical;
            first_pc = false;
        }

        jc += nc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        b: &[f64],
        alpha: f64,
        beta: f64,
        c: &mut [f64],
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                let cv = c[i * n + j];
                c[i * n + j] = alpha * acc + beta * cv;
            }
        }
    }

    fn run_case(m: usize, n: usize, k: usize, alpha: f64, beta: f64) {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.013 - 0.7).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.011 + 0.3).collect();
        let mut c_ours: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.005 - 1.0).collect();
        let mut c_ref = c_ours.clone();

        unsafe {
            gemm_avx2_f64(
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                beta,
                c_ours.as_mut_ptr(),
                n,
            );
        }
        naive(m, n, k, &a, &b, alpha, beta, &mut c_ref);

        for i in 0..m * n {
            let diff = (c_ours[i] - c_ref[i]).abs();
            let scale = c_ref[i].abs().max(1.0);
            assert!(
                diff / scale < 1e-10,
                "{m}x{n}x{k} alpha={alpha} beta={beta}: idx={i} ours={} ref={} diff={}",
                c_ours[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn gemm_8x6x4() {
        run_case(8, 6, 4, 1.0, 0.0);
    }
    #[test]
    fn gemm_8x6x32() {
        run_case(8, 6, 32, 1.0, 0.0);
    }
    #[test]
    fn gemm_16x12x32() {
        run_case(16, 12, 32, 1.0, 0.0);
    }
    #[test]
    fn gemm_17x13x9() {
        run_case(17, 13, 9, 1.0, 0.0);
    }
    #[test]
    fn gemm_64x64x64() {
        run_case(64, 64, 64, 1.0, 0.0);
    }
    #[test]
    fn gemm_100x100x100() {
        run_case(100, 100, 100, 1.0, 0.0);
    }
    #[test]
    fn gemm_128x128x128() {
        run_case(128, 128, 128, 1.0, 0.0);
    }
    #[test]
    fn gemm_alpha_beta() {
        run_case(50, 50, 50, 0.7, 1.3);
    }
    #[test]
    fn gemm_alpha_zero_beta() {
        run_case(40, 30, 20, 1.5, 0.0);
    }
    #[test]
    fn gemm_300x300x300_kc_split() {
        run_case(300, 300, 300, 1.0, 0.0);
    }
    #[test]
    fn gemm_8x6x300_kc_split() {
        run_case(8, 6, 300, 1.0, 0.0);
    }
    #[test]
    fn gemm_3x3x3_tiny() {
        run_case(3, 3, 3, 1.0, 0.0);
    }
    #[test]
    fn gemm_1x1x10() {
        run_case(1, 1, 10, 1.0, 0.0);
    }
    /// Exercises the M-slab split with a 512-row problem (the typical
    /// regression-target size) — verifies all slabs reconstruct correctly.
    #[test]
    fn gemm_512x512x512_parallel() {
        run_case(512, 512, 512, 1.0, 0.0);
    }
    /// Verifies M-slab split when M is not a multiple of (workers * MR).
    #[test]
    fn gemm_513x257x129_parallel_irregular() {
        run_case(513, 257, 129, 1.0, 0.0);
    }
    /// Verifies M-slab split with beta != 0 (accumulation across slabs).
    #[test]
    fn gemm_512x512x512_parallel_beta() {
        run_case(512, 512, 512, 0.5, 0.7);
    }
}
