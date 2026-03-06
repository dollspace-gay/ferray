#!/usr/bin/env python3
"""Speed benchmark: NumPy vs ferrum.

Measures wall-clock time for both implementations on identical workloads.
ferrum timing is measured internally (excludes JSON I/O overhead).

Usage:
    python3 benchmarks/speed_benchmark.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42
WARMUP_ITERS = 3
BENCH_ITERS = 10

FERRUM_BENCH_DIR = Path(__file__).parent / "ferrum_bench"


def find_ferrum_bench():
    for profile in ["release", "debug"]:
        candidate = FERRUM_BENCH_DIR / "target" / profile / "ferrum-bench"
        if candidate.exists():
            return str(candidate)
    return None


def time_numpy(func, data, iters):
    """Time a NumPy function, return median time in microseconds."""
    # Warmup
    for _ in range(WARMUP_ITERS):
        func(data)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        func(data)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns -> us

    times.sort()
    return times[len(times) // 2]  # median


def time_ferrum(bench_bin, func_name, size_str, data, iters):
    """Time a ferrum function via the bench binary, return median internal time in microseconds."""
    input_json = json.dumps(data.tolist())

    # Warmup
    for _ in range(WARMUP_ITERS):
        subprocess.run(
            [bench_bin, func_name, size_str],
            input=input_json, capture_output=True, text=True, timeout=120,
        )

    times = []
    for _ in range(iters):
        result = subprocess.run(
            [bench_bin, func_name, size_str],
            input=input_json, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None
        output = json.loads(result.stdout)
        elapsed_ns = output.get("elapsed_ns", 0)
        times.append(elapsed_ns / 1000.0)  # ns -> us

    times.sort()
    return times[len(times) // 2]  # median


def format_time(us):
    """Format microseconds to human-readable string."""
    if us < 1:
        return f"{us * 1000:.0f} ns"
    elif us < 1000:
        return f"{us:.1f} us"
    elif us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1_000_000:.2f} s"


def main():
    bench_bin = find_ferrum_bench()
    if bench_bin is None:
        print("ERROR: ferrum-bench binary not found. Build with:")
        print(f"  cd {FERRUM_BENCH_DIR} && cargo build --release")
        sys.exit(1)

    rng = np.random.default_rng(SEED)

    print("=" * 80)
    print("Speed Benchmark: NumPy vs ferrum")
    print("=" * 80)
    print(f"NumPy version:  {np.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Iterations:     {BENCH_ITERS} (median of)")
    print(f"Warmup:         {WARMUP_ITERS} iterations")
    print()

    # Define benchmarks: (name, category, sizes, numpy_func, ferrum_func_name, input_gen)
    benchmarks = []

    # Ufuncs
    ufunc_sizes = [1_000, 10_000, 100_000, 1_000_000]
    ufunc_list = [
        ("sin", np.sin, lambda n, rng: rng.uniform(-np.pi, np.pi, n)),
        ("cos", np.cos, lambda n, rng: rng.uniform(-np.pi, np.pi, n)),
        ("tan", np.tan, lambda n, rng: rng.uniform(-1.5, 1.5, n)),
        ("exp", np.exp, lambda n, rng: rng.uniform(-5.0, 5.0, n)),
        ("log", np.log, lambda n, rng: rng.uniform(0.01, 100.0, n)),
        ("sqrt", np.sqrt, lambda n, rng: rng.uniform(0.0, 1000.0, n)),
        ("arctan", np.arctan, lambda n, rng: rng.uniform(-100.0, 100.0, n)),
        ("tanh", np.tanh, lambda n, rng: rng.uniform(-5.0, 5.0, n)),
    ]

    for func_name, np_func, input_gen in ufunc_list:
        for size in ufunc_sizes:
            data = input_gen(size, rng)
            benchmarks.append({
                "name": func_name,
                "category": "ufunc",
                "size": size,
                "data": data,
                "numpy_func": np_func,
                "ferrum_name": func_name,
            })

    # Stats reductions
    stats_sizes = [1_000, 10_000, 100_000, 1_000_000]
    stats_list = [
        ("sum", np.sum),
        ("mean", np.mean),
        ("var", lambda x: np.var(x, ddof=0)),
        ("std", lambda x: np.std(x, ddof=0)),
    ]

    for func_name, np_func in stats_list:
        for size in stats_sizes:
            data = rng.standard_normal(size)
            benchmarks.append({
                "name": func_name,
                "category": "stats",
                "size": size,
                "data": data,
                "numpy_func": np_func,
                "ferrum_name": func_name,
            })

    # Matmul
    matmul_sizes = [(10, 10), (50, 50), (100, 100)]
    for rows, cols in matmul_sizes:
        a = rng.standard_normal((rows, cols))
        b = rng.standard_normal((cols, rows))
        combined = np.concatenate([a.ravel(), b.ravel()])
        benchmarks.append({
            "name": "matmul",
            "category": "linalg",
            "size": rows,
            "data": combined,
            "numpy_func": lambda d, r=rows, c=cols: (
                d[:r*c].reshape(r, c) @ d[r*c:].reshape(c, r)
            ),
            "ferrum_name": "matmul",
            "size_str": f"{rows}x{cols}",
        })

    # FFT
    fft_sizes = [64, 1024, 16384, 65536]
    for size in fft_sizes:
        data = rng.standard_normal(size)
        benchmarks.append({
            "name": "fft",
            "category": "fft",
            "size": size,
            "data": data,
            "numpy_func": np.fft.fft,
            "ferrum_name": "fft",
        })

    # Run benchmarks
    results = []
    total = len(benchmarks)

    for i, bench in enumerate(benchmarks):
        size_str = bench.get("size_str", str(bench["size"]))
        label = f"{bench['name']}/{size_str}"
        sys.stdout.write(f"  [{i+1:3d}/{total}] {label:<25} ")
        sys.stdout.flush()

        # Time NumPy
        np_us = time_numpy(bench["numpy_func"], bench["data"], BENCH_ITERS)

        # Time ferrum
        fe_us = time_ferrum(
            bench_bin, bench["ferrum_name"], size_str,
            bench["data"], BENCH_ITERS,
        )

        if fe_us is not None and fe_us > 0:
            ratio = fe_us / np_us
            winner = "ferrum" if ratio < 1.0 else "numpy"
            speedup = f"{1/ratio:.2f}x" if ratio < 1.0 else f"{ratio:.2f}x"
            sys.stdout.write(
                f"numpy={format_time(np_us):>10}  ferrum={format_time(fe_us):>10}  "
                f"{'<--' if winner == 'ferrum' else '   '} {speedup} {'faster' if winner == 'ferrum' else 'slower'}\n"
            )
        else:
            ratio = None
            sys.stdout.write(f"numpy={format_time(np_us):>10}  ferrum=ERROR\n")

        results.append({
            "name": bench["name"],
            "category": bench["category"],
            "size": bench["size"],
            "size_str": size_str,
            "numpy_us": np_us,
            "ferrum_us": fe_us,
            "ratio": ratio,
        })

    # Print summary table
    print()
    print("=" * 80)
    print("Summary Table (median time, lower is better)")
    print("=" * 80)
    print()

    current_cat = None
    hdr = f"{'Function':<12} {'Size':<10} {'NumPy':>12} {'ferrum':>12} {'Ratio':>8} {'Winner':>8}"
    print(hdr)
    print("-" * 70)

    for r in results:
        if r["category"] != current_cat:
            if current_cat is not None:
                print()
            current_cat = r["category"]
            print(f"  [{current_cat.upper()}]")

        np_t = format_time(r["numpy_us"])
        if r["ferrum_us"] is not None:
            fe_t = format_time(r["ferrum_us"])
            ratio = r["ratio"]
            if ratio < 1.0:
                ratio_str = f"{1/ratio:.1f}x"
                winner = "ferrum"
            else:
                ratio_str = f"{ratio:.1f}x"
                winner = "numpy"
        else:
            fe_t = "ERROR"
            ratio_str = "N/A"
            winner = "N/A"

        print(f"  {r['name']:<12} {r['size_str']:<10} {np_t:>12} {fe_t:>12} {ratio_str:>8} {winner:>8}")

    print("-" * 70)

    # Output JSON for BENCHMARKS.md generation
    json_path = Path(__file__).parent / "speed_results.json"
    json_results = []
    for r in results:
        json_results.append({
            "name": r["name"],
            "category": r["category"],
            "size": r["size"],
            "size_str": r["size_str"],
            "numpy_us": r["numpy_us"],
            "ferrum_us": r["ferrum_us"],
            "ratio": r["ratio"],
        })
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
