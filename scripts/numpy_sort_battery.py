#!/usr/bin/env python3
"""Critic battery: diff ferray argsort/argpartition vs live numpy 2.4.5, index-for-index.

Generates adversarial arrays (size thresholds, heavy ties, NaN/inf/signed-zero,
depth-exhausting, KNN-shaped), computes the numpy oracle, streams the SAME arrays
through ferray's harness binary, and reports any index mismatch with a minimal repro.
"""
import subprocess, sys, random, struct
import numpy as np

assert np.__version__ == "2.4.5", np.__version__

HARNESS = sys.argv[1]

def tok(v):
    if np.isnan(v): return "nan"
    if v == np.inf: return "inf"
    if v == -np.inf: return "-inf"
    # preserve signed zero
    if v == 0.0 and struct.pack('<d', v) == struct.pack('<d', -0.0):
        return "-0"
    return repr(float(v))

def rec_S(a):
    return "S " + str(len(a)) + " " + " ".join(tok(v) for v in a)

def rec_P(a, k):
    return "P " + str(k) + " " + str(len(a)) + " " + " ".join(tok(v) for v in a)

records = []   # (label, kind, array(list of float), kth_or_None, expected_numpy_indices)

def add_sort(label, a):
    a = [float(x) for x in a]
    exp = list(map(int, np.argsort(np.array(a, dtype=np.float64), kind='quicksort')))
    records.append((label, "S", a, None, exp))

def add_part(label, a, k):
    a = [float(x) for x in a]
    if k >= len(a):
        return
    exp = list(map(int, np.argpartition(np.array(a, dtype=np.float64), k)))
    records.append((label, "P", a, k, exp))

rng = random.Random(20260606)
npr = np.random.default_rng(20260606)

SIZES = [2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,127,128,129,200,255,256,257,300,511,512,513,1000,2048,4096]

# 1. SIZE THRESHOLDS: random + heavy-tie at each size, multiple seeds
for n in SIZES:
    for s in range(8):
        a = npr.standard_normal(n)
        add_sort(f"rand_n{n}_s{s}", a)
        for k in sorted(set([0,1,n//2,max(0,n-2),n-1])):
            add_part(f"rand_n{n}_s{s}_k{k}", a, k)
    # heavy tie: small value set
    for s in range(6):
        a = npr.integers(0, 3, size=n).astype(np.float64)   # {0,1,2}
        add_sort(f"tie3_n{n}_s{s}", a)
        for k in sorted(set([0,1,n//2,max(0,n-2),n-1])):
            add_part(f"tie3_n{n}_s{s}_k{k}", a, k)
    # two-valued
    for s in range(4):
        a = npr.integers(0, 2, size=n).astype(np.float64)   # {0,1}
        add_sort(f"tie2_n{n}_s{s}", a)
        for k in sorted(set([0,1,n//2,max(0,n-2),n-1])):
            add_part(f"tie2_n{n}_s{s}_k{k}", a, k)
    # all equal
    add_sort(f"alleq_n{n}", [7.0]*n)
    add_part(f"alleq_n{n}_kmid", [7.0]*n, n//2)
    # few distinct many dup
    for s in range(3):
        vals = [npr.standard_normal() for _ in range(4)]
        a = [vals[rng.randrange(4)] for _ in range(n)]
        add_sort(f"few4_n{n}_s{s}", a)
        add_part(f"few4_n{n}_s{s}_kmid", a, n//2)

# 2. blocks of ties at partition boundaries (the KNN-killer shape)
for n in [50,100,256,257,500,1000]:
    base = list(npr.standard_normal(n))
    # force a tie block in the middle
    for i in range(n//2 - 5, n//2 + 5):
        base[i] = 0.0
    add_sort(f"tieblock_n{n}", base)
    for k in [n//2-3, n//2, n//2+3]:
        add_part(f"tieblock_n{n}_k{k}", base, k)

# 3. DEPTH-EXHAUSTING patterns
for n in [300, 512, 1000, 2048, 4096]:
    # already sorted
    add_sort(f"sorted_n{n}", list(range(n)))
    add_part(f"sorted_n{n}_kmid", list(range(n)), n//2)
    # reverse sorted
    add_sort(f"revsorted_n{n}", list(range(n,0,-1)))
    add_part(f"revsorted_n{n}_kmid", list(range(n,0,-1)), n//2)
    # organ pipe
    op = list(range(n//2)) + list(range(n - n//2, 0, -1))
    add_sort(f"organ_n{n}", op)
    add_part(f"organ_n{n}_kmid", op, n//2)
    # median-of-3 killer-ish: sawtooth
    saw = [ (i*7919) % n for i in range(n) ]
    add_sort(f"saw_n{n}", saw)
    add_part(f"saw_n{n}_kmid", saw, n//2)
    # ramp with duplicated halves
    dup = list(range(n//2)) * 2
    dup = dup[:n]
    add_sort(f"duphalf_n{n}", dup)
    add_part(f"duphalf_n{n}_kmid", dup, n//2)

# 4. NaN / inf / -inf / -0.0
for n in [5, 8, 16, 17, 64, 100, 257, 1000]:
    for s in range(6):
        a = npr.standard_normal(n)
        # sprinkle special values
        idxs = npr.choice(n, size=max(1, n//8), replace=False)
        for j, ix in enumerate(idxs):
            choice = (j + s) % 5
            a[ix] = [np.nan, np.inf, -np.inf, 0.0, -0.0][choice]
        add_sort(f"special_n{n}_s{s}", a)
        for k in sorted(set([0,1,n//2,n-1])):
            add_part(f"special_n{n}_s{s}_k{k}", a, k)
    # all-NaN, leading/trailing NaN
    add_sort(f"allnan_n{n}", [np.nan]*n)
    a = list(npr.standard_normal(n)); a[0]=np.nan; a[-1]=np.nan
    add_sort(f"edgenan_n{n}", a)
    add_part(f"edgenan_n{n}_kmid", a, n//2)
    # signed zeros only
    add_sort(f"signedzero_n{n}", [(-0.0 if i%2 else 0.0) for i in range(n)])

# 6. KNN-shaped two-step with EXACT integer-coordinate tie distances
def knn_two_step(dist, k):
    ap = np.argpartition(np.array(dist, dtype=np.float64), k-1)
    sel = ap[:k]
    sub = np.array(dist, dtype=np.float64)[sel]
    order = np.argsort(sub, kind='quicksort')
    return list(map(int, sel[order]))

# We can't run the two-step through one harness call, but each step is its own
# record; the composed correctness follows if both argpartition and argsort match
# index-for-index. We add the per-step records here with integer-coord ties.
for trial in range(400):
    n = rng.choice([10, 20, 50, 100, 200, 400])
    dim = rng.choice([1,2,3])
    # integer coordinates -> many exact-tie squared distances
    pts = npr.integers(-3, 4, size=(n, dim)).astype(np.float64)
    q = npr.integers(-3, 4, size=dim).astype(np.float64)
    dist = ((pts - q)**2).sum(axis=1)
    k = rng.choice([1,3,5,7, min(n,10)])
    if k <= n:
        add_part(f"knn_t{trial}_n{n}_k{k}", list(dist), k-1)
    add_sort(f"knndist_t{trial}_n{n}", list(dist))

# Build input stream
lines = []
for (label, kind, a, k, exp) in records:
    if kind == "S":
        lines.append(rec_S(a))
    else:
        lines.append(rec_P(a, k))

inp = "\n".join(lines) + "\n"
proc = subprocess.run([HARNESS], input=inp, capture_output=True, text=True)
if proc.returncode != 0:
    print("HARNESS FAILED", proc.returncode)
    print(proc.stderr[-3000:])
    sys.exit(2)
out_lines = proc.stdout.strip("\n").split("\n")
assert len(out_lines) == len(records), (len(out_lines), len(records))

mismatch = 0
examples = []
for (label, kind, a, k, exp), ol in zip(records, out_lines):
    got = [] if ol == "NONE" else list(map(int, ol.split()))
    if got != exp:
        mismatch += 1
        if len(examples) < 12:
            examples.append((label, kind, k, a, exp, got))

print(f"records={len(records)} mismatches={mismatch}")
for (label, kind, k, a, exp, got) in examples:
    # find first differing position
    diff_pos = next((i for i in range(min(len(exp),len(got))) if exp[i]!=got[i]), None)
    print("----")
    print(f"label={label} kind={kind} kth={k} n={len(a)} first_diff_idx={diff_pos}")
    if len(a) <= 40:
        print("  array =", a)
    print("  numpy =", exp[:40], "..." if len(exp)>40 else "")
    print("  ferray=", got[:40], "..." if len(got)>40 else "")
sys.exit(1 if mismatch else 0)
