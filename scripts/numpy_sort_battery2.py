#!/usr/bin/env python3
"""Battery 2: high-volume stress on the riskiest paths (unrolled partition,
exact size boundaries, dense ties, depth fallback, kth sweep, big NaN arrays)."""
import subprocess, sys, random, struct
import numpy as np
assert np.__version__ == "2.4.5", np.__version__
HARNESS = sys.argv[1]

def tok(v):
    if np.isnan(v): return "nan"
    if v == np.inf: return "inf"
    if v == -np.inf: return "-inf"
    if v == 0.0 and struct.pack('<d', v) == struct.pack('<d', -0.0): return "-0"
    return repr(float(v))

records = []
def add_sort(label, a):
    a=[float(x) for x in a]
    exp=list(map(int,np.argsort(np.array(a,dtype=np.float64),kind='quicksort')))
    records.append((label,"S",a,None,exp))
def add_part(label,a,k):
    a=[float(x) for x in a]
    if k>=len(a) or k<0: return
    exp=list(map(int,np.argpartition(np.array(a,dtype=np.float64),k)))
    records.append((label,"P",a,k,exp))

npr=np.random.default_rng(424242)
rng=random.Random(424242)

# A. Exhaustive small-N exact (N=2..40) all permutations-ish via many random + tie variants
for n in range(2,41):
    for s in range(40):
        if s < 20:
            a = npr.integers(0, max(2, n//2), size=n).astype(np.float64)  # dense ties
        else:
            a = npr.standard_normal(n)
        add_sort(f"smallA_n{n}_s{s}", a)
        for k in range(n):
            if rng.random() < 0.3 or k in (0,1,n//2,n-1,n-2):
                add_part(f"smallA_n{n}_s{s}_k{k}", a, k)

# B. Unrolled-partition stress: N in the unrolled regime (>128 inside a quicksort
# sub-range means total N must exceed 256; push 257..5000 with dense ties so
# pivots land in tie blocks and compress-store paths are exercised hard.
for n in [257, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]:
    for s in range(10):
        nd = rng.choice([2,3,4,8,16])   # number of distinct values
        a = npr.integers(0, nd, size=n).astype(np.float64)
        add_sort(f"unroll_n{n}_nd{nd}_s{s}", a)
        for k in sorted(set([0,1,3,n//4,n//2,3*n//4,n-2,n-1])):
            add_part(f"unroll_n{n}_nd{nd}_s{s}_k{k}", a, k)
    # continuous random too
    for s in range(6):
        a = npr.standard_normal(n)
        add_sort(f"unrollc_n{n}_s{s}", a)
        for k in sorted(set([0,1,n//2,n-1])):
            add_part(f"unrollc_n{n}_s{s}_k{k}", a, k)

# C. Depth-exhaustion: large adversarial patterns that defeat median-of-4 pivot.
def make_killer(n):
    # classic median-of-3/quickselect killer (Musser): build so the chosen
    # pivots are always extreme. Approximate with a recursive bad pattern.
    a=[0]*n
    # ascending then a few big spikes
    for i in range(n): a[i]=i
    for i in range(0,n,2):
        a[i], a[min(i+1,n-1)] = a[min(i+1,n-1)], a[i]
    return a
for n in [300, 512, 1024, 2048, 4096, 8000]:
    add_sort(f"killer_n{n}", make_killer(n))
    add_part(f"killer_n{n}_kmid", make_killer(n), n//2)
    add_part(f"killer_n{n}_k1", make_killer(n), 1)
    add_part(f"killer_n{n}_klast", make_killer(n), n-1)
    # sorted/reverse with ties
    add_sort(f"sortedtie_n{n}", [ (i//4) for i in range(n)])
    add_part(f"sortedtie_n{n}_kmid", [ (i//4) for i in range(n)], n//2)
    add_sort(f"revtie_n{n}", [ ((n-i)//4) for i in range(n)])

# D. Big NaN arrays (nth_element / std::sort NaN-last fallback)
for n in [257, 500, 1000, 2048, 4096]:
    for s in range(8):
        a = npr.standard_normal(n)
        frac = rng.choice([0.01,0.1,0.3,0.5,0.9])
        m = max(1,int(n*frac))
        idxs = npr.choice(n, size=m, replace=False)
        a[idxs] = np.nan
        add_sort(f"bignan_n{n}_f{frac}_s{s}", a)
        for k in sorted(set([0,1,n//2,n-1])):
            add_part(f"bignan_n{n}_f{frac}_s{s}_k{k}", a, k)
    # mixed nan+inf+signedzero
    for s in range(4):
        a = npr.standard_normal(n)
        for ix in npr.choice(n, size=n//4, replace=False):
            a[ix] = rng.choice([np.nan, np.inf, -np.inf, 0.0, -0.0])
        add_sort(f"bigmix_n{n}_s{s}", a)
        add_part(f"bigmix_n{n}_s{s}_kmid", a, n//2)

# E. tie block straddling kth across all relevant kth
for n in [256, 257, 258, 300, 512]:
    a=list(npr.standard_normal(n))
    lo, hi = n//2-8, n//2+8
    for i in range(lo,hi): a[i]=42.0
    add_sort(f"straddle_n{n}", a)
    for k in range(lo-2, hi+2):
        add_part(f"straddle_n{n}_k{k}", a, k)

def rec(kind,a,k):
    if kind=="S": return "S "+str(len(a))+" "+" ".join(tok(v) for v in a)
    return "P "+str(k)+" "+str(len(a))+" "+" ".join(tok(v) for v in a)

inp="\n".join(rec(kind,a,k) for (_,kind,a,k,_) in records)+"\n"
proc=subprocess.run([HARNESS],input=inp,capture_output=True,text=True)
if proc.returncode!=0:
    print("HARNESS FAILED",proc.returncode); print(proc.stderr[-3000:]); sys.exit(2)
out=proc.stdout.strip("\n").split("\n")
assert len(out)==len(records),(len(out),len(records))
mm=0; ex=[]
for (label,kind,a,k,exp),ol in zip(records,out):
    got=[] if ol=="NONE" else list(map(int,ol.split()))
    if got!=exp:
        mm+=1
        if len(ex)<15:
            dp=next((i for i in range(min(len(exp),len(got))) if exp[i]!=got[i]),None)
            ex.append((label,kind,k,len(a),dp,a,exp,got))
print(f"records={len(records)} mismatches={mm}")
for (label,kind,k,n,dp,a,exp,got) in ex:
    print("----",label,"kind",kind,"kth",k,"n",n,"first_diff",dp)
    if n<=32: print("  a=",a)
    print("  np=",exp[:32]); print("  fr=",got[:32])
sys.exit(1 if mm else 0)
