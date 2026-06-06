#!/usr/bin/env python3
"""High-volume random fuzz + early-exit/max_iters edge probes."""
import subprocess, sys, random, struct
import numpy as np
assert np.__version__=="2.4.5"
HARNESS=sys.argv[1]
def tok(v):
    if np.isnan(v): return "nan"
    if v==np.inf: return "inf"
    if v==-np.inf: return "-inf"
    if v==0.0 and struct.pack('<d',v)==struct.pack('<d',-0.0): return "-0"
    return repr(float(v))
npr=np.random.default_rng(999)
rng=random.Random(999)
records=[]
def add_sort(a):
    a=[float(x) for x in a]
    records.append(("S",a,None,list(map(int,np.argsort(np.array(a,dtype=np.float64),kind='quicksort')))))
def add_part(a,k):
    a=[float(x) for x in a]
    if k>=len(a) or k<0: return
    records.append(("P",a,k,list(map(int,np.argpartition(np.array(a,dtype=np.float64),k)))))

# 1. pure random continuous, many sizes, many seeds
for _ in range(6000):
    n=rng.randint(2,1200)
    scale=rng.choice([1e-300,1e-6,1.0,1e6,1e300])
    a=npr.standard_normal(n)*scale
    add_sort(a)
    add_part(a, rng.randint(0,n-1))

# 2. nearly-sorted (1-2 swaps) to probe is_sorted early-exit boundary
for _ in range(800):
    n=rng.randint(5,1000)
    a=np.arange(n,dtype=np.float64)
    for _ in range(rng.randint(1,3)):
        i,j=rng.randrange(n),rng.randrange(n)
        a[i],a[j]=a[j],a[i]
    add_sort(a)
    add_part(a, rng.randint(0,n-1))
    # exactly sorted (should hit early-exit)
    add_sort(np.arange(n,dtype=np.float64))

# 3. tiny float perturbations near ties (subnormal-level)
for _ in range(800):
    n=rng.randint(8,300)
    base=npr.standard_normal(n)
    a=base.copy()
    for ix in npr.choice(n,size=n//4,replace=False):
        a[ix]=base[rng.randrange(n)]+rng.choice([0.0, 5e-324, -5e-324])
    add_sort(a)
    add_part(a, rng.randint(0,n-1))

# 4. powers-of-two sizes exactly (network boundary alignment)
for n in [4,8,16,32,64,128,256]:
    for _ in range(50):
        a=npr.standard_normal(n)
        add_sort(a)
        for k in range(n):
            add_part(a,k)

def rec(kind,a,k):
    if kind=="S": return "S "+str(len(a))+" "+" ".join(tok(v) for v in a)
    return "P "+str(k)+" "+str(len(a))+" "+" ".join(tok(v) for v in a)
inp="\n".join(rec(k,a,kk) for (k,a,kk,_) in records)+"\n"
proc=subprocess.run([HARNESS],input=inp,capture_output=True,text=True)
if proc.returncode!=0:
    print("HARNESS FAIL",proc.stderr[-2000:]); sys.exit(2)
out=proc.stdout.strip("\n").split("\n")
assert len(out)==len(records),(len(out),len(records))
mm=0; ex=[]
for (kind,a,k,exp),ol in zip(records,out):
    got=[] if ol=="NONE" else list(map(int,ol.split()))
    if got!=exp:
        mm+=1
        if len(ex)<10:
            dp=next((i for i in range(min(len(exp),len(got))) if exp[i]!=got[i]),None)
            ex.append((kind,k,len(a),dp,a,exp,got))
print(f"fuzz records={len(records)} mismatches={mm}")
for (kind,k,n,dp,a,exp,got) in ex:
    print("----",kind,"kth",k,"n",n,"first_diff",dp)
    if n<=24: print("  a=",a)
    print("  np=",exp[:24]); print("  fr=",got[:24])
sys.exit(1 if mm else 0)
