#!/usr/bin/env python3
"""End-to-end KNN two-step: argpartition(dist,k-1)[:k] then argsort(dist[sel]).
Compare the COMPOSED neighbor index list ferray-vs-numpy, with exact-tie dists."""
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

npr=np.random.default_rng(7)
rng=random.Random(7)

# Build many distance arrays. We run TWO harness passes: first the argpartition
# of every dist array, then (in a second pass) the argsort of each selected
# sub-array. To keep it single-process-correct we just call the harness twice.

cases=[]  # (dist, k)
for t in range(2000):
    n=rng.choice([5,10,20,50,100,200,500])
    dim=rng.choice([1,2,3,4])
    pts=npr.integers(-3,4,size=(n,dim)).astype(np.float64)
    q=npr.integers(-3,4,size=dim).astype(np.float64)
    dist=((pts-q)**2).sum(axis=1)
    k=rng.choice([1,2,3,5,7,10])
    if k<=n:
        cases.append((list(dist),k))

def run(lines):
    inp="\n".join(lines)+"\n"
    p=subprocess.run([HARNESS],input=inp,capture_output=True,text=True)
    if p.returncode!=0:
        print("HARNESS FAIL",p.stderr[-2000:]); sys.exit(2)
    return p.stdout.strip("\n").split("\n")

# pass 1: argpartition
p1_lines=["P "+str(k-1)+" "+str(len(d))+" "+" ".join(tok(v) for v in d) for (d,k) in cases]
p1=run(p1_lines)

# build sub-arrays for pass 2 from FERRAY's selection (the real pipeline uses
# ferray's own argpartition output to pick the sub-array, then ferray's argsort)
p2_lines=[]; sels=[]
for (d,k),ol in zip(cases,p1):
    ap=list(map(int,ol.split()))
    sel=ap[:k]
    sels.append(sel)
    sub=[d[i] for i in sel]
    p2_lines.append("S "+str(len(sub))+" "+" ".join(tok(v) for v in sub))
p2=run(p2_lines)

mm=0; ex=[]
for (d,k),sel,ol in zip(cases,sels,p2):
    order=list(map(int,ol.split()))
    ferray_neigh=[sel[o] for o in order]
    # numpy reference end-to-end
    arr=np.array(d,dtype=np.float64)
    nap=np.argpartition(arr,k-1)
    nsel=nap[:k]
    norder=np.argsort(arr[nsel],kind='quicksort')
    np_neigh=list(map(int,nsel[norder]))
    if ferray_neigh!=np_neigh:
        mm+=1
        if len(ex)<12: ex.append((d,k,np_neigh,ferray_neigh))
print(f"knn_e2e cases={len(cases)} mismatches={mm}")
for (d,k,npn,frn) in ex:
    print("----k",k,"n",len(d))
    print("  dist=",[round(x,2) for x in d][:30])
    print("  np_neigh=",npn)
    print("  fr_neigh=",frn)
sys.exit(1 if mm else 0)
