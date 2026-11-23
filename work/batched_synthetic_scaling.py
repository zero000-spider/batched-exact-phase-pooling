"""Synthetic compute workloads only: no classification labels or generalization claims."""
from batched_study import *

def run(length,budget,family):
    rng=np.random.default_rng(np.random.SeedSequence([20261116,length,budget,0 if family=='white' else 1]));n=80
    if family=='white':a=rng.normal(size=(n,length))
    else:
        t=np.arange(length)/length*2*np.pi;a=np.zeros((n,length))
        for harmonic in range(1,5):a+=rng.normal(size=(n,1))*np.cos(harmonic*t)[None,:]/harmonic+rng.normal(size=(n,1))*np.sin(harmonic*t)[None,:]/harmonic
        a+=.01*rng.normal(size=a.shape)
    a=((a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)).astype(np.float32);train=a[:64];batch=np.ascontiguousarray(a[64:]);p=params(train,'eippv',23,budget);order=prepare_orders(p[1],p[2])
    dest=STUDY/'synthetic';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'L{length}_F{budget}_{family}_input.npz',train=train,test=batch)
    methods={'hybrid_EI':lambda:batched_features(batch,*p,order,0,length),'cached_EI':lambda:batched_features(batch,*p,order,-2),'K32_cached_fixed':lambda:batched_features(batch,*p,order,-32),'K1024_cached_fixed':lambda:batched_features(batch,*p,order,-1024)}
    times={k:[] for k in methods}
    with threadpool_limits(limits=1):
        ref=features(batch,*p,0,False);np.testing.assert_allclose(methods['hybrid_EI'](),ref,rtol=0,atol=2e-7)
        np.testing.assert_array_equal(methods['K32_cached_fixed'](),features(batch,*p,32,True))
        for fn in methods.values():fn()
        order_rng=np.random.default_rng(2026111627)
        for repeat in range(5):
            for name in order_rng.permutation(list(methods)):
                gc.collect();start=time.perf_counter();methods[name]();times[name].append(1000*(time.perf_counter()-start))
    rows=[dict(length=length,budget=budget,actual_features=len(p[2]),family=family,batch=len(batch),method=k,median_ms=float(np.median(v)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75)),raw_ms=json.dumps(v),scope='generated workload; no labels; feature extraction only') for k,v in times.items()]
    writecsv(dest/f'L{length}_F{budget}_{family}.csv',rows);print(f'SYNTHETIC {length} {budget} {family} DONE',flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
    else:
        for length in (32,64,128,256,512):
            for budget in BUDGETS:
                for family in ('white','smooth'):
                    if (STUDY/f'synthetic/L{length}_F{budget}_{family}.csv').exists():continue
                    proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),str(length),str(budget),family],capture_output=True,text=True)
                    if proc.returncode:raise RuntimeError(proc.stderr)
                    print(proc.stdout.strip(),flush=True)
