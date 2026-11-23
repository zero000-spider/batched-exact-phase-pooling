from batched_study import *

def case_benchmark(case,budget):
    set_num_threads(2);d,_=input_data(case);path=STUDY/'replay'/f'{case}_s23_f{budget}_hybrid'
    r=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));p=params(d['train'],'eippv',23,budget);order=prepare_orders(p[1],p[2])
    methods={'direct_EI':lambda a:features(a,*p,0,False),'cached_EI':lambda a:batched_features(a,*p,order,-2),'hybrid_EI':lambda a:batched_features(a,*p,order,0,a.shape[1]),'K32_cached_fixed':lambda a:batched_features(a,*p,order,-32),'K1024_cached_fixed':lambda a:batched_features(a,*p,order,-1024)}
    rows=[]
    with threadpool_limits(limits=1):
        ft=methods['hybrid_EI'](d['train']).astype(np.float64);sc=StandardScaler().fit(ft);m=RidgeClassifier(alpha=r['alpha']).fit(sc.transform(ft),d['labels'])
        for batchsize in (1,64):
            batch=np.ascontiguousarray(d['views'][1,:batchsize]);prediction=lambda fn:m.predict(sc.transform(fn(batch).astype(np.float64)))
            exact=prediction(methods['hybrid_EI'])
            with np.load(path.with_suffix('.npz')) as z:np.testing.assert_array_equal(exact,z['predictions'][1,:batchsize])
            for name in methods:prediction(methods[name])
            times={name:[] for name in methods};rng=np.random.default_rng(2026111623);mismatch={}
            for repeat in range(9):
                for name in rng.permutation(list(methods)):
                    gc.collect();start=time.perf_counter();pred=prediction(methods[name]);elapsed=1000*(time.perf_counter()-start);times[name].append(elapsed);mismatch[name]=int((pred!=exact).sum())
            for name,v in times.items():rows.append(dict(case=case,budget=budget,actual_features=len(p[2]),batch=batchsize,method=name,median_ms=float(np.median(v)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75)),raw_ms=json.dumps(v),prediction_mismatches_on_batch=mismatch[name],classifier='same EI-trained model/scaler for all feature backends',scope='full feature extraction + float64 conversion + scaling + prediction; compilation/fitting/input reads excluded'))
    writecsv(STUDY/'benchmarks'/f'{case}_f{budget}_e2e.csv',rows);print(f'{case} F{budget} END_TO_END_DONE',flush=True)

if __name__=='__main__':
    if len(sys.argv)>2:case_benchmark(sys.argv[1],int(sys.argv[2]))
    else:
        for case in CASES:
            for budget in (9996,49728):
                marker=STUDY/'benchmarks'/f'{case}_f{budget}_e2e.csv'
                if marker.exists():continue
                proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),case,str(budget)],capture_output=True,text=True)
                if proc.returncode:raise RuntimeError(proc.stderr)
                print(proc.stdout.strip(),flush=True)
