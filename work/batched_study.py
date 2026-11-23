"""Numerical preservation, full-pipeline replay, and fair matched cost study."""
from batched_pooling import *
from pathlib import Path
import csv,gc,subprocess
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
STUDY=HERE.parent/'outputs/shared_threshold_study'
OLD=HERE.parent/'outputs/evidence_chain_strict'
BUDGETS=(756,9996,49728)

def writecsv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fast_transform(signals, parameters, backend='hybrid'):
    """Prepare threshold orders once per call and select the requested pooling backend."""
    threshold_orders = prepare_orders(parameters[1], parameters[2])
    pooling_mode = {'hybrid': 0, 'cached': -2, 'K32': -32, 'K1024': -1024}[backend]
    return batched_features(np.ascontiguousarray(signals), *parameters, threshold_orders, pooling_mode, len(signals[0]) if backend == 'hybrid' else 0)
def old_path(case,seed):return OLD/(f"{case}_s{seed}_eippv_f9996"+('_groupcv' if case.startswith('mpeg7') else ''))

def initialize():
    STUDY.mkdir(exist_ok=True);(STUDY/'replay').mkdir(exist_ok=True);(STUDY/'benchmarks').mkdir(exist_ok=True)
    protocol=dict(date='2026-11-16',hypothesis='Exploit multiple thresholds on each shared convolution response to reduce exact phase-averaging cost without changing the statistical target. No claim of inventing generic CDF sweeps or threshold duration.',
        interpolation='Same periodic piecewise-linear response and strict > threshold semantics as prior EI.',
        algorithm='Segment-uniform mixture survival plus atoms for flat segments; compensated event sweep sorted by response values. Bias orders precomputed once per fitted transform. Hybrid uses direct evaluation if B<8, cached endpoints if 8<=B<L, and sweep if B>=L.',
        design_status='Development used only Fish training inputs and synthetic numerical checks. All original datasets previously inspected. This is an implementation/complexity contribution, not an unseen-data accuracy confirmation.',
        baselines=['original direct EI','cached-endpoint EI','sweep EI','hybrid EI','original optimized finite K32 counts','cached endpoints with compile-time K32','cached endpoints with compile-time K1024','independent sorted phase-sample K32 with floating-point tie guard','sorted circular hard PPV','pinned original MiniRocket transform'],
        fairness='Strengthened finite-K code shares endpoints and specializes fixed K. Training-only backend profiling considers original/cached/sorted K32. No baseline replaced by a deliberately slow enumerator. All matched methods share data/budgets/threads.',
        numerical=dict(cases=CASES,seeds=SEEDS,budget=9996,feature_atol=2e-7,groundtruth='original direct EI plus high-precision synthetic references',workflow='Refit thresholds/scaler/alpha inside each original inner fold, compare selected alpha and all test predictions with prior formal outputs. Full train/test feature differences and finite-K equality audited on representative full-data passes.'),
        timing=dict(cases=CASES,budgets=BUDGETS,seed=23,batch=64,warmed_repeats=9,order_seed=2026111619,numba_threads=2,blas_threads=1,exclude='compilation, input reads, one-time parameter preparation; those preparation costs reported separately',stage='after all numerical/replay workers finish',interpretation='Local measured implementation performance, no hardware-independent acceleration guarantee'),
        source_sha256={name:sha(HERE/name) for name in ['batched_pooling.py','audit_batched_pooling.py','batched_study.py']})
    target=STUDY/'frozen_protocol.json'
    if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==json.loads(json.dumps(protocol))
    else:dump(target,protocol)

def replay(case,seed,budget=9996,backend='hybrid'):
    set_num_threads(2);start=time.perf_counter();d,ip=input_data(case);x,y,g=d['train'],d['labels'],d['train_groups'];cv=[];correct=np.zeros(len(ALPHAS))
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261116).split(x,y,g)) if len(set(g))<len(g) else list(StratifiedKFold(n_splits=3,shuffle=True,random_state=20261116).split(x,y))
    with threadpool_limits(limits=1):
        for fold,(tr,va) in enumerate(folds):
            assert not set(g[tr])&set(g[va]);p=params(x[tr],'eippv',seed,budget)
            ft=fast_transform(x[tr],p,backend).astype(np.float64);fv=fast_transform(x[va],p,backend).astype(np.float64)
            sc=StandardScaler().fit(ft);ft=sc.transform(ft);fv=sc.transform(fv)
            for ai,a in enumerate(ALPHAS):
                m=RidgeClassifier(alpha=a).fit(ft,y[tr]);c=int((m.predict(fv)==y[va]).sum());correct[ai]+=c;cv.append(dict(fold=fold,alpha=a,correct=c,total=len(va)))
        alpha=ALPHAS[int(np.argmax(correct))];p=params(x,'eippv',seed,budget);ft=fast_transform(x,p,backend).astype(np.float64);sc=StandardScaler().fit(ft);m=RidgeClassifier(alpha=alpha).fit(sc.transform(ft),y)
        pred=[]
        for view in d['views']:
            pieces=[]
            for i in range(0,len(view),64):pieces.extend(m.predict(sc.transform(fast_transform(view[i:i+64],p,backend).astype(np.float64))))
            pred.append(pieces)
    pred=np.array(pred);row=dict(case=case,seed=seed,budget=budget,backend=backend,alpha=alpha,feature_dim=ft.shape[1],accuracy_pct=100*float((pred[1:]==d['truth']).mean()),flip_pct=100*float((pred[1:]!=pred[0]).mean()),input_sha256=sha(ip),cv=cv,seconds=time.perf_counter()-start)
    if budget==9996 and backend=='hybrid':
        suffix='_groupcv' if case.startswith('mpeg7') else '';reference=OLD/f'{case}_s{seed}_eippv_f9996{suffix}'
        refrow=json.loads(reference.with_suffix('.json').read_text(encoding='utf-8'))
        with np.load(reference.with_suffix('.npz')) as z:
            with np.load(ip) as src:keep=np.array([r.tobytes() not in {t.tobytes() for t in src['train']} for r in src['views'][0]])
            mismatch=int((z['predictions'][:,keep]!=pred).sum());np.testing.assert_array_equal(z['object_ids'][keep],d['object_ids'])
        row.update(prior_predictions=pred.size,prediction_mismatches=mismatch,alpha_identical=alpha==refrow['selected']['alpha'],cv_counts_identical=cv==refrow['fold_validation'])
    path=STUDY/'replay'/f'{case}_s{seed}_f{budget}_{backend}';np.savez_compressed(path.with_suffix('.npz'),predictions=pred,truth=d['truth'],object_ids=d['object_ids'],phases=d['phases']);dump(path.with_suffix('.json'),row)
    print(json.dumps({k:v for k,v in row.items() if k!='cv'}),flush=True)

def audit_case(case):
    set_num_threads(2);d,_=input_data(case);p=params(d['train'],'eippv',23,9996);orders=prepare_orders(p[1],p[2]);maxdiff=0.;changed=0;values=0;finite_equal=True
    with threadpool_limits(limits=1):
        for matrix in [d['train'],*d['views']]:
            for i in range(0,len(matrix),64):
                a=np.ascontiguousarray(matrix[i:i+64]);ref=features(a,*p,0,False);fast=batched_features(a,*p,orders,0,a.shape[1]);delta=abs(ref.astype(np.float64)-fast)
                maxdiff=max(maxdiff,float(delta.max()));changed+=int((ref!=fast).sum());values+=ref.size;assert float(delta.max())<=2e-7,(case,i,float(delta.max()))
                base=features(a,*p,32,True);alt=batched_features(a,*p,orders,-32);np.testing.assert_array_equal(base,alt)
        # The independent sorting backend is much slower; check a fixed held-out batch.
        a=np.ascontiguousarray(d['views'][1,:16]);np.testing.assert_array_equal(features(a,*p,32,True),batched_features(a,*p,orders,32))
    row=dict(case=case,seed=23,feature_values=values,non_bitwise_equal_values=changed,max_abs_difference=maxdiff,tolerance=2e-7,cached_K32_bitwise_equal=True,sorted_K32_held_batch_bitwise_equal=True)
    dump(STUDY/f'numerical_{case}.json',row);print(json.dumps(row),flush=True)

def benchmark(case,budget):
    set_num_threads(2);d,_=input_data(case);p=params(d['train'],'eippv',23,budget);start=time.perf_counter();orders=prepare_orders(p[1],p[2]);prep_ms=1000*(time.perf_counter()-start)
    batch=np.ascontiguousarray(d['views'][1,:64]);trainbatch=np.ascontiguousarray(d['train'][:32]);seed_rng(23);op=original.fit(d['train'],num_features=budget)
    methods={
        'direct_EI':lambda a:features(a,*p,0,False),'cached_EI':lambda a:batched_features(a,*p,orders,-2),
        'sweep_EI':lambda a:batched_features(a,*p,orders,0),'hybrid_EI':lambda a:batched_features(a,*p,orders,0,a.shape[1]),
        'K32_original_count':lambda a:features(a,*p,32,True),'K32_cached_fixed':lambda a:batched_features(a,*p,orders,-32),
        'K1024_cached_fixed':lambda a:batched_features(a,*p,orders,-1024),'K32_sorted':lambda a:batched_features(a,*p,orders,32),
        'hard_sorted':lambda a:batched_features(a,*p,orders,-1),'original_MiniRocket':lambda a:original.transform(a,op)}
    pilot=[]
    with threadpool_limits(limits=1):
        for name in ['K32_original_count','K32_cached_fixed','K32_sorted']:
            fn=methods[name];fn(trainbatch);tt=[]
            for _ in range(3):
                t=time.perf_counter();fn(trainbatch);tt.append(1000*(time.perf_counter()-t))
            pilot.append(dict(case=case,budget=budget,method=name,batch=len(trainbatch),median_ms=float(np.median(tt)),raw_ms=json.dumps(tt)))
        selected=min(pilot,key=lambda x:x['median_ms'])['method']
        timing_names=[k for k in methods if k!='K32_sorted']
        if selected=='K32_sorted':timing_names.append(selected)
        for name in timing_names:methods[name](batch)
        rng=np.random.default_rng(2026111619);timings={n:[] for n in timing_names}
        for repeat in range(9):
            for name in rng.permutation(timing_names):
                gc.collect();t=time.perf_counter();a=methods[name](batch);elapsed=1000*(time.perf_counter()-t);assert np.isfinite(a).all();timings[name].append(elapsed)
    rows=[dict(case=case,budget=budget,actual_features=len(p[2]),method=k,batch=len(batch),median_ms=float(np.median(v)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75)),raw_ms=json.dumps(v),bias_order_preparation_ms=prep_ms if k in ('sweep_EI','hybrid_EI','hard_sorted','K32_sorted') else 0.,additional_order_bytes=orders.nbytes if k in ('sweep_EI','hybrid_EI','hard_sorted','K32_sorted') else 0,selected_K32_training_backend=selected) for k,v in timings.items()]
    path=STUDY/'benchmarks'/f'{case}_f{budget}';writecsv(path.with_suffix('.csv'),rows);writecsv(path.with_name(path.name+'_pilot.csv'),pilot);print(f'{case} F{budget} BENCHMARK_DONE',flush=True)

def run_stage(stage):
    if stage=='replay':jobs=[('replay',c,str(s),'9996','hybrid') for c in CASES for s in SEEDS]
    elif stage=='audit':jobs=[('audit_case',c) for c in CASES]
    elif stage=='benchmark':jobs=[('benchmark_case',c,str(b)) for c in CASES for b in BUDGETS]
    elif stage=='large':jobs=[('replay',c,'23','49728',method) for c in CASES for method in ('hybrid','K32')]
    else:raise ValueError(stage)
    for i,args in enumerate(jobs):
        if args[0]=='replay':marker=STUDY/'replay'/f'{args[1]}_s{args[2]}_f{args[3]}_{args[4]}.json'
        elif args[0]=='audit_case':marker=STUDY/f'numerical_{args[1]}.json'
        else:marker=STUDY/'benchmarks'/f'{args[1]}_f{args[2]}.csv'
        if marker.exists():continue
        proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
        if proc.returncode:
            (STUDY/f'{stage}_failure_{i}.txt').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8');raise RuntimeError(proc.stderr)
        if proc.stderr:(STUDY/f'{stage}_warnings_{i}.txt').write_text(proc.stderr,encoding='utf-8')
        print(f'{i+1}/{len(jobs)} '+proc.stdout.strip(),flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='init':initialize()
    elif action=='replay' and len(sys.argv)>2:replay(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
    elif action=='audit_case':audit_case(sys.argv[2])
    elif action=='benchmark_case':benchmark(sys.argv[2],int(sys.argv[3]))
    else:run_stage(action)
