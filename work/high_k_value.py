"""Separate fixed-classifier quadrature error from independently refitted accuracy."""
from modern_extension import *
import csv,gc

def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run_case(case,defer_timings=False):
    set_num_threads(2);d,path=input_data(case);dest=OUT/'high_k';dest.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        p,sc,mod,selected=strict_fit(d,'eippv',23,9996)
        rows=[];allpred={};truth=d['truth'];ks=(16,32,64,128,1024)
        exactpred=[];kp={k:[] for k in ks};agg={k:dict(abs_sum=0.,sq_sum=0.,count=0,max_error=0.,score_error_max=0.,certified=0,total=0) for k in ks}
        for vi,view in enumerate(d['views']):
            fe=transform(view,p,'eippv').astype(np.float64);ze=sc.transform(fe);pe=mod.predict(ze);se=mod.decision_function(ze);exactpred.append(pe)
            for k in ks:
                fk=features(np.ascontiguousarray(view),*p,k,True).astype(np.float64)
                delta=fk-fe;zk=sc.transform(fk);pk=mod.predict(zk);sk=mod.decision_function(zk);kp[k].append(pk)
                if vi==0:continue
                a=agg[k];a['abs_sum']+=float(abs(delta).sum());a['sq_sum']+=float((delta**2).sum());a['count']+=delta.size;a['max_error']=max(a['max_error'],float(abs(delta).max()))
                err=abs(sk-se);a['score_error_max']=max(a['score_error_max'],float(err.max()))
                if se.ndim==1:cert=abs(se)>err
                else:
                    top=np.sort(se,axis=1)[:,-2:];margin=top[:,-1]-top[:,-2];cert=margin>2*err.max(axis=1)
                assert np.all(pk[cert]==pe[cert])
                a['certified']+=int(cert.sum());a['total']+=len(view)
                assert float(abs(delta).max())<=1/k+3e-6
            del fe,ze
        exactpred=np.array(exactpred)
        allpred['exact']=exactpred
        for k in ks:
            a=agg[k];pred=np.array(kp[k]);allpred[f'K{k}']=pred
            rows.append(dict(case=case,K=k,feature_mae=a['abs_sum']/a['count'],feature_rmse=np.sqrt(a['sq_sum']/a['count']),feature_max_abs_error=a['max_error'],deterministic_bound=1/k,prediction_agreement_pct=100*float((pred[1:]==exactpred[1:]).mean()),margin_certified_same_pct=100*a['certified']/a['total'],exact_accuracy_pct=100*float((exactpred[1:]==truth).mean()),approx_same_classifier_accuracy_pct=100*float((pred[1:]==truth).mean()),maximum_score_error=a['score_error_max'],classifier='EI-trained frozen classifier',model_seed=23))
        write_csv(dest/f'{case}_features.csv',rows)
        np.savez_compressed(dest/f'{case}_predictions.npz',**allpred,truth=truth,object_ids=d['object_ids'],phases=d['phases'])
        np.savez_compressed(dest/f'{case}_pooling_params.npz',dilations=p[0],counts=p[1],biases=p[2])
        dump(dest/f'{case}_model_selection.json',selected)
        print(case+' HIGH_K_FEATURES_DONE',flush=True)
    if not defer_timings:timing_case(case)

def timing_case(case):
    set_num_threads(2);d,_=input_data(case);dest=OUT/'high_k'
    with np.load(dest/f'{case}_pooling_params.npz') as z:p=tuple(z[k] for k in ['dilations','counts','biases'])
    with threadpool_limits(limits=1):
        batch=np.ascontiguousarray(d['views'][1,:64]);methods=[0,16,32,64,128,1024];timings={k:[] for k in methods};rng=np.random.default_rng(2026111609)
        for k in methods:features(batch,*p,k,k>1)
        for repeat in range(7):
            for k in rng.permutation(methods):
                gc.collect();start=time.perf_counter();f=features(batch,*p,int(k),k>1);elapsed=time.perf_counter()-start
                assert np.isfinite(f).all();timings[int(k)].append(1000*elapsed)
        times=[dict(case=case,method='exact' if k==0 else f'K{k}',batch=len(batch),median_ms=float(np.median(v)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75)),raw_ms=json.dumps(v),scope='convolution plus pooling; shared inputs and fitted parameters; compile/scaling/classifier excluded; run after other experiment workers finish') for k,v in timings.items()]
        write_csv(dest/f'{case}_timings.csv',times)
    print(case+' HIGH_K_TIMINGS_DONE',flush=True)

if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'all'
    if action=='feature_case':run_case(sys.argv[2],True)
    elif action=='timing_case':timing_case(sys.argv[2])
    elif action in ('features','timings','all'):
        for c in CASES:
            mode='timing_case' if action=='timings' else 'feature_case' if action=='features' else c
            marker=OUT/f'high_k/{c}_timings.csv' if action in ('all','timings') else OUT/f'high_k/{c}_features.csv'
            if marker.exists():continue
            args=[mode,c] if mode.endswith('_case') else [c]
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),*args],capture_output=True,text=True)
            if proc.returncode:raise RuntimeError(proc.stderr)
            print(proc.stdout.strip(),flush=True)
    else:run_case(action)
