"""Auditable author-recipe baselines and new split-robustness experiments."""
import os
os.environ.update(NUMBA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,json,hashlib,random,time,importlib.util,warnings,subprocess
import numpy as np
from numba import set_num_threads
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier,RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ.get('EIPPV_PROJECT',str(HERE/'eippv_extension/project')))
OUT=Path(os.environ.get('EIPPV_EXTENSION_OUT',str(HERE.parent/'outputs/modern_baselines_stability')))
VENDOR=HERE/'modern_baselines'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from rocket_official_experiment import seed_rng,original,fit_biases
from rocket_mpeg7_confirm import features,resample,contours
from rocket_directional_pool import load
CASES=tuple(f'{s}_{n}' for s in ('mpeg7','hand','fish','osuleaf') for n in (32,64))
SEEDS=(23,37,71);SPLITS=(101,211,307,401,503)
ALPHAS=(.1,1,10,100,1000)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def author_multi():
    path=VENDOR/'multirocket_ridge_compat.py'
    spec=importlib.util.spec_from_file_location('multirocket_ridge_compat',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod
def initialize():
    OUT.mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True)
    code=(VENDOR/'MultiRocket/multirocket/multirocket.py').read_text(encoding='utf-8')
    assert code.count('from multirocket.logistic_regression import LogisticRegression')==1
    assert code.count('normalize=False')==1
    adapted=code.replace('from multirocket.logistic_regression import LogisticRegression','')
    adapted=adapted.replace('normalize=False','')
    (VENDOR/'multirocket_ridge_compat.py').write_text(adapted,encoding='utf-8')
    sources={}
    for name in ['MultiRocket','SelF-Rocket']:
        p=VENDOR/name
        commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        sources[name]=dict(commit=commit,url='https://github.com/'+('ChangWeiTan/MultiRocket' if name=='MultiRocket' else 'ANR-MYEL/SelF-Rocket'),files={str(f.relative_to(p)).replace('\\','/'):sha(f) for f in p.rglob('*.py')})
    protocol=dict(date='2026-11-16',sources=sources,compatibility='MultiRocket ridge-only adapter removes unused torch logistic import and obsolete normalize=False. Feature-generating code unchanged. SelF-Rocket code unmodified; warnings reset to visible after import.',
        cases=CASES,seeds=SEEDS,standard_recipes=['MultiRocket default requested 50000, actual 49728; author RidgeClassifierCV alpha grid','SelF-Rocket default 10 runs, 5000 sampled features per mini-classifier, all 15 IR/PO candidates; selected final dimension 9996 or 19992; internal generated dimension 99960'],
        controlled_recipe='MultiRocket and EI both actual 9408 features; whole transform and scaler refit inside grouped/stratified 3-fold training CV; alpha grid .1,1,10,100,1000. MultiRocket requested10000 produces9408 due 8-way feature allocation. No default50k baseline is replaced by this budget control.',
        standard_validation_caveat='Author recipes preserve internal training-only selection, including feature calibration/scaling before internal validation. No test data enters fitting. These recipes are not labeled fold-local strict CV; MPEG duplicate groups are not implemented inside untouched author recipes.',
        stability=dict(split_seeds=SPLITS,model_seed=23,cases=CASES,methods=['circular','phase16','phase32','eippv'],budget=9996,design='Five stratified random resplits of pooled existing sources, preserving approximate source train proportion; one random assignment per exact-duplicate group across raw/32/64 representations; all views of an object remain on one side. Not new unseen data. Whole feature transform/scaler refit in every inner fold. Repeated splits overlap and are not independent datasets.'),
        high_k='Fixed EI-trained classifier; evaluate K16/32/64/128 feature error, score error, prediction agreement and matched timings. No tuning of K on test labels. No assertion of equivalence from nonsignificance.',
        original_input_sha={c:sha(ROOT/f'contour_submission/experiments/input_{c}.npz') for c in CASES},
        analysis='Report source-specific effects. Object-paired intervals for original split are descriptive. Repeated-split ranges/SD and win counts are descriptive, not an independent n=5 significance test. Statistical differences cannot establish novelty.',
        scope='Follow-up on previously inspected sources. No untouched test source or fixed-pose repeated acquisition is claimed.')
    target=OUT/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text(encoding='utf-8'))==json.loads(json.dumps(protocol)), 'Protocol already frozen'
    else:dump(target,protocol)

def prepare_splits():
    """Union exact duplicates before allocating objects, keeping all phases together."""
    report=[]
    for source in ('mpeg7','hand','fish','osuleaf'):
        if source=='mpeg7':x,y=contours();fraction=.5
        else:
            tr,yt,te,ye=load({'hand':'HandOutlines','fish':'Fish','osuleaf':'OSULeaf'}[source])
            x=np.vstack((tr,te));y=np.r_[yt,ye];fraction=len(tr)/len(x)
        base={n:resample(x,n,0.) for n in (32,64)}
        parent=np.arange(len(x))
        def root(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for arr in [x,base[32],base[64]]:
            seen={}
            for i,row in enumerate(arr):
                k=row.tobytes()
                if k in seen:
                    a,b=root(i),root(seen[k]);parent[a]=b
                else:seen[k]=i
        groups=np.array([root(i) for i in range(len(x))]);unique=np.unique(groups)
        glabel=[]
        for g in unique:
            labels=np.unique(y[groups==g]);assert len(labels)==1,(source,g,labels)
            glabel.append(labels[0])
        phases=np.r_[0.,np.random.default_rng(20261108).uniform(0,1,8)]
        allviews={n:np.array([resample(x,n,p) for p in phases]) for n in (32,64)}
        for split in SPLITS:
            gt,gv=train_test_split(unique,train_size=fraction,random_state=split,stratify=glabel)
            ti=np.flatnonzero(np.isin(groups,gt));vi=np.flatnonzero(np.isin(groups,gv))
            assert not set(groups[ti])&set(groups[vi])
            for n in (32,64):
                assert not {r.tobytes() for r in base[n][ti]}&{r.tobytes() for r in base[n][vi]}
                path=OUT/f'inputs/{source}_{n}_split{split}.npz'
                if not path.exists():np.savez_compressed(path,train=base[n][ti],labels=y[ti],views=allviews[n][:,vi],truth=y[vi],object_ids=vi,train_ids=ti,train_groups=groups[ti],test_groups=groups[vi],phases=phases)
                report.append(dict(case=f'{source}_{n}',split=split,train=len(ti),test=len(vi),unique_groups=len(unique),duplicate_records=len(x)-len(unique),train_test_group_overlap=0,input_sha256=sha(path)))
    dump(OUT/'resplit_input_audit.json',report)

def input_data(case,split=0):
    p=OUT/f'inputs/{case}_split{split}.npz' if split else ROOT/f'contour_submission/experiments/input_{case}.npz'
    with np.load(p) as z:d={k:z[k] for k in z.files}
    if not split:
        known={r.tobytes() for r in d['train']};keep=np.array([r.tobytes() not in known for r in d['views'][0]])
        for k in ['views']:d[k]=d[k][:,keep]
        for k in ['truth','object_ids']:d[k]=d[k][keep]
        d['train_groups']=np.array([hashlib.sha256(r.tobytes()).hexdigest() for r in d['train']])
    return d,p

def params(x,method,seed,budget):
    seed_rng(seed)
    if method=='multi':
        m=author_multi();a=np.ascontiguousarray(x,dtype=np.float64)
        return m.fit(a,num_features=budget//8),m.fit(np.ascontiguousarray(np.diff(a,axis=1)),num_features=budget//8)
    d,c=original._fit_dilations(x.shape[1],budget,32);q=original._quantiles(84*int(c.sum()))
    return d,c,fit_biases(x,d,c,q,seed,True)

_MULTI=None
def transform(x,p,method):
    global _MULTI
    if method=='multi':
        if _MULTI is None:_MULTI=author_multi()
        a=np.ascontiguousarray(x,dtype=np.float64)
        return np.nan_to_num(_MULTI.transform(a,np.ascontiguousarray(np.diff(a,axis=1)),*p,4))
    step={'circular':1,'phase16':16,'phase32':32,'eippv':0}[method]
    return features(np.ascontiguousarray(x),*p,step,step>1)

def strict_fit(inputs, method, seed, budget):
    """Select ridge alpha with fold-local feature fitting, then refit on all training data."""
    train_signals, train_labels = (inputs['train'], inputs['labels'])
    training_groups = inputs['train_groups']
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261116).split(train_signals, train_labels, training_groups)) if len(set(training_groups)) < len(training_groups) else list(StratifiedKFold(n_splits=3, shuffle=True, random_state=20261116).split(train_signals, train_labels))
    correct_per_alpha = np.zeros(len(ALPHAS))
    cv_records = []
    # Threshold calibration and scaling belong inside each training fold;
    # validation objects must not influence either fitted transform.
    for fold_index, (fit_indices, validation_indices) in enumerate(folds):
        assert not set(training_groups[fit_indices]) & set(training_groups[validation_indices])
        parameters = params(train_signals[fit_indices], method, seed, budget)
        fit_features = transform(train_signals[fit_indices], parameters, method).astype(np.float64)
        validation_features = transform(train_signals[validation_indices], parameters, method).astype(np.float64)
        scaler = StandardScaler().fit(fit_features)
        fit_features = scaler.transform(fit_features)
        validation_features = scaler.transform(validation_features)
        for alpha_index, ridge_alpha in enumerate(ALPHAS):
            classifier = RidgeClassifier(alpha=ridge_alpha).fit(fit_features, train_labels[fit_indices])
            num_correct = int((classifier.predict(validation_features) == train_labels[validation_indices]).sum())
            correct_per_alpha[alpha_index] += num_correct
            cv_records.append(dict(fold=fold_index, alpha=ridge_alpha, correct=num_correct, total=len(validation_indices)))
    # ALPHAS is ordered: argmax retains the first candidate when scores tie.
    alpha = ALPHAS[int(np.argmax(correct_per_alpha))]
    parameters = params(train_signals, method, seed, budget)
    fit_features = transform(train_signals, parameters, method).astype(np.float64)
    scaler = StandardScaler().fit(fit_features)
    model = RidgeClassifier(alpha=alpha).fit(scaler.transform(fit_features), train_labels)
    return (parameters, scaler, model, dict(alpha=alpha, feature_dim=fit_features.shape[1], cv=cv_records))

def worker(case,recipe,seed,split=0):
    set_num_threads(2);start=time.perf_counter();d,inputpath=input_data(case,split)
    random.seed(seed);np.random.seed(seed);seed_rng(seed)
    selected={};p=None
    with threadpool_limits(limits=1):
        if recipe=='multi_standard':
            m=author_multi().MultiRocket(num_features=50000,classifier='ridge')
            m.fit(np.ascontiguousarray(d['train'],dtype=np.float64),d['labels'],predict_on_train=False)
            pred=np.array([m.predict(np.ascontiguousarray(v,dtype=np.float64)) for v in d['views']])
            selected=dict(alpha=float(m.classifier[-1].alpha_),feature_dim=int(sum(m.base_parameters[1])*84*8))
        elif recipe=='self_standard':
            sys.path.insert(0,str(VENDOR/'SelF-Rocket/code'))
            from self_rocket import SelFRocket
            warnings.resetwarnings();m=SelFRocket()
            m.fit(np.ascontiguousarray(d['train'],dtype=np.float64),d['labels'])
            pred=np.array([m.predict(np.ascontiguousarray(v,dtype=np.float64)) for v in d['views']])
            dim=(m.nb_kernels//84)*84*(2 if 'MIX' in m.selected_comb_po else 1)
            selected=dict(alpha=float(m.classifier.alpha_),feature_dim=dim,generated_feature_dim=(m.nb_kernels//84)*84*10,selected_pooling=m.selected_comb_po,num_runs=m.num_runs,mini_classifier_features=m.num_features_pc)
        else:
            method=recipe.removesuffix('_controlled');budget=9408 if recipe.endswith('_controlled') else 9996
            p,sc,model,selected=strict_fit(d,method,seed,budget)
            pred=np.array([model.predict(sc.transform(transform(v,p,method).astype(np.float64))) for v in d['views']])
    acc=(pred[1:]==d['truth']).mean(axis=0);flip=(pred[1:]!=pred[0]).mean(axis=0)
    stem=f'{case}_{recipe}_s{seed}_split{split}';dest=OUT/'raw'/stem
    np.savez_compressed(dest.with_suffix('.npz'),predictions=pred,truth=d['truth'],object_ids=d['object_ids'],phases=d['phases'])
    row=dict(case=case,recipe=recipe,seed=seed,split=split,accuracy_pct=100*float(acc.mean()),flip_pct=100*float(flip.mean()),objects=len(acc),input_sha256=sha(inputpath),selected=selected,seconds=time.perf_counter()-start)
    dump(dest.with_suffix('.json'),row)
    print(json.dumps({k:v for k,v in row.items() if k!='selected'}),flush=True)

def run_jobs():
    jobs=[]
    for case in CASES:
        for seed in SEEDS:
            for recipe in ('multi_standard','self_standard','multi_controlled','eippv_controlled'):jobs.append((case,recipe,seed,0))
    for case in CASES:
        for split in SPLITS:
            for recipe in ('circular','phase16','phase32','eippv'):jobs.append((case,recipe,23,split))
    dump(OUT/'job_manifest.json',[dict(case=c,recipe=r,seed=s,split=t) for c,r,s,t in jobs])
    for i,(case,recipe,seed,split) in enumerate(jobs):
        stem=f'{case}_{recipe}_s{seed}_split{split}'
        if (OUT/'raw'/f'{stem}.json').exists():continue
        proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker',case,recipe,str(seed),str(split)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        if proc.stderr:(OUT/'raw'/f'{stem}.warnings.txt').write_text(proc.stderr,encoding='utf-8')
        if proc.returncode:
            (OUT/'raw'/f'{stem}.error.txt').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
            print(f'FAILED {i+1}/{len(jobs)} {stem}',flush=True)
        else:print(f'{i+1}/{len(jobs)} '+proc.stdout.strip(),flush=True)
    print('JOBS_FINISHED',flush=True)

def run_additional():
    jobs=[(c,r,23,t) for c in CASES for t in SPLITS for r in ('multi_standard','self_standard')]
    manifest=[dict(case=c,recipe=r,seed=s,split=t) for c,r,s,t in jobs]
    path=OUT/'additional_manifest.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==manifest
    else:
        assert not any((OUT/'raw'/f'{c}_{r}_s{s}_split{t}.json').exists() for c,r,s,t in jobs)
        dump(OUT/'baseline_resplit_addendum.json',dict(date='2026-11-16',before_additional_results=True,reason='After original-split strong baselines completed, extend both author recipes to all five pre-existing resplits so their comparison with EI is not restricted to one partition. Original protocol and results retained; this is a follow-up design addition, not preregistration.',configurations=80,model_seed=23,split_seeds=SPLITS,recipes=['multi_standard','self_standard'],cases=CASES))
        dump(path,manifest)
    for i,(case,recipe,seed,split) in enumerate(jobs):
        stem=f'{case}_{recipe}_s{seed}_split{split}'
        if (OUT/'raw'/f'{stem}.json').exists():continue
        proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker',case,recipe,str(seed),str(split)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        if proc.stderr:(OUT/'raw'/f'{stem}.warnings.txt').write_text(proc.stderr,encoding='utf-8')
        if proc.returncode:
            (OUT/'raw'/f'{stem}.error.txt').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8');print(f'ADDITIONAL FAILED {i+1}/80 {stem}',flush=True)
        else:print(f'ADDITIONAL {i+1}/80 '+proc.stdout.strip(),flush=True)
    print('ADDITIONAL_FINISHED',flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='init':initialize();prepare_splits()
    elif action=='worker':
        if sys.argv[3]=='circular' and int(sys.argv[5])!=0:
            from fused_stability import joint
            joint(sys.argv[2],int(sys.argv[4]),int(sys.argv[5]))
        else:worker(sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]))
    elif action=='run':run_jobs()
    elif action=='additional':run_additional()
