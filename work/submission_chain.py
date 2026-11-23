"""Strict fold-local confirmation with contour baselines. Never alter prior results."""
import os
os.environ.update(NUMBA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from complete_experiments import ROOT,HERE,sha
import sys,json,time,subprocess,itertools
import numpy as np
from numba import set_num_threads
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold,GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT))
from rocket_official_experiment import original,seed_rng,fit_biases
from rocket_mpeg7_confirm import features,fit_blur_biases

OUT=HERE.parent/'outputs/evidence_chain_strict';OUT.mkdir(parents=True,exist_ok=True)
CASES=tuple(f'{s}_{n}' for s in ('mpeg7','hand','fish','osuleaf') for n in (32,64))
METHODS=('original','circular','blur','phase4','phase16','phase32','eippv','fourier_svm','cyclic_1nn')
ALPHAS=(.1,1,10,100,1000)

def parameters(x,seed,budget,method):
 seed_rng(seed)
 if method=='original':return original.fit(x,num_features=budget)
 d,c=original._fit_dilations(x.shape[1],budget,32);q=original._quantiles(84*int(c.sum()))
 b=fit_blur_biases(x,d,c,q,seed) if method=='blur' else fit_biases(x,d,c,q,seed,True)
 return d,c,b

def transform(x,p,method):
 if method=='original':return original.transform(x,p)
 step={'circular':1,'blur':-1,'phase4':4,'phase16':16,'phase32':32,'eippv':0}[method]
 return features(x,*p,step,step>1)

def stem(case,seed,method,budget):
 suffix='_groupcv' if case.startswith('mpeg7') and budget==9996 and os.environ.get('CHAIN_MPEG_GROUPED')=='1' else ''
 return f'{case}_s{seed}_{method}_f{budget}'+suffix

def worker(case,seed,method,budget):
 set_num_threads(2);start=time.monotonic();target=OUT/stem(case,seed,method,budget)
 with np.load(ROOT/f'contour_submission/experiments/input_{case}.npz') as z:
  x=z['train'];y=z['labels'];views=z['views'];truth=z['truth'];ids=z['object_ids'];phases=z['phases']
 assert views.shape[0]==9
 import hashlib
 groups=np.array([hashlib.sha256(r.tobytes()).hexdigest() for r in x])
 grouped=len(set(groups))<len(groups)
 folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261116).split(x,y,groups)) if grouped else list(StratifiedKFold(n_splits=3,shuffle=True,random_state=20261116).split(x,y))
 if grouped:
  for tr,va in folds:assert not set(groups[tr])&set(groups[va])
 validation=[];cv=[]
 with threadpool_limits(limits=1):
  if method=='cyclic_1nn':
   # Minimum Euclidean distance over all integer circular alignments, via FFT correlation.
   fx=np.fft.rfft(x,axis=1);norm=(x*x).sum(axis=1);pred=[]
   for view in views:
    pp=[]
    for v in view:
     corr=np.fft.irfft(fx*np.conj(np.fft.rfft(v)),n=x.shape[1],axis=1).max(axis=1)
     distance=norm+(v*v).sum()-2*corr
     pp.append(y[np.argmin(distance)])
    pred.append(pp)
   pred=np.array(pred);selected={};feature_dim=x.shape[1]
  elif method=='fourier_svm':
   # Magnitudes of scalar radius/outline-sequence Fourier coefficients, not elliptic Fourier coordinates.
   def spec(a):return abs(np.fft.rfft(a,axis=1))[:,1:]
   search=GridSearchCV(make_pipeline(StandardScaler(),SVC(kernel='rbf')),{'svc__C':(.1,1,10,100),'svc__gamma':('scale',.1,1)},cv=folds,n_jobs=1,scoring='accuracy',error_score='raise')
   search.fit(spec(x),y);pred=np.array([search.predict(spec(v)) for v in views]);selected=search.best_params_;feature_dim=spec(x).shape[1]
   cv=[dict(params=p,mean_score=float(score)) for p,score in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'])]
  else:
   sums=np.zeros(len(ALPHAS));denom=0
   for fold,(tr,va) in enumerate(folds):
    assert not set(tr)&set(va)
    p=parameters(x[tr],seed,budget,method)
    ft=transform(x[tr],p,method).astype(np.float64);fv=transform(x[va],p,method).astype(np.float64)
    scale=StandardScaler().fit(ft);ft=scale.transform(ft);fv=scale.transform(fv)
    for ai,alpha in enumerate(ALPHAS):
     model=RidgeClassifier(alpha=alpha).fit(ft,y[tr]);correct=int((model.predict(fv)==y[va]).sum());sums[ai]+=correct
     validation.append(dict(fold=fold,alpha=alpha,correct=correct,total=len(va)))
    denom+=len(va)
   best=int(np.argmax(sums));alpha=ALPHAS[best];selected={'alpha':alpha}
   cv=[dict(alpha=a,mean_score=float(c/denom)) for a,c in zip(ALPHAS,sums)]
   p=parameters(x,seed,budget,method);ft=transform(x,p,method).astype(np.float64);feature_dim=ft.shape[1]
   scale=StandardScaler().fit(ft);model=RidgeClassifier(alpha=alpha).fit(scale.transform(ft),y)
   pred=np.array([model.predict(scale.transform(transform(v,p,method).astype(np.float64))) for v in views])
 accuracy=(pred[1:]==truth).mean(axis=0);flip=(pred[1:]!=pred[0]).mean(axis=0)
 np.savez_compressed(target.with_suffix('.npz'),predictions=pred,truth=truth,object_ids=ids,phases=phases)
 row=dict(case=case,seed=seed,method=method,budget=budget,feature_dim=feature_dim,accuracy=float(accuracy.mean()),flip=float(flip.mean()),clean_accuracy=float((pred[0]==truth).mean()),worst_phase_accuracy=float((pred[1:]==truth).mean(axis=1).min()),selected=selected,cv=cv,fold_validation=validation,validation_scheme='StratifiedGroupKFold_exact_sequence' if grouped else 'StratifiedKFold',seconds=time.monotonic()-start,input_sha256=sha(ROOT/f'contour_submission/experiments/input_{case}.npz'))
 target.with_suffix('.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
 print(json.dumps({k:row[k] for k in ('case','seed','method','budget','accuracy','flip','seconds')}),flush=True)

def main():
 protocol=dict(cases=CASES,methods=METHODS,seeds=[23,37,71],fold_seed=20261116,folds=3,alphas=ALPHAS,primary='EI minus circular: accuracy and flip on four 32-point sources; Holm family 8',secondary='EI versus smoothing, phase4/16/32, Fourier-magnitude RBF-SVM, cyclic Euclidean 1NN; descriptive paired intervals',pooling_budget=9996,extra_budget=756,validation='Convolution thresholds and scaler refit within every training fold. Final model refitted on full training partition; test used only once.',scope='Follow-up analysis on previously inspected datasets; not preregistered.',inputs={c:sha(ROOT/f'contour_submission/experiments/input_{c}.npz') for c in CASES})
 if not (OUT/'protocol.json').exists():(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 jobs=[(c,s,m,9996) for c in CASES for s in (23,37,71) for m in METHODS if m!='cyclic_1nn' or s==23]
 jobs += [(c,s,m,756) for c in CASES if c.endswith('_32') for s in (23,37,71) for m in ('circular','eippv')]
 for i,(case,seed,method,budget) in enumerate(jobs):
  target=OUT/f'{case}_s{seed}_{method}_f{budget}.json'
  if target.exists():continue
  proc=subprocess.run([sys.executable,'-B',__file__,case,str(seed),method,str(budget)],capture_output=True,text=True,cwd=ROOT)
  if proc.returncode:
   (OUT/f'{case}_s{seed}_{method}_f{budget}.error.txt').write_text(proc.stderr,encoding='utf-8');raise RuntimeError(proc.stderr)
  if proc.stderr:(OUT/f'{case}_s{seed}_{method}_f{budget}.warnings.txt').write_text(proc.stderr,encoding='utf-8')
  print(f'{i+1}/{len(jobs)} '+proc.stdout.strip(),flush=True)
 print('ALL '+str(len(jobs))+' CONFIGURATIONS COMPLETE',flush=True)

if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
 else:main()
