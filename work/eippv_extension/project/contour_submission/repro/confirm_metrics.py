"""Serial, isolated-worker benchmark and phase-stability confirmation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
from numba import set_num_threads
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from rocket_mpeg7_confirm import contours, features, resample, split_shapes
from rocket_official_experiment import fit_biases, original, seed_rng
from rocket_directional_pool import load

OUT = Path(os.environ.get('EIPPV_OUTPUT_DIR', str(ROOT/'contour_submission'/'experiments')))
METHODS = ('original','circular','phase2','phase4','phase8','phase16','phase32','eippv')
REPEATS = 5
BATCH = 64
MODEL_SEED = int(os.environ.get('EIPPV_MODEL_SEED', '23'))


def prepare():
    OUT.mkdir(exist_ok=True)
    radii,labels=contours()
    tr,te=split_shapes(radii,labels)
    sources={'mpeg7':(radii[tr],labels[tr],radii[te],labels[te],te)}
    train,y,test,truth=load('HandOutlines')
    known={r.tobytes() for r in train}
    keep=np.array([r.tobytes() not in known for r in test])
    sources['hand']=(train,y,test[keep],truth[keep],np.flatnonzero(keep))
    phases=np.random.default_rng(20261108).uniform(0,1,8)
    for source,(x,y,test,truth,ids) in sources.items():
        for n in (32,64):
            target=OUT/f'input_{source}_{n}.npz'
            if target.exists():
                raise FileExistsError(target)
            train=resample(x,n,0.)
            views=np.array([resample(test,n,p) for p in (0.,*phases)])
            np.savez_compressed(target,train=train,labels=y,views=views,truth=truth,
                                object_ids=ids,phases=np.r_[0.,phases])


def stats(values):
    return dict(median=float(np.median(values)),q25=float(np.quantile(values,.25)),
                q75=float(np.quantile(values,.75)),min=float(min(values)),max=float(max(values)),
                raw=[float(x) for x in values])


def worker(case,method):
    set_num_threads(2)
    with threadpool_limits(limits=1),np.load(OUT/f'input_{case}.npz',allow_pickle=False) as source:
        train=source['train']; labels=source['labels']; views=source['views']; truth=source['truth']
        phases=source['phases']; ids=source['object_ids']
        def parameters():
            seed_rng(MODEL_SEED)
            if method=='original':
                return original.fit(train,num_features=10000)
            d,c=original._fit_dilations(train.shape[1],10000,32)
            q=original._quantiles(84*int(c.sum()))
            return d,c,fit_biases(train,d,c,q,MODEL_SEED,True)
        params=parameters()
        def transform(x):
            if method=='original':
                return original.transform(x,params)
            steps={'circular':1,'phase2':2,'phase4':4,'phase8':8,'phase16':16,'phase32':32,'eippv':0}[method]
            return features(x,*params,steps,steps>1)
        batch=np.ascontiguousarray(views[1,:BATCH])
        transform(batch)
        d,c=original._fit_dilations(train.shape[1],10000,32)
        q=original._quantiles(84*int(c.sum()))
        reference=(d,c,fit_biases(train,d,c,q,MODEL_SEED,True))
        features(batch,*reference,0,False)
        calibration_times=[]; feature_times=[]; class_times=[]; total_times=[]
        for repeat in range(REPEATS):
            gc.collect()
            start=perf_counter(); params=parameters(); calibrated=perf_counter()
            ftrain=transform(train); extracted=perf_counter()
            scaler=StandardScaler().fit(ftrain)
            model=RidgeClassifierCV(alphas=(.1,1,10,100,1000)).fit(scaler.transform(ftrain),labels)
            end=perf_counter()
            calibration_times.append(1000*(calibrated-start))
            feature_times.append(1000*(extracted-calibrated))
            class_times.append(1000*(end-extracted));total_times.append(1000*(end-start))
            del ftrain
        extraction=[];inference=[]
        for repeat in range(REPEATS):
            gc.collect();start=perf_counter();fbatch=transform(batch);end_features=perf_counter()
            model.predict(scaler.transform(fbatch));end=perf_counter()
            extraction.append(1000*(end_features-start));inference.append(1000*(end-start))
        predictions=[]; errors=[];error_square=[];error_max=[];crossings_bound=[]
        for index,view in enumerate(views):
            matrix=transform(np.ascontiguousarray(view))
            predictions.append(model.predict(scaler.transform(matrix)))
            # Audit a fixed batch of 64 objects at every random phase.
            if method!='original' and index>0:
                exact=features(np.ascontiguousarray(view[:BATCH]),*reference,0,False)
                delta=np.asarray(matrix[:BATCH],dtype=np.float64)-exact
                errors.append(float(abs(delta).mean()));error_square.append(float((delta**2).mean()))
                error_max.append(float(abs(delta).max()))
                if method.startswith('phase'):
                    k=int(method[5:]);assert abs(delta).max()<=1/k+2e-7
            del matrix
        predictions=np.array(predictions)
        correct=predictions==truth
        accuracy=correct[1:].mean(axis=1)
        stable=(predictions==predictions[0]).all(axis=0)
        flipped=(predictions[1:]!=predictions[0]).mean(axis=0)
        memory=psutil.Process().memory_info()
        np.savez_compressed(OUT/f'predictions_{case}_{method}.npz',predictions=predictions,
                            truth=truth,object_ids=ids,phases=phases)
        result=dict(case=case,method=method,model_seed=MODEL_SEED,features=84*int(c.sum()),samples=int(train.shape[1]),
                    train=len(train),test=len(truth),phases=phases.tolist(),threads=2,blas_threads=1,
                    repeats=REPEATS,batch=BATCH,accuracy=float(accuracy.mean()),
                    accuracy_sd=float(accuracy.std(ddof=1)),worst_phase_accuracy=float(accuracy.min()),
                    phase_accuracy=accuracy.tolist(),clean_accuracy=float(correct[0].mean()),
                    flip_rate=float(flipped.mean()),all_phase_consistency=float(stable.mean()),
                    paired_object_correct=correct[1:].mean(axis=0).tolist(),
                    calibration_ms=stats(calibration_times),train_feature_ms=stats(feature_times),
                    classifier_ms=stats(class_times),total_train_ms=stats(total_times),
                    batch_feature_ms=stats(extraction),batch_inference_ms=stats(inference),
                    peak_worker_rss_mib=float(getattr(memory,'peak_wset',memory.rss)/(1024**2)),
                    train_feature_storage_mib=float(len(train)*84*c.sum()*4/(1024**2)),
                    batch_feature_storage_mib=float(BATCH*84*c.sum()*4/(1024**2)),
                    feature_mae=None if not errors else float(np.mean(errors)),
                    feature_rmse=None if not errors else float(np.sqrt(np.mean(error_square))),
                    feature_max_error=None if not errors else max(error_max),
                    feature_error_objects=BATCH,feature_error_phases=8,
                    selected_alpha=float(model.alpha_),platform=platform.platform(),
                    cpu=os.environ.get('PROCESSOR_IDENTIFIER','unknown'))
        print(json.dumps(result),flush=True)


def run():
    path=OUT/'cost_stability_v1.jsonl'
    if path.exists():
        raise FileExistsError('Set EIPPV_OUTPUT_DIR to a new directory; raw results are never overwritten')
    environment=os.environ.copy()
    environment['NUMBA_NUM_THREADS']='2'
    environment['OMP_NUM_THREADS']='1'
    environment['MKL_NUM_THREADS']='1'
    environment['OPENBLAS_NUM_THREADS']='1'
    for case in ('mpeg7_32','mpeg7_64','hand_32','hand_64'):
        for method in METHODS:
            result=subprocess.run([sys.executable,'-B',__file__,'worker','--case',case,'--method',method],
                                  capture_output=True,text=True,env=environment,check=True)
            row=json.loads(result.stdout.strip())
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
            print(case,method,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker'))
    parser.add_argument('--case');parser.add_argument('--method',choices=METHODS)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:worker(args.case,args.method)
