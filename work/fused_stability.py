"""Compute four pre-specified poolings together, retaining original arithmetic."""
from modern_extension import *
from numba import njit,prange
from rocket_official_experiment import KERNELS,response_parts
from rocket_mpeg7_confirm import interval

@njit(parallel=True,cache=True)
def fused(x,dilations,counts,biases):
    out=np.empty((4,len(x),len(biases)),dtype=np.float32)
    for example in prange(len(x)):
        offset=0
        for di in range(len(dilations)):
            alpha,gamma=response_parts(x[example],dilations[di],True)
            for i,j,k in KERNELS:
                v=alpha+gamma[i]+gamma[j]+gamma[k]
                for _ in range(counts[di]):
                    b=np.float64(biases[offset]);s0=0.;s16=0.;s32=0.;se=0.
                    for t in range(len(v)):
                        a=np.float64(v[t])-b;c=np.float64(v[(t+1)%len(v)])-b
                        s0+=interval(a,c,1,False);s16+=interval(a,c,16,True)
                        s32+=interval(a,c,32,True);se+=interval(a,c,0,False)
                    out[0,example,offset]=s0/len(v);out[1,example,offset]=s16/len(v)
                    out[2,example,offset]=s32/len(v);out[3,example,offset]=se/len(v)
                    offset+=1
    return out

METHODS=('circular','phase16','phase32','eippv')
def audit():
    set_num_threads(2);rows=[]
    for case in CASES:
        d,_=input_data(case,101);p=params(d['train'],'eippv',23,9996)
        x=np.ascontiguousarray(d['views'][2,:5]);ff=fused(x,*p)
        for i,m in enumerate(METHODS):
            ref=transform(x,p,m);np.testing.assert_array_equal(ff[i],ref)
            rows.append(dict(case=case,method=m,values=ref.size,bitwise_identical=True))
    dump(OUT/'fused_transform_audit.json',rows)
    print('FUSED32 CHECKS PASS',flush=True)

def joint(case,seed,split):
    start=time.perf_counter();set_num_threads(2);d,ip=input_data(case,split);x,y,g=d['train'],d['labels'],d['train_groups']
    if len(set(g))<len(g):folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261116).split(x,y,g))
    else:folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=20261116).split(x,y))
    correct=np.zeros((4,len(ALPHAS)));cv=[[] for _ in METHODS]
    with threadpool_limits(limits=1):
        for fold,(tr,va) in enumerate(folds):
            assert not set(g[tr])&set(g[va]);p=params(x[tr],'eippv',seed,9996)
            ft=fused(x[tr],*p);fv=fused(x[va],*p)
            for mi,m in enumerate(METHODS):
                a=ft[mi].astype(np.float64);b=fv[mi].astype(np.float64);sc=StandardScaler().fit(a);a=sc.transform(a);b=sc.transform(b)
                for ai,alpha in enumerate(ALPHAS):
                    mod=RidgeClassifier(alpha=alpha).fit(a,y[tr]);c=int((mod.predict(b)==y[va]).sum());correct[mi,ai]+=c
                    cv[mi].append(dict(fold=fold,alpha=alpha,correct=c,total=len(va)))
            del ft,fv
        p=params(x,'eippv',seed,9996);ft=fused(x,*p);models=[];scalers=[];selected=[]
        for mi,m in enumerate(METHODS):
            alpha=ALPHAS[int(np.argmax(correct[mi]))];a=ft[mi].astype(np.float64);sc=StandardScaler().fit(a);mod=RidgeClassifier(alpha=alpha).fit(sc.transform(a),y)
            models.append(mod);scalers.append(sc);selected.append(dict(alpha=alpha,feature_dim=a.shape[1],cv=cv[mi]))
        del ft
        predictions=[[] for _ in METHODS]
        for view in d['views']:
            fv=fused(view,*p)
            for mi in range(4):predictions[mi].append(models[mi].predict(scalers[mi].transform(fv[mi].astype(np.float64))))
    elapsed=time.perf_counter()-start
    for mi,m in enumerate(METHODS):
        pred=np.array(predictions[mi]);acc=(pred[1:]==d['truth']).mean(axis=0);flip=(pred[1:]!=pred[0]).mean(axis=0)
        stem=f'{case}_{m}_s{seed}_split{split}';path=OUT/'raw'/stem
        np.savez_compressed(path.with_suffix('.npz'),predictions=pred,truth=d['truth'],object_ids=d['object_ids'],phases=d['phases'])
        row=dict(case=case,recipe=m,seed=seed,split=split,accuracy_pct=100*float(acc.mean()),flip_pct=100*float(flip.mean()),objects=len(acc),input_sha256=sha(ip),selected=selected[mi],seconds=elapsed,execution='four-method shared-transform batch; seconds is joint total, not individual cost')
        dump(path.with_suffix('.json'),row)
    print(json.dumps(dict(case=case,split=split,joint_methods=4,seconds=elapsed)),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='audit':audit()
    else:joint(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
