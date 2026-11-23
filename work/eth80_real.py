"""Frozen object-disjoint real multiview stress test; no synthetic phase views."""
import os
os.environ.update(NUMBA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from complete_experiments import ROOT,HERE,sha
import sys,json,tarfile,re,io,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
sys.path.insert(0,str(ROOT))
from rocket_mpeg7_confirm import resample,features
from rocket_official_experiment import original,fit_biases,seed_rng
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from numba import set_num_threads

GROUPED=os.environ.get('ETH80_GROUPED')=='1'
FLOAT64=os.environ.get('ETH80_FLOAT64')=='1'
OUT=HERE.parent/('outputs/eth80_multiview' if FLOAT64 else 'outputs/eth80_grouped_diagnostic' if GROUPED else 'outputs/eth80_legacy_diagnostic');OUT.mkdir(parents=True,exist_ok=True)
ARCHIVE=HERE/'jena/eth80-contours.tgz'
TRAIN_VIEWS=('000-000','045-000','045-090','045-180','045-270')
METHODS=('original','circular','phase16','phase32','eippv')
protocol=dict(dataset='ETH80 official contour release',archive_sha256=sha(ARCHIVE),source='https://datasets.d2.mpi-inf.mpg.de/eth80/eth80-contours.tgz',split_seed=23,train_objects_per_class=5,test_objects_per_class=5,train_views=TRAIN_VIEWS,test_views='all 41 real camera views',samples=[32,64],model_seeds=[23,37,71],methods=METHODS,features=9996,alphas=[.1,1,10,100,1000],primary='EI minus circular: object mean accuracy and anchor-view disagreement at both lengths; Holm family 4',anchor='000-000',scope='Real camera viewpoint changes, not repeated identical-pose capture or pure phase shifts. Provider contours used; no evaluation of foreground segmentation.',exclusion='Ignore .xvpics thumbnails; require all 80 objects x41 valid closed outer contours; stop on extraction failure, never select based on performance.')
protocol['alpha_selection']='5-fold StratifiedGroupKFold by training physical object; StandardScaler fitted separately inside each fold' if GROUPED else 'legacy RidgeClassifierCV; training views not grouped'
protocol['classifier_dtype']='float64' if FLOAT64 else 'float32'
if not (OUT/'protocol.json').exists():(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
cache=OUT/'contours.npz'
if GROUPED and not cache.exists() and (HERE.parent/'outputs/eth80_legacy_diagnostic/contours.npz').exists():
 import shutil
 shutil.copy2(HERE.parent/'outputs/eth80_legacy_diagnostic/contours.npz',cache)
if not cache.exists():
    rows=[];pat=re.compile(r'/([a-z]+[0-9]+)/\1-([0-9]{3}-[0-9]{3})-contour.png$')
    with tarfile.open(ARCHIVE) as tar:
        members=[(m,pat.search(m.name)) for m in tar.getmembers() if m.isfile() and pat.search(m.name)]
        for m,match in members:
            im=np.asarray(Image.open(io.BytesIO(tar.extractfile(m).read())).convert('L'))
            filled=binary_fill_holes(im<128)
            cs=find_contours(np.pad(filled,1),.5)
            c=max(cs,key=lambda c:abs(np.dot(c[:-1,0],c[1:,1])-np.dot(c[:-1,1],c[1:,0])))
            assert np.array_equal(c[0],c[-1]),m.name
            length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))]
            xy=np.column_stack([np.interp(np.linspace(0,length[-1],512,endpoint=False),length,c[:,i]) for i in (0,1)])
            radius=np.linalg.norm(xy-xy.mean(axis=0),axis=1)
            rows.append((match[1],match[2],radius))
    rows.sort(key=lambda r:(r[0],r[1]));assert len(rows)==3280
    np.savez_compressed(cache,objects=np.array([r[0] for r in rows]),views=np.array([r[1] for r in rows]),radii=np.array([r[2] for r in rows]))
z=np.load(cache);objects=z['objects'];views=z['views'];raw=z['radii']
classes=sorted({re.sub(r'\d+$','',o) for o in objects});rng=np.random.default_rng(23);tr=[];te=[]
for label in classes:
    ids=sorted({o for o in objects if re.sub(r'\d+$','',o)==label},key=lambda o:int(re.search(r'\d+$',o)[0]))
    assert len(ids)==10
    shuffled=rng.permutation(ids);tr.extend(sorted(shuffled[:5]));te.extend(sorted(shuffled[5:]))
assert not set(tr)&set(te);assert all((objects==o).sum()==41 for o in tr+te)
train_index=np.array([i for i,(o,v) in enumerate(zip(objects,views)) if o in tr and v in TRAIN_VIEWS]);test_index=np.concatenate([np.flatnonzero(objects==o) for o in te]);assert len(train_index)==200 and len(test_index)==1640
labels=np.array([re.sub(r'\d+$','',o) for o in objects]);truth=labels[test_index].reshape(40,41)
assert all(views[test_index].reshape(40,41)[:,0]=='000-000')
(OUT/'split.json').write_text(json.dumps(dict(train_objects=tr,test_objects=te,train_indices=train_index.tolist(),test_indices=test_index.tolist(),all_view_ids=views[test_index].reshape(40,41)[0].tolist()),indent=2),encoding='utf-8')
set_num_threads(2)
with threadpool_limits(limits=1):
 for n in (32,64):
  train=resample(raw[train_index],n,0.);test=resample(raw[test_index],n,0.)
  for seed in (23,37,71):
   for method in METHODS:
    record=OUT/f'n{n}_seed{seed}_{method}.json';prediction=record.with_suffix('.npz')
    if record.exists():continue
    start=time.monotonic();seed_rng(seed)
    if method=='original':
     params=original.fit(train,num_features=10000)
     def transform(x):return original.transform(x,params)
    else:
     d,c=original._fit_dilations(n,10000,32);q=original._quantiles(84*int(c.sum()));b=fit_biases(train,d,c,q,seed,True)
     step={'circular':1,'phase16':16,'phase32':32,'eippv':0}[method]
     def transform(x):return features(x,d,c,b,step,step>1)
    f=transform(train)
    if FLOAT64:f=f.astype(np.float64)
    if GROUPED:
     cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=23)
     folds=list(cv.split(f,labels[train_index],objects[train_index]))
     for aa,bb in folds:assert not set(objects[train_index][aa])&set(objects[train_index][bb])
     model=GridSearchCV(make_pipeline(StandardScaler(),RidgeClassifier()),{'ridgeclassifier__alpha':(.1,1,10,100,1000)},cv=folds,n_jobs=1,scoring='accuracy').fit(f,labels[train_index])
     ftest=transform(test)
     if FLOAT64:ftest=ftest.astype(np.float64)
     pred=model.predict(ftest).reshape(40,41);selected_alpha=model.best_params_['ridgeclassifier__alpha']
    else:
     scale=StandardScaler().fit(f);model=RidgeClassifierCV(alphas=(.1,1,10,100,1000)).fit(scale.transform(f),labels[train_index])
     pred=model.predict(scale.transform(transform(test))).reshape(40,41);selected_alpha=model.alpha_
    acc=(pred==truth).mean(axis=1);flip=(pred[:,1:]!=pred[:,:1]).mean(axis=1)
    np.savez_compressed(prediction,predictions=pred,truth=truth,object_ids=np.array(te),view_ids=views[test_index].reshape(40,41))
    row=dict(n=n,seed=seed,method=method,accuracy=float(acc.mean()),flip=float(flip.mean()),alpha=float(selected_alpha),seconds=time.monotonic()-start,train_objects=40,test_objects=40,train_images=200,test_images=1640)
    record.write_text(json.dumps(row,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
print('30 configurations COMPLETE',flush=True)
