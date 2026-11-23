"""Independent MPEG-7 contour confirmation for MiniRocket pooling."""
import argparse
from collections import Counter
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit, prange, set_num_threads
from PIL import Image
from skimage.measure import find_contours
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from rocket_official_experiment import KERNELS, fit_biases, original, response_parts, seed_rng
from rocket_directional_pool import load

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "rocket_pilot_data" / "mpeg7_shape_b.zip"
SHA256 = "331b197caf51b0a3cfd76852aeec46b1fb839fa9374b2a7302754cb92fe0680c"
PUBLIC_SHA256 = "ce8d87409d1b87b07352ebf6e0eafb8884bea6e4027cb7a23c228e7a3685bd0d"
METHODS = (("circular",1,False),("sample4",4,False),("sample8",8,False),
           ("sample16",16,False),("count16",16,True),("duration",0,False))


def contours(archive_path=None):
    archive_path = ARCHIVE if archive_path is None else Path(archive_path)
    with archive_path.open("rb") as stream:
        assert file_digest(stream,"sha256").hexdigest() in {SHA256,PUBLIC_SHA256}
    radii, labels = [], []
    with ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            stem = Path(member).stem
            name, sep, index = stem.rpartition("-")
            if not sep or not index.isdigit() or not member.lower().endswith(".gif"):
                continue
            with Image.open(BytesIO(archive.read(member))) as image:
                pixels = np.asarray(image.convert("L"))
            border = np.r_[pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]]
            light = pixels > 127
            foreground = ~light if np.median(border) > 127 else light
            boundaries = find_contours(np.pad(foreground,1), .5)
            outline = max(boundaries,key=lambda xy:abs(np.dot(xy[:-1,0],xy[1:,1])
                                                       -np.dot(xy[:-1,1],xy[1:,0])))
            assert np.array_equal(outline[0],outline[-1])
            length = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(outline,axis=0),axis=1))]
            uniform = np.linspace(0,length[-1],512,endpoint=False)
            xy = np.column_stack([np.interp(uniform,length,outline[:,axis]) for axis in (0,1)])
            radii.append(np.linalg.norm(xy-xy.mean(axis=0),axis=1))
            labels.append(name)
    assert len(labels) == 1400 and Counter(Counter(labels).values()) == {20:70}
    return np.array(radii),np.array(labels)


def resample(x,n,phase,acquisition="linear"):
    phase = np.broadcast_to(phase,(len(x),))
    grid = ((np.arange(n)+phase[:,None])*x.shape[1]/n)%x.shape[1]
    start = grid.astype(int)
    portion = grid-start
    values = (1-portion)*np.take_along_axis(x,start,axis=1)
    values += portion*np.take_along_axis(x,(start+1)%x.shape[1],axis=1)
    if acquisition == "cubic":
        curve = CubicSpline(np.arange(x.shape[1]+1),np.c_[x,x[:,0]],axis=1,bc_type="periodic")
        c = curve.c[:,start,np.arange(len(x))[:,None]]
        values = ((c[0]*portion+c[1])*portion+c[2])*portion+c[3]
    return np.ascontiguousarray((values-values.mean(1,keepdims=True)) /
                                np.maximum(values.std(1,keepdims=True),1e-5),dtype=np.float32)


@njit
def interval(a,b,steps,fast):
    if steps == 1:
        return 1. if a > 0 else 0.
    low, high = min(a,b),max(a,b)
    if low > 0:
        return 1.
    if high <= 0:
        return 0.
    if steps == 0:
        return high/(high-low)
    if fast:
        if a > 0:
            return np.ceil(steps*a/(a-b))/steps
        return (steps-1-np.floor(steps*(-a)/(b-a)))/steps
    count = 0
    for j in range(steps):
        count += (steps-j)*a+j*b > 0
    return count/steps


@njit
def fit_blur_biases(x,dilations,counts,quantiles,seed):
    np.random.seed(seed)
    biases = np.empty(len(quantiles),dtype=np.float32)
    offset = 0
    for di in range(len(dilations)):
        for i,j,k in KERNELS:
            sample = x[np.random.randint(len(x))]
            alpha,gamma = response_parts(sample,dilations[di],True)
            v = alpha+gamma[i]+gamma[j]+gamma[k]
            v = (np.roll(v,1)+2*v+np.roll(v,-1))/4
            end = offset+counts[di]
            biases[offset:end] = np.quantile(v,quantiles[offset:end])
            offset = end
    return biases


@njit(parallel=True)
def features(x,dilations,counts,biases,steps,fast):
    result = np.empty((len(x),len(biases)),dtype=np.float32)
    for example in prange(len(x)):
        offset = 0
        for di in range(len(dilations)):
            alpha,gamma = response_parts(x[example],dilations[di],True)
            for i,j,k in KERNELS:
                v = alpha+gamma[i]+gamma[j]+gamma[k]
                if steps == -1:
                    v = (np.roll(v,1)+2*v+np.roll(v,-1))/4
                for _ in range(counts[di]):
                    b = np.float64(biases[offset])
                    total = 0.
                    for t in range(len(v)):
                        total += interval(np.float64(v[t])-b,np.float64(v[(t+1)%len(v)])-b,
                                          1 if steps == -1 else steps,fast)
                    result[example,offset] = total/len(v)
                    offset += 1
    return result


def score(matrix,y,truth):
    scale = StandardScaler().fit(matrix[:len(y)])
    model = RidgeClassifierCV(alphas=(.1,1,10,100,1000)).fit(scale.transform(matrix[:len(y)]),y)
    pred = model.predict(scale.transform(matrix[len(y):])).reshape(4,len(truth))
    return pred[0] == truth, pred[1:] == truth, pred[1:] != pred[0]


def split_shapes(raw, labels, allow_duplicates=False, split_seed=23):
    tr,te = train_test_split(np.arange(len(labels)),test_size=.5,stratify=labels,random_state=split_seed)
    if allow_duplicates:
        return tr,te
    keys = [row.tobytes() for row in raw]
    frequency = Counter(keys)
    for test_slot,index in enumerate(te):
        if keys[index] in {keys[i] for i in tr}:
            train_slot = next(slot for slot,other in enumerate(tr)
                              if labels[other] == labels[index] and frequency[keys[other]] == 1)
            tr[train_slot],te[test_slot] = index,tr[train_slot]
    assert not {keys[i] for i in tr} & {keys[i] for i in te}
    return tr,te


def evaluate(raw,labels,n,budget,seed,allow_duplicates=False,archive_split=None,
             split_seed=23,phase_seed=20261107,acquisition="linear",timing_repeats=5,
             blur_only=False):
    name = "MPEG7"
    if archive_split is None:
        tr,te = split_shapes(raw,labels,allow_duplicates,split_seed)
        train_raw,train_y,test_raw,test_y = raw[tr],labels[tr],raw[te],labels[te]
    else:
        name = "HandOutlines"
        test_raw,test_y = archive_split
        training_rows = {row.tobytes() for row in raw}
        keep = np.array([row.tobytes() not in training_rows for row in test_raw])
        train_raw,train_y,test_raw,test_y = raw,labels,test_raw[keep],test_y[keep]
    train = resample(train_raw,n,0.,acquisition)
    rng = np.random.default_rng(phase_seed)
    phases = rng.uniform(0.,1.,size=(3,len(test_y)))
    views = [resample(test_raw,n,p,acquisition) for p in (0.,*phases)]
    x = np.vstack([train,*views])
    seed_rng(seed)
    dilations,counts,baseline = original.fit(train,num_features=budget)
    q = original._quantiles(len(baseline))
    np.testing.assert_allclose(fit_biases(train,dilations,counts,q,seed,False),baseline,rtol=2e-5,atol=2e-5)
    cyclic = fit_biases(train,dilations,counts,q,seed,True)
    batch = views[1][:64]
    transforms = {"original":lambda v:original.transform(v,(dilations,counts,baseline))}
    selected = (("duration",0,False),) if blur_only else METHODS
    for method,steps,fast in selected:
        transforms[method] = lambda v,s=steps,f=fast:features(v,dilations,counts,cyclic,s,f)
    if blur_only:
        blur = fit_blur_biases(train,dilations,counts,q,seed)
        transforms["blur"] = lambda v:features(v,dilations,counts,blur,-1,False)
    for transform in transforms.values():
        transform(batch)
    timings = {method:[] for method in transforms}
    for repeat in range(timing_repeats):
        for method in rng.permutation(list(transforms)):
            t = perf_counter()
            transforms[method](batch)
            timings[method].append(1000*(perf_counter()-t))
    direct16 = None
    for method,transform in transforms.items():
        matrix = transform(x)
        if method == "sample16":
            direct16 = matrix.copy()
        if method == "count16":
            np.testing.assert_allclose(direct16,matrix,atol=1e-7,rtol=1e-7)
            del direct16
        original_ok, shift_ok, flipped = score(matrix,train_y,test_y)
        print(json.dumps(dict(data=name,method=method,samples=n,features=len(cyclic),seed=seed,
                              train=len(train_y),test=len(test_y),phase_views=3,timing_batch=len(batch),
                              feature_ms=median(timings[method]) if timings[method] else None,
                              original=float(original_ok.mean()),
                              shifted=float(shift_ok.mean()),flip=float(flipped.mean()),
                              split_seed=split_seed,phase_seed=phase_seed,acquisition=acquisition,
                              per_shape_shift=shift_ok.mean(axis=0).tolist())),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples",type=int,choices=(32,64),required=True)
    p.add_argument("--budget",type=int,choices=(756,10000),default=756)
    p.add_argument("--seed",type=int,default=23)
    p.add_argument("--legacy-split",action="store_true",help="Replay the original split before duplicate audit")
    p.add_argument("--hand",action="store_true",help="Use the separate HandOutlines archive")
    p.add_argument("--split-seed",type=int,default=23)
    p.add_argument("--phase-seed",type=int,default=20261107)
    p.add_argument("--acquisition",choices=("linear","cubic"),default="linear")
    p.add_argument("--no-timing",action="store_true")
    p.add_argument("--blur-only",action="store_true",help="Compare refitted binomial smoothing against duration")
    a = p.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=1):
        options = dict(split_seed=a.split_seed,phase_seed=a.phase_seed,
                       acquisition=a.acquisition,timing_repeats=0 if a.no_timing else 5,
                       blur_only=a.blur_only)
        if a.hand:
            train,y,test,truth = load("HandOutlines")
            evaluate(train,y,a.samples,a.budget,a.seed,archive_split=(test,truth),**options)
        else:
            evaluate(*contours(),a.samples,a.budget,a.seed,a.legacy_split,**options)
