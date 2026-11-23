"""UCR pilot for rotation-invariant signed lag pooling on MiniRocket kernels."""
import argparse
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent / "rocket_pilot_data"
METHODS = ("ppv", "global", "symmetric", "directional", "both")


def load(name):
    with ZipFile(ROOT/f"{name}.zip") as archive:
        result = []
        for part in ("TRAIN", "TEST"):
            member = next(x for x in archive.namelist() if x.endswith(f"{name}_{part}.ts"))
            lines = archive.read(member).decode("utf-8-sig").splitlines()
            start = next(i for i,s in enumerate(lines) if s.strip().lower()=="@data")
            rows = [s.rsplit(":", 1) for s in lines[start+1:] if s.strip()]
            x = np.stack([np.fromstring(s, sep=",", dtype=np.float32) for s,_ in rows])
            y = np.array([label.strip() for _,label in rows])
            assert np.isfinite(x).all()
            x = (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-5)
            result.extend((x,y))
    return result


def pooling(train, test, seed):
    a = np.vstack((train,test))
    base, plus, symmetric, directional = [],[],[],[]
    rng = np.random.default_rng(seed)
    lags = [1,4,8]
    for derivative in (False,True):
        x = (a - np.roll(a,1,axis=1)) if derivative else a
        for dilation in (1,2,4,8):
            if dilation*8 >= x.shape[1]:
                continue
            rolled = [np.roll(x,i*dilation,axis=1) for i in range(9)]
            summed = np.add.reduce(rolled)
            for indices in itertools.combinations(range(9),3):
                values = 3*(rolled[indices[0]]+rolled[indices[1]]+rolled[indices[2]])-summed
                picked = rng.choice(len(train), min(len(train),16), replace=False)
                low, high = np.quantile(values[picked],[.25,.75])
                pos = values > high
                neg = values < low
                count = pos.sum(1)
                base.append(count/values.shape[1])
                plus.extend((np.maximum(values-high,0).sum(1)/np.maximum(count,1),values.max(1)))
                for lag in lags:
                    forward = (pos & np.roll(neg,-lag,axis=1)).mean(1)
                    backward = (neg & np.roll(pos,-lag,axis=1)).mean(1)
                    symmetric.append(forward+backward)
                    directional.append(forward-backward)
    ppv=np.column_stack(base)
    glob=np.column_stack(base+plus)
    return dict(ppv=ppv, global_=glob, symmetric=np.c_[glob,np.column_stack(symmetric)],
                directional=np.c_[glob,np.column_stack(directional)],
                both=np.c_[glob,np.column_stack(symmetric+directional)])


def evaluate(name, seed):
    a, y, b, truth = load(name)
    features = pooling(a,b,seed)
    for method, matrix in features.items():
        scaler=StandardScaler().fit(matrix[:len(a)])
        model=RidgeClassifierCV(alphas=np.logspace(-1,4,10)).fit(scaler.transform(matrix[:len(a)]),y)
        pred=model.predict(scaler.transform(matrix[len(a):]))
        print(json.dumps(dict(data=name,seed=seed,method=method,n=len(b),features=matrix.shape[1],
                              accuracy=float(np.mean(pred==truth)),
                              balanced=float(balanced_accuracy_score(truth,pred)))),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",nargs="+")
    p.add_argument("--seed",type=int,default=23)
    args=p.parse_args()
    with threadpool_limits(limits=1):
        for name in args.data:
            evaluate(name,args.seed)
