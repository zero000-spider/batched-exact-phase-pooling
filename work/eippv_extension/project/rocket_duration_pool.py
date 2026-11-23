"""Threshold-exceedance duration pooling for resampled closed contours."""
import argparse
import itertools
import json
import numpy as np
from scipy.special import expit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from rocket_directional_pool import load


def resample(x, n, phase):
    coordinate = (np.arange(n)+phase)*x.shape[1]/n
    left = np.floor(coordinate).astype(int)
    fraction = coordinate-left
    observed = x[:,left%x.shape[1]]*(1-fraction)+x[:,(left+1)%x.shape[1]]*fraction
    return ((observed-observed.mean(1,keepdims=True)) /
            np.maximum(observed.std(1,keepdims=True),1e-5)).astype(np.float32)


def duration_pool(left, right):
    """Integrate the positive fraction of each linear response segment."""
    low = np.minimum(left,right)
    high = np.maximum(left,right)
    crossing = (low <= 0) & (high > 0)
    fraction = np.divide(high, high-low, out=np.zeros_like(high),where=crossing)
    return ((low>0)+fraction).mean(axis=1)


def transform(train, views, seed):
    data=np.vstack((train,*views))
    rng=np.random.default_rng(seed)
    methods={key:[] for key in ("hard","sigmoid_narrow","sigmoid","sigmoid_wide","sample2","sample4","duration")}
    for d in np.unique(np.geomspace(1,(data.shape[1]-1)/8,3).astype(int)):
        rolled=[np.roll(data,(i-4)*d,axis=1) for i in range(9)]
        total=np.add.reduce(rolled)
        for i,j,k in itertools.combinations(range(9),3):
            v=3*(rolled[i]+rolled[j]+rolled[k])-total
            sample=rng.choice(len(train),min(16,len(train)),replace=False)
            quantiles=np.quantile(v[sample],[.2,.5,.8])
            temperature=max(float((quantiles[-1]-quantiles[0])*.1),1e-6)
            next_v=np.roll(v,-1,axis=1)
            for b in quantiles:
                a,z=v-b,next_v-b
                methods["hard"].append((a>0).mean(1))
                methods["sigmoid_narrow"].append(expit(a/(.5*temperature)).mean(1))
                methods["sigmoid"].append(expit(a/temperature).mean(1))
                methods["sigmoid_wide"].append(expit(a/(2*temperature)).mean(1))
                methods["sample2"].append(.5*((a>0).mean(1)+((a+z)>0).mean(1)))
                methods["sample4"].append(.25*((a>0).mean(1)+((3*a+z)>0).mean(1)
                                               +((a+z)>0).mean(1)+((a+3*z)>0).mean(1)))
                methods["duration"].append(duration_pool(a,z))
    return {key:np.column_stack(cols) for key,cols in methods.items()}


def main(name,seed,factor):
    train,y,test,truth=load(name)
    n=max(24,test.shape[1]//factor)
    phases=(0.,.25,.5,.75)
    a=resample(train,n,0.)
    views=[resample(test,n,phase) for phase in phases]
    all_features=transform(a,views,seed)
    offset=len(a)
    for key, features in all_features.items():
        x=features[:offset]
        ts=features[offset:].reshape(len(phases),len(test),-1)
        scaler=StandardScaler().fit(x)
        model=RidgeClassifierCV(alphas=(.1,1,10,100,1000)).fit(scaler.transform(x),y)
        pred=np.array([model.predict(scaler.transform(view)) for view in ts])
        sensitivity=np.abs(ts[1:]-ts[0]).mean()
        scaled_sensitivity=np.abs(scaler.transform(ts[1:].reshape(-1,x.shape[1])).reshape(ts[1:].shape)
                                 -scaler.transform(ts[0])).mean()
        print(json.dumps(dict(data=name,seed=seed,factor=factor,samples=n,method=key,n=len(test),
                              original=float((pred[0]==truth).mean()),
                              shifted=float((pred[1:]==truth).mean()),
                              flip=float((pred[1:]!=pred[0]).mean()),
                              distance=float(sensitivity),scaled_distance=float(scaled_sensitivity))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",nargs="+")
    p.add_argument("--seed",type=int,default=23)
    p.add_argument("--factor",type=int,choices=(2,4,8),default=4)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        for name in a.data:
            main(name,a.seed,a.factor)
