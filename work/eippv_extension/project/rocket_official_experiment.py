"""Matched MiniRocket experiment using the pinned author implementation."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from minirocket_reference import minirocket as original
from rocket_duration_pool import resample
from rocket_directional_pool import load

DATA = ("Adiac", "SwedishLeaf", "Fish", "OSULeaf", "ShapesAll", "BeetleFly",
        "BirdChicken", "DiatomSizeReduction", "ArrowHead", "Car", "Herring")
KERNELS = np.array(list(itertools.combinations(range(9), 3)), dtype=np.int32)


@njit
def seed_rng(seed):
    np.random.seed(seed)


@njit
def response_parts(x, dilation, circular):
    alpha = -x.copy()
    triple = x+x+x
    gamma = np.zeros((9, len(x)), dtype=np.float32)
    gamma[4] = triple
    for tap in range(9):
        if tap == 4:
            continue
        shift = (4-tap)*dilation
        for i in range(len(x)):
            src = i-shift
            if circular:
                src %= len(x)
            if 0 <= src < len(x):
                alpha[i] -= x[src]
                gamma[tap,i] = triple[src]
    return alpha, gamma


@njit
def fit_biases(x, dilations, counts, quantiles, seed, circular):
    np.random.seed(seed)
    biases = np.empty(len(quantiles), dtype=np.float32)
    feature = 0
    for di in range(len(dilations)):
        for k in range(len(KERNELS)):
            sample = x[np.random.randint(len(x))]
            alpha, gamma = response_parts(sample, dilations[di], circular)
            i,j,k3 = KERNELS[k]
            v = alpha+gamma[i]+gamma[j]+gamma[k3]
            end = feature+counts[di]
            biases[feature:end] = np.quantile(v,quantiles[feature:end])
            feature = end
    return biases


@njit(parallel=True)
def pool_features(x, dilations, counts, biases):
    features = np.empty((3,len(x),len(biases)),dtype=np.float32)
    for example in prange(len(x)):
        feature = 0
        for di in range(len(dilations)):
            alpha, gamma = response_parts(x[example],dilations[di],True)
            for kernel in KERNELS:
                i,j,k = kernel
                v = alpha+gamma[i]+gamma[j]+gamma[k]
                for _ in range(counts[di]):
                    bias = biases[feature]
                    hard, sampled, duration = 0,0,0.
                    for t in range(len(v)):
                        a = float(v[t])-float(bias)
                        b = float(v[(t+1)%len(v)])-float(bias)
                        hard += a > 0
                        sampled += (a > 0) + (3*a+b > 0) + (a+b > 0) + (a+3*b > 0)
                        low, high = min(a,b),max(a,b)
                        if low > 0:
                            duration += 1
                        elif high > 0:
                            duration += high/(high-low)
                    features[0,example,feature] = hard/len(v)
                    features[1,example,feature] = sampled/(4*len(v))
                    features[2,example,feature] = duration/len(v)
                    feature += 1
    return features


def evaluate(name, factor, budget, seed, exclude_duplicates=False):
    raw, y, test, truth = load(name)
    if exclude_duplicates:
        training_rows = {row.tobytes() for row in raw}
        keep = np.array([row.tobytes() not in training_rows for row in test])
        test, truth = test[keep], truth[keep]
    n = max(24,test.shape[1]//factor)
    train = resample(raw,n,0.)
    x = np.vstack([train]+[resample(test,n,p) for p in (0.,.25,.5,.75)])
    seed_rng(seed)
    dilations,counts,biases = original.fit(train,num_features=budget)
    # Confirm the adapter consumes the same sequence of training draws
    # and uses the same convolution before changing the boundary.
    q = original._quantiles(len(biases))
    reproduced = fit_biases(train,dilations,counts,q,seed,False)
    np.testing.assert_allclose(reproduced,biases,rtol=2e-5,atol=2e-5)
    circular = fit_biases(train,dilations,counts,q,seed,True)
    cyclic = pool_features(x,dilations,counts,circular)
    rotated = np.ascontiguousarray(np.roll(x[:2],5,axis=1))
    np.testing.assert_allclose(pool_features(rotated,dilations,counts,circular),
                               cyclic[:,:2],rtol=1e-6,atol=1e-7)
    features = [original.transform(x,(dilations,counts,biases)),*cyclic]
    for name_method, matrix in zip(("original","circular","sample4","duration"),features):
        scaler = StandardScaler().fit(matrix[:len(train)])
        clf = RidgeClassifierCV(alphas=(.1,1,10,100,1000))
        clf.fit(scaler.transform(matrix[:len(train)]),y)
        pred = clf.predict(scaler.transform(matrix[len(train):])).reshape(4,len(test))
        print(json.dumps(dict(data=name, factor=factor, budget=budget, seed=seed, n=len(test),
                              samples=n, features=len(biases), method=name_method,
                              original=float((pred[0]==truth).mean()),
                              shifted=float((pred[1:]==truth).mean()),
                              flip=float((pred[1:]!=pred[0]).mean()),
                              commit=Path(original.__file__).with_name("commit.txt").read_text().strip())),
              flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",nargs="*",default=DATA)
    p.add_argument("--factor",type=int,choices=(2,4,8),default=4)
    p.add_argument("--budget",type=int,choices=(756,10000),default=756)
    p.add_argument("--seed",type=int,default=23)
    p.add_argument("--exclude-test-duplicates",action="store_true")
    args=p.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=1):
        for dataset in args.data:
            evaluate(dataset,args.factor,args.budget,args.seed,args.exclude_test_duplicates)
