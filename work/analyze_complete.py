"""Shared paired-inference routines for the released analyses."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ci(diff,truth,rng):
    values=[]; groups=[np.flatnonzero(truth==x) for x in np.unique(truth)]
    for _ in range(40):
        estimates=np.zeros(250)
        for ids in groups:
            estimates+=diff[rng.choice(ids,(250,len(ids)),replace=True)].sum(axis=1)
        values.extend(estimates/len(truth))
    return np.quantile(values,[.025,.975])*100

def pvalue(diff,rng):
    count=0;obs=abs(diff.mean())
    for _ in range(100):
        signs=rng.integers(0,2,(1000,len(diff)))*2-1
        count+=int((abs((signs*diff).mean(axis=1))>=obs-1e-15).sum())
    return (count+1)/100001

