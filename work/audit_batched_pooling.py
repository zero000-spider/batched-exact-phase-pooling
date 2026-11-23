from batched_pooling import *
from decimal import Decimal,localcontext
import itertools

DEST=HERE.parent/'outputs/shared_threshold_study'
def decimal_reference(v,bs):
    with localcontext() as ctx:
        ctx.prec=180
        vals=[Decimal.from_float(float(a)) for a in v];out=[]
        for threshold in bs:
            b=Decimal.from_float(float(threshold));s=Decimal(0)
            for i,a in enumerate(vals):
                c=vals[(i+1)%len(vals)];lo=min(a,c);hi=max(a,c)
                if b<lo:s+=1
                elif b>=hi:pass
                else:s+=(hi-b)/(hi-lo)
            out.append(float(s/len(vals)))
    return np.array(out)

def main():
    DEST.mkdir(exist_ok=True);rng=np.random.default_rng(2026111607);count=0;values=0;maxerr=0.;max_decimal=0.;finite_discrepancies=[];decimal_cases=0
    cases=[]
    for n in (3,4,9,32,64,128,256):
        for _ in range(100):cases.append(rng.normal(size=n).astype(np.float32))
        for scale in (0.,1.,1e-30,1e30):
            cases.extend([np.full(n,scale,np.float32),np.resize(np.array([-scale,0,0,scale],np.float32),n)])
        for small in (1e-5,1e-10,1e-20,1e-35):cases.append(np.resize(np.array([0,small,1,-1],np.float32),n))
        cases.append((1+np.finfo(np.float32).eps*rng.integers(-3,4,n)).astype(np.float32))
    for ci,v in enumerate(cases):
        bs=np.r_[v,np.nextafter(v,np.float32(-np.inf)),np.nextafter(v,np.float32(np.inf)),rng.normal(size=17)].astype(np.float32)
        order=np.argsort(bs,kind='stable').astype(np.int32);ref=direct_queries(v,bs);fast=sweep_queries(v,bs,order);cache=cached_endpoint_queries(v,bs,0)
        delta=max(float(abs(fast.astype(np.float64)-ref).max()),float(abs(cache.astype(np.float64)-ref).max()))
        if delta>2e-7:
            np.savez_compressed(DEST/f'audit_failure_{ci}.npz',v=v,biases=bs,reference=ref,fast=fast,cached=cache)
            raise AssertionError((ci,len(v),delta))
        maxerr=max(maxerr,delta);count+=1;values+=len(bs)
        if ci%41==0 or (float(np.max(v))-float(np.min(v))>0 and min(abs(float(v[i])-float(v[(i+1)%len(v)])) for i in range(len(v)) if v[i]!=v[(i+1)%len(v)])<1e-9):
            truth=decimal_reference(v,bs);de=float(abs(fast.astype(np.float64)-truth).max());max_decimal=max(max_decimal,de);assert de<=2e-7,(ci,de);decimal_cases+=1
        # Finite-K comparisons on well-scaled cases; log rather than hide any discrepancy.
        if ci%17==0:
            for k in (4,16,32,128):
                direct=np.array([np.mean([interval(float(v[i])-float(b),float(v[(i+1)%len(v)])-float(b),k,True) for i in range(len(v))]) for b in bs],np.float32)
                a=cached_endpoint_queries(v,bs,k);b=sorted_sample_queries(v,bs,order,k)
                if not np.array_equal(a,direct) or not np.array_equal(b,direct):finite_discrepancies.append(dict(case=ci,length=len(v),K=k,cached_max_difference=float(abs(a-direct).max()),sorted_max_difference=float(abs(b-direct).max())))
    result=dict(response_cases=count,threshold_queries=values,maximum_EI_feature_difference=maxerr,decimal_reference_cases=decimal_cases,maximum_decimal_difference=max_decimal,tolerance=2e-7,finite_K_discrepancies=finite_discrepancies)
    dump(DEST/'numerical_audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':set_num_threads(2);main()
