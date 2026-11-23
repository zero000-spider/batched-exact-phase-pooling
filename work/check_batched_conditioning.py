"""Follow up the asymmetric short-edge cancellation concern without tuning on labels."""
from batched_pooling import *
from audit_batched_pooling import decimal_reference
DEST=HERE.parent/'outputs/shared_threshold_study'
def main():
    rng=np.random.default_rng(2026111631);count=0;queries=0;maxdiff=0.;worst=None;dc=0;de=0.
    for n in (4,9,32,64,128,512):
        for exponent in (-10,-9.3,-9,-8.3,-8,-7,-5):
            for trial in range(10):
                v=rng.uniform(-2,3,n).astype(np.float32)
                for i in range(0,n-1,4):v[i]=0;v[i+1]=np.float32(10.**exponent*rng.uniform(.8,1.2))
                bs=np.r_[v,np.nextafter(v,np.float32(-np.inf)),np.nextafter(v,np.float32(np.inf)),rng.uniform(-2,3,80)].astype(np.float32)
                order=np.argsort(bs,kind='stable').astype(np.int32);a=sweep_queries(v,bs,order);b=direct_queries(v,bs);delta=float(abs(a.astype(np.float64)-b).max())
                if delta>maxdiff:maxdiff=delta;worst=dict(length=n,exponent=exponent,trial=trial)
                if delta>2e-7:
                    np.savez_compressed(DEST/'asymmetric_conditioning_failure.npz',response=v,thresholds=bs,sweep=a,direct=b)
                    raise AssertionError((n,exponent,trial,delta))
                if n==4 and trial==0:
                    truth=decimal_reference(v,bs);err=float(abs(a.astype(np.float64)-truth).max());assert err<=2e-7,err;de=max(de,err);dc+=1
                count+=1;queries+=len(bs)
    result=dict(response_cases=count,threshold_queries=queries,maximum_abs_difference=maxdiff,worst_case=worst,decimal_cases=dc,maximum_decimal_difference=de,tolerance=2e-7,pooling_source_sha256=sha(HERE/'batched_pooling.py'),scope='follow-up asymmetric short-edge cancellation audit; no labels; frozen implementation unchanged')
    dump(DEST/'asymmetric_conditioning_audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':set_num_threads(2);main()
