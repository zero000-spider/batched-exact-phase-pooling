from modern_extension import *
from rocket_mpeg7_confirm import interval
import itertools,ast
def main():
    pairs=list(itertools.product(np.arange(-16,17)/8,repeat=2))+list(map(tuple,np.random.default_rng(91).normal(size=(10000,2))))
    rows=[]
    for k in (64,128,1024):
        for a,b in pairs:
            literal=interval(a,b,k,False);fast=interval(a,b,k,True);exact=interval(a,b,0,False)
            assert abs(literal-fast)<1e-12
            assert abs(literal-exact)<=int(min(a,b)<=0<max(a,b))/k+1e-12
        rows.append(dict(test='optimized_vs_literal_and_bound',K=k,pairs=len(pairs),passed=True))
    m=author_multi()
    for n in (32,64):
        d,_=input_data(f'fish_{n}');x=np.ascontiguousarray(d['train'][:7],dtype=np.float64)
        seed_rng(23);a=m.fit(x,num_features=1250)
        seed_rng(23);b=m.fit(x,num_features=1176)
        for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
        rows.append(dict(test='MR_requested10000_and_actual9408_parameter_equivalence',length=n,actual_final_dimension=int(sum(a[1])*84*8),passed=True))
    dump(OUT/'high_k_implementation_audit.json',rows)
    print('HIGH-K ENUMERATION AND BUDGET AUDIT PASS')
if __name__=='__main__':main()
