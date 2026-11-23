from batched_study import *
def main():
    rows=[];set_num_threads(2)
    with threadpool_limits(limits=1):
        for case in CASES:
            d,path=input_data(case);a=np.ascontiguousarray(d['views'][1,:64])
            for budget in BUDGETS:
                p=params(d['train'],'eippv',23,budget);order=prepare_orders(p[1],p[2]);ref=features(a,*p,0,False);delta=0.
                for mode,cut in [(-2,0),(0,0),(0,a.shape[1])]:
                    alt=batched_features(a,*p,order,mode,cut);de=float(abs(alt.astype(np.float64)-ref).max());delta=max(delta,de);assert de<=2e-7,(case,budget,mode,de)
                k32=features(a,*p,32,True);np.testing.assert_array_equal(k32,batched_features(a,*p,order,-32));np.testing.assert_array_equal(k32,batched_features(a,*p,order,32))
                np.testing.assert_array_equal(features(a,*p,1024,True),batched_features(a,*p,order,-1024))
                np.testing.assert_array_equal(features(a,*p,1,False),batched_features(a,*p,order,-1))
                rows.append(dict(case=case,budget=budget,batch=len(a),input_sha256=sha(path),max_EI_difference=delta,K32_cached_and_sorted_equal=True,K1024_cached_equal=True,sorted_hard_equal=True))
                print(case+' '+str(budget)+' INPUT_CHECK_PASS',flush=True)
    dump(STUDY/'benchmark_input_checks.json',rows)
if __name__=='__main__':main()
