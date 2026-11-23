from submission_chain import *
os.environ['CHAIN_MPEG_GROUPED']='1'
from analyze_complete import write_csv
import gc
rows=[];set_num_threads(2)
with threadpool_limits(limits=1):
 for case in CASES:
  with np.load(ROOT/f'contour_submission/experiments/input_{case}.npz') as z:x=z['train'];y=z['labels'];batch=z['views'][1,:64]
  for method in METHODS:
   record=json.loads((OUT/(stem(case,23,method,9996)+'.json')).read_text())
   if method=='cyclic_1nn':
    fx=np.fft.rfft(x,axis=1);norm=(x*x).sum(1)
    def predict(a):
     return np.array([y[np.argmin(norm+(v*v).sum()-2*np.fft.irfft(fx*np.conj(np.fft.rfft(v)),n=x.shape[1],axis=1).max(1))] for v in a])
   elif method=='fourier_svm':
    def spec(a):return abs(np.fft.rfft(a,axis=1))[:,1:]
    selected=record['selected'];model=make_pipeline(StandardScaler(),SVC(C=selected['svc__C'],gamma=selected['svc__gamma'],kernel='rbf')).fit(spec(x),y)
    def predict(a):return model.predict(spec(a))
   else:
    p=parameters(x,23,9996,method);ft=transform(x,p,method).astype(np.float64)
    scale=StandardScaler().fit(ft);model=RidgeClassifier(alpha=record['selected']['alpha']).fit(scale.transform(ft),y)
    def predict(a):return model.predict(scale.transform(transform(a,p,method).astype(np.float64)))
   expected=np.load(OUT/(stem(case,23,method,9996)+'.npz'))['predictions'][1,:len(batch)]
   np.testing.assert_array_equal(predict(batch),expected)
   times=[]
   for repeat in range(5):
    gc.collect();start=time.perf_counter();actual=predict(batch);times.append(1000*(time.perf_counter()-start))
    np.testing.assert_array_equal(actual,expected)
   rows.append(dict(case=case,method=method,batch=len(batch),median_ms=float(np.median(times)),q25_ms=float(np.quantile(times,.25)),q75_ms=float(np.quantile(times,.75)),raw_ms=json.dumps(times),seed=23,compiled_and_io_excluded=True,matching_predictions=True))
  print(case+' benchmark complete',flush=True)
write_csv(OUT/'matched_inference_cost.csv',rows)
