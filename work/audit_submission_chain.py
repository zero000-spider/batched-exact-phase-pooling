from submission_chain import *
import itertools
from rocket_mpeg7_confirm import interval
results=[]
pairs=list(itertools.product(np.arange(-16,17)/8,repeat=2))+list(map(tuple,np.random.default_rng(91).normal(size=(10000,2))))
for k in (2,4,8,16,32):
 for a,b in pairs:
  literal=interval(a,b,k,False);optimized=interval(a,b,k,True);exact=interval(a,b,0,False)
  assert abs(literal-optimized)<1e-12
  assert abs(literal-exact)<=int(min(a,b)<=0<max(a,b))/k+1e-12
 results.append(dict(test='quadrature',K=k,pairs=len(pairs),passed=True))
rng=np.random.default_rng(20261116);x=rng.normal(size=(10,32)).astype(np.float32)
set_num_threads(2)
for m in ('circular','blur','phase4','phase16','phase32','eippv'):
 p=parameters(x,23,756,m);base=transform(x,p,m)
 error=max(float(abs(transform(np.ascontiguousarray(np.roll(x,k,axis=1)),p,m)-base).max()) for k in (1,7,19))
 assert error<2e-6,(m,error)
 results.append(dict(test='integer_shift_feature_invariance',method=m,max_error=error,passed=True))
for n in (32,64):
 train=rng.normal(size=(12,n));test=rng.normal(size=(7,n))
 for query in test:
  corr=np.fft.irfft(np.fft.rfft(train,axis=1)*np.conj(np.fft.rfft(query)),n=n,axis=1).max(axis=1)
  fft=(train*train).sum(1)+(query*query).sum()-2*corr
  literal=np.array([min(((np.roll(query,k)-t)**2).sum() for k in range(n)) for t in train])
  np.testing.assert_allclose(fft,literal,atol=1e-10)
 results.append(dict(test='cyclic_distance_fft_vs_enumeration',length=n,pairs=84,passed=True))
(OUT/'implementation_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('ALL IMPLEMENTATION CHECKS PASS')
