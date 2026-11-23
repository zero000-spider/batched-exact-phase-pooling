"""Rebuild all eight derived inputs from verified raw archives, retaining old caches."""
from complete_experiments import ROOT,HERE,sha
import sys,json,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'contour_submission/repro'))
import confirm_metrics as core
from rocket_directional_pool import load
from rocket_mpeg7_confirm import resample
dest=HERE.parent/'outputs/evidence_chain_strict/inputs_rebuilt';dest.mkdir(parents=True,exist_ok=True)
metadata=json.loads((ROOT/'contour_submission/repro/datasets.json').read_text())
for name in ('MPEG7','HandOutlines','Fish','OSULeaf'):
 if name=='MPEG7':
  assert sha(ROOT/'rocket_pilot_data/mpeg7_shape_b.zip') in {'331b197caf51b0a3cfd76852aeec46b1fb839fa9374b2a7302754cb92fe0680c','ce8d87409d1b87b07352ebf6e0eafb8884bea6e4027cb7a23c228e7a3685bd0d'}
 else:
  expected=next(r['sha256'] for r in metadata if r['name']==name)
  assert sha(ROOT/f'rocket_pilot_data/{name}.zip')==expected
core.OUT=dest
if not any(dest.glob('input_mpeg7*.npz')):core.prepare()
for name,alias in [('Fish','fish'),('OSULeaf','osuleaf')]:
 x,y,test,truth=load(name);known={r.tobytes() for r in x};keep=np.array([r.tobytes() not in known for r in test]);ids=np.flatnonzero(keep);test=test[keep];truth=truth[keep]
 phases=np.r_[0.,np.random.default_rng(20261108).uniform(0,1,8)]
 for n in (32,64):
  path=dest/f'input_{alias}_{n}.npz'
  if not path.exists():np.savez_compressed(path,train=resample(x,n,0.),labels=y,views=np.array([resample(test,n,p) for p in phases]),truth=truth,object_ids=ids,phases=phases)
report=[]
for new in sorted(dest.glob('input_*.npz')):
 old=ROOT/'contour_submission/experiments'/new.name
 preexisting=old.exists()
 if not preexisting:
  old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(new,old)
 with np.load(old) as a,np.load(new) as b:
  assert a.files==b.files
  for key in a.files:np.testing.assert_array_equal(a[key],b[key])
 report.append(dict(file=new.name,compared_to_preexisting_cache=preexisting,arrays_identical=True,old_sha256=sha(old),rebuilt_sha256=sha(new)))
assert len(report)==8
(dest/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('ALL EIGHT INPUTS REBUILT FROM RAW DATA: ARRAY-IDENTICAL')
