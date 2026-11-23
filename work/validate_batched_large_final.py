"""Resolve large-budget implementation provenance by refitting final-source pipelines."""
from pathlib import Path
import sys,subprocess,json
from concurrent.futures import ThreadPoolExecutor
import batched_study as s
BASE=s.STUDY
DEST=BASE/'final_large_validation'
def worker(case,backend):
    path=DEST/'replay'/f'{case}_s23_f49728_{backend}'
    if not path.with_suffix('.json').exists():
        child="import batched_study as s; from pathlib import Path; s.STUDY=Path(sys.argv[1]); (s.STUDY/'replay').mkdir(parents=True,exist_ok=True); s.replay(sys.argv[2],23,49728,sys.argv[3])"
        p=subprocess.run([sys.executable,'-B','-c','import sys; '+child,str(DEST),case,backend],cwd=s.HERE,encoding='utf-8',errors='replace',capture_output=True)
        if p.returncode:raise RuntimeError(p.stderr)
    a=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    b=json.loads((BASE/'replay'/path.name).with_suffix('.json').read_text(encoding='utf-8'))
    for key in ['alpha','cv','accuracy_pct','flip_pct','input_sha256']:assert a[key]==b[key],(case,backend,key)
    with s.np.load(path.with_suffix('.npz')) as az,s.np.load((BASE/'replay'/path.name).with_suffix('.npz')) as bz:
        for k in az.files:s.np.testing.assert_array_equal(az[k],bz[k])
        predictions=az['predictions'].size
    print(case,backend,'FINAL_LARGE_MATCH',flush=True)
    return dict(case=case,backend=backend,predictions=predictions)
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(lambda args:worker(*args),[(c,b) for c in s.CASES for b in ('hybrid','K32')]))
    s.dump(BASE/'final_large_validation.json',dict(configurations=len(rows),predictions=sum(r['predictions'] for r in rows),all_predictions_alpha_cv_identical=True,pooling_source_sha256=s.sha(s.HERE/'batched_pooling.py'),configurations_checked=rows))
