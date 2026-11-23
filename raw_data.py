"""Obtain provider archives and reconstruct the study's deterministic inputs."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, urllib.request, urllib.error

ROOT = Path(__file__).resolve().parent
WORK = ROOT/'work'
PROJECT = WORK/'eippv_extension/project'
for folder in (ROOT, WORK, PROJECT): sys.path.insert(0, str(folder))
SOURCES = json.loads((ROOT/'raw_sources.json').read_text(encoding='utf-8'))
EXPECTED = json.loads((ROOT/'input_arrays.json').read_text(encoding='utf-8'))
ALIASES = dict(fish='Fish', hand='HandOutlines', osuleaf='OSULeaf', mpeg7='mpeg7_shape_b')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def archive_path(row):
    return (WORK/'jena' if row['name']=='ETH80' else PROJECT/'rocket_pilot_data')/row['filename']

def fetch(dataset='all', local=None):
    rows=[r for r in SOURCES if dataset=='all' or r['name']==dataset]
    if not rows: raise ValueError('Unknown dataset: '+dataset)
    records=[]
    for row in rows:
        path=archive_path(row)
        accepted={row['sha256'], *row.get('alternative_sha256', [])}
        url=row.get('download_url',row['url'])
        path.parent.mkdir(parents=True,exist_ok=True)
        if local is not None:
            if len(rows)!=1: raise ValueError('--archive requires one --dataset')
            if sha(Path(local)) not in accepted: raise ValueError('Unrecognized archive checksum')
            if Path(local).resolve()!=path.resolve(): shutil.copyfile(local,path)
        downloaded=False
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.download')
            request=urllib.request.Request(url,headers={'User-Agent':'EI-phase-pooling-reproduction/1.0'})
            try:
                with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as out:
                    shutil.copyfileobj(response,out)
                if sha(temporary) not in accepted: raise ValueError('Unrecognized provider archive: '+row['name'])
                temporary.replace(path); downloaded=True
            except urllib.error.URLError as error:
                raise SystemExit(f'Provider download failed for {row["name"]}: {error}. '
                    f'Download from {row["url"]}, then run raw_data.py fetch '
                    f'--dataset {row["name"]} --archive PATH_TO_ARCHIVE.') from None
            finally:
                if temporary.exists(): temporary.unlink()
        if sha(path) not in accepted: raise ValueError('Archive checksum mismatch: '+str(path))
        records.append(dict(dataset=row['name'],sha256=sha(path),source=url,downloaded=downloaded))
    return records

def verify_input(path):
    import numpy as np
    path=Path(path)
    expected=EXPECTED[path.name]['arrays']
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(expected): raise ValueError('Input fields differ: '+path.name)
        for key, reference in expected.items():
            a=z[key]
            actual=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            if actual!=reference: raise ValueError('Input array mismatch: '+path.name+'/'+key)
    return dict(file=path.name,arrays=len(expected),all_array_hashes_match=True,archive_sha256=sha(path))

def rebuild(dataset='all'):
    import numpy as np
    from rocket_directional_pool import load
    from rocket_mpeg7_confirm import resample, contours, split_shapes
    names=list(ALIASES) if dataset=='all' else [dataset if dataset in ALIASES else next(k for k,v in ALIASES.items() if v==dataset)]
    dest=PROJECT/'contour_submission/experiments'
    dest.mkdir(parents=True,exist_ok=True)
    report=[]
    for name in names:
        records=fetch(ALIASES[name])
        paths=[dest/f'input_{name}_{n}.npz' for n in (32,64)]
        if not all(p.exists() for p in paths):
            if name=='mpeg7':
                radii,yall=contours();tr,te=split_shapes(radii,yall)
                x,y,test,truth,ids=radii[tr],yall[tr],radii[te],yall[te],te
            else:
                x,y,test,truth=load(ALIASES[name]);known={r.tobytes() for r in x}
                keep=np.array([r.tobytes() not in known for r in test]);ids=np.flatnonzero(keep)
                test,truth=test[keep],truth[keep]
            phases=np.r_[0.,np.random.default_rng(20261108).uniform(0,1,8)]
            for n,path in zip((32,64),paths):
                if path.exists(): continue
                np.savez_compressed(path,train=resample(x,n,0.),labels=y,views=np.array([resample(test,n,p) for p in phases]),truth=truth,object_ids=ids,phases=phases)
        report.append(dict(dataset=name,provider_archives=records,inputs=[verify_input(p) for p in paths]))
    out=ROOT/'reproduced/data';out.mkdir(parents=True,exist_ok=True)
    (out/f'{dataset}_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('INPUT_ARRAYS_VERIFIED '+', '.join(names),flush=True)
    return report

def prepare_resplits(dest):
    """Regenerate all forty grouped resplit inputs; verify before experiments."""
    rebuild()
    import modern_extension as m
    old=m.OUT
    try:
        m.OUT=Path(dest);(m.OUT/'inputs').mkdir(parents=True,exist_ok=True)
        m.prepare_splits()
        records=[verify_input(p) for p in sorted((m.OUT/'inputs').glob('*.npz'))]
        if len(records)!=40: raise ValueError('Expected forty resplit inputs')
        (m.OUT/'inputs_verification.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    finally: m.OUT=old

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['fetch','rebuild','resplits'])
    p.add_argument('--dataset',default='all');p.add_argument('--archive',type=Path)
    args=p.parse_args()
    if args.action=='fetch': print(json.dumps(fetch(args.dataset,args.archive),indent=2))
    elif args.action=='rebuild': rebuild(args.dataset)
    else: prepare_resplits(ROOT/'reproduced/modern')
