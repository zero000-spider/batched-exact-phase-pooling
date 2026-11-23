"""Verify the retained and currently downloadable MPEG-7 ZIPs contain identical study inputs."""
from pathlib import Path
from zipfile import ZipFile
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--retained',required=True,type=Path)
parser.add_argument('--public',required=True,type=Path)
parser.add_argument('--check-arrays',action='store_true')
args=parser.parse_args()
def sha(b):return hashlib.sha256(b).hexdigest()
def selected(path):
    with ZipFile(path) as archive:
        entries=[n for n in sorted(archive.namelist()) if Path(n).stem.rpartition('-')[2].isdigit() and n.lower().endswith('.gif')]
        names=[Path(n).name for n in entries]
        assert len(names)==len(set(names))==1400
        return names,{Path(n).name:sha(archive.read(n)) for n in entries}
oldnames,old=selected(args.retained);newnames,new=selected(args.public)
assert oldnames==newnames and old==new
canonical=sha(''.join(n+'\0'+old[n]+'\n' for n in oldnames).encode())
result=dict(source_url='https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip',
            retained_container_sha256=sha(args.retained.read_bytes()),public_container_sha256=sha(args.public.read_bytes()),
            retained_bytes=args.retained.stat().st_size,public_bytes=args.public.stat().st_size,
            selected_gifs=1400,identical_names=True,identical_processing_order=True,identical_file_bytes=True,
            selected_member_manifest_sha256=canonical,
            distinction='ZIP directory prefixes and auxiliary entries differ; all 1400 study GIFs are byte-identical.')
if args.check_arrays:
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from rocket_mpeg7_confirm import contours
    a,ay=contours(args.retained);b,by=contours(args.public)
    assert np.array_equal(a,b) and np.array_equal(ay,by)
    result['processed_radius_arrays_identical']=True
    result['processed_shape']=list(a.shape)
dest=ROOT/'contour_submission'/'repro'
(dest/'mpeg7_container_equivalence.json').write_text(json.dumps(result,indent=2))
(dest/'mpeg7_shape_member_hashes.json').write_text(json.dumps(old,indent=2))
print(json.dumps(result,indent=2))
