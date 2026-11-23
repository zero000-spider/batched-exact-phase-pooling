"""Portable entry points; frozen results are read-only references, never run markers."""
from pathlib import Path
import os,sys,json,hashlib,argparse,importlib.metadata as metadata,subprocess,shutil,runpy
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent;WORK=ROOT/'work';RESULTS=ROOT/'reproduced'
os.environ.update(NUMBA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.environ['EIPPV_PROJECT']=str(WORK/'eippv_extension/project')
cache=str((RESULTS/('nc_adapter' if len(sys.argv)>1 and sys.argv[1]=='adapter' else 'nc')).resolve())
os.environ['NUMBA_CACHE_DIR']=('\\\\?\\'+cache) if os.name=='nt' else cache
os.environ['MPLCONFIGDIR']=str(RESULTS/'plot_cache')
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(WORK))
sys.path.insert(0,str(ROOT))

def verify_input(path):
    from raw_data import verify_input as check
    return check(path)

def prepare_case(case):
    from raw_data import rebuild
    rebuild(case.rsplit('_',1)[0])

def reference_sha(path):
    """Use frozen container identifiers only after exact array-hash verification."""
    from raw_data import EXPECTED
    path=Path(path)
    if path.name in EXPECTED:
        verify_input(path)
        return EXPECTED[path.name]['archive_sha256']
    return sha(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def runtime():
    import numpy,scipy,numba,sklearn,matplotlib,PIL,skimage
    mods=[numpy,scipy,numba,sklearn,matplotlib,PIL,skimage]
    prefix=Path(sys.prefix).resolve()
    for mod in mods:assert Path(mod.__file__).resolve().is_relative_to(prefix),(mod.__name__,mod.__file__,str(prefix))
    files=[ROOT/'reproduce.py',ROOT/'raw_data.py',*WORK.rglob('*.py')]
    return dict(python=sys.version,prefix=str(prefix),isolated=sys.flags.isolated,base_prefix=sys.base_prefix,modules={m.__name__:str(Path(m.__file__).resolve()) for m in mods},packages={d.metadata['Name'].lower():d.version for d in metadata.distributions()},sources={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(files)})
def same_arrays(a,b):
    import numpy as np
    with np.load(a) as x,np.load(b) as y:
        assert set(x.files)==set(y.files)
        for key in x.files:np.testing.assert_array_equal(x[key],y[key])
        return int(x['predictions'].size)
def core(args,environment):
    import batched_study as s
    reference=s.STUDY;s.STUDY=RESULTS/'core';(s.STUDY/'replay').mkdir(parents=True,exist_ok=True)
    name=f'{args.case}_s{args.seed}_f{args.budget}_{args.backend}'
    flag=s.STUDY/'checks'/f'{name}.json';target=s.STUDY/'replay'/name;old=reference/'replay'/name
    if flag.exists():
        record=json.loads(flag.read_text(encoding='utf-8'));assert record['environment']==environment
        assert sha(target.with_suffix('.json'))==record['metadata_sha256'] and sha(target.with_suffix('.npz'))==record['predictions_sha256']
    else:s.replay(args.case,args.seed,args.budget,args.backend)
    new=json.loads(target.with_suffix('.json').read_text(encoding='utf-8'));prior=json.loads(old.with_suffix('.json').read_text(encoding='utf-8'))
    for key in ('case','seed','budget','backend','alpha','feature_dim','accuracy_pct','flip_pct','cv'):assert new[key]==prior[key],(name,key)
    _,inputpath=s.input_data(args.case);verify_input(inputpath)
    assert new['input_sha256']==sha(inputpath)
    from raw_data import EXPECTED
    assert prior['input_sha256']==EXPECTED[inputpath.name]['archive_sha256']
    n=same_arrays(target.with_suffix('.npz'),old.with_suffix('.npz'))
    write(flag,dict(name=name,predictions=n,all_predictions_and_selection_identical=True,environment=environment,metadata_sha256=sha(target.with_suffix('.json')),predictions_sha256=sha(target.with_suffix('.npz'))))
    print('CORE_VERIFIED '+name+' '+str(n),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('suite',choices=['verify','adapter','core','chain','modern','numerical','costs','synthetic','statistics','eth80']);p.add_argument('--case');p.add_argument('--seed',type=int,default=23);p.add_argument('--budget',type=int,default=9996);p.add_argument('--backend',choices=['hybrid','K32'],default='hybrid');p.add_argument('--method',default='circular');p.add_argument('--recipe',default='multi_standard');p.add_argument('--split',type=int,default=0);p.add_argument('--jobs',type=int,default=2);args=p.parse_args()
    if args.suite=='verify':
        manifest=json.loads((ROOT/'RELEASE_SHA256.json').read_text(encoding='utf-8'))
        for name,value in manifest.items():assert sha(ROOT/name)==value,name
        protocol=json.loads((ROOT/'outputs/shared_threshold_study/formal_cost_protocol.json').read_text(encoding='utf-8'))
        for name,value in protocol['source_sha256'].items():assert sha(WORK/name)==value,name
        write(RESULTS/'source_verification.json',dict(files=len(manifest),all_hashes_match=True,formal_sources_match_release_manifest=True))
        print('SOURCE_AND_EVIDENCE_VERIFIED',flush=True);return
    env=runtime();write(RESULTS/'environment.json',env)
    import numpy as np
    import batched_study as s
    if args.suite=='adapter':
        from audit_adapter_equivalence import main as check_adapter
        check_adapter(RESULTS/'adapter_equivalence.json')
    elif args.suite=='core':
        if args.case:prepare_case(args.case)
        else:
            from raw_data import rebuild
            rebuild()
        if args.case:core(args,env)
        else:
            jobs=[(c,k,9996,'hybrid') for c in s.CASES for k in s.SEEDS]+[(c,23,49728,b) for c in s.CASES for b in ('hybrid','K32')]
            def run(job):
                c,k,b,backend=job;cmd=[sys.executable,'-I','-B',str(ROOT/'reproduce.py'),'core','--case',c,'--seed',str(k),'--budget',str(b),'--backend',backend]
                log=RESULTS/'logs'/f'{c}_s{k}_f{b}_{backend}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('w',encoding='utf-8') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
                if proc.returncode:raise RuntimeError(f'{job} failed; see {log}')
                print('FRESH_CORE_DONE '+str(job),flush=True)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,jobs))
            records=[json.loads((RESULTS/'core/checks'/f'{c}_s{k}_f{b}_{backend}.json').read_text(encoding='utf-8')) for c,k,b,backend in jobs]
            assert all(r['all_predictions_and_selection_identical'] for r in records)
            write(RESULTS/'core_acceptance.json',dict(configurations=len(records),predictions=sum(r['predictions'] for r in records),all_predictions_and_selection_identical=True,environment=env))
    elif args.suite=='chain':
        import submission_chain as chain
        os.environ['CHAIN_MPEG_GROUPED']='1';old=chain.OUT;chain.OUT=RESULTS/'chain';chain.OUT.mkdir(parents=True,exist_ok=True)
        case=args.case or 'fish_32';prepare_case(case);chain.worker(case,args.seed,args.method,args.budget);name=chain.stem(case,args.seed,args.method,args.budget)
        count=same_arrays(chain.OUT/f'{name}.npz',old/f'{name}.npz')
        a=json.loads((chain.OUT/f'{name}.json').read_text());b=json.loads((old/f'{name}.json').read_text())
        for key in ('accuracy','flip','clean_accuracy','worst_phase_accuracy','selected','cv','fold_validation'):assert a[key]==b[key],key
        verify_input(WORK/f'eippv_extension/project/contour_submission/experiments/input_{case}.npz')
        write(chain.OUT/f'{name}_verification.json',dict(predictions=count,all_arrays_and_selection_identical=True));print('CHAIN_VERIFIED '+name,flush=True)
    elif args.suite=='modern':
        import modern_extension as m
        old=m.OUT;m.OUT=RESULTS/'modern';(m.OUT/'raw').mkdir(parents=True,exist_ok=True);(m.OUT/'inputs').mkdir(exist_ok=True);case=args.case or 'fish_32'
        prepare_case(case)
        if args.split:
            from raw_data import prepare_resplits
            prepare_resplits(m.OUT)
        recipes=[args.recipe]
        if args.recipe=='circular' and args.split:
            import fused_stability as f
            f.OUT=m.OUT;f.joint(case,args.seed,args.split);recipes=list(f.METHODS)
        else:m.worker(case,args.recipe,args.seed,args.split)
        result=[]
        for recipe in recipes:
            name=f'{case}_{recipe}_s{args.seed}_split{args.split}';count=same_arrays(m.OUT/'raw'/f'{name}.npz',old/'raw'/f'{name}.npz');a=json.loads((m.OUT/'raw'/f'{name}.json').read_text());b=json.loads((old/'raw'/f'{name}.json').read_text());assert a['selected']==b['selected']
            _,ip=m.input_data(case,args.split);verify_input(ip);assert a['input_sha256']==sha(ip)
            result.append(dict(name=name,predictions=count,all_arrays_and_selection_identical=True))
        write(m.OUT/f'{case}_{args.recipe}_split{args.split}_verification.json',result);print('MODERN_VERIFIED '+str(result),flush=True)
    elif args.suite=='numerical':
        import audit_batched_pooling as a,check_batched_conditioning as c
        dest=RESULTS/'numerical';dest.mkdir(parents=True,exist_ok=True);a.DEST=dest;c.DEST=dest;a.main();c.main()
    elif args.suite=='costs':
        import batched_e2e_benchmark as e
        s.STUDY=RESULTS/'costs';e.STUDY=s.STUDY
        for name in ['replay','benchmarks']:(s.STUDY/name).mkdir(parents=True,exist_ok=True)
        case=args.case or 'fish_32';prepare_case(case);s.replay(case,args.seed,args.budget,'hybrid');s.benchmark(case,args.budget)
        if args.budget in (9996,49728):e.case_benchmark(case,args.budget)
    elif args.suite=='synthetic':
        import batched_synthetic_scaling as a
        a.STUDY=RESULTS/'synthetic';a.STUDY.mkdir(parents=True,exist_ok=True)
        for n in (32,64,128,256,512):
            for b in s.BUDGETS:
                for family in ('white','smooth'):a.run(n,b,family)
    elif args.suite=='statistics':
        from raw_data import rebuild
        rebuild()
        import submission_chain as chain
        original=chain.OUT;chain.OUT=RESULTS/'statistics';chain.OUT.mkdir(parents=True,exist_ok=True);chain.sha=reference_sha
        for path in original.iterdir():
            if path.suffix in ('.json','.npz'):shutil.copy2(path,chain.OUT/path.name)
        runpy.run_path(str(WORK/'analyze_submission_chain.py'),run_name='__main__')
        for name in ['primary_tests.csv','paired_effects.csv','summary.csv','budget_effects.csv']:
            assert (chain.OUT/name).read_bytes()==(original/name).read_bytes(),name
        write(chain.OUT/'verification.json',dict(primary_tests_and_effects_and_summary_byte_identical=True));print('STATISTICS_VERIFIED',flush=True)
    elif args.suite=='eth80':
        import importlib.util
        spec=importlib.util.spec_from_file_location('review_raw_data',ROOT/'raw_data.py')
        raw_data=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw_data)
        raw_data.fetch('ETH80')
        text=(WORK/'eth80_real.py').read_text(encoding='utf-8');marker="ARCHIVE=HERE/'jena/eth80-contours.tgz'";assert marker in text
        replacement="OUT=Path("+repr(str(RESULTS/'eth80'))+");OUT.mkdir(parents=True,exist_ok=True)\n"+marker
        text=text.replace(marker,replacement);os.environ.update(ETH80_GROUPED='1',ETH80_FLOAT64='1')
        exec(compile(text,str(WORK/'eth80_real.py'),'exec'),{'__name__':'__main__','__file__':str(WORK/'eth80_real.py')})
        old=ROOT/'outputs/eth80_multiview';count=0
        for path in (RESULTS/'eth80').glob('n*_seed*.npz'):count+=same_arrays(path,old/path.name)
        assert len(list((RESULTS/'eth80').glob('n*_seed*.npz')))==30
        write(RESULTS/'eth80/verification.json',dict(configurations=30,predictions=count,all_arrays_identical=True))
if __name__=='__main__':main()
