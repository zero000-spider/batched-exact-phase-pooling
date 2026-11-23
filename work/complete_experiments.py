"""Frozen multi-seed confirmation; serial execution preserves timing comparability."""
import os, sys, json, shutil, subprocess, hashlib, time
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE / ('project' if (HERE/'project').exists() else 'eippv_extension/project')
OUT = Path(os.environ.get('EIPPV_CONFIRMATION_OUT', str(HERE/'results' if (HERE/'project').exists() else HERE.parent/'outputs/initial_confirmation')))
CASES = ('mpeg7_32','mpeg7_64','hand_32','hand_64')
METHODS = ('original','circular','phase16','phase32','eippv')
SEEDS = (23,37,71)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=CASES,
        methods=METHODS,seeds=SEEDS,features=9996,phase_seed=20261108,
        primary='EI-PPV minus circular, accuracy and flip, four cases, Holm family of 8',
        secondary='EI-PPV minus original/phase16/phase32, descriptive intervals',
        inference='Average phases and model seeds within object; class-stratified object bootstrap. Conditional on fixed split and 3 seeds; seeds are not independent datasets.',
        source_worker_sha256=sha(ROOT/'contour_submission/repro/confirm_metrics.py'))
    target=OUT/'protocol.json'
    if not target.exists(): target.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    env=os.environ.copy()
    env.update(NUMBA_NUM_THREADS='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic()
    for seed in SEEDS:
        folder=OUT/f'raw/seed{seed}'; folder.mkdir(parents=True,exist_ok=True)
        env.update(EIPPV_OUTPUT_DIR=str(folder),EIPPV_MODEL_SEED=str(seed))
        for case in CASES:
            cached=folder/f'input_{case}.npz'
            if not cached.exists(): shutil.copy2(ROOT/f'contour_submission/experiments/input_{case}.npz',cached)
            for method in METHODS:
                record=folder/f'{case}_{method}.json'
                if record.exists(): continue
                t=time.monotonic()
                proc=subprocess.run([sys.executable,'-B',str(ROOT/'contour_submission/repro/confirm_metrics.py'),'worker','--case',case,'--method',method],env=env,cwd=ROOT,capture_output=True,text=True)
                if proc.returncode:
                    (folder/f'{case}_{method}.error.txt').write_text(proc.stderr,encoding='utf-8')
                    raise RuntimeError(proc.stderr)
                row=json.loads(proc.stdout); row['wall_seconds']=time.monotonic()-t
                record.write_text(json.dumps(row,indent=2),encoding='utf-8')
                print(f'{seed} {case} {method}: accuracy={row["accuracy"]:.6f}, flip={row["flip_rate"]:.6f}, {row["wall_seconds"]:.1f}s; elapsed {time.monotonic()-started:.0f}s',flush=True)
    print('ALL 60 WORKERS COMPLETE',flush=True)

if __name__=='__main__': main()
