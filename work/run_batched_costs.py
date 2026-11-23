from pathlib import Path
import subprocess,sys,json,hashlib
here=Path(__file__).resolve().parent;dest=here.parent/'outputs/shared_threshold_study'
protocol=json.loads((dest/'formal_cost_protocol.json').read_text(encoding='utf-8'))
for name,h in protocol['source_sha256'].items():assert hashlib.sha256((here/name).read_bytes()).hexdigest()==h,name
checks=json.loads((dest/'benchmark_input_checks.json').read_text(encoding='utf-8'));assert len(checks)==24
pipeline=json.loads((dest/'final_pipeline_validation.json').read_text(encoding='utf-8'));assert pipeline['all_predictions_identical'] and pipeline['configurations']==24
for args in [('batched_study.py','benchmark'),('batched_e2e_benchmark.py',),('batched_synthetic_scaling.py',)]:
    print('START_SERIAL_STAGE '+args[0]+' '+(' '.join(args[1:])),flush=True)
    subprocess.run([sys.executable,'-B',str(here/args[0]),*args[1:]],check=True)
print('ALL_COST_STAGES_COMPLETE',flush=True)
