"""Revalidate final strengthened implementation without replacing development records."""
import batched_study as study
from pathlib import Path
import json

base=study.STUDY
study.STUDY=base/'final_pipeline_validation'
(study.STUDY/'replay').mkdir(parents=True,exist_ok=True)
count=0;predictions=0
for case in study.CASES:
    for seed in study.SEEDS:
        path=study.STUDY/'replay'/f'{case}_s{seed}_f9996_hybrid.json'
        if not path.exists():study.replay(case,seed,9996,'hybrid')
        r=json.loads(path.read_text(encoding='utf-8'))
        assert r['prediction_mismatches']==0 and r['alpha_identical'] and r['cv_counts_identical'],r
        count+=1;predictions+=r['prior_predictions']
assert count==24
study.dump(base/'final_pipeline_validation.json',dict(configurations=count,predictions=predictions,all_predictions_identical=True,all_alpha_identical=True,all_cv_counts_identical=True,pooling_source_sha256=study.sha(study.HERE/'batched_pooling.py')))
print('FINAL IMPLEMENTATION REPLAYS PASS',count,predictions,flush=True)
