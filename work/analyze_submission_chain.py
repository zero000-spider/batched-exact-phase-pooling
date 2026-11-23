import os
os.environ['CHAIN_MPEG_GROUPED']='1'
from submission_chain import ROOT,HERE,OUT,CASES,METHODS,sha,stem
from analyze_complete import ci,pvalue,write_csv
import json,shutil,zipfile
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})

rows=[];scores={};audit=[];data_audit=[];masks={};raw_rows=[];dummy=[]
for case in CASES:
 with np.load(ROOT/f'contour_submission/experiments/input_{case}.npz') as z:
  truth=z['truth'];ids=z['object_ids'];phases=z['phases'];known={r.tobytes() for r in z['train']};keep=np.array([r.tobytes() not in known for r in z['views'][0]])
  masks[case]=keep
  data_audit.append(dict(case=case,train=len(z['train']),test_raw=len(truth),test_retained=int(keep.sum()),excluded_ids=json.dumps(ids[~keep].tolist()),exact_train_duplicates=len(z['train'])-len(known)))
  classes,counts=np.unique(z['labels'],return_counts=True);majority=classes[np.argmax(counts)]
  dummy.append(dict(case=case,train_majority_class=str(majority),accuracy_pct=100*float((truth[keep]==majority).mean()),flip_pct=0.0))
  data_audit[-1]['train_class_counts']=json.dumps(dict(zip(classes.tolist(),counts.tolist())))
  classes,counts=np.unique(truth[keep],return_counts=True);data_audit[-1]['test_class_counts']=json.dumps(dict(zip(classes.tolist(),counts.tolist())))
 for seed in (23,37,71):
  for method in METHODS:
   effective_seed=23 if method=='cyclic_1nn' else seed
   path=OUT/stem(case,effective_seed,method,9996)
   row=json.loads(path.with_suffix('.json').read_text())
   assert row['input_sha256']==sha(ROOT/f'contour_submission/experiments/input_{case}.npz')
   with np.load(path.with_suffix('.npz')) as z:
    pred=z['predictions']
    for key,value in (('truth',truth),('object_ids',ids),('phases',phases)):np.testing.assert_array_equal(z[key],value)
   assert pred.shape==(9,len(truth))
   acc=(pred[1:]==truth).mean(axis=0);flip=(pred[1:]!=pred[0]).mean(axis=0)
   np.testing.assert_allclose(acc.mean(),row['accuracy'],atol=1e-14);np.testing.assert_allclose(flip.mean(),row['flip'],atol=1e-14)
   scores[case,seed,method]=(acc[keep],flip[keep],truth[keep])
   if seed==effective_seed:
    raw_rows.append(dict(case=case,seed=seed,method=method,accuracy_pct=100*row['accuracy'],flip_pct=100*row['flip']))
    row=dict(row);row.update(accuracy=float(acc[keep].mean()),flip=float(flip[keep].mean()),clean_accuracy=float((pred[0,keep]==truth[keep]).mean()),worst_phase_accuracy=float((pred[1:,keep]==truth[keep]).mean(axis=1).min()))
    rows.append(row);audit.append(dict(case=case,seed=seed,method=method,objects=len(truth),prediction_count=pred.size,identity_and_metric_check='PASS'))
write_csv(OUT/'prediction_audit.csv',audit)
write_csv(OUT/'data_audit.csv',data_audit);write_csv(OUT/'full_test_sensitivity.csv',raw_rows)
write_csv(OUT/'majority_dummy.csv',dummy)
summary=[]
for case in CASES:
 for method in METHODS:
  rr=[r for r in rows if r['case']==case and r['method']==method]
  summary.append(dict(case=case,method=method,runs=len(rr),feature_dim=rr[0]['feature_dim'],accuracy_pct=100*np.mean([r['accuracy'] for r in rr]),accuracy_seed_sd_pp=100*np.std([r['accuracy'] for r in rr],ddof=1) if len(rr)>1 else '',flip_pct=100*np.mean([r['flip'] for r in rr]),worst_phase_pct=100*np.mean([r['worst_phase_accuracy'] for r in rr])))
write_csv(OUT/'summary.csv',summary)
rng=np.random.default_rng(2026111602);effects=[]
for case in CASES:
 for comparator in METHODS:
  if comparator=='eippv':continue
  for i,metric in enumerate(('accuracy','flip')):
   ds=[scores[case,s,'eippv'][i]-scores[case,s,comparator][i] for s in (23,37,71)]
   diff=np.mean(ds,axis=0);truth=scores[case,23,'eippv'][2];lo,hi=ci(diff,truth,rng)
   primary=case.endswith('_32') and comparator=='circular'
   effects.append(dict(case=case,comparator=comparator,metric=metric,effect_pp=100*diff.mean(),ci95_low_pp=lo,ci95_high_pp=hi,p=pvalue(diff,rng) if primary else '',holm_p='',analysis='primary_followup' if primary else 'descriptive'))
primary=[r for r in effects if r['analysis']=='primary_followup'];running=0
for rank,r in enumerate(sorted(primary,key=lambda r:r['p'])):
 running=max(running,(len(primary)-rank)*r['p']);r['holm_p']=min(1,running)
write_csv(OUT/'paired_effects.csv',effects);write_csv(OUT/'primary_tests.csv',primary)
budget=[]
for case in CASES:
 if not case.endswith('_32'):continue
 with np.load(ROOT/f'contour_submission/experiments/input_{case}.npz') as source:
  source_truth=source['truth'];source_ids=source['object_ids'];source_phases=source['phases']
 for b in (756,9996):
  for method in ('circular','eippv'):
   rr=[]
   for s in (23,37,71):
    r=json.loads((OUT/(stem(case,s,method,b)+'.json')).read_text())
    with np.load(OUT/(stem(case,s,method,b)+'.npz')) as z:
     for key,value in (('truth',source_truth),('object_ids',source_ids),('phases',source_phases)):np.testing.assert_array_equal(z[key],value)
     np.testing.assert_allclose((z['predictions'][1:]==z['truth']).mean(),r['accuracy'],atol=1e-14)
     np.testing.assert_allclose((z['predictions'][1:]!=z['predictions'][0]).mean(),r['flip'],atol=1e-14)
     pred=z['predictions'][:,masks[case]];truth=z['truth'][masks[case]]
    if b==756:audit.append(dict(case=case,seed=s,method=method,objects=len(source_truth),prediction_count=9*len(source_truth),identity_and_metric_check='PASS'))
    r['accuracy']=float((pred[1:]==truth).mean());r['flip']=float((pred[1:]!=pred[0]).mean());rr.append(r)
   budget.append(dict(case=case,budget=b,method=method,actual_features=rr[0]['feature_dim'],accuracy_pct=100*np.mean([r['accuracy'] for r in rr]),flip_pct=100*np.mean([r['flip'] for r in rr])))
write_csv(OUT/'budget_control.csv',budget)
write_csv(OUT/'prediction_audit.csv',audit)
budget_effects=[]
for case in CASES:
 if not case.endswith('_32'):continue
 for b in (756,9996):
  arrays={}
  for method in ('circular','eippv'):
   values=[]
   for s in (23,37,71):
    with np.load(OUT/(stem(case,s,method,b)+'.npz')) as z:
     pred=z['predictions'][:,masks[case]];truth=z['truth'][masks[case]]
    values.append(((pred[1:]==truth).mean(axis=0),(pred[1:]!=pred[0]).mean(axis=0)))
   arrays[method]=values
  for i,metric in enumerate(('accuracy','flip')):
   diff=np.mean([arrays['eippv'][j][i]-arrays['circular'][j][i] for j in range(3)],axis=0);lo,hi=ci(diff,truth,rng)
   if b==9996:
    canonical=next(r for r in primary if r['case']==case and r['metric']==metric)
    lo,hi=canonical['ci95_low_pp'],canonical['ci95_high_pp']
   budget_effects.append(dict(case=case,budget=b,metric=metric,effect_pp=100*diff.mean(),ci95_low_pp=lo,ci95_high_pp=hi,analysis='descriptive'))
write_csv(OUT/'budget_effects.csv',budget_effects)
def save(fig,name):
 fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.png',dpi=300);plt.close(fig)
source_names={'mpeg7':'MPEG-7','hand':'HandOutlines','fish':'Fish','osuleaf':'OSULeaf'}
def case_title(case):
 source,n=case.rsplit('_',1);return f'{source_names[source]} / {n} points'
for metric,label in [('accuracy_pct','Phase-mean accuracy (%)'),('flip_pct','Prediction flip rate (%)')]:
 fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
 for ax,case in zip(axes.flat,CASES):
  rr=[next(r for r in summary if r['case']==case and r['method']==m) for m in METHODS]
  ax.plot(range(len(METHODS)),[r[metric] for r in rr],'o-',color='#326b96')
  dummy_value=next(r[metric] for r in dummy if r['case']==case)
  ax.text(.02,.02,f'Train-majority dummy: {dummy_value:.2f}%',transform=ax.transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
  ax.set_xticks(range(len(METHODS)),['Orig','Hard','Blur','K4','K16','K32','EI','FFT-SVM','CyclicNN'],rotation=35,fontsize=8)
  ax.set_title(case_title(case));ax.set_ylabel(label);ax.grid(alpha=.2)
 save(fig,'chain_'+metric)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,metric in zip(axes,('accuracy','flip')):
 rr=[r for r in primary if r['metric']==metric]
 for i,r in enumerate(rr):ax.errorbar(r['effect_pp'],i,xerr=[[r['effect_pp']-r['ci95_low_pp']],[r['ci95_high_pp']-r['effect_pp']]],fmt='o',capsize=4)
 ax.axvline(0,ls='--',color='gray');ax.set_yticks(range(len(rr)),[r['case'] for r in rr]);ax.set_title(metric+'; 95% object bootstrap CI');ax.set_xlabel('EI minus hard PPV (pp)');ax.grid(alpha=.2)
save(fig,'primary_effects')
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,metric in zip(axes,('accuracy_pct','flip_pct')):
 for case in CASES:
  if not case.endswith('_32'):continue
  for method,style in [('circular','--'),('eippv','-')]:
   rr=[r for r in budget if r['case']==case and r['method']==method]
   color={'mpeg7_32':'#1f77b4','hand_32':'#ff7f0e','fish_32':'#2ca02c','osuleaf_32':'#d62728'}[case]
   label=source_names[case.rsplit('_',1)[0]]+(' hard' if method=='circular' else ' EI')
   ax.plot([r['budget'] for r in rr],[r[metric] for r in rr],style,marker='o',color=color,label=label)
 ax.set_xscale('log');ax.set_xticks([756,9996],['756','9996']);ax.set_xlabel('Feature budget');ax.set_ylabel('Mean phase accuracy (%)' if metric=='accuracy_pct' else 'Prediction flip rate (%)');ax.grid(alpha=.2)
axes[1].legend(fontsize=6,ncol=2);save(fig,'feature_budget')
print(json.dumps({'verified_configs':len(rows)+24,'primary':primary},indent=2))
