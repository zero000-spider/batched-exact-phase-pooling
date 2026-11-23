from complete_experiments import ROOT,HERE,sha
from analyze_complete import ci,pvalue,write_csv
import numpy as np,json,shutil,zipfile,os
import matplotlib.pyplot as plt
GROUPED=os.environ.get('ETH80_GROUPED')=='1'
FLOAT64=os.environ.get('ETH80_FLOAT64')=='1'
OUT=HERE.parent/('outputs/eth80_multiview' if FLOAT64 else 'outputs/eth80_grouped_diagnostic' if GROUPED else 'outputs/eth80_legacy_diagnostic')
rows=[];scores={};anchor=None
for n in (32,64):
 for s in (23,37,71):
  for m in ('original','circular','phase16','phase32','eippv'):
   p=OUT/f'n{n}_seed{s}_{m}.npz'
   with np.load(p) as z:
    pred=z['predictions'];truth=z['truth'];ids=z['object_ids'];views=z['view_ids']
   if anchor is None:anchor=(truth,ids,views)
   for a,b in zip(anchor,(truth,ids,views)):np.testing.assert_array_equal(a,b)
   acc=(pred==truth).mean(axis=1);flip=(pred[:,1:]!=pred[:,:1]).mean(axis=1)
   r=json.loads(p.with_suffix('.json').read_text());np.testing.assert_allclose(acc.mean(),r['accuracy']);np.testing.assert_allclose(flip.mean(),r['flip'])
   scores[n,s,m]=(acc,flip);rows.append(r)
write_csv(OUT/'all_results.csv',rows)
tests=[];rng=np.random.default_rng(2026111504)
for n in (32,64):
 for comparator in ('circular','original','phase16','phase32'):
  for i,metric in enumerate(('accuracy','flip')):
   diff=np.mean([scores[n,s,'eippv'][i]-scores[n,s,comparator][i] for s in (23,37,71)],axis=0)
   lo,hi=ci(diff,truth[:,0],rng)
   tests.append(dict(n=n,comparator=comparator,metric=metric,effect_pp=100*diff.mean(),ci95_low_pp=lo,ci95_high_pp=hi,p=pvalue(diff,rng) if comparator=='circular' else '',holm_p=''))
primary=[t for t in tests if t['comparator']=='circular'];running=0
for i,t in enumerate(sorted(primary,key=lambda t:t['p'])):
 running=max(running,(4-i)*t['p']);t['holm_p']=min(1,running)
write_csv(OUT/'paired_tests.csv',tests)
methods=('original','circular','phase16','phase32','eippv')
for metric in ('accuracy','flip'):
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for ax,n in zip(axes,(32,64)):
  for s in (23,37,71):ax.plot(range(5),[100*next(r[metric] for r in rows if r['n']==n and r['seed']==s and r['method']==m) for m in methods],'o-',label=f'Seed {s}')
  ax.set_xticks(range(5),['Original','Hard','K=16','K=32','EI-PPV'],rotation=20);ax.set_title(f'ETH80 / {n} points');ax.set_ylabel(metric+' (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.savefig(OUT/f'{metric}.pdf');fig.savefig(OUT/f'{metric}.png',dpi=300);plt.close(fig)
summary=[]
for n in (32,64):
 for m in methods:
  mean=100*np.mean([r['accuracy'] for r in rows if r['n']==n and r['method']==m])
  summary.append(dict(n=n,method=m,accuracy_pct=mean))
write_csv(OUT/'summary.csv',summary)
print(json.dumps(dict(configurations=len(rows),primary=primary),indent=2))
