from modern_extension import *
import csv
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
OLD=HERE.parent/'outputs/evidence_chain_strict'
def writecsv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def interval(diff,truth,rng):
    classes=np.unique(truth);idx=[np.flatnonzero(truth==c) for c in classes]
    reps=np.zeros(10000)
    for ii in idx:reps+=diff[rng.choice(ii,size=(10000,len(ii)),replace=True)].sum(axis=1)
    return (100*np.quantile(reps/len(truth),[.025,.975])).tolist()
def metrics(p,y):return ((p[1:]==y).mean(axis=0),(p[1:]!=p[0]).mean(axis=0))
@lru_cache(maxsize=None)
def load_old(case,seed,method):
    suffix='_groupcv' if case.startswith('mpeg7') else ''
    path=OLD/f'{case}_s{seed}_{method}_f9996{suffix}.npz'
    d,_=input_data(case)
    with np.load(path) as z:
        known={r.tobytes() for r in d['train']};originalpath=ROOT/f'contour_submission/experiments/input_{case}.npz'
        with np.load(originalpath) as orig:keep=np.array([r.tobytes() not in known for r in orig['views'][0]])
        assert np.array_equal(z['truth'][keep],d['truth']);assert np.array_equal(z['object_ids'][keep],d['object_ids'])
        return z['predictions'][:,keep]

def main():
    manifest=json.loads((OUT/'job_manifest.json').read_text(encoding='utf-8'))+json.loads((OUT/'additional_manifest.json').read_text(encoding='utf-8'));audit=[];rows=[];preds={};labels={}
    for j in manifest:
        stem=f"{j['case']}_{j['recipe']}_s{j['seed']}_split{j['split']}";path=OUT/'raw'/stem
        assert path.with_suffix('.json').exists(),stem
        r=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));d,ip=input_data(j['case'],j['split'])
        assert r['input_sha256']==sha(ip)
        with np.load(path.with_suffix('.npz')) as z:
            for key in ['truth','object_ids','phases']:np.testing.assert_array_equal(z[key],d[key])
            p=z['predictions'];assert p.shape==(9,len(d['truth']))
        acc,flip=metrics(p,d['truth'])
        assert abs(100*acc.mean()-r['accuracy_pct'])<1e-10
        assert abs(100*flip.mean()-r['flip_pct'])<1e-10
        if j['recipe'].endswith('_controlled'):assert r['selected']['feature_dim']==9408
        if j['recipe']=='multi_standard':assert r['selected']['feature_dim']==49728
        if j['recipe']=='self_standard':assert r['selected']['num_runs']==10 and r['selected']['feature_dim'] in [9996,19992]
        audit.append(dict(configuration=stem,predictions=p.size,passed=True))
        rows.append({k:r[k] for k in ['case','recipe','seed','split','accuracy_pct','flip_pct','objects','seconds']}|dict(anchor_accuracy_pct=100*float((p[0]==d['truth']).mean()),worst_phase_pct=100*float((p[1:]==d['truth']).mean(axis=1).min()),feature_dim=r['selected']['feature_dim'],selected_pooling=r['selected'].get('selected_pooling',''),execution_scope=r.get('execution','single configuration wall time; not a matched inference benchmark')))
        preds[j['case'],j['recipe'],j['seed'],j['split']]=(acc,flip)
        labels[j['case'],j['split']]=d['truth']
    writecsv(OUT/'prediction_audit.csv',audit);writecsv(OUT/'all_run_metrics.csv',rows)
    oldsummary=readcsv(OLD/'summary.csv');standard=[]
    for case in CASES:
        for recipe in ['multi_standard','self_standard','multi_controlled','eippv_controlled']:
            rr=[r for r in rows if r['case']==case and r['recipe']==recipe and r['split']==0]
            standard.append(dict(case=case,method=recipe,feature_dim='/'.join(map(str,sorted(set(r['feature_dim'] for r in rr)))),accuracy_pct=float(np.mean([r['accuracy_pct'] for r in rr])),seed_sd_pp=float(np.std([r['accuracy_pct'] for r in rr],ddof=1)),flip_pct=float(np.mean([r['flip_pct'] for r in rr])),anchor_accuracy_pct=float(np.mean([r['anchor_accuracy_pct'] for r in rr])),worst_phase_pct=float(np.mean([r['worst_phase_pct'] for r in rr]))))
        for method in ['original','circular','phase16','phase32','eippv']:
            r=next(r for r in oldsummary if r['case']==case and r['method']==method)
            yy=labels[case,0];anchor=100*float(np.mean([(load_old(case,s,method)[0]==yy).mean() for s in SEEDS]))
            standard.append(dict(case=case,method=method,feature_dim=r['feature_dim'],accuracy_pct=float(r['accuracy_pct']),seed_sd_pp=float(r['accuracy_seed_sd_pp']),flip_pct=float(r['flip_pct']),anchor_accuracy_pct=anchor,worst_phase_pct=float(r['worst_phase_pct'])))
    writecsv(OUT/'strong_baseline_summary.csv',standard)
    rng=np.random.default_rng(2026111617);effects=[]
    for case in CASES:
        for comparator in ['multi_standard','self_standard','multi_controlled']:
            ei='eippv_controlled' if comparator=='multi_controlled' else 'eippv'
            for mi,metric in enumerate(['accuracy','flip']):
                ds=[]
                for s in SEEDS:
                    a=preds[case,ei,s,0][mi] if ei=='eippv_controlled' else metrics(load_old(case,s,'eippv'),labels[case,0])[mi]
                    ds.append(a-preds[case,comparator,s,0][mi])
                diff=np.mean(ds,axis=0);lo,hi=interval(diff,labels[case,0],rng)
                effects.append(dict(case=case,EI_recipe=ei,comparator=comparator,metric=metric,effect_pp=100*float(diff.mean()),ci95_low_pp=lo,ci95_high_pp=hi,interpretation='descriptive object-paired conditional interval; not multiplicity-adjusted; standard recipes have their own internal selection'))
    writecsv(OUT/'strong_baseline_paired_effects.csv',effects)
    stability=[];by_split=[]
    for case in CASES:
        for comparator in ['circular','phase16','phase32','multi_standard','self_standard']:
            differences=[];flips=[]
            for split in SPLITS:
                a=preds[case,'eippv',23,split];b=preds[case,comparator,23,split]
                da=100*float((a[0]-b[0]).mean());df=100*float((a[1]-b[1]).mean());differences.append(da);flips.append(df)
                by_split.append(dict(case=case,split=split,comparator=comparator,accuracy_effect_pp=da,flip_effect_pp=df))
            stability.append(dict(case=case,comparator=comparator,splits=len(SPLITS),accuracy_effect_mean_pp=float(np.mean(differences)),accuracy_effect_sd_pp=float(np.std(differences,ddof=1)),accuracy_effect_min_pp=min(differences),accuracy_effect_max_pp=max(differences),accuracy_wins=sum(d>1e-12 for d in differences),accuracy_ties=sum(abs(d)<=1e-12 for d in differences),flip_effect_mean_pp=float(np.mean(flips)),flip_effect_min_pp=min(flips),flip_effect_max_pp=max(flips),interpretation='descriptive overlapping resplits; model seed fixed23; no independent-dataset significance test'))
    writecsv(OUT/'split_effects.csv',by_split);writecsv(OUT/'split_stability_summary.csv',stability)
    high_audit=[]
    for case in CASES:
        with np.load(OUT/f'high_k/{case}_predictions.npz') as z:
            reference=load_old(case,23,'eippv');np.testing.assert_array_equal(z['exact'],reference)
            np.testing.assert_array_equal(z['truth'],labels[case,0])
            rr=readcsv(OUT/f'high_k/{case}_features.csv')
            for r in rr:
                k=int(r['K']);pred=z[f'K{k}'];agreement=100*float((pred[1:]==z['exact'][1:]).mean());accuracy=100*float((pred[1:]==z['truth']).mean())
                assert abs(agreement-float(r['prediction_agreement_pct']))<1e-10
                assert abs(accuracy-float(r['approx_same_classifier_accuracy_pct']))<1e-10
                assert float(r['feature_max_abs_error'])<=1/k+3e-6
            high_audit.append(dict(case=case,exact_predictions_match_previous=True,finite_K_metrics_recomputed=True,feature_bound_pass=True,K_values=json.dumps([int(r['K']) for r in rr])))
    writecsv(OUT/'high_k_prediction_audit.csv',high_audit)
    for kind in ['features','timings']:
        rr=[]
        for case in CASES:rr+=readcsv(OUT/f'high_k/{case}_{kind}.csv')
        writecsv(OUT/f'high_k_{kind}.csv',rr)
    # Source-wise figures, avoiding an artificial pooled leaderboard.
    fig,axes=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
    method_order=['circular','phase32','eippv','multi_standard','self_standard','multi_controlled','eippv_controlled']
    for ax,case in zip(axes.flat,CASES):
        rr=[next(r for r in standard if r['case']==case and r['method']==m) for m in method_order]
        ax.errorbar(range(len(rr)),[r['accuracy_pct'] for r in rr],yerr=[r['seed_sd_pp'] for r in rr],fmt='o',capsize=3,color='#386885')
        ax.set_xticks(range(len(rr)),['Hard','K32','EI','MR50k','SelF','MR9408','EI9408'],rotation=30,fontsize=8);ax.set_ylim(50,100);ax.set_title(case+' (mean +/- seed SD)');ax.set_ylabel('Accuracy (%)');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'strong_baseline_accuracy.pdf');fig.savefig(OUT/'strong_baseline_accuracy.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
    for ax,case in zip(axes.flat,CASES):
        for method,color in [('circular','#777777'),('phase16','#288cbd'),('phase32','#dc8b21')]:
            rr=[r for r in by_split if r['case']==case and r['comparator']==method]
            ax.plot(range(1,6),[r['accuracy_effect_pp'] for r in rr],marker='o',label='EI - '+method,color=color)
        ax.axhline(0,color='black',linewidth=.7);ax.set_title(case);ax.set_xticks(range(1,6));ax.set_xlabel('Resplit index');ax.set_ylabel('Accuracy difference (pp)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'split_stability.pdf');fig.savefig(OUT/'split_stability.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
    for ax,case in zip(axes.flat,CASES):
        for method,color in [('multi_standard','#337a89'),('self_standard','#a6608d')]:
            rr=[r for r in by_split if r['case']==case and r['comparator']==method]
            ax.plot(range(1,6),[r['accuracy_effect_pp'] for r in rr],marker='o',label='EI - '+method,color=color)
        ax.axhline(0,color='black',linewidth=.7);ax.set_title(case);ax.set_xticks(range(1,6));ax.set_xlabel('Resplit index');ax.set_ylabel('Accuracy difference (pp)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'modern_split_stability.pdf');fig.savefig(OUT/'modern_split_stability.png',dpi=300);plt.close(fig)
    auditreport=dict(configurations=len(rows),prediction_checks=len(audit),total_predictions_checked=sum(r['predictions'] for r in audit),standard_configurations=sum(r['split']==0 for r in rows),resplit_configurations=sum(r['split']!=0 for r in rows),controlled_dimensions_equal=True,original_results_unchanged=True)
    dump(OUT/'extension_audit.json',auditreport)
    print(json.dumps(auditreport),flush=True)
if __name__=='__main__':main()
