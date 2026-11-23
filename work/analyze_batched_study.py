from batched_study import *
import hashlib,platform,importlib.metadata as metadata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})

def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def validate_times(rows,repeats):
    for r in rows:
        v=json.loads(r['raw_ms']);assert len(v)==repeats and min(v)>0
        fields=[('median_ms',np.median(v))]
        if repeats!=3:fields += [('q25_ms',np.quantile(v,.25)),('q75_ms',np.quantile(v,.75))]
        for field,expected in fields:assert abs(float(r[field])-float(expected))<1e-7
def main():
    proto=json.loads((STUDY/'formal_cost_protocol.json').read_text(encoding='utf-8'))
    for name,h in proto['source_sha256'].items():assert sha(HERE/name)==h,name
    pipeline=json.loads((STUDY/'final_pipeline_validation.json').read_text(encoding='utf-8'))
    assert pipeline['configurations']==24 and pipeline['predictions']==80244 and pipeline['all_predictions_identical']
    numerical=[json.loads((STUDY/f'numerical_{c}.json').read_text(encoding='utf-8')) for c in CASES]
    assert all(r['max_abs_difference']<=2e-7 and r['cached_K32_bitwise_equal'] for r in numerical)
    replay=[]
    for path in sorted((STUDY/'replay').glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'));d,ip=input_data(r['case']);assert r['input_sha256']==sha(ip)
        with np.load(path.with_suffix('.npz')) as z:
            for key in ('truth','object_ids','phases'):np.testing.assert_array_equal(z[key],d[key])
            assert abs(100*(z['predictions'][1:]==z['truth']).mean()-r['accuracy_pct'])<1e-10
            assert abs(100*(z['predictions'][1:]!=z['predictions'][0]).mean()-r['flip_pct'])<1e-10
        replay.append({k:r[k] for k in ['case','seed','budget','backend','alpha','feature_dim','accuracy_pct','flip_pct']})
    assert len(replay)==40
    final_recomputed=0
    for case in CASES:
        for seed in SEEDS:
            stem=f'{case}_s{seed}_f9996_hybrid'
            path=STUDY/'final_pipeline_validation/replay'/stem
            r=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            dev=json.loads((STUDY/'replay'/stem).with_suffix('.json').read_text(encoding='utf-8'))
            assert r['alpha']==dev['alpha'] and r['cv']==dev['cv']
            with np.load(path.with_suffix('.npz')) as a,np.load((STUDY/'replay'/stem).with_suffix('.npz')) as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                final_recomputed+=a['predictions'].size
    assert final_recomputed==80244
    writecsv(STUDY/'classification_replays.csv',replay)
    large=[]
    for c in CASES:
        e=next(r for r in replay if r['case']==c and r['budget']==49728 and r['backend']=='hybrid')
        k=next(r for r in replay if r['case']==c and r['budget']==49728 and r['backend']=='K32')
        normal=next(r for r in replay if r['case']==c and r['budget']==9996 and r['seed']==23)
        large.append(dict(case=c,EI9996_seed23_accuracy_pct=normal['accuracy_pct'],EI49728_seed23_accuracy_pct=e['accuracy_pct'],K32_49728_seed23_accuracy_pct=k['accuracy_pct'],EI_minus_K32_pp=e['accuracy_pct']-k['accuracy_pct'],interpretation='one fixed seed; strict training-only parameter selection; capacity study, not additional independent datasets'))
    writecsv(STUDY/'large_budget_accuracy.csv',large)
    times=[];pilots=[];feature_effects=[]
    for c in CASES:
        for b in BUDGETS:
            rr=readcsv(STUDY/f'benchmarks/{c}_f{b}.csv');validate_times(rr,9);times+=rr
            pp=readcsv(STUDY/f'benchmarks/{c}_f{b}_pilot.csv');validate_times(pp,3);pilots+=pp
            selected=min(pp,key=lambda r:float(r['median_ms']))['method'];assert all(r['selected_K32_training_backend']==selected for r in rr)
            table={r['method']:r for r in rr};h=float(table['hybrid_EI']['median_ms']);km=float(table[selected]['median_ms'])
            best=min(float(r['median_ms']) for r in rr if r['method'].startswith('K32_'))
            feature_effects.append(dict(case=c,budget=b,hybrid_ms=h,selected_K32_backend=selected,selected_K32_ms=km,speedup_vs_selected_K32=km/h,speedup_vs_best_measured_K32=best/h,speedup_vs_cached_EI=float(table['cached_EI']['median_ms'])/h,speedup_vs_old_direct_EI=float(table['direct_EI']['median_ms'])/h,speedup_vs_cached_K1024=float(table['K1024_cached_fixed']['median_ms'])/h,hard_sorted_ms=float(table['hard_sorted']['median_ms']),original_MiniRocket_ms=float(table['original_MiniRocket']['median_ms']),ratio_definition='comparator median / hybrid median; >1 means hybrid faster'))
    writecsv(STUDY/'feature_timings.csv',times);writecsv(STUDY/'training_backend_pilots.csv',pilots);writecsv(STUDY/'feature_speedups.csv',feature_effects)
    e2e=[];e2e_effects=[]
    for c in CASES:
        for b in (9996,49728):
            rr=readcsv(STUDY/f'benchmarks/{c}_f{b}_e2e.csv');validate_times(rr,9);e2e+=rr
            for batch in (1,64):
                table={r['method']:r for r in rr if int(r['batch'])==batch};h=float(table['hybrid_EI']['median_ms'])
                assert int(table['direct_EI']['prediction_mismatches_on_batch'])==0 and int(table['cached_EI']['prediction_mismatches_on_batch'])==0
                e2e_effects.append(dict(case=c,budget=b,batch=batch,hybrid_ms=h,K32_ms=float(table['K32_cached_fixed']['median_ms']),K1024_ms=float(table['K1024_cached_fixed']['median_ms']),speedup_vs_K32=float(table['K32_cached_fixed']['median_ms'])/h,speedup_vs_K1024=float(table['K1024_cached_fixed']['median_ms'])/h,speedup_vs_old_direct_EI=float(table['direct_EI']['median_ms'])/h,scope='same EI-trained head; extraction+conversion+scaling+prediction'))
    writecsv(STUDY/'end_to_end_timings.csv',e2e);writecsv(STUDY/'end_to_end_speedups.csv',e2e_effects)
    synth=[]
    for p in sorted((STUDY/'synthetic').glob('*.csv')):
        rr=readcsv(p);validate_times(rr,5);synth+=rr
    assert len(synth)==120
    writecsv(STUDY/'synthetic_scaling.csv',synth)
    # All cases shown; no pooled cross-task accuracy or cherry-picked speed profile.
    labels={'mpeg7':'MPEG-7','hand':'HandOutlines','fish':'Fish','osuleaf':'OSULeaf'}
    name=lambda c:labels[c.rsplit('_',1)[0]]+' / '+c.rsplit('_',1)[1]
    fig,axes=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
    for ax,c in zip(axes.flat,CASES):
        rr=[r for r in feature_effects if r['case']==c]
        for key,label,color in [('speedup_vs_best_measured_K32','Best measured K32','#2d728f'),('speedup_vs_cached_EI','Cached EI','#898989'),('speedup_vs_old_direct_EI','Old direct EI','#d77c36')]:
            ax.plot([r['budget'] for r in rr],[r[key] for r in rr],'-o',label=label,color=color)
        ax.axhline(1,color='black',linewidth=.7);ax.set_xscale('log');ax.set_yscale('log',base=2);ax.set_title(name(c));ax.set_xlabel('Features');ax.set_ylabel('Comparator / hybrid time');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(STUDY/'feature_speedups.pdf');fig.savefig(STUDY/'feature_speedups.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
    for ax,c in zip(axes.flat,CASES):
        for batch,color in [(1,'#557f38'),(64,'#8e5e9d')]:
            rr=[r for r in e2e_effects if r['case']==c and r['batch']==batch]
            ax.plot([r['budget'] for r in rr],[r['speedup_vs_K32'] for r in rr],'-o',label=f'Batch {batch}',color=color)
        ax.axhline(1,color='black',linewidth=.7);ax.set_xscale('log');ax.set_title(name(c));ax.set_xlabel('Features');ax.set_ylabel('K32 / hybrid end-to-end time');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(STUDY/'end_to_end_speedups.pdf');fig.savefig(STUDY/'end_to_end_speedups.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,family in zip(axes,('white','smooth')):
        for budget,color in zip(BUDGETS,('#777777','#2d728f','#d77c36')):
            xx=[];yy=[]
            for n in (32,64,128,256,512):
                table={r['method']:float(r['median_ms']) for r in synth if int(r['length'])==n and int(r['budget'])==budget and r['family']==family}
                xx.append(n);yy.append(table['K32_cached_fixed']/table['hybrid_EI'])
            ax.plot(xx,yy,'-o',label=str(budget),color=color)
        ax.axhline(1,color='black',linewidth=.7);ax.set_xscale('log',base=2);ax.set_title(family+' synthetic workload');ax.set_xlabel('Response length');ax.set_ylabel('K32 / hybrid feature time');ax.legend(title='Features');ax.grid(alpha=.2)
    fig.savefig(STUDY/'synthetic_scaling.pdf');fig.savefig(STUDY/'synthetic_scaling.png',dpi=300);plt.close(fig)
    digest={}
    for b in BUDGETS:
        rr=[r for r in feature_effects if r['budget']==b];v=[r['speedup_vs_best_measured_K32'] for r in rr]
        digest[str(b)]=dict(feature_speedup_min=min(v),feature_speedup_max=max(v),faster_cases=sum(x>1 for x in v),cases=len(v))
        if b!=756:
            for batch in (1,64):
                v=[r['speedup_vs_K32'] for r in e2e_effects if r['budget']==b and r['batch']==batch]
                digest[str(b)][f'e2e_batch{batch}']=dict(min=min(v),max=max(v),faster_cases=sum(x>1 for x in v))
    audit=dict(final_replay=pipeline,real_feature_values_checked=sum(r['feature_values'] for r in numerical),non_bitwise_values=sum(r['non_bitwise_equal_values'] for r in numerical),max_abs_feature_difference=max(r['max_abs_difference'] for r in numerical),large_budget_configurations=16,feature_conditions=24,end_to_end_conditions=16,synthetic_workloads=30,all_timing_statistics_recomputed=True,frozen_cost_sources_match=True,speed_summary=digest)
    dump(STUDY/'iteration_audit.json',audit)
    dump(STUDY/'runtime_environment.json',dict(python=sys.version,platform=platform.system()+' '+platform.release(),processor=platform.processor(),libraries={n:metadata.version(n) for n in ['numpy','scipy','numba','llvmlite','scikit-learn','matplotlib']},numba_threads=2,blas_threads=1,affinity_or_frequency_fixed=False))
    print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
