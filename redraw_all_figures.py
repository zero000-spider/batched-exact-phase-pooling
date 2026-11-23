"""Redraw every archived experimental chart from its recorded numerical tables."""
from pathlib import Path
import argparse,csv,json,hashlib,os,shutil
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reproduced/plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
CHAIN=ROOT/'outputs/evidence_chain_strict'
MODERN=ROOT/'outputs/modern_baselines_stability'
BAT=ROOT/'outputs/shared_threshold_study'
ETH=ROOT/'outputs/eth80_multiview'
BOUNDARY=CHAIN/'boundary_evidence'
CASES=tuple(f'{s}_{n}' for s in ('mpeg7','hand','fish','osuleaf') for n in (32,64))
METHODS=('original','circular','blur','phase4','phase16','phase32','eippv','fourier_svm','cyclic_1nn')
BUDGETS=(756,9996,49728)
USED={}
def read(path):
    USED[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    def scalar(s):
        try:return int(s)
        except ValueError:
            try:return float(s)
            except ValueError:return s
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k:scalar(v) for k,v in row.items()} for row in csv.DictReader(f)]

def redraw_chain(dest):
    OUT=dest/'outputs/evidence_chain_strict';OUT.mkdir(parents=True,exist_ok=True)
    summary=read(CHAIN/'summary.csv');dummy=read(CHAIN/'majority_dummy.csv');primary=read(CHAIN/'primary_tests.csv');budget=read(CHAIN/'budget_control.csv')
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
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

def redraw_cost(dest):
    OUT=dest/'outputs/evidence_chain_strict';OUT.mkdir(parents=True,exist_ok=True)
    cost=read(CHAIN/'matched_inference_cost.csv');plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
    for ax,case in zip(axes.flat,CASES):
     rr=[next(r for r in cost if r['case']==case and r['method']==m) for m in METHODS]
     med=[float(r['median_ms']) for r in rr]
     ax.errorbar(range(len(METHODS)),med,yerr=[[v-float(r['q25_ms']) for v,r in zip(med,rr)],[float(r['q75_ms'])-v for v,r in zip(med,rr)]],fmt='o-',capsize=3);ax.set_yscale('log')
     ax.set_xticks(range(len(METHODS)),['Orig','Hard','Blur','K4','K16','K32','EI','FFT-SVM','CyclicNN'],rotation=35,fontsize=8)
     source,n=case.rsplit('_',1);name={'mpeg7':'MPEG-7','hand':'HandOutlines','fish':'Fish','osuleaf':'OSULeaf'}[source]
     ax.set_title(f'{name} / {n} points');ax.set_ylabel('Batch-64 inference (ms; log scale)');ax.grid(alpha=.2)
    fig.savefig(OUT/'matched_inference_cost.pdf');fig.savefig(OUT/'matched_inference_cost.png',dpi=300);plt.close(fig)

def redraw_modern(dest):
    OUT=dest/'outputs/modern_baselines_stability';OUT.mkdir(parents=True,exist_ok=True)
    standard=read(MODERN/'strong_baseline_summary.csv');by_split=read(MODERN/'split_effects.csv');plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
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

def redraw_highk(dest):
    OUT=dest/'outputs/modern_baselines_stability';OUT.mkdir(parents=True,exist_ok=True)
    hk=read(MODERN/'high_k_features.csv');plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    # Plot actual feature error and prediction agreement, without treating numerical convergence as accuracy improvement.
    fig,axes=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
    for ax,c in zip(axes.flat,CASES):
        rr=[r for r in hk if r['case']==c];xx=[int(r['K']) for r in rr]
        ax.plot(xx,[float(r['feature_mae']) for r in rr],'-o',color='#326f9f');ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('K');ax.set_ylabel('Feature MAE',color='#326f9f');ax.set_title(c);ax.grid(alpha=.2)
        right=ax.twinx();right.plot(xx,[float(r['prediction_agreement_pct']) for r in rr],'--s',color='#bd742e');right.set_ylabel('Prediction agreement (%)',color='#bd742e');right.set_ylim(95,100.05)
    fig.savefig(OUT/'high_k_error_agreement.pdf');fig.savefig(OUT/'high_k_error_agreement.png',dpi=300);plt.close(fig)

def redraw_batched(dest):
    STUDY=dest/'outputs/shared_threshold_study';STUDY.mkdir(parents=True,exist_ok=True)
    feature_effects=read(BAT/'feature_speedups.csv');e2e_effects=read(BAT/'end_to_end_speedups.csv');synth=read(BAT/'synthetic_scaling.csv');plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
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

def redraw_eth80(dest):
    OUT=dest/'outputs/eth80_multiview';OUT.mkdir(parents=True,exist_ok=True)
    rows=read(ETH/'all_results.csv')
    methods=('original','circular','phase16','phase32','eippv')
    for metric in ('accuracy','flip'):
     fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
     for ax,n in zip(axes,(32,64)):
      for s in (23,37,71):ax.plot(range(5),[100*next(r[metric] for r in rows if r['n']==n and r['seed']==s and r['method']==m) for m in methods],'o-',label=f'Seed {s}')
      ax.set_xticks(range(5),['Original','Hard','K=16','K=32','EI-PPV'],rotation=20);ax.set_title(f'ETH80 / {n} points');ax.set_ylabel(metric+' (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
     fig.savefig(OUT/f'{metric}.pdf');fig.savefig(OUT/f'{metric}.png',dpi=300);plt.close(fig)

def redraw_legacy_error(dest):
    OUT=dest/'outputs/evidence_chain_strict/boundary_evidence';OUT.mkdir(parents=True,exist_ok=True)
    CASES=('mpeg7_32','mpeg7_64','hand_32','hand_64');PHASE_METHODS=('phase2','phase4','phase8','phase16','phase32');K_VALUES=np.array([2,4,8,16,32]);index={(r['case'],r['method']):r for r in read(BOUNDARY/'prior_phase_sweep.csv')}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    for ax, case in zip(axes.flat, CASES):
        values = [index[case, method] for method in PHASE_METHODS]
        ax.plot(K_VALUES, [row["feature_mae"] for row in values], "o-", color="#6a3d9a", label="MAE")
        ax.plot(K_VALUES, [row["feature_max_error"] for row in values], "s--", color="#1b9e77", label="Max error")
        ax.plot(K_VALUES, 1 / K_VALUES, ":", color="#555555", label="1/K bound")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_xlabel("Finite phases K")
        ax.set_ylabel("Feature error vs EI-PPV")
        ax.set_title(case.replace("_", " / "))
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.savefig(OUT / "feature_error_vs_k.png", dpi=240)
    fig.savefig(OUT / "feature_error_vs_k.pdf")
    plt.close(fig)

def redraw_legacy_noncontour(dest):
    OUT=dest/'outputs/evidence_chain_strict/boundary_evidence';OUT.mkdir(parents=True,exist_ok=True)
    DATASETS=('GunPoint','Coffee','ECG200');METHODS=('original','circular','phase4','phase16','eippv');summary=read(BOUNDARY/'prior_noncontour_summary.csv')
    x = np.arange(len(DATASETS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    for offset, method in enumerate(METHODS):
        vals = [next(row["accuracy_mean"] for row in summary if row["dataset"] == dataset and row["method"] == method) for dataset in DATASETS]
        ax.bar(x + (offset - 2) * width, 100 * np.array(vals), width=width, label=method)
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel("Official-test accuracy (%)")
    ax.set_ylim(88, 101)
    ax.set_title("Exploratory non-contour boundary check (mean of 3 fixed seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(OUT / "noncontour_accuracy.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "noncontour_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced/all_figures')
    parser.add_argument('--check',action='store_true',help='Compare every redrawn PNG with the archived pixels')
    args=parser.parse_args();dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
    for draw in (redraw_chain,redraw_cost,redraw_modern,redraw_highk,redraw_batched,redraw_eth80,redraw_legacy_error,redraw_legacy_noncontour):
        with plt.rc_context():
            plt.rcdefaults();draw(dest)
    # Historical copies refer to the same numerical charts, not new experiments.
    for ext in ('png','pdf'):
        for stem in ('accuracy','flip'):
            shutil.copyfile(dest/f'outputs/eth80_multiview/{stem}.{ext}',dest/f'outputs/evidence_chain_strict/boundary_evidence/prior_ETH80_{stem}.{ext}')
        for stem in ('feature_error_vs_k','noncontour_accuracy'):
            p=dest/f'outputs/evidence_chain_strict/boundary_evidence/{stem}.{ext}'
            p.replace(p.with_name('prior_'+p.name))
    (dest/'source_hashes.json').write_text(json.dumps(USED,indent=2),encoding='utf-8')
    if args.check:
        from PIL import Image
        comparisons=[]
        for original in sorted((ROOT/'outputs').rglob('*.png')):
            relative=original.relative_to(ROOT);redrawn=dest/relative
            with Image.open(original) as a,Image.open(redrawn) as b:
                aa=np.asarray(a.convert('RGBA'));bb=np.asarray(b.convert('RGBA'))
                same=aa.shape==bb.shape and np.array_equal(aa,bb)
            comparisons.append(dict(file=relative.as_posix(),pixel_identical=bool(same)))
        (dest/'pixel_comparisons.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
        if not all(r['pixel_identical'] for r in comparisons):
            raise SystemExit('Pixel differences found; inspect pixel_comparisons.json and the rendering environment.')
        print('ARCHIVED_FIGURE_PIXELS_VERIFIED '+str(len(comparisons)),flush=True)
    print('REDREW_ARCHIVED_FIGURES '+str(len(list(dest.rglob('*.png')))),flush=True)
if __name__=='__main__':main()
