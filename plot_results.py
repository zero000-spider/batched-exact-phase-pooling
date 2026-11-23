"""Render the five manuscript figures from frozen numerical CSVs."""
from pathlib import Path
import argparse, csv, hashlib, json, os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reproduced/plot_cache'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'reproduced/paper_figures')
args=parser.parse_args()
ASSETS=args.output.resolve();ASSETS.mkdir(parents=True,exist_ok=True)
CHAIN=ROOT/'outputs/evidence_chain_strict'
MODERN=ROOT/'outputs/modern_baselines_stability'
BAT=ROOT/'outputs/shared_threshold_study'
used={}
def read(path):
    used[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
primary=read(CHAIN/'primary_tests.csv')
hk=read(MODERN/'high_k_features.csv')
fs=read(BAT/'feature_speedups.csv');es=read(BAT/'end_to_end_speedups.csv')
synth=read(BAT/'synthetic_scaling.csv')
cases=[f'{s}_{n}' for s in ('mpeg7','hand','fish','osuleaf') for n in (32,64)]
names={'mpeg7':'MPEG-7','hand':'HandOutlines','fish':'Fish','osuleaf':'OSULeaf'}
def name(c):
    s,n=c.rsplit('_',1);return names[s]+'/'+n
def one(rows,**keys):return next(r for r in rows if all(str(r[k])==str(v) for k,v in keys.items()))

def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(6.6,3.1),layout='constrained')
    for ax,metric,title in zip(axs,['accuracy','flip'],['Accuracy effect','Prediction disagreement effect']):
        rr=[r for r in primary if r['metric']==metric]
        for i,r in enumerate(rr):
            v=float(r['effect_pp']);lo=float(r['ci95_low_pp']);hi=float(r['ci95_high_pp'])
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color='#27647e')
        ax.axvline(0,color='gray',ls='--',lw=.8);ax.set_yticks(range(4),[names[r['case'].rsplit('_',1)[0]] for r in rr]);ax.set_xlabel('EI minus hard PPV (pp)');ax.set_title(title);ax.grid(axis='x',alpha=.18)
    fig.savefig(ASSETS/'primary.png',dpi=240);fig.savefig(ASSETS/'primary.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(6.6,2.8),layout='constrained')
    budgets=[756,9996,49728]
    for ax,b in zip(axs,budgets):
        rr=[one(fs,case=c,budget=b) for c in cases]
        vals=[float(r['speedup_vs_best_measured_K32']) for r in rr]
        ax.barh(range(8),vals,color=['#286980' if c.endswith('_32') else '#94b9c6' for c in cases]);ax.axvline(1,color='black',ls='--',lw=.8)
        ax.set_yticks(range(8),[name(c).replace('HandOutlines','Hand') for c in cases] if b==756 else []);ax.set_xlim(0,3.25);ax.set_title(f'{b:,} features');ax.set_xlabel('Best K32 / EI time');ax.invert_yaxis()
    fig.savefig(ASSETS/'feature_cost.png',dpi=240);fig.savefig(ASSETS/'feature_cost.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(6.6,3.1),layout='constrained')
    for ax,batch in zip(axs,[1,64]):
        for i,c in enumerate(cases):
            vals=[float(one(es,case=c,budget=b,batch=batch)['speedup_vs_K32']) for b in [9996,49728]]
            ax.plot([0,1],vals,'-o',label=name(c).replace('HandOutlines','Hand'),ms=3,lw=.8)
        ax.set_xticks([0,1],['9,996','49,728']);ax.set_xlabel('Features');ax.set_title(f'Batch {batch}');ax.set_ylabel('K32 / EI full inference time');ax.axhline(1,color='gray',ls='--');ax.grid(alpha=.15)
    axs[1].legend(fontsize=6.5,ncol=2);fig.savefig(ASSETS/'e2e.png',dpi=240);fig.savefig(ASSETS/'e2e.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(6.6,3.0),layout='constrained')
    for c in cases:
        rr=sorted([r for r in hk if r['case']==c],key=lambda r:int(r['K']))
        xx=[int(r['K']) for r in rr]
        axs[0].plot(xx,[float(r['feature_max_abs_error']) for r in rr],'-o',ms=2,lw=.8,label=name(c).replace('HandOutlines','Hand'))
        axs[1].plot(xx,[float(r['prediction_agreement_pct']) for r in rr],'-o',ms=2,lw=.8)
    axs[0].plot([16,32,64,128,1024],[1/k for k in [16,32,64,128,1024]],'k--',lw=.8,label='1/K bound');axs[0].set_yscale('log');axs[0].set_ylabel('Maximum absolute feature error');axs[0].legend(fontsize=5.5,ncol=2)
    axs[1].set_ylabel('Prediction agreement (%)');axs[1].set_ylim(99.7,100.02)
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xlabel('K');ax.grid(alpha=.2)
    fig.savefig(ASSETS/'highk.png',dpi=240);fig.savefig(ASSETS/'highk.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(6.6,2.8),layout='constrained')
    for ax,family in zip(axs,['white','smooth']):
        for b in budgets:
            xx=[];yy=[]
            for length in [32,64,128,256,512]:
                a=one(synth,length=length,budget=b,family=family,method='K32_cached_fixed');e=one(synth,length=length,budget=b,family=family,method='hybrid_EI')
                xx.append(length);yy.append(float(a['median_ms'])/float(e['median_ms']))
            ax.plot(xx,yy,'-o',label=f'{b:,}',ms=3)
        ax.set_xscale('log',base=2);ax.axhline(1,color='gray',ls='--');ax.set_xlabel('Response length');ax.set_ylabel('K32 / EI feature time');ax.set_title(f'{family} synthetic workload');ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.savefig(ASSETS/'synthetic.png',dpi=240);fig.savefig(ASSETS/'synthetic.pdf');plt.close(fig)
    print('5 figures generated from frozen CSV records',flush=True)
if __name__=='__main__':
    plots()
    (ASSETS/'source_hashes.json').write_text(json.dumps(used,indent=2),encoding='utf-8')
