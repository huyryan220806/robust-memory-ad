"""Generate diagnostic figures; all threshold-based panels are exploratory."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_handoff import build_index, locate, transform_mask


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('results/evaluation/pilot01'))
    parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args()
    metric=read(args.results/'pixel_metrics.csv')
    quality=read(args.results/'quality.csv')
    decisions=read(args.results/'image_decisions.csv')
    runs=sorted({r['run_id'] for r in metric})
    lights=['regular','overexposed','underexposed','shift_1','shift_2','shift_3','shift_4']
    colors=['#2563eb','#0d9488','#ea580c']
    fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    for run,color in zip(runs,colors):
        name=run.removeprefix('pilot01-').removesuffix('-s0')
        values={r['lighting_group']:float(r['image_auroc']) for r in metric if r['run_id']==run}
        axes[0].plot(lights,[values[l] for l in lights],marker='o',label=name,color=color)
        rows={r['lighting_group']:r for r in quality if r['run_id']==run and float(r['target_calibration_fpr'])==.05 and float(r['small_area_cutoff'])==.001}
        axes[1].plot(lights,[float(rows[l]['false_positive_rate']) for l in lights],marker='o',label=name,color=color)
    axes[0].set(title='Image AUROC by lighting (5 normal + 15 anomalous images per group)',ylabel='AUROC',ylim=(0,1.05))
    axes[1].set(title='Exploratory FPR: threshold calibrated on 41 validation images (target 5%)',ylabel='False-positive rate',ylim=(-.03,1.05))
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(ncol=3);ax.tick_params(axis='x',rotation=20)
    fig.savefig(args.results/'lighting_diagnostics.png',dpi=160);plt.close(fig)
    files=build_index(args.assets)
    selected=[r for r in decisions if r['run_id']=='pilot01-greedy-s0' and float(r['target_calibration_fpr'])==.05]
    for outcome in ['FP','FN']:
        cases=[r for r in selected if r['outcome']==outcome]
        if not cases:continue
        case=sorted(cases,key=lambda r:r['image_id'])[0]
        iid=case['image_id']
        image_path=locate(files,'/'+iid+'.png')
        with Image.open(image_path) as im: rgb=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240)))
        if case['y_true']=='1':
            path=locate(files,'/vial/test_public/ground_truth/bad/'+iid.split('/')[-1]+'_mask.png')
            with Image.open(path) as im: mask=transform_mask(im.convert('L'))
        else:mask=np.zeros((224,224),dtype=bool)
        maps=[np.load(locate(files,'/'+run+'/anomaly_maps/'+iid.replace('/','__')+'.npy'),allow_pickle=False) for run in runs]
        lo=min(m.min() for m in maps);hi=max(m.max() for m in maps)
        fig,axes=plt.subplots(1,4,figsize=(13,4),layout='constrained')
        axes[0].imshow(rgb);axes[0].set_title('Input crop / GT outline')
        if mask.any():axes[0].contour(mask,levels=[.5],colors='red')
        for ax,run,amap in zip(axes[1:],runs,maps):
            ax.imshow(rgb);heat=ax.imshow(amap,cmap='magma',alpha=.7,vmin=lo,vmax=hi)
            if mask.any():ax.contour(mask,levels=[.5],colors='cyan')
            ax.set_title(run.removeprefix('pilot01-'))
        for ax in axes:ax.axis('off')
        fig.colorbar(heat,ax=list(axes[1:]),shrink=.65,label='Map score (shared scale)')
        fig.suptitle(f'Exploratory greedy {outcome}: {iid}\nscore={float(case["image_score"]):.4f}, threshold={float(case["threshold"]):.4f}; alpha=5%')
        fig.savefig(args.results/f'case_{outcome.lower()}.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
