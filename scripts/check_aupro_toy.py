"""Small, explicit AU-PRO teaching fixtures; no benchmark scores are computed."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import label


def toy_aupro(scores, masks, limit):
    """Exact score-event curve, region-balanced overlap, linear boundary clipping.

    For teaching fixtures only; validate against official evaluation before
    applying a chosen convention to benchmark data. Eight-connectivity per image.
    """
    scores, masks = np.asarray(scores,dtype=float), np.asarray(masks)
    if scores.shape!=masks.shape or scores.ndim!=3 or not 0<limit<=1:
        raise ValueError('Expected matching N,H,W and limit in (0,1]')
    if not np.isfinite(scores).all() or not np.isin(masks,[0,1]).all():
        raise ValueError('Invalid scores or mask')
    masks=masks.astype(bool)
    weights=np.zeros_like(scores)
    n_regions=0
    for i,mask in enumerate(masks):
        cc,n=label(mask,np.ones((3,3),dtype=int))
        for region in range(1,n+1):
            selected=cc==region
            weights[i][selected]=1/selected.sum()
        n_regions+=n
    negatives=int((~masks).sum())
    if not n_regions or not negatives:raise ValueError('Need regions and negative pixels')
    weights/=n_regions
    points=[(0.,0.)]
    for threshold in sorted(np.unique(scores),reverse=True):
        alarm=scores>=threshold
        points.append((float((alarm & ~masks).sum()/negatives),float(weights[alarm].sum())))
    area=0.
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if x1==x0:continue
        if x0>=limit:break
        end=min(x1,limit)
        end_y=y0+(y1-y0)*(end-x0)/(x1-x0)
        area+=(end-x0)*(y0+end_y)/2
    return area/limit,points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anomalib',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('results/evaluation/pilot01/followup/aupro-toy.json'))
    args=parser.parse_args()
    mask=np.zeros((2,8,8),dtype=np.uint8)
    mask[0,1,1]=1
    mask[0,5,3:6]=1
    perfect=np.where(mask,.9,.1)
    fixtures={'perfect':perfect,'reversed':1-perfect,'all_tied':np.full(mask.shape,.5)}
    results=[]
    for name,scores in fixtures.items():
        for limit in (.05,.3):
            expected={'perfect':1.,'reversed':0.,'all_tied':limit/2}[name]
            result,points=toy_aupro(scores,mask,limit)
            if abs(result-expected)>1e-12:raise ValueError('Toy expectation mismatch')
            row=dict(case=name,fpr_limit=limit,expected_under_declared_interpolation=expected,
                     independent_result=result,curve=points)
            if args.anomalib:
                import torch
                from anomalib.metrics import AUPRO
                metric=AUPRO(fields=['anomaly_map','gt_mask'],fpr_limit=limit,num_thresholds=None)
                try:
                    metric.update(SimpleNamespace(anomaly_map=torch.tensor(scores,dtype=torch.float32),gt_mask=torch.tensor(mask,dtype=torch.int64)))
                    value=float(metric.compute())
                    row['anomalib_result']=value if np.isfinite(value) else None
                    if not np.isfinite(value):
                        row['anomalib_status']='diagnostic_nonfinite'
                        row['anomalib_nonfinite_value']=str(value)
                    else:
                        row['anomalib_status']='matched' if abs(value-result)<1e-5 else 'diagnostic_difference'
                except Exception as exc:
                    row['anomalib_result']=None
                    row['anomalib_status']='diagnostic_error'
                    row['anomalib_error']=f'{type(exc).__name__}: {exc}'
            results.append(row)
    payload=dict(scope='synthetic teaching fixtures only; not pilot AU-PRO',
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 convention='8-connected regions per image; equal region weight; global negative pixels; grouped ties; linear clipping at beta; divide by beta',
                 fixture={'shape':[2,8,8],'regions':2,'positive_pixels':4,'negative_pixels':124},
                 region_weighting_example={'detected_pixels':1,'positive_pixels':4,'pixel_recall':.25,'region_overlaps':[1.,0.],'PRO':.5},
                 results=results)
    packages=['numpy','scipy']+(['anomalib','torch','torchmetrics'] if args.anomalib else [])
    payload['versions']={name:importlib.metadata.version(name) for name in packages}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
