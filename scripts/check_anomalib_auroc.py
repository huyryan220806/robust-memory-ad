"""Independent Anomalib cross-check of CSV-based image AUROC (no training)."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from anomalib.metrics import AUROC
from sklearn.metrics import roc_auc_score

from audit_image_pilot import auc_rank, label, sha256


def check(ys, scores):
    original = np.asarray(scores, dtype=np.float64)
    # TorchMetrics treats out-of-[0,1] scores as logits; avoid unnecessary
    # sigmoid and finite-precision saturation using an affine transform.
    span = np.ptp(original)
    scaled = (original-original.min())/span if span else np.zeros_like(original)
    batch = SimpleNamespace(pred_score=torch.tensor(scaled,dtype=torch.float64),
                            gt_label=torch.tensor(ys,dtype=torch.int64))
    metric = AUROC(fields=['pred_score','gt_label'])
    metric.update(batch)
    result = float(metric.compute())
    reference = float(roc_auc_score(ys,original))
    ranked = auc_rank(list(ys),list(original))
    if abs(result-reference)>1e-6 or abs(ranked-reference)>1e-12:
        raise ValueError('AUROC disagreement')
    return dict(anomalib=result,sklearn=reference,rank=ranked,absolute_difference=abs(result-reference))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/evaluation/pilot01/anomalib-check.json'))
    args=parser.parse_args()
    with args.predictions.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    results={}
    for run in sorted({r['run_id'] for r in rows}):
        selected=[r for r in rows if r['run_id']==run and r['image_id'].split('/')[1]=='test_public']
        results[run]=check([label(r['image_id']) for r in selected],[float(r['image_score']) for r in selected])
    result=dict(versions={p:importlib.metadata.version(p) for p in ['anomalib','torchmetrics','torch','scikit-learn']},
                toy=check([0,0,1,1],[.1,.2,.8,.9]),ties=check([0,1],[1.,1.]),runs=results,
                predictions_sha256=sha256(args.predictions),score_transform='per-run positive affine minmax to [0,1]; float64; rankings preserved')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
