"""Independent source-to-row audit and descriptive analysis of two fixed cases.

Does not import the evaluation pipeline. No model or operating-policy changes.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as label_components

ROOT = Path(__file__).resolve().parents[1]
RUN = 'pilot01-greedy-s0'
CASES = ['vial/test_public/good/000_underexposed', 'vial/test_public/bad/003_overexposed']


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=ROOT/'results/evaluation/pilot01')
    parser.add_argument('--output', type=Path, default=ROOT/'results/evaluation/pilot01/followup')
    args = parser.parse_args()
    source = args.assets/'outputs/pilot01'
    data = args.assets/'data/mvtec_ad_2'
    manifest_path, prediction_path = source/'manifest_images.csv', source/'predictions.csv'
    manifest, predictions = read(manifest_path), read(prediction_path)
    by_id = {r['image_id']: r for r in manifest}
    pred = {(r['run_id'], r['image_id']): r for r in predictions}
    if len(by_id) != len(manifest) or len(pred) != len(predictions):
        raise ValueError('Duplicate source key')
    scores = {iid: Decimal(r['image_score']) for (run, iid), r in pred.items() if run == RUN}
    if not all(s.is_finite() for s in scores.values()):
        raise ValueError('Non-finite scores')
    validation = sorted([r for r in manifest if r['split'] == 'validation'],
                        key=lambda r: (scores[r['image_id']], r['image_id']))
    if len(validation) != 41 or any('/good/' not in r['image_id'] for r in validation):
        raise ValueError('Unexpected calibration set')
    # Independent equivalent of the documented policy for six-decimal CSV scores:
    # k=2, third-highest validation score is the boundary; alarms are strictly above it.
    boundary = scores[validation[-3]['image_id']]
    threshold = math.nextafter(float(boundary), math.inf)
    val_rows = [dict(rank_ascending=i+1,image_id=r['image_id'],score=str(scores[r['image_id']]),
                     above_boundary=int(scores[r['image_id']] > boundary)) for i,r in enumerate(validation)]
    evidence, mask_hashes = [], []
    small_count = small_detected = normal_count = false_positives = 0
    test = sorted([r for r in manifest if r['split'] == 'test_public'],key=lambda r:r['image_id'])
    for row in test:
        iid = row['image_id']
        y = int(iid.split('/')[2] == 'bad')
        alarm = scores[iid] > boundary
        if alarm != (float(scores[iid]) >= threshold):
            raise ValueError('Decimal boundary/float threshold disagree')
        areas, ratio, eligible = [], '', False
        if y:
            mask_path = data/'vial/test_public/ground_truth/bad'/(iid.split('/')[-1]+'_mask.png')
            raw = np.asarray(Image.open(mask_path).convert('L'))
            if raw.shape != (1900,1400) or not set(np.unique(raw)).issubset({0,255}):
                raise ValueError('Unexpected original mask')
            regions, count = label_components(raw >= 128, np.ones((3,3), dtype=int))
            areas = [int(np.count_nonzero(regions == region)) for region in range(1, count+1)]
            if not areas: raise ValueError('Empty bad mask')
            ratio = max(areas)/raw.size
            # Exact integer comparison to cutoff 0.005 (=1/200).
            eligible = max(areas)*200 <= raw.size
            mask_hashes.append(dict(image_id=iid,sha256=sha(mask_path)))
        normal_count += not y
        false_positives += (not y) and alarm
        small_count += eligible
        small_detected += eligible and alarm
        evidence.append(dict(image_id=iid,lighting=row['lighting'],y_true=y,image_score=str(scores[iid]),
                             alarm=int(alarm),component_areas_pixels=';'.join(map(str,areas)),
                             max_component_area_ratio=ratio,eligible_small_0_005=int(eligible)))
    quality = [r for r in read(args.results/'quality.csv') if r['run_id']==RUN and r['lighting_group']=='all'
               and Decimal(r['target_calibration_fpr'])==Decimal('.05') and Decimal(r['small_area_cutoff'])==Decimal('.005')]
    if len(quality)!=1:raise ValueError('Expected one target quality row')
    actual = dict(normal_count=normal_count,false_positives=false_positives,small_count=small_count,small_detected=small_detected)
    if any(int(quality[0][k]) != v for k,v in actual.items()):raise ValueError('Export/source disagreement')
    if float(quality[0]['threshold']) != threshold:raise ValueError('Threshold disagreement')
    case_rows = []
    runs = sorted({r['run_id'] for r in predictions})
    calibration = read(args.results/'calibration.csv')
    for iid in CASES:
        mask = np.zeros((224,224), dtype=bool)
        image_path = data/(iid+'.png')
        if '/bad/' in iid:
            p = data/'vial/test_public/ground_truth/bad'/(iid.split('/')[-1]+'_mask.png')
            with Image.open(p) as im:
                mask=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST).crop((16,16,240,240)))>=128
        for run in runs:
            p = source/'runs'/run/'anomaly_maps'/(iid.replace('/','__')+'.npy')
            amap=np.load(p,allow_pickle=False)
            if amap.shape!=(224,224) or not np.isfinite(amap).all():raise ValueError('Invalid case map')
            thr=float(next(r['threshold'] for r in calibration if r['run_id']==run and Decimal(r['target_calibration_fpr'])==Decimal('.05')))
            score=float(pred[run,iid]['image_score'])
            y,x=np.unravel_index(np.argmax(amap),amap.shape)
            case_rows.append(dict(image_id=iid,run_id=run,image_score=score,threshold=thr,score_minus_threshold=score-thr,
                                  image_alarm=int(score>=thr),gt_defect_pixels=int(mask.sum()),
                                  map_max=float(amap.max()),map_argmax_y=int(y),map_argmax_x=int(x),
                                  map_argmax_in_gt=int(mask[y,x]),
                                  mean_map_inside_gt=float(amap[mask].mean()) if mask.any() else '',
                                  mean_map_outside_gt=float(amap[~mask].mean()),
                                  image_sha256=sha(image_path),map_sha256=sha(p)))
    summary=dict(status='passed',run_id=RUN,alpha=.05,cutoff=.005,
                 threshold=threshold,validation_boundary=str(boundary),validation_boundary_id=validation[-3]['image_id'],
                 validation_alerts=sum(r['above_boundary'] for r in val_rows),**actual,
                 false_positive_ids=[r['image_id'] for r in evidence if r['y_true']==0 and r['alarm']],
                 small_detected_ids=[r['image_id'] for r in evidence if r['eligible_small_0_005'] and r['alarm']],
                 manifest_sha256=sha(manifest_path),predictions_sha256=sha(prediction_path),
                 quality_sha256=sha(args.results/'quality.csv'),mask_hashes=mask_hashes,
                 script_sha256=sha(Path(__file__)),
                 note='Independent recomputation by assistant; user/teammate manual sign-off not yet recorded.')
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/'validation_rank_check.csv',val_rows)
    save(args.output/'source_row_check.csv',evidence)
    save(args.output/'case_analysis.csv',case_rows)
    (args.output/'source_row_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='mask_hashes'},indent=2))
    print(json.dumps(case_rows,indent=2))


if __name__=='__main__': main()
