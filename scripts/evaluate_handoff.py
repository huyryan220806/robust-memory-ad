"""Reproducible image/mask/map evaluation of pilot01. No baseline training.

Threshold and small-area grids are exploratory, pending team review.
Input files stay external to Git; file inventory records their SHA256.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from audit_image_pilot import audit, read_csv, label, auc_rank, RUNS

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows: {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibrate(scores, alpha):
    """Conservative empirical FPR budget with >=, including tied scores.

Permit floor(alpha*n) alerts; place threshold just ABOVE the (k+1)th
largest score. This is an empirical constraint, not a population guarantee.
"""
    scores = np.asarray(scores, dtype=np.float64)
    if not 0 <= alpha < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid calibration input')
    k = math.floor(alpha * len(scores))
    boundary = np.sort(scores)[len(scores) - k - 1]
    threshold = float(np.nextafter(boundary, np.inf))
    count = int((scores >= threshold).sum())
    if count > k:
        raise ValueError('Calibration FPR budget violated')
    return threshold, count


def components(binary):
    labels, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    areas = np.bincount(labels.ravel())[1:]
    return labels, areas


def transform_mask(image):
    return np.asarray(image.resize((256, 256), Image.Resampling.NEAREST).crop((16, 16, 240, 240))) >= 128


def mask_record(image_id, path):
    with Image.open(path) as image:
        image = image.convert('L')
        raw = np.asarray(image)
        if raw.shape != (1900, 1400):
            raise ValueError(f'Unexpected original mask H,W: {image_id}: {raw.shape}')
        if not set(np.unique(raw)).issubset({0, 255}):
            raise ValueError(f'Mask is not 0/255: {image_id}')
        binary = raw >= 128
        component_labels, areas = components(binary)
        if len(areas) == 0:
            raise ValueError(f'Empty bad mask: {image_id}')
        resized = np.asarray(image.resize((256, 256), Image.Resampling.NEAREST)) >= 128
        cropped = transform_mask(image)
        # Track identities through resampling: counts need not equal connected
        # components after resizing, which can merge or split original regions.
        ids_resized = np.asarray(Image.fromarray(component_labels.astype(np.int32)).resize((256, 256), Image.Resampling.NEAREST))
        ids_crop = ids_resized[16:240, 16:240]
        kept_resize = set(np.unique(ids_resized)) - {0}
        kept_crop = set(np.unique(ids_crop)) - {0}
        record = dict(image_id=image_id, original_height=raw.shape[0], original_width=raw.shape[1],
                      original_defect_pixels=int(binary.sum()), original_component_count=len(areas),
                      max_component_area_ratio=float(areas.max() / binary.size),
                      total_defect_area_ratio=float(binary.mean()),
                      resized_defect_pixels=int(resized.sum()), cropped_defect_pixels=int(cropped.sum()),
                      crop_pixels_lost=int(resized.sum() - cropped.sum()),
                      components_lost_resize=len(areas)-len(kept_resize),
                      components_lost_crop=len(kept_resize-kept_crop),
                      cropped_mask_nonempty=bool(cropped.any()))
        return record, cropped


def build_index(root):
    return [p for p in root.rglob('*') if p.is_file()]


def locate(files, suffix):
    matches = [p for p in files if p.as_posix().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected one asset ending {suffix}; found {len(matches)}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/evaluation/pilot01-proposed.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/evaluation/pilot01')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    fixed = {'calibration_split':'validation','evaluation_split':'test_public',
             'threshold_rule':'nextafter_upper_order_statistic','alarm_rule':'score >= threshold',
             'small_area_denominator':'full_original_image',
             'small_image_rule':'all_original_components_at_or_below_cutoff'}
    if any(cfg[k] != v for k,v in fixed.items()):
        raise ValueError('Unsupported protocol rule; update implementation explicitly')
    if not all(0 < c <= 1 for c in cfg['small_component_area_cutoffs']):
        raise ValueError('Area cutoffs must be ratios in (0,1]')
    if (cfg['resize'],cfg['center_crop'],cfg['component_connectivity']) != (256,224,8):
        raise ValueError('Unsupported geometry/connectivity; update implementation explicitly')
    manifest = read_csv(args.manifest, {'image_id','category','split','lighting','mask_path','image_path'})
    predictions = read_csv(args.predictions, {'run_id','image_id','image_score','map_path'})
    issues = audit(manifest, predictions)
    if issues:
        raise ValueError('\n'.join(issues))
    files = build_index(args.assets)
    test = sorted([m for m in manifest if m['split']=='test_public'], key=lambda m:m['image_id'])
    validation_ids = {m['image_id'] for m in manifest if m['split']=='validation'}
    image_ids = [m['image_id'] for m in test]
    ys = np.array([label(i) for i in image_ids])
    lookup = {(p['run_id'],p['image_id']):p for p in predictions}
    mask_rows, masks, asset_inventory = [], [], []
    for m in test:
        iid = m['image_id']
        if label(iid):
            mask = locate(files, '/vial/test_public/ground_truth/bad/'+iid.split('/')[-1]+'_mask.png')
            rec, transformed = mask_record(iid, mask)
            mask_rows.append(rec)
            asset_inventory.append(dict(kind='mask',image_id=iid,run_id='',relative_path=mask.relative_to(args.assets).as_posix(),sha256=digest(mask)))
        else:
            transformed = np.zeros((224,224),dtype=bool)
        masks.append(transformed)
    masks = np.stack(masks)
    mask_lookup = {r['image_id']:r for r in mask_rows}
    quality, calibration, decisions, pixel_rows = [], [], [], []
    groups = ['all'] + sorted({m['lighting'] for m in test})
    for run in sorted(RUNS):
        maps = []
        for p in sorted([p for p in predictions if p['run_id']==run], key=lambda p:p['image_id']):
            iid = p['image_id']
            path = locate(files, '/'+run+'/anomaly_maps/'+iid.replace('/','__')+'.npy')
            amap = np.load(path, allow_pickle=False)
            if amap.shape != (224,224) or amap.dtype != np.float32 or not np.isfinite(amap).all():
                raise ValueError(f'Invalid map: {run}, {iid}')
            asset_inventory.append(dict(kind='map', image_id=iid, run_id=run,relative_path=path.relative_to(args.assets).as_posix(),sha256=digest(path)))
            if iid in image_ids:
                maps.append((iid,amap))
        maps = np.stack([v for _,v in sorted(maps,key=lambda item:item[0])])
        scores = np.array([float(lookup[run,i]['image_score']) for i in image_ids])
        val = [float(lookup[run,i]['image_score']) for i in sorted(validation_ids)]
        # AP is non-interpolated sklearn average_precision_score, not trapezoidal AUPR.
        for group in groups:
            idx = np.array([group=='all' or m['lighting']==group for m in test])
            auroc = float(roc_auc_score(ys[idx],scores[idx]))
            if abs(auroc-auc_rank(ys[idx].tolist(),scores[idx].tolist()))>1e-12:
                raise ValueError('AUROC cross-check failed')
            pixel_rows.append(dict(run_id=run,lighting_group=group,images=int(idx.sum()),
                                   positive_pixels=int(masks[idx].sum()),total_pixels=int(masks[idx].size),
                                   pixel_ap=float(average_precision_score(masks[idx].ravel(),maps[idx].ravel())),
                                   image_auroc=auroc,scope='224x224_crop',status=cfg['status']))
        for alpha in cfg['calibration_fpr_targets']:
            threshold,val_alerts = calibrate(val,alpha)
            calibration.append(dict(run_id=run,target_calibration_fpr=alpha,threshold=threshold,
                                    calibration_count=len(val),calibration_alerts=val_alerts,
                                    empirical_calibration_fpr=val_alerts/len(val),rule=cfg['threshold_rule'],status=cfg['status']))
            alarm = scores>=threshold
            for i,m in enumerate(test):
                decisions.append(dict(run_id=run,image_id=m['image_id'],lighting=m['lighting'],y_true=int(ys[i]),
                                      image_score=scores[i],target_calibration_fpr=alpha,threshold=threshold,
                                      alarm=int(alarm[i]),outcome=('TP' if ys[i] else 'FP') if alarm[i] else ('FN' if ys[i] else 'TN')))
            for cutoff in cfg['small_component_area_cutoffs']:
                small = np.array([bool(label(i) and mask_lookup[i]['max_component_area_ratio']<=cutoff) for i in image_ids])
                for group in groups:
                    idx = np.array([group=='all' or m['lighting']==group for m in test])
                    normal = idx & (ys==0)
                    anomalous = idx & (ys==1)
                    eligible = idx & small
                    ns = int(eligible.sum())
                    fp = int((alarm & normal).sum())
                    tp = int((alarm & anomalous).sum())
                    stp = int((alarm & eligible).sum())
                    quality.append(dict(run_id=run,category='vial',split='test_public',lighting_group=group,
                                        target_calibration_fpr=alpha,threshold=threshold,small_area_cutoff=cutoff,
                                        false_positives=fp,normal_count=int(normal.sum()),false_positive_rate=fp/int(normal.sum()),
                                        detected_anomalies=tp,anomaly_count=int(anomalous.sum()),all_anomaly_recall=tp/int(anomalous.sum()),
                                        small_detected=stp,small_count=ns,small_recall=stp/ns if ns else '',
                                        missing_reason='' if ns else 'no_eligible_small_images',
                                        status=cfg['status'],protocol_version=cfg['protocol_version']))
        print(f'Completed {run}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('quality',quality),('calibration',calibration),('image_decisions',decisions),
                      ('mask_audit',mask_rows),('pixel_metrics',pixel_rows),('asset_inventory',asset_inventory)]:
        dump_csv(args.output/(name+'.csv'),rows)
    audit_report = dict(protocol=cfg,manifest_sha256=digest(args.manifest),predictions_sha256=digest(args.predictions),
                        config_sha256=digest(args.config),script_sha256=digest(__file__),
                        baseline_commit='6d7e8d8f0127838aae8f9c2dd3f2c85a87b94141',
                        evaluated_masks=len(mask_rows),evaluated_maps=sum(r['kind']=='map' for r in asset_inventory),
                        masks_nonempty_after_crop=sum(r['cropped_mask_nonempty'] for r in mask_rows),
                        masks_losing_pixels_in_crop=sum(r['crop_pixels_lost']>0 for r in mask_rows),
                        components_lost_resize=sum(r['components_lost_resize'] for r in mask_rows),
                        components_lost_crop=sum(r['components_lost_crop'] for r in mask_rows),
                        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','Pillow']},
                        limitations=['Exploratory policies pending team approval.',
                                     'test_public already inspected during development; not untouched final test.',
                                     'No independent specimen confirmation or confidence intervals.',
                                     'No baseline re-training, latency measurement, or AU-PRO claimed.'])
    (args.output/'audit.json').write_text(json.dumps(audit_report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit_report,indent=2))


if __name__=='__main__':
    main()
