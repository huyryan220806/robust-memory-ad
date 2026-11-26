"""Reconcile all exported rates to per-image decisions and mask metadata."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results/evaluation/pilot01'
    tables={p.stem:list(csv.DictReader(p.open(encoding='utf8',newline=''))) for p in folder.glob('*.csv')}
    expected={'quality':216,'calibration':9,'image_decisions':1260,'mask_audit':105,'pixel_metrics':24,'asset_inventory':648}
    for name,n in expected.items():
        if len(tables[name])!=n: raise ValueError(f'Unexpected rows: {name}')
    masks={r['image_id']:float(r['max_component_area_ratio']) for r in tables['mask_audit']}
    for r in tables['quality']:
        decisions=[x for x in tables['image_decisions'] if x['run_id']==r['run_id'] and x['target_calibration_fpr']==r['target_calibration_fpr'] and (r['lighting_group']=='all' or x['lighting']==r['lighting_group'])]
        small=[x for x in decisions if x['y_true']=='1' and masks[x['image_id']]<=float(r['small_area_cutoff'])]
        counts={'false_positives':sum(x['outcome']=='FP' for x in decisions),
                'normal_count':sum(x['y_true']=='0' for x in decisions),
                'detected_anomalies':sum(x['outcome']=='TP' for x in decisions),
                'anomaly_count':sum(x['y_true']=='1' for x in decisions),
                'small_count':len(small),'small_detected':sum(x['alarm']=='1' for x in small)}
        if any(int(r[k])!=v for k,v in counts.items()):raise ValueError(f'Count mismatch: {r}')
        for metric,numerator,denominator in [('false_positive_rate','false_positives','normal_count'),('all_anomaly_recall','detected_anomalies','anomaly_count'),('small_recall','small_detected','small_count')]:
            if counts[denominator]:
                if abs(float(r[metric])-counts[numerator]/counts[denominator])>1e-12:raise ValueError('Rate mismatch')
            elif r[metric]!='':raise ValueError('Missing denominator represented as a number')
    audit=json.loads((folder/'audit.json').read_text(encoding='utf8'))
    actual=hashlib.sha256((ROOT/'scripts/evaluate_handoff.py').read_bytes()).hexdigest()
    if audit['script_sha256']!=actual:raise ValueError('Stale script hash')
    print('216 result rows reconciled to image decisions; 648 asset hashes; script hash current.')


if __name__=='__main__':main()
