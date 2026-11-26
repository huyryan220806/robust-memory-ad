"""Audit the vial seed-0 handoff and evaluate image AUROC. Python stdlib only.

No threshold is chosen here: FPR/recall await a reviewed calibration policy.
No small-defect labels are inferred from filenames, scores, or the overlay.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = {'pilot01-full-s0', 'pilot01-random-s0', 'pilot01-greedy-s0'}
LIGHTS = {'regular', 'overexposed', 'underexposed', 'shift_1', 'shift_2', 'shift_3', 'shift_4'}
REFERENCE = {'pilot01-full-s0': .8879, 'pilot01-random-s0': .7518, 'pilot01-greedy-s0': .8914}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing required columns {required}')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'{path}: malformed row')
    return rows


def label(image_id):
    parts = image_id.split('/')
    if len(parts) != 4 or parts[2] not in {'good', 'bad'}:
        raise ValueError(f'Unrecognized label path: {image_id}')
    return int(parts[2] == 'bad')


def auc_rank(labels, scores):
    """Mann-Whitney AUROC, average ranks for exact score ties."""
    if len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError('Invalid labels/scores')
    if any(not math.isfinite(s) for s in scores):
        raise ValueError('Non-finite score')
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auc_pairs(labels, scores):
    """Independent pairwise implementation for audit at this pilot's scale."""
    good = [s for y, s in zip(labels, scores) if y == 0]
    bad = [s for y, s in zip(labels, scores) if y == 1]
    if not good or not bad:
        return None
    return sum((p > n) + .5 * (p == n) for p in bad for n in good) / (len(good) * len(bad))


def audit(manifest, predictions):
    issues = []
    ids = [m['image_id'] for m in manifest]
    issues += [f'duplicate manifest ID: {k}' for k, n in Counter(ids).items() if n > 1]
    if len(manifest) != 472:
        issues.append(f'manifest count {len(manifest)} != 472')
    for m in manifest:
        parts = m['image_id'].split('/')
        label(m['image_id'])
        if parts[0] != m['category'] or parts[1] != m['split'] or m['category'] != 'vial':
            issues.append(f'category/split conflict: {m["image_id"]}')
        if not parts[-1].endswith('_' + m['lighting']):
            issues.append(f'lighting/ID conflict: {m["image_id"]}')
        if not m['image_path'].replace('\\', '/').endswith(m['image_id'] + '.png'):
            issues.append(f'image path conflict: {m["image_id"]}')
        if label(m['image_id']) and m['mask_path'] in ('', 'chua_xac_dinh'):
            issues.append(f'missing bad mask reference: {m["image_id"]}')
    counts = Counter((m['split'], label(m['image_id'])) for m in manifest)
    expected_counts = {('train', 0): 291, ('validation', 0): 41, ('test_public', 0): 35, ('test_public', 1): 105}
    if counts != expected_counts:
        issues.append(f'split/label counts differ: {counts}')
    test = [m for m in manifest if m['split'] == 'test_public']
    if {m['lighting'] for m in test} != LIGHTS:
        issues.append('unexpected test lighting groups')
    for light in LIGHTS:
        if Counter(label(m['image_id']) for m in test if m['lighting'] == light) != {0: 5, 1: 15}:
            issues.append(f'lighting counts differ: {light}')
    if any(m['lighting'] != 'regular' for m in manifest if m['split'] in ('train', 'validation')):
        issues.append('non-regular train/validation lighting')
    keys = [(p['run_id'], p['image_id']) for p in predictions]
    issues += [f'duplicate prediction key: {k}' for k, n in Counter(keys).items() if n > 1]
    if {p['run_id'] for p in predictions} != RUNS:
        issues.append('unexpected run IDs')
    expected_ids = {m['image_id'] for m in manifest if m['split'] in ('validation', 'test_public')}
    for run in sorted(RUNS):
        actual = {p['image_id'] for p in predictions if p['run_id'] == run}
        issues += [f'{run}: missing {k}' for k in sorted(expected_ids - actual)]
        issues += [f'{run}: extra {k}' for k in sorted(actual - expected_ids)]
    for p in predictions:
        try:
            score = float(p['image_score'])
            if not math.isfinite(score):
                raise ValueError()
        except ValueError:
            issues.append(f'invalid score: {p["run_id"]}, {p["image_id"]}')
        expected_map = p['image_id'].replace('/', '__') + '.npy'
        if not p['map_path'].replace('\\', '/').endswith('/' + expected_map):
            issues.append(f'map basename conflict: {p["image_id"]}')
    return issues


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/handoff/vial_seed0')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/pilot/vial_seed0')
    args = parser.parse_args()
    manifest_path = args.input_dir / 'manifest_images.csv'
    prediction_path = args.input_dir / 'predictions.csv'
    manifest = read_csv(manifest_path, {'image_id', 'category', 'split', 'image_path', 'lighting', 'mask_path'})
    predictions = read_csv(prediction_path, {'run_id', 'image_id', 'image_score', 'map_path'})
    issues = audit(manifest, predictions)
    if issues:
        raise ValueError('Input audit failed; no results written:\n' + '\n'.join(issues))
    lookup = {m['image_id']: m for m in manifest}
    joined = [{**lookup[p['image_id']], **p, 'y_true': label(p['image_id'])} for p in predictions]
    quality = []
    for run in sorted(RUNS):
        for light in ['all'] + sorted(LIGHTS):
            subset = [r for r in joined if r['run_id'] == run and r['split'] == 'test_public'
                      and (light == 'all' or r['lighting'] == light)]
            ys = [r['y_true'] for r in subset]
            scores = [float(r['image_score']) for r in subset]
            auc = auc_rank(ys, scores)
            if not math.isclose(auc, auc_pairs(ys, scores), abs_tol=1e-12):
                raise ValueError('Independent AUROC implementations disagree')
            quality.append(dict(run_id=run, category='vial', split='test_public', lighting_group=light,
                                normal_count=len(ys)-sum(ys), anomaly_count=sum(ys), image_auroc=auc,
                                threshold='', false_positives='', false_positive_rate='',
                                small_detected='', small_count='', small_recall='',
                                missing_reason='threshold_not_agreed; masks_not_received; small_definition_not_agreed'))
    overall = [r for r in quality if r['lighting_group'] == 'all']
    comparisons = [{**r, 'reference_auroc': REFERENCE[r['run_id']],
                    'matches_reference_4dp': round(r['image_auroc'], 4) == REFERENCE[r['run_id']]} for r in overall]
    # Repeated specimen prefixes may indicate matched lighting variants, not independent specimens.
    test_rows = [m for m in manifest if m['split'] == 'test_public']
    potential_units = Counter(m['image_id'].rsplit('/', 1)[0] + '/' + m['image_id'].rsplit('/', 1)[1].removesuffix('_' + m['lighting']) for m in test_rows)
    report = dict(status='passed', scope='CSV structure/counts/joins and image AUROC only',
                  manifest_rows=len(manifest), prediction_rows=len(predictions),
                  run_counts=dict(Counter(p['run_id'] for p in predictions)),
                  sources={p.name: sha256(p) for p in (manifest_path, prediction_path)},
                  script_sha256=sha256(__file__), python=platform.python_version(),
                  command=sys.argv, auroc_backend='stdlib average-rank; independently checked by pair counting',
                  anomalib_verified=False, comparisons=comparisons,
                  accessible_image_paths=sum(Path(m['image_path']).is_file() for m in manifest),
                  accessible_bad_mask_paths=sum(Path(m['mask_path']).is_file() for m in test_rows if label(m['image_id'])),
                  accessible_map_paths=sum(Path(p['map_path']).is_file() for p in predictions),
                  candidate_specimen_prefix_counts=dict(potential_units),
                  limitations=['Mask/map shape and preservation claims not independently verified.',
                               'Lighting fields match filenames; acquisition metadata not independently verified.',
                               'Candidate specimen prefixes require confirmation before independent-sample inference.',
                               'No threshold, small-defect recall, pixel metrics, or efficiency claims.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'quality.csv', quality)
    write_csv(args.output_dir / 'joined_predictions.csv', joined)
    (args.output_dir / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
