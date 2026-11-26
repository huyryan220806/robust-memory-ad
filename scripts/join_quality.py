"""Expand run-level efficiency next to each distinct quality protocol row.

No metric is averaged or summed. Repeated efficiency is one measurement,
not additional observations. Blank metrics and missing_reason stay unchanged.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

QUALITY_KEYS = ['run_id', 'category', 'split', 'lighting_group',
                'target_calibration_fpr', 'threshold', 'small_area_cutoff']
MEASUREMENT_KEYS = ['run_id', 'measurement_id', 'device']


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    return rows


def identity(row, fields, numeric=()):
    values = []
    for field in fields:
        value = row.get(field, '').strip()
        if not value:
            raise ValueError(f'Missing identity column/value: {field}')
        if field in numeric:
            try:
                value = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f'Invalid identity number: {field}') from exc
            if not value.is_finite():
                raise ValueError(f'Non-finite identity number: {field}')
        values.append(value)
    return tuple(values)


def join_run_level(efficiency, quality):
    by_run, quality_seen = {}, set()
    for row in quality:
        key = identity(row, QUALITY_KEYS, QUALITY_KEYS[-3:])
        if key in quality_seen:
            raise ValueError(f'Duplicate quality identity: {key}')
        quality_seen.add(key)
        by_run.setdefault(row['run_id'], []).append(row)
    output, measurement_seen = [], set()
    for measurement in efficiency:
        key = identity(measurement, MEASUREMENT_KEYS)
        if key in measurement_seen:
            raise ValueError(f'Duplicate efficiency measurement: {key}')
        measurement_seen.add(key)
        if measurement.get('kind') == 'instrumentation_smoke':
            raise ValueError('Smoke instrumentation is not model efficiency')
        candidates = by_run.get(measurement['run_id'])
        if not candidates:
            raise ValueError(f'No quality rows for {measurement["run_id"]}')
        for quality_row in candidates:
            if measurement.get('category') and measurement['category'] != quality_row['category']:
                raise ValueError('Efficiency category differs from quality category')
            merged = dict(quality_row)
            for name, value in measurement.items():
                if name == 'run_id':
                    continue
                # Preserve quality's exact identity columns; explicitly label profiling metadata collisions.
                target = 'profiling_' + name if name in merged else name
                if target in merged:
                    raise ValueError(f'Ambiguous input column: {target}')
                merged[target] = value
            merged['profiling_scope'] = 'run'
            merged['profiling_repeated_across_quality_rows'] = str(len(candidates) > 1).lower()
            merged['profiling_aggregation_key'] = '|'.join(str(x) for x in key)
            output.append(merged)
    return output


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--efficiency', type=Path, nargs='+', required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--scope', choices=['run'], required=True,
                        help='Explicitly acknowledge run-level performance repeated across quality groups')
    parser.add_argument('--lighting-group', help='Optional explicit filter; e.g. all (never sums subgroups)')
    parser.add_argument('--quality-commit', help='Optional pinned evaluation source commit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    quality = read_csv(args.quality)
    if args.lighting_group:
        quality = [row for row in quality if row['lighting_group'] == args.lighting_group]
        if not quality:
            raise ValueError('No quality rows match the explicit lighting filter')
    efficiency = [row for path in args.efficiency for row in read_csv(path)]
    rows = join_run_level(efficiency, quality)
    audit = {'quality_sha256': file_hash(args.quality), 'quality_commit': args.quality_commit,
             'efficiency_inputs': [{'path': str(path), 'sha256': file_hash(path)} for path in args.efficiency],
             'input_measurements': len(efficiency), 'selected_quality_rows': len(quality),
             'joined_rows': len(rows), 'profiling_scope': args.scope,
             'quality_identity': QUALITY_KEYS, 'measurement_identity': MEASUREMENT_KEYS,
             'aggregation_performed': False,
             'note': 'Count each measurement once; never sum lighting_group=all with subgroup rows.'}
    audit_path = args.output.with_suffix(args.output.suffix + '.audit.json')
    if args.output.exists() or audit_path.exists():
        raise FileExistsError('Output or audit already exists; use a new filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(name for row in rows for name in row)))
        writer.writeheader()
        writer.writerows(rows)
    with audit_path.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
