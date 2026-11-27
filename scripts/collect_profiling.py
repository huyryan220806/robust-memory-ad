"""Validate Huy's L4 log folders and collect reproducible efficiency tables.

This tool reads logs; it never runs a model, training, or Colab. No final table
is created until all three runs meet the confirmed pilot01 protocol.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import shutil

from scripts.baseline_checks import read_reference_scores, sha256
from scripts.join_quality import join_run_level, read_csv
from scripts.profile import quantile, write_csv, write_json

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / 'configs/profiling/pilot01-l4.json'
RAW_FILES = ['config.json', 'environment.json', 'artifacts.json', 'images.json',
             'validation.json', 'warmup.csv', 'timings.csv', 'rss_samples.csv',
             'memory.json', 'efficiency.csv', 'stability.json', 'status.json']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    result = float(value)
    require(math.isfinite(result), f'{name}: expected a finite number')
    return result


def integer(value, name):
    require(not isinstance(value, bool) and re.fullmatch(r'\d+', str(value)) is not None,
            f'{name}: expected a nonnegative integer')
    return int(value)


def close(actual, expected, name):
    require(math.isclose(number(actual, name), expected, rel_tol=0, abs_tol=1e-9),
            f'{name}: summary differs from raw data')


def lighting(image_id):
    parts = image_id.split('/')
    require(len(parts) == 4 and parts[:2] == ['vial', 'test_public'] and parts[2] in {'good', 'bad'},
            f'Image outside vial/test_public: {image_id}')
    return parts[3].split('_', 1)[-1]


def check_distribution(ids, protocol):
    require(len(ids) == len(set(ids)) == protocol['images'], 'Need 140 distinct test_public images')
    groups = Counter(lighting(i) for i in ids)
    require(groups == Counter({group: protocol['images_per_lighting'] for group in protocol['lighting_groups']}),
            'Need all seven lighting conditions, 20 images each')
    require(sum('/good/' in i for i in ids) == 35 and sum('/bad/' in i for i in ids) == 105,
            'Need 35 good and 105 bad test_public images')
    require(all(sum('/good/' in i and lighting(i) == group for i in ids) == 5 for group in groups),
            'Need five good specimens per lighting condition')
    return dict(groups)


def check_rows(rows, config, ids, count):
    require(len(rows) == count, f'Expected {count} timing rows')
    values = []
    for i, row in enumerate(rows):
        for key in ['run_id', 'measurement_id', 'device', 'image_group']:
            require(row[key] == config[key], f'Timing identity differs: {key}')
        require(integer(row['iteration'], 'iteration') == i, 'Timing iteration order differs')
        require(row['image_id'] == ids[i % len(ids)], 'Timing image order differs from config')
        value = number(row['latency_ms'], 'latency_ms')
        require(value > 0, 'Latency must be positive milliseconds')
        values.append(value)
    return values


def check_memory(memory, samples):
    require(len(samples) >= 2, 'Need RSS baseline and final samples')
    times = [integer(r['time_ns'], 'RSS time_ns') for r in samples]
    require(all(a < b for a, b in zip(times, times[1:])), 'RSS timestamps are not monotonic')
    rss_values = [integer(r['rss_bytes'], 'rss_bytes') for r in samples]
    require(min(rss_values) > 0, 'RSS values must be positive')
    rss = memory['rss']
    baseline, peak = rss_values[0], max(rss_values)
    require(integer(rss['baseline_bytes'], 'RSS baseline') == baseline and
            integer(rss['peak_observed_bytes'], 'RSS peak') == peak and
            integer(rss['peak_observed_increment_bytes'], 'RSS increment') == peak - baseline,
            'RSS memory summary differs from raw samples')
    require(rss['interval_s'] == .005, 'Expected 5 ms RSS sampling')
    gpu = memory['torch_cuda']
    require(isinstance(gpu, dict), 'Missing PyTorch CUDA allocator counters')
    counters = {k: integer(gpu[k], k) for k in ['baseline_allocated_bytes', 'baseline_reserved_bytes',
                'peak_allocated_bytes', 'peak_reserved_bytes']}
    require(counters['baseline_allocated_bytes'] > 0 and
            counters['peak_allocated_bytes'] >= counters['baseline_allocated_bytes'] and
            counters['peak_reserved_bytes'] >= counters['baseline_reserved_bytes'] and
            counters['peak_reserved_bytes'] >= counters['peak_allocated_bytes'] and
            counters['baseline_reserved_bytes'] >= counters['baseline_allocated_bytes'],
            'Inconsistent PyTorch allocated/reserved counters')
    require('PyTorch' in gpu['scope'] and 'warm-up' in gpu['reset'], 'Missing allocator scope/reset declaration')
    return {'rss_baseline_bytes': baseline, 'rss_peak_observed_bytes': peak,
            'rss_peak_increment_bytes': peak - baseline,
            **{'torch_' + k: v for k, v in counters.items()},
            'torch_peak_allocated_increment_bytes': counters['peak_allocated_bytes'] - counters['baseline_allocated_bytes'],
            'torch_peak_reserved_increment_bytes': counters['peak_reserved_bytes'] - counters['baseline_reserved_bytes']}


def validate_run(folder, run_id, references, protocol):
    folder = Path(folder)
    for name in RAW_FILES:
        path = folder / name
        require(path.is_file() and not path.is_symlink(), f'Missing or linked raw file: {path}')
    data = {name[:-5]: json.loads((folder / name).read_text(encoding='utf-8'))
            for name in RAW_FILES if name.endswith('.json')}
    cfg, env, summary = data['config'], data['environment'], read_csv(folder / 'efficiency.csv')
    require(data['status']['status'] == 'complete', f'{run_id}: run did not complete')
    require(cfg.get('kind', '').startswith('model') and data['status']['kind'] == cfg['kind'], 'Not a model-inference run')
    require(len(summary) == 1, 'Each run must have exactly one efficiency row')
    summary = summary[0]
    require(cfg['run_id'] == run_id, 'Run directory and config run_id differ')
    for key in ['device', 'require_l4', 'batch_size', 'dtype', 'seed', 'resolution', 'baseline_commit',
                'patchcore_commit', 'score_atol', 'score_rtol', 'warmup', 'query']:
        require(cfg[key] == protocol[key], f'{run_id}: protocol field differs: {key}')
    require(cfg['iterations'] == protocol['images'] and cfg['baseline_commit_verified'] is True,
            'Wrong iteration count or unverified baseline')
    spec = protocol['runs'][run_id]
    require(cfg['num_patches'] == cfg['budget'] == spec['num_patches'], 'Wrong patch budget')
    boundary = cfg['timing_boundary'].lower()
    require(all(word in boundary for word in ['decoded', 'rgb', 'ram', 'score', 'map']), 'Missing confirmed timing boundary')
    require(re.fullmatch(r'[0-9a-f]{40}', cfg['profiling_commit']) is not None and
            env['profiling_commit'] == cfg['profiling_commit'], 'Profiling commit differs from environment')
    require(env['profile_source_sha256'] == protocol['profiler_source_sha256'], 'Unknown profiler source version')
    gpu_names = [line.split(',')[0].strip() for line in str(env['nvidia_smi']).splitlines()]
    require('NVIDIA L4' in gpu_names, 'Environment does not identify NVIDIA L4')
    require(integer(env['ram_total_bytes'], 'RAM total') > 0, 'Missing system RAM metadata')
    require(env['packages'].get('torch') and env['packages'].get('psutil'), 'Missing runtime package versions')
    ids = [r['image_id'] for r in cfg['images']]
    groups = check_distribution(ids, protocol)
    expected_ids = {iid for rid, iid in references if rid == run_id and iid.startswith('vial/test_public/')}
    require(set(ids) == expected_ids, 'Measured image set differs from handed-over predictions')
    images = data['images']
    require([r['image_id'] for r in images] == ids and
            all(re.fullmatch(r'[0-9a-f]{64}', r['sha256']) for r in images), 'Missing image hashes/order')
    checks = data['validation']
    require([r['image_id'] for r in checks] == ids, 'Need reference validation for all 140 images')
    for row, image_cfg in zip(checks, cfg['images']):
        expected = references[(run_id, row['image_id'])]
        require(row['expected'] == image_cfg['expected_score'] == expected, 'Expected score differs from predictions')
        require(row['passed'] is True and abs(number(row['actual'], 'actual score') - expected) <= protocol['score_atol'],
                f'Reference score mismatch: {row["image_id"]}')
    artifacts = data['artifacts']
    for role, size, digest in [('bank_index', spec['index_bytes'], spec['index_sha256']),
                               ('metadata', protocol['params_bytes'], protocol['params_sha256'])]:
        candidates = [r for r in artifacts['files'] if r['role'] == role]
        require(len(candidates) == 1 and candidates[0]['bytes'] == size and candidates[0]['sha256'] == digest,
                f'Wrong or ambiguous bank artifact: {role}')
    require(artifacts['bank_index_bytes'] == spec['index_bytes'], 'Wrong artifact byte total')
    close(artifacts['bank_index_MB'], spec['index_bytes'] / 1e6, 'Bank MB')
    samples = check_rows(read_csv(folder / 'timings.csv'), cfg, ids, protocol['images'])
    check_rows(read_csv(folder / 'warmup.csv'), cfg, ids, protocol['warmup'])
    counters = check_memory(data['memory'], read_csv(folder / 'rss_samples.csv'))
    for key in ['run_id', 'measurement_id', 'image_group', 'device', 'kind', 'baseline_commit']:
        require(summary[key] == cfg[key], f'Efficiency identity differs: {key}')
    require(integer(summary['n'], 'n') == len(samples) and integer(summary['warmup'], 'warmup') == cfg['warmup'],
            'Efficiency sample count differs from logs')
    require(integer(summary['bank_index_bytes'], 'bank byte') == spec['index_bytes'] and
            integer(summary['num_patches'], 'patch count') == spec['num_patches'] and
            integer(summary['budget'], 'budget') == spec['num_patches'] and
            integer(summary['seed'], 'seed') == 0, 'Efficiency bank metadata differs')
    close(summary['bank_index_MB'], spec['index_bytes'] / 1e6, 'Efficiency bank MB')
    close(summary['p50_ms'], quantile(samples, .5), 'p50_ms')
    close(summary['p95_ms'], quantile(samples, .95), 'p95_ms')
    require(number(summary['p50_ms'], 'p50') <= number(summary['p95_ms'], 'p95'), 'p50 exceeds p95')
    require(integer(summary['rss_peak_observed_bytes'], 'RSS summary') == counters['rss_peak_observed_bytes'] and
            integer(summary['torch_peak_allocated_bytes'], 'CUDA summary') == counters['torch_peak_allocated_bytes'],
            'Efficiency memory differs from memory.json')
    require(summary['timings_file'] == 'timings.csv' and summary['memory_file'] == 'memory.json', 'Unexpected log links')
    close(data['stability']['first_half_p95_ms'], quantile(samples[:70], .95), 'First-half p95')
    close(data['stability']['second_half_p95_ms'], quantile(samples[70:], .95), 'Second-half p95')
    result = {**summary, 'category': protocol['category'], 'split': protocol['split'],
              'device_name': protocol['device_name'], 'profiling_scope': 'run',
              'protocol_id': protocol['protocol_id'], **counters,
              'params_bytes': protocol['params_bytes'],
              'deployment_bank_and_params_bytes': spec['index_bytes'] + protocol['params_bytes'],
              'timings_file': f'raw/{run_id}/timings.csv', 'memory_file': f'raw/{run_id}/memory.json',
              'environment_file': f'raw/{run_id}/environment.json', 'hardware_file': 'raw/hardware.txt'}
    return {'summary': result, 'ids': ids, 'image_hashes': {r['image_id']: r['sha256'] for r in images},
            'runtime_signature': {key: env[key] for key in ['cpu', 'os', 'ram_total_bytes', 'packages', 'nvidia_smi']},
            'lighting_counts': groups, 'raw_hashes': {name: sha256(folder / name) for name in RAW_FILES}}


def collect(logs_root, predictions, output, quality=None, quality_commit=None, protocol=None):
    protocol = protocol or json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    logs_root, output = Path(logs_root), Path(output)
    require(not output.exists(), 'Use a new output directory; previous collections are retained')
    require(sha256(predictions) == protocol['predictions_sha256'], 'Predictions hash differs from the handover')
    references = read_reference_scores(predictions)
    hardware = logs_root / 'hardware.txt'
    require(hardware.is_file() and not hardware.is_symlink(), 'Include Huy\'s top-level hardware.txt')
    require(re.search(r'torch CUDA\s+\d+\.\d+', hardware.read_text(encoding='utf-8')) is not None,
            'Hardware log lacks a real CUDA version')
    validated = [validate_run(logs_root / rid, rid, references, protocol) for rid in protocol['runs']]
    require(len({v['summary']['measurement_id'] for v in validated}) == len(validated), 'Measurement IDs are not distinct')
    require(all(v['ids'] == validated[0]['ids'] and v['image_hashes'] == validated[0]['image_hashes'] for v in validated),
            'Image order/content differs across configurations')
    require(all(v['runtime_signature'] == validated[0]['runtime_signature'] for v in validated),
            'Runtime metadata differs across configurations; keep sessions separate')
    summaries = [v['summary'] for v in validated]
    joined = join_run_level(summaries, read_csv(quality)) if quality else None
    # Validate everything before creating final artifacts. Copy only the fixed log filenames.
    output.mkdir(parents=True, exist_ok=False)
    try:
        for rid, checked in zip(protocol['runs'], validated):
            target = output / 'raw' / rid
            target.mkdir(parents=True)
            for name in RAW_FILES:
                shutil.copyfile(logs_root / rid / name, target / name)
                require(sha256(target / name) == checked['raw_hashes'][name], 'Raw log changed while collecting')
        shutil.copyfile(hardware, output / 'raw/hardware.txt')
        write_csv(output / 'efficiency.csv', summaries)
        if joined is not None:
            shutil.copyfile(quality, output / 'quality-input.csv')
            write_csv(output / 'quality-efficiency.csv', joined)
        warnings = ['RSS is sampled at 5 ms and may miss brief peaks; allocator counters are not total GPU process VRAM.',
                    'The pinned notebook sets TF32 off; the legacy environment log does not independently record those flags.',
                    'Review p95 stability and repeat in a new measurement_id when necessary.']
        audit = {'status': 'complete', 'protocol': protocol, 'predictions_sha256': sha256(predictions),
                 'hardware_sha256': sha256(hardware), 'run_ids': list(protocol['runs']),
                 'samples_per_run': protocol['images'], 'measurement_count': len(summaries),
                 'raw_hashes': {rid: v['raw_hashes'] for rid, v in zip(protocol['runs'], validated)},
                 'quality_sha256': sha256(quality) if quality else None, 'quality_commit': quality_commit,
                 'joined_rows': len(joined) if joined is not None else 0,
                 'aggregation_performed': False, 'warnings': warnings}
        write_json(output / 'collection-audit.json', audit)
        write_json(output / 'collection-status.json', {'status': 'complete'})
    except Exception as exc:
        write_json(output / 'collection-status.json', {'status': 'failed', 'error': str(exc)})
        raise
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs-root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quality', type=Path)
    parser.add_argument('--quality-commit')
    args = parser.parse_args()
    audit = collect(args.logs_root, args.predictions, args.output, args.quality, args.quality_commit)
    print(json.dumps({'status': audit['status'], 'measurement_count': audit['measurement_count'],
                      'samples_per_run': audit['samples_per_run'], 'joined_rows': audit['joined_rows']}, indent=2))


if __name__ == '__main__':
    main()
