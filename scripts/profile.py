"""Batch-one profiling harness; no PatchCore implementation is assumed.

Adapter contract is documented in docs/profiling.md. Synthetic smoke output
checks the instrumentation only and must never be reported as model results.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path


def quantile(values, q):
    """Linear interpolation at (n - 1) * q (NumPy method='linear')."""
    if not values or not 0 <= q <= 1:
        raise ValueError('Need samples and 0 <= q <= 1')
    values = sorted(float(x) for x in values)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Samples must be finite nonnegative milliseconds')
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(entries, base):
    """Count explicitly declared deployed files once; keep other roles separate."""
    rows, seen = [], set()
    for entry in entries:
        if entry['role'] not in {'bank_index', 'backbone', 'intermediate', 'metadata'}:
            raise ValueError('Unknown artifact role')
        path = (base / entry['path']).resolve(strict=True)
        if path in seen:
            raise ValueError(f'Artifact listed twice: {entry["path"]}')
        seen.add(path)
        rows.append({**entry, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    total = sum(row['bytes'] for row in rows if row['role'] == 'bank_index')
    return {'files': rows, 'bank_index_bytes': total, 'bank_index_MB': total / 1_000_000}


class RSSSampler:
    """Sample this process RSS; a sampled peak can miss short-lived allocations."""
    def __init__(self, interval_s=0.005):
        import psutil
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples = []
        self.error = None

    def sample(self):
        self.samples.append({'time_ns': time.perf_counter_ns(), 'rss_bytes': self.process.memory_info().rss})

    def loop(self):
        try:
            while not self.stop_event.wait(self.interval_s):
                self.sample()
        except Exception as exc:
            self.error = exc

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        if self.error:
            raise self.error
        baseline = self.samples[0]['rss_bytes']
        peak = max(x['rss_bytes'] for x in self.samples)
        return {'tool': 'psutil.Process.memory_info().rss', 'scope': 'current process; sampled every 5 ms; inference loop plus logging overhead',
                'baseline_bytes': baseline, 'peak_observed_bytes': peak,
                'peak_observed_increment_bytes': peak - baseline, 'interval_s': self.interval_s}


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'psutil', 'faiss-cpu', 'faiss-gpu', 'Pillow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def command(args):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        gpu = command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])
    except FileNotFoundError:
        gpu = None
    import psutil
    return {'python': sys.version, 'os': platform.platform(), 'cpu': platform.processor(),
            'ram_total_bytes': psutil.virtual_memory().total, 'packages': packages,
            'nvidia_smi': gpu, 'profile_source_sha256': sha256(Path(__file__)),
            'profiling_commit': command(['git', 'rev-parse', 'HEAD']),
            'profiling_git_status': command(['git', 'status', '--porcelain']),
            'created_utc': datetime.now(timezone.utc).isoformat()}


def runtime(device, require_l4):
    if device == 'cpu':
        if require_l4:
            raise ValueError('Official target requires CUDA and NVIDIA L4')
        return (lambda: None), nullcontext, None
    if not device.startswith('cuda:'):
        raise ValueError('Device must be cpu or cuda:N')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.cuda.set_device(torch.device(device))
    name = torch.cuda.get_device_name(device)
    if require_l4 and 'L4' not in name.upper().split():
        raise RuntimeError(f'Expected L4; got {name}')
    return lambda: torch.cuda.synchronize(device), torch.inference_mode, torch


def timed_call(function, synchronize):
    synchronize()
    start = time.perf_counter_ns()
    output = function()
    synchronize()
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    return output, elapsed


def validate_config(config):
    for name in ['run_id', 'measurement_id', 'image_group', 'device', 'baseline_commit',
                 'dtype', 'resolution', 'query', 'timing_boundary', 'bank_layout']:
        if not config.get(name):
            raise ValueError(f'Missing config field: {name}')
    if config.get('batch_size') != 1:
        raise ValueError('Only batch_size=1 is supported')
    if config.get('warmup', 10) < 0 or config.get('iterations', 50) < 1:
        raise ValueError('Invalid warm-up or iteration count')
    if not config.get('images') or len({x['image_id'] for x in config['images']}) != len(config['images']):
        raise ValueError('Images must have unique image_id values')
    for row in config['images']:
        if not math.isfinite(float(row['expected_score'])):
            raise ValueError('Every image needs a finite reference image_score')
    for key in ['score_atol', 'score_rtol']:
        tolerance = config.get(key, 1e-5)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Invalid score tolerance')
    if config.get('kind') != 'instrumentation_smoke':
        if not config.get('artifacts') or not any(x['role'] == 'bank_index' for x in config['artifacts']):
            raise ValueError('Real runs need declared bank/index artifacts')
        if not config.get('baseline_commit_verified'):
            raise ValueError('Adapter must verify baseline checkout matches baseline_commit')


def run(config, adapter, output, base):
    """Load once, verify reference scores, warm up, then measure one image per row."""
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment())
    stage = 'setup'
    try:
        synchronize, inference_context, torch = runtime(config['device'], config.get('require_l4', False))
        artifacts = inventory(config.get('artifacts', []), base)
        write_json(output / 'artifacts.json', artifacts)
        # Adapter owns model.eval(), dtype, RNG seeds, preprocessing and FAISS stream handling.
        adapter.load(config, base)
        images = [(spec, adapter.decode(spec, base)) for spec in config['images']]
        image_files = [{'image_id': spec['image_id'], 'path': spec['path'], 'sha256': sha256((base / spec['path']).resolve())}
                       for spec, _ in images if 'path' in spec]
        write_json(output / 'images.json', image_files)
        def sync():
            synchronize()
            adapter.synchronize()  # Complete external FAISS/non-PyTorch work too.
        checks = []
        with inference_context():
            stage = 'reference_validation'
            for spec, image in images:
                result = adapter.predict(image)
                sync()
                score = float(adapter.image_score(result))
                passed = math.isfinite(score) and math.isclose(score, float(spec['expected_score']),
                          rel_tol=config.get('score_rtol', 1e-5), abs_tol=config.get('score_atol', 1e-5))
                checks.append({'image_id': spec['image_id'], 'expected': spec['expected_score'], 'actual': score, 'passed': passed})
                del result
            write_json(output / 'validation.json', checks)
            if not all(row['passed'] for row in checks):
                raise ValueError('Reference score mismatch: timing aborted')
            stage = 'warmup'
            for phase, count in [('warmup', config.get('warmup', 10)), ('measured', config.get('iterations', 50))]:
                rows = []
                sampler = None
                if phase == 'measured':
                    stage = 'measurement'
                    sync()
                    gpu_memory = None
                    if torch:
                        torch.cuda.reset_peak_memory_stats(config['device'])
                        gpu_memory = {'baseline_allocated_bytes': torch.cuda.memory_allocated(config['device']),
                                      'baseline_reserved_bytes': torch.cuda.memory_reserved(config['device'])}
                    sampler = RSSSampler()
                    sampler.start()
                path = output / ('timings.csv' if phase == 'measured' else 'warmup.csv')
                try:
                    with path.open('w', newline='', encoding='utf-8') as handle:
                        fields = ['run_id', 'measurement_id', 'image_group', 'device', 'image_id', 'iteration', 'latency_ms']
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        for i in range(count):
                            spec, image = images[i % len(images)]
                            result, elapsed = timed_call(lambda: adapter.predict(image), sync)
                            del result  # Release outside the timed region.
                            row = {key: config[key] for key in fields[:4]}
                            row.update(image_id=spec['image_id'], iteration=i, latency_ms=elapsed)
                            rows.append(row)
                            writer.writerow(row)
                            handle.flush()
                finally:
                    if sampler:
                        memory = sampler.stop()
                        write_csv(output / 'rss_samples.csv', sampler.samples)
                        if torch:
                            gpu_memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(config['device']),
                                              peak_reserved_bytes=torch.cuda.max_memory_reserved(config['device']),
                                              scope='PyTorch allocator only; excludes external FAISS/CUDA allocations',
                                              reset='after validation and warm-up, before measured loop')
                        write_json(output / 'memory.json', {'rss': memory, 'torch_cuda': gpu_memory,
                                   'gpu_process_peak': 'not measured; external index VRAM requires process-level instrumentation'})
        samples = [row['latency_ms'] for row in rows]
        summary = {key: config[key] for key in ['run_id', 'measurement_id', 'image_group', 'device']}
        summary.update(kind=config.get('kind', 'model'), baseline_commit=config['baseline_commit'],
                       seed=config.get('seed'), budget=config.get('budget'), num_patches=config.get('num_patches'),
                       bank_index_bytes=artifacts['bank_index_bytes'], bank_index_MB=artifacts['bank_index_MB'],
                       warmup=config.get('warmup', 10), n=len(samples), p50_ms=quantile(samples, .5), p95_ms=quantile(samples, .95),
                       rss_peak_observed_bytes=memory['peak_observed_bytes'],
                       torch_peak_allocated_bytes=gpu_memory['peak_allocated_bytes'] if gpu_memory else '',
                       timings_file='timings.csv', memory_file='memory.json')
        write_csv(output / 'efficiency.csv', [summary])
        # Diagnostic windows; no automatic claim of stability from 50 observations.
        half = len(samples) // 2
        write_json(output / 'stability.json', {'first_half_p95_ms': quantile(samples[:half], .95) if half else None,
                   'second_half_p95_ms': quantile(samples[half:], .95),
                   'verdict': 'manual review required; repeat in a new measurement_id if p95 varies'})
        write_json(output / 'status.json', {'status': 'complete', 'kind': summary['kind']})
        return summary
    except Exception as exc:
        write_json(output / 'status.json', {'status': 'failed', 'stage': stage, 'error': str(exc)})
        raise


def measure_selection(select, synchronize, metadata, output):
    """Call after train features exist. select must return the chosen bank/indices."""
    for key in ['run_id', 'measurement_id', 'seed', 'budget', 'input_patches', 'boundary']:
        if key not in metadata:
            raise ValueError(f'Missing selection metadata: {key}')
    # Caller configures the RNG and prepares train features before invoking this helper.
    with Path(output).open('x', encoding='utf-8') as handle:
        selected, elapsed = timed_call(select, synchronize)
        json.dump({**metadata, 'selection_ms': elapsed, 'unit': 'ms'}, handle, indent=2, allow_nan=False)
    return selected


class SmokeAdapter:
    """Deterministic CPU toy work; not an anomaly detector."""
    def load(self, config, base):
        pass

    def decode(self, spec, base):
        return spec['value']

    def predict(self, image):
        return sum((i % 17) * image for i in range(10000))

    def image_score(self, output):
        return output

    def synchronize(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--adapter', help='importable_module:factory; factory() returns an adapter')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.config or args.adapter:
            parser.error('--smoke cannot be combined with --config/--adapter')
        config = {'kind': 'instrumentation_smoke', 'run_id': 'smoke-only', 'measurement_id': args.output.name,
                  'image_group': 'synthetic', 'device': 'cpu', 'baseline_commit': 'not-a-model',
                  'batch_size': 1, 'dtype': 'python-int', 'resolution': 'not-an-image', 'query': 'toy summation',
                  'timing_boundary': 'toy computation only', 'bank_layout': 'no bank', 'warmup': 10, 'iterations': 50,
                  'images': [{'image_id': 'toy-1', 'value': 1, 'expected_score': 79974}], 'artifacts': []}
        adapter, base = SmokeAdapter(), Path.cwd()
    else:
        if not args.config or not args.adapter:
            parser.error('Need --config and --adapter (or --smoke)')
        config = json.loads(args.config.read_text(encoding='utf-8'))
        config['adapter'] = args.adapter
        module, factory = args.adapter.split(':', 1)
        adapter = getattr(importlib.import_module(module), factory)()
        base = args.config.resolve().parent
    print(json.dumps(run(config, adapter, args.output, base), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
