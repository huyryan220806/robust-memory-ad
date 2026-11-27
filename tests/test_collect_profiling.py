"""Synthetic log fixtures only: these tests do not run or measure a GPU."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.collect_profiling import PROTOCOL_PATH, collect
from scripts.profile import quantile, write_csv, write_json


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    write_json(path, data)


def fixtures(root):
    protocol = json.loads(PROTOCOL_PATH.read_text())
    groups = protocol['lighting_groups']
    ids = sorted(f'vial/test_public/{kind}/{i:03}_{group}'
                 for kind, n in [('good', 5), ('bad', 15)] for i in range(n) for group in groups)
    predictions, quality = root / 'predictions.csv', root / 'quality.csv'
    refs = [{'run_id': rid, 'image_id': iid, 'image_score': 1 + i / 1000}
            for rid in protocol['runs'] for i, iid in enumerate(ids)]
    write_csv(predictions, refs)
    protocol['predictions_sha256'] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    write_csv(quality, [{'run_id': rid, 'category': 'vial', 'split': 'test_public',
               'lighting_group': group, 'target_calibration_fpr': '.05', 'threshold': '2',
               'small_area_cutoff': '.005', 'small_recall': '', 'missing_reason': 'test_missing',
               'status': 'synthetic_test_fixture'} for rid in protocol['runs'] for group in ['all', 'regular']])
    logs = root / 'logs'
    logs.mkdir()
    (logs / 'hardware.txt').write_text('Synthetic test CPU\ntorch CUDA 12.6\n')
    for rid, spec in protocol['runs'].items():
        folder = logs / rid
        folder.mkdir()
        cfg = {k: protocol[k] for k in ['device', 'require_l4', 'batch_size', 'dtype', 'seed', 'resolution',
                'baseline_commit', 'patchcore_commit', 'score_atol', 'score_rtol', 'warmup', 'query']}
        cfg.update(run_id=rid, kind='model_inference', measurement_id='fixture-' + rid,
                   image_group='test_public_all', iterations=140, baseline_commit_verified=True,
                   profiling_commit='1' * 40, num_patches=spec['num_patches'], budget=spec['num_patches'],
                   timing_boundary='decoded RGB in RAM to score and smoothed map in RAM',
                   images=[{'image_id': iid, 'expected_score': 1 + i / 1000} for i, iid in enumerate(ids)])
        write_json(folder / 'config.json', cfg)
        write_json(folder / 'environment.json', {'profiling_commit': '1' * 40,
                   'profile_source_sha256': protocol['profiler_source_sha256'],
                   'nvidia_smi': 'NVIDIA L4, 23034 MiB, synthetic-test-driver', 'cpu': 'synthetic-test-cpu',
                   'os': 'synthetic-test-os', 'ram_total_bytes': 8 * 1024**3,
                   'packages': {'torch': 'synthetic-test', 'psutil': 'synthetic-test'}})
        write_json(folder / 'images.json', [{'image_id': iid, 'sha256': hashlib.sha256(iid.encode()).hexdigest()} for iid in ids])
        write_json(folder / 'artifacts.json', {'bank_index_bytes': spec['index_bytes'],
                   'bank_index_MB': spec['index_bytes'] / 1e6, 'files': [
                   {'role': 'bank_index', 'bytes': spec['index_bytes'], 'sha256': spec['index_sha256']},
                   {'role': 'metadata', 'bytes': protocol['params_bytes'], 'sha256': protocol['params_sha256']}]})
        write_json(folder / 'validation.json', [{'image_id': iid, 'expected': 1 + i / 1000,
                   'actual': 1 + i / 1000, 'passed': True} for i, iid in enumerate(ids)])
        identity = {k: cfg[k] for k in ['run_id', 'measurement_id', 'image_group', 'device']}
        measured = [{**identity, 'image_id': iid, 'iteration': i, 'latency_ms': float(i + 1)} for i, iid in enumerate(ids)]
        write_csv(folder / 'timings.csv', measured)
        write_csv(folder / 'warmup.csv', measured[:10])
        write_csv(folder / 'rss_samples.csv', [{'time_ns': i * 100, 'rss_bytes': rss}
                  for i, rss in enumerate([1000, 1100, 1050], 1)])
        write_json(folder / 'memory.json', {'rss': {'baseline_bytes': 1000, 'peak_observed_bytes': 1100,
                   'peak_observed_increment_bytes': 100, 'interval_s': .005},
                   'torch_cuda': {'baseline_allocated_bytes': 100, 'baseline_reserved_bytes': 200,
                   'peak_allocated_bytes': 150, 'peak_reserved_bytes': 250,
                   'scope': 'PyTorch allocator only', 'reset': 'after warm-up, before measured loop'}})
        samples = [r['latency_ms'] for r in measured]
        write_csv(folder / 'efficiency.csv', [{**identity, 'kind': cfg['kind'],
                  'baseline_commit': cfg['baseline_commit'], 'seed': 0, 'budget': spec['num_patches'],
                  'num_patches': spec['num_patches'], 'bank_index_bytes': spec['index_bytes'],
                  'bank_index_MB': spec['index_bytes'] / 1e6, 'n': 140, 'warmup': 10,
                  'p50_ms': quantile(samples, .5), 'p95_ms': quantile(samples, .95),
                  'rss_peak_observed_bytes': 1100, 'torch_peak_allocated_bytes': 150,
                  'timings_file': 'timings.csv', 'memory_file': 'memory.json'}])
        write_json(folder / 'stability.json', {'first_half_p95_ms': quantile(samples[:70], .95),
                   'second_half_p95_ms': quantile(samples[70:], .95)})
        write_json(folder / 'status.json', {'status': 'complete', 'kind': cfg['kind']})
    return protocol, logs, predictions, quality


class CollectTests(unittest.TestCase):
    def test_collect_recomputes_and_keeps_raw_and_missing_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, logs, predictions, quality = fixtures(root)
            out = root / 'collected'
            audit = collect(logs, predictions, out, quality=quality, protocol=protocol)
            self.assertEqual(audit['measurement_count'], 3)
            self.assertEqual(audit['samples_per_run'], 140)
            self.assertEqual(audit['joined_rows'], 6)
            self.assertFalse(audit['aggregation_performed'])
            with (out / 'efficiency.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['torch_peak_reserved_bytes'] == '250' for r in rows))
            for row in rows:
                self.assertEqual((out / row['timings_file']).read_bytes(),
                                 (logs / row['run_id'] / 'timings.csv').read_bytes())
            with (out / 'quality-efficiency.csv').open() as handle:
                self.assertTrue(all(row['small_recall'] == '' and row['missing_reason'] == 'test_missing'
                                    for row in csv.DictReader(handle)))
            with self.assertRaisesRegex(ValueError, 'new output'):
                collect(logs, predictions, out, protocol=protocol)

    def test_protocol_drift_is_rejected_before_output(self):
        for field, value in [('query', 'FAISS GPU'), ('device', 'cpu'), ('batch_size', 2),
                             ('num_patches', 2281), ('iterations', 50), ('score_rtol', .0001)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                protocol, logs, predictions, _ = fixtures(root)
                edit_json(logs / 'pilot01-full-s0/config.json', lambda cfg: cfg.update({field: value}))
                with self.assertRaises(ValueError):
                    collect(logs, predictions, root / 'out', protocol=protocol)
                self.assertFalse((root / 'out').exists())

    def test_failed_run_is_not_accepted_with_stale_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, logs, predictions, _ = fixtures(root)
            edit_json(logs / 'pilot01-full-s0/status.json', lambda s: s.update(status='failed'))
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                collect(logs, predictions, root / 'out', protocol=protocol)

    def test_passed_flag_does_not_hide_a_score_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, logs, predictions, _ = fixtures(root)
            edit_json(logs / 'pilot01-full-s0/validation.json', lambda s: s[0].update(actual=9))
            with self.assertRaisesRegex(ValueError, 'score mismatch'):
                collect(logs, predictions, root / 'out', protocol=protocol)

    def test_image_hashes_and_runtime_must_match_across_runs(self):
        for filename, change, error in [
                ('images.json', lambda s: s[0].update(sha256='0' * 64), 'Image order/content'),
                ('environment.json', lambda s: s.update(cpu='different-cpu'), 'Runtime metadata'),
                ('environment.json', lambda s: s.update(nvidia_smi='NVIDIA L40S, 46068 MiB, test'), 'does not identify')]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                protocol, logs, predictions, _ = fixtures(root)
                edit_json(logs / 'pilot01-random-s0' / filename, change)
                with self.assertRaisesRegex(ValueError, error):
                    collect(logs, predictions, root / 'out', protocol=protocol)

    def test_missing_samples_and_fabricated_summary_are_rejected(self):
        for filename, edit in [('timings.csv', 'drop'), ('efficiency.csv', 'p95')]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                protocol, logs, predictions, _ = fixtures(root)
                p = logs / 'pilot01-full-s0' / filename
                with p.open() as handle:
                    rows = list(csv.DictReader(handle))
                if edit == 'drop':
                    rows.pop()
                else:
                    rows[0]['p95_ms'] = '1'
                write_csv(p, rows)
                with self.assertRaises(ValueError):
                    collect(logs, predictions, root / 'out', protocol=protocol)

    def test_memory_peaks_are_checked_against_raw_samples(self):
        for change, error in [
                (lambda s: s['rss'].update(peak_observed_bytes=9999), 'RSS memory summary'),
                (lambda s: s['torch_cuda'].update(baseline_allocated_bytes=0, baseline_reserved_bytes=0,
                                                peak_allocated_bytes=0, peak_reserved_bytes=0), 'PyTorch')]:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                protocol, logs, predictions, _ = fixtures(root)
                edit_json(logs / 'pilot01-full-s0/memory.json', change)
                with self.assertRaisesRegex(ValueError, error):
                    collect(logs, predictions, root / 'out', protocol=protocol)

    def test_predictions_must_be_the_handed_over_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, logs, predictions, _ = fixtures(root)
            predictions.write_text('run_id,image_id,image_score\nwrong,wrong,1\n')
            with self.assertRaisesRegex(ValueError, 'Predictions hash'):
                collect(logs, predictions, root / 'out', protocol=protocol)
