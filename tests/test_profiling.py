import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.profile import SmokeAdapter, inventory, measure_selection, quantile, run, timed_call


def config():
    return {'kind': 'instrumentation_smoke', 'run_id': 'pilot-test', 'measurement_id': 'session-1',
            'image_group': 'toy', 'device': 'cpu', 'baseline_commit': 'not-a-model',
            'dtype': 'int', 'resolution': 'toy', 'query': 'toy', 'timing_boundary': 'toy',
            'bank_layout': 'none', 'batch_size': 1, 'warmup': 2, 'iterations': 5,
            'images': [{'image_id': 'a', 'value': 1, 'expected_score': 79974},
                       {'image_id': 'b', 'value': 2, 'expected_score': 159948}], 'artifacts': []}


class ProfilingTests(unittest.TestCase):
    def test_quantiles(self):
        self.assertEqual(quantile([8, 9, 10, 11, 12], .5), 10)
        self.assertAlmostEqual(quantile([8, 9, 10, 11, 12], .95), 11.8)
        self.assertEqual(quantile([3], .95), 3)
        for values in [[], [-1], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                quantile(values, .95)

    def test_synchronization_surrounds_clock_and_operation(self):
        events = []
        def clock():
            events.append('clock')
            return len(events) * 1_000_000
        with patch('scripts.profile.time.perf_counter_ns', clock):
            result, elapsed = timed_call(lambda: events.append('predict') or 42,
                                         lambda: events.append('sync'))
        self.assertEqual(events, ['sync', 'clock', 'predict', 'sync', 'clock'])
        self.assertEqual(result, 42)
        self.assertEqual(elapsed, 3)

    def test_full_run_preserves_rows_and_excludes_warmup(self):
        class CountingAdapter(SmokeAdapter):
            loads = 0
            calls = 0
            def load(self, config, base):
                self.loads += 1
            def predict(self, image):
                self.calls += 1
                return super().predict(image)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            adapter = CountingAdapter()
            summary = run(config(), adapter, output, Path(directory))
            self.assertEqual(adapter.loads, 1)
            self.assertEqual(adapter.calls, 2 + 2 + 5)  # reference checks + warmup + measured
            with (output / 'timings.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            with (output / 'warmup.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            self.assertEqual(len(rows), 5)
            self.assertEqual([x['image_id'] for x in rows], ['a', 'b', 'a', 'b', 'a'])
            values = [float(x['latency_ms']) for x in rows]
            self.assertEqual(summary['p50_ms'], quantile(values, .5))
            self.assertEqual(summary['p95_ms'], quantile(values, .95))
            with self.assertRaises(FileExistsError):
                run(config(), adapter, output, Path(directory))

    def test_score_mismatch_stops_before_timing(self):
        bad = config()
        bad['images'][0]['expected_score'] = -100
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bad'
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                run(bad, SmokeAdapter(), output, Path(directory))
            self.assertFalse((output / 'timings.csv').exists())
            self.assertEqual(json.loads((output / 'status.json').read_text())['stage'], 'reference_validation')

    def test_target_l4_never_falls_back_to_cpu(self):
        bad = config()
        bad['require_l4'] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'L4'):
                run(bad, SmokeAdapter(), Path(directory) / 'bad', Path(directory))

    def test_inventory_separates_backbone_and_intermediates(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for filename, length in [('index', 20), ('weights', 100), ('copy', 20)]:
                (base / filename).write_bytes(b'x' * length)
            entries = [{'path': 'index', 'role': 'bank_index'}, {'path': 'weights', 'role': 'backbone'},
                       {'path': 'copy', 'role': 'intermediate'}]
            result = inventory(entries, base)
            self.assertEqual(result['bank_index_bytes'], 20)
            self.assertEqual(result['bank_index_MB'], .00002)
            with self.assertRaisesRegex(ValueError, 'twice'):
                inventory([entries[0], entries[0]], base)

    def test_selection_clock_only_wraps_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            metadata = {'run_id': 'a', 'measurement_id': 'a1', 'seed': 0, 'budget': .1,
                        'input_patches': 20, 'boundary': 'features ready to selected indices'}
            selected = measure_selection(lambda: [1, 3], lambda: None, metadata, path)
            self.assertEqual(selected, [1, 3])
            result = json.loads(path.read_text())
            self.assertGreaterEqual(result['selection_ms'], 0)


if __name__ == '__main__':
    unittest.main()
