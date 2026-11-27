import unittest
from scripts.join_quality import join_run_level


def quality(**changes):
    return {'run_id': 'pilot01-full-s0', 'category': 'vial', 'split': 'test_public',
            'lighting_group': 'all', 'target_calibration_fpr': '0.01', 'threshold': '4.2',
            'small_area_cutoff': '0.001', 'small_recall': '',
            'missing_reason': 'no_eligible_small_images', 'status': 'exploratory_pending_team_review', **changes}


def efficiency(**changes):
    return {'run_id': 'pilot01-full-s0', 'measurement_id': 'l4-session-1',
            'device': 'cuda:0', 'p50_ms': '10', 'kind': 'model', **changes}


class QualityJoinTests(unittest.TestCase):
    def test_expands_protocol_rows_and_keeps_blanks(self):
        rows = join_run_level([efficiency()], [quality(), quality(lighting_group='regular'),
                             quality(small_area_cutoff='0.005'), quality(target_calibration_fpr='0.05')])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row['small_recall'] == '' for row in rows))
        self.assertTrue(all(row['missing_reason'] == 'no_eligible_small_images' for row in rows))
        self.assertTrue(all(row['status'] == 'exploratory_pending_team_review' for row in rows))
        self.assertEqual(len({row['profiling_aggregation_key'] for row in rows}), 1)

    def test_sessions_and_devices_are_not_collapsed(self):
        rows = join_run_level([efficiency(), efficiency(measurement_id='cpu-session-1', device='cpu')], [quality()])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['device'] for row in rows}, {'cpu', 'cuda:0'})

    def test_duplicate_quality_identity_is_rejected_even_with_equivalent_numbers(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate quality'):
            join_run_level([efficiency()], [quality(), quality(target_calibration_fpr='1e-2')])

    def test_missing_run_duplicate_measurement_and_smoke_fail(self):
        for measurements, rows, error in [([efficiency(run_id='missing')], [quality()], 'No quality'),
                                          ([efficiency(), efficiency()], [quality()], 'Duplicate efficiency'),
                                          ([efficiency(kind='instrumentation_smoke')], [quality()], 'Smoke')]:
            with self.assertRaisesRegex(ValueError, error):
                join_run_level(measurements, rows)

    def test_quality_identity_survives_profiling_metadata_collision(self):
        result = join_run_level([efficiency(split='eval_all')], [quality()])[0]
        self.assertEqual(result['split'], 'test_public')
        self.assertEqual(result['profiling_split'], 'eval_all')
