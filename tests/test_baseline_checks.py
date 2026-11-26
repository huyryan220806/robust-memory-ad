import csv
import math
import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.baseline_checks import absolute_score_check, image_path, read_plain_params, read_reference_scores


class BaselineChecks(unittest.TestCase):
    def test_tolerance_is_absolute_and_inclusive(self):
        self.assertTrue(absolute_score_check(.0001, 0, .0001)['passed'])
        self.assertFalse(absolute_score_check(100.0002, 100, .0001)['passed'])
        for score in [math.inf, math.nan]:
            with self.assertRaises(ValueError):
                absolute_score_check(score, 0, .001)

    def test_identifiers_are_portable_and_cannot_escape_root(self):
        self.assertEqual(image_path('data', 'vial/test_public/bad/011_overexposed'),
                         Path('data/vial/test_public/bad/011_overexposed.png'))
        for iid in ['../vial/bad/test', 'vial/test_public/../test', 'C:/data/bad/test', 'vial\\test_public/bad/image']:
            with self.assertRaises(ValueError):
                image_path('data', iid)

    def test_duplicate_and_nonfinite_reference_scores_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'scores.csv'
            p.write_text('run_id,image_id,image_score\na,b,1\na,b,2\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_reference_scores(p)
            p.write_text('run_id,image_id,image_score\na,b,nan\n')
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                read_reference_scores(p)

    def test_params_reject_objects_before_upstream_unpickle(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'params.pkl'
            p.write_bytes(pickle.dumps(Path('untrusted')))
            with self.assertRaisesRegex(ValueError, 'plain values'):
                read_plain_params(p)
