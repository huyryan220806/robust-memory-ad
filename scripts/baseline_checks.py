"""Dependency-light checks shared by baseline reload and unit tests."""
import csv
import hashlib
import io
import math
import pickle
import subprocess
from pathlib import Path

PATCHCORE_COMMIT = 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
RUN_IDS = ('pilot01-full-s0', 'pilot01-random-s0', 'pilot01-greedy-s0')


def absolute_score_check(actual, expected, atol):
    actual, expected, atol = float(actual), float(expected), float(atol)
    if not all(math.isfinite(x) for x in (actual, expected, atol)) or atol < 0:
        raise ValueError('Scores and tolerance must be finite; tolerance must be nonnegative')
    difference = abs(actual - expected)
    return {'expected_score': expected, 'actual_score': actual, 'absolute_difference': difference,
            'atol': atol, 'rtol': 0, 'passed': difference <= atol}


def read_reference_scores(path):
    result = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            key = (row['run_id'], row['image_id'])
            if key in result:
                raise ValueError(f'Duplicate prediction: {key}')
            score = float(row['image_score'])
            if not math.isfinite(score):
                raise ValueError(f'Non-finite prediction: {key}')
            result[key] = score
    if not result:
        raise ValueError('Empty predictions table')
    return result


def image_path(data_root, image_id):
    # image_id is a portable identifier, never a Windows path from the source machine.
    parts = image_id.split('/')
    if len(parts) != 4 or any(not x or x in {'.', '..'} or '\\' in x or ':' in x for x in parts):
        raise ValueError(f'Invalid image_id: {image_id}')
    return Path(data_root).joinpath(*parts).with_suffix('.png')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_patchcore_checkout(root):
    root = Path(root).resolve(strict=True)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PATCHCORE_COMMIT:
        raise ValueError(f'Expected PatchCore {PATCHCORE_COMMIT}, got {actual}')
    if subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', 'src']).returncode:
        raise ValueError('PatchCore src differs from the pinned commit')
    if not (root / 'src/patchcore/patchcore.py').is_file():
        raise ValueError('PatchCore src is missing')
    return actual


class PlainParamsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Params must contain plain values only: {module}.{name}')


def read_plain_params(path):
    params = PlainParamsUnpickler(io.BytesIO(Path(path).read_bytes())).load()
    expected = {'backbone.name': 'wideresnet50', 'layers_to_extract_from': ['layer2', 'layer3'],
                'input_shape': (3, 224, 224), 'pretrain_embed_dimension': 1024,
                'target_embed_dimension': 1024, 'patchsize': 3, 'patchstride': 1,
                'anomaly_scorer_num_nn': 1}
    if params != expected:
        raise ValueError('Bank parameters do not match the handed-over vial baseline')
    return params
