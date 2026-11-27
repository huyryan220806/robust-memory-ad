"""Reload handed-over banks on CPU or an explicit CUDA device and check one image.

No feature cache, anomaly-map collection or ground-truth masks are required.
This is a correctness check, not an inference latency benchmark.
"""
import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.baseline_checks import (RUN_IDS, absolute_score_check, image_path,
                                     read_plain_params, read_reference_scores,
                                     sha256, verify_patchcore_checkout)


def check_banks(args):
    commit = verify_patchcore_checkout(args.patchcore_root)
    sys.path.insert(0, str(args.patchcore_root.resolve() / 'src'))
    import faiss
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    import patchcore.common
    import patchcore.patchcore
    import patchcore.utils
    from scripts.mvtec_ad2_dataset import MVTecAD2Dataset, DatasetSplit

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this command never falls back to CPU')
    if args.require_l4:
        if device.type != 'cuda' or 'L4' not in torch.cuda.get_device_name(device).upper().split():
            raise RuntimeError('An actual NVIDIA L4 CUDA device is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patchcore.utils.fix_seeds(0, device)
    refs = read_reference_scores(args.pilot_root / 'predictions.csv')
    sample_path = image_path(args.data_root, args.image_id)
    # Reuse exactly the baseline transform. Do not fetch a cached embedding.
    dataset = MVTecAD2Dataset(args.data_root, 'vial', split=DatasetSplit.TEST, resize=256, imagesize=224)
    with Image.open(sample_path) as image:
        tensor = dataset.transform_img(image.convert('RGB')).unsqueeze(0)
    atol = args.atol if args.atol is not None else (1e-4 if device.type == 'cuda' else 1e-3)
    results = []
    for run_id in args.run_ids:
        key = (run_id, args.image_id)
        if key not in refs:
            raise ValueError(f'Reference score missing for {key}')
        config_path = args.config_root / f'{run_id}.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        count = config['patch_counts']['B_after_reduction']
        bank = args.pilot_root / 'runs' / run_id / 'bank'
        index_path = bank / 'nnscorer_search_index.faiss'
        params_path = bank / 'patchcore_params.pkl'
        read_plain_params(params_path)
        selected = np.load(bank / 'selected_indices.npy', allow_pickle=False)
        if selected.dtype != np.int64 or selected.ndim != 1 or len(selected) != count:
            raise ValueError(f'Incorrect selected indices for {run_id}')
        if len(np.unique(selected)) != count or selected.min() < 0 or selected.max() >= 228144:
            raise ValueError(f'Nonunique/out-of-domain selected indices for {run_id}')
        expected_bytes = config['bank_storage']['faiss_index_size_bytes']
        if index_path.stat().st_size != expected_bytes:
            raise ValueError(f'Index byte count differs from config: {run_id}')
        model = patchcore.patchcore.PatchCore(device)
        model.load_from_path(str(bank), device=device,
                             nn_method=patchcore.common.FaissNN(on_gpu=False, num_workers=4), prepend='')
        model.eval()
        index = model.anomaly_scorer.nn_method.search_index
        if not isinstance(index, faiss.IndexFlatL2) or int(index.ntotal) != count or int(index.d) != 1024 or index.metric_type != faiss.METRIC_L2:
            raise ValueError(f'Incorrect FAISS index metadata: {run_id}')
        with torch.inference_mode():
            scores, maps = model._predict(tensor.to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        check = absolute_score_check(scores[0], refs[key], atol)
        amap = np.asarray(maps[0])
        if amap.shape != (224, 224) or not np.isfinite(amap).all():
            raise ValueError(f'Invalid output map: {run_id}')
        row = {'run_id': run_id, 'image_id': args.image_id, **check,
               'num_patches': int(index.ntotal), 'embedding_dim': int(index.d),
               'faiss_index_type': type(index).__name__, 'faiss_threads': faiss.omp_get_max_threads(),
               'index_bytes': index_path.stat().st_size, 'index_MB': index_path.stat().st_size / 1_000_000,
               'params_bytes': params_path.stat().st_size,
               'index_sha256': sha256(index_path), 'params_sha256': sha256(params_path),
               'selected_indices_sha256': sha256(bank / 'selected_indices.npy'),
               'map_shape': list(amap.shape), 'map_finite': True}
        results.append(row)
        print(f'{run_id}: expected={check["expected_score"]:.6f}, actual={check["actual_score"]:.6f}, '
              f'diff={check["absolute_difference"]:.3g}, atol={atol:g}, passed={check["passed"]}', flush=True)
        del model, index, scores, maps, amap
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    weights_url = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1.url
    weights_path = Path(torch.hub.get_dir()) / 'checkpoints' / weights_url.rsplit('/', 1)[1]
    return {'kind': 'bank_reload_correctness_only', 'device': str(device),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            'patchcore_commit': commit, 'verifier_source_sha256': sha256(Path(__file__)),
            'baseline_checks_source_sha256': sha256(Path(__file__).with_name('baseline_checks.py')),
            'dataset_source_sha256': sha256(Path(__file__).with_name('mvtec_ad2_dataset.py')),
            'predictions_sha256': sha256(args.pilot_root / 'predictions.csv'),
            'image_sha256': sha256(sample_path),
            'weights': {'name': 'IMAGENET1K_V1', 'url': weights_url,
                        'sha256': sha256(weights_path) if weights_path.is_file() else None},
            'environment': {'python': sys.version, 'os': platform.platform(), 'torch': torch.__version__,
                            'torchvision': torchvision.__version__, 'faiss': faiss.__version__,
                            'numpy': np.__version__, 'cuda': torch.version.cuda,
                            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                            'packages': {name: importlib.metadata.version(name) for name in ['timm', 'pretrainedmodels', 'scipy', 'Pillow']}},
            'all_passed': all(row['passed'] for row in results), 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patchcore-root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config-root', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/pilot')
    parser.add_argument('--image-id', default='vial/test_public/bad/011_overexposed')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
    parser.add_argument('--require-l4', action='store_true')
    parser.add_argument('--run-ids', nargs='+', choices=RUN_IDS, default=list(RUN_IDS))
    parser.add_argument('--atol', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before setup so a repeated run never overwrites an old result.
    with args.output.open('x', encoding='utf-8') as handle:
        try:
            report = check_banks(args)
        except Exception as exc:
            json.dump({'all_passed': False, 'error': str(exc), 'device': args.device}, handle, indent=2)
            raise
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    if not report['all_passed']:
        raise SystemExit('Reference score mismatch; report retained, do not start benchmarking')


if __name__ == '__main__':
    main()
