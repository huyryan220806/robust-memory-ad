"""
Giai đoạn 4: Chạy 3 cấu hình Baseline PatchCore trên MVTec AD 2:
  1. pilot01-full-s0   (Full bank)
  2. pilot01-random-s0 (Random coreset, đúng B vị trí duy nhất, seed=0)
  3. pilot01-greedy-s0 (Greedy coreset của patchcore-inspection: ApproximateGreedyCoresetSampler,
                        chiếu tuyến tính ngẫu nhiên 1024 -> 128 chiều, 10 điểm xuất phát, đúng B vị trí duy nhất, seed=0)

Đầu ra cho mỗi cấu hình tại D:\paper1\outputs\pilot01\runs\<run_id>\:
  - config.json, command.txt, run.log
  - bank/ (patchcore_params.pkl, nnscorer_search_index.faiss, selected_indices.npy, selected_indices.csv, bank_info.json)
  - anomaly_maps/<split>__<anomaly>__<stem>.npy
  - predictions.csv (run_id,image_id,image_score,map_path)
Đồng thời gộp chung 3 cấu hình vào D:\paper1\outputs\pilot01\predictions.csv và lưu cấu hình nhẹ vào repo nhóm configs/pilot/.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_SRC = Path(
    os.environ.get("PATCHCORE_SRC", str(SCRIPT_DIR.parents[1] / "patchcore-inspection" / "src"))
)
if str(PATCHCORE_SRC) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_SRC))

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils


def compute_approx_greedy_indices(
    features_np: np.ndarray,
    budget_b: int,
    seed: int,
    device: torch.device,
    number_of_starting_points: int = 10,
    proj_dim: int = 128,
) -> np.ndarray:
    """
    Thực thi thuật toán ApproximateGreedyCoresetSampler của patchcore-inspection
    với đúng số lượng mẫu cố định budget_b (B) và trả về mảng B chỉ số duy nhất thuộc miền train.
    Ghi nhận rõ:
      - Có dùng bản xấp xỉ (ApproximateGreedyCoresetSampler với 10 điểm khởi tạo ngẫu nhiên).
      - Có dùng phép chiếu giảm chiều tuyến tính ngẫu nhiên torch.nn.Linear(D, 128, bias=False).
    """
    patchcore.utils.fix_seeds(seed, device)
    n_samples, feat_dim = features_np.shape
    assert 0 < budget_b <= n_samples, f"Invalid budget_b={budget_b} for N={n_samples}"

    features_t = torch.from_numpy(features_np).to(torch.float32).to(device)
    with torch.no_grad():
        if feat_dim != proj_dim:
            mapper = torch.nn.Linear(feat_dim, proj_dim, bias=False).to(device)
            reduced_features = mapper(features_t)
        else:
            reduced_features = features_t

        num_start = int(np.clip(number_of_starting_points, 1, n_samples))
        start_points = np.random.choice(n_samples, num_start, replace=False).tolist()

        approx_dist_matrix = patchcore.sampler.GreedyCoresetSampler._compute_batchwise_differences(
            reduced_features, reduced_features[start_points]
        )
        anchor_distances = torch.mean(approx_dist_matrix, dim=-1).reshape(-1, 1)

        coreset_indices = []
        selected_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)

        for _ in range(budget_b):
            # Đảm bảo không chọn trùng chỉ số đã chọn do sai số làm tròn dấu phẩy động
            masked_distances = anchor_distances.clone()
            masked_distances[selected_mask] = -1.0
            select_idx = int(torch.argmax(masked_distances).item())
            coreset_indices.append(select_idx)
            selected_mask[select_idx] = True

            select_dist = patchcore.sampler.GreedyCoresetSampler._compute_batchwise_differences(
                reduced_features, reduced_features[select_idx : select_idx + 1]
            )
            anchor_distances = torch.minimum(anchor_distances, select_dist)

    indices_arr = np.array(coreset_indices, dtype=np.int64)
    assert len(indices_arr) == budget_b, f"Expected {budget_b} indices, got {len(indices_arr)}"
    assert len(set(indices_arr.tolist())) == budget_b, "Duplicate indices detected in Greedy Coreset!"
    return indices_arr


def save_selected_indices_csv(
    csv_path: Path,
    indices: np.ndarray,
    train_image_ids: list,
    patch_grid: list,
):
    h_grid, w_grid = patch_grid[0], patch_grid[1]
    patches_per_img = h_grid * w_grid
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coreset_rank", "flat_patch_index", "train_image_id", "patch_row", "patch_col"])
        for rank, flat_idx in enumerate(indices.tolist()):
            img_idx = flat_idx // patches_per_img
            rem = flat_idx % patches_per_img
            p_row = rem // w_grid
            p_col = rem % w_grid
            writer.writerow([rank, flat_idx, train_image_ids[img_idx], p_row, p_col])


def infer_from_cached_eval_features(
    anomaly_scorer: patchcore.common.NearestNeighbourScorer,
    eval_features: np.ndarray,
    patch_grid: list,
    target_size: tuple,
    device: torch.device,
):
    """
    Thực hiện truy vấn láng giềng gần nhất và tái tạo bản đồ bất thường (anomaly map)
    khớp 100% với PatchCore._predict trong patchcore-inspection.
    """
    num_eval, patches_per_img, embed_dim = eval_features.shape
    h_grid, w_grid = patch_grid[0], patch_grid[1]
    segmentor = patchcore.common.RescaleSegmentor(device=device, target_size=target_size)

    # Gộp toàn bộ truy vấn (num_eval * patches_per_img, embed_dim) hoặc chạy theo từng ảnh để đo thời gian
    flat_queries = eval_features.reshape(num_eval * patches_per_img, embed_dim)
    patch_scores_flat, _, _ = anomaly_scorer.predict([flat_queries])
    patch_scores_per_img = patch_scores_flat.reshape(num_eval, patches_per_img)

    # Điểm cấp ảnh: max trên các patch (patchcore.PatchMaker.score)
    image_scores = np.max(patch_scores_per_img, axis=-1).astype(np.float32)

    # Bản đồ bất thường: reshape về (num_eval, H_grid, W_grid) -> nội suy bilinear lên (224, 224) + Gaussian sigma=4
    patch_maps = patch_scores_per_img.reshape(num_eval, h_grid, w_grid)
    anomaly_maps = segmentor.convert_to_segmentation(patch_maps)
    return image_scores, anomaly_maps


def run_single_config(
    mode: str,
    run_id: str,
    train_features: np.ndarray,
    train_meta: dict,
    eval_features: np.ndarray,
    eval_meta: dict,
    budget_b: int,
    coreset_percentage: float,
    seed: int,
    device: torch.device,
    device_name: str,
    output_root: Path,
    repo_config_dir: Path,
):
    run_dir = output_root / "runs" / run_id
    bank_dir = run_dir / "bank"
    maps_dir = run_dir / "anomaly_maps"
    run_dir.mkdir(parents=True, exist_ok=True)
    bank_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    n_total, embed_dim = train_features.shape
    patchcore.utils.fix_seeds(seed, device)

    cmd_str = (
        f"D:\\paper1\\.venv-patchcore\\Scripts\\python.exe "
        f"D:\\paper1\\robust-memory-ad\\scripts\\run_baseline_pilot.py "
        f"--mode {mode} --run_id {run_id} --seed {seed} --budget_b {budget_b}"
    )
    (run_dir / "command.txt").write_text(cmd_str + "\n", encoding="utf-8")
    logger.info(f"Starting run_id={run_id} | mode={mode} | seed={seed} | device={device_name}")
    logger.info(f"Command: {cmd_str}")

    t_select_start = time.time()
    if mode == "full":
        selected_indices = np.arange(n_total, dtype=np.int64)
        bank_features = train_features
        sampler_desc = {
            "name": "IdentitySampler (Full Bank)",
            "approximate": False,
            "random_projection": False,
            "projection_dim": None,
            "num_starting_points": None,
        }
    elif mode == "random":
        patchcore.utils.fix_seeds(seed, device)
        selected_indices = np.random.choice(n_total, budget_b, replace=False).astype(np.int64)
        bank_features = np.ascontiguousarray(train_features[selected_indices], dtype=np.float32)
        sampler_desc = {
            "name": "RandomSampler (np.random.choice, replace=False)",
            "approximate": False,
            "random_projection": False,
            "projection_dim": None,
            "num_starting_points": None,
        }
    elif mode == "greedy":
        selected_indices = compute_approx_greedy_indices(
            train_features,
            budget_b=budget_b,
            seed=seed,
            device=device,
            number_of_starting_points=10,
            proj_dim=128,
        )
        bank_features = np.ascontiguousarray(train_features[selected_indices], dtype=np.float32)
        sampler_desc = {
            "name": "ApproximateGreedyCoresetSampler (patchcore-inspection)",
            "approximate": True,
            "random_projection": True,
            "projection_type": "torch.nn.Linear(1024, 128, bias=False)",
            "projection_dim": 128,
            "num_starting_points": 10,
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    selection_time_sec = time.time() - t_select_start
    n_selected = int(len(selected_indices))
    n_unique = int(len(set(selected_indices.tolist())))
    min_idx = int(selected_indices.min())
    max_idx = int(selected_indices.max())

    logger.info(
        f"Selection done in {selection_time_sec:.3f}s | N_total={n_total} -> N_selected={n_selected} "
        f"(unique={n_unique}, range=[{min_idx}, {max_idx}])"
    )

    # Kiểm chứng điều kiện bắt buộc của Giai đoạn 4
    if mode == "full":
        assert n_selected == n_total, f"Full bank size {n_selected} != N_total {n_total}"
    else:
        assert n_selected == budget_b, f"Selected count {n_selected} != budget B {budget_b}"
        assert n_unique == budget_b, f"Unique count {n_unique} != budget B {budget_b}"
        assert 0 <= min_idx and max_idx < n_total, f"Indices out of train domain [0, {n_total})"

    # Lưu danh sách chỉ số đã chọn cho random và greedy (và cả full bank metadata)
    np.save(bank_dir / "selected_indices.npy", selected_indices)
    if mode in ("random", "greedy"):
        save_selected_indices_csv(
            bank_dir / "selected_indices.csv",
            selected_indices,
            train_meta["image_ids"],
            train_meta["patch_grid"],
        )

    # Xây dựng FAISS IndexFlatL2 (CPU-compatible để nạp lại trên cả CPU lẫn GPU)
    t_fit_start = time.time()
    nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=4)
    anomaly_scorer = patchcore.common.NearestNeighbourScorer(
        n_nearest_neighbours=1, nn_method=nn_method
    )
    anomaly_scorer.fit(detection_features=[bank_features])
    fit_time_sec = time.time() - t_fit_start

    faiss_ntotal = int(anomaly_scorer.nn_method.search_index.ntotal)
    assert faiss_ntotal == n_selected, f"FAISS index ntotal ({faiss_ntotal}) != n_selected ({n_selected})"

    # Lưu chuẩn PatchCore save_to_path (nnscorer_search_index.faiss + patchcore_params.pkl)
    anomaly_scorer.save(str(bank_dir), save_features_separately=False, prepend="")
    input_shape = (
        3,
        train_meta["preprocessing"]["center_crop"][0],
        train_meta["preprocessing"]["center_crop"][1],
    )
    patchcore_params = {
        "backbone.name": "wideresnet50",
        "layers_to_extract_from": ["layer2", "layer3"],
        "input_shape": input_shape,
        "pretrain_embed_dimension": 1024,
        "target_embed_dimension": 1024,
        "patchsize": 3,
        "patchstride": 1,
        "anomaly_scorer_num_nn": 1,
    }
    params_path = bank_dir / "patchcore_params.pkl"
    with open(params_path, "wb") as pf:
        pickle.dump(patchcore_params, pf, protocol=pickle.HIGHEST_PROTOCOL)

    faiss_index_path = bank_dir / "nnscorer_search_index.faiss"
    faiss_size_bytes = int(faiss_index_path.stat().st_size)
    params_size_bytes = int(params_path.stat().st_size)
    logger.info(
        f"Saved bank to {bank_dir} | FAISS ntotal={faiss_ntotal} | "
        f"index_bytes={faiss_size_bytes} ({faiss_size_bytes/(1024*1024):.2f} MiB)"
    )

    # Suy luận trên toàn bộ tập đánh giá (181 ảnh = 41 validation + 140 test_public)
    t_infer_start = time.time()
    image_scores, anomaly_maps = infer_from_cached_eval_features(
        anomaly_scorer=anomaly_scorer,
        eval_features=eval_features,
        patch_grid=eval_meta["patch_grid"],
        target_size=(input_shape[1], input_shape[2]),
        device=device,
    )
    infer_time_sec = time.time() - t_infer_start

    records = eval_meta["records"]
    assert len(image_scores) == len(records) == len(anomaly_maps)

    pred_rows = []
    for idx, rec in enumerate(records):
        image_id = rec["image_id"]
        score_val = float(image_scores[idx])
        amap = np.asarray(anomaly_maps[idx], dtype=np.float32)

        assert not np.isnan(score_val) and not np.isinf(score_val), f"NaN/Inf score for {image_id}"
        assert not np.isnan(amap).any() and not np.isinf(amap).any(), f"NaN/Inf map for {image_id}"

        safe_map_name = image_id.replace("/", "__") + ".npy"
        map_file = maps_dir / safe_map_name
        np.save(map_file, amap)

        pred_rows.append({
            "run_id": run_id,
            "image_id": image_id,
            "image_score": f"{score_val:.6f}",
            "map_path": map_file.resolve().as_posix(),
        })

    pred_csv_path = run_dir / "predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "image_id", "image_score", "map_path"])
        writer.writeheader()
        writer.writerows(pred_rows)

    # Tính nhanh AUROC trên test_public để ghi vào log tham khảo
    test_scores = [float(image_scores[i]) for i, r in enumerate(records) if r["split"] == "test_public"]
    test_labels = [int(r["is_anomaly"]) for r in records if r["split"] == "test_public"]
    test_auroc = float(roc_auc_score(test_labels, test_scores))
    val_scores = [float(image_scores[i]) for i, r in enumerate(records) if r["split"] == "validation"]

    logger.info(
        f"Inference finished on {len(records)} eval images in {infer_time_sec:.3f}s "
        f"({(infer_time_sec/len(records))*1000:.2f} ms/img from cached features) | "
        f"test_public Image AUROC = {test_auroc:.4f}"
    )

    config_payload = {
        "run_id": run_id,
        "mode": mode,
        "seed": seed,
        "category": train_meta["category"],
        "patchcore_inspection_commit": "fcaa92f124fb1ad74a7acf56726decd4b27cbcad",
        "device": device_name,
        "dtype": "float32",
        "preprocessing": train_meta["preprocessing"],
        "backbone": train_meta["backbone"],
        "sampler": sampler_desc,
        "patch_counts": {
            "num_train_images": train_meta["num_images"],
            "patches_per_image": train_meta["patches_per_image"],
            "N_total_before_reduction": n_total,
            "coreset_percentage": 1.0 if mode == "full" else coreset_percentage,
            "B_after_reduction": n_selected,
            "unique_selected_indices": n_unique,
            "min_selected_index": min_idx,
            "max_selected_index": max_idx,
        },
        "bank_storage": {
            "bank_dir": bank_dir.resolve().as_posix(),
            "faiss_index_path": faiss_index_path.resolve().as_posix(),
            "patchcore_params_path": params_path.resolve().as_posix(),
            "selected_indices_npy": (bank_dir / "selected_indices.npy").resolve().as_posix(),
            "faiss_ntotal": faiss_ntotal,
            "faiss_index_size_bytes": faiss_size_bytes,
            "faiss_index_size_MiB": round(faiss_size_bytes / (1024 * 1024), 4),
            "params_size_bytes": params_size_bytes,
        },
        "evaluation_summary": {
            "total_eval_images": len(records),
            "validation_good_images": len(val_scores),
            "test_public_images": len(test_scores),
            "test_public_image_auroc": round(test_auroc, 4),
            "selection_time_sec": round(selection_time_sec, 4),
            "faiss_fit_time_sec": round(fit_time_sec, 4),
            "nn_query_and_map_time_sec": round(infer_time_sec, 4),
            "predictions_csv": pred_csv_path.resolve().as_posix(),
            "anomaly_maps_dir": maps_dir.resolve().as_posix(),
        },
        "command": cmd_str,
    }

    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (bank_dir / "bank_info.json").write_text(
        json.dumps(config_payload["bank_storage"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Lưu bản cấu hình nhẹ (dùng đường dẫn chuẩn D:/paper1/...) vào configs/pilot/ trong repo nhóm
    repo_config_dir.mkdir(parents=True, exist_ok=True)
    (repo_config_dir / f"{run_id}.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return pred_rows, config_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path(r"D:\paper1\outputs\pilot01"))
    parser.add_argument("--repo_config_dir", type=Path, default=Path(r"D:\paper1\robust-memory-ad\configs\pilot"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coreset_percentage", type=float, default=0.01)
    parser.add_argument("--budget_b", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["all", "full", "random", "greedy"], default="all")
    parser.add_argument("--run_id", type=str, default="")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    cache_dir = args.output_root / "feature_cache"
    train_features = np.load(cache_dir / "train_features.npy")
    train_meta = json.loads((cache_dir / "train_cache_meta.json").read_text(encoding="utf-8"))
    eval_features = np.load(cache_dir / "eval_features.npy")
    eval_meta = json.loads((cache_dir / "eval_cache_meta.json").read_text(encoding="utf-8"))

    n_total = int(train_features.shape[0])
    budget_b = args.budget_b if args.budget_b > 0 else int(n_total * args.coreset_percentage)

    if args.mode == "all":
        configs_to_run = [
            ("full", f"pilot01-full-s{args.seed}"),
            ("random", f"pilot01-random-s{args.seed}"),
            ("greedy", f"pilot01-greedy-s{args.seed}"),
        ]
    else:
        rid = args.run_id if args.run_id else f"pilot01-{args.mode}-s{args.seed}"
        configs_to_run = [(args.mode, rid)]

    all_pred_rows = []
    all_summaries = []

    for mode, rid in configs_to_run:
        rows, summary = run_single_config(
            mode=mode,
            run_id=rid,
            train_features=train_features,
            train_meta=train_meta,
            eval_features=eval_features,
            eval_meta=eval_meta,
            budget_b=budget_b,
            coreset_percentage=args.coreset_percentage,
            seed=args.seed,
            device=device,
            device_name=device_name,
            output_root=args.output_root,
            repo_config_dir=args.repo_config_dir,
        )
        all_pred_rows.extend(rows)
        all_summaries.append(summary)

    if args.mode == "all":
        combined_pred_csv = args.output_root / "predictions.csv"
        with open(combined_pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "image_id", "image_score", "map_path"])
            writer.writeheader()
            writer.writerows(all_pred_rows)
        (args.output_root / "baseline_runs_summary.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[DONE] Đã lưu bảng dự đoán tổng hợp (3 run_ids, {len(all_pred_rows)} dòng) tại: {combined_pred_csv}")


if __name__ == "__main__":
    main()
