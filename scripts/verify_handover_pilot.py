"""
Giai đoạn 5: Kiểm tra chéo độc lập (Cross-Verification) cho cả 2 đầu bàn giao:
1. Kiểm chứng predictions.csv & Anomaly Map + Sinh ảnh Overlay đối chiếu GT mask (cho Đỗ Hoàng Phúc).
2. Tiến trình Python độc lập nạp lại từng file bank/index của 3 cấu hình (pilot01-full-s0, pilot01-random-s0, pilot01-greedy-s0)
   trên cả GPU lẫn CPU và đối chiếu image_score với predictions.csv (cho Lê Quyết Tiến).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_SRC = Path(
    os.environ.get("PATCHCORE_SRC", str(SCRIPT_DIR.parents[1] / "patchcore-inspection" / "src"))
)
for p in (str(SCRIPT_DIR), str(PATCHCORE_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import patchcore.common
import patchcore.patchcore
from mvtec_ad2_dataset import MVTecAD2Dataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD


def denormalize_img(tensor_chw: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    arr = tensor_chw.detach().cpu().numpy() * std + mean
    return np.clip(arr.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"D:\paper1\data\mvtec_ad_2"))
    parser.add_argument("--category", type=str, default="vial")
    parser.add_argument("--output_root", type=Path, default=Path(r"D:\paper1\outputs\pilot01"))
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--sample_image_id", type=str, default="")
    args = parser.parse_args()

    ver_dir = args.output_root / "verification"
    ver_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc manifest_images.csv để lấy tập ảnh đánh giá chuẩn (validation + test_public)
    manifest_csv = args.output_root / "manifest_images.csv"
    with open(manifest_csv, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    eval_manifest_ids = {
        r["image_id"] for r in manifest_rows if r["split"] in ("validation", "test_public")
    }
    num_expected_eval = len(eval_manifest_ids)
    print(f"[Manifest] Tổng số ảnh đánh giá (validation + test_public): {num_expected_eval}")

    # 2. Kiểm chứng predictions.csv tổng hợp và từng run_id
    combined_csv = args.output_root / "predictions.csv"
    with open(combined_csv, "r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))

    run_ids = ["pilot01-full-s0", "pilot01-random-s0", "pilot01-greedy-s0"]
    seen_pairs = set()
    rows_by_run = {rid: {} for rid in run_ids}
    map_shape_set = set()

    for row in pred_rows:
        rid = row["run_id"]
        iid = row["image_id"]
        score_val = float(row["image_score"])
        mpath = Path(row["map_path"])
        if not mpath.exists():
            # Tự động ánh xạ đường dẫn tương đối khi chạy trên máy khác (vd: Colab L4 Linux)
            mpath = args.output_root / "runs" / rid / "anomaly_maps" / Path(row["map_path"].replace("\\", "/")).name

        assert rid in rows_by_run, f"Unexpected run_id: {rid}"
        pair = (rid, iid)
        assert pair not in seen_pairs, f"Duplicate (run_id, image_id) pair: {pair}"
        seen_pairs.add(pair)

        assert not np.isnan(score_val) and not np.isinf(score_val), f"NaN/Inf score for {pair}"
        assert mpath.exists(), f"Missing anomaly map file: {mpath}"
        amap = np.load(mpath)
        assert not np.isnan(amap).any() and not np.isinf(amap).any(), f"NaN/Inf inside map: {mpath}"
        map_shape_set.add(tuple(amap.shape))

        rows_by_run[rid][iid] = {
            "score": score_val,
            "score_str": row["image_score"],
            "map_path": mpath.resolve().as_posix(),
        }

    for rid in run_ids:
        assert len(rows_by_run[rid]) == num_expected_eval, (
            f"Run {rid} has {len(rows_by_run[rid])} rows != expected {num_expected_eval}"
        )
        assert set(rows_by_run[rid].keys()) == eval_manifest_ids, (
            f"Run {rid} image_ids do not match eval manifest IDs!"
        )

    print(
        f"[PASSED] predictions.csv check: {len(pred_rows)} total rows "
        f"({num_expected_eval} rows x 3 runs), all unique pairs, all maps exist, 0 NaN/Inf, map shapes={map_shape_set}"
    )

    # 3. Chọn 1 ảnh lỗi có ground-truth mask để vừa sinh ảnh Overlay vừa kiểm chứng nạp lại Bank
    ds_eval = MVTecAD2Dataset(
        args.data_root,
        classname=args.category,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.EVAL_ALL,
    )

    target_idx = None
    if args.sample_image_id:
        for i, item_tuple in enumerate(ds_eval.data_to_iterate):
            if item_tuple[5] == args.sample_image_id:
                target_idx = i
                break
    if target_idx is None:
        # Chọn ảnh lỗi đầu tiên trong test_public/bad có vùng lỗi rõ ràng trong khung crop
        for i, item_tuple in enumerate(ds_eval.data_to_iterate):
            if item_tuple[1] == "bad":
                item = ds_eval[i]
                if item["mask"].sum().item() > 25:
                    target_idx = i
                    break

    sample_item = ds_eval[target_idx]
    sample_id = sample_item["image_id"]
    img_rgb = denormalize_img(sample_item["image"])
    gt_mask = sample_item["mask"].squeeze(0).numpy()

    # Sinh ảnh Overlay minh họa (Ảnh sau crop | GT Mask | Full Bank Map | Random Map | Greedy Map)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Cropped RGB ({args.imagesize}x{args.imagesize})\n{sample_id}")
    axes[0].axis("off")

    axes[1].imshow(img_rgb)
    axes[1].imshow(gt_mask, cmap="Reds", alpha=0.5)
    axes[1].set_title("Ground Truth Mask\n(H, W aligned)")
    axes[1].axis("off")

    for k, rid in enumerate(run_ids):
        rec = rows_by_run[rid][sample_id]
        amap = np.load(rec["map_path"])
        ax = axes[2 + k]
        ax.imshow(img_rgb)
        im_heat = ax.imshow(amap, cmap="jet", alpha=0.5)
        ax.contour(gt_mask, levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(f"{rid}\nscore={rec['score']:.4f} | shape={amap.shape}")
        ax.axis("off")

    fig.tight_layout()
    overlay_path = ver_dir / "overlay_verification_vial.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    print(f"[PASSED] Đã sinh ảnh Overlay kiểm chứng thứ tự trục (H, W) tại: {overlay_path}")

    # 4. Kiểm chứng nạp lại Bank độc lập trên cả GPU và CPU cho cả 3 cấu hình
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    reload_results = {}

    for rid in run_ids:
        bank_dir = args.output_root / "runs" / rid / "bank"
        csv_score = rows_by_run[rid][sample_id]["score"]

        # Nạp lại trên GPU (hoặc thiết bị chạy chính)
        pc_gpu = patchcore.patchcore.PatchCore(gpu_device)
        pc_gpu.load_from_path(
            load_path=str(bank_dir),
            device=gpu_device,
            nn_method=patchcore.common.FaissNN(on_gpu=False, num_workers=4),
            prepend="",
        )
        s_gpu, m_gpu = pc_gpu._predict(sample_item["image"].unsqueeze(0).to(gpu_device))
        score_gpu = float(s_gpu[0])
        diff_gpu = abs(score_gpu - csv_score)

        # Nạp lại trên CPU
        pc_cpu = patchcore.patchcore.PatchCore(cpu_device)
        pc_cpu.load_from_path(
            load_path=str(bank_dir),
            device=cpu_device,
            nn_method=patchcore.common.FaissNN(on_gpu=False, num_workers=4),
            prepend="",
        )
        s_cpu, m_cpu = pc_cpu._predict(sample_item["image"].unsqueeze(0).to(cpu_device))
        score_cpu = float(s_cpu[0])
        diff_cpu = abs(score_cpu - csv_score)

        ntotal = int(pc_gpu.anomaly_scorer.nn_method.search_index.ntotal)
        print(
            f"  [Reload Check] {rid} (ntotal={ntotal}): "
            f"CSV={csv_score:.6f} | Reload_GPU={score_gpu:.6f} (|diff|={diff_gpu:.2e}) | "
            f"Reload_CPU={score_cpu:.6f} (|diff|={diff_cpu:.2e})"
        )
        assert diff_gpu < 1e-5 and diff_cpu < 1e-3, f"Reload score mismatch on {rid}!"

        reload_results[rid] = {
            "bank_dir": bank_dir.resolve().as_posix(),
            "faiss_ntotal": ntotal,
            "sample_image_id": sample_id,
            "csv_score": csv_score,
            "reloaded_gpu_score": score_gpu,
            "reloaded_cpu_score": score_cpu,
            "abs_diff_gpu": diff_gpu,
            "abs_diff_cpu": diff_cpu,
        }

    report = {
        "predictions_verification": {
            "combined_predictions_csv": combined_csv.resolve().as_posix(),
            "manifest_images_csv": manifest_csv.resolve().as_posix(),
            "total_rows": len(pred_rows),
            "rows_per_run": num_expected_eval,
            "unique_pairs_verified": True,
            "all_maps_exist_on_disk": True,
            "no_nan_or_inf": True,
            "map_dimensions_HxW": list(map_shape_set)[0],
            "overlay_image_path": overlay_path.resolve().as_posix(),
        },
        "bank_reload_verification": reload_results,
    }
    report_path = ver_dir / "handover_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== BÁO CÁO KIỂM CHỨNG BÀN GIAO GIAI ĐOẠN 5 ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
