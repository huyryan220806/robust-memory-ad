"""
Giai đoạn 3: Chạy thử quy mô nhỏ (Smoke Test), Thiết kế Feature Cache tách biệt Train/Eval & Chốt ngân sách B.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
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

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from mvtec_ad2_dataset import MVTecAD2Dataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD


def denormalize_img(tensor_chw: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    arr = tensor_chw.detach().cpu().numpy() * std + mean
    return np.clip(arr.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)


def build_patchcore_instance(device: torch.device, input_shape=(3, 224, 224), backbone_name="wideresnet50"):
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name = backbone_name
    backbone.seed = None
    nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=4)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=nn_method,
    )
    return pc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"D:\paper1\data\mvtec_ad_2"))
    parser.add_argument("--category", type=str, default="vial")
    parser.add_argument("--output_root", type=Path, default=Path(r"D:\paper1\outputs\pilot01"))
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coreset_percentage", type=float, default=0.01)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Device] Using {device} ({device_name})", flush=True)

    patchcore.utils.fix_seeds(args.seed, device)

    smoke_dir = args.output_root / "smoke_test"
    cache_dir = args.output_root / "feature_cache"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    ds_train = MVTecAD2Dataset(
        args.data_root, classname=args.category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.TRAIN
    )
    ds_eval = MVTecAD2Dataset(
        args.data_root, classname=args.category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.EVAL_ALL
    )

    # 2. Bước 3.1: Smoke test trên 8 ảnh train đầu tiên + xuất ảnh kiểm tra resize/crop
    print("\n--- [Bước 3.1] Smoke Test trích xuất đặc trưng trên 8 ảnh train ---", flush=True)
    pc = build_patchcore_instance(device, input_shape=(3, args.imagesize, args.imagesize))

    smoke_imgs = torch.stack([ds_train[i]["image"] for i in range(8)], dim=0).to(device)
    with torch.no_grad():
        smoke_feats, patch_shapes = pc._embed(smoke_imgs, provide_patch_shapes=True)
    smoke_feats_np = np.asarray(smoke_feats)
    patches_per_img = int(patch_shapes[0][0] * patch_shapes[0][1])
    embed_dim = int(smoke_feats_np.shape[1])
    feat_dtype = str(smoke_feats_np.dtype)

    print(f"  Số ảnh smoke batch        : {smoke_imgs.shape[0]}")
    print(f"  Kích thước tensor ảnh     : {tuple(smoke_imgs.shape)} (dtype={smoke_imgs.dtype})")
    print(f"  Kích thước lưới patch     : {patch_shapes[0]} -> {patches_per_img} patches/ảnh")
    print(f"  Ma trận đặc trưng batch   : {smoke_feats_np.shape} (D={embed_dim}, dtype={feat_dtype})")

    # Tìm 1 ảnh lỗi có vùng mask nhỏ để kiểm tra sau resize/crop chi tiết nhỏ vẫn nằm trong khung
    bad_indices = [i for i, x in enumerate(ds_eval.data_to_iterate) if x[1] == "bad"]
    best_small_idx = None
    best_mask_ratio = 1.0
    preserved_masks = 0
    for idx in bad_indices:
        item = ds_eval[idx]
        m_sum = float(item["mask"].sum().item())
        ratio = m_sum / (args.imagesize * args.imagesize)
        if m_sum > 0:
            preserved_masks += 1
            if 0.001 <= ratio < best_mask_ratio:
                best_mask_ratio = ratio
                best_small_idx = idx
    if best_small_idx is None:
        best_small_idx = bad_indices[0]

    sample_bad = ds_eval[best_small_idx]
    orig_pil = PIL.Image.open(sample_bad["image_path"]).convert("RGB")
    orig_mask_pil = PIL.Image.open(sample_bad["mask_path"]).convert("L")
    cropped_rgb = denormalize_img(sample_bad["image"])
    cropped_mask = sample_bad["mask"].squeeze(0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(orig_pil)
    axes[0].set_title(f"Original ({orig_pil.width}x{orig_pil.height})\n{sample_bad['image_id']}")
    axes[0].axis("off")
    axes[1].imshow(orig_mask_pil, cmap="gray")
    axes[1].set_title("Original GT Mask")
    axes[1].axis("off")
    axes[2].imshow(cropped_rgb)
    axes[2].set_title(f"Resized({args.resize}) + Crop({args.imagesize})")
    axes[2].axis("off")
    axes[3].imshow(cropped_rgb)
    axes[3].imshow(cropped_mask, cmap="jet", alpha=0.45)
    axes[3].set_title(f"Cropped Mask Overlay\n(Defect area={best_mask_ratio*100:.2f}%)")
    axes[3].axis("off")
    fig.tight_layout()
    crop_check_path = smoke_dir / "crop_check_sample.png"
    fig.savefig(crop_check_path, dpi=150)
    plt.close(fig)
    print(f"  Đã lưu ảnh kiểm chứng resize/crop tại: {crop_check_path}")
    print(f"  Tỷ lệ ảnh lỗi bảo toàn vùng mask sau crop: {preserved_masks}/{len(bad_indices)}")

    # 3. Bước 3.2: Trích xuất & lưu Feature Cache tách biệt rõ TRAIN và EVAL
    print("\n--- [Bước 3.2] Trích xuất và lưu Feature Cache tách biệt TRAIN & EVAL ---", flush=True)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    eval_loader = torch.utils.data.DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False, num_workers=0)

    t0 = time.time()
    train_feats_list = []
    train_image_ids = []
    with torch.no_grad():
        for batch in train_loader:
            imgs = batch["image"].to(torch.float).to(device)
            emb = pc._embed(imgs)
            train_feats_list.append(np.asarray(emb, dtype=np.float32))
            train_image_ids.extend(batch["image_id"])

    train_features = np.concatenate(train_feats_list, axis=0)
    train_time = time.time() - t0
    train_cache_path = cache_dir / "train_features.npy"
    np.save(train_cache_path, train_features)

    train_hash = hashlib.sha256(train_features[:100].tobytes()).hexdigest()[:16]
    train_meta = {
        "split": "train",
        "category": args.category,
        "num_images": len(ds_train),
        "patches_per_image": patches_per_img,
        "patch_grid": list(patch_shapes[0]),
        "total_patches": int(train_features.shape[0]),
        "embed_dim": int(train_features.shape[1]),
        "dtype": str(train_features.dtype),
        "memory_bytes": int(train_features.nbytes),
        "memory_mib": round(train_features.nbytes / (1024 * 1024), 2),
        "extraction_time_sec": round(train_time, 2),
        "device": device_name,
        "preprocessing": {
            "resize": [args.resize, args.resize],
            "center_crop": [args.imagesize, args.imagesize],
            "square_resize": True,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "backbone": {
            "name": "wideresnet50",
            "torchvision_arch": "wide_resnet50_2",
            "weights": "IMAGENET1K_V1 (pretrained=True)",
            "layers_to_extract_from": ["layer2", "layer3"],
            "patchsize": 3,
            "patchstride": 1,
            "pretrain_embed_dimension": 1024,
            "target_embed_dimension": 1024,
        },
        "head_sha256_16": train_hash,
        "image_ids": train_image_ids,
    }
    train_meta_path = cache_dir / "train_cache_meta.json"
    train_meta_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Đã lưu TRAIN cache: {train_cache_path} | shape={train_features.shape} ({train_meta['memory_mib']} MiB) trong {train_time:.2f}s", flush=True)

    t1 = time.time()
    eval_feats_list = []
    eval_records = []
    with torch.no_grad():
        for batch in eval_loader:
            imgs = batch["image"].to(torch.float).to(device)
            emb = np.asarray(pc._embed(imgs), dtype=np.float32)
            bs = imgs.shape[0]
            emb_reshaped = emb.reshape(bs, patches_per_img, embed_dim)
            eval_feats_list.append(emb_reshaped)
            for i in range(bs):
                eval_records.append({
                    "image_id": batch["image_id"][i],
                    "split": batch["split"][i],
                    "anomaly": batch["anomaly"][i],
                    "is_anomaly": int(batch["is_anomaly"][i].item()),
                    "lighting": batch["lighting"][i],
                    "image_path": batch["image_path"][i],
                    "mask_path": batch["mask_path"][i],
                })

    eval_features = np.concatenate(eval_feats_list, axis=0)
    eval_time = time.time() - t1
    eval_cache_path = cache_dir / "eval_features.npy"
    np.save(eval_cache_path, eval_features)

    eval_hash = hashlib.sha256(eval_features[:5].tobytes()).hexdigest()[:16]
    eval_meta = {
        "split": "validation+test_public",
        "category": args.category,
        "num_images": len(ds_eval),
        "patches_per_image": patches_per_img,
        "patch_grid": list(patch_shapes[0]),
        "feature_tensor_shape": list(eval_features.shape),
        "embed_dim": embed_dim,
        "dtype": str(eval_features.dtype),
        "memory_bytes": int(eval_features.nbytes),
        "memory_mib": round(eval_features.nbytes / (1024 * 1024), 2),
        "extraction_time_sec": round(eval_time, 2),
        "device": device_name,
        "preprocessing": train_meta["preprocessing"],
        "backbone": train_meta["backbone"],
        "head_sha256_16": eval_hash,
        "records": eval_records,
    }
    eval_meta_path = cache_dir / "eval_cache_meta.json"
    eval_meta_path.write_text(json.dumps(eval_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Đã lưu EVAL cache : {eval_cache_path} | shape={eval_features.shape} ({eval_meta['memory_mib']} MiB) trong {eval_time:.2f}s", flush=True)

    # 4. Bước 3.3: Chạy 1 ảnh mẫu trọn vẹn -> lưu thử bank -> nạp lại bank kiểm tra sai số dấu phẩy động
    print("\n--- [Bước 3.3] Kiểm chứng 1 ảnh mẫu trọn vẹn + Lưu & Nạp lại Bank ---", flush=True)
    # Dùng 1,000 patch đầu tiên làm sample bank nhanh để test cơ chế save_to_path / load_from_path
    pc_sample = build_patchcore_instance(device, input_shape=(3, args.imagesize, args.imagesize))
    pc_sample.anomaly_scorer.fit(detection_features=[train_features[:2000]])

    sample_item = ds_eval[best_small_idx]
    sample_img_tensor = sample_item["image"].unsqueeze(0).to(device)
    scores_orig, masks_orig = pc_sample._predict(sample_img_tensor)
    score_orig = float(scores_orig[0])
    map_orig = np.asarray(masks_orig[0], dtype=np.float32)

    sample_bank_dir = smoke_dir / "sample_bank"
    sample_bank_dir.mkdir(parents=True, exist_ok=True)
    pc_sample.save_to_path(str(sample_bank_dir), prepend="")

    sample_map_path = smoke_dir / "sample_anomaly_map.npy"
    np.save(sample_map_path, map_orig)

    # Nạp lại bank trên GPU và trên CPU độc lập để kiểm chứng
    pc_reload_gpu = patchcore.patchcore.PatchCore(device)
    pc_reload_gpu.load_from_path(
        load_path=str(sample_bank_dir),
        device=device,
        nn_method=patchcore.common.FaissNN(on_gpu=False, num_workers=4),
        prepend="",
    )
    scores_reload_gpu, masks_reload_gpu = pc_reload_gpu._predict(sample_img_tensor)
    score_reload_gpu = float(scores_reload_gpu[0])
    diff_gpu = abs(score_orig - score_reload_gpu)

    cpu_device = torch.device("cpu")
    pc_reload_cpu = patchcore.patchcore.PatchCore(cpu_device)
    pc_reload_cpu.load_from_path(
        load_path=str(sample_bank_dir),
        device=cpu_device,
        nn_method=patchcore.common.FaissNN(on_gpu=False, num_workers=4),
        prepend="",
    )
    scores_reload_cpu, _ = pc_reload_cpu._predict(sample_item["image"].unsqueeze(0).to(cpu_device))
    score_reload_cpu = float(scores_reload_cpu[0])
    diff_cpu = abs(score_orig - score_reload_cpu)

    print(f"  Sample image_id        : {sample_item['image_id']}")
    print(f"  Score gốc (lúc tạo)    : {score_orig:.8f}")
    print(f"  Score nạp lại (GPU)    : {score_reload_gpu:.8f} (|diff| = {diff_gpu:.2e})")
    print(f"  Score nạp lại (CPU)    : {score_reload_cpu:.8f} (|diff| = {diff_cpu:.2e})")
    assert diff_gpu < 1e-5 and diff_cpu < 1e-3, "LỖI: Điểm nạp lại bank lệch quá ngưỡng dấu phẩy động!"

    # Xuất predictions_sample.csv (1 dòng mẫu gửi sớm cho Đỗ Hoàng Phúc)
    pred_sample_csv = smoke_dir / "predictions_sample.csv"
    with open(pred_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "image_id", "image_score", "map_path"])
        writer.writeheader()
        writer.writerow({
            "run_id": "pilot01-smoke-s0",
            "image_id": sample_item["image_id"],
            "image_score": f"{score_orig:.6f}",
            "map_path": sample_map_path.resolve().as_posix(),
        })

    # 5. Bước 3.4: Tính toán ngân sách B
    n_total = int(train_features.shape[0])
    budget_b = int(n_total * args.coreset_percentage)
    full_bank_mib = round(train_features.nbytes / (1024 * 1024), 2)
    coreset_bank_mib = round((budget_b * embed_dim * 4) / (1024 * 1024), 2)

    smoke_report = {
        "category": args.category,
        "device": device_name,
        "num_train_images": len(ds_train),
        "num_eval_images": len(ds_eval),
        "input_image_tensor_shape": [3, args.imagesize, args.imagesize],
        "patches_per_image": patches_per_img,
        "embed_dim_D": embed_dim,
        "dtype": feat_dtype,
        "N_total_train_patches": n_total,
        "full_bank_size_bytes": int(train_features.nbytes),
        "full_bank_size_MiB": full_bank_mib,
        "coreset_percentage": args.coreset_percentage,
        "budget_B_patches": budget_b,
        "coreset_bank_size_MiB": coreset_bank_mib,
        "preserved_bad_masks_after_crop": f"{preserved_masks}/{len(bad_indices)}",
        "sample_verification": {
            "image_id": sample_item["image_id"],
            "image_path": sample_item["image_path"],
            "score_original": score_orig,
            "score_reloaded_gpu": score_reload_gpu,
            "score_reloaded_cpu": score_reload_cpu,
            "abs_diff_gpu": diff_gpu,
            "abs_diff_cpu": diff_cpu,
            "sample_bank_dir": sample_bank_dir.resolve().as_posix(),
            "predictions_sample_csv": pred_sample_csv.resolve().as_posix(),
            "sample_map_path": sample_map_path.resolve().as_posix(),
            "crop_check_image": crop_check_path.resolve().as_posix(),
        },
    }
    report_path = smoke_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(smoke_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== BÁO CÁO SMOKE TEST & CHỐT NGÂN SÁCH B ===")
    print(json.dumps(smoke_report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
