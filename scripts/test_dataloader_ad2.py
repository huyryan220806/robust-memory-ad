"""
Kiểm chứng Giai đoạn 2 - Bước 3:
Đọc thử 1 batch từ MVTecAD2Dataset (train, validation, test_public, eval_all)
và xác nhận nhãn, split, lighting, kích thước tensor ảnh & mask chính xác.
"""

import argparse
import json
import sys
from pathlib import Path
import torch

import os

SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_SRC = Path(
    os.environ.get("PATCHCORE_SRC", str(SCRIPT_DIR.parents[1] / "patchcore-inspection" / "src"))
)
for p in (str(SCRIPT_DIR), str(PATCHCORE_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mvtec_ad2_dataset import MVTecAD2Dataset, DatasetSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"D:\paper1\data\mvtec_ad_2")
    parser.add_argument("--category", type=str, default="vial")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    ds_train = MVTecAD2Dataset(args.data_root, classname=args.category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.TRAIN)
    ds_val = MVTecAD2Dataset(args.data_root, classname=args.category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.VAL)
    ds_test = MVTecAD2Dataset(args.data_root, classname=args.category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.TEST)
    ds_eval = MVTecAD2Dataset(args.data_root, classname=args.category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.EVAL_ALL)

    print(f"Dataset counts for '{args.category}':")
    print(f"  TRAIN      : {len(ds_train)}")
    print(f"  VALIDATION : {len(ds_val)}")
    print(f"  TEST_PUBLIC: {len(ds_test)}")
    print(f"  EVAL_ALL   : {len(ds_eval)} (validation + test_public)")

    assert len(ds_eval) == len(ds_val) + len(ds_test), "LỖI: EVAL_ALL không bằng tổng VAL + TEST_PUBLIC!"

    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=False)
    batch_train = next(iter(loader_train))

    print("\n--- Train Batch Check ---")
    print("  image shape :", tuple(batch_train["image"].shape), "dtype:", batch_train["image"].dtype)
    print("  mask shape  :", tuple(batch_train["mask"].shape), "dtype:", batch_train["mask"].dtype)
    print("  is_anomaly  :", batch_train["is_anomaly"].tolist())
    print("  split       :", batch_train["split"])
    print("  lighting    :", batch_train["lighting"])
    print("  image_id[0] :", batch_train["image_id"][0])

    assert tuple(batch_train["image"].shape) == (args.batch_size, 3, args.imagesize, args.imagesize)
    assert tuple(batch_train["mask"].shape) == (args.batch_size, 1, args.imagesize, args.imagesize)
    assert all(x == 0 for x in batch_train["is_anomaly"].tolist()), "LỖI: Tập train chứa ảnh lỗi!"

    # Lấy 1 batch từ cuối test_public (chứa ảnh lỗi 'bad' có mask)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    batch_test = next(iter(loader_test))

    print("\n--- Test_Public Batch Check ---")
    print("  image shape :", tuple(batch_test["image"].shape), "dtype:", batch_test["image"].dtype)
    print("  mask shape  :", tuple(batch_test["mask"].shape), "dtype:", batch_test["mask"].dtype)
    print("  is_anomaly  :", batch_test["is_anomaly"].tolist())
    print("  anomaly     :", batch_test["anomaly"])
    print("  lighting    :", batch_test["lighting"])
    print("  image_id    :", batch_test["image_id"])
    print("  mask_path   :", batch_test["mask_path"])

    assert tuple(batch_test["image"].shape) == (args.batch_size, 3, args.imagesize, args.imagesize)
    assert tuple(batch_test["mask"].shape) == (args.batch_size, 1, args.imagesize, args.imagesize)
    print("\n[PASSED] Kiểm chứng DataLoader MVTec AD 2 thành công!")


if __name__ == "__main__":
    main()
