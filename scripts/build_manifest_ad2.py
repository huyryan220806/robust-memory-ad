"""
Script khảo sát dữ liệu MVTec AD 2 và sinh file danh sách ảnh chuẩn hóa (manifest_images.csv).
Tuân thủ quy ước Giai đoạn 2:
- Cột: image_id,category,split,image_path,lighting,mask_path
- image_id ổn định theo đường dẫn tương đối: <category>/<split>/<subfolder>/<stem>
- Ghi "chua_xac_dinh" cho trường thiếu thông tin (ví dụ mask_path của ảnh good hoặc lighting không rõ).
- Kiểm chứng tự động:
  + Không có dòng hay image_id trùng lặp.
  + Giao của image_id giữa các tập split (train, validation, test_public) bằng rỗng.
  + Mọi đường dẫn image_path và mask_path (khác "chua_xac_dinh") đều tồn tại thực tế trên đĩa.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from PIL import Image


def extract_lighting_from_stem(stem: str) -> str:
    """Trích xuất điều kiện ánh sáng từ tên file MVTec AD 2 (vd: 000_regular -> regular, 013_shift_3 -> shift_3)."""
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1]:
        return parts[1]
    return "chua_xac_dinh"


def build_manifest(data_root: Path, category: str, output_csv: Path, summary_json: Path = None):
    cat_root = data_root / category
    if not cat_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục category: {cat_root}")

    rows = []
    resolutions = Counter()
    modes = Counter()
    split_defect_counts = Counter()
    lighting_counts = Counter()

    # Cấu trúc chuẩn MVTec AD 2:
    # train/good/*.png
    # validation/good/*.png
    # test_public/good/*.png
    # test_public/bad/*.png (kèm mask tại test_public/ground_truth/bad/<stem>_mask.png)
    split_specs = [
        ("train", "good", cat_root / "train" / "good"),
        ("validation", "good", cat_root / "validation" / "good"),
        ("test_public", "good", cat_root / "test_public" / "good"),
        ("test_public", "bad", cat_root / "test_public" / "bad"),
    ]

    for split_name, defect_type, folder in split_specs:
        if not folder.exists():
            continue
        img_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        for img_path in img_files:
            stem = img_path.stem
            image_id = f"{category}/{split_name}/{defect_type}/{stem}"
            lighting = extract_lighting_from_stem(stem)

            if split_name == "test_public" and defect_type == "bad":
                candidate_mask = cat_root / "test_public" / "ground_truth" / "bad" / f"{stem}_mask{img_path.suffix}"
                if candidate_mask.exists():
                    mask_path_str = candidate_mask.resolve().as_posix()
                else:
                    raise FileNotFoundError(f"Thiếu ground-truth mask cho ảnh lỗi: {img_path}")
            else:
                mask_path_str = "chua_xac_dinh"

            with Image.open(img_path) as im:
                resolutions[f"{im.width}x{im.height}"] += 1
                modes[im.mode] += 1

            split_defect_counts[f"{split_name}/{defect_type}"] += 1
            lighting_counts[f"{split_name}:{lighting}"] += 1

            rows.append({
                "image_id": image_id,
                "category": category,
                "split": split_name,
                "image_path": img_path.resolve().as_posix(),
                "lighting": lighting,
                "mask_path": mask_path_str,
            })

    # --- KIỂM CHỨNG TỰ ĐỘNG ---
    if not rows:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong {cat_root}")

    all_ids = [r["image_id"] for r in rows]
    assert len(all_ids) == len(set(all_ids)), "LỖI: Phát hiện image_id trùng lặp trong manifest!"

    ids_by_split = {}
    for r in rows:
        ids_by_split.setdefault(r["split"], set()).add(r["image_id"])

    splits = list(ids_by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            inter = ids_by_split[s1].intersection(ids_by_split[s2])
            assert len(inter) == 0, f"LỖI RÒ RỈ DỮ LIỆU: Giao giữa {s1} và {s2} khác rỗng: {inter}"

    for r in rows:
        assert Path(r["image_path"]).exists(), f"File ảnh không tồn tại: {r['image_path']}"
        if r["mask_path"] != "chua_xac_dinh":
            assert Path(r["mask_path"]).exists(), f"File mask không tồn tại: {r['mask_path']}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "category", "split", "image_path", "lighting", "mask_path"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "category": category,
        "data_root": cat_root.resolve().as_posix(),
        "manifest_path": output_csv.resolve().as_posix(),
        "total_images": len(rows),
        "split_counts": {k: len(v) for k, v in ids_by_split.items()},
        "split_defect_counts": dict(split_defect_counts),
        "resolutions_WxH": dict(resolutions),
        "image_modes": dict(modes),
        "lighting_breakdown": dict(sorted(lighting_counts.items())),
        "leakage_check_passed": True,
        "duplicate_check_passed": True,
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== KẾT QUẢ KHẢO SÁT & KIỂM CHỨNG MANIFEST MVTec AD 2 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path(r"D:\paper1\data\mvtec_ad_2"))
    parser.add_argument("--category", type=str, default="vial")
    parser.add_argument("--output_csv", type=Path, default=Path(r"D:\paper1\outputs\pilot01\manifest_images.csv"))
    parser.add_argument("--summary_json", type=Path, default=Path(r"D:\paper1\outputs\pilot01\dataset_survey.json"))
    args = parser.parse_args()
    build_manifest(args.data_root, args.category, args.output_csv, args.summary_json)
