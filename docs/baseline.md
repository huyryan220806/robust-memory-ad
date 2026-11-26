# Báo cáo Kỹ thuật & Hướng dẫn Tái lập Baseline PatchCore (Pilot Tuần 2)

**Người phụ trách:** Hữu Huỳnh Phúc  
**Nhánh làm việc:** `baseline/pilot`  
**Đề tài:** *Defect-Preserving Robust Coreset Selection under Memory and Latency Constraints for Industrial Anomaly Detection*

---

## 1. Cấu hình Phần cứng, Môi trường & Mã phiên bản

| Hạng mục | Thông số thực tế |
|---|---|
| **Hệ điều hành** | Microsoft Windows 11 Home Single Language (`10.0.26300`, Build `26300`) |
| **CPU** | 13th Gen Intel(R) Core(TM) i7-13620H (10 nhân vật lý, 16 luồng) |
| **RAM hệ thống** | `15.71 GB` |
| **GPU & VRAM** | `NVIDIA GeForce RTX 4060 Laptop GPU` (`8188 MiB` VRAM, Driver `610.78`) |
| **Môi trường ảo Python** | `D:\paper1\.venv-patchcore` (`Python 3.11.9`) |
| **PyTorch & Torchvision** | `torch==2.8.0+cu126`, `torchvision==0.23.0+cu126` (`allow_tf32=False`, chuẩn IEEE-754 `float32`) |
| **Thư viện tìm láng giềng** | `faiss-cpu==1.16.0` (`IndexFlatL2`, `num_workers=4`, lưu chuẩn CPU-compatible để nạp lại trên cả GPU và CPU) |
| **Thư viện phụ trợ** | `timm==1.0.31`, `pretrainedmodels==0.7.4`, `scikit-learn==1.9.1`, `scikit-image==0.26.0`, `scipy==1.17.1`, `numpy==2.4.6`, `pillow==13.0.0`, `matplotlib==3.11.2`, `click==8.5.1` |
| **Repo `patchcore-inspection`** | Commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` (đặt tại `D:\paper1\patchcore-inspection`, tách biệt cạnh `D:\paper1\robust-memory-ad`) |
| **Repo `robust-memory-ad`** | Commit gốc `7d46ffcf54ee11c56f025b8c14b6bf15b165878c` (nhánh `baseline/pilot`) |

---

## 2. Dữ liệu MVTec AD 2 (`vial`) & Quy ước `manifest_images.csv`

- **Đường dẫn dữ liệu (ngoài Git):** `D:\paper1\data\mvtec_ad_2\vial`
- **File danh sách ảnh chuẩn hóa:** `D:\paper1\outputs\pilot01\manifest_images.csv` (472 dòng ảnh + 1 dòng tiêu đề).
- **Cấu trúc cột chuẩn:** `image_id,category,split,image_path,lighting,mask_path`
  - Quy ước `image_id` ổn định theo đường dẫn tương đối: `vial/<split>/<defect_type>/<stem>` (ví dụ: `vial/train/good/000_regular`, `vial/validation/good/000_regular`, `vial/test_public/bad/011_overexposed`).
  - Ghi `"chua_xac_dinh"` cho `mask_path` của các ảnh chuẩn (`good`).
  - Kiểm chứng tự động đạt 100%: **0 dòng trùng lặp**, **giao `image_id` giữa `train`, `validation`, `test_public` bằng rỗng ($\emptyset$)**.

| Tập (`split`) | Nhãn (`anomaly`) | Số lượng ảnh | Độ phân giải gốc ($W \times H$) | Điều kiện ánh sáng (`lighting`) | Vai trò trong Pilot |
|---|---|---:|---|---|---|
| `train` | `good` | **291** | `1400 x 1900` (Grayscale `L` $\to$ `RGB`) | `regular` (291) | Trích xuất đặc trưng dựng Memory Bank |
| `validation` | `good` | **41** | `1400 x 1900` | `regular` (41) | Đánh giá (hiệu chỉnh ngưỡng / đo báo động giả) |
| `test_public` | `good` (35) + `bad` (105) | **140** | `1400 x 1900` (kèm 105 GT mask) | 7 mức: `regular`, `overexposed`, `underexposed`, `shift_1`, `shift_2`, `shift_3`, `shift_4` (mỗi mức 20 ảnh: 5 `good`, 15 `bad`) | Đánh giá chất lượng phát hiện & định vị lỗi |
| **Tổng cộng** | — | **472** | `1400 x 1900` | — | Tập đánh giá suy luận (`eval_all`) = `41 + 140 = 181` ảnh |

---

## 3. Tiền xử lý, Backbone, Feature Cache & Chốt ngân sách Coreset `B`

1. **Tiền xử lý ảnh (`MVTecAD2Dataset`):**
   - `Resize((256, 256), BILINEAR)` (square resize để giữ trọn chiều cao lọ `1400x1900`) $\to$ `CenterCrop((224, 224))` $\to$ `ToTensor()` $\to$ `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
   - Với ground-truth mask: `Resize((256, 256), NEAREST)` $\to$ `CenterCrop((224, 224))`.
   - **Kiểm chứng bảo toàn lỗi nhỏ sau crop:** `105/105` (100%) ảnh lỗi trong `test_public/bad` giữ nguyên vùng mask lỗi bên trong khung `224x224` (đã lưu hình minh họa tại `D:\paper1\outputs\pilot01\smoke_test\crop_check_sample.png`).
2. **Backbone & Trích xuất đặc trưng PatchCore:**
   - Backbone: `wideresnet50` (`torchvision.models.wide_resnet50_2`, trọng số `IMAGENET1K_V1`).
   - Tầng trích xuất: `["layer2", "layer3"]`.
   - Tham số PatchMaker & Aggregator: `patchsize = 3`, `patchstride = 1`, `pretrain_embed_dimension = 1024`, `target_embed_dimension = 1024`.
   - Kích thước lưới patch mỗi ảnh: `28 x 28 = 784` patches/ảnh, số chiều `D = 1024`, kiểu số `dtype = float32`.
3. **Feature Cache tách biệt `TRAIN` và `EVAL`:**
   - `TRAIN cache`: `D:\paper1\outputs\pilot01\feature_cache\train_features.npy` — shape `(228144, 1024)`, `float32`, dung lượng `891.19 MiB` (kèm `train_cache_meta.json`).
   - `EVAL cache`: `D:\paper1\outputs\pilot01\feature_cache\eval_features.npy` — shape `(181, 784, 1024)`, `float32`, dung lượng `554.31 MiB` (kèm `eval_cache_meta.json`).
4. **Ngân sách Coreset `B` và `seed`:**
   - `seed = 0`
   - Tổng số patch train trước rút gọn: `N_total = 291 x 784 = 228,144` patches (`891.19 MiB`).
   - Tỷ lệ rút gọn chuẩn: `1%` (`coreset_percentage = 0.01`) $\implies$ **`B = 2,281` patches** (`8.91 MiB`, giảm **100 lần** dung lượng bank).

---

## 4. Bảng Tóm tắt 3 Cấu hình Baseline (`Full bank`, `Random`, `Greedy`)

Cả 3 cấu hình chạy trên cùng tập đánh giá `181` ảnh (`41 validation` + `140 test_public`), cùng backbone `wideresnet50`, cùng tiền xử lý `256 -> 224`, cùng `dtype = float32`, và cùng bộ truy vấn `FaissNN(on_gpu=False, num_workers=4)` (`k = 1`):

| Mã lượt chạy (`run_id`) | Bộ chọn (`sampler`) | Số patch trước rút gọn (`N_total`) | Số patch lưu bank (`B`) | Dung lượng FAISS Index (`bytes` / `MiB`) | Thời gian chọn coreset (`s`) | Thời gian truy vấn + tạo map 181 ảnh (`s` / `ms/ảnh`) | Image AUROC (`test_public`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`pilot01-full-s0`** | `IdentitySampler` (Full bank) | `228,144` | `228,144` | `934,477,869 B` (`891.19 MiB`) | `0.001 s` | `135.322 s` (`747.64 ms/ảnh`) | `0.8879` |
| **`pilot01-random-s0`** | `RandomSampler` (`np.random.choice`, `replace=False`, `seed=0`) | `228,144` | `2,281` (duy nhất, miền `[81, 228101]`) | `9,343,021 B` (`8.91 MiB`) | `0.009 s` | `1.810 s` (`10.00 ms/ảnh`) | `0.7518` |
| **`pilot01-greedy-s0`** | `ApproximateGreedyCoresetSampler` (xấp xỉ 10 điểm khởi tạo + chiếu tuyến tính ngẫu nhiên `Linear(1024 -> 128, bias=False)`, `seed=0`) | `228,144` | `2,281` (duy nhất, miền `[158, 228102]`) | `9,343,021 B` (`8.91 MiB`) | `3.689 s` | `1.670 s` (`9.23 ms/ảnh`) | **`0.8914`** |

---

## 5. Lệnh Tái lập Từng bước (PowerShell)

### Bước 1: Sinh `manifest_images.csv` và kiểm chứng DataLoader MVTec AD 2
```powershell
D:\paper1\.venv-patchcore\Scripts\python.exe D:\paper1\robust-memory-ad\scripts\build_manifest_ad2.py `
  --data_root D:\paper1\data\mvtec_ad_2 `
  --category vial `
  --output_csv D:\paper1\outputs\pilot01\manifest_images.csv `
  --summary_json D:\paper1\outputs\pilot01\dataset_survey.json

D:\paper1\.venv-patchcore\Scripts\python.exe D:\paper1\robust-memory-ad\scripts\test_dataloader_ad2.py `
  --data_root D:\paper1\data\mvtec_ad_2 `
  --category vial
```

### Bước 2: Chạy Smoke Test, tách cache `train`/`eval` và chốt ngân sách `B`
```powershell
D:\paper1\.venv-patchcore\Scripts\python.exe D:\paper1\robust-memory-ad\scripts\run_smoke_and_cache.py `
  --data_root D:\paper1\data\mvtec_ad_2 `
  --category vial `
  --output_root D:\paper1\outputs\pilot01 `
  --resize 256 `
  --imagesize 224 `
  --batch_size 16 `
  --seed 0 `
  --coreset_percentage 0.01
```

### Bước 3: Tạo Bank & Suy luận cho 3 cấu hình Baseline (`pilot01-full-s0`, `pilot01-random-s0`, `pilot01-greedy-s0`)
```powershell
# Chạy đồng loạt cả 3 cấu hình:
D:\paper1\.venv-patchcore\Scripts\python.exe D:\paper1\robust-memory-ad\scripts\run_baseline_pilot.py `
  --output_root D:\paper1\outputs\pilot01 `
  --repo_config_dir D:\paper1\robust-memory-ad\configs\pilot `
  --seed 0 `
  --coreset_percentage 0.01 `
  --budget_b 2281 `
  --mode all
```

### Bước 4: Nạp lại Bank độc lập (trên GPU & CPU) và Kiểm chứng Bàn giao
```powershell
D:\paper1\.venv-patchcore\Scripts\python.exe D:\paper1\robust-memory-ad\scripts\verify_handover_pilot.py `
  --data_root D:\paper1\data\mvtec_ad_2 `
  --category vial `
  --output_root D:\paper1\outputs\pilot01 `
  --resize 256 `
  --imagesize 224 `
  --sample_image_id "vial/test_public/bad/011_overexposed"
```
