# Kiểm ngược một hàng quality từ dữ liệu gốc

Ngày 26/11/2026 · Đỗ Hoàng Phúc · pilot01, vial, seed 0.

**Đã tính lại bằng script độc lập với evaluator và khớp số liệu. Phúc và đồng tác giả chưa ký xác nhận kiểm thủ công.** Huy đã chọn mốc 5% và cutoff 0,5% cho pilot thăm dò sau khi có kết quả; dữ liệu gốc vẫn mang nhãn lịch sử `v0.3-proposed`.

## Hàng cần giải thích

Trong [quality.csv](../results/evaluation/pilot01/quality.csv), chọn đúng tổ hợp:

| Trường | Giá trị |
|---|---|
| run_id | pilot01-greedy-s0 |
| category / split / lighting_group | vial / test_public / all |
| target_calibration_fpr | 0.05 |
| small_area_cutoff | 0.005 |
| threshold | 4.402119000000001 |
| false_positives / normal_count | 2 / 35 |
| small_detected / small_count | 7 / 21 |

`all` gộp bảy điều kiện ánh sáng. `0.05` là mục tiêu cảnh báo trên validation good, còn `0.005` là mức cắt tỷ lệ diện tích **mỗi vùng lỗi**. Hai tham số này có ý nghĩa khác nhau.

## 1. Lấy ngưỡng từ 41 ảnh validation good

Với alpha=0.05, `floor(41 × 0.05)=2`: cho phép tối đa hai cảnh báo trên chính tập hiệu chỉnh. Sắp 41 score tăng dần, lấy score lớn thứ ba làm biên rồi chọn số float64 ngay phía trên biên đó. Năm điểm cao nhất:

| Hạng tăng dần | ID sau vial/validation/good/ | Score | Cảnh báo |
|---:|---|---:|---|
| 37 | 006_regular | 4.282448 | Không |
| 38 | 004_regular | 4.339485 | Không |
| 39 | 013_regular | 4.402119 | Không, đây là biên |
| 40 | 021_regular | 4.581844 | Có |
| 41 | 005_regular | 4.884100 | Có |

Quy tắc chạy là `score >= 4.402119000000001`. Nếu thay bằng ngưỡng làm tròn `4.402119`, ảnh ở biên sẽ bị tính thêm thành cảnh báo. Số cảnh báo validation là **2/41 ≈ 4.878%**; đây không phải cam kết FPR trên ảnh mới.

Xem toàn bộ [41 điểm validation đã xếp hạng](../results/evaluation/pilot01/followup/validation_rank_check.csv). Script kiểm độc lập dùng `Decimal` để đối chiếu `score > biên`, rồi xác nhận từng quyết định khớp quy tắc float64 của evaluator.

## 2. Đếm báo động giả trên test_public

Trong 35 ảnh good, chỉ hai ảnh vượt ngưỡng, đều thuộc underexposed:

| ID sau vial/test_public/good/ | Score | Cảnh báo |
|---|---:|---|
| 000_underexposed | 4.460339 | Có |
| 001_underexposed | 4.353128 | Không |
| 002_underexposed | 4.593967 | Có |
| 003_underexposed | 4.375233 | Không |
| 004_underexposed | 4.252352 | Không |

FPR toàn test là **2/35 = 5.714%**; FPR riêng underexposed là **2/5 = 40%**. Hai số đúng cho hai mẫu số khác nhau. Không cộng hàng `all` với các hàng ánh sáng, vì sẽ đếm trùng ảnh. Mỗi nhóm chỉ có năm good nên một ảnh làm FPR thay đổi 20 điểm phần trăm.

## 3. Đếm ảnh chỉ có lỗi nhỏ từ mask gốc

Mask gốc có `1900 × 1400 = 2,660,000` pixel. Mức cắt 0.005 tương ứng **13,300 pixel cho mỗi vùng liên thông 8**. Ảnh đủ điều kiện nếu có lỗi và vùng lớn nhất không quá 13,300 pixel. Không dùng diện tích sau resize để quyết định và không cộng diện tích các vùng rồi so với mức cắt này.

Script đọc lại cả 105 PNG mask, gán nhãn vùng và đếm pixel từng vùng. So sánh bằng số nguyên `max_area × 200 <= 2,660,000` để tránh sai số ở biên.

| Tiền tố tên ảnh bad | Số ảnh đủ điều kiện | Greedy cảnh báo |
|---|---:|---:|
| 007, qua bảy mức sáng | 7 | 0 |
| 010, qua bảy mức sáng | 7 | 7 |
| 011, qua bảy mức sáng | 7 | 0 |
| Tổng | 21 | 7 |

Vậy **small_recall = 7/21 = 33.333%**. Đây là recall cảnh báo **cấp ảnh** trong tập ảnh chỉ có lỗi nhỏ; chưa chứng minh định vị được từng vùng lỗi. Hữu Phúc xác nhận 21 ảnh này thuộc ba lọ lỗi, mỗi lọ chụp qua bảy mức sáng; không coi chúng là 21 mẫu vật độc lập.

## Bằng chứng và cách tái lập

- [140 quyết định và diện tích vùng đọc lại](../results/evaluation/pilot01/followup/source_row_check.csv).
- [Log khớp số liệu và SHA-256 đầu vào, mask, script](../results/evaluation/pilot01/followup/source_row_check.json).
- [Script kiểm độc lập](../scripts/prepare_followup_evidence.py), không import hàm tính của evaluator chính.

Từ thư mục repo, với bộ bàn giao hiện có nằm ngoài Git:

```powershell
python -B scripts/prepare_followup_evidence.py --assets ../data/handoff/drive_pilot
```

Để tự kiểm trước khi giải thích với nhóm, Phúc có thể mở CSV validation tìm ba điểm cao nhất; mở hai dòng good bị cảnh báo; chọn một mask thuộc 007, 010 hoặc 011 để đối chiếu diện tích và quyết định. Khi đã tự kiểm, ghi người kiểm, ngày và phạm vi kiểm trong biên bản riêng. Trạng thái hiện tại chỉ xác nhận phép tính độc lập bằng code đã khớp.
