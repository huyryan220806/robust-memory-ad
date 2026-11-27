# Đánh giá pilot của Đỗ Hoàng Phúc

Ngày 26/11/2026. Nhánh `evaluation/pilot`, phát triển từ baseline commit `6d7e8d8f0127838aae8f9c2dd3f2c85a87b94141`.

## Phạm vi và trạng thái

Đánh giá đầu ra ba baseline Full/Random/Approximate Greedy trên vial seed 0. Không chạy lại backbone, tạo bank hoặc đo hiệu năng. Random và Greedy đều có 2.281 patch theo config, Full có 228.144 patch. Các số thời gian trong bàn giao baseline là truy vấn đặc trưng cache theo batch và tạo map, không phải p50/p95 độ trễ end-to-end.

**Ngày 27/11/2026, Huy đã chọn quy tắc 5% và cutoff 0,5% cho pilot**, đồng thời yêu cầu giữ bảy ảnh bị crop một phần khi tính pixel AP. Xem [ghi nhận quyết định](pilot-huy-decision.md). Bộ kết quả và [config v0.3-proposed](../configs/evaluation/pilot01-proposed.json) là bản xuất trước phản hồi, giữ nguyên tên, hash và nhãn trạng thái lịch sử; chỉ chọn hàng 5%/0,5% để báo cáo chính cho pilot. Các mức khác là phân tích độ nhạy thăm dò. Quyết định được đưa ra sau khi kết quả test đã được xem, không phải đăng ký trước; `test_public` không còn là tập đánh giá cuối chưa từng sử dụng.

## Nguồn dữ liệu và cách chạy

[Drive bàn giao của Hữu Phúc](https://drive.google.com/drive/folders/171zEixi2Xb07Njhon9B_lsEORCbI6kz2). Tải các thư mục cần thiết ra ngoài Git và giữ hậu tố đường dẫn:

```text
<assets>/data/mvtec_ad_2/vial/test_public/good/*.png
<assets>/data/mvtec_ad_2/vial/test_public/bad/*.png
<assets>/data/mvtec_ad_2/vial/test_public/ground_truth/bad/*_mask.png
<assets>/outputs/pilot01/manifest_images.csv
<assets>/outputs/pilot01/predictions.csv
<assets>/outputs/pilot01/runs/<run_id>/anomaly_maps/*.npy
```

CSV giữ đường dẫn Windows gốc. Code ánh xạ theo hậu tố duy nhất dưới `--assets`; không sửa CSV nguồn, không dùng đường dẫn tuyệt đối của người gửi để tải hay ghi file. Thiếu hoặc nhiều file cùng hậu tố đều gây lỗi. Không tải bank/cache lớn cho phần đánh giá này. Nếu danh sách Drive chỉ hiện 50 file, tải cả folder hoặc kiểm số lượng bằng danh sách đầy đủ; tuyệt đối không đánh giá phần đầu rồi coi là đủ.

Tạo môi trường riêng, cài `requirements-evaluation.txt`. Từ repo:

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/evaluate_handoff.py --manifest <assets>/outputs/pilot01/manifest_images.csv --predictions <assets>/outputs/pilot01/predictions.csv --assets <assets>
python -B scripts/plot_evaluation.py --assets <assets>
python -B scripts/verify_evaluation_results.py
```

Môi trường đã chạy: Python 3.12.10, numpy 2.5.3, scipy 1.18.0, scikit-learn 1.9.0, Pillow 12.2.0. Phiên bản và hash lưu trong audit. Kiểm torchvision là test tùy chọn, xác nhận PIL resize/crop khớp phép biến đổi baseline trên mask nhị phân.

Kiểm Anomalib tách khỏi yêu cầu chạy evaluator chính. Môi trường kiểm riêng dùng Anomalib 2.2.0, TorchMetrics 1.8.2, torch 2.6.0+cu124, torchvision 0.21.0+cu124. Sau khi chuẩn bị môi trường Anomalib tương thích:

```powershell
python -B scripts/check_anomalib_auroc.py --predictions <assets>/outputs/pilot01/predictions.csv
```

API: `AUROC(fields=['pred_score','gt_label'])`. Điểm được biến đổi affine đơn điệu về [0,1], float64, trước khi đưa vào TorchMetrics để tránh tự áp sigmoid cho score ngoài [0,1]. Thứ tự và điểm bằng nhau được giữ; kiểm trực tiếp với scikit-learn trên điểm gốc. [Tài liệu Anomalib](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/metrics/index.html).

## Dữ liệu được dùng

Manifest gồm 472 ảnh: 291 train good, 41 validation good, 140 test_public (35 good, 105 bad). Predictions gồm 543 hàng, ba run × 181 ảnh. Nhãn từ thành phần `good`/`bad` trong ID; điểm cao là bất thường theo người gửi. Mỗi nhóm ánh sáng test có 5 good, 15 bad. Ánh sáng được suy từ tên file theo mã baseline, chưa phải xác minh độc lập điều kiện thu nhận.

- Train chỉ dựng bank, không tham gia đánh giá này.
- Validation good dùng hiệu chỉnh ngưỡng. Tỷ lệ cảnh báo trên validation là chẩn đoán hiệu chỉnh, không phải FPR độc lập.
- Test_public dùng đo chất lượng pilot, phân tích theo ánh sáng, tìm ca lỗi.
- Hữu Phúc xác nhận 20 lọ/cảnh test được chụp lặp qua bảy mức sáng: 5 good và 15 bad; định danh là `(good hoặc bad, tiền tố)`, vì good/000 và bad/000 khác nhau. Thiết kế theo ánh sáng là đo lặp theo mẫu vật; không coi 140 ảnh là 140 đơn vị độc lập. Xem [ghi nhận và phân tích ghép cặp](specimen-handoff.md). Chưa có bằng chứng về cách lấy mẫu để khẳng định tính độc lập thống kê giữa các lọ hoặc giữa validation và test.

## Ngưỡng cảnh báo đề xuất

Quy tắc `score >= threshold`. Với n=41 ảnh validation good và mục tiêu alpha thuộc {1%,5%,10%}, đặt k=floor(alpha*n). Sắp điểm tăng dần, lấy điểm tại chỉ số n-k-1 (0-based), đặt ngưỡng là số float64 kế tiếp lớn hơn điểm đó (`np.nextafter(...,+inf)`). Cách này bảo đảm tối đa k cảnh báo trên tập hiệu chỉnh kể cả điểm trùng. Đây chỉ là ràng buộc thực nghiệm, không bảo đảm FPR quần thể.

Ngưỡng riêng cho từng run, cùng alpha và cùng 41 ảnh hiệu chỉnh. Giữ nguyên ngưỡng khi đổi ánh sáng. Không dùng nhãn test để chọn ngưỡng. Một ảnh calibration tương ứng 1/41≈2,44%; một good mỗi nhóm test tương ứng FPR 20 điểm phần trăm.

FPR = số good cảnh báo / tổng good. Recall tất cả lỗi = số bad cảnh báo / tổng bad. Mỗi hàng có tử số và mẫu số; dữ liệu không đủ để tính tỷ lệ thì để trống, không thay bằng 0.

## Định nghĩa lỗi nhỏ đề xuất và kiểm mask

Đọc mask gốc grayscale, yêu cầu giá trị chỉ {0,255}, kích thước PIL W×H=1400×1900, numpy H×W=(1900,1400). Nguồn baseline dùng `ToTensor()>0.5`; evaluator dùng raw>=128, tương đương với các mask này. Không ngầm áp raw>0 nếu mask có giá trị khác.

Vùng liên thông 8-neighbor trên mask gốc; diện tích mỗi vùng chia cho **toàn bộ ảnh gốc**. Huy chọn mức cắt 0,5% cho pilot: ảnh chỉ có lỗi nhỏ khi có vùng lỗi và **mọi** vùng có tỷ lệ <= 0,5%. Ảnh có cả vùng lớn và nhỏ không vào mẫu số lỗi nhỏ, nhưng vẫn thuộc mẫu số recall tất cả lỗi. Các mức 0,1% và 1% được giữ như phân tích độ nhạy thăm dò.

Để khớp map: PIL NEAREST resize 256×256, crop box `(16,16,240,240)` thành224×224. Kiểm số pixel trước/sau crop và theo dõi ID từng vùng gốc qua resize/crop. Mask còn ít nhất một pixel không đồng nghĩa giữ nguyên mọi vùng/diện tích. Ảnh lỗi bị mất vùng do crop vẫn nằm trong phép đo cấp ảnh theo định nghĩa gốc; không âm thầm bỏ ca khó.

## Chỉ số và giới hạn

- Image AUROC tính từ score CSV (đã làm tròn 6 chữ số trong baseline), không dùng max của map. Baseline tạo score từ max patch trước khi nội suy/làm mượt map. Không gọi điểm là xác suất.
- Pixel AP dùng `sklearn.metrics.average_precision_score` trên tất cả pixel của 140 ảnh test sau biến đổi. Good có mask toàn 0 theo nhãn good; bad thiếu mask bị dừng, không thay bằng 0. Đây là average precision dạng bậc thang, không phải diện tích hình thang AUPR.
- Không chuẩn hóa map theo từng ảnh. AP dùng score map gốc, toàn tập và từng nhóm. Map yêu cầu float32, shape224×224, hữu hạn. Không kết luận về vùng ảnh gốc đã bị crop bỏ.
- Độ nhạy lỗi nhỏ ở đây là **cảnh báo cấp ảnh**, không chứng minh định vị trúng lỗi nhỏ.
- AU-PRO chưa chạy; cần thống nhất giới hạn FPR và cài đặt trước giai đoạn báo cáo chỉ số này. Đây không phải phần đã hoàn thành trong pilot hiện tại.
- Chưa đánh giá seed khác, category khác, phương pháp mới của Huy hoặc test nhãn ẩn; chưa thể khẳng định phương pháp nào tốt hơn tổng quát.

## Đầu ra để bàn giao

Thư mục `results/evaluation/pilot01/` chứa:

| File | Ý nghĩa |
|---|---|
| `quality.csv` | Tử số/mẫu số, FPR, recall tất cả lỗi và recall lỗi nhỏ trên toàn lưới đề xuất |
| `calibration.csv` | Ngưỡng, mục tiêu và số cảnh báo trên validation |
| `image_decisions.csv` | Quyết định TP/FP/FN/TN từng ảnh để kiểm ngược |
| `mask_audit.csv` | Diện tích gốc, mất pixel/vùng khi resize/crop |
| `pixel_metrics.csv` | Image AUROC và pixel AP toàn tập/theo ánh sáng |
| `asset_inventory.csv` | ID, đường dẫn tương đối và hash của 105 mask + 543 map |
| `audit.json` | Giao thức, phiên bản, hash đầu vào và tổng hợp kiểm tra |
| `anomalib-check.json` | Đối chiếu AUROC bằng Anomalib, sklearn và công thức hạng |
| `lighting_diagnostics.png`, `case_fp.png`, `case_fn.png` | Hình chẩn đoán và ca lỗi theo quy tắc đề xuất nếu có |

Không cộng hàng `all` với các hàng nhóm con. Không trung bình các tỷ lệ để thay tỷ lệ gộp. Khi ghép với Tiến, giữ run ID, nhóm ảnh và các tham số giao thức; không nhân kết quả từ nhiều mức alpha/cutoff vào cùng một số hiệu năng mà không ghi các tham số.

Mọi file đầu ra chỉ xuất sau khi đọc và kiểm đủ đầu vào. Nếu chạy thất bại, đầu ra lần trước còn trên đĩa không đại diện cho lần chạy lỗi mới. Kiểm exit code và hash trong audit khi bàn giao.
