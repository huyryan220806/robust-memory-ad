# Chuẩn bị đánh giá nhiều seed và phương pháp tiếp theo

Ngày 26/11/2026. Các mẫu dưới đây chỉ có tiêu đề cột, **chưa có kết quả của seed hoặc phương pháp mới**. Đây là đề xuất cho đợt sau; `quality.csv` và quy ước đã gửi Tiến ở pilot vẫn giữ nguyên.

## Ba mẫu đã chuẩn bị

| File | Một hàng đại diện cho |
|---|---|
| [run_registry.csv](../templates/evaluation/run_registry.csv) | Một run chạy một phương pháp, category và seed, cùng nguồn code/config/dữ liệu |
| [quality_long.csv](../templates/evaluation/quality_long.csv) | Một metric của một run trên một nhóm ảnh và một lựa chọn giao thức cụ thể |
| [seed_summary.csv](../templates/evaluation/seed_summary.csv) | Trung bình và độ lệch chuẩn của một metric qua các seed có điều kiện đánh giá tương thích |

Sao chép mẫu sang thư mục kết quả của đợt mới rồi điền; giữ bản mẫu trống. CSV dùng UTF-8, dấu phẩy phân cột, dấu chấm thập phân. Không điền 0 cho kết quả chưa chạy. Chưa có bộ chuyển đổi tự động từ bảng pilot sang định dạng đề xuất này.

## Đăng ký run và tập ảnh

`run_id` phải trỏ về đúng lần chạy trong manifest dự đoán của baseline/phương pháp mới; không suy đoán bằng tên người. Với mẫu hiện tại, một run chỉ có một category và seed. Nếu pipeline dùng run bao nhiều category thì phải thống nhất lại khóa trước khi nhập.

Trong registry, ghi `method`, `seed`, `category`, `bank_patch_count`, `source_commit`, hash config, hash manifest, hash danh sách ID calibration và đánh giá, `protocol_version`, `status`. Hash danh sách ID tính trên chuỗi UTF-8 gồm các ID duy nhất sắp thứ tự tăng dần, mỗi ID kết thúc bằng LF; từ chối ID trùng. Manifest hash phản ánh nội dung file; hash ID phản ánh thành viên tập. Hash tài sản ảnh/mask/map tiếp tục nằm trong asset inventory của từng đợt.

`evaluation_set_id` trong bảng chất lượng xác định **tập đánh giá cha** gồm dataset/release/split và hash ID. `lighting_group=all` hoặc tên ánh sáng xác định nhóm con; metadata phân nhóm phải được cố định trong manifest đã hash. `calibration_id` xác định đúng tập calibration và quy tắc; ngưỡng thực tế vẫn lưu riêng cho từng run. Không dùng chung một threshold số học cho các run có thang điểm khác nhau.

## Khóa hàng chất lượng và giá trị thiếu

Khóa của `quality_long.csv` là tổ hợp:

```text
run_id + category + split + lighting_group + evaluation_set_id
+ protocol_version + metric + calibration_id + target_calibration_fpr
+ small_definition_id + small_area_cutoff
```

Các trường không áp dụng dùng ô trống nhất quán khi kiểm trùng khóa. `threshold` là giá trị phát sinh từ calibration đã nêu, không tạo thêm hàng chỉ vì cách làm tròn khác nhau. `log_path` trỏ tới log tái lập của đợt đó. `status` ghi rõ kết quả thăm dò hay theo giao thức đã chốt; không lấy bản mẫu làm bằng chứng phê duyệt.

| metric đề xuất | Tham số áp dụng | numerator / denominator |
|---|---|---|
| image_auroc | Không có ngưỡng/calibration/cutoff lỗi nhỏ | Để trống; giá trị từ toàn bộ score của nhóm |
| pixel_ap | Không có ngưỡng/calibration/cutoff lỗi nhỏ | Để trống; lưu số pixel trong log |
| false_positive_rate | calibration_id, alpha, threshold; không có cutoff lỗi nhỏ | FP / số good |
| all_anomaly_recall | calibration_id, alpha, threshold; không có cutoff lỗi nhỏ | TP / số bad |
| small_recall | calibration và định nghĩa/mức cắt lỗi nhỏ | Bad đủ điều kiện được cảnh báo / bad đủ điều kiện |

Không lặp AUROC/AP theo từng alpha và cutoff. Không lặp FPR/recall tất cả lỗi theo cutoff lỗi nhỏ. Với AU-PRO sau này, phải bổ sung giới hạn tích phân vào khóa hoặc tên metric, chẳng hạn `aupro_fpr005`, và cố định triển khai trong protocol; chưa có hàng AU-PRO thật ở mẫu này.

`value` là tỷ lệ [0,1] khi tính được. Nếu mẫu số bằng 0 thì để `value` trống, vẫn ghi mẫu số 0 và lý do như `no_eligible_small_images`. Nếu run chưa thực hiện thì chưa tạo hàng chất lượng; nếu đo thất bại thì ghi trạng thái/lý do và không tạo số giả. AUROC cần có cả hai lớp; các trường hợp không xác định phải được ghi là thiếu.

## Tổng hợp qua seed

Chỉ gộp khi giống phương pháp, ngân sách bank/cấu hình so sánh, category, tập ảnh, nhóm ánh sáng, metric và giao thức. Nếu phương pháp có nhiều ngân sách bank, `comparison_config_id` phải tách chúng. Các metric tại ngưỡng phải có cùng **quy tắc** calibration và alpha; ngưỡng cụ thể được tính riêng mỗi run và không đem trung bình để tạo ngưỡng mới.

Với n seed có kết quả hợp lệ: `mean = sum(value)/n`; `sample_sd = sqrt(sum((value-mean)^2)/(n-1))` khi n>=2. Với n=1, ghi mean của seed đó, SD để trống và lý do `single_seed_no_sample_sd`. SD qua seed mô tả biến thiên giữa lần chạy; không phải khoảng tin cậy theo mẫu ảnh. Ghi seed yêu cầu, seed hiện có, run nguồn và trạng thái thiếu để không che một phương pháp bị thiếu run.

Giữ kết quả từng category. Nếu báo trung bình macro các category, tính mỗi category một trọng số và ghi đúng tên; không coi AUROC gộp mọi score giữa category/run là tương đương. Không cộng hàng `all` với các nhóm ánh sáng. Các phép so sánh ghép cặp phải dùng cùng tập seed và cùng tập ảnh; không chỉ chọn seed tốt nhất của mỗi phương pháp.

Số seed, category mở rộng, ngân sách bank và tiêu chí kết luận cần được ghi trong kế hoạch đợt tiếp theo. Mẫu không tự đặt số lần chạy hay một ngưỡng “đạt paper”.

## Ghép với profiling của Tiến

Tiến giữ `measurement_id` và `device` trong bảng đo hiệu năng; bảng chất lượng gắn với run và phạm vi đánh giá. Khi ghép, chọn rõ một hàng metric/ngưỡng/nhóm ảnh và một phiên đo. Nếu một run có nhiều phép đo thiết bị, chất lượng có thể xuất hiện trên nhiều hàng sau join nhưng vẫn là cùng một phép đo chất lượng; không coi các bản sao là seed mới.

Không thay hợp đồng bàn giao pilot bằng mẫu mới mà chưa thống nhất. Với pilot hiện tại, dùng [quality.csv](../results/evaluation/pilot01/quality.csv) kèm khóa đã mô tả trong [giao thức](evaluation.md).

## Đầu vào tiếp theo và người cung cấp

| Nội dung cần chốt/nhận | Người phối hợp | Phúc đã chuẩn bị |
|---|---|---|
| Protocol xác nhận, phương pháp mới và thiết kế so sánh | Huy/nhóm | Bản thảo, giao thức đề xuất, mẫu bảng |
| Run mới, seed/config/hash và cách lấy mẫu giữa các split | Hữu Phúc | Bộ kiểm đầu vào, truy ngược hàng, quy tắc đăng ký run; quan hệ mẫu vật qua ánh sáng đã được xác nhận |
| Kết quả đo gắn phiên đo và thiết bị | Tiến | Quy tắc ghép chất lượng và tránh nhân đôi kết quả |
| AU-PRO được đối chiếu triển khai | Phúc, nhóm rà quy ước | Ví dụ và log chênh lệch, [danh sách kiểm tiếp](aupro-preparation.md) |

Khi nhận dữ liệu mới: kiểm manifest/predictions/assets đầy đủ → ghi registry và protocol → chạy evaluator theo cấu hình tương ứng → đối chiếu tử số/mẫu số → xuất bảng từng run → tổng hợp seed → cập nhật bản thảo với giới hạn phù hợp. Không cần dựng bank hoặc đo thiết bị thay phần việc của người khác để hoàn thiện các bước chuẩn bị này.
