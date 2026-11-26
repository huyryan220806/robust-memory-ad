# Chuẩn bị AU-PRO: khái niệm, ví dụ và kiểm triển khai

Ngày 26/11/2026. **Đã hoàn thành ví dụ giả lập và ghi nhận chênh lệch triển khai. Chưa tính AU-PRO trên dữ liệu pilot.** File này chuẩn bị cho quyết định và kiểm triển khai tiếp theo, không bổ sung một chỉ số đã xác nhận vào bảng chất lượng.

## Ba đại lượng cần phân biệt

| Đại lượng | Tử số / mẫu số hoặc ý nghĩa |
|---|---|
| FPR cấp ảnh | Ảnh good bị cảnh báo / toàn bộ ảnh good |
| FPR cấp pixel | Pixel nền bị đánh dấu lỗi / toàn bộ pixel nền trong tập đánh giá |
| PRO tại một ngưỡng pixel | Trung bình tỷ lệ vùng lỗi được phủ, mỗi vùng liên thông có trọng số bằng nhau |

Với vùng ground truth thứ r là R_r và vùng dự đoán lỗi A(t), overlap của vùng r bằng `|A(t) ∩ R_r| / |R_r|`. PRO là trung bình của các overlap trên mọi vùng của mọi ảnh. Mỗi ảnh được gán nhãn vùng riêng để hai vùng ở cùng tọa độ trên hai ảnh không bị ghép thành một vùng.

Ví dụ tự dựng: có một vùng 1 pixel và một vùng 3 pixel. Nếu chỉ phát hiện trọn vùng 1 pixel, pixel recall là 1/4=0.25, nhưng PRO=(1+0)/2=0.5. PRO giúp mỗi vùng đóng góp ngang nhau, song tự nó không phải chỉ số chỉ dành cho lỗi nhỏ.

Quét ngưỡng pixel để có đường PRO theo FPR pixel. Trong quy ước ví dụ ở đây:

```text
AU-PRO_beta = integral từ 0 đến beta của PRO(FPR) dFPR / beta
```

Chia cho beta để chuẩn hóa về [0,1]. Nội suy/cắt đúng tại beta, xử lý điểm bằng nhau và điểm đầu đường cong phải được ghi rõ. Beta là giới hạn tích phân FPR pixel; alpha trong quy tắc hiệu chỉnh cảnh báo ảnh là một tham số khác, kể cả khi cùng bằng 0.05.

## Chọn giới hạn theo benchmark, không dùng mặc định ngầm

Bài MVTec AD 2, mục IV-B, dùng giới hạn **0.05** cho AU-PRO, chặt hơn giới hạn 0.30 thường dùng trước đó. Vì vậy hướng chuẩn bị là đối chiếu AU-PRO_0.05; nhóm cần ghi rõ giao thức cuối cùng. [Nguồn tác giả MVTec AD 2](https://arxiv.org/html/2503.21622v1#S4.SS2).

Mã Anomalib 2.2.0 đang cài đặt có mặc định `fpr_limit=0.3`, `num_thresholds=None`. Chỉ gọi tên metric mà không truyền giới hạn sẽ không thể hiện lựa chọn 0.05. Đối chiếu crop 224 × 224 của pilot cũng chưa đủ để gọi số đo là kết quả benchmark chính thức: cần khớp tập ảnh, hệ tọa độ mask/map, vùng được đánh giá và cách nội suy với giao thức đích.

## Những kiểm tra đã chạy

[Script ví dụ](../scripts/check_aupro_toy.py) tự dựng hai ảnh 8 × 8, gồm hai vùng lỗi diện tích 1 và 3 pixel trên ảnh đầu; ảnh thứ hai hoàn toàn good. Tổng 4 pixel lỗi và 124 pixel nền. Các score trong [0,1] để tránh nhánh tự sigmoid của thư viện.

Quy ước của ví dụ: liên thông 8, duyệt mọi mức score phân biệt, gom các điểm bằng nhau, thêm điểm không dự đoán pixel nào, nội suy tuyến tính và chia diện tích cho đúng beta. Hàm này phục vụ học và kiểm ca biên, chưa được chứng nhận tương đương mã benchmark và chưa tối ưu cho map thật.

| Ví dụ | beta | Đáp án theo quy ước đã khai báo | Anomalib 2.2.0 tại máy |
|---|---:|---:|---:|
| Score phân biệt hoàn hảo lỗi/nền | 0.05 | 1.0 | 0.5 |
| Score phân biệt hoàn hảo lỗi/nền | 0.30 | 1.0 | 0.5 |
| Score đảo ngược lỗi/nền | 0.05 | 0.0 | NaN |
| Score đảo ngược lỗi/nền | 0.30 | 0.0 | NaN |
| Mọi score bằng nhau | 0.05 | 0.025 | NaN |
| Mọi score bằng nhau | 0.30 | 0.15 | NaN |

Ở ví dụ mọi score bằng nhau, đường nội suy nối (0,0) với (1,1), nên diện tích chuẩn hóa đến beta bằng beta/2. Đây là hệ quả của quy ước nội suy đã nêu, không phải một điểm vận hành thực sự đạt được bằng ngưỡng xác định trên các score bằng nhau.

[Log JSON](../results/evaluation/pilot01/followup/aupro-toy.json) giữ phiên bản thư viện, hash script, các điểm đường cong và trạng thái từng phép đối chiếu; NaN được ghi `null` kèm nguyên nhân, không đổi thành 0. Bốn test kiểm đáp án các đường cong, trọng số vùng, tách vùng theo ảnh/biến đổi score affine tăng và trường hợp thiếu lớp đã qua.

Đây là **chênh lệch đã tái lập trên các ví dụ này**, chưa phải kết luận rằng mọi kết quả Anomalib đều sai hoặc xác định được duy nhất nguyên nhân. Mã cài tại máy chuẩn hóa bằng `fpr[-1]`; cần kiểm giá trị đầu cuối và cách dựng đường cong. [Issue #3718 của dự án Anomalib](https://github.com/open-edge-platform/anomalib/issues/3718) có báo cáo liên quan đến chuẩn hóa ở các bản cũ; nội dung issue là báo cáo của người dùng, không thay thế việc kiểm độc lập trên phiên bản sử dụng.

Kiểm AUROC trước đây dùng một metric khác và đã đối chiếu khớp; chênh lệch AU-PRO này không tự động phủ định AUROC đã kiểm.

## Cách tái lập và việc cần hoàn tất trước khi báo số thật

```powershell
# Chỉ chạy ví dụ độc lập; không cần Anomalib
python -B scripts/check_aupro_toy.py --output aupro-toy-reference.json

# Môi trường Anomalib riêng đã có ở máy làm việc
.venv-evaluation/Scripts/python.exe -B scripts/check_aupro_toy.py --anomalib

python -B -m unittest discover -s tests -v
```

Trước khi đưa AU-PRO thật vào bản thảo, cần hoàn tất:

1. Ghi rõ beta, liên thông, phạm vi ROI/crop, tập ảnh và phiên bản mã tham chiếu. Phân biệt số đo trên crop nội bộ với benchmark ở độ phân giải/giao thức khác.
2. Đối chiếu mã tham chiếu benchmark bằng các ví dụ có score trùng, vùng khác diện tích, nhiều ảnh, ảnh good, vùng nằm sát biên và đường cong không có điểm đúng tại beta. Giải thích chênh lệch hiện có trước khi chọn triển khai.
3. Kiểm đường cong đơn điệu theo FPR, giới hạn [0,1], điểm cắt đúng beta và cách trả thiếu dữ liệu. Không âm thầm chặn số sai về [0,1] hoặc thay NaN bằng 0.
4. Nếu dùng lấy mẫu ngưỡng để giảm chi phí, kiểm sai số khi tăng số ngưỡng so với tham chiếu. Không chuẩn hóa score riêng từng ảnh; giữ chung thang score trong tập tính metric.
5. Giữ ảnh good trong mẫu số nền; lưu hash map/mask, số ảnh/vùng/pixel và cấu hình cùng kết quả. Kiểm ảnh mất vùng do crop trước khi diễn giải độ nhạy.

Chưa cần đợi đầu ra mô hình mới để làm các kiểm triển khai trên. Chọn quy tắc báo cáo chính thức và chạy thí nghiệm xác nhận cần gắn với giao thức nhóm đã thống nhất.
