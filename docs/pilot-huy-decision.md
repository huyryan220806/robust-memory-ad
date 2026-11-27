# Phản hồi của Huy: quy tắc báo cáo pilot

Ghi nhận ngày 27/11/2026 từ nội dung Huy gửi cho Đỗ Hoàng Phúc. Phạm vi là pilot vial seed 0; quyết định được truyền đạt sau khi đã có kết quả thăm dò, không phải đăng ký trước cho đánh giá xác nhận.

- Mỗi baseline dùng 41 score validation good tăng dần, ngưỡng là float64 ngay trên score thứ 39 theo thứ tự 1-based; quy tắc `score >= threshold`, cố định cho bảy điều kiện sáng. Đây là mốc calibration 5%, không bảo đảm FPR quần thể 5%.
- FPR test là số báo động giả chia 35 ảnh good của test_public. Với alpha 5%: Full 0/35, Random 0/35, Greedy 2/35.
- Một ảnh chỉ có lỗi nhỏ khi mọi vùng lỗi liên thông 8 hướng trên mask gốc có diện tích <= 0,5% diện tích toàn ảnh gốc. Ảnh chứa cả lỗi lớn và nhỏ vẫn có mặt trong recall tất cả lỗi.
- Kết quả trên 21 ảnh đủ điều kiện là Full 8/21, Random 5/21, Greedy 7/21. Theo xác nhận của Hữu Phúc, 21 ảnh chỉ đến từ ba lọ lỗi vật lý được chụp bảy lần; kết quả này mang tính thăm dò.
- Giữ đủ bảy ảnh bad/014 bị crop mất một phần vùng lỗi khi tính pixel AP; báo rõ hạn chế. Pixel AP toàn test hiện có gồm cả bảy ảnh này và toàn bộ 140 ảnh test.

Các quy tắc được Huy chọn khớp các hàng đã tính, nên không sửa số liệu, config hoặc nhãn `exploratory_pending_team_review` trong các file máy đọc được đã xuất trước đó. Nhãn đó phản ánh **trạng thái tại thời điểm xuất**, không phản ánh phản hồi mới. Lọc `target_calibration_fpr=0.05`, `small_area_cutoff=0.005`, `lighting_group=all` trong [quality.csv](../results/evaluation/pilot01/quality.csv) để lấy hàng báo cáo cho từng run; chọn hàng ánh sáng riêng khi phân tích theo nhóm. Các mức khác vẫn là phân tích độ nhạy, không phải lựa chọn của Huy cho báo cáo chính của pilot.
