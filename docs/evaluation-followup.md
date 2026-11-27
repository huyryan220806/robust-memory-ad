# Phần chủ động chuẩn bị sau bàn giao pilot

Ngày 26/11/2026 · Đỗ Hoàng Phúc · Nhánh `evaluation/pilot`.

Đã thực hiện năm việc có thể làm với dữ liệu hiện có. Đây là phần kiểm chứng và chuẩn bị bản thảo/thí nghiệm tiếp theo; chưa đồng nghĩa hoàn tất toàn bộ trách nhiệm coauthor trong paper.

| Việc | Sản phẩm | Trạng thái |
|---|---|---|
| Truy ngược một hàng chất lượng | [Hướng dẫn đọc số liệu](evaluation-row-walkthrough.md), [log độc lập](../results/evaluation/pilot01/followup/source_row_check.json) | Khớp dữ liệu gốc: validation 2/41, test FP 2/35, lỗi nhỏ 7/21 ở cấu hình được chọn |
| Phân tích báo nhầm và bỏ sót | [Hai ca lỗi](evaluation-case-analysis.md), [số liệu sáu hàng](../results/evaluation/pilot01/followup/case_analysis.csv) | Đã đối chiếu score, ngưỡng, mask và vị trí cực đại map |
| Viết phần đánh giá cho bản thảo | [Methods, results và limitations bằng tiếng Anh](manuscript/evaluation-draft.md) | Bản nháp có dẫn nguồn số liệu, chờ tác giả rà |
| Học và kiểm AU-PRO | [Giải thích, ví dụ, việc cần kiểm tiếp](aupro-preparation.md), [log giả lập](../results/evaluation/pilot01/followup/aupro-toy.json) | Ví dụ độc lập khớp đáp án; Anomalib có chênh lệch cần giải quyết trước khi báo số thật |
| Chuẩn bị nhiều seed/phương pháp | [Hướng dẫn và ba mẫu CSV](evaluation-next-runs.md) | Mẫu trống và quy tắc khóa/tổng hợp đã có; chưa có thí nghiệm mới |

## Kiểm chứng kỹ thuật

- Script kiểm hàng đọc trực tiếp CSV nguồn và 105 mask, không dùng hàm tính của evaluator chính; lưu hash để tái lập.
- 11 unit test đã qua, gồm 7 test đánh giá trước đó và 4 test cho ví dụ AU-PRO. Các test qua không có nghĩa phép đối chiếu Anomalib AU-PRO đã khớp; log ghi rõ khác biệt.
- Bảng kết quả pilot có 216 hàng được đối chiếu với quyết định từng ảnh; dữ liệu mới không thay đổi các kết quả baseline đã bàn giao.
- Hình ca lỗi giữ tại máy và có lệnh tái tạo; Git chứa code, số liệu và tài liệu.

## Thứ tự Phúc nên đọc

1. Đọc hướng dẫn kiểm ngược một hàng để tự giải thích ngưỡng, tử số và mẫu số.
2. Xem hai ca lỗi; phân biệt bỏ sót cảnh báo ảnh với việc map không có tín hiệu trên vùng lỗi.
3. Rà bản nháp tiếng Anh và các giới hạn trước khi chuyển cho Huy ghép bài.
4. Đọc phần AU-PRO trước khi quyết định cách đưa metric này vào pipeline.

Chưa có xác nhận Phúc đã tự kiểm thủ công. Huy đã chọn quy tắc cho pilot; xem [quyết định](pilot-huy-decision.md). Hữu Phúc đã xác nhận quan hệ mẫu vật qua ánh sáng; xem [bàn giao mẫu vật](specimen-handoff.md). Thiết kế đánh giá xác nhận và các run tiếp theo vẫn cần phối hợp với nhóm.
