# Phân tích hai ca lỗi pilot01

Ngày 26/11/2026 · Thăm dò trên vial, seed 0. Hai ca được chọn bằng quy tắc lấy image ID đầu tiên trong nhóm Greedy FP và FN tại mục tiêu calibration FPR 5%; không đại diện cho tần suất lỗi toàn bộ tập.

Nguồn số liệu: [case_analysis.csv](../results/evaluation/pilot01/followup/case_analysis.csv), gồm hash ảnh/map và sáu hàng cho hai ảnh × ba run. Các ngưỡng dưới đây được làm tròn để đọc; quyết định dùng số đầy đủ trong CSV. Điểm bất thường không phải xác suất.

## Ca báo động giả: good/000_underexposed

ID đầy đủ: `vial/test_public/good/000_underexposed`. Nhãn good nên ground truth khi đánh giá pixel là mask toàn 0.

| Run | Image score | Ngưỡng | Score − ngưỡng | Cảnh báo |
|---|---:|---:|---:|---|
| Full | 3.141232 | 3.963692 | −0.822460 | Không |
| Random | 4.114296 | 5.870096 | −1.755800 | Không |
| Greedy | 4.460339 | 4.402119 | +0.058220 | Có: FP |

Trên overlay Greedy, map kích hoạt cả nền quanh lọ, rõ ở phía phải. Cực đại map ở `(y=90, x=171)`, giá trị khoảng 4.001400; đây là vị trí trong crop 224 × 224, không phải tọa độ ảnh gốc. **Image score cao hơn ngưỡng một khoảng nhỏ**, nên ảnh bị cảnh báo theo giao thức hiện tại.

Full và Random không cảnh báo ở cùng mục tiêu calibration, nhưng mỗi run có thang điểm/ngưỡng riêng. Không so độ lớn score hoặc khoảng cách tới ngưỡng giữa các run như xác suất hay độ tự tin đã hiệu chỉnh.

Giả thuyết cần kiểm tiếp: đặc trưng nhạy với ánh sáng hoặc sự khác biệt của bank có thể làm điểm good tăng. Hữu Phúc xác nhận các ảnh cùng nhánh và tiền tố là cùng mẫu vật/cảnh; có thể so sánh ghép cặp qua điều kiện sáng và qua seed. Một ảnh chưa chứng minh nguyên nhân. Không tăng ngưỡng chỉ để loại ca FP đã xem này.

## Ca bỏ sót: bad/003_overexposed

ID đầy đủ: `vial/test_public/bad/003_overexposed`. Mask sau biến đổi có **819 pixel lỗi**.

| Run | Image score | Score − ngưỡng | Cực đại map trong GT? | Quyết định |
|---|---:|---:|---|---|
| Full | 2.893761 | −1.069931 | Có, (y=81, x=114) | FN |
| Random | 4.424709 | −1.445387 | Không, (y=116, x=120) | FN |
| Greedy | 3.669401 | −0.732718 | Có, (y=82, x=112) | FN |

Overlay cho thấy ground truth ở phần phía trên thân lọ. Full và Greedy có cực đại map **nằm trong vùng lỗi**, mặc dù không phát cảnh báo cấp ảnh. Với Greedy, score map trung bình trong GT khoảng 2.951986, ngoài GT khoảng 1.892481. Quan sát này cho thấy có tín hiệu trên vùng lỗi; nó chưa chứng minh phân đoạn chính xác toàn vùng hoặc mọi thành phần lỗi.

Không dùng ngưỡng image score để cắt anomaly map: baseline tính image score từ patch trước nội suy/làm mượt; map cuối cùng có phép biến đổi khác. Cực đại map không bằng image score và chưa có ngưỡng pixel được hiệu chỉnh.

Mask gốc có hai vùng liên thông, diện tích **15,954 và 17,235 pixel**. Vùng lớn nhất chiếm `17,235 / 2,660,000 ≈ 0.6479%` toàn ảnh. Vì vậy ca này **không thuộc nhóm chỉ có lỗi nhỏ ở mức cắt 0.5%**, nhưng thuộc nhóm ở mức cắt 1%. Không dùng nó để minh họa một lỗi thuộc mẫu số 21 ảnh của báo cáo 0.5%.

Giả thuyết cần kiểm tiếp: ngưỡng hiệu chỉnh trên good regular có thể tạo đánh đổi giữa báo nhầm và bỏ sót khi chuyển ánh sáng; chất lượng đặc trưng cũng có thể góp phần. Cần phân tích cả phân phối good/bad và các run bổ sung để tách các khả năng. Việc cả ba baseline bỏ sót cùng ảnh chỉ là kết quả tại ngưỡng đang xét.

## Hình và câu có thể dùng trong bản thảo

Hai hình đã được dựng và xem tại máy: `results/evaluation/pilot01/case_fp.png` và `case_fn.png`. Hình chứa ảnh dữ liệu gốc nên không nằm trong commit. Tái tạo từ repo:

```powershell
python -B scripts/plot_evaluation.py --assets ../data/handoff/drive_pilot
```

Các panel dùng chung thang màu **trong từng hình** để tránh hiểu sai do đổi thang hiển thị; điều đó không hiệu chỉnh score giữa các phương pháp.

Đề xuất diễn đạt thận trọng:

> In the inspected normal underexposed image, Greedy exceeded its image-level threshold while Full and Random did not. In the inspected anomalous overexposed image, all three configurations missed the image-level alarm; however, the maximum map responses of Full and Greedy lay within the defect mask. These examples distinguish image-level alarm failure from the absence of a spatial anomaly response and do not establish a causal effect of lighting.
