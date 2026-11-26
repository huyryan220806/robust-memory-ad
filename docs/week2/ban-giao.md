# Bàn giao của Nguyễn Đình Huy · Tuần 2

**Ngày đối chiếu 26/11/2026.** Quyết định làm việc cho nhóm dựa trên ba nhánh bàn giao và tệp Drive của pilot01. Các quyết định về phép đo dưới đây được ghi sau khi xem kết quả vial/test_public. [Đề cương v1](../research-question.md) · [Bản PDF 4 trang](De-cuong-v1.pdf)

## Hữu Huỳnh Phúc · baseline và khả năng chạy lại

Chốt vial làm nhóm pilot, seed 0. Full giữ 228.144 patch, random và approximate greedy cùng giữ 2.281 patch. Dùng WRN50-2 V1, layer2/layer3, FP32, FAISS CPU IndexFlatL2, k = 1. [Cấu hình greedy](https://github.com/huyryan220806/robust-memory-ad/blob/6d7e8d8f0127838aae8f9c2dd3f2c85a87b94141/configs/pilot/pilot01-greedy-s0.json)

- Tải manifest và predictions gốc từ gói bàn giao, kiểm SHA-256 so với audit của Hoàng Phúc. Kiểm 472 ID khác nhau trong manifest, ba run có cùng 181 ID đánh giá và tổng cộng 543 điểm hữu hạn.
- Lần theo `pilot01-greedy-s0` và `vial/test_public/bad/000_regular`. Tệp FAISS có 9.343.021 byte, đặc trưng cache của ảnh gồm 784 vector 1.024 chiều. Tính lại 1-NN từ cache và bank được score 6,836819 so với 6,836815 trong CSV. Ở α = 5%, ngưỡng greedy là 4,402119000000001, nên ảnh này được tính vào TP. [Kiểm bank từ cache](cached-reload.json)
- Từ cùng ảnh PNG gốc, chạy WRN50-2 V1 và mã PatchCore của tác giả trên CPU. Score 6,836821, lệch 0,0000056 so với CSV, map 224 × 224 hữu hạn. [Log ảnh gốc, phiên bản và SHA-256](raw-image-replay.json)
- Sửa diễn giải về mask khi viết bài. 105/105 mask vẫn có vùng lỗi sau crop. Trong 7 mask có tiền tố bad/014, crop làm mất 194 trên 1.785 pixel dương cộng gộp trong hệ tọa độ 256². Giữ cả các ca này trong mẫu số.
- Hữu Phúc xác nhận 5 lọ chuẩn và 15 lọ lỗi, mỗi lọ xuất hiện dưới bảy điều kiện sáng. Ghép theo (good/bad, tiền tố ID), lưu mask riêng cho từng ảnh. Khi chia tập hoặc tính độ bất định, giữ toàn bộ ảnh của cùng lọ trong cùng nhóm. [Bàn giao định danh lọ](https://github.com/huyryan220806/robust-memory-ad/blob/56b0221824c457aa748d2e10638c8e11581f82f8/docs/specimen-handoff.md)

## Đỗ Hoàng Phúc · quyết định về giao thức đánh giá

Dùng v0.3 làm giao thức **thăm dò pilot01**. Duyệt cách hiệu chỉnh riêng từng run trên 41 ảnh validation chuẩn. Giữ lưới α ∈ {1%, 5%, 10%} và ba mức diện tích {0,1%, 0,5%, 1%}. Khi báo cáo một điểm vận hành, đọc α = 5% và mức 0,5% kèm cả lưới. Quyết định này áp dụng cho phát triển tiếp theo; cần khóa quy tắc trước khi chạy trên tập xác nhận độc lập. [Giao thức chi tiết](https://github.com/huyryan220806/robust-memory-ad/blob/56b0221824c457aa748d2e10638c8e11581f82f8/docs/evaluation.md)

- Định nghĩa lỗi nhỏ theo mọi vùng liên thông 8 trong mask gốc, mẫu số diện tích là toàn ảnh gốc. Ảnh chứa cả vùng lớn và nhỏ thuộc tập “tất cả lỗi”, tách khỏi nhóm “chỉ có lỗi nhỏ”. Đo phát hiện ở cấp ảnh và đo định vị trên map thành hai đầu ra.
- Đã chạy 11 unit test của nhánh đánh giá và đối soát 216 hàng tổng hợp với các quyết định theo ảnh. Độc lập tính lại AUROC cho cả ba run trực tiếp từ CSV Drive bằng xác suất điểm ảnh lỗi xếp trên ảnh chuẩn, tính ½ cho hai điểm bằng nhau. Tái tạo ngưỡng 5%, số báo nhầm, số phát hiện và nhóm lỗi nhỏ. Đã chạy thêm script Anomalib của Hoàng Phúc trên cùng CSV. AUROC khớp scikit-learn và phép tính theo thứ hạng ở cả ba run, sai số lớn nhất 4,21 × 10⁻⁸. Pixel AP trích từ bảng của Hoàng Phúc. [Đối chiếu Anomalib](anomalib-recheck.json) [Log kiểm độc lập](verification.json)
- Ở α = 5%, cả hai báo nhầm của greedy thuộc underexposed (2/5 ảnh chuẩn). Năm ảnh chuẩn trong điều kiện này tương ứng năm lọ chuẩn cũng có mặt ở các điều kiện khác. Phần thảo luận dùng số đếm và phạm vi pilot.
- Đối soát theo từng lọ ở α = 5%: từ sáng chuẩn sang thiếu sáng, greedy bật thêm cảnh báo cho 2/5 lọ chuẩn và 3/15 lọ lỗi, không có lọ nào đổi theo chiều ngược lại. Full và random giữ nguyên quyết định từng lọ khi so hai điều kiện này. 21 ảnh của nhóm lỗi nhỏ 0,5% chỉ thuộc 3 lọ bad/007, 010 và 011. [Bảng đếm ghép cặp](verification.json)
- Tạm giữ AU-PRO khỏi bảng kết quả thật cho tới khi Hoàng Phúc giải thích được chênh lệch giữa toy và triển khai thư viện. Crop 224 vẫn là cấu hình của pilot01. Thử resize trực tiếp 256 cần sinh lại map và hiệu chỉnh lại ngưỡng trên validation.
- Tập vial/test_public đã sử dụng cho phát triển. Đánh dấu thăm dò mọi quyết định phát sinh từ tập này. Với đánh giá xác nhận, nhóm ghi danh sách mẫu hoặc category giữ lại và phiên bản cấu hình trước khi mở nhãn, hoặc dùng dịch vụ chấm private theo điều kiện benchmark.

## Lê Quyết Tiến · quyết định cho lượt đo L4

Đã chạy lại 7 test của PR ban đầu và 18 test của nhánh profiling cập nhật. Smoke CPU riêng dùng 10 warm-up và 50 mẫu. Tính lại từ 50 mẫu gốc của Tiến được p50 = 0,409150 ms và p95 = 0,651060 ms cho phép cộng mô phỏng. Kết luận review cục bộ: bộ ghi log và cách tính phân vị có thể chuyển sang bước tích hợp. Tiến đã kiểm cả ba bank trên CPU. Index cộng tham số lần lượt là 934.478.099, 9.343.251 và 9.343.251 byte. Huy đã chuẩn bị [notebook chạy ba bank trên Colab L4](pilot01-l4.ipynb). Đã kiểm đoạn nạp bank, tiền xử lý, trả score/map và ghi log trong notebook bằng một PNG và bank greedy trên CPU. Lượt đo thật cần một phiên Colab L4 có quyền truy cập Drive. [Bảng byte và score CPU của Tiến](https://github.com/huyryan220806/robust-memory-ad/blob/15ab4dac972d60885aa6ec229d68727d52523bfe/docs/evidence/handover-20261126/bank-sizes-and-cpu-reload.csv) [PR 2](https://github.com/huyryan220806/robust-memory-ad/pull/2)

| Thông số Tiến cần | Quyết định cho lượt tiếp theo |
|---|---|
| Máy và truy vấn | NVIDIA L4 trên Colab theo bàn giao của Tiến. Backbone chạy L4; dùng FAISS CPU 4 luồng như baseline. Ghi model CPU phiên Colab, driver, CUDA, dtype và phiên bản thư viện. |
| Input/output tính giờ | Một ảnh RGB đã giải mã trong RAM → resize/crop/normalize → chuyển thiết bị → backbone → truy vấn → điểm và map 224² về RAM. Batch 1. Nạp bank, tải tệp và giải mã ảnh trước đồng hồ. |
| So điểm trước khi đo | Dùng CSV bàn giao làm tham chiếu. Dùng sai số tuyệt đối GPU ≤ 1e-4 và CPU ≤ 1e-3 như phép kiểm baseline, rtol = 0. Khi lệch, lưu expected/actual, cấu hình và dừng trước khi tính độ trễ. |
| Danh sách và số lần | Dùng cùng một danh sách ID, cùng thứ tự cho mọi run. Đo một lượt qua đủ 140 ảnh test_public sau 10 lượt warm-up cho từng cấu hình. Lặp thêm phiên nếu p95 dao động. Lưu từng mẫu, hash ảnh và điểm đối chiếu. |
| Khóa ghép dữ liệu | Ghép `quality.run_id` với `efficiency.run_id`. Mỗi lần đo có measurement_id riêng, kèm device, nhóm ảnh và phạm vi tính giờ. Baseline commit 6d7e8d8. |
| Phạm vi bộ nhớ | Thống kê index byte, metadata cần nạp và backbone riêng. Báo peak RAM/VRAM theo đúng phạm vi công cụ. Giữ cả log raw, p50, p95 và n. |

Trước khi có bảng ba cấu hình đo cùng phiên L4, trạng thái đánh giá ràng buộc độ trễ là **chờ đo**. Thời gian truy vấn theo batch trên cache của baseline chưa bao gồm tiền xử lý, trích đặc trưng và toàn bộ chi phí trả ảnh.

## Lê Nam · buổi đọc cùng Huy

Đã chuẩn bị [phiếu đọc có mẫu](phieu-doc-nam.md) từ hai PDF của tác giả. Nam nhận bài EfficientAD để tự điền một câu. Huy rà lại câu trả lời khi nhận được. Việc gửi cho Nam và buổi trao đổi trực tiếp thuộc phần phối hợp giữa hai thành viên.

## Hai ưu tiên tuần tới

1. Huy vận hành notebook L4 và chuyển log của ba run để Tiến ghép với bảng chất lượng. Từ đó xác lập p95 tối đa khả thi và giới hạn bộ nhớ ngoài index.
2. Huy cùng hai bạn Phúc chọn tập xác nhận tách theo mẫu vật, chốt biên độ suy giảm độ nhạy cho phép và các đối chứng trước khi huấn chỉnh bộ chọn mới.

## Cách lặp lại phần đối soát

Từ clone của repo đã fetch các nhánh, chạy `python scripts/verify_week2_handover.py <manifest_images.csv> <predictions.csv>`. Script đọc các Git blob theo SHA cố định và xác nhận hash của hai CSV trước khi tính. [Drive bàn giao trong hướng dẫn đánh giá](https://github.com/huyryan220806/robust-memory-ad/blob/56b0221824c457aa748d2e10638c8e11581f82f8/docs/evaluation.md)

### Chạy lại trên máy Huy

Mở PowerShell ở thư mục gốc `Big_Project`. Hai CSV, ảnh PNG, trọng số V1 và bank greedy đã kiểm hash nằm trong `.week2-work/handover`. Checkout mã tác giả và các môi trường Python nằm trong cùng thư mục làm việc. Các số đo từ phiên L4 cần được lưu riêng theo measurement ID.

```powershell
$inputs = Resolve-Path '.week2-work/handover'
& ./.week2-work/baseline/Scripts/python.exe ./robust-memory-ad/scripts/replay_greedy_image.py "$inputs/000_regular.png" "$inputs/wrn.pth" "$inputs/greedy.faiss" "$inputs/predictions.csv" ./.week2-work/patchcore-inspection
& ./.week2-work/evaluation/Scripts/python.exe ./robust-memory-ad/scripts/verify_week2_handover.py "$inputs/manifest.csv" "$inputs/predictions.csv"
& ./.week2-work/evaluation/Scripts/python.exe ./.week2-work/evaluation-source/scripts/check_anomalib_auroc.py --predictions "$inputs/predictions.csv" --output ./.week2-work/anomalib-latest.json
```
