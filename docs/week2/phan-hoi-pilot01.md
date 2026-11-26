# Huy trả lời Hoàng Phúc và Tiến về pilot01

Nguyễn Đình Huy · 26/11/2026 · MVTec AD 2, nhóm vial, seed 0

## Đỗ Hoàng Phúc

Phúc dùng giao thức v0.3 cho phần đánh giá thăm dò của pilot này nhé. Mình chốt ba điểm để bạn ghi vào báo cáo.

1. **Ngưỡng cảnh báo.** Hiệu chỉnh riêng từng run trên cùng 41 ảnh validation chuẩn. Mốc chính là α = 5%. Sắp 41 điểm tăng dần, lấy điểm thứ 39 (đếm từ 1), rồi lấy số float64 kế tiếp lớn hơn nó bằng `nextafter(s[38], +inf)`. Cảnh báo khi `score ≥ ngưỡng`. Cách này cho tối đa 2/41 cảnh báo trên validation, kể cả khi có điểm bằng nhau. Giữ nguyên ngưỡng đó cho cả bảy điều kiện sáng. Báo FPR trên 35 ảnh chuẩn của `test_public` và số đếm trên 5 ảnh chuẩn ở từng điều kiện. Giữ cả α = 1% và 10% trong bảng phân tích độ nhạy.
2. **Lỗi nhỏ và ROI.** Tách các vùng lỗi liên thông 8 hướng trên mask gốc. Chia diện tích từng vùng cho toàn bộ ảnh gốc 1.400 × 1.900 pixel. Ở mốc chính 0,5%, ảnh thuộc nhóm “chỉ có lỗi nhỏ” khi có ít nhất một vùng lỗi và **mọi** vùng đều chiếm tối đa 0,5%. Ảnh có cả lỗi lớn lẫn nhỏ vẫn tính vào recall tất cả lỗi và đánh giá định vị, nhưng bỏ khỏi mẫu số recall “chỉ có lỗi nhỏ”. Báo thêm hai mức 0,1% và 1%. Nếu mẫu số bằng 0, ghi N/A.
3. **Bảy mask bị crop.** Giữ đủ 105 ảnh lỗi khi tính recall. Đánh dấu bảy ảnh của lọ `bad/014` bị mất một phần vùng lỗi ở biên. Khi đo pixel AP, so map với mask đã căn chỉnh trong khung 224 × 224. Khi phân nhóm kích thước lỗi, dùng mask gốc như trên.

Ở mốc 5% và 0,5%, full phát hiện 8/21 ảnh chỉ có lỗi nhỏ, random 5/21 và greedy 7/21. 21 ảnh này thuộc **ba lọ** được chụp lặp dưới bảy điều kiện sáng. Khi so giữa hai điều kiện, ghép theo cùng lọ. Các mốc trên được chốt sau khi xem pilot. Để đánh giá xác nhận, nhóm sẽ khóa trước quy tắc và dùng một tập giữ lại riêng. [Giao thức và cách tính](https://github.com/huyryan220806/robust-memory-ad/blob/56b0221824c457aa748d2e10638c8e11581f82f8/docs/evaluation.md) · [Số liệu Huy đối soát](verification.json)

## Lê Quyết Tiến

Tiến dùng năm quy ước dưới đây nhé. Anh vận hành Colab L4 và bàn giao log gốc để em tổng hợp.

1. **Phạm vi đo.** Đúng, batch 1. Bắt đầu khi ảnh RGB đã giải mã nằm trong RAM. Tính resize/crop/normalize, chuyển thiết bị, backbone, truy vấn FAISS và tạo score/map. Đọc Drive, đọc đĩa, giải mã ảnh và nạp bank thực hiện trước khi bấm giờ.
2. **Đầu ra trên CPU.** Có tính thời gian chuyển về CPU. Dừng đồng hồ khi cả score và map 224 × 224 đã sẵn sàng trong RAM. Đồng bộ CUDA ở hai đầu phép đo. Ghi tệp kết quả sau khi dừng đồng hồ.
3. **Nơi lưu và quyền.** Ảnh nằm ở `data/mvtec_ad_2/vial`, bank ở `outputs/pilot01/runs/<run_id>/bank` trong Drive bàn giao của Hữu Phúc. Huy là đầu mối xử lý quyền đọc nguồn với Hữu Phúc nếu em thiếu quyền. Log mới do anh ghi vào `MyDrive/robust-memory-ad-results/<UTC>/<run_id>/` trên Drive của anh. Sau lượt đo, anh sẽ chia sẻ thư mục phiên cho em quyền xem/tải để ghép bảng. [Drive đầu vào](https://drive.google.com/drive/folders/171zEixi2Xb07Njhon9B_lsEORCbI6kz2)
4. **Dữ liệu và ngân sách.** Dựng bank từ 291 ảnh train chuẩn. Đo trên cùng 140 ảnh `vial/test_public`, đủ bảy điều kiện sáng và cùng thứ tự cho ba run. Full có 228.144 patch. Random và approximate greedy cùng có đúng 2.281 patch. Mỗi bank kiểm score với CSV bàn giao (sai số tuyệt đối GPU ≤ 0,0001), làm nóng 10 lượt rồi đo đủ 140 ảnh. Tính p50/p95 từ 140 mẫu của từng run.
5. **VRAM và FAISS.** Lượt này backbone chạy L4, FAISS `IndexFlatL2` chạy CPU 4 luồng. Ghi peak allocated/reserved của PyTorch sau warm-up, peak RSS quan sát được của tiến trình và byte của index/tham số ở các cột riêng. Peak VRAM toàn tiến trình hiện để “chưa đo”. Nếu thử FAISS GPU, mở run riêng và lấy mẫu bộ nhớ GPU theo PID bằng NVML hoặc `nvidia-smi`; ghi chu kỳ lấy mẫu và báo đây là đỉnh quan sát được. Khi thiết bị không cung cấp số theo PID, để N/A.

Hiện đã kiểm nạp bank trên CPU. p50/p95 và VRAM trên L4 sẽ được điền từ log của phiên chạy thật. [Notebook Huy vận hành](pilot01-l4.ipynb) · [Hướng dẫn và bằng chứng của Tiến](https://github.com/huyryan220806/robust-memory-ad/blob/15ab4dac972d60885aa6ec229d68727d52523bfe/docs/profiling-handover.md)
