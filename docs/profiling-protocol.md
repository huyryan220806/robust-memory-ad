# Giao thức profiling đã chốt — 27/11/2026

Huy đã xác nhận trực tiếp qua Tiến các điều kiện dưới đây. Quyết định cũng được ghi trên nhánh của Huy ở [bàn giao tuần 2](https://github.com/huyryan220806/robust-memory-ad/blob/bb443f244770e74681fc1ad46c6dfc1c50f1020c/docs/week2/ban-giao.md). Bản đọc bằng máy: [configs/profiling/pilot01-l4.json](../configs/profiling/pilot01-l4.json).

**Trạng thái:** đã khóa phép đo; **chưa nhận log L4 thực đo**. Chưa có p50/p95 hoặc VRAM/RAM benchmark để điền. Huy vận hành train, Colab và quyền truy cập Drive; Tiến chuẩn bị/kiểm công cụ và tổng hợp log. Không cần Tiến thao tác Colab.

## Điều kiện cố định

| Mục | Đã chốt |
|---|---|
| Thiết bị | Backbone NVIDIA L4, FAISS CPU IndexFlatL2 1-NN, 4 luồng |
| Batch, dtype | 1, float32; TF32 tắt theo bàn giao baseline |
| Bắt đầu đồng hồ | Ảnh RGB đã giải mã hoàn toàn trong RAM |
| Trong đồng hồ | Resize vuông 256, crop 224, normalize, chuyển thiết bị, backbone, FAISS CPU, score, nội suy và làm mượt map |
| Kết thúc đồng hồ | Cả score và map 224×224 đã sẵn trong RAM CPU; đồng bộ GPU trước t0 và sau tính toán cuối |
| Ngoài đồng hồ | Tải/đọc đĩa, decode/convert RGB, nạp model/bank, ghi CSV |
| Ảnh | Đúng 140 vial/test_public: 7 điều kiện sáng × 20 ảnh, cùng thứ tự ở cả ba run |
| Làm nóng, số mẫu | 10 warm-up mỗi bank, sau đó đo mỗi ảnh một lần: 140 mẫu/run |
| Đối chiếu trước đo | Đủ 140 score, sai số tuyệt đối ≤1e-4, rtol=0; không đo tiếp khi lệch |
| RAM | RSS tiến trình, baseline và peak quan sát lấy mẫu 5 ms; ghi rõ có thể bỏ sót peak ngắn |
| VRAM | Riêng allocator PyTorch: baseline/peak allocated và reserved, reset sau warm-up; không gọi là tổng VRAM hệ thống |
| FAISS GPU | Một lượt/protocol riêng, bổ sung VRAM theo PID; không ghép vào bộ CPU FAISS này |

| run_id | Patch | Index byte | Params byte |
|---|---:|---:|---:|
| pilot01-full-s0 | 228144 | 934477869 | 230 |
| pilot01-random-s0 | 2281 | 9343021 | 230 |
| pilot01-greedy-s0 | 2281 | 9343021 | 230 |

Index đã chứa vector. Không cộng feature cache hoặc selected_indices vào dung lượng bank triển khai; trọng số backbone tách riêng. Các số byte và hash đã kiểm trên tệp thật ngày 26/11, chưa phải số peak RAM/VRAM.

## Đầu ra Huy chuyển cho Tiến

Huy đã có [notebook đo ba bank](https://github.com/huyryan220806/robust-memory-ad/blob/bb443f244770e74681fc1ad46c6dfc1c50f1020c/docs/week2/pilot01-l4.ipynb) dùng bộ đo của Tiến tại commit `15ab4da`. Công cụ tổng hợp mới đọc đúng định dạng này, nên không cần sửa notebook hoặc tạo một luồng chạy khác.

Nhận nguyên thư mục kết quả của một phiên, gồm `hardware.txt` ở gốc và ba thư mục theo run ID. Trong mỗi run cần:

- `config.json`, `environment.json`: commit, thiết bị, thư viện, batch, phạm vi và danh sách ảnh.
- `artifacts.json`, `images.json`, `validation.json`: hash bank/ảnh và điểm đối chiếu của 140 ảnh.
- `warmup.csv`, `timings.csv`: 10 lượt làm nóng riêng và 140 mẫu ms sau warm-up.
- `rss_samples.csv`, `memory.json`: mẫu RSS gốc và bộ đếm allocator GPU.
- `efficiency.csv`, `stability.json`, `status.json`: tổng hợp và trạng thái lượt đo.

Mỗi lượt lặp có measurement_id mới và thư mục riêng. Giữ mọi log gốc, cả lượt lỗi. Chỉ nhận một bộ so sánh khi đủ ba run `status=complete`, cùng ảnh/thứ tự/hash và metadata môi trường. Không tự trộn các phiên khác nhau. Hardware log của notebook có CPU và phiên bản torch CUDA; environment có NVIDIA model/driver/RAM/thư viện. Cờ TF32 được notebook đặt false; định dạng environment cũ chưa ghi độc lập cờ này, nên audit sẽ ghi giới hạn đó.

## Lệnh Tiến dùng khi đã nhận log

Từ gốc repo, trong môi trường có phụ thuộc profiling:

```bash
python -m scripts.collect_profiling --logs-root PATH_TO_HUY_LOG_FOLDER --predictions artifacts/pilot01/predictions.csv --output outputs/collected-UNIQUE_SESSION
```

Muốn ghép cùng quality.csv của Hoàng Phúc trong một lần:

```bash
python -m scripts.collect_profiling --logs-root PATH_TO_HUY_LOG_FOLDER --predictions artifacts/pilot01/predictions.csv --quality PATH_TO_QUALITY.csv --quality-commit 56b0221824c457aa748d2e10638c8e11581f82f8 --output outputs/collected-UNIQUE_SESSION
```

Bộ nhận log kiểm hash predictions/bank, cấu hình, 140 ảnh đủ nhóm sáng, điểm đối chiếu, số dòng/thứ tự, warm-up, p50/p95 tính lại, RSS peak so mẫu raw và quan hệ baseline/peak allocator. Khi không khớp sẽ dừng trước khi tạo bảng cuối. Không chạy model, train hoặc Colab; không tạo số thay thế khi chưa có log.

Đầu ra gồm:

- `efficiency.csv`: ba hàng, mỗi hàng một measurement; thêm RAM baseline/peak/tăng thêm và VRAM PyTorch allocated/reserved baseline/peak/tăng thêm. Đường dẫn mỗi hàng truy tới log được giữ nguyên trong `raw/<run_id>/`.
- `quality-efficiency.csv` khi có quality: nối theo run_id, giữ các tham số giao thức chất lượng, measurement_id/device và ô thiếu. Một số đo toàn run có thể xuất hiện cạnh nhiều hàng chất lượng; không cộng lặp. Không cộng `lighting_group=all` với nhóm con.
- `collection-audit.json`, `collection-status.json`: hash nguồn, giao thức áp dụng, số hàng và giới hạn công cụ; trạng thái complete ghi sau cùng.

Phân vị vẫn là nội suy tuyến tính ở `(n-1)*q`. Không xóa outlier. `stability.json` chỉ là chẩn đoán; nếu p95 dao động, trao đổi với Huy để lặp cùng ca đo và ghi lý do. Peak RSS là peak lấy mẫu của tiến trình, gồm chi phí logger và có thể gồm bộ nhớ giữ lại từ lượt trước; báo baseline/độ tăng kèm peak tuyệt đối.

Đã chạy **26 unit test** của repo; tám test mới của bộ nhận log dùng dữ liệu mô phỏng trong thư mục tạm, không phải đo trên L4. Sau khi Huy chia sẻ log thật, Tiến mới kiểm, tổng hợp và hoàn tất bảng hiệu năng tuần 2. Thời gian chọn coreset là log một lần do Huy/Phúc cung cấp; không gộp vào độ trễ từng ảnh.
