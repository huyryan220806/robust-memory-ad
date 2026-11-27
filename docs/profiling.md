# Bộ đo profiling độc lập — Lê Quyết Tiến

Cập nhật **27/11/2026**. Huy đã [chốt giao thức L4](profiling-protocol.md): batch 1, 140 ảnh test_public, RGB RAM → score/map CPU, backbone L4 và FAISS CPU 4 luồng. Dụng cụ đo đã tự kiểm trên CPU; [ba bank đã kiểm nạp](profiling-handover.md). **Chưa nhận log benchmark L4 thực đo.** Huy phụ trách train/Colab; Tiến kiểm và tổng hợp log bằng `scripts.collect_profiling`.

## Phần đã hoàn thành và có trong PR

- `scripts/profile.py`: ghi thời gian bằng đồng hồ đơn điệu, xuất từng mẫu CSV, tách warm-up và tính p50/p95.
- Liệt kê byte/SHA-256 các tệp được khai báo, tách vai trò bank/index, backbone, metadata và bản trung gian; quy đổi 1 MB = 1.000.000 byte.
- Ghi môi trường, cấu hình, trạng thái lượt đo và RSS tiến trình lấy mẫu; giữ log khi phép kiểm đầu ra thất bại, không ghi đè thư mục lượt cũ.
- Helper `measure_selection` ghi thời gian một callback chọn coreset với metadata do bên gọi cung cấp; đã kiểm bằng callback giả lập.
- Bảy phép kiểm tự động và log smoke CPU gồm 10 warm-up + 50 mẫu đo.

Đã nhận bank/config và schema chất lượng từ hai bạn Phúc, cùng phạm vi timing được Huy xác nhận. PR có công cụ kiểm bank, nhận log L4 và ghép chất lượng theo schema đã xác nhận. Dùng notebook do Huy phụ trách; không bổ sung notebook Colab vào phần việc của Tiến.

## Chạy phần đã tự kiểm

Từ thư mục gốc repo:

```bash
python -m pip install -r requirements-profiling.txt
python -m unittest discover -s tests -v
python -m scripts.profile --smoke --output outputs/smoke-cpu-UNIQUE_ID
```

Thay `UNIQUE_ID` mỗi lần chạy; thư mục đã tồn tại sẽ bị từ chối để giữ nguyên log trước đó. Lệnh smoke không cần PyTorch hoặc dữ liệu mô hình. Nó dùng phép cộng số nguyên giả lập CPU, không đọc ảnh, không nạp bank và không chạy anomaly detection.

## Cách tính và log đã xác minh

Đơn vị độ trễ: **ms**, từ chênh lệch `time.perf_counter_ns()`. Hàm `timed_call` gọi callback đồng bộ trước đồng hồ và sau callback công việc; phép kiểm xác nhận thứ tự này bằng mô phỏng, chưa xác minh CUDA thật.

Phân vị dùng nội suy tuyến tính ở vị trí `(n-1)*q` của dãy đã sắp xếp. Dãy `8,9,10,11,12` có **p50 = 10 ms**, **p95 = 11,8 ms**. Dãy âm, rỗng hoặc không hữu hạn bị từ chối.

Log tham khảo: `docs/evidence/smoke-cpu-20261125/`. Đã tính lại p50/p95 từ 50 dòng `timings.csv` và đối chiếu hash nguồn bộ đo trong `environment.json`. Đây là log tự kiểm, không phải bảng hiệu năng mô hình.

| Tệp | Nội dung |
|---|---|
| `config.json`, `environment.json` | Tham số thử và môi trường thực; commit, trạng thái cây mã và hash bộ đo |
| `validation.json`, `status.json` | Đối chiếu đầu ra giả lập và trạng thái hoàn tất/thất bại |
| `warmup.csv` | Các lượt làm nóng, không đưa vào phân vị |
| `timings.csv`, `efficiency.csv` | Từng mẫu sau warm-up và bảng tổng hợp có thể tính lại |
| `rss_samples.csv`, `memory.json` | RSS trước vòng đo, các mẫu RSS, peak quan sát và phạm vi công cụ |
| `artifacts.json`, `images.json` | Danh mục/hash theo đầu vào; smoke không có bank hoặc ảnh |
| `stability.json` | p95 hai nửa dãy để xem xét; không tự kết luận số đo ổn định |

Môi trường smoke: Windows 11, i5-12450H, RAM 16.866.680.832 byte, Python 3.12.5, psutil 7.2.2. PyTorch đang cài là bản CPU; RTX 4050 trên máy không được dùng trong smoke.

## Giao diện kỹ thuật của dụng cụ

`run(config, adapter, output, base)` là giao diện tổng quát của bộ đo, chưa phải pipeline nhóm đã tích hợp. Adapter cung cấp `load(config, base)`, `decode(spec, base)`, `predict(input)`, `image_score(result)` và `synchronize()`. Bộ đo gọi load một lần, đối chiếu điểm trước warm-up và giữ warm-up ngoài CSV đo chính. Đường dẫn đầu vào tương đối với thư mục config.

Các trường `run_id`, `measurement_id`, `image_group`, `device` trong CSV là metadata do bên gọi cung cấp. Giao thức pilot dùng run_id baseline, measurement_id riêng từng lượt và category=vial/split=test_public để ghép quality. Hiệu năng toàn run được đặt cạnh các hàng lighting/threshold/cutoff có phân biệt đầy đủ. Không dùng giá trị giả lập làm số liệu mô hình thật.

## Phạm vi xác minh và giới hạn

Đã kiểm: phân vị, thứ tự đồng bộ/đồng hồ qua mock, nạp một lần, số dòng và thứ tự mẫu, tách warm-up, dừng khi điểm tham chiếu sai, không ghi đè output, kiểm byte theo vai trò và helper chọn coreset. Bảy phép kiểm hiện tại không yêu cầu baseline từ thành viên khác.

RSS dùng psutil lấy mẫu mỗi 5 ms trong tiến trình hiện tại. Peak quan sát có thể bỏ sót cấp phát ngắn và gồm cả chi phí ghi log; không bao quát tiến trình con. Nhánh CUDA và bộ đếm PyTorch có trong dụng cụ tổng quát nhưng **chưa được chạy xác minh trên GPU**; không có khẳng định về tổng peak VRAM, FAISS hoặc kết quả L4 từ bản bàn giao này.

Phạm vi đo, tolerance và cấu hình chính thức hiện có tại [giao thức đã chốt](profiling-protocol.md). Bộ nhận log được kiểm bằng fixture mô phỏng; chỉ xuất bảng L4 sau khi Huy giao đủ log thật đạt kiểm tra. Các bản nháp cũ giữ cục bộ để tra lịch sử.
