# Bằng chứng tự kiểm dụng cụ — không phải benchmark mô hình

`smoke-cpu-20261125/` là log của phép cộng giả lập CPU trên máy Tiến: nạp adapter một lần, 10 warm-up, 50 mẫu đo. Không có ảnh MVTec, bank hay PatchCore; **không suy ra hiệu năng L4 hoặc chất lượng phát hiện lỗi từ đây**.

Mở `status.json` để biết lượt hoàn tất, `timings.csv` để tính lại p50/p95 trong `efficiency.csv`, `warmup.csv` để xem dãy làm nóng. `memory.json` và `rss_samples.csv` chỉ mô tả RSS tiến trình của phép thử giả lập.

Mã đo có SHA-256 trong `environment.json`. Commit môi trường có thể là commit trước thay đổi này; `profiling_git_status` ghi rõ cây mã đang thay đổi khi tạo bằng chứng. Không xem hash commit đó là commit chứa toàn bộ phần profiling.

Kiểm tra lại bằng `python -m unittest discover -s tests -v` và một thư mục output mới với `--smoke`. Thời gian thật có thể khác giữa các lượt; phép kiểm yêu cầu log và thống kê nhất quán, không yêu cầu khớp các ms của máy Tiến.
