# Bằng chứng kiểm bàn giao ngày 26/11/2026

- `random-greedy-cpu-reload.json`: nạp hai bank Phúc giao, chạy ảnh `vial/test_public/bad/011_overexposed` qua transform/backbone/query/map trên **CPU**, so với predictions.csv. Random lệch 4.283691406e-6; greedy lệch 3.508789063e-6; đều ≤1e-3. Báo cáo có hash đầu vào, trọng số và nguồn bộ kiểm. Không phải số liệu p50/p95 hoặc số đo L4.
- `quality-schema-audit.json`: đọc đúng quality.csv ở commit nguồn được ghi, xác nhận 216 hàng/72 hàng mỗi run và phép ghép giữ ô trống. Metadata dùng để kiểm cấu trúc, **không có measurement hiệu năng thật** và không sinh ms/MB giả.

Script train/cache của Phúc không được sửa hoặc chạy trong đợt kiểm này. Huy phụ trách train và Colab. Trạng thái kiểm bank full sẽ được ghi riêng khi có đủ index và chạy xong; không suy từ kết quả hai bank nhỏ.
