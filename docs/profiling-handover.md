# Kiểm bank và ghép chất lượng — bàn giao của Tiến

Cập nhật **26/11/2026**. **Huy phụ trách train và Colab**, theo Tiến xác nhận. Các lệnh dưới đây phục vụ kiểm bank đã bàn giao và xử lý bảng của Tiến; không chạy train, chọn lại coreset, dựng lại bank hoặc notebook Colab.

## Nguồn đã nhận

- Nhánh `baseline/pilot`, commit `6d7e8d8f0127838aae8f9c2dd3f2c85a87b94141`: dataset adapter, ba config và scripts baseline của Phúc. Nhánh này hiện đã có trên repo chung. Phần profiling dùng lại mã đó; các script train/cache giữ nguyên.
- PatchCore tác giả: `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`; verifier từ chối checkout khác hoặc sửa `src`.
- [Thư mục Drive bàn giao](https://drive.google.com/drive/folders/171zEixi2Xb07Njhon9B_lsEORCbI6kz2): bank, metadata, predictions và dữ liệu. Không đưa bank/ảnh/trọng số lên Git.
- [quality.csv của Hoàng Phúc](https://github.com/huyryan220806/robust-memory-ad/blob/ae33ffaa53b50da82bce7e6d343778ee35d5e8a4/results/evaluation/pilot01/quality.csv): 216 hàng, 72 hàng mỗi run; giao thức vẫn `exploratory_pending_team_review`.

## 1. Lấy đúng đầu vào kiểm nạp bank

`configs/pilot/handover-assets.json` liệt kê Drive file ID và byte thực từ metadata. Bộ này chỉ lấy bank đã có, các tệp bàn giao và **một ảnh đánh giá**; không lấy feature matrix hoặc ảnh train để huấn luyện.

```bash
python -m pip install -r requirements-baseline-check.txt
python -m scripts.download_pilot_assets --root PATH_TO_HANDOVER
```

Dùng môi trường baseline có PyTorch/torchvision phù hợp. Bộ tải dùng link Drive gốc; không lưu token hoặc URL phiên vào log. Index full khoảng 934 MB nên tải qua mạng có thể lâu. Tệp hiện có được kiểm byte rồi giữ nguyên; chạy verifier để kiểm tiếp hash, nội dung index và score. Tệp tải dở được giữ với đuôi `.download-part`, không âm thầm coi là hoàn tất.

## 2. Nạp lại và đối chiếu một ảnh

```bash
python -m scripts.verify_bank_reload --patchcore-root PATH_TO_PINNED_PATCHCORE --pilot-root PATH_TO_HANDOVER/artifacts/pilot01 --data-root PATH_TO_HANDOVER/data/mvtec_ad_2 --device cpu --output outputs/bank-reload-UNIQUE_ID.json
```

Đường dẫn dùng file ID/image ID, không dùng đường dẫn `D:/paper1` của máy Phúc. Mặc định kiểm ba run `pilot01-full-s0`, `pilot01-random-s0`, `pilot01-greedy-s0`. Có thể chọn `--run-ids` để kiểm một phần và ghi rõ phần đã kiểm.

Ảnh tham chiếu: `vial/test_public/bad/011_overexposed`. Quy trình dùng **đúng transform của dataset adapter** rồi `_predict` của PatchCore, không dùng feature cache để thay suy luận ảnh. Backbone lấy trọng số ImageNet có sẵn; đây là nạp trọng số để suy luận, không huấn luyện.

Verifier kiểm: commit nguồn, params plain values, IndexFlatL2 1024 chiều, số patch, indices int64 duy nhất và đúng miền, byte index, score hữu hạn và map 224×224 hữu hạn. Báo cáo lưu hash index/params/indices/ảnh/predictions/trọng số và nguồn verifier.

Điều kiện điểm là **sai số tuyệt đối**, `rtol=0`: CPU ≤1e-3; CUDA ≤1e-4. TF32 tắt. Script không tự đổi CUDA thành CPU; `--require-l4` chỉ dành cho người vận hành máy L4. Bản sửa `verify_handover_pilot.py` cũng dùng mặc định ≤1e-4 cho GPU và từ chối image ID chỉ định không tồn tại.

**Kết quả kiểm bank không phải số liệu latency/memory benchmark.** Không lấy thời gian tải tệp, nạp trọng số hoặc nạp index làm thời gian xử lý một ảnh.

## 3. Ghép quality.csv theo câu trả lời của Hoàng Phúc

Một hàng chất lượng được phân biệt bằng:

`run_id, category, split, lighting_group, target_calibration_fpr, threshold, small_area_cutoff`.

Phần chất lượng hiện không có `image_group`. Bộ ghép nối theo **run_id**, giữ nguyên bảy trường trên, `missing_reason`, ô chỉ số trống và trạng thái thăm dò. Phía hiệu năng giữ `measurement_id`, `device` và toàn bộ số/log gốc. Nếu metadata hiệu năng trùng tên cột chất lượng, nó được ghi với tiền tố `profiling_` để giữ rõ phạm vi.

```bash
python -m scripts.join_quality --efficiency PATH_TO_EFFICIENCY.csv --quality PATH_TO_QUALITY.csv --scope run --quality-commit ae33ffaa53b50da82bce7e6d343778ee35d5e8a4 --output outputs/quality-efficiency-UNIQUE_ID.csv
```

`--scope run` xác nhận đây là hiệu năng **toàn run**, được đặt cạnh nhiều hàng chất lượng. Script gắn khóa `profiling_aggregation_key` và cờ lặp; **không cộng hoặc trung bình số đo**. Với một measurement của mỗi baseline, bảng chất lượng hiện tạo 216 hàng ghép nhưng vẫn chỉ có ba measurement hiệu năng. Nếu đo lặp hoặc thêm thiết bị thì mỗi measurement giữ riêng.

Không cộng hàng `lighting_group=all` với các nhóm con. `--lighting-group all` chỉ lọc rõ hàng toàn tập, không chọn ngưỡng hoặc mức cắt giúp nhóm. Không thay ô trống bằng 0, không tự chốt các ngưỡng thăm dò. Khóa chất lượng trùng, measurement trùng, run không khớp hoặc số liệu `instrumentation_smoke` đều làm script dừng.

`*.csv.audit.json` lưu hash đầu vào, commit nguồn chất lượng nếu được cung cấp, số measurement, số hàng ghép và xác nhận không thực hiện phép gộp. Đợt kiểm schema trên 216 hàng thật dùng metadata giả lập không có ms/MB; không tạo kết quả hiệu năng giả.

## 4. Phần còn chờ Huy

Phạm vi đo chính thức chưa được chốt: ảnh RAM sau decode/convert RGB, tiền xử lý, các chuyển thiết bị bắt buộc của pipeline và thời điểm score/map CPU sẵn. Bộ đo độc lập đã có nhưng chưa có benchmark PatchCore chính thức của Tiến. Huy vận hành train/Colab; Tiến bàn giao công cụ và nhận log gốc để tổng hợp sau khi thống nhất phạm vi.
