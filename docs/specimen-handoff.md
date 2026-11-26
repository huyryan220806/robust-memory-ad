# Xác nhận mẫu vật và tác động tới đánh giá

Thông tin do Hữu Phúc bàn giao qua tin nhắn cho Đỗ Hoàng Phúc ngày 26/11/2026; đây là xác nhận từ người bàn giao sau khi kiểm ảnh gốc, chưa phải kiểm tra độc lập của nhóm đánh giá.

- Trong cùng nhánh good hoặc bad, bảy ảnh cùng tiền tố là cùng một lọ vật lý, cùng cảnh và camera cố định, thay đổi điều kiện chiếu sáng/phơi sáng.
- Test public gồm 5 lọ good và 15 lọ bad, mỗi lọ bảy ảnh. Khóa mẫu vật phải bao gồm cả nhánh và tiền tố: `good/000` và `bad/000` khác lọ. Dùng mask PNG riêng cho từng ảnh vì đường biên có thể khác nhẹ giữa các điều kiện.
- Hữu Phúc mô tả mẫu bad/014 có một lỗi ở giữa thân và một lỗi gần mép nắp; crop cắt mất một phần vùng gần mép. Điều này phù hợp với kiểm đếm trước đó rằng bảy mask bad/014 mất một phần pixel sau crop, nhưng mô tả vị trí và nguyên nhân chưa được kiểm độc lập ở đây.

## Hệ quả cho phân tích

Mỗi nhóm ánh sáng có 20 ảnh từ 20 lọ khác nhau; bảy nhóm ánh sáng là **đo lặp trên cùng 20 lọ**. Điều đó không tự chứng minh 20 lọ độc lập thống kê: cần biết cách lấy mẫu, lô sản xuất và quan hệ giữa train, validation và test. So sánh ghép cặp theo `(nhánh, tiền tố)` loại khác biệt danh tính mẫu giữa hai ảnh trong cặp, nhưng không tự chứng minh mọi chênh lệch hoàn toàn do ánh sáng: cần kiểm soát thứ tự chụp, thời gian và các thay đổi đồng thời khác.

Đối với phép so sánh theo cặp, giữ cùng mẫu vật ở mọi điều kiện, báo các thay đổi score/quyết định theo từng lọ và số cặp đầy đủ. Với khoảng tin cậy gộp ảnh, có thể **đề xuất** bootstrap theo cụm mẫu vật, lấy lại cả bảy quan sát của lọ; với AUROC cần xem xét lấy mẫu phân tầng good/bad để duy trì hai lớp và dùng cùng cụm cho các phương pháp/ánh sáng khi so sánh. Chỉ 5 lọ good và 15 lọ bad nên độ bất định có thể lớn; phương pháp, số lần lấy mẫu và diễn giải cần nhóm duyệt trước. Chưa tính hay báo khoảng tin cậy trong lần cập nhật này.

## Đề xuất bỏ center crop

Resize thẳng 256 × 256 có thể tránh loại bỏ dải biên do center crop, nhưng **không đảm bảo giữ nguyên 100% diện tích pixel gốc hoặc mọi vùng lỗi** sau downsampling. Cần kiểm từng thành phần và so sánh trong cùng hệ tọa độ. Hơn nữa đây là thay đổi preprocessing của mô hình: phải chạy lại dự đoán/map tương ứng ở độ phân giải mới, hiệu chỉnh lại ngưỡng bằng validation và kiểm lại mask căn chỉnh. Không chỉ đổi mask hoặc kéo map 224 lên 256 rồi xem như cùng thí nghiệm. Cần thống nhất cấu hình mới với Huy và Hữu Phúc trước khi đối chiếu; kết quả cũ vẫn là pilot crop 224.
