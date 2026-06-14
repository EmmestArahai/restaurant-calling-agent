# =============================================================
# Dockerfile
# Mục đích: Đóng gói toàn bộ ứng dụng vào một container
# =============================================================

# ── 1. Base image — Python 3.11 phiên bản nhẹ (slim) ─────────
# slim = không có các công cụ thừa → image nhỏ hơn ~200MB
FROM python:3.11-slim

# ── 2. Đặt thư mục làm việc bên trong container ──────────────
# Mọi lệnh tiếp theo đều chạy từ /app
WORKDIR /app

# ── 3. Copy requirements trước — tận dụng Docker cache ───────
# Docker cache layer: nếu requirements.txt không đổi,
# bước pip install sẽ được cache lại → build lần sau nhanh hơn
COPY requirements.txt .

# ── 4. Cài thư viện ───────────────────────────────────────────
# --no-cache-dir: không lưu cache pip → giảm kích thước image
RUN pip install --no-cache-dir -r requirements.txt

# ── 5. Copy toàn bộ source code vào container ─────────────────
COPY . .

# ── 6. Tạo thư mục cần thiết bên trong container ─────────────
RUN mkdir -p models logs

# ── 7. Huấn luyện model ngay khi build image ─────────────────
# Đảm bảo file models/intent_model.pkl luôn có sẵn trong image
RUN python scripts/train_model.py

# ── 8. Mở cổng 8000 để bên ngoài có thể truy cập ─────────────
EXPOSE 8000

# ── 9. Lệnh chạy khi container khởi động ─────────────────────
# host=0.0.0.0 bắt buộc phải có trong Docker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]