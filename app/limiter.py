# =============================================================
# app/limiter.py
# Mục đích: Cấu hình Rate Limiter dùng chung cho toàn bộ app
# =============================================================

from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Khởi tạo Limiter ──────────────────────────────────────────
# get_remote_address: lấy IP của client làm key để đếm request
# Mỗi IP được tính riêng biệt — IP A không ảnh hưởng IP B
limiter = Limiter(key_func=get_remote_address)