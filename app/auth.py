# =============================================================
# app/auth.py
# Mục đích: Xác thực API Key trên mỗi request
# =============================================================

import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# ── 1. Load biến môi trường ───────────────────────────────────
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ── 2. Khai báo nơi FastAPI tìm API Key ──────────────────────
# Client phải gửi key trong HTTP Header tên "X-API-Key"
# auto_error=False để ta tự xử lý lỗi thay vì FastAPI tự động
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── 3. Hàm dependency — FastAPI gọi hàm này trước mỗi request
def verify_api_key(key: str = Security(api_key_header)) -> str:
    """
    Kiểm tra API Key từ header X-API-Key.
    - Nếu không có key      → 401 Unauthorized
    - Nếu key sai           → 403 Forbidden
    - Nếu key đúng          → cho phép request đi tiếp
    """
    # Trường hợp 1: Client không gửi key nào
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Missing API Key",
                "message": "Vui long gui API Key trong header 'X-API-Key'"
            }
        )

    # Trường hợp 2: Client gửi key nhưng sai
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Invalid API Key",
                "message": "API Key khong hop le"
            }
        )

    # Trường hợp 3: Key đúng → trả về key để endpoint có thể dùng nếu cần
    return key