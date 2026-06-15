# =============================================================
# app/database.py
# Mục đích: Kết nối SQLite, định nghĩa bảng, cung cấp session
# =============================================================

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── 1. Cấu hình đường dẫn database ───────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/calling_agent.db")

# Tạo thư mục data/ nếu chưa có
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── 2. Tạo engine kết nối SQLite ─────────────────────────────
# connect_args: cho phép dùng SQLite với nhiều thread cùng lúc
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False}
)

# ── 3. Tạo Base class — tất cả model kế thừa từ đây ──────────
Base = declarative_base()

# ── 4. Định nghĩa bảng call_logs ─────────────────────────────
# Mỗi dòng = 1 lần khách gọi API /predict
class CallLog(Base):
    __tablename__ = "call_logs"

    # Khoá chính — tự động tăng
    id          = Column(Integer, primary_key=True, index=True)

    # Câu nói gốc của khách
    text        = Column(Text, nullable=False)

    # Kết quả phân loại
    intent      = Column(String(50), nullable=False)
    confidence  = Column(Float, nullable=False)

    # Entity được trích xuất
    time        = Column(String(50), nullable=True)
    food_item   = Column(String(100), nullable=True)
    location    = Column(String(100), nullable=True)

    # Metadata
    created_at  = Column(DateTime, default=datetime.utcnow)

# ── 4b. Định nghĩa bảng feedbacks ────────────────────────────
# Mỗi dòng = 1 lần nhà hàng báo model đoán sai
class Feedback(Base):
    __tablename__ = "feedbacks"

    id                = Column(Integer, primary_key=True, index=True)

    # Liên kết đến bản ghi call_logs bị đoán sai
    call_log_id       = Column(Integer, nullable=False)

    # Intent model đã đoán (sai)
    predicted_intent  = Column(String(50), nullable=False)

    # Intent đúng do người dùng cung cấp
    correct_intent    = Column(String(50), nullable=False)

    # Ghi chú thêm nếu muốn
    note              = Column(Text, nullable=True)

    created_at        = Column(DateTime, default=datetime.utcnow)

# ── 5. Tạo bảng trong database (nếu chưa có) ─────────────────
Base.metadata.create_all(bind=engine)

# ── 6. Session factory — dùng để thao tác với database ───────
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ── 7. Dependency — FastAPI gọi hàm này để lấy session ───────
def get_db():
    """
    Tạo database session cho mỗi request.
    Đảm bảo session luôn được đóng sau khi request xong
    dù có lỗi hay không — nhờ khối finally.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()