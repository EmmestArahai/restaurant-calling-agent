# =============================================================
# app/main.py
# Phiên bản 2.0 — Thêm: Logging, GET /intents
# =============================================================

import os
import logging
import joblib
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.entity_extractor import extract_all_entities
from app.auth import verify_api_key
from app.limiter import limiter
from app.database import get_db, CallLog, Feedback
from sqlalchemy.orm import Session

# ── 1. Load biến môi trường ───────────────────────────────────
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/intent_model.pkl")
LOG_PATH   = os.getenv("LOG_PATH", "logs/api.log")

# ── 2. Cấu hình Logging ───────────────────────────────────────
# Tạo thư mục logs/ nếu chưa có
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Tạo logger riêng cho ứng dụng
logger = logging.getLogger("restaurant_api")
logger.setLevel(logging.INFO)

# Handler 1: ghi ra FILE (lưu vĩnh viễn)
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Handler 2: in ra TERMINAL (tiện theo dõi khi dev)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format log: [2024-01-15 20:30:00] INFO — nội dung
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ── 3. Khởi tạo FastAPI ───────────────────────────────────────
app = FastAPI(
    title="🍽️ Restaurant AI Calling Agent API",
    description=(
        "API phân loại ý định (Intent Classification) từ câu nói của khách hàng. "
        "Hỗ trợ 6 intent: order, greeting, reservation, query, complaint, cancel."
    ),
    version="3.0.0",
)

# ── Gắn Rate Limiter vào app ───────────────────────────────────
# state.limiter: FastAPI cần biết limiter nào đang dùng
# add_exception_handler: tự động trả về lỗi 429 khi bị chặn
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── 4. Danh sách intent hợp lệ ───────────────────────────────
# Định nghĩa tập trung ở đây, dùng lại ở nhiều chỗ
SUPPORTED_INTENTS = [
    {
        "intent": "order",
        "description": "Customer wants to order food",
        "example": "I want to order a pizza"
    },
    {
        "intent": "greeting",
        "description": "Customer is greeting",
        "example": "Hello, good evening"
    },
    {
        "intent": "reservation",
        "description": "Customer wants to book a table",
        "example": "Book a table for 2 at 7 PM"
    },
    {
        "intent": "query",
        "description": "Customer is asking for information",
        "example": "What are your opening hours?"
    },
    {
        "intent": "complaint",
        "description": "Customer is complaining about service",
        "example": "My order is late and food is cold"
    },
    {
        "intent": "cancel",
        "description": "Customer wants to cancel order or reservation",
        "example": "Cancel my reservation please"
    },
]

# ── 5. Load model khi server khởi động ───────────────────────
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"❌ Không tìm thấy model tại '{MODEL_PATH}'. "
            "Hãy chạy scripts/train_model.py trước!"
        )
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded thành công từ: {MODEL_PATH}")
    logger.info(f"API v2.0.0 khởi động — Log file: {LOG_PATH}")

# ── 6. Schema Request / Response ─────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="I want to order a pizza at 7 PM"
    )

class PredictResponse(BaseModel):
    id: int                  # ID bản ghi trong database
    text: str
    intent: str
    confidence: float
    time: str | None
    food_item: str | None
    location: str | None
    created_at: str          # Thời điểm ghi nhận

# ── Schema cho POST /feedback ─────────────────────────────────
class FeedbackRequest(BaseModel):
    call_log_id: int = Field(
        ...,
        description="ID của bản ghi cần phản hồi (lấy từ response của /predict)",
        example=1
    )
    correct_intent: str = Field(
        ...,
        description="Intent đúng thực sự",
        example="cancel"
    )
    note: str | None = Field(
        default=None,
        description="Ghi chú thêm (không bắt buộc)",
        example="Customer said cancel but model predicted order"
    )

class FeedbackResponse(BaseModel):
    id: int
    call_log_id: int
    predicted_intent: str
    correct_intent: str
    note: str | None
    created_at: str
    message: str

# ── 7. GET / — Health Check ───────────────────────────────────
@app.get("/", summary="Health Check")
def root():
    logger.info("Health check endpoint được gọi")
    return {
        "status": "online",
        "version": "2.0.0",
        "message": "🍽️ Restaurant AI Calling Agent API đang hoạt động!",
        "docs": "/docs"
    }

# ── 8. GET /intents — Danh sách intent được hỗ trợ ───────────
@app.get(
    "/intents",
    summary="Lấy danh sách tất cả Intent được hỗ trợ",
    tags=["Info"]
)
def get_intents():
    """
    Trả về danh sách 6 intent mà model có thể phân loại,
    kèm mô tả và câu ví dụ cho từng intent.
    Hữu ích cho frontend hoặc developer tích hợp API.
    """
    logger.info("GET /intents được gọi")
    return {
        "total": len(SUPPORTED_INTENTS),
        "intents": SUPPORTED_INTENTS
    }

# ── POST /feedback — Nhận phản hồi khi model đoán sai ────────
@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Báo cáo khi model dự đoán sai intent",
    tags=["Feedback"],
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("30/minute")
def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Nhận phản hồi từ nhà hàng khi model đoán sai.
    - Tìm bản ghi call_log theo call_log_id
    - Lưu feedback kèm intent đúng vào database
    - Dữ liệu này dùng để train lại model sau
    """
    # ── Kiểm tra call_log_id có tồn tại không ────────────────
    call_log = db.query(CallLog).filter(
        CallLog.id == body.call_log_id
    ).first()

    if not call_log:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy bản ghi với id={body.call_log_id}"
        )

    # ── Kiểm tra correct_intent có hợp lệ không ──────────────
    valid_intents = [i["intent"] for i in SUPPORTED_INTENTS]
    if body.correct_intent not in valid_intents:
        raise HTTPException(
            status_code=400,
            detail=f"Intent không hợp lệ. Các intent hợp lệ: {valid_intents}"
        )

    # ── Lưu feedback vào database ─────────────────────────────
    feedback = Feedback(
        call_log_id      = body.call_log_id,
        predicted_intent = call_log.intent,
        correct_intent   = body.correct_intent,
        note             = body.note,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    logger.info(
        f"FEEDBACK | call_log_id={body.call_log_id} | "
        f"predicted={call_log.intent} | "
        f"correct={body.correct_intent}"
    )

    return FeedbackResponse(
        id               = feedback.id,
        call_log_id      = feedback.call_log_id,
        predicted_intent = feedback.predicted_intent,
        correct_intent   = feedback.correct_intent,
        note             = feedback.note,
        created_at       = feedback.created_at.isoformat(),
        message          = f"Cam on! Da ghi nhan: '{feedback.predicted_intent}' -> '{feedback.correct_intent}'"
    )

# ── 9. POST /predict — Phân loại Intent chính ────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phân loại Intent từ câu nói của khách hàng",
    tags=["NLP"],
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("60/minute")
def predict_intent(
    request: Request,
    body: PredictRequest,
    db: Session = Depends(get_db)     # Inject database session
):
    """
    Nhận câu nói → trả về intent, confidence và các entity.
    Mỗi request được lưu vào database để phân tích sau.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    # ── Dự đoán ───────────────────────────────────────────────
    user_text         = body.text.strip()
    probabilities     = model.predict_proba([user_text])[0]
    predicted_idx     = probabilities.argmax()
    predicted_intent  = model.classes_[predicted_idx]
    confidence_score  = round(float(probabilities[predicted_idx]), 4)
    entities          = extract_all_entities(user_text)

    # ── Lưu vào database ──────────────────────────────────────
    log_entry = CallLog(
        text       = user_text,
        intent     = predicted_intent,
        confidence = confidence_score,
        time       = entities["time"],
        food_item  = entities["food_item"],
        location   = entities["location"],
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)   # Lấy lại id và created_at vừa được sinh ra

    # ── Ghi log file ──────────────────────────────────────────
    logger.info(
        f"PREDICT | id={log_entry.id} | text='{user_text}' | "
        f"intent={predicted_intent} | confidence={confidence_score} | "
        f"entities={entities}"
    )

    return PredictResponse(
        id         = log_entry.id,
        text       = user_text,
        intent     = predicted_intent,
        confidence = confidence_score,
        time       = entities["time"],
        food_item  = entities["food_item"],
        location   = entities["location"],
        created_at = log_entry.created_at.isoformat(),
    )

# ── 10. Chạy trực tiếp ───────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)