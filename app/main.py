# =============================================================
# app/main.py
# Phiên bản 2.0 — Thêm: Logging, GET /intents
# =============================================================

import os
import logging
import joblib
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.entity_extractor import extract_all_entities
from app.auth import verify_api_key

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
    version="2.0.0",
)

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
    text: str
    intent: str
    confidence: float
    time: str | None
    food_item: str | None
    location: str | None

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

# ── 9. POST /predict — Phân loại Intent chính ────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phân loại Intent từ câu nói của khách hàng",
    tags=["NLP"],
    dependencies=[Depends(verify_api_key)]
)
def predict_intent(request: PredictRequest):
    """
    Nhận câu nói → trả về intent, confidence và các entity
    (time, food_item, location) được trích xuất từ câu nói.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    user_text    = request.text.strip()
    probabilities     = model.predict_proba([user_text])[0]
    predicted_idx     = probabilities.argmax()
    predicted_intent  = model.classes_[predicted_idx]
    confidence_score  = round(float(probabilities[predicted_idx]), 4)
    entities          = extract_all_entities(user_text)

    # Ghi log mỗi request: input và output
    logger.info(
        f"PREDICT | text='{user_text}' | "
        f"intent={predicted_intent} | "
        f"confidence={confidence_score} | "
        f"entities={entities}"
    )

    return PredictResponse(
        text=user_text,
        intent=predicted_intent,
        confidence=confidence_score,
        time=entities["time"],
        food_item=entities["food_item"],
        location=entities["location"],
    )

# ── 10. Chạy trực tiếp ───────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)