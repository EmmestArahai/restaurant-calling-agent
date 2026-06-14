# =============================================================
# app/main.py
# Mục đích: Khởi tạo FastAPI server, load model, tạo endpoint /predict
# =============================================================

import os
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── 1. Load biến môi trường từ file .env ──────────────────────
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/intent_model.pkl")

# ── 2. Khởi tạo ứng dụng FastAPI ─────────────────────────────
# title, description, version sẽ hiển thị đẹp trên Swagger UI
app = FastAPI(
    title="🍽️ Restaurant AI Calling Agent API",
    description=(
        "API phân loại ý định (Intent Classification) từ câu nói của khách hàng. "
        "Hỗ trợ 6 intent: order, greeting, reservation, query, complaint, cancel."
    ),
    version="1.0.0",
)

# ── 3. Load model khi server khởi động ────────────────────────
# Biến toàn cục để tái sử dụng model ở mọi request
# Chỉ load 1 lần duy nhất → tiết kiệm tài nguyên
model = None

@app.on_event("startup")
def load_model():
    """Hàm này tự động chạy khi FastAPI server khởi động."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"❌ Không tìm thấy file model tại '{MODEL_PATH}'. "
            "Hãy chạy scripts/train_model.py trước!"
        )
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model đã được load thành công từ: {MODEL_PATH}")

# ── 4. Định nghĩa schema cho Request (đầu vào) ────────────────
# Pydantic tự động validate kiểu dữ liệu và sinh docs cho Swagger
class PredictRequest(BaseModel):
    text: str = Field(
        ...,                          # ... nghĩa là bắt buộc phải có
        min_length=1,
        max_length=500,
        example="I want to order a pizza please"
    )

# ── 5. Định nghĩa schema cho Response (đầu ra) ────────────────
class PredictResponse(BaseModel):
    text: str           # Câu nói gốc của khách
    intent: str         # Intent được dự đoán
    confidence: float   # Độ tin cậy (0.0 → 1.0)

# ── 6. Endpoint GET / — Kiểm tra server còn sống không ────────
@app.get("/", summary="Health Check")
def root():
    """Endpoint kiểm tra server đang hoạt động."""
    return {
        "status": "online",
        "message": "🍽️ Restaurant AI Calling Agent API đang hoạt động!",
        "docs": "/docs"
    }

# ── 7. Endpoint POST /predict — TRUNG TÂM CỦA TOÀN BỘ API ────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phân loại Intent từ câu nói của khách hàng",
    tags=["NLP"]
)
def predict_intent(request: PredictRequest):
    """
    Nhận vào câu nói của khách hàng và trả về:
    - **intent**: Ý định được phân loại (order/greeting/reservation/query/complaint/cancel)
    - **confidence**: Độ tin cậy của dự đoán (càng gần 1.0 càng chắc chắn)
    """
    # Kiểm tra model đã được load chưa
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    # Lấy text từ request và dự đoán
    user_text = request.text.strip()

    # predict_proba trả về xác suất cho từng intent
    # Lấy index của xác suất cao nhất → đó là intent được chọn
    probabilities  = model.predict_proba([user_text])[0]
    predicted_idx  = probabilities.argmax()
    predicted_intent    = model.classes_[predicted_idx]
    confidence_score    = round(float(probabilities[predicted_idx]), 4)

    return PredictResponse(
        text=user_text,
        intent=predicted_intent,
        confidence=confidence_score
    )

# ── 8. Chạy trực tiếp bằng lệnh `python app/main.py` ─────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)