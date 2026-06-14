# =============================================================
# scripts/train_model.py
# Mục đích: Đọc dữ liệu, huấn luyện mô hình NLP, lưu model ra file
# =============================================================

import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv

# ── 1. Load biến môi trường từ file .env ──────────────────────
load_dotenv()
DATA_PATH  = os.getenv("DATA_PATH",  "data/ai_calling_agent_dataset_1000.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/intent_model.pkl")

# ── 2. Đọc dữ liệu từ file CSV ────────────────────────────────
print("📂 Đang đọc dữ liệu từ:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Kiểm tra nhanh dữ liệu
print(f"✅ Tổng số mẫu: {len(df)}")
print(f"✅ Các intent có trong dữ liệu:\n{df['intent'].value_counts()}\n")

# ── 3. Tách dữ liệu thành X (đầu vào) và y (nhãn) ────────────
# X: cột 'text' — câu nói của khách hàng
# y: cột 'intent' — nhãn ý định cần dự đoán
X = df["text"]
y = df["intent"]

# ── 4. Chia dữ liệu thành tập Train (80%) và Test (20%) ───────
# random_state=42 để đảm bảo kết quả tái hiện được
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Tập Train: {len(X_train)} mẫu | Tập Test: {len(X_test)} mẫu")

# ── 5. Xây dựng Pipeline scikit-learn ─────────────────────────
# Pipeline gộp 2 bước thành 1 khối duy nhất:
#   Bước 1 — TfidfVectorizer: chuyển text thô → ma trận số (TF-IDF)
#             ngram_range=(1,2): học cả từ đơn ("order") lẫn cụm 2 từ ("cancel booking")
#   Bước 2 — LogisticRegression: dùng ma trận số đó để phân loại intent
#             max_iter=1000: tăng số vòng lặp để model hội tụ đủ
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=5.0
    ))
])

# ── 6. Huấn luyện model trên tập Train ────────────────────────
print("\n🚀 Bắt đầu huấn luyện model...")
pipeline.fit(X_train, y_train)
print("✅ Huấn luyện hoàn tất!")

# ── 7. Đánh giá model trên tập Test ──────────────────────────
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {accuracy * 100:.2f}%")
print("\n📋 BÁO CÁO CHI TIẾT THEO TỪNG INTENT:")
print(classification_report(y_test, y_pred))

# ── 8. Lưu model đã huấn luyện ra file .pkl ───────────────────
# Tạo thư mục models/ nếu chưa tồn tại
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"💾 Model đã được lưu tại: {MODEL_PATH}")