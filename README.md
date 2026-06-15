# 🍽️ Restaurant AI Calling Agent — NLP API

## 📖 Giới thiệu

Hệ thống phân loại ý định (Intent Classification) từ câu nói của khách hàng,
được xây dựng bằng Machine Learning và đóng gói thành RESTful API bằng FastAPI.
Dự án mô phỏng bộ não xử lý ngôn ngữ cho AI Calling Agent của nhà hàng.

**6 Intent được hỗ trợ:**
| Intent | Ý nghĩa |
|---|---|
| `order` | Đặt món ăn |
| `greeting` | Chào hỏi |
| `reservation` | Đặt bàn |
| `query` | Hỏi thông tin |
| `complaint` | Phàn nàn |
| `cancel` | Huỷ đơn / huỷ đặt bàn |

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| NLP / ML | scikit-learn (TF-IDF + Logistic Regression) |
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Serialization | Joblib |
| Cấu hình | python-dotenv |

---

## 📂 Cấu trúc dự án
restaurant-calling-agent/

├── app/

│   ├── init.py

│   └── main.py           # FastAPI server & endpoint /predict

├── scripts/

│   └── train_model.py    # Script huấn luyện và lưu model

├── models/               # Chứa file .pkl

├── data/                 # Chứa file CSV

├── .env                  # Biến môi trường

├── .gitignore

├── requirements.txt

└── README.md
---

## ⚙️ Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/restaurant-calling-agent.git
cd restaurant-calling-agent
```

### 2. Tạo và kích hoạt môi trường ảo
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Tạo file `.env`
```bash
# Tạo file .env và thêm nội dung sau:
MODEL_PATH=models/xxx.pkl
DATA_PATH=data/xxx.csv
```

### 5. Đặt file dữ liệu
Đặt file `xxx.csv` vào thư mục `data/`.

### 6. Huấn luyện model
```bash
python scripts/train_model.py
```

### 7. Khởi động API server
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 🌐 Live Demo

| | URL |
|---|---|
| **Base URL** | https://restaurant-calling-agent.onrender.com |
| **Swagger UI** | https://restaurant-calling-agent.onrender.com/docs |
| **API Status** | https://stats.uptimerobot.com/Hc3JvxepFv |

> ⚠️ Free tier: request đầu tiên có thể mất 30-60 giây nếu server đang ngủ.