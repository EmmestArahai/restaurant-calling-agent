# =============================================================
# app/entity_extractor.py
# Mục đích: Trích xuất các entity (time, food_item, location)
# từ câu nói của khách hàng bằng Regex + Keyword Matching
# =============================================================

import re
from typing import Optional

# ── 1. Danh sách keyword cho từng entity ──────────────────────
# Lấy trực tiếp từ dữ liệu thực tế trong dataset

FOOD_KEYWORDS = [
    "pizza", "burger", "pasta", "biryani", "sandwich",
    "chicken biryani", "chicken", "rice", "noodle", "steak"
]

LOCATION_KEYWORDS = [
    "city center", "main road", "near the mall", "near mall",
    "downtown", "city centre", "mall", "center", "centre"
]

# ── 2. Hàm trích xuất TIME bằng Regex ─────────────────────────
def extract_time(text: str) -> Optional[str]:
    """
    Tìm kiếm pattern giờ trong câu nói.
    Ví dụ: "at 7 PM", "8pm", "9 AM", "tonight at 6"
    """
    text_lower = text.lower()

    # Pattern 1: "7 PM", "10 AM", "6pm", "8 am"
    match = re.search(r'\b(\d{1,2})\s*(am|pm)\b', text_lower)
    if match:
        hour = match.group(1)
        period = match.group(2).upper()
        return f"{hour} {period}"

    # Pattern 2: "7:30 PM", "19:00"
    match = re.search(r'\b(\d{1,2}:\d{2})\s*(am|pm)?\b', text_lower)
    if match:
        return match.group(0).strip().upper()

    # Pattern 3: từ khoá thời gian tự nhiên
    if "tonight" in text_lower:
        return "tonight"
    if "tomorrow" in text_lower:
        return "tomorrow"
    if "now" in text_lower or "right now" in text_lower:
        return "now"

    return None  # Không tìm thấy

# ── 3. Hàm trích xuất FOOD ITEM bằng Keyword Matching ─────────
def extract_food_item(text: str) -> Optional[str]:
    """
    Tìm kiếm tên món ăn trong câu nói.
    Ưu tiên cụm từ dài hơn (chicken biryani) trước từ ngắn (chicken).
    """
    text_lower = text.lower()

    # Sắp xếp theo độ dài giảm dần để match cụm dài trước
    sorted_keywords = sorted(FOOD_KEYWORDS, key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in text_lower:
            return keyword

    return None  # Không tìm thấy

# ── 4. Hàm trích xuất LOCATION bằng Keyword Matching ──────────
def extract_location(text: str) -> Optional[str]:
    """
    Tìm kiếm địa điểm trong câu nói.
    Ưu tiên cụm từ dài hơn (city center) trước từ ngắn (city).
    """
    text_lower = text.lower()

    # Sắp xếp theo độ dài giảm dần
    sorted_keywords = sorted(LOCATION_KEYWORDS, key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in text_lower:
            return keyword

    return None  # Không tìm thấy

# ── 5. Hàm tổng hợp — gọi cả 3 hàm trên cùng lúc ─────────────
def extract_all_entities(text: str) -> dict:
    """
    Trích xuất toàn bộ entity từ một câu nói.
    Trả về dict với 3 key: time, food_item, location.
    Giá trị là None nếu không tìm thấy entity đó trong câu.
    """
    return {
        "time":      extract_time(text),
        "food_item": extract_food_item(text),
        "location":  extract_location(text),
    }