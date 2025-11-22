# classes.py
# Danh sách 25 lớp QuickDraw theo thứ tự ABC (hoặc thứ tự tương ứng với đầu ra của mô hình)

QUICKDRAW_CLASSES = [
     "apple", "bowtie", "circle", "cloud", "cup",
    "diamond", "fish", "guitar", "hat", "headphones",
    "ladder", "laptop", "leaf", "moon", "paints",
    "pencil", "smiley_face", "soccer_ball", "sock", "star",
    "sun", "t-shirt", "triangle", "watermelon", "wine_class"
    # LƯU Ý: Đảm bảo danh sách này khớp chính xác với chỉ mục đầu ra (0-24) của mô hình.
    # Nếu mô hình được train với thứ tự khác, hãy điều chỉnh ở đây.
]

def get_class_name(index):
    """
    Trả về tên lớp từ chỉ mục (index) đầu ra của mô hình.
    """
    if 0 <= index < len(QUICKDRAW_CLASSES):
        return QUICKDRAW_CLASSES[index]
    return "Unknown Class"