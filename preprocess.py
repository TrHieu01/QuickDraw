import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from cnn_model import QuickDrawV2 # Import trực tiếp QuickDrawV2
import os

# --- Load model ---
# LƯU Ý: Đặt file best_model.pth trong thư mục con 'model'
model_path = r"model/best_model.pth" 
num_classes = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QuickDrawV2(num_classes)
model.is_loaded = False # Thêm flag để kiểm tra trạng thái load

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File mô hình không tồn tại tại: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Xử lý trường hợp mô hình được lưu với DataParallel (thêm tiền tố 'module.')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    model.is_loaded = True
    print(f"Mô hình QuickDrawV2 đã được tải thành công trên {DEVICE}.")
except FileNotFoundError as e:
    print(f"LỖI QUAN TRỌNG: {e}. Ứng dụng sẽ chạy, nhưng dự đoán sẽ không hoạt động.")
except Exception as e:
    print(f"LỖI khi tải mô hình: {e}. Ứng dụng sẽ chạy, nhưng dự đoán sẽ không hoạt động.")


preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_for_model(canvas_bgr):
    """
    Tiền xử lý ảnh vẽ từ canvas (nền trắng, nét đen):
    - Chuyển xám
    - Đảo ngược màu (nền đen, nét trắng)
    - Cắt vùng chứa nét vẽ
    - Chuẩn hóa tỉ lệ & pad để vuông
    - Resize xuống 28x28
    - Chuẩn hóa tensor [-1,1]
    """
    if not model.is_loaded: 
        return None
    
    try:
        # Chuyển sang grayscale
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        # Đảo màu: nét trắng trên nền đen
        gray = cv2.bitwise_not(gray)
        # Ngưỡng hóa nhẹ để loại nhiễu
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        # Tìm toạ độ vùng chứa nét vẽ
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y:y+h, x:x+w]
        
        # Làm cho ảnh vuông bằng cách pad thêm viền đen
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        
        # Thêm viền (margin)
        margin_ratio = 0.40 
        margin = int(size * margin_ratio) 
        
        square = cv2.copyMakeBorder(
            square,
            margin, margin, margin, margin,
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        # Resize xuống 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Chuẩn hóa tensor cho model
        tensor = preprocess_transform(resized)
        tensor = tensor.unsqueeze(0).to(DEVICE)  # [1,1,28,28]
        return tensor
        
    except Exception as e:
        # print(f"Lỗi trong preprocess_for_model: {e}")
        return None