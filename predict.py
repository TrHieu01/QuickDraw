# predict.py
import torch
import torch.nn.functional as F
import numpy as np
# Import trực tiếp vì cùng thư mục
from preprocess import preprocess_for_model, model, DEVICE 
from classes import QUICKDRAW_CLASSES 

# --- Load model và classes đã được thực hiện trong preprocess.py ---

def predict_class(canvas_bgr):
    """
    Thực hiện tiền xử lý và dự đoán lớp từ ảnh vẽ (canvas_bgr).
    """
    
    # Kiểm tra trạng thái load model
    if not model.is_loaded:
         return {
            'prediction': 'Model Not Ready',
            'probability': 0.0,
            'top_k': []
        }
        
    # 1. Tiền xử lý ảnh
    input_tensor = preprocess_for_model(canvas_bgr)
    
    if input_tensor is None:
        return {
            'prediction': 'No drawing found',
            'probability': 0.0,
            'top_k': []
        }

    # 2. Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        
    # 3. Tính toán xác suất (Softmax)
    probabilities = F.softmax(output, dim=1)
    
    # 4. Lấy dự đoán cao nhất
    max_prob, predicted_index = torch.max(probabilities, 1)
    
    predicted_class = QUICKDRAW_CLASSES[predicted_index.item()]
    confidence = max_prob.item() * 100.0
    
    # 5. Lấy Top-K
    top_k_probs, top_k_indices = torch.topk(probabilities, 5)
    
    top_k_results = []
    # Chuyển tensor sang list Python để xử lý
    top_k_probs_list = top_k_probs[0].tolist()
    top_k_indices_list = top_k_indices[0].tolist()

    for idx, prob in zip(top_k_indices_list, top_k_probs_list):
        top_k_results.append({
            'class': QUICKDRAW_CLASSES[idx],
            'probability': f"{prob * 100.0:.2f}%"
        })
        
    return {
        'prediction': predicted_class,
        'probability': f"{confidence:.2f}%",
        'top_k': top_k_results
    }

# --- Ví dụ kiểm tra nhanh (chỉ chạy khi predict.py được chạy trực tiếp) ---
if __name__ == '__main__':
    print("Logic dự đoán QuickDraw sẵn sàng.")