import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import numpy as np
import cv2
import torch
import torch.nn.functional as F # Cần cho Softmax
import os
import mediapipe as mp
from threading import Lock
import time

# --- Import các module của bạn ---
# Đảm bảo các file này đã được tách logic load model ra khỏi global scope
from cnn_model import QuickDrawV2
from classes import QUICKDRAW_CLASSES
from preprocess import preprocess_for_model 

# --- Cấu hình Streamlit ---
st.set_page_config(
    page_title="QuickDraw Classifier (WebRTC)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load Model (Sử dụng Caching) ---
@st.cache_resource
def load_pytorch_model():
    """Tải model chỉ một lần và trả về model đã load."""
    model_path = "model/best_model.pth" 
    num_classes = len(QUICKDRAW_CLASSES)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QuickDrawV2(num_classes)
    is_loaded = False
    
    try:
        if not os.path.exists(model_path):
            st.error(f"LỖI: Không tìm thấy file mô hình tại {model_path}. Dự đoán sẽ không hoạt động.")
            return model, False, DEVICE

        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        is_loaded = True
        st.sidebar.success(f"Mô hình đã sẵn sàng trên {DEVICE}.")
        return model, is_loaded, DEVICE
        
    except Exception as e:
        st.error(f"LỖI khi tải mô hình: {e}. Vui lòng kiểm tra file .pth và kiến trúc model.")
        return model, False, DEVICE

MODEL, IS_MODEL_LOADED, DEVICE = load_pytorch_model()

# --- 2. Hàm Dự đoán Chính (Tích hợp logic từ predict.py và thêm Icon Path) ---
def _predict_drawing(canvas_bgr, model, device, classes):
    """
    Thực hiện tiền xử lý và dự đoán lớp từ ảnh vẽ (canvas_bgr), 
    trả về kết quả kèm theo đường dẫn icon.
    """
    if not IS_MODEL_LOADED:
        return {'prediction': 'Model Not Ready', 'probability': '0.00%', 'top_k': [], 'icon_path': None}

    # 1. Tiền xử lý ảnh (sử dụng hàm từ preprocess.py)
    # Gán model và device vào global scope của preprocess để hàm preprocess_for_model có thể dùng
    # NOTE: Đây là một hack để tránh việc truyền model và device vào hàm preprocess_for_model, 
    # nhưng trong môi trường Streamlit/multithread, cách tốt nhất là pass trực tiếp.
    # Tuy nhiên, do cấu trúc code hiện tại, chúng ta tạm thời dùng global model/device
    # cho việc gọi preprocess_for_model.

    # 1. Tiền xử lý ảnh (sử dụng hàm từ preprocess.py)
    input_tensor = preprocess_for_model(canvas_bgr)
    
    if input_tensor is None:
        return {
            'prediction': 'No drawing found',
            'probability': '0.00%',
            'top_k': [],
            'icon_path': None
        }

    # Đưa tensor lên thiết bị (CPU/GPU)
    # input_tensor đã được to(DEVICE) trong preprocess_for_model, nhưng thêm check để đảm bảo an toàn
    if input_tensor.device != device:
        input_tensor = input_tensor.to(device) 

    # 2. Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        
    # 3. Tính toán xác suất (Softmax)
    probabilities = F.softmax(output, dim=1)
    
    # 4. Lấy dự đoán cao nhất
    max_prob, predicted_index = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted_index.item()]
    confidence = max_prob.item() * 100.0
    
    # 5. Lấy Top-K
    top_k_probs, top_k_indices = torch.topk(probabilities, 5)
    
    top_k_results = []
    for i in range(5):
        idx = top_k_indices[0][i].item()
        prob = top_k_probs[0][i].item() * 100.0
        top_k_results.append({
            'class': classes[idx],
            'probability': f"{prob:.2f}%"
        })
        
    # 6. Thêm Icon Path
    # Chuyển tên lớp thành tên file (chữ thường, thay thế khoảng trắng bằng gạch dưới)
    icon_filename = f"images/{predicted_class.lower().replace(' ', '_').replace('-', '_')}.png"
    # Kiểm tra xem file có tồn tại không (tùy chọn)
    if not os.path.exists(icon_filename):
        icon_filename = None # Nếu không tìm thấy, đặt là None
        
    return {
        'prediction': predicted_class,
        'probability': f"{confidence:.2f}%",
        'top_k': top_k_results,
        'icon_path': icon_filename
    }


# --- 3. Định nghĩa Bộ Xử lý Video (MediaPipe & Drawing Logic) ---

class AirDrawingProcessor(VideoProcessorBase):
    """
    Xử lý từng khung hình video từ camera để nhận diện tay và vẽ.
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.CANVAS_W, self.CANVAS_H = 400, 400
        # Canvas là nơi nét vẽ được ghi lại
        self.canvas = np.full((self.CANVAS_H, self.CANVAS_H, 3), 255, dtype=np.uint8) # Trắng
        self.last_point = None
        self.is_drawing = False
        self.has_drawn_since_clear = False
        self.lock = Lock()
        self.prediction_result = None

    # Logic kiểm tra cử chỉ (giữ nguyên)
    def _is_index_finger_extended(self, hand_landmarks):
        index_extended = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_closed = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_closed = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_closed = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        return index_extended and middle_closed and ring_closed and pinky_closed

    def _is_open_hand(self, hand_landmarks):
        fingers_and_pips = [
            (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP), 
            (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP), 
            (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
        ]
        all_extended = True
        for tip, pip in fingers_and_pips:
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                all_extended = False
                break
        thumb_extended = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x
        return all_extended and thumb_extended

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        # BỎ BƯỚC cv2.flip(img, 1) - WebRTC đã lật ảnh, lật lại sẽ gây ngược
        
        AIR_AREA_SIZE = 300
        x_start = (w - AIR_AREA_SIZE) // 2
        y_start = (h - AIR_AREA_SIZE) // 2
        x_end = x_start + AIR_AREA_SIZE
        y_end = y_start + AIR_AREA_SIZE
        
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.putText(img, "AIR DRAWING AREA", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x_cam = int(index_finger_tip.x * w)
                y_cam = int(index_finger_tip.y * h)

                pointing = self._is_index_finger_extended(hand_landmarks)
                open_hand = self._is_open_hand(hand_landmarks)
                
                in_draw_area = x_start <= x_cam <= x_end and y_start <= y_cam <= y_end
                                
                if in_draw_area:
                    # Ánh xạ tọa độ từ khung camera sang canvas 400x400
                    x_norm = (x_cam - x_start) / AIR_AREA_SIZE
                    y_norm = (y_cam - y_start) / AIR_AREA_SIZE
                    x_canvas = int(x_norm * self.CANVAS_W)
                    y_canvas = int(y_norm * self.CANVAS_H)
                    
                    if pointing:
                        cv2.circle(img, (x_cam, y_cam), 10, (0, 255, 0), -1) 
                        with self.lock:
                            # VẼ NÉT MỚI: Chỉ vẽ khi đang ở trạng thái 'is_drawing' (từ lần trước)
                            if self.is_drawing and self.last_point:
                                cv2.line(self.canvas, self.last_point, (x_canvas, y_canvas), (0, 0, 0), 15)
                            
                            self.last_point = (x_canvas, y_canvas)
                            self.is_drawing = True
                            self.has_drawn_since_clear = True
                    else:
                        self.last_point = None
                        self.is_drawing = False
                        
                    # Dự đoán Tự động (Open Hand)
                    if open_hand and self.has_drawn_since_clear and not self.is_drawing:
                        with self.lock:
                            self.prediction_result = self._predict_and_clear()
                        
                else:
                    self.last_point = None
                    self.is_drawing = False
            else:
                self.last_point = None
                self.is_drawing = False

        # WebRTC Streamer cần trả về đối tượng MediaPipe Frame
        return frame

    def _predict_and_clear(self):
        """Thực hiện dự đoán và xóa canvas."""
        
        # GỌI HÀM DỰ ĐOÁN MỚI
        result = _predict_drawing(self.canvas, MODEL, DEVICE, QUICKDRAW_CLASSES)
        
        # Xóa canvas sau khi dự đoán
        self.canvas = np.full((self.CANVAS_H, self.CANVAS_H, 3), 255, dtype=np.uint8) # Trắng
        self.has_drawn_since_clear = False
        return result

# --- 4. Giao diện Streamlit ---

st.title("QuickDraw Classifier Web App")
st.markdown("Chọn chế độ **'Vẽ Chuột'** (Mouse) hoặc **'Vẽ Không khí'** (Air Drawing) bằng camera.")

# Tabs
tab1, tab2 = st.tabs(["Vẽ Chuột (Mouse)", "Vẽ Không khí (Air Drawing)"])

# --- Tab 1: Vẽ Chuột (Mouse) ---
with tab1:
    col1_mouse, col2_mouse = st.columns([1, 1])

    with col1_mouse:
        st.header("Canvas Vẽ (Chuột)")
        
        CANVAS_SIZE = 400
        
        with st.sidebar:
            st.subheader("Công cụ Vẽ Chuột")
            stroke_width = st.slider("Độ dày nét vẽ", 10, 40, 20, key="mouse_stroke_width")
        
        # Cập nhật fill_color và background_color để đảm bảo nền trắng tuyệt đối
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)", 
            stroke_width=stroke_width,
            stroke_color="#000000",              
            background_color="#FFFFFF",
            update_streamlit=True,
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="mouse_canvas",
        )

        trigger_prediction_mouse = st.button("Dự đoán Nét vẽ (Mouse)", key="btn_mouse_predict")

        if canvas_result.image_data is not None and trigger_prediction_mouse:
            canvas_data_np = canvas_result.image_data.astype(np.uint8)
            rgb_image = canvas_data_np[:, :, :3]
            canvas_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            with st.spinner("Đang xử lý dự đoán..."):
                # GỌI HÀM DỰ ĐOÁN MỚI
                result = _predict_drawing(canvas_bgr, MODEL, DEVICE, QUICKDRAW_CLASSES)
            
            st.session_state['mouse_prediction'] = result
            st.session_state['show_mouse_result'] = True
        
    with col2_mouse:
        st.header("Kết quả Mouse Drawing")
        
        if 'show_mouse_result' in st.session_state and st.session_state['show_mouse_result']:
            result = st.session_state['mouse_prediction']
            
            if result['prediction'] == 'Model Not Ready':
                st.error("LỖI: Không thể dự đoán vì mô hình chưa được tải.")
            elif result['prediction'] == 'No drawing found':
                st.warning("Không tìm thấy nét vẽ hợp lệ trên canvas. Vui lòng vẽ rõ hơn.")
            else:
                # HIỂN THỊ ICON
                if result['icon_path']:
                    st.image(result['icon_path'], caption=f"Icon của lớp dự đoán: {result['prediction']}", width=100)
                else:
                    st.warning(f"Không tìm thấy icon cho lớp '{result['prediction']}' tại đường dẫn dự kiến.")
                
                st.subheader("🏆 Kết quả Phân loại:")
                st.metric(
                    label=f"Dự đoán Chính xác nhất", 
                    value=f"{result['prediction'].upper()}", 
                    delta=f"{result['probability']} Confidence"
                )
                st.markdown("**Top 5 Dự đoán:**")
                top_k_data = result['top_k']
                for item in top_k_data:
                    st.write(f"- **{item['class']}**: {item['probability']}")

# --- Tab 2: Vẽ Không khí (Air Drawing) ---
with tab2:
    st.header("Vẽ Không khí (Air Drawing) - Camera")
    
    col1_air, col2_air = st.columns([1, 1])

    with col1_air:
        st.info("Khu vực hiển thị camera (WebRTC Streamer)")
        ctx = webrtc_streamer(
            key="air_drawing_stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=AirDrawingProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
    with col2_air:
        st.header("Canvas và Kết quả Air Drawing")
        
        # Cập nhật trạng thái Canvas hiển thị trong session_state
        if ctx.video_processor:
            # Dùng st.session_state để trigger redraw nếu Canvas thay đổi
            if 'air_canvas_version' not in st.session_state:
                st.session_state['air_canvas_version'] = 0

            # Cố gắng lấy canvas và kết quả trong khi khóa mutex
            with ctx.video_processor.lock:
                canvas_display = ctx.video_processor.canvas.copy()
                result = ctx.video_processor.prediction_result.copy() if ctx.video_processor.prediction_result else None
            
            st.image(canvas_display, caption="Canvas vẽ bằng tay", width=400)
            
            if result:
                if result['prediction'] == 'Model Not Ready':
                    st.error("LỖI: Không thể dự đoán vì mô hình chưa được tải.")
                elif result['prediction'] == 'No drawing found':
                    st.warning("Vừa rồi không tìm thấy nét vẽ nào. Vui lòng thử lại.")
                else:
                    st.subheader("🏆 Kết quả Tự động:")
                    
                    # HIỂN THỊ ICON
                    if result['icon_path']:
                        st.image(result['icon_path'], caption=f"Icon của lớp dự đoán: {result['prediction']}", width=100)
                    else:
                        st.warning(f"Không tìm thấy icon cho lớp '{result['prediction']}' tại đường dẫn dự kiến.")

                    st.metric(
                        label=f"Dự đoán Chính xác nhất", 
                        value=f"{result['prediction'].upper()}", 
                        delta=f"{result['probability']} Confidence"
                    )
                    st.markdown("**Top 5 Dự đoán:**")
                    for item in result['top_k']:
                        st.write(f"- **{item['class']}**: {item['probability']}")

        else:
            st.info("Đang chờ kết nối camera...")