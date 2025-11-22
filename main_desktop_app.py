import cv2
import numpy as np
import mediapipe as mp
import time
from predict import predict_class 
from classes import QUICKDRAW_CLASSES 
import os 

# --- Cấu hình chung ---
CANVAS_W, CANVAS_H = 400, 400
CAM_W, CAM_H = 640, 480 
DRAW_COLOR = (0, 0, 0) # Nét vẽ màu đen
CANVAS_COLOR = (255, 255, 255) # Nền trắng
BRUSH_THICKNESS = 15

# Khởi tạo MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    print("LỖI: Không tìm thấy thư viện MediaPipe. Chế độ AIR sẽ không hoạt động.")
    class DummyHands:
        def process(self, image): return None
        def close(self): pass
    hands = DummyHands()
    mp_hands = None

# --- Biến trạng thái ---
canvas = np.full((CANVAS_H, CANVAS_W, 3), CANVAS_COLOR, dtype=np.uint8)
prediction_result = None
last_point = None
drawing_mode = 'MOUSE' # 'MOUSE' hoặc 'AIR'
is_drawing = False 
has_drawn_since_clear = False 

# --- Chức năng Kiểm tra Cử chỉ ---
# LandMark IDs quan trọng:
# TIP (Đầu ngón): 8 (Trỏ), 12 (Giữa), 16 (Áp út), 20 (Út)
# PIP (Đốt giữa): 6 (Trỏ), 10 (Giữa), 14 (Áp út), 18 (Út)
# DIP (Đốt cuối): 7 (Trỏ), 11 (Giữa), 15 (Áp út), 19 (Út)

def is_index_finger_extended(hand_landmarks):
    """
    Kiểm tra xem chỉ có ngón trỏ duỗi ra không (chế độ chỉ trỏ để vẽ).
    """
    if not mp_hands: return False
    
    # 1. Ngón trỏ phải duỗi (TIP (8) phải ở trên PIP (6))
    index_extended = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

    # 2. Các ngón khác phải co lại (TIP phải ở dưới PIP)
    middle_closed = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_closed = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    
    # SỬA LỖI: PINKY_FINGER_PIP -> PINKY_PIP
    pinky_closed = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    
    # 3. Ngón cái phải nằm ngoài (đảm bảo không chạm ngón trỏ)
    thumb_closed = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    
    # Chỉ trỏ (Draw) khi: Ngón trỏ duỗi VÀ các ngón khác co
    is_pointing = index_extended and middle_closed and ring_closed and pinky_closed
    return is_pointing

def is_open_hand(hand_landmarks):
    """
    Kiểm tra xem toàn bộ bàn tay có mở ra không (dùng để dự đoán/clear).
    """
    if not mp_hands: return False
    
    # SỬA LỖI: Dùng PINKY_PIP thay vì PINKY_FINGER_PIP
    fingers_and_pips = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP), 
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP), 
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP) # Đã sửa lỗi
    ]
    
    all_extended = True
    for tip, pip in fingers_and_pips:
        # Nếu ngón nào co (tip y > pip y) thì không phải open hand
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            all_extended = False
            break
            
    # Ngón cái phải nằm ngoài (duỗi)
    thumb_extended = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    
    return all_extended and thumb_extended


# --- Chức năng Mouse Drawing (Vẽ chuột) ---
def draw_mouse(event, x, y, flags, param):
    """Xử lý sự kiện vẽ bằng chuột trên Canvas."""
    global last_point, canvas
    
    # Canvas nằm ở nửa bên phải của cửa sổ, bắt đầu từ vị trí CAM_W
    x_canvas = x - CAM_W 
    # Offset để căn giữa Canvas 400x400 vào khung 480x400
    y_offset_display = (CAM_H - CANVAS_H) // 2 
    y_canvas = y - y_offset_display
    
    if drawing_mode == 'MOUSE' and x_canvas >= 0 and x_canvas < CANVAS_W and y_canvas >= 0 and y_canvas < CANVAS_H:
        
        if event == cv2.EVENT_LBUTTONDOWN:
            last_point = (x_canvas, y_canvas)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if last_point:
                cv2.line(canvas, last_point, (x_canvas, y_canvas), DRAW_COLOR, BRUSH_THICKNESS)
                last_point = (x_canvas, y_canvas)
        elif event == cv2.EVENT_LBUTTONUP:
            last_point = None
    else:
        last_point = None 

# --- Main App Loop ---
def run_app():
    global canvas, prediction_result, drawing_mode, last_point, is_drawing, has_drawn_since_clear
    
    # Khởi tạo Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("LỖI: Không thể mở camera. Vui lòng kiểm tra webcam.")
        return
        
    cv2.namedWindow('QuickDraw Classifier App')
    cv2.setMouseCallback('QuickDraw Classifier App', draw_mouse)
    
    print("\n--- QuickDraw Classifier App ---")
    print("CHẾ ĐỘ AIR:")
    print("   - Vẽ: Chỉ duỗi ngón trỏ (chỉ trỏ).")
    print("   - Dừng/Dự đoán: Mở rộng toàn bộ bàn tay (Open Hand).")
    print("CHẾ ĐỘ CHUỘT: Dùng chuột vẽ trên Canvas bên phải.")
    print("Điều khiển: 'M' (Chuyển mode), 'C' (Xóa), 'P' (Dự đoán thủ công), 'Q' (Thoát)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1) 
        
        # --- 1. Xử lý MediaPipe (Chế độ AIR) ---
        if drawing_mode == 'AIR' and mp_hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Định nghĩa khu vực vẽ Air Drawing
            AIR_AREA_X_START, AIR_AREA_Y_START = 100, 100
            AIR_AREA_X_END, AIR_AREA_Y_END = CAM_W - 100, CAM_H - 100
            
            cv2.rectangle(frame, (AIR_AREA_X_START, AIR_AREA_Y_START), 
                          (AIR_AREA_X_END, AIR_AREA_Y_END), (255, 0, 0), 2)
            cv2.putText(frame, "AIR DRAWING AREA", (AIR_AREA_X_START, AIR_AREA_Y_START - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Lấy tọa độ ngón trỏ
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_cam = int(index_finger_tip.x * CAM_W)
                    y_cam = int(index_finger_tip.y * CAM_H)

                    # --- Logic Kích hoạt/Dừng/Vẽ ---
                    pointing = is_index_finger_extended(hand_landmarks)
                    open_hand = is_open_hand(hand_landmarks)
                    
                    # Ánh xạ tọa độ Camera sang Canvas (chỉ trong khu vực vẽ)
                    in_draw_area = AIR_AREA_X_START <= x_cam <= AIR_AREA_X_END and \
                                   AIR_AREA_Y_START <= y_cam <= AIR_AREA_Y_END
                                   
                    if in_draw_area:
                        x_norm = (x_cam - AIR_AREA_X_START) / (AIR_AREA_X_END - AIR_AREA_X_START)
                        y_norm = (y_cam - AIR_AREA_Y_START) / (AIR_AREA_Y_END - AIR_AREA_Y_START)
                        x_canvas = int(x_norm * CANVAS_W)
                        y_canvas = int(y_norm * CANVAS_H)
                        
                        if pointing:
                            cv2.circle(frame, (x_cam, y_cam), 10, (0, 255, 0), -1) # Màu xanh khi đang vẽ
                            if is_drawing:
                                cv2.line(canvas, last_point, (x_canvas, y_canvas), DRAW_COLOR, BRUSH_THICKNESS)
                            last_point = (x_canvas, y_canvas)
                            is_drawing = True
                            has_drawn_since_clear = True
                        else:
                            last_point = None
                            is_drawing = False
                            
                        # Logic Tự động Dự đoán (Open Hand)
                        # Kích hoạt khi: Tay mở VÀ đã vẽ được VÀ không còn đang trong trạng thái vẽ (vừa thả tay ra)
                        if open_hand and has_drawn_since_clear and not is_drawing:
                            prediction_result = predict_class(canvas)
                            if prediction_result and prediction_result['prediction'] != 'No drawing found' and prediction_result['prediction'] != 'Model Not Ready':
                                print(f"Dự đoán tự động (Open Hand): {prediction_result['prediction']} - {prediction_result['probability']}")
                                # Sau khi dự đoán, reset trạng thái vẽ/canvas
                                # TẠM THỜI KHÔNG XÓA CANVAS TỰ ĐỘNG để người dùng xem kết quả
                                # canvas = np.full((CANVAS_H, CANVAS_W, 3), CANVAS_COLOR, dtype=np.uint8)
                                has_drawn_since_clear = False
                                # Đặt lại last_point và is_drawing để ngăn dự đoán lặp lại
                                last_point = None
                                is_drawing = False
                            else:
                                print("Dự đoán tự động: Không tìm thấy nét vẽ hoặc Model Not Ready. Xóa Canvas.")
                                # Xóa canvas nếu không có gì để dự đoán
                                canvas = np.full((CANVAS_H, CANVAS_W, 3), CANVAS_COLOR, dtype=np.uint8)
                                has_drawn_since_clear = False
                                last_point = None
                                is_drawing = False
                    else:
                        last_point = None
                        is_drawing = False
            else:
                last_point = None
                is_drawing = False

        # --- 2. Tạo Khung Tổng hợp (Camera + Canvas) ---
        y_offset_display = (CAM_H - CANVAS_H) // 2 
        canvas_display = np.full((CAM_H, CANVAS_W, 3), CANVAS_COLOR, dtype=np.uint8)
        canvas_display[y_offset_display:y_offset_display+CANVAS_H, :] = canvas
        combined_frame = np.concatenate((frame, canvas_display), axis=1)

        # --- 3. Hiển thị thông tin lên Canvas ---
        display_canvas_area = combined_frame[:, CAM_W:]
        cv2.putText(display_canvas_area, f"Mode: {drawing_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_canvas_area, "Press P to Predict, C to Clear", (10, CANVAS_H + y_offset_display - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        if prediction_result and prediction_result['prediction'] not in ['No drawing found', 'Model Not Ready']:
            main_pred = prediction_result['prediction'].upper()
            main_prob = prediction_result['probability']
            cv2.putText(display_canvas_area, f"PREDICTION: {main_pred}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
            cv2.putText(display_canvas_area, f"Confidence: {main_prob}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)

            top_k = prediction_result['top_k']
            cv2.putText(display_canvas_area, "Top 5:", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            for i, res in enumerate(top_k):
                text = f"{i+1}. {res['class']} ({res['probability']})"
                cv2.putText(display_canvas_area, text, (10, 140 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        elif prediction_result and prediction_result['prediction'] == 'Model Not Ready':
            cv2.putText(display_canvas_area, "LỖI: Chưa tải được Model!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow('QuickDraw Classifier App', combined_frame)

        # --- 4. Xử lý phím bấm ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('c'): # Xóa Canvas
            canvas = np.full((CANVAS_H, CANVAS_W, 3), CANVAS_COLOR, dtype=np.uint8)
            prediction_result = None
            last_point = None
            has_drawn_since_clear = False
            print("Canvas đã được xóa.")
        elif key == ord('p'): # Dự đoán thủ công
            prediction_result = predict_class(canvas)
            if prediction_result:
                print(f"Dự đoán thủ công: {prediction_result['prediction']} - {prediction_result['probability']}")
        elif key == ord('m'): # Chuyển đổi chế độ
            if drawing_mode == 'MOUSE':
                drawing_mode = 'AIR'
                print("Đã chuyển sang chế độ AIR. Dùng ngón trỏ để vẽ, mở tay để dự đoán/xóa.")
            else:
                drawing_mode = 'MOUSE'
                print("Đã chuyển sang chế độ MOUSE. Dùng chuột vẽ trên Canvas.")
            
            # Reset trạng thái
            canvas = np.full((CANVAS_H, CANVAS_W, 3), CANVAS_COLOR, dtype=np.uint8)
            prediction_result = None
            last_point = None
            is_drawing = False
            has_drawn_since_clear = False
    
    # Giải phóng tài nguyên
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_app()