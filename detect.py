import cv2
import time
from predict import InferenceEngine

if __name__ == "__main__":
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    frame_shape = (1440, 720)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Codec MJPG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])

    # Sử dụng lớp InferenceEngine
    model_path = 'weights/yolov10s320.onnx'
    class_path = 'weights/classes.txt'
    confidence_threshold = 0.7
    input_shape = 320
    engine = InferenceEngine(model_path, class_path, confidence_threshold, input_shape)

    # Biến để tính toán FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # Đọc khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy khung hình từ camera.")
            break

        # Xử lý khung hình
        anno_frame = engine.predict(frame, frame_shape)

        # Tính toán FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Hiển thị FPS lên khung hình
        cv2.putText(anno_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Hiển thị khung hình
        # cv2.imshow('Detection', anno_frame)

        # Nhấn 'q' để thoát khỏi vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
