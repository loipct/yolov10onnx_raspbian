# tests/test_inference_engine.py
import sys
import os
# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from predict import InferenceEngine  # Đảm bảo bạn đã cài đặt class InferenceEngine


model_path = r"weights/yolov10s320.onnx"
input_shape = 320
confidence_threshold = 0.5

def test_import_onnxruntime():
    """
    Test that ONNX Runtime is imported correctly.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.fail("onnxruntime is not installed or not importable")

def test_inference_engine_initialization():
    """
    Test the initialization of the InferenceEngine class.
    """

    engine = InferenceEngine(model_path, confidence_threshold, input_shape)
    
    assert engine.session is not None, "Inference session should be initialized"
    assert engine.confidence_threshold == confidence_threshold, "Confidence threshold should match"
    assert engine.input_shape == input_shape, "Input shape should match"

def test_preprocess_image():
    """
    Test the preprocess_image method.
    """

    engine = InferenceEngine(model_path, confidence_threshold, input_shape)
    
    # Tạo hình ảnh mẫu
    cv2_image = np.zeros((input_shape, input_shape, 3), dtype=np.uint8)
    
    processed_image = engine.preprocess_image(cv2_image)
    
    assert processed_image.shape == (1, 3, input_shape, input_shape), "Processed image shape is incorrect"
    assert np.max(processed_image) <= 1.0 and np.min(processed_image) >= 0.0, "Image normalization failed"

def test_run_inference():
    """
    Test the run_inference method.
    """

    engine = InferenceEngine(model_path, confidence_threshold, input_shape)
    
    # Tạo tensor đầu vào mẫu
    input_tensor = np.random.rand(1, 3, input_shape, input_shape).astype(np.float32)
    
    results = engine.run_inference(input_tensor)
    
    assert isinstance(results, np.ndarray), "Inference results should be a numpy array"

def test_filter_detections():
    """
    Test the filter_detections method.
    """

    engine = InferenceEngine(model_path, confidence_threshold, input_shape)
    
    # Tạo kết quả phát hiện mẫu
    results = [np.array([
        [50, 50, 150, 150, 0.6, 1],
        [30, 30, 100, 100, 0.4, 2]
    ])]
    
    image_shape = (640, 480)
    
    detections = engine.filter_detections(results, image_shape)
    
    assert len(detections) == 1, "There should be one detection"
    assert detections[0]['confidence'] == 0.6, "Confidence value is incorrect"
    assert detections[0]['class_id'] == 1, "Class ID is incorrect"

def test_draw_labels():
    """
    Test the draw_labels method.
    """

    engine = InferenceEngine(model_path, confidence_threshold, input_shape)
    
    # Tạo hình ảnh mẫu
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    
    # Tạo kết quả phát hiện mẫu
    detections = [{
        'confidence': 0.6,
        'bbox': (50, 50, 150, 150),
        'class_id': 1,
        'class_name': 'coca'
    }]
    
    image_with_labels = engine.draw_labels(image, detections)
    
    assert image_with_labels is not None, "Image with labels should not be None"

if __name__ == "__main__":
    pytest.main()
