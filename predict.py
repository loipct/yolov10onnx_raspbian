import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import time

class InferenceEngine:
    def __init__(self, model_path, confidence_threshold, input_shape):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        # Danh sách các tên lớp đối tượng (điều chỉnh theo mô hình của bạn)
        self.class_names = {
                        0: '7up',
                        1: 'coca',
                        2: 'lemona',
                        3: 'mat_ong',
                        4: 'nhadam',
                        5: 'olong',
                        6: 'proby',
                        7: 'revine',
                        8: 'satori',
                        9: 'warrior'
                    }
        
    @staticmethod
    def opencv_to_pil(cv2_image):
        """
        Convert an image from OpenCV (BGR) format to PIL (RGB) format.

        Args:
            cv2_image (np.ndarray): The image in OpenCV format (BGR).

        Returns:
            PIL.Image.Image: The image in PIL format (RGB).
        """
        # Chuyển đổi từ BGR sang RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Chuyển đổi từ NumPy Array sang PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    def preprocess_image(self, image):
        """
        Prepares input image before inference.

        Args:
            image_path (str): Path to the image file.
        """
        img = self.opencv_to_pil(image)
        
        # Thay đổi kích thước hình ảnh để phù hợp với kích thước đầu vào của mô hình
        img = img.resize((self.input_shape, self.input_shape))  # Hoặc kích thước đầu vào mong muốn của mô hình

        # Chuyển đổi hình ảnh thành mảng numpy và chuẩn bị dữ liệu đầu vào cho mô hình
        img = np.array(img).astype(np.float32)
        img /= 255.0  # Normalize hình ảnh từ [0, 255] thành [0.0, 1.0]

        # Thay đổi thứ tự kênh từ HWC sang CHW (Channel-Height-Width)
        img = np.transpose(img, (2, 0, 1))

        # Thêm chiều batch nếu cần
        img = np.expand_dims(img, axis=0)


        return img
  
    def run_inference(self, input_tensor):
        """
        Runs inference on the input tensor.

        Args:
            input_tensor (np.ndarray): Input tensor with shape (B, C, H, W).
        
        Returns:
            np.ndarray: Output tensor from the model.
        """
        input_name = self.get_input_name()
        output_name = self.get_output_name()

        # Đảm bảo đầu vào có dạng mảng 4 chiều
        inputs = {input_name: input_tensor}
        outputs = self.session.run([output_name], inputs)
        
        # Trả về mảng đầu ra mà không cần flatten
        return outputs[0]

    def filter_detections(self, results, image_shape):
        detections = []
        for detection in results[0]:
            left, top, right, bottom, confidence, class_id = detection
            if confidence >= self.confidence_threshold:
                x1 = int(left * image_shape[0]/self.input_shape )
                y1 = int(top * image_shape[1]/self.input_shape)
                x2 = int(right * image_shape[0]/self.input_shape)
                y2 = int(bottom * image_shape[1]/self.input_shape)
                detections.append({
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'class_name': self.class_names[int(class_id)]
                })
        return detections

    
    def draw_labels(self, image, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            print(label)
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def get_input_name(self):
        return self.session.get_inputs()[0].name

    def get_output_name(self):
        return self.session.get_outputs()[0].name
    
    def predict(self, image, image_shape):
        start_time = time.time()
        input_tensor_values = self.preprocess_image(image)
        results = self.run_inference(input_tensor_values)
        detections = self.filter_detections(results, image_shape)
        inference_time = time.time() - start_time
        print(f"Inference Time: {inference_time:.4f} seconds")
        image_with_labels = self.draw_labels(image, detections)
        return image_with_labels