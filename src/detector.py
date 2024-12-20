from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from utils import draw_detections, save_detection_results

class BasketballDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the basketball detector with a YOLO model.
        Args:
            model_path (str): Path to the YOLO model weights
        """
        self.model = YOLO(model_path)
        # COCO classes: person=0, sports ball=32
        self.target_classes = [0, 32]
        self.class_names = {0: 'person', 32: 'basketball'}
    
    def detect_image(self, image_path, conf_threshold=0.25):
        """
        Detect basketball players and balls in an image.
        Args:
            image_path (str): Path to the input image
            conf_threshold (float): Confidence threshold for detections
        Returns:
            tuple: (annotated image, list of detections)
        """
        results = self.model(image_path, conf=conf_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        # Draw detections on the image
        img = cv2.imread(str(image_path))
        annotated_img = draw_detections(img.copy(), detections)
        
        return annotated_img, detections
    
    def detect_video(self, video_path, output_path=None, show_video=True):
        """
        Detect basketball players and balls in a video.
        Args:
            video_path (str): Path to the input video
            output_path (str): Path to save the output video (optional)
            show_video (bool): Whether to display the video while processing
        """
        cap = cv2.VideoCapture(video_path if video_path else 0)
        
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            results = self.model(frame)
            annotated_frame = results[0].plot()
            
            if show_video:
                cv2.imshow('Basketball Detection', annotated_frame)
            
            if output_path:
                out.write(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

def main():
    # Example usage
    detector = BasketballDetector()
    
    # Image detection example
    image_path = 'data/sample.jpg'
    if Path(image_path).exists():
        annotated_img, detections = detector.detect_image(image_path)
        cv2.imshow('Detection Result', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save results
        output_path = 'output/detection_result.jpg'
        Path('output').mkdir(exist_ok=True)
        cv2.imwrite(output_path, annotated_img)
        save_detection_results(detections, 'output/detections.json')
    
    # Video detection example
    video_path = 'data/sample.mp4'
    if Path(video_path).exists():
        detector.detect_video(video_path, output_path='output/detection_result.mp4')

if __name__ == '__main__':
    main()