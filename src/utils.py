import cv2
import json
from pathlib import Path

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels on the image.
    Args:
        image (numpy.ndarray): Input image
        detections (list): List of detection dictionaries
    Returns:
        numpy.ndarray: Annotated image
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Set color based on class (green for person, red for basketball)
        color = (0, 255, 0) if class_name == 'person' else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'{class_name} {confidence:.2f}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def save_detection_results(detections, output_path):
    """
    Save detection results to a JSON file.
    Args:
        detections (list): List of detection dictionaries
        output_path (str): Path to save the JSON file
    """
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)

def create_data_dirs():
    """
    Create necessary directories for the project.
    """
    dirs = ['data', 'output', 'models']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)