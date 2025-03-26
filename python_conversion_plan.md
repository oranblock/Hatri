# Hat Detection System - Conversion Plan

## Project Overview
Converting the web-based hat detection system (React, TypeScript, TensorFlow.js) to a Python application running on Raspberry Pi 5 with Hailo-8 AI accelerator.

## Source Analysis
The original project implements a web-based hat detection system with these key components:

1. **TensorflowContext.tsx**: Handles model loading (COCO-SSD and BlazeFace)
2. **HatDetector.tsx**: Implements the computer vision pipeline including:
   - Object detection with COCO-SSD
   - Face detection with BlazeFace
   - K-means clustering for hat color analysis
   - Multi-object tracking with persistence
   - Distance estimation via face reference
   - Movement prediction and trajectory visualization

## Target Implementation

### Platform
- Raspberry Pi 5
- Hailo-8 AI accelerator
- Python-based implementation
- OpenCV, NumPy, Hailo SDK

### Functional Requirements
- Detect multiple hats simultaneously
- Identify hat colors using k-means clustering
- Track hats across video frames
- Estimate distance of hats
- Predict hat movement and trajectory
- Performance: 30+ FPS with <40ms latency

## Project Structure

```
hat_detection_rpi/
├── README.md                   # Setup and usage instructions
├── requirements.txt            # Python dependencies
├── install.sh                  # Installation script
├── models/                     # Directory for AI models
│   ├── hailo_models/           # Compiled Hailo models
│   └── model_conversion/       # Scripts for model conversion
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration parameters
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py         # Main detector class
│   │   ├── hailo_detector.py   # Hailo-specific detection
│   │   ├── color_analyzer.py   # K-means color clustering
│   │   └── hat_validator.py    # Hat probability validation
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py          # Multi-object tracker
│   │   ├── trajectory.py       # Movement prediction
│   │   └── distance.py         # Distance estimation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py    # Drawing and UI helpers
│   │   └── performance.py      # Performance monitoring
│   └── ui/
│       ├── __init__.py
│       └── display.py          # OpenCV window or GTK interface
└── tests/                      # Unit and integration tests
```

## Key Algorithm Ports

### 1. K-means Color Clustering
Convert the `detectDominantColors` function to Python:
```python
# From HatDetector.tsx line ~464
def detect_dominant_colors(image_data, num_clusters=3):
    # Sample pixels for performance
    pixels = []
    # Extract pixels from OpenCV image data
    # Use NumPy for efficient operations
    
    # K-means clustering implementation
    color_clusters = k_means_clustering(pixels, num_clusters)
    
    # Find the most saturated/colorful cluster
    dominant_cluster = color_clusters[0]
    highest_saturation = color_saturation(dominant_cluster.centroid)
    
    for cluster in color_clusters[1:]:
        saturation = color_saturation(cluster.centroid)
        if saturation > highest_saturation:
            highest_saturation = saturation
            dominant_cluster = cluster
    
    r, g, b = map(int, dominant_cluster.centroid)
    color_name = find_nearest_named_color(r, g, b)
    
    return {
        'r': r,
        'g': g,
        'b': b,
        'color_name': color_name
    }
```

### 2. Hat Probability Estimation
Convert the `isLikelyHat` and `calculateHatConfidence` functions:
```python
# From HatDetector.tsx line ~640
def is_likely_hat(prediction, face_detections):
    # Check if the prediction might be a hat
    possible_hat_classes = ['hat', 'cap', 'helmet', 'sports ball', 'frisbee', 'bowl']
    
    # Direct class match
    if any(cls in prediction['class'].lower() for cls in possible_hat_classes):
        return True
    
    # Position relative to faces
    for face in face_detections:
        # Extract position data
        # Check if object is above and centered with a face
        # Validate size and shape
        pass
    
    return False

# From HatDetector.tsx line ~678
def calculate_hat_confidence(prediction, face_detections):
    # Start with model confidence
    confidence = prediction['score']
    
    # Adjust for different classes
    direct_hat_classes = ['hat', 'cap', 'helmet']
    secondary_hat_classes = ['sports ball', 'frisbee', 'bowl']
    
    # Apply position-based confidence adjustments
    # Use the same weighting factors as original
    
    return min(1.0, max(0.0, confidence))
```

### 3. Non-Maximum Suppression
Use OpenCV's built-in NMS or implement:
```python
# From HatDetector.tsx line ~746
def non_maximum_suppression(detections, overlap_threshold=0.5):
    # Sort detections by confidence
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    selected_detections = []
    
    while sorted_detections:
        current = sorted_detections.pop(0)
        selected_detections.append(current)
        
        # Filter remaining detections
        remaining = []
        for detection in sorted_detections:
            overlap = calculate_iou(current['bbox'], detection['bbox'])
            if overlap < overlap_threshold:
                remaining.append(detection)
        
        sorted_detections = remaining
    
    return selected_detections
```

## Hailo-Specific Implementations

### 1. Model Conversion
```python
# model_conversion/convert_models.py
from hailo_model_zoo import convert_model

def convert_coco_ssd():
    """Convert COCO-SSD model to Hailo format"""
    model_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    convert_model(model_url, "models/hailo_models/coco_ssd")

def convert_blazeface():
    """Convert BlazeFace model to Hailo format"""
    model_url = "https://github.com/tensorflow/tfjs-models/tree/master/blazeface"
    convert_model(model_url, "models/hailo_models/blazeface")
```

### 2. Hailo Inference
```python
# detection/hailo_detector.py
from hailo_platform import HailoInfer

class HailoDetector:
    def __init__(self):
        self.object_detector = HailoInfer("models/hailo_models/coco_ssd")
        self.face_detector = HailoInfer("models/hailo_models/blazeface")
    
    def detect(self, frame):
        """Run parallel detection using Hailo acceleration"""
        # Use Hailo's zero-copy API for optimal performance
        object_results = self.object_detector.infer(frame)
        face_results = self.face_detector.infer(frame)
        
        return object_results, face_results
```

## Performance Optimizations

1. **Zero-copy inference** with Hailo SDK
2. **Vectorized operations** using NumPy
3. **Optimized OpenCV functions** for image processing
4. **Parallel processing** where applicable
5. **Frame skipping** based on processing power (similar to source)

## UI Implementation

```python
# ui/display.py
import cv2
import numpy as np

class HatDetectionUI:
    def __init__(self, window_name="Hat Detection System"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def display_frame(self, frame, detections, tracked_hats):
        # Draw detection boxes and tracking info
        for hat_id, hat in tracked_hats.items():
            # Draw bounding box
            cv2.rectangle(frame, 
                          (int(hat['x']), int(hat['y'])), 
                          (int(hat['x'] + hat['width']), int(hat['y'] + hat['height'])),
                          (int(hat['color_rgb']['r']), int(hat['color_rgb']['g']), int(hat['color_rgb']['b'])),
                          2)
            
            # Draw labels with color and confidence
            cv2.putText(frame, 
                        f"{hat['color']} hat ({int(hat['confidence']*100)}%)",
                        (int(hat['x']), int(hat['y'] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
            # Draw movement vector if moving
            if abs(hat['velocity_x']) > 1 or abs(hat['velocity_y']) > 1:
                # Draw trajectory
                pass
        
        # Show performance metrics
        cv2.putText(frame, 
                    f"FPS: {self.fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow(self.window_name, frame)
        
    def handle_keys(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        return True
```

## Installation Script

```bash
#!/bin/bash
# install.sh

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "Installing dependencies..."
sudo apt install -y python3-pip python3-opencv python3-numpy

# Install Hailo SDK
echo "Installing Hailo SDK..."
pip3 install hailo-platform

# Install Python requirements
echo "Installing Python requirements..."
pip3 install -r requirements.txt

# Create model directories
mkdir -p models/hailo_models

# Convert models
echo "Converting models for Hailo (this may take a while)..."
python3 models/model_conversion/convert_models.py

echo "Installation complete!"
echo "Run 'python3 src/main.py' to start the application"
```

## Implementation Timeline

1. Basic project setup and camera integration (1 day)
2. Port core detection algorithms to Python/OpenCV (2 days)
3. Implement K-means color clustering (1 day)
4. Create multi-object tracking system (2 days)
5. Integrate with Hailo SDK and optimize models (2 days)
6. Develop UI and visualization (1 day)
7. Testing and performance optimization (2 days)

## Next Steps

1. Set up Raspberry Pi 5 with Hailo-8 accelerator
2. Install required dependencies
3. Implement core detection modules
4. Port tracking algorithms
5. Integrate with Hailo SDK
6. Build UI
7. Test and optimize performance