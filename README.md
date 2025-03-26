# Hatri - Advanced Hat Detection & Tracking System

A comprehensive hat detection and tracking system with two implementations:

1. **Web-based Implementation** (TypeScript/React/TensorFlow.js)
2. **Raspberry Pi Implementation** (Python/OpenCV/Hailo-8)

## Project Overview

This project demonstrates advanced computer vision techniques for real-time hat detection, color analysis, and movement tracking. The system uses machine learning models for detection, k-means clustering for color identification, and custom tracking algorithms for persistent object tracking.

## Web Implementation

The original implementation uses TensorFlow.js to run computer vision models in the browser, leveraging WebGL acceleration for real-time performance.

### Features
- Object detection with ML model classification
- Face-relative hat positioning analysis
- K-means clustering for color analysis
- Non-maximum suppression for overlapping detections
- Multi-object tracking with persistence
- Trajectory prediction and visualization
- Distance estimation via face size reference
- Movement direction classification
- **Fullscreen mode** for better visualization
- **Front/back camera switching**
- **Adjustable processing power settings**

### Running the Web Implementation

#### Development Mode
```
npm run dev
```

If you encounter "too many open files" errors, try:
```
./dev.sh
```

#### Production Mode (Recommended)
Build the application:
```
npm run build
```

Serve using Python:
```
./serve.sh
```
This will serve the application at http://localhost:8000

Alternative server (if Python is not available):
```
./serve-alt.sh
```
This will serve the application at http://localhost:8080

## Raspberry Pi Implementation

The Python-based implementation is optimized for Raspberry Pi 5 with Hailo-8 AI accelerator, designed to run efficiently on edge hardware.

### Features
- Hardware-accelerated object and face detection
- Optimized k-means clustering for color analysis
- Multi-object tracking with movement prediction
- Distance estimation using face size reference
- Trajectory visualization with predictive paths
- Multiple processing power modes
- Simple OpenCV-based UI with performance statistics

### Running the Raspberry Pi Implementation

```bash
cd hat_detection_rpi
./install.sh
source venv/bin/activate
python -m src.main
```

See the detailed instructions in [hat_detection_rpi/README.md](hat_detection_rpi/README.md)

## Implementation Details

Both implementations feature the same core algorithms:

- **Hat Detection Pipeline**: Object detection models identify potential hats based on class and position
- **Color Analysis**: K-means clustering extracts dominant colors from hat regions
- **Face Detection**: Face coordinates provide reference for hat positioning and distance estimation
- **Tracking System**: Multi-object tracking maintains hat identities across frames
- **Trajectory Prediction**: Linear prediction of movement paths based on velocity

## Conversion Project

This repository contains a complete conversion of the web implementation to a Python-based solution for embedded hardware. Read about the conversion process in:

- [Conversion Plan](python_conversion_plan.md): Detailed plan for porting the application
- [Conversion Summary](hat_conversion_summary.md): Overview of the conversion process and key decisions

## Technical Details

### Web Implementation
- Uses TensorFlow.js for neural network inference
- Employs the COCO-SSD model for object detection
- Uses BlazeFace for face detection
- Implements custom k-means clustering for color analysis
- Provides motion tracking and trajectory prediction
- Adapts model complexity based on selected processing power

### Raspberry Pi Implementation
- Uses Hailo-8 NPU for hardware-accelerated inference
- Optimized Python implementation of computer vision algorithms
- OpenCV-based visualization and camera handling
- Multiple performance levels for different hardware capabilities

## License

MIT

## Credits

This project demonstrates advanced computer vision techniques for real-time object detection, tracking, and analysis, with implementations for both web browsers and embedded hardware.