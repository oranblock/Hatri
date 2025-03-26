# Advanced Hat Detection & Tracking System

This project uses TensorFlow.js with advanced computer vision algorithms for real-time hat detection and tracking. The system performs k-means clustering for color analysis, maintains object persistence across frames, and provides detailed analytics on each detected hat including distance estimation, movement vectors, and trajectory prediction.

## Features
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

## Running the Application

### Development Mode
```
npm run dev
```

If you encounter "too many open files" errors, try:
```
./dev.sh
```

### Production Mode (Recommended)
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

## Instructions
1. Allow camera access when prompted
2. Point the camera at people wearing hats
3. The system will detect hats, analyze their colors, and track their movements
4. See real-time analytics in the table below the video feed

## Camera Controls
- **Camera On/Off**: Toggle the camera feed
- **Switch Camera**: Switch between front and back cameras (if available)
- **Processing Power**: Adjust the balance between performance and quality
  - Low: Better performance, reduced quality
  - Medium: Balanced (default)
  - High: Better quality, may be slower on some devices

## Fullscreen Mode
- Click the "Fullscreen" button in the top-right corner to enter fullscreen mode
- In fullscreen mode, the hat detection canvas will maximize to fill your screen
- A small overlay will show the number of hats being tracked, camera mode, and processing power
- Press "Exit Fullscreen" or ESC key to return to normal mode

## Technical Details
- Uses TensorFlow.js for neural network inference
- Employs the COCO-SSD model for object detection
- Uses BlazeFace for face detection
- Implements k-means clustering for advanced color analysis
- Provides motion tracking and trajectory prediction
- Adapts model complexity based on selected processing power level