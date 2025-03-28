# Hatri: Hat Detection System for Raspberry Pi with Hailo-8

A real-time hat detection and tracking system optimized for Raspberry Pi 5 with Hailo-8 AI accelerator. This implementation is a Python port of the web-based Hatri system, redesigned to leverage hardware acceleration for edge deployment.

![Hatri System](https://github.com/oranblock/Hatri/blob/assets/hatri-rpi-demo.png)

## Features

- **Multi-Hat Detection**: Identify multiple hats simultaneously in real-time video
- **Advanced Color Analysis**: K-means clustering for accurate hat color identification
- **Multi-Object Tracking**: Track hats across video frames with persistent identities
- **Distance Estimation**: Calculate hat distances using face size references
- **Trajectory Prediction**: Predict movement patterns and future positions
- **Hardware Acceleration**: Optimized for Hailo-8 NPU on Raspberry Pi 5
- **Adaptive Processing**: Three performance modes to balance quality and speed
- **Lightweight UI**: Clean interface with performance metrics and visualization controls

## System Requirements

### Hardware
- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- Hailo-8 AI accelerator module
- USB webcam or Raspberry Pi Camera Module
- Display connected to Raspberry Pi

### Software
- Raspberry Pi OS (64-bit recommended)
- Python 3.9 or newer
- OpenCV 4.5+
- NumPy
- Hailo Runtime and SDK

## Quick Start

1. Install the application:
   ```bash
   ./install.sh
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run the application:
   ```bash
   python -m src.main
   ```

## Installation Details

The installation script performs the following steps:

1. Updates system packages
2. Installs required system dependencies
3. Creates a Python virtual environment
4. Installs Python packages
5. Sets up the Hailo SDK (if hardware is detected)
6. Prepares model files for inference
7. Creates a desktop shortcut (on Raspberry Pi)

If the Hailo-8 hardware is not detected, the system will run in simulation mode using CPU-based inference.

## Configuration

The application can be configured via command-line arguments:

```bash
python -m src.main --camera 0 --width 640 --height 480 --fps 30 --power medium
```

Options:
- `--camera`: Camera device ID (default: 0)
- `--width`: Capture width (default: 640)
- `--height`: Capture height (default: 480)  
- `--fps`: Target frame rate (default: 30)
- `--power`: Processing power level (low/medium/high, default: medium)

## Performance Modes

| Mode   | Resolution | Processing            | Features                               |
|--------|------------|----------------------|----------------------------------------|
| Low    | 320x240    | Frame skipping: 2/3  | Basic tracking, reduced color analysis |
| Medium | 640x480    | Frame skipping: 1/2  | Full features, balanced performance    |
| High   | 1280x720   | All frames processed | Maximum quality with detailed analysis |

## Keyboard Controls

| Key       | Function                        |
|-----------|---------------------------------|
| `ESC`     | Exit application                |
| `F`       | Toggle fullscreen mode          |
| `C`       | Toggle controls display         |
| `D`       | Toggle debug information        |
| `1/2/3`   | Low/Medium/High processing      |
| `H`       | Toggle trajectory visualization |
| `G`       | Toggle face detection boxes     |

## System Architecture

### Detection Pipeline

The hat detection pipeline consists of these key components:

1. **Hailo Detector**: Interfaces with the Hailo-8 NPU to run object and face detection models
2. **Color Analyzer**: Uses k-means clustering to identify dominant colors
3. **Hat Validator**: Determines if detected objects are likely hats based on position and shape

### Tracking System

The tracking system provides object persistence across frames:

1. **Hat Tracker**: Maintains identity of hats between frames
2. **Trajectory Predictor**: Calculates movement vectors and predicts future positions
3. **Distance Estimator**: Calculates physical distances using face references

### Visualization

The UI system provides interactive controls and real-time feedback:

1. **Display Manager**: Handles window creation and user interaction
2. **Visualizer**: Renders detections, tracking data, and performance metrics
3. **Performance Monitor**: Tracks and reports system performance

## Development Mode

For development without Hailo hardware, the system includes OpenCV-based fallbacks:

```bash
# Install required simulation dependencies
pip install tensorflow opencv-python-headless

# Run in simulation mode (automatic when no Hailo device is detected)  
python -m src.main
```

## Custom Model Integration

To use your own detection models:

1. Place TensorFlow model files in `models/custom/`
2. Run the conversion script with custom paths:
   ```bash
   python -m models.model_conversion.convert_models --output-dir models/custom_hailo
   ```
3. Update paths in `config.py` to point to your custom models

## Troubleshooting

### Camera Access Issues
- Ensure the camera is properly connected
- Check permissions: `sudo usermod -a -G video $USER`
- Try a different camera ID: `python -m src.main --camera 1`

### Performance Issues
- Set lower processing mode: `python -m src.main --power low`
- Reduce resolution: `python -m src.main --width 320 --height 240`
- Check CPU throttling: `vcgencmd measure_temp`

### Hailo Connectivity
- Check USB connection
- Verify Hailo Runtime installation
- Try reinstalling the Hailo SDK

## Credits

This project is a Python port of the web-based Hat Detection System, adapted for Raspberry Pi and hardware acceleration. The core algorithms are preserved while optimizing for edge deployment.

## License

MIT