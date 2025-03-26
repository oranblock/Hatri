# Hat Detection System - Conversion Project

This repository contains two implementations of the Hat Detection System:

1. **Web-based Implementation** (original project)
   - Located in: `/src` directory
   - Uses React, TypeScript, and TensorFlow.js
   - Runs in a browser with WebGL acceleration
   
2. **Raspberry Pi Implementation** (ported version)
   - Located in: `/hat_detection_rpi` directory
   - Uses Python, OpenCV, and Hailo-8 AI accelerator
   - Runs on Raspberry Pi 5 with hardware acceleration

## Conversion Overview

This project demonstrates the conversion of a web-based computer vision application to an embedded system with hardware acceleration. Key aspects of the conversion include:

1. **Algorithm preservation**: The same core algorithms were maintained (k-means clustering, hat validation, movement tracking)
2. **Platform adaptation**: Code was ported from TypeScript to Python with consideration for embedded constraints
3. **Hardware optimization**: The detection pipeline was adapted for Hailo-8 neural processing unit
4. **Performance tuning**: Multiple processing levels were implemented to balance quality and speed

## Repository Structure

- `/src` - Original TypeScript implementation
- `/hat_detection_rpi` - Python implementation for Raspberry Pi with Hailo-8
- `/python_conversion_plan.md` - Detailed conversion plan with implementation notes
- `/hat_conversion_summary.md` - This summary file

## Key Components

Both implementations feature:

- Multiple hat detection and tracking
- K-means clustering for color identification
- Face detection for relative positioning
- Trajectory prediction
- Distance estimation

## Getting Started

### Web Version (original)
```bash
npm install
npm run dev
```

### Raspberry Pi Version
```bash
cd hat_detection_rpi
./install.sh
source venv/bin/activate
python -m src.main
```

## Project Implementation Notes

The conversion project demonstrates several important patterns for porting web/browser applications to embedded systems:

1. **Modular Architecture**: The code was refactored into focused modules with clear responsibilities
2. **Hardware Abstraction**: Detection logic was separated from hardware-specific acceleration
3. **Simulation Mode**: A fallback mode allows development without specialized hardware
4. **Performance Monitoring**: Both implementations track and report performance metrics
5. **Consistent Algorithms**: Core computer vision algorithms remained consistent across platforms