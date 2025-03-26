"""
Main entry point for the Hat Detection System for Raspberry Pi with Hailo-8.
"""

import logging
import time
import argparse
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from . import config
from .detection.detector import HatDetector
from .tracking.tracker import HatTracker
from .utils.performance import PerformanceMonitor, FrameRateController
from .ui.display import DisplayManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class HatDetectionApp:
    """
    Main application class for hat detection.
    """
    
    def __init__(self, 
                camera_id: int = config.CAMERA_ID, 
                processing_power: str = config.DEFAULT_PROCESSING_POWER,
                camera_width: int = config.CAMERA_WIDTH,
                camera_height: int = config.CAMERA_HEIGHT,
                camera_fps: int = config.CAMERA_FPS):
        """
        Initialize the hat detection application.
        
        Args:
            camera_id: Camera device ID
            processing_power: Initial processing power level
            camera_width: Camera capture width
            camera_height: Camera capture height
            camera_fps: Camera frame rate
        """
        self.camera_id = camera_id
        self.processing_power = processing_power
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        
        # Flag to control application loop
        self.running = False
        
        logger.info(f"Initializing HatDetectionApp with camera_id={camera_id}, "
                   f"processing_power={processing_power}")
        
        # Initialize components
        try:
            # Hat detector
            self.hat_detector = HatDetector(processing_power)
            
            # Hat tracker
            self.hat_tracker = HatTracker()
            
            # Performance monitor
            self.performance_monitor = PerformanceMonitor()
            
            # Frame rate controller
            self.fps_controller = FrameRateController(target_fps=camera_fps)
            
            # Display manager
            self.display_manager = DisplayManager()
            self.display_manager.set_power_change_callback(self.set_processing_power)
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            raise
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            Success flag
        """
        try:
            logger.info(f"Opening camera {self.camera_id}")
            
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            # Read first frame to verify camera is working
            ret, _ = self.cap.read()
            
            if not ret:
                logger.error("Failed to read frame from camera")
                return False
            
            logger.info(f"Camera initialized successfully with resolution "
                       f"{self.camera_width}x{self.camera_height} at {self.camera_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def run(self) -> None:
        """
        Run the hat detection application main loop.
        """
        # Initialize camera
        if not self.initialize_camera():
            logger.error("Camera initialization failed, exiting")
            return
        
        # Start application loop
        self.running = True
        frame_count = 0
        
        logger.info("Starting main detection loop")
        
        try:
            while self.running:
                # Start frame timing
                self.performance_monitor.start_frame()
                
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame, skipping")
                    continue
                
                # Increment frame counter
                frame_count += 1
                
                # Detect hats in the frame
                hat_detections = self.hat_detector.detect_hats(frame)
                
                # Update tracking information
                tracked_hats = self.hat_tracker.update(hat_detections)
                
                # End frame timing
                self.performance_monitor.end_frame(len(tracked_hats))
                
                # Get performance statistics
                performance_stats = self.performance_monitor.get_stats()
                
                # Display results
                self.display_manager.display_frame(
                    frame, 
                    tracked_hats, 
                    None,  # We don't pass face detections to display
                    performance_stats
                )
                
                # Handle user input
                continue_running, new_power = self.display_manager.handle_keyboard(
                    self.processing_power
                )
                
                # Update running flag
                self.running = continue_running
                
                # Update processing power if changed
                if new_power != self.processing_power:
                    self.set_processing_power(new_power)
                
                # Control frame rate
                self.fps_controller.wait()
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, exiting")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # Clean up resources
            self.cleanup()
    
    def set_processing_power(self, processing_power: str) -> None:
        """
        Change the processing power setting.
        
        Args:
            processing_power: New processing power level
        """
        if self.processing_power != processing_power:
            logger.info(f"Changing processing power from {self.processing_power} to {processing_power}")
            self.processing_power = processing_power
            
            # Update detector
            self.hat_detector.set_processing_power(processing_power)
            
            # Update resolution if camera is initialized
            if hasattr(self, 'cap') and self.cap.isOpened():
                new_resolution = config.PROCESSING_SETTINGS[processing_power]["resolution"]
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_resolution[1])
                logger.info(f"Updated camera resolution to {new_resolution}")
    
    def cleanup(self) -> None:
        """Release resources and cleanup."""
        logger.info("Cleaning up resources")
        
        # Release camera
        if hasattr(self, 'cap'):
            self.cap.release()
        
        # Close display window
        self.display_manager.close()
        
        logger.info("Cleanup complete")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hat Detection System for Raspberry Pi with Hailo-8")
    
    parser.add_argument("--camera", type=int, default=config.CAMERA_ID,
                        help=f"Camera device ID (default: {config.CAMERA_ID})")
    
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH,
                        help=f"Camera capture width (default: {config.CAMERA_WIDTH})")
    
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT,
                        help=f"Camera capture height (default: {config.CAMERA_HEIGHT})")
    
    parser.add_argument("--fps", type=int, default=config.CAMERA_FPS,
                        help=f"Target frames per second (default: {config.CAMERA_FPS})")
    
    parser.add_argument("--power", type=str, default=config.DEFAULT_PROCESSING_POWER,
                        choices=[config.ProcessingPower.LOW, 
                                config.ProcessingPower.MEDIUM, 
                                config.ProcessingPower.HIGH],
                        help=f"Processing power level (default: {config.DEFAULT_PROCESSING_POWER})")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting Hat Detection System for Raspberry Pi with Hailo-8")
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    try:
        # Create and run the application
        app = HatDetectionApp(
            camera_id=args.camera,
            processing_power=args.power,
            camera_width=args.width,
            camera_height=args.height,
            camera_fps=args.fps
        )
        
        # Run the application
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    
    logger.info("Application terminated")

if __name__ == "__main__":
    main()