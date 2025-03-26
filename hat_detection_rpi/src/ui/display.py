"""
Display module for hat detection visualization.
"""

import logging
import cv2
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable

from .. import config
from ..utils.visualization import Visualizer

logger = logging.getLogger(__name__)

class DisplayManager:
    """
    Manages the display window and user interface for hat detection.
    """
    
    def __init__(self, window_name: str = config.WINDOW_NAME):
        """
        Initialize the display manager.
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.visualizer = Visualizer(
            show_faces=config.SHOW_FACES,
            show_trajectory=config.SHOW_TRAJECTORY
        )
        
        # UI state
        self.fullscreen = False
        self.show_controls = True
        self.show_debug = False
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Callback for processing power change
        self.on_power_change = None
        
        logger.info(f"DisplayManager initialized with window_name={window_name}")
    
    def display_frame(self, 
                     frame: np.ndarray, 
                     hat_detections: Dict[str, Any], 
                     face_detections: Optional[List[Dict[str, Any]]] = None,
                     performance_stats: Optional[Dict[str, Any]] = None) -> None:
        """
        Display a frame with detections and UI.
        
        Args:
            frame: The video frame to display
            hat_detections: Dictionary of tracked hats
            face_detections: Optional list of face detections
            performance_stats: Optional performance statistics
        """
        try:
            # Draw detections on the frame
            display_frame = self.visualizer.draw_detections(
                frame, hat_detections, face_detections, performance_stats
            )
            
            # Add UI controls if enabled
            if self.show_controls:
                self._draw_controls(display_frame, hat_detections, performance_stats)
            
            # Show debug info if enabled
            if self.show_debug and performance_stats:
                self._draw_debug_info(display_frame, performance_stats)
            
            # Display the frame
            cv2.imshow(self.window_name, display_frame)
            
            # Handle fullscreen toggle if needed
            if self.fullscreen:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
    
    def handle_keyboard(self, processing_power: str = config.DEFAULT_PROCESSING_POWER) -> Tuple[bool, str]:
        """
        Handle keyboard input.
        
        Args:
            processing_power: Current processing power level
            
        Returns:
            Tuple of (continue_running, new_processing_power)
        """
        key = cv2.waitKey(1) & 0xFF
        
        # Continue flag and current processing power
        continue_running = True
        new_power = processing_power
        
        # Process key presses
        if key == 27:  # ESC key
            continue_running = False
        elif key == ord('f'):  # Toggle fullscreen
            self.fullscreen = not self.fullscreen
            logger.info(f"Fullscreen mode: {self.fullscreen}")
        elif key == ord('c'):  # Toggle controls
            self.show_controls = not self.show_controls
            logger.info(f"Controls display: {self.show_controls}")
        elif key == ord('d'):  # Toggle debug info
            self.show_debug = not self.show_debug
            logger.info(f"Debug display: {self.show_debug}")
        elif key == ord('1'):  # Set low processing power
            new_power = config.ProcessingPower.LOW
            if self.on_power_change:
                self.on_power_change(new_power)
            logger.info(f"Processing power set to: {new_power}")
        elif key == ord('2'):  # Set medium processing power
            new_power = config.ProcessingPower.MEDIUM
            if self.on_power_change:
                self.on_power_change(new_power)
            logger.info(f"Processing power set to: {new_power}")
        elif key == ord('3'):  # Set high processing power
            new_power = config.ProcessingPower.HIGH
            if self.on_power_change:
                self.on_power_change(new_power)
            logger.info(f"Processing power set to: {new_power}")
        elif key == ord('h'):  # Toggle hat trajectory
            self.visualizer.set_show_trajectory(not self.visualizer.show_trajectory)
            logger.info(f"Show trajectory: {self.visualizer.show_trajectory}")
        elif key == ord('g'):  # Toggle face display
            self.visualizer.set_show_faces(not self.visualizer.show_faces)
            logger.info(f"Show faces: {self.visualizer.show_faces}")
        
        return continue_running, new_power
    
    def set_power_change_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback for processing power changes.
        
        Args:
            callback: Function to call when processing power changes
        """
        self.on_power_change = callback
    
    def _draw_controls(self, frame: np.ndarray, hat_detections: Dict[str, Any], stats: Optional[Dict[str, Any]]) -> None:
        """
        Draw control panel and information.
        
        Args:
            frame: The frame to draw on
            hat_detections: Dictionary of tracked hats
            stats: Performance statistics
        """
        height, width = frame.shape[:2]
        
        # Control panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 150), (250, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Hat Detection System", (20, height - 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tracking info
        hats_count = len(hat_detections)
        cv2.putText(frame, f"Tracking {hats_count} hats", (20, height - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Performance info
        if stats:
            fps = stats.get("fps", 0)
            proc_time = stats.get("avg_processing_time_ms", 0)
            cv2.putText(frame, f"FPS: {fps:.1f} ({proc_time:.1f}ms/frame)", (20, height - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Controls:", (20, height - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ESC: Exit | F: Fullscreen | D: Debug", (20, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "1/2/3: Low/Med/High power | H: Toggle trajectory", (20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_debug_info(self, frame: np.ndarray, stats: Dict[str, Any]) -> None:
        """
        Draw detailed performance and debug information.
        
        Args:
            frame: The frame to draw on
            stats: Performance statistics
        """
        height, width = frame.shape[:2]
        
        # Debug panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 310, 10), (width - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Debug Information", (width - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance details
        line_y = 60
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if 'time' in key or 'fps' in key:
                    # Format time/fps with 1 decimal place
                    display_value = f"{value:.1f}"
                else:
                    # Format other numbers as integers
                    display_value = f"{int(value)}"
            else:
                display_value = str(value)
            
            # Format key for display (convert snake_case to Title Case)
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            
            cv2.putText(frame, f"{display_key}: {display_value}", (width - 300, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            line_y += 20
    
    def close(self) -> None:
        """Close the display window."""
        cv2.destroyWindow(self.window_name)