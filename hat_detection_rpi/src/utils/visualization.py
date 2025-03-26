"""
Visualization utilities for the hat detection system.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from .. import config

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Handles visualization of detections and tracking information.
    """
    
    def __init__(self, show_faces: bool = True, show_trajectory: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            show_faces: Whether to display face detection boxes
            show_trajectory: Whether to display movement vectors and trajectories
        """
        self.show_faces = show_faces
        self.show_trajectory = show_trajectory
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
        logger.info(f"Visualizer initialized with show_faces={show_faces}, show_trajectory={show_trajectory}")
    
    def draw_detections(self, 
                       frame: np.ndarray, 
                       hat_detections: Dict[str, Any], 
                       face_detections: Optional[List[Dict[str, Any]]] = None,
                       performance_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Draw hat detections, tracking info, and performance stats on a frame.
        
        Args:
            frame: The video frame to draw on
            hat_detections: Dictionary of tracked hats
            face_detections: Optional list of face detections
            performance_stats: Optional performance statistics
            
        Returns:
            Frame with visualizations added
        """
        # Make a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw face detections if enabled and available
        if self.show_faces and face_detections:
            self._draw_faces(vis_frame, face_detections)
        
        # Draw hat detections
        for hat_id, hat in hat_detections.items():
            self._draw_hat(vis_frame, hat)
        
        # Draw performance stats if available
        if performance_stats:
            self._draw_performance_stats(vis_frame, performance_stats)
        
        return vis_frame
    
    def _draw_faces(self, frame: np.ndarray, face_detections: List[Dict[str, Any]]) -> None:
        """
        Draw face detection boxes.
        
        Args:
            frame: The frame to draw on
            face_detections: List of face detections
        """
        for face in face_detections:
            try:
                if "topLeft" in face and "bottomRight" in face:
                    # BlazeFace format
                    x1, y1 = map(int, face["topLeft"])
                    x2, y2 = map(int, face["bottomRight"])
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                else:
                    # OpenCV format
                    x, y, w, h = map(int, face["bbox"])
                
                # Draw semi-transparent face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0, 128), 1)
                
            except Exception as e:
                logger.error(f"Error drawing face: {e}")
                continue
    
    def _draw_hat(self, frame: np.ndarray, hat: Dict[str, Any]) -> None:
        """
        Draw a hat detection with tracking information.
        
        Args:
            frame: The frame to draw on
            hat: Hat detection data
        """
        try:
            # Extract hat dimensions
            x = int(hat["x"])
            y = int(hat["y"])
            width = int(hat["width"])
            height = int(hat["height"])
            color_rgb = hat["color_rgb"]
            color_name = hat["color"]
            confidence = hat.get("confidence", 0.0)
            frames_tracked = hat.get("frames_tracked", 1)
            
            # Draw bounding box - thicker for hats tracked longer
            line_width = min(5, 1 + int(frames_tracked / 10))
            color = (int(color_rgb["b"]), int(color_rgb["g"]), int(color_rgb["r"]))  # BGR format
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, line_width)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - 30), (x + width, y), color, -1)  # Filled rectangle
            
            # Draw label text (color name and confidence)
            text = f"{color_name} hat ({int(confidence * 100)}%)"
            cv2.putText(frame, text, (x + 5, y - 10), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # Draw distance info if available
            if "distance_label" in hat:
                cv2.putText(frame, f"Distance: {hat['distance_label']}", 
                           (x + 5, y + height + 15), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # Draw tracking ID
            short_id = hat["id"][:8] if len(hat["id"]) > 8 else hat["id"]
            cv2.putText(frame, f"ID: {short_id}", 
                       (x + 5, y + height + 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            # Draw movement vector if hat is moving and trajectory visualization is enabled
            if self.show_trajectory:
                center_x = int(hat["center_x"])
                center_y = int(hat["center_y"])
                velocity_x = hat.get("velocity_x", 0)
                velocity_y = hat.get("velocity_y", 0)
                
                if abs(velocity_x) > 1 or abs(velocity_y) > 1:
                    # Draw velocity vector
                    end_x = int(center_x + velocity_x * 5)
                    end_y = int(center_y + velocity_y * 5)
                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
                    
                    # Draw predicted future position if available
                    if "predicted_x" in hat and "predicted_y" in hat:
                        pred_x = int(hat["predicted_x"])
                        pred_y = int(hat["predicted_y"])
                        cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 255), -1)
            
        except Exception as e:
            logger.error(f"Error drawing hat: {e}")
    
    def _draw_performance_stats(self, frame: np.ndarray, stats: Dict[str, Any]) -> None:
        """
        Draw performance statistics on the frame.
        
        Args:
            frame: The frame to draw on
            stats: Performance statistics
        """
        try:
            # Create a semi-transparent background for stats
            height, width = frame.shape[:2]
            bg_width = 200
            bg_height = 20
            
            # Create a black rectangle with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (5 + bg_width, 5 + bg_height), (0, 0, 0), -1)
            
            # Apply transparency
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw FPS text
            fps = stats.get("fps", 0)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                       self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
        except Exception as e:
            logger.error(f"Error drawing performance stats: {e}")
    
    def set_show_faces(self, show_faces: bool) -> None:
        """Set whether to show face detections."""
        self.show_faces = show_faces
    
    def set_show_trajectory(self, show_trajectory: bool) -> None:
        """Set whether to show trajectory predictions."""
        self.show_trajectory = show_trajectory