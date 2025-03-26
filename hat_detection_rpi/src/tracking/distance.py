"""
Distance estimation module for hat detection.
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class DistanceEstimator:
    """
    Estimates distances of hats based on face size reference.
    """
    
    def __init__(self, 
                 reference_face_width: float = 300, 
                 scale_factor: float = 100,
                 camera_focal_length: Optional[float] = None,
                 camera_sensor_width: Optional[float] = None):
        """
        Initialize the distance estimator.
        
        Args:
            reference_face_width: Reference face width in pixels at 1 meter
            scale_factor: Scaling factor for distance calculation
            camera_focal_length: Optional camera focal length in mm
            camera_sensor_width: Optional camera sensor width in mm
        """
        self.reference_face_width = reference_face_width
        self.scale_factor = scale_factor
        self.camera_focal_length = camera_focal_length
        self.camera_sensor_width = camera_sensor_width
        
        logger.info(f"DistanceEstimator initialized with reference width {reference_face_width}px")
        
        # If camera parameters are provided, use them for more accurate estimation
        self.use_camera_params = (camera_focal_length is not None and camera_sensor_width is not None)
        if self.use_camera_params:
            logger.info(f"Using camera parameters: focal length={camera_focal_length}mm, sensor width={camera_sensor_width}mm")
    
    def estimate_from_face(self, face_width: float) -> Tuple[int, str]:
        """
        Estimate distance based on face width.
        
        Args:
            face_width: Width of the face in pixels
            
        Returns:
            Tuple of (distance_value, distance_label)
        """
        if face_width <= 0:
            return (None, "unknown")
        
        try:
            # Basic inverse relationship between face width and distance
            distance = int((self.scale_factor * self.reference_face_width) / face_width)
            
            # Format distance label
            if distance < 100:
                distance_label = f"~{distance}cm"
            else:
                distance_m = distance / 100.0
                distance_label = f"~{distance_m:.1f}m"
                
            return (distance, distance_label)
            
        except Exception as e:
            logger.error(f"Error estimating distance: {e}")
            return (None, "unknown")
    
    def estimate_from_camera_params(self, face_width: float, real_face_width: float = 15.0) -> Tuple[int, str]:
        """
        Estimate distance using camera parameters for more accuracy.
        
        Args:
            face_width: Width of the face in pixels
            real_face_width: Real width of a human face in cm (approx. 15cm average)
            
        Returns:
            Tuple of (distance_value, distance_label)
        """
        if not self.use_camera_params or face_width <= 0:
            return self.estimate_from_face(face_width)
        
        try:
            # Use camera parameters and face width for more accurate estimation
            # Distance = (Real face width * Focal length * Image width) / (Face width in pixels * Sensor width)
            
            # Convert focal length to pixels
            focal_length_px = self.camera_focal_length * face_width / (real_face_width / 100)
            
            # Calculate distance in cm
            distance = int((real_face_width * focal_length_px) / face_width)
            
            # Format distance label
            if distance < 100:
                distance_label = f"~{distance}cm"
            else:
                distance_m = distance / 100.0
                distance_label = f"~{distance_m:.1f}m"
                
            return (distance, distance_label)
            
        except Exception as e:
            logger.error(f"Error estimating distance with camera params: {e}")
            return self.estimate_from_face(face_width)
    
    def find_nearest_face(self, 
                         hat_center: Tuple[float, float], 
                         faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the nearest face to a hat.
        
        Args:
            hat_center: (x, y) center coordinates of the hat
            faces: List of detected faces
            
        Returns:
            Nearest face or None if no faces
        """
        if not faces:
            return None
            
        hat_x, hat_y = hat_center
        min_distance = float('inf')
        nearest_face = None
        
        for face in faces:
            try:
                if "topLeft" in face and "bottomRight" in face:
                    # BlazeFace format
                    face_center_x = (face["topLeft"][0] + face["bottomRight"][0]) / 2
                    face_center_y = (face["topLeft"][1] + face["bottomRight"][1]) / 2
                else:
                    # OpenCV format
                    fx, fy, fw, fh = face["bbox"]
                    face_center_x = fx + fw / 2
                    face_center_y = fy + fh / 2
                
                # Calculate squared distance (no need for sqrt since we're just comparing)
                dx = hat_x - face_center_x
                dy = hat_y - face_center_y
                dist_squared = dx*dx + dy*dy
                
                if dist_squared < min_distance:
                    min_distance = dist_squared
                    nearest_face = face
            except Exception as e:
                logger.error(f"Error finding nearest face: {e}")
                continue
        
        return nearest_face
    
    def get_face_width(self, face: Dict[str, Any]) -> float:
        """
        Get the width of a face from detection data.
        
        Args:
            face: Face detection data
            
        Returns:
            Face width in pixels
        """
        try:
            if "topLeft" in face and "bottomRight" in face:
                # BlazeFace format
                return face["bottomRight"][0] - face["topLeft"][0]
            else:
                # OpenCV format
                return face["bbox"][2]  # Width is the third element
        except Exception as e:
            logger.error(f"Error getting face width: {e}")
            return 0.0