"""
Main detector module that combines object detection, face detection, and hat validation.
"""

import logging
import time
import cv2
import numpy as np
import uuid
from typing import Dict, List, Any, Tuple, Optional

from .hailo_detector import HailoDetector
from .color_analyzer import ColorAnalyzer
from .hat_validator import HatValidator
from .. import config

logger = logging.getLogger(__name__)

class HatDetector:
    """
    Main detector class that combines all detection components.
    """
    
    def __init__(self, processing_power: str = config.DEFAULT_PROCESSING_POWER):
        """
        Initialize the hat detector with all required components.
        
        Args:
            processing_power: Low, medium, or high processing level
        """
        self.processing_power = processing_power
        self.settings = config.PROCESSING_SETTINGS[processing_power]
        
        logger.info(f"Initializing HatDetector with {processing_power} processing power")
        
        # Initialize components
        try:
            self.hailo_detector = HailoDetector(processing_power)
            self.color_analyzer = ColorAnalyzer()
            self.hat_validator = HatValidator()
            
            # Performance stats
            self.frame_count = 0
            self.detection_count = 0
            self.performance_times = []
            
            logger.info("HatDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HatDetector: {e}")
            raise
    
    def detect_hats(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hats in a frame.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            
        Returns:
            List of hat detections with position, color, and confidence info
        """
        start_time = time.time()
        
        # Increment frame count for stats
        self.frame_count += 1
        
        # Check for frame skipping based on processing power
        if (self.processing_power == config.ProcessingPower.LOW and self.frame_count % 3 != 0) or \
           (self.processing_power == config.ProcessingPower.MEDIUM and self.frame_count % 2 != 0):
            # Skip processing for this frame
            return []
        
        try:
            # 1. Run object and face detection
            object_detections, face_detections = self.hailo_detector.detect(frame)
            
            # 2. Identify potential hats
            potential_hats = []
            for detection in object_detections:
                # Check if the object is likely a hat
                if self.hat_validator.is_likely_hat(detection, face_detections):
                    # Calculate hat confidence
                    hat_confidence = self.hat_validator.calculate_hat_confidence(detection, face_detections)
                    
                    # If confidence is above threshold, consider it a potential hat
                    if hat_confidence >= config.HAT_PROBABILITY_THRESHOLD:
                        x, y, width, height = detection["bbox"]
                        center_x = x + width / 2
                        center_y = y + height / 2
                        
                        # Generate an ID for this potential hat
                        hat_id = f"hat_{int(center_x)}_{int(center_y)}"
                        
                        potential_hats.append({
                            "id": hat_id,
                            "bbox": detection["bbox"],
                            "confidence": hat_confidence,
                            "class": detection["class"]
                        })
            
            # 3. Apply non-maximum suppression to remove overlapping detections
            filtered_hats = self.hat_validator.non_maximum_suppression(
                potential_hats, 
                overlap_threshold=config.NMS_THRESHOLD
            )
            
            # 4. Process filtered hats for color and distance
            hat_detections = []
            for hat in filtered_hats:
                try:
                    # Extract hat region for color analysis
                    x, y, width, height = hat["bbox"]
                    
                    # Ensure coordinates are within frame boundaries
                    x = max(0, int(x))
                    y = max(0, int(y))
                    width = min(int(width), frame.shape[1] - x)
                    height = min(int(height), frame.shape[0] - y)
                    
                    # Extract hat region for color analysis
                    hat_region = frame[y:y+height, x:x+width]
                    
                    # Skip if region is empty
                    if hat_region.size == 0:
                        continue
                    
                    # Sample from center of object for color analysis
                    color_x = int(x + width / 4)
                    color_y = int(y + height / 4)
                    color_width = int(width / 2)
                    color_height = int(height / 2)
                    
                    # Ensure coordinates are within frame boundaries
                    color_width = min(color_width, frame.shape[1] - color_x)
                    color_height = min(color_height, frame.shape[0] - color_y)
                    
                    # Extract region for color analysis
                    if color_width > 0 and color_height > 0:
                        color_region = frame[color_y:color_y+color_height, color_x:color_x+color_width]
                        # Analyze color
                        color_info = self.color_analyzer.detect_dominant_colors(
                            color_region, 
                            num_clusters=self.settings["color_clusters"]
                        )
                    else:
                        color_info = {"r": 128, "g": 128, "b": 128, "color_name": "gray"}
                    
                    # 5. Estimate distance based on nearest face
                    distance_value = None
                    distance_label = "unknown"
                    nearest_face_id = None
                    
                    # Find the nearest face to this hat
                    if face_detections:
                        center_x = x + width / 2
                        center_y = y + height / 2
                        
                        min_distance = float('inf')
                        nearest_face = None
                        
                        for face in face_detections:
                            if "topLeft" in face and "bottomRight" in face:
                                # BlazeFace format
                                face_center_x = (face["topLeft"][0] + face["bottomRight"][0]) / 2
                                face_center_y = (face["topLeft"][1] + face["bottomRight"][1]) / 2
                                face_width = face["bottomRight"][0] - face["topLeft"][0]
                            else:
                                # OpenCV format
                                fx, fy, fw, fh = face["bbox"]
                                face_center_x = fx + fw / 2
                                face_center_y = fy + fh / 2
                                face_width = fw
                            
                            # Calculate distance between hat center and face center
                            dx = center_x - face_center_x
                            dy = center_y - face_center_y
                            dist_squared = dx*dx + dy*dy
                            
                            if dist_squared < min_distance:
                                min_distance = dist_squared
                                nearest_face = face
                                if "topLeft" in face:
                                    face_width = face["bottomRight"][0] - face["topLeft"][0]
                                    nearest_face_id = f"face_{face['topLeft'][0]}_{face['topLeft'][1]}"
                                else:
                                    face_width = face["bbox"][2]
                                    nearest_face_id = f"face_{face['bbox'][0]}_{face['bbox'][1]}"
                        
                        if nearest_face is not None:
                            # Estimate distance based on face size
                            distance_value = self._estimate_distance(face_width)
                            distance_label = f"~{distance_value}cm"
                    
                    # 6. Generate unique ID based on position and color
                    center_x = x + width / 2
                    center_y = y + height / 2
                    hat_id = f"hat_{int(center_x)}_{int(center_y)}_{color_info['color_name']}"
                    
                    # 7. Create hat detection object
                    hat_detection = {
                        "id": hat_id,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "center_x": center_x,
                        "center_y": center_y,
                        "color": color_info["color_name"],
                        "color_rgb": {
                            "r": color_info["r"],
                            "g": color_info["g"],
                            "b": color_info["b"]
                        },
                        "distance": distance_value,
                        "distance_label": distance_label,
                        "confidence": hat["confidence"],
                        "timestamp": int(time.time() * 1000),
                        "nearest_face_id": nearest_face_id,
                        "velocity_x": 0,
                        "velocity_y": 0,
                        "distance_change": 0,
                        "frames_tracked": 1
                    }
                    
                    hat_detections.append(hat_detection)
                    
                except Exception as e:
                    logger.error(f"Error processing hat detection: {e}")
                    continue
            
            # Update detection count for stats
            self.detection_count += len(hat_detections)
            
            # Calculate frame processing time
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # in ms
            self.performance_times.append(processing_time)
            
            # Keep only last 30 measurements
            if len(self.performance_times) > 30:
                self.performance_times.pop(0)
            
            return hat_detections
            
        except Exception as e:
            logger.error(f"Error in detect_hats: {e}")
            return []
    
    def _estimate_distance(self, face_width: float) -> int:
        """
        Estimate distance based on face size.
        
        Args:
            face_width: Width of face in pixels
            
        Returns:
            Estimated distance in cm
        """
        max_face_size = 300
        scale_factor = 100
        return int((scale_factor * max_face_size) / face_width)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance stats
        """
        avg_time = 0
        fps = 0
        
        if self.performance_times:
            avg_time = sum(self.performance_times) / len(self.performance_times)
            fps = 1000 / avg_time if avg_time > 0 else 0
        
        return {
            "frames_processed": self.frame_count,
            "detections_total": self.detection_count,
            "avg_processing_time_ms": avg_time,
            "fps": fps,
            "power_level": self.processing_power
        }
    
    def set_processing_power(self, processing_power: str) -> None:
        """
        Change the processing power setting.
        
        Args:
            processing_power: New processing power level
        """
        if processing_power not in config.PROCESSING_SETTINGS:
            logger.warning(f"Invalid processing power: {processing_power}. Using default.")
            processing_power = config.DEFAULT_PROCESSING_POWER
            
        self.processing_power = processing_power
        self.settings = config.PROCESSING_SETTINGS[processing_power]
        
        # Update Hailo detector
        self.hailo_detector.set_processing_power(processing_power)
        
        logger.info(f"Processing power changed to {processing_power}")