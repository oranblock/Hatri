"""
Hat validation module to determine if detected objects are likely hats.
Ported from the original TypeScript implementation in HatDetector.tsx.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple

from .. import config

logger = logging.getLogger(__name__)

class HatValidator:
    """
    Validates if detected objects are likely hats based on 
    their position, shape, and relationship to detected faces.
    """
    
    def __init__(self):
        """Initialize the hat validator"""
        logger.info("HatValidator initialized")
    
    def is_likely_hat(self, prediction: Dict[str, Any], face_detections: List[Dict[str, Any]]) -> bool:
        """
        Check if the detected object is likely a hat based on position and shape.
        
        Args:
            prediction: Object detection result
            face_detections: List of detected faces
            
        Returns:
            Boolean indicating if the object is likely a hat
        """
        # These are common classes that may represent hats or head items
        possible_hat_classes = ['hat', 'cap', 'helmet', 'sports ball', 'frisbee', 'bowl']
        
        # If the prediction includes "hat" or similar in its class name, consider it likely
        if any(cls in prediction["class"].lower() for cls in possible_hat_classes):
            return True
        
        # For other objects, check if they're positioned above a face
        for face in face_detections:
            try:
                # Extract face dimensions
                if "topLeft" in face and "bottomRight" in face:
                    # BlazeFace format
                    face_top = face["topLeft"][1]
                    face_left = face["topLeft"][0]
                    face_right = face["bottomRight"][0]
                    face_bottom = face["bottomRight"][1]
                else:
                    # OpenCV format
                    x, y, w, h = face["bbox"]
                    face_top = y
                    face_left = x
                    face_right = x + w
                    face_bottom = y + h
                
                face_width = face_right - face_left
                face_center_x = (face_left + face_right) / 2
                
                # Extract object dimensions
                obj_x, obj_y, obj_width, obj_height = prediction["bbox"]
                obj_bottom = obj_y + obj_height
                obj_center_x = obj_x + obj_width / 2
                
                # Check if object is above and roughly centered with a face
                is_above_face = obj_bottom >= face_top - 30 and obj_bottom <= face_top + 40
                is_near_face_center_x = abs(obj_center_x - face_center_x) < face_width * 0.7
                has_reasonable_size = obj_width >= face_width * 0.4 and obj_width <= face_width * 2.2
                has_reasonable_shape = obj_width / obj_height >= 1.0  # Hats tend to be wider than tall
                
                if is_above_face and is_near_face_center_x and has_reasonable_size and has_reasonable_shape:
                    return True
                    
            except Exception as e:
                logger.error(f"Error checking hat likelihood: {e}")
                continue
        
        return False
    
    def calculate_hat_confidence(self, prediction: Dict[str, Any], face_detections: List[Dict[str, Any]]) -> float:
        """
        Calculate hat-like confidence score for an object.
        
        Args:
            prediction: Object detection result
            face_detections: List of detected faces
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # These are common classes that may represent hats or head items
        direct_hat_classes = ['hat', 'cap', 'helmet']
        secondary_hat_classes = ['sports ball', 'frisbee', 'bowl']
        
        # Start with the model's confidence
        confidence = prediction["score"]
        
        # Boost confidence for hat-related classes
        if any(cls in prediction["class"].lower() for cls in direct_hat_classes):
            confidence = min(1.0, confidence * 1.2)
            return confidence
        
        if any(cls in prediction["class"].lower() for cls in secondary_hat_classes):
            confidence = min(1.0, confidence * 1.1)
        
        # For other objects, assess positional relationship with faces
        for face in face_detections:
            try:
                # Extract face dimensions
                if "topLeft" in face and "bottomRight" in face:
                    # BlazeFace format
                    face_top = face["topLeft"][1]
                    face_left = face["topLeft"][0]
                    face_right = face["bottomRight"][0]
                    face_bottom = face["bottomRight"][1]
                else:
                    # OpenCV format
                    x, y, w, h = face["bbox"]
                    face_top = y
                    face_left = x
                    face_right = x + w
                    face_bottom = y + h
                
                face_width = face_right - face_left
                face_center_x = (face_left + face_right) / 2
                
                # Extract object dimensions
                obj_x, obj_y, obj_width, obj_height = prediction["bbox"]
                obj_bottom = obj_y + obj_height
                obj_center_x = obj_x + obj_width / 2
                
                # Assess position metrics and adjust confidence accordingly
                vertical_position = abs(obj_bottom - face_top) / face_width
                horizontal_alignment = abs(obj_center_x - face_center_x) / face_width
                size_ratio = obj_width / face_width
                aspect_ratio = obj_width / obj_height
                
                # Ideal hat is directly above face, centered, with reasonable size and aspect ratio
                positional_confidence = 0.5
                
                # Vertical position score - highest when just above face
                if vertical_position < 0.2: 
                    positional_confidence += 0.2
                elif vertical_position < 0.4: 
                    positional_confidence += 0.1
                else: 
                    positional_confidence -= 0.1
                
                # Horizontal alignment score - highest when centered above face
                if horizontal_alignment < 0.2: 
                    positional_confidence += 0.2
                elif horizontal_alignment < 0.5: 
                    positional_confidence += 0.1
                else: 
                    positional_confidence -= 0.1
                
                # Size ratio score - highest when hat width is similar to face width
                if 0.7 < size_ratio < 1.5: 
                    positional_confidence += 0.2
                elif 0.5 < size_ratio < 2.0: 
                    positional_confidence += 0.1
                else: 
                    positional_confidence -= 0.1
                
                # Aspect ratio score - hats tend to be wider than tall
                if 1.2 < aspect_ratio < 3.0: 
                    positional_confidence += 0.1
                
                # Combine original and positional confidence
                confidence = (confidence + positional_confidence) / 2
                break  # Only use the best face for this calculation
                
            except Exception as e:
                logger.error(f"Error calculating hat confidence: {e}")
                continue
        
        return max(0, min(1, confidence))
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            bbox1: First bounding box [x, y, width, height]
            bbox2: Second bounding box [x, y, width, height]
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        # Extract coordinates
        x1, y1, width1, height1 = bbox1
        x2, y2, width2, height2 = bbox2
        
        # Calculate areas
        area1 = width1 * height1
        area2 = width2 * height2
        
        # Calculate intersection coordinates
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + width1, x2 + width2)
        yy2 = min(y1 + height1, y2 + height2)
        
        # Check if there is an overlap
        if xx2 < xx1 or yy2 < yy1:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (xx2 - xx1) * (yy2 - yy1)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou
    
    def non_maximum_suppression(self, 
                                detections: List[Dict[str, Any]], 
                                overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply non-maximum suppression to remove duplicate detections.
        
        Args:
            detections: List of detection results with bbox and confidence
            overlap_threshold: IoU threshold for considering duplicates
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
            
        # Sort detections by confidence score (highest first)
        sorted_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        selected_detections = []
        
        while len(sorted_detections) > 0:
            # Select the detection with highest confidence
            current_detection = sorted_detections[0]
            selected_detections.append(current_detection)
            
            # Remove the current detection from the list
            sorted_detections.pop(0)
            
            # Filter remaining detections to remove overlapping ones
            remaining_detections = []
            
            for detection in sorted_detections:
                # Calculate overlap between current detection and this one
                overlap = self.calculate_iou(current_detection["bbox"], detection["bbox"])
                
                # Keep it only if it doesn't overlap too much
                if overlap < overlap_threshold:
                    remaining_detections.append(detection)
            
            # Update the list with non-overlapping detections
            sorted_detections = remaining_detections
        
        return selected_detections