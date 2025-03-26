"""
Multi-object tracker for hat tracking across video frames.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from .. import config
from .trajectory import TrajectoryPredictor

logger = logging.getLogger(__name__)

class HatTracker:
    """
    Tracks multiple hats across video frames, maintaining their identities
    and tracking motion and trajectory.
    """
    
    def __init__(self):
        """
        Initialize the hat tracker.
        """
        # Dictionary of currently tracked hats
        self.tracked_hats = {}
        
        # For trajectory prediction
        self.trajectory_predictor = TrajectoryPredictor()
        
        logger.info("HatTracker initialized")
    
    def update(self, detections: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Update tracked hats with new detections.
        
        Args:
            detections: List of hat detections from the current frame
            
        Returns:
            Dictionary of tracked hats with their information
        """
        current_time = int(time.time() * 1000)
        current_frame_hats = {}
        
        try:
            # Process new detections
            for detection in detections:
                # Extract detection data
                hat_id = detection["id"]
                center_x = detection["center_x"]
                center_y = detection["center_y"]
                color = detection["color"]
                
                # Check if this hat was previously tracked
                existing_hat_key = self._find_matching_hat(detection)
                
                if existing_hat_key:
                    # Update existing hat with new information
                    existing_hat = self.tracked_hats[existing_hat_key]
                    
                    # Calculate velocities and update tracking info
                    velocity_x = center_x - existing_hat["center_x"]
                    velocity_y = center_y - existing_hat["center_y"]
                    
                    # Calculate distance change if applicable
                    distance_change = 0
                    if detection["distance"] is not None and existing_hat["distance"] is not None:
                        distance_change = detection["distance"] - existing_hat["distance"]
                    
                    # Update the hat's information
                    updated_hat = {
                        **existing_hat,  # Keep existing data
                        **detection,     # Update with new detection data
                        "velocity_x": velocity_x,
                        "velocity_y": velocity_y,
                        "distance_change": distance_change,
                        "frames_tracked": existing_hat["frames_tracked"] + 1,
                        "timestamp": current_time
                    }
                    
                    # Store with the existing key to maintain identity
                    current_frame_hats[existing_hat_key] = updated_hat
                    
                else:
                    # This is a new hat, add it to tracking
                    current_frame_hats[hat_id] = detection
            
            # Update trajectory predictions for hats that have been tracked for a while
            for hat_id, hat in current_frame_hats.items():
                if hat["frames_tracked"] > 5 and (abs(hat["velocity_x"]) > 1 or abs(hat["velocity_y"]) > 1):
                    # Predict future position
                    predicted_position = self.trajectory_predictor.predict_position(
                        hat["center_x"], 
                        hat["center_y"],
                        hat["velocity_x"],
                        hat["velocity_y"],
                        steps=10  # Predict 10 frames ahead
                    )
                    
                    hat["predicted_x"] = predicted_position[0]
                    hat["predicted_y"] = predicted_position[1]
            
            # Update the main tracking dictionary
            self.tracked_hats = current_frame_hats
            
            # Remove old hats (not seen for more than MAX_TRACKING_AGE)
            self._remove_expired_hats(current_time)
            
            return self.tracked_hats
            
        except Exception as e:
            logger.error(f"Error updating hat tracking: {e}")
            return self.tracked_hats
    
    def _find_matching_hat(self, detection: Dict[str, Any]) -> Optional[str]:
        """
        Find a previously tracked hat that matches this detection.
        
        Args:
            detection: Current hat detection
            
        Returns:
            Key of matching hat or None if no match found
        """
        for key, existing_hat in self.tracked_hats.items():
            # Calculate distance between centers
            distance = np.sqrt(
                (existing_hat["center_x"] - detection["center_x"]) ** 2 +
                (existing_hat["center_y"] - detection["center_y"]) ** 2
            )
            
            # Check if it's approximately the same hat based on position and color
            if (distance < config.TRACKING_MATCH_THRESHOLD and 
                existing_hat["color"] == detection["color"]):
                return key
        
        return None
    
    def _remove_expired_hats(self, current_time: int) -> None:
        """
        Remove hats that haven't been seen for too long.
        
        Args:
            current_time: Current timestamp in milliseconds
        """
        expired_keys = []
        
        for key, hat in self.tracked_hats.items():
            if current_time - hat["timestamp"] > config.MAX_TRACKING_AGE:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.tracked_hats[key]
    
    def get_movement_direction(self, hat_id: str) -> str:
        """
        Get human-readable movement direction for a tracked hat.
        
        Args:
            hat_id: ID of the hat to check
            
        Returns:
            Direction string (e.g., "Up", "Down-Right")
        """
        if hat_id not in self.tracked_hats:
            return "Unknown"
            
        hat = self.tracked_hats[hat_id]
        vx = hat.get("velocity_x", 0)
        vy = hat.get("velocity_y", 0)
        
        if abs(vx) < 1 and abs(vy) < 1:
            return "Stationary"
            
        angle = np.arctan2(vy, vx) * 180 / np.pi
        
        if -22.5 < angle <= 22.5:
            return "Right"
        elif 22.5 < angle <= 67.5:
            return "Down-Right"
        elif 67.5 < angle <= 112.5:
            return "Down"
        elif 112.5 < angle <= 157.5:
            return "Down-Left"
        elif 157.5 < angle or angle <= -157.5:
            return "Left"
        elif -157.5 < angle <= -112.5:
            return "Up-Left"
        elif -112.5 < angle <= -67.5:
            return "Up"
        elif -67.5 < angle <= -22.5:
            return "Up-Right"
            
        return "Unknown"
    
    def get_formatted_detections(self) -> List[Dict[str, str]]:
        """
        Get formatted hat detections for display.
        
        Returns:
            List of formatted hat detections
        """
        results = []
        
        for hat_id, hat in self.tracked_hats.items():
            # Determine if the hat is moving
            is_moving = abs(hat.get("velocity_x", 0)) > 1 or abs(hat.get("velocity_y", 0)) > 1
            
            results.append({
                "id": hat_id[:8],  # Truncate ID for display
                "type": "Hat",
                "color": hat["color"],
                "distance": hat.get("distance_label", "unknown"),
                "confidence": f"{hat.get('confidence', 0):.2f}",
                "frames_tracked": str(hat.get("frames_tracked", 1)),
                "moving": "Yes" if is_moving else "No",
                "direction": self.get_movement_direction(hat_id)
            })
            
        return results
        
    def clear(self) -> None:
        """Clear all tracked hats."""
        self.tracked_hats = {}