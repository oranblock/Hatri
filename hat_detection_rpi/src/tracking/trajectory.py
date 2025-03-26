"""
Trajectory prediction module for estimating future hat positions.
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class TrajectoryPredictor:
    """
    Predicts future positions of hats based on their velocity.
    """
    
    def __init__(self, smoothing_factor: float = 0.7):
        """
        Initialize the trajectory predictor.
        
        Args:
            smoothing_factor: Weight for current velocity vs historical data (0-1)
        """
        self.smoothing_factor = smoothing_factor
        self.velocity_history = {}  # Stores velocity histories for each hat
        
        logger.info(f"TrajectoryPredictor initialized with smoothing factor {smoothing_factor}")
    
    def predict_position(self, 
                         x: float, 
                         y: float, 
                         vx: float, 
                         vy: float, 
                         steps: int = 10, 
                         hat_id: Optional[str] = None) -> Tuple[float, float]:
        """
        Predict future position based on current position and velocity.
        
        Args:
            x: Current x position
            y: Current y position
            vx: Velocity in x direction
            vy: Velocity in y direction
            steps: Number of steps to predict ahead
            hat_id: Optional ID for tracking velocity history
            
        Returns:
            Tuple of (predicted_x, predicted_y)
        """
        try:
            # Apply simple linear prediction by default
            if hat_id is None:
                return (x + vx * steps, y + vy * steps)
            
            # Use smoothed velocity if we're tracking history
            smoothed_vx, smoothed_vy = self._get_smoothed_velocity(hat_id, vx, vy)
            
            # Calculate predicted position using smoothed velocity
            predicted_x = x + smoothed_vx * steps
            predicted_y = y + smoothed_vy * steps
            
            return (predicted_x, predicted_y)
            
        except Exception as e:
            logger.error(f"Error predicting trajectory: {e}")
            # Fall back to current position
            return (x, y)
    
    def _get_smoothed_velocity(self, hat_id: str, vx: float, vy: float) -> Tuple[float, float]:
        """
        Get smoothed velocity using historical data.
        
        Args:
            hat_id: Hat identifier
            vx: Current velocity in x direction
            vy: Current velocity in y direction
            
        Returns:
            Tuple of (smoothed_vx, smoothed_vy)
        """
        # Initialize history for this hat if it doesn't exist
        if hat_id not in self.velocity_history:
            self.velocity_history[hat_id] = {
                "vx": vx,
                "vy": vy,
                "samples": 1
            }
            return (vx, vy)
        
        # Get history
        history = self.velocity_history[hat_id]
        
        # Apply exponential smoothing
        smoothed_vx = vx * self.smoothing_factor + history["vx"] * (1 - self.smoothing_factor)
        smoothed_vy = vy * self.smoothing_factor + history["vy"] * (1 - self.smoothing_factor)
        
        # Update history
        history["vx"] = smoothed_vx
        history["vy"] = smoothed_vy
        history["samples"] += 1
        
        return (smoothed_vx, smoothed_vy)
    
    def clear_history(self, hat_id: Optional[str] = None) -> None:
        """
        Clear velocity history for a hat or all hats.
        
        Args:
            hat_id: Hat ID to clear, or None to clear all
        """
        if hat_id is None:
            self.velocity_history = {}
        elif hat_id in self.velocity_history:
            del self.velocity_history[hat_id]
    
    def predict_multiple(self, 
                         hats: Dict[str, Dict[str, Any]], 
                         steps: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Predict trajectories for multiple hats.
        
        Args:
            hats: Dictionary of tracked hats
            steps: Number of steps to predict ahead
            
        Returns:
            Dictionary of hats with predicted positions added
        """
        predictions = {}
        
        for hat_id, hat in hats.items():
            if hat.get("frames_tracked", 0) > 5 and (abs(hat.get("velocity_x", 0)) > 1 or abs(hat.get("velocity_y", 0)) > 1):
                x = hat.get("center_x", 0)
                y = hat.get("center_y", 0)
                vx = hat.get("velocity_x", 0)
                vy = hat.get("velocity_y", 0)
                
                predicted_x, predicted_y = self.predict_position(x, y, vx, vy, steps, hat_id)
                
                # Create a copy with predictions added
                hat_copy = hat.copy()
                hat_copy["predicted_x"] = predicted_x
                hat_copy["predicted_y"] = predicted_y
                
                predictions[hat_id] = hat_copy
            else:
                # Just copy the hat without predictions
                predictions[hat_id] = hat.copy()
        
        return predictions