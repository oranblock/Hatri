"""
Performance monitoring utilities for the hat detection system.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitors and reports performance metrics for the detection system.
    """
    
    def __init__(self, max_samples: int = 30):
        """
        Initialize the performance monitor.
        
        Args:
            max_samples: Maximum number of samples to keep for statistics
        """
        self.max_samples = max_samples
        
        # Performance metrics
        self.frames_processed = 0
        self.detections_count = 0
        self.processing_times = deque(maxlen=max_samples)
        
        # Timing variables
        self.frame_start_time = 0
        self.last_log_time = time.time()
        self.log_interval = 10  # Log performance every 10 seconds
        
        logger.info(f"PerformanceMonitor initialized with max_samples={max_samples}")
    
    def start_frame(self) -> None:
        """Mark the start of frame processing."""
        self.frame_start_time = time.time()
    
    def end_frame(self, detections_count: int = 0) -> float:
        """
        Mark the end of frame processing and update metrics.
        
        Args:
            detections_count: Number of detections in this frame
            
        Returns:
            Processing time for this frame in milliseconds
        """
        end_time = time.time()
        processing_time = (end_time - self.frame_start_time) * 1000  # Convert to ms
        
        # Update metrics
        self.frames_processed += 1
        self.detections_count += detections_count
        self.processing_times.append(processing_time)
        
        # Log performance metrics at intervals
        if end_time - self.last_log_time > self.log_interval:
            self._log_performance()
            self.last_log_time = end_time
        
        return processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate FPS from processing times
        fps = 0
        avg_processing_time = 0
        
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_count,
            "avg_detections_per_frame": self.detections_count / max(1, self.frames_processed),
            "avg_processing_time_ms": avg_processing_time,
            "min_processing_time_ms": min(self.processing_times) if self.processing_times else 0,
            "max_processing_time_ms": max(self.processing_times) if self.processing_times else 0,
            "fps": fps
        }
    
    def reset(self) -> None:
        """Reset all performance metrics."""
        self.frames_processed = 0
        self.detections_count = 0
        self.processing_times.clear()
        self.last_log_time = time.time()
    
    def _log_performance(self) -> None:
        """Log performance metrics."""
        stats = self.get_stats()
        logger.info(
            f"Performance: {stats['fps']:.1f} FPS, "
            f"{stats['avg_processing_time_ms']:.1f}ms/frame, "
            f"{stats['avg_detections_per_frame']:.1f} detections/frame"
        )

class FrameRateController:
    """
    Controls frame rate for consistent performance.
    """
    
    def __init__(self, target_fps: int = 30):
        """
        Initialize the frame rate controller.
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_frame_time = time.time()
        
        logger.info(f"FrameRateController initialized with target_fps={target_fps}")
    
    def wait(self) -> float:
        """
        Wait to maintain target frame rate.
        
        Returns:
            Actual FPS achieved
        """
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Calculate time to wait
        wait_time = max(0, self.target_frame_time - elapsed)
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Update frame time after waiting
        actual_frame_time = time.time() - self.last_frame_time
        self.last_frame_time = time.time()
        
        # Calculate actual FPS
        actual_fps = 1.0 / actual_frame_time if actual_frame_time > 0 else 0
        
        return actual_fps