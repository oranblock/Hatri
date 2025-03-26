"""
Color analysis module implementing k-means clustering for hat color detection.
Ported from the original TypeScript implementation in HatDetector.tsx.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Union

from .. import config

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    """
    Analyzes image regions to detect dominant colors using k-means clustering.
    """
    
    def __init__(self, named_colors: List[Dict[str, Any]] = None):
        """
        Initialize the color analyzer with a list of named colors.
        
        Args:
            named_colors: List of color definitions with name and RGB values
        """
        self.named_colors = named_colors if named_colors is not None else config.NAMED_COLORS
        logger.info(f"ColorAnalyzer initialized with {len(self.named_colors)} named colors")
    
    def detect_dominant_colors(self, image_data: np.ndarray, num_clusters: int = 3) -> Dict[str, Any]:
        """
        Detect dominant colors in an image region using k-means clustering.
        
        Args:
            image_data: Region of the image to analyze (OpenCV BGR format)
            num_clusters: Number of color clusters to identify
            
        Returns:
            Dictionary with dominant color info (r, g, b values and color name)
        """
        try:
            # Sample pixels (for performance optimization)
            pixels = self._sample_pixels(image_data)
            
            if len(pixels) < num_clusters:
                # Not enough pixels, reduce clusters
                num_clusters = max(1, len(pixels) // 2)
                logger.debug(f"Not enough pixels, reducing clusters to {num_clusters}")
            
            # Apply k-means clustering
            color_clusters = self._k_means_clustering(pixels, num_clusters)
            
            # Find the most saturated/colorful cluster (likely to be the hat color)
            dominant_cluster = color_clusters[0]
            highest_saturation = self._color_saturation(dominant_cluster["centroid"])
            
            for cluster in color_clusters[1:]:
                saturation = self._color_saturation(cluster["centroid"])
                if saturation > highest_saturation:
                    highest_saturation = saturation
                    dominant_cluster = cluster
            
            # Get RGB values (note: OpenCV uses BGR)
            b, g, r = map(int, dominant_cluster["centroid"])
            
            # Match to named color
            color_name = self._find_nearest_named_color(r, g, b)
            
            return {
                "r": r,
                "g": g,
                "b": b,
                "color_name": color_name
            }
            
        except Exception as e:
            logger.error(f"Error in detect_dominant_colors: {e}")
            # Return a default color in case of error
            return {"r": 128, "g": 128, "b": 128, "color_name": "gray"}
    
    def _sample_pixels(self, image_data: np.ndarray) -> np.ndarray:
        """
        Sample pixels from the image data for performance optimization.
        
        Args:
            image_data: Image data in BGR format
            
        Returns:
            NumPy array of pixel values
        """
        # Check if image is valid
        if image_data is None or image_data.size == 0:
            return np.array([[128, 128, 128]])  # Default to gray
        
        # Reshape image for easier processing
        height, width, channels = image_data.shape
        
        # Downsample by taking every 4th pixel for performance
        # Convert to 1D array of 3D pixels
        pixels = image_data[::4, ::4].reshape(-1, 3).astype(np.float32)
        
        # Ensure we have some pixels
        if len(pixels) == 0:
            return np.array([[128, 128, 128]])  # Default to gray
        
        return pixels
    
    def _k_means_clustering(self, points: np.ndarray, k: int, max_iterations: int = 10) -> List[Dict[str, Any]]:
        """
        Perform k-means clustering on pixel data.
        
        Args:
            points: Array of pixel values
            k: Number of clusters
            max_iterations: Maximum number of iterations
            
        Returns:
            List of cluster information
        """
        # Use OpenCV's built-in k-means function for efficiency
        # Termination criteria: 10 iterations or epsilon = 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1.0)
        
        # Apply k-means
        _, labels, centers = cv2.kmeans(
            points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Group points by cluster
        clusters = []
        for i in range(k):
            # Get points in this cluster
            cluster_points = points[labels.flatten() == i]
            
            clusters.append({
                "centroid": centers[i],
                "points": cluster_points,
                "size": len(cluster_points)
            })
        
        return clusters
    
    def _color_saturation(self, rgb: np.ndarray) -> float:
        """
        Calculate color saturation (higher for more vibrant colors).
        
        Args:
            rgb: RGB color values
            
        Returns:
            Saturation value (0.0 to 1.0)
        """
        # Convert BGR to RGB if necessary
        r, g, b = rgb if len(rgb) == 3 else (rgb[2], rgb[1], rgb[0])
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # Avoid division by zero
        if max_val == 0:
            return 0.0
        
        return (max_val - min_val) / max_val
    
    def _euclidean_distance(self, p1: Union[List[float], np.ndarray], p2: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Euclidean distance between two RGB points.
        
        Args:
            p1: First RGB point
            p2: Second RGB point
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(
            np.sum(np.square(np.array(p1) - np.array(p2)))
        )
    
    def _find_nearest_named_color(self, r: int, g: int, b: int) -> str:
        """
        Map RGB to named color based on Euclidean distance.
        
        Args:
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
            
        Returns:
            Name of closest color
        """
        min_distance = float('inf')
        closest_color = "unknown"
        
        for color in self.named_colors:
            # Compare RGB values
            distance = self._euclidean_distance([r, g, b], color["rgb"])
            if distance < min_distance:
                min_distance = distance
                closest_color = color["name"]
        
        return closest_color