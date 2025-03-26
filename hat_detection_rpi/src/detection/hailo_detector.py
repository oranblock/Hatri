"""
Hailo-accelerated detector for object and face detection
"""

import logging
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional

# Replace with actual imports when Hailo SDK is available
# from hailo_platform import HailoInfer, HailoStreamInterface

from .. import config

logger = logging.getLogger(__name__)

class HailoDetector:
    """
    Detector class that uses Hailo-8 accelerator for inference.
    
    This class handles the initialization and inference for both
    object detection (COCO-SSD) and face detection (BlazeFace) models.
    """
    
    def __init__(self, processing_power: str = config.DEFAULT_PROCESSING_POWER):
        """
        Initialize the Hailo detector with the specified processing power.
        
        Args:
            processing_power: Low, medium, or high processing level
        """
        self.processing_power = processing_power
        self.settings = config.PROCESSING_SETTINGS[processing_power]
        
        logger.info(f"Initializing HailoDetector with {processing_power} processing power")
        
        # Model paths
        self.coco_model_path = config.COCO_SSD_MODEL_PATH
        self.face_model_path = config.BLAZEFACE_MODEL_PATH
        
        # Initialize Hailo runtime
        try:
            logger.info("Initializing Hailo runtime...")
            # In actual implementation, this would use real Hailo SDK
            # self.hailo_runtime = HailoStreamInterface()
            
            # Load object detection model
            logger.info(f"Loading object detection model from {self.coco_model_path}")
            # self.object_detector = HailoInfer(self.coco_model_path)
            
            # Load face detection model with parameters based on processing power
            face_params = {
                "max_faces": self.settings["max_faces"],
                "score_threshold": self.settings["score_threshold"]
            }
            logger.info(f"Loading face detection model from {self.face_model_path} with params {face_params}")
            # self.face_detector = HailoInfer(self.face_model_path, **face_params)
            
            # For simulation without actual Hailo hardware
            self._initialize_opencv_fallbacks()
            
            logger.info("Hailo detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Hailo detector: {e}")
            raise
    
    def _initialize_opencv_fallbacks(self):
        """
        Initialize OpenCV-based detectors as fallbacks when Hailo hardware is not available.
        This is for development and testing only.
        """
        # Load OpenCV DNN for object detection
        try:
            # Use MobileNet SSD as a fallback
            weights = "models/fallback/mobilenet_ssd.pb"
            config_file = "models/fallback/mobilenet_ssd.pbtxt"
            self.object_detector_fallback = cv2.dnn.readNetFromTensorflow(weights, config_file)
            
            # OpenCV Face detector - use Haar cascade for simplicity
            self.face_detector_fallback = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            logger.info("OpenCV fallback detectors initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV fallbacks: {e}")
            # Continue without fallbacks
            self.object_detector_fallback = None
            self.face_detector_fallback = None
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run detection on a frame to find objects and faces.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            
        Returns:
            Tuple of (object_detections, face_detections)
        """
        # Resize frame based on processing power settings
        target_size = self.settings["resolution"]
        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame = cv2.resize(frame, target_size)
        
        try:
            # In actual implementation with real Hailo hardware:
            # object_results = self.object_detector.infer(frame)
            # face_results = self.face_detector.infer(frame)
            
            # For development/simulation, use OpenCV fallbacks
            object_results = self._detect_objects_fallback(frame)
            face_results = self._detect_faces_fallback(frame)
            
            return object_results, face_results
        
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [], []
    
    def _detect_objects_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback object detection using OpenCV DNN when Hailo hardware is not available.
        """
        if self.object_detector_fallback is None:
            return []
        
        # COCO class names
        classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                  "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", 
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", 
                  "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
                  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
                  "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
                  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
                  "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", 
                  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", 
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        
        self.object_detector_fallback.setInput(blob)
        detections = self.object_detector_fallback.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > config.OBJECT_CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Format to be consistent with expected Hailo output
                results.append({
                    "class": classes[class_id],
                    "score": float(confidence),
                    "bbox": [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
                })
                
        return results
    
    def _detect_faces_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback face detection using OpenCV Haar cascade when Hailo hardware is not available.
        """
        if self.face_detector_fallback is None:
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector_fallback.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Format to be consistent with BlazeFace output style
            # BlazeFace returns landmarks too, but we'll omit those in the fallback
            results.append({
                "topLeft": [x, y],
                "bottomRight": [x + w, y + h],
                "landmarks": [],  # Empty in fallback
                "probability": 0.9  # Dummy value for Haar cascade
            })
            
        # Limit to max_faces based on processing power
        max_faces = self.settings["max_faces"]
        return results[:max_faces]
    
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
        
        # Update face detector parameters if using real Hailo
        # if hasattr(self, 'face_detector') and self.face_detector is not None:
        #    self.face_detector.set_params(max_faces=self.settings["max_faces"],
        #                                  score_threshold=self.settings["score_threshold"])
        
        logger.info(f"Processing power changed to {processing_power}")