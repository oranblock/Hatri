"""
Configuration settings for Hat Detection System
"""

# Camera settings
CAMERA_ID = 0  # Default camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Processing power options
class ProcessingPower:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Default processing level
DEFAULT_PROCESSING_POWER = ProcessingPower.MEDIUM

# Processing settings based on power level
PROCESSING_SETTINGS = {
    ProcessingPower.LOW: {
        "resolution": (320, 240),
        "skip_frames": 2,
        "max_faces": 1,
        "score_threshold": 0.7,
        "color_clusters": 2
    },
    ProcessingPower.MEDIUM: {
        "resolution": (640, 480),
        "skip_frames": 1,
        "max_faces": 3,
        "score_threshold": 0.5,
        "color_clusters": 3
    },
    ProcessingPower.HIGH: {
        "resolution": (1280, 720),
        "skip_frames": 0,
        "max_faces": 5,
        "score_threshold": 0.5,
        "color_clusters": 4
    }
}

# Detection settings
OBJECT_CONFIDENCE_THRESHOLD = 0.5
HAT_PROBABILITY_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

# Tracking settings
MAX_TRACKING_AGE = 2000  # ms before removing a tracked object
TRACKING_MATCH_THRESHOLD = 50  # pixels

# Model paths
COCO_SSD_MODEL_PATH = "models/hailo_models/coco_ssd"
BLAZEFACE_MODEL_PATH = "models/hailo_models/blazeface"

# Named colors with RGB values
NAMED_COLORS = [
    {"name": "red", "rgb": [255, 0, 0]},
    {"name": "green", "rgb": [0, 255, 0]},
    {"name": "blue", "rgb": [0, 0, 255]},
    {"name": "yellow", "rgb": [255, 255, 0]},
    {"name": "cyan", "rgb": [0, 255, 255]},
    {"name": "magenta", "rgb": [255, 0, 255]},
    {"name": "black", "rgb": [0, 0, 0]},
    {"name": "white", "rgb": [255, 255, 255]},
    {"name": "gray", "rgb": [128, 128, 128]},
    {"name": "orange", "rgb": [255, 165, 0]},
    {"name": "purple", "rgb": [128, 0, 128]},
    {"name": "brown", "rgb": [165, 42, 42]},
    {"name": "pink", "rgb": [255, 192, 203]}
]

# UI settings
WINDOW_NAME = "Hat Detection System"
SHOW_FACES = True
SHOW_TRAJECTORY = True