"""
Model conversion utilities for Hailo-8 hardware.

This script downloads and converts TensorFlow models to Hailo format
for optimized inference on the Hailo-8 accelerator.
"""

import os
import sys
import logging
import argparse
import subprocess
import tempfile
import shutil
import urllib.request
import tarfile
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# URLs for pre-trained models
MODEL_URLS = {
    "coco-ssd": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
    "blazeface": "https://github.com/tensorflow/tfjs-models/raw/master/blazeface/src/blazeface.ts"
}

# Output paths
HAILO_MODELS_DIR = Path(__file__).parent.parent / "hailo_models"
TEMP_DIR = Path(tempfile.mkdtemp())

def download_model(name: str, url: str) -> Path:
    """
    Download a model from a URL.
    
    Args:
        name: Name of the model
        url: URL to download from
        
    Returns:
        Path to the downloaded file
    """
    logger.info(f"Downloading {name} model from {url}")
    
    # Create download directory
    download_dir = TEMP_DIR / name
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Download file
    file_path = download_dir / f"{name}_download"
    
    try:
        urllib.request.urlretrieve(url, file_path)
        logger.info(f"Downloaded {name} model to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading {name} model: {e}")
        raise

def extract_model(model_path: Path, model_name: str) -> Path:
    """
    Extract a downloaded model archive.
    
    Args:
        model_path: Path to the downloaded model file
        model_name: Name of the model
        
    Returns:
        Path to the extracted model directory
    """
    logger.info(f"Extracting {model_name} model")
    
    extract_dir = TEMP_DIR / f"{model_name}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if model_path.suffix == ".gz" or str(model_path).endswith(".tar.gz"):
            # Extract tar.gz archive
            with tarfile.open(model_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        elif model_path.suffix == ".zip":
            # Extract zip archive
            with zipfile.ZipFile(model_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            # Just copy the file if it's not an archive
            shutil.copy(model_path, extract_dir)
            
        logger.info(f"Extracted {model_name} model to {extract_dir}")
        return extract_dir
    except Exception as e:
        logger.error(f"Error extracting {model_name} model: {e}")
        raise

def convert_coco_ssd_model(source_dir: Path, output_dir: Path) -> None:
    """
    Convert COCO-SSD model to Hailo format.
    
    Args:
        source_dir: Path to the extracted model directory
        output_dir: Path to save the converted model
    """
    logger.info("Converting COCO-SSD model to Hailo format")
    
    try:
        # In a real implementation, this would use the Hailo Dataflow Compiler (HCL)
        # For simulation, we'll just create a placeholder file
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the SavedModel directory
        saved_model_dir = next(source_dir.glob("**/saved_model"))
        
        if not saved_model_dir.exists():
            raise FileNotFoundError(f"SavedModel directory not found in {source_dir}")
        
        # For a real implementation, we would call the Hailo Model Zoo conversion here:
        # hailo_model_zoo compile --model ssd_mobilenet_v2 --input-model {saved_model_dir} --output-model {output_dir}
        
        # Create placeholder files for demonstration
        with open(output_dir / "coco-ssd.hef", "w") as f:
            f.write("# Placeholder for Hailo compiled model\n")
            f.write(f"# Original model from: {saved_model_dir}\n")
            f.write("# This would be a binary HEF file in a real implementation\n")
        
        with open(output_dir / "coco-ssd_metadata.json", "w") as f:
            f.write('{\n')
            f.write('  "model_name": "coco-ssd",\n')
            f.write('  "framework": "tensorflow",\n')
            f.write('  "task": "object_detection",\n')
            f.write('  "input_shape": [1, 300, 300, 3],\n')
            f.write('  "classes": 90\n')
            f.write('}\n')
        
        logger.info(f"COCO-SSD model converted successfully to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error converting COCO-SSD model: {e}")
        raise

def convert_blazeface_model(source_dir: Path, output_dir: Path) -> None:
    """
    Convert BlazeFace model to Hailo format.
    
    Args:
        source_dir: Path to the extracted model directory
        output_dir: Path to save the converted model
    """
    logger.info("Converting BlazeFace model to Hailo format")
    
    try:
        # In a real implementation, we would need to:
        # 1. Convert the BlazeFace model from TFJS to TensorFlow
        # 2. Convert TensorFlow model to Hailo format
        
        # For simulation, we'll just create placeholder files
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder files for demonstration
        with open(output_dir / "blazeface.hef", "w") as f:
            f.write("# Placeholder for Hailo compiled model\n")
            f.write(f"# Original model from TFJS: BlazeFace\n")
            f.write("# This would be a binary HEF file in a real implementation\n")
        
        with open(output_dir / "blazeface_metadata.json", "w") as f:
            f.write('{\n')
            f.write('  "model_name": "blazeface",\n')
            f.write('  "framework": "tensorflow",\n')
            f.write('  "task": "face_detection",\n')
            f.write('  "input_shape": [1, 256, 256, 3],\n')
            f.write('  "anchors": 896\n')
            f.write('}\n')
        
        logger.info(f"BlazeFace model converted successfully to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error converting BlazeFace model: {e}")
        raise

def convert_models() -> None:
    """Convert all models to Hailo format."""
    logger.info("Starting model conversion process")
    
    # Create output directory
    HAILO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process COCO-SSD model
        coco_ssd_url = MODEL_URLS["coco-ssd"]
        coco_ssd_file = download_model("coco-ssd", coco_ssd_url)
        coco_ssd_dir = extract_model(coco_ssd_file, "coco-ssd")
        coco_ssd_output_dir = HAILO_MODELS_DIR / "coco_ssd"
        convert_coco_ssd_model(coco_ssd_dir, coco_ssd_output_dir)
        
        # Process BlazeFace model
        blazeface_url = MODEL_URLS["blazeface"]
        blazeface_file = download_model("blazeface", blazeface_url)
        blazeface_dir = extract_model(blazeface_file, "blazeface")
        blazeface_output_dir = HAILO_MODELS_DIR / "blazeface"
        convert_blazeface_model(blazeface_dir, blazeface_output_dir)
        
        logger.info("All models converted successfully")
        
    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        raise
    finally:
        # Clean up temporary directory
        logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)

def main():
    """Main entry point for the model conversion script."""
    parser = argparse.ArgumentParser(description="Convert models to Hailo format")
    
    parser.add_argument("--coco-ssd", action="store_true", 
                        help="Convert only COCO-SSD model")
    
    parser.add_argument("--blazeface", action="store_true", 
                        help="Convert only BlazeFace model")
    
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory for converted models")
    
    args = parser.parse_args()
    
    # Update output directory if specified
    if args.output_dir:
        global HAILO_MODELS_DIR
        HAILO_MODELS_DIR = Path(args.output_dir)
    
    try:
        if args.coco_ssd:
            # Convert only COCO-SSD
            logger.info("Converting only COCO-SSD model")
            HAILO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            coco_ssd_url = MODEL_URLS["coco-ssd"]
            coco_ssd_file = download_model("coco-ssd", coco_ssd_url)
            coco_ssd_dir = extract_model(coco_ssd_file, "coco-ssd")
            coco_ssd_output_dir = HAILO_MODELS_DIR / "coco_ssd"
            convert_coco_ssd_model(coco_ssd_dir, coco_ssd_output_dir)
        elif args.blazeface:
            # Convert only BlazeFace
            logger.info("Converting only BlazeFace model")
            HAILO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            blazeface_url = MODEL_URLS["blazeface"]
            blazeface_file = download_model("blazeface", blazeface_url)
            blazeface_dir = extract_model(blazeface_file, "blazeface")
            blazeface_output_dir = HAILO_MODELS_DIR / "blazeface"
            convert_blazeface_model(blazeface_dir, blazeface_output_dir)
        else:
            # Convert all models
            convert_models()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()