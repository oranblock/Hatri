import React, { useEffect, useRef, useState, useCallback } from 'react';
// Use the tensorflow context instead of direct imports
import { useTensorflow } from './TensorflowContext';

// Import only the icons we need directly
import Activity from 'lucide-react/dist/esm/icons/activity';
import Camera from 'lucide-react/dist/esm/icons/camera';
import Zap from 'lucide-react/dist/esm/icons/zap';
import Maximize from 'lucide-react/dist/esm/icons/maximize';
import Minimize from 'lucide-react/dist/esm/icons/minimize';
import CameraOff from 'lucide-react/dist/esm/icons/camera-off';
import FlipHorizontal from 'lucide-react/dist/esm/icons/flip-horizontal';
import Settings from 'lucide-react/dist/esm/icons/settings';
import Cpu from 'lucide-react/dist/esm/icons/cpu';
import type { TrackedHat, HatDetection, ColorInfo } from '../types/hat';

// Processing power levels
type ProcessingPower = 'low' | 'medium' | 'high';

// Camera facing mode
type CameraFacing = 'user' | 'environment';

// Named color definitions with RGB values
const namedColors = [
  { name: 'red', rgb: [255, 0, 0] },
  { name: 'green', rgb: [0, 255, 0] },
  { name: 'blue', rgb: [0, 0, 255] },
  { name: 'yellow', rgb: [255, 255, 0] },
  { name: 'cyan', rgb: [0, 255, 255] },
  { name: 'magenta', rgb: [255, 0, 255] },
  { name: 'black', rgb: [0, 0, 0] },
  { name: 'white', rgb: [255, 255, 255] },
  { name: 'gray', rgb: [128, 128, 128] },
  { name: 'orange', rgb: [255, 165, 0] },
  { name: 'purple', rgb: [128, 0, 128] },
  { name: 'brown', rgb: [165, 42, 42] },
  { name: 'pink', rgb: [255, 192, 203] }
];

// Calculate Euclidean distance between two RGB points
function euclideanDistance(p1: number[], p2: number[]): number {
  return Math.sqrt(
    Math.pow(p2[0] - p1[0], 2) +
    Math.pow(p2[1] - p1[1], 2) +
    Math.pow(p2[2] - p1[2], 2)
  );
}

// Map RGB to named color
function findNearestNamedColor(r: number, g: number, b: number): string {
  let minDistance = Infinity;
  let closestColor = 'unknown';
  
  for (const color of namedColors) {
    const distance = euclideanDistance([r, g, b], color.rgb);
    if (distance < minDistance) {
      minDistance = distance;
      closestColor = color.name;
    }
  }
  
  return closestColor;
}

// Calculate color saturation (higher for more vibrant colors)
function colorSaturation(rgb: number[]): number {
  const [r, g, b] = rgb;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  
  // Avoid division by zero
  if (max === 0) return 0;
  
  return (max - min) / max;
}

// K-means clustering algorithm for color quantization
function kMeansClustering(points: number[][], k: number, maxIterations = 10): {
  centroid: number[],
  points: number[][]
}[] {
  // Initialize centroids randomly from the data points
  const centroids = points.length >= k 
    ? points.sort(() => 0.5 - Math.random()).slice(0, k) 
    : Array(k).fill(0).map(() => [
        Math.random() * 255,
        Math.random() * 255,
        Math.random() * 255
      ]);
    
  const clusters = Array(k).fill(null).map(() => ({ 
    centroid: [0, 0, 0], 
    points: [] as number[][] 
  }));
  
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // Reset clusters
    clusters.forEach(cluster => {
      cluster.points = [];
    });
    
    // Assign points to nearest centroid
    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      let minDistance = Infinity;
      let closestClusterIndex = 0;
      
      for (let j = 0; j < centroids.length; j++) {
        const distance = euclideanDistance(point, centroids[j]);
        if (distance < minDistance) {
          minDistance = distance;
          closestClusterIndex = j;
        }
      }
      
      clusters[closestClusterIndex].points.push(point);
    }
    
    // Calculate new centroids
    let changed = false;
    
    for (let i = 0; i < k; i++) {
      if (clusters[i].points.length === 0) continue;
      
      const newCentroid = [0, 0, 0];
      
      for (let j = 0; j < clusters[i].points.length; j++) {
        newCentroid[0] += clusters[i].points[j][0];
        newCentroid[1] += clusters[i].points[j][1];
        newCentroid[2] += clusters[i].points[j][2];
      }
      
      newCentroid[0] /= clusters[i].points.length;
      newCentroid[1] /= clusters[i].points.length;
      newCentroid[2] /= clusters[i].points.length;
      
      if (euclideanDistance(newCentroid, centroids[i]) > 1) {
        changed = true;
      }
      
      centroids[i] = newCentroid;
      clusters[i].centroid = newCentroid;
    }
    
    // If centroids didn't change significantly, we're done
    if (!changed) break;
  }
  
  return clusters;
}

// Get human-readable movement direction
function getMovementDirection(vx: number, vy: number): string {
  if (Math.abs(vx) < 1 && Math.abs(vy) < 1) return 'Stationary';
  
  const angle = Math.atan2(vy, vx) * 180 / Math.PI;
  
  if (angle > -22.5 && angle <= 22.5) return 'Right';
  if (angle > 22.5 && angle <= 67.5) return 'Down-Right';
  if (angle > 67.5 && angle <= 112.5) return 'Down';
  if (angle > 112.5 && angle <= 157.5) return 'Down-Left';
  if (angle > 157.5 || angle <= -157.5) return 'Left';
  if (angle > -157.5 && angle <= -112.5) return 'Up-Left';
  if (angle > -112.5 && angle <= -67.5) return 'Up';
  if (angle > -67.5 && angle <= -22.5) return 'Up-Right';
  
  return 'Unknown';
}

// Calculate Intersection over Union for two bounding boxes
function calculateIoU(bbox1: number[], bbox2: number[]): number {
  const [x1, y1, width1, height1] = bbox1;
  const [x2, y2, width2, height2] = bbox2;
  
  const xOverlap = Math.max(0, Math.min(x1 + width1, x2 + width2) - Math.max(x1, x2));
  const yOverlap = Math.max(0, Math.min(y1 + height1, y2 + height2) - Math.max(y1, y2));
  const intersectionArea = xOverlap * yOverlap;
  
  const bbox1Area = width1 * height1;
  const bbox2Area = width2 * height2;
  const unionArea = bbox1Area + bbox2Area - intersectionArea;
  
  return intersectionArea / unionArea;
}

const HatDetector: React.FC = () => {
  // Get TensorFlow.js and models from context
  const { tf, cocossd, blazeface } = useTensorflow();
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState('Loading models...');
  const [detections, setDetections] = useState<HatDetection[]>([]);
  const [trackedHats, setTrackedHats] = useState<Record<string, TrackedHat>>({});
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>('user');
  const [processingPower, setProcessingPower] = useState<ProcessingPower>('medium');
  const [cameraActive, setCameraActive] = useState(true);
  const streamRef = useRef<MediaStream | null>(null);
  const trackedHatsRef = useRef<Record<string, TrackedHat>>({});
  const statsRef = useRef({
    frames: 0,
    detections: 0,
    performance: [] as number[]
  });

  // Debugging data
  const [debugInfo, setDebugInfo] = useState<string[]>([]);
  const [errorInfo, setErrorInfo] = useState<string[]>([]);
  
  // Debug logging
  const debug = {
    log: (message: string, data?: any) => {
      console.log(`[HatDetector] ${message}`, data !== undefined ? data : '');
      setStatus(`[Debug] ${message}`);
      setDebugInfo(prev => [...prev, `${message}${data ? ': ' + JSON.stringify(data) : ''}`]);
    },
    error: (message: string, error?: any) => {
      const errorMsg = error instanceof Error ? error.message : (typeof error === 'string' ? error : 'Unknown error');
      const fullError = error instanceof Error ? error.stack || error.message : (typeof error === 'string' ? error : 'Unknown error');
      console.error(`[HatDetector] ${message}`, error);
      setStatus(`[Error] ${message}: ${errorMsg}`);
      setErrorInfo(prev => [...prev, `${message}: ${fullError}`]);
    }
  };

  // Define helper functions outside of useEffect using useCallback
  const [objectDetector, setObjectDetector] = useState<any>(null);
  const [faceDetector, setFaceDetector] = useState<any>(null);
  const [animationId, setAnimationId] = useState<number | null>(null);
  
  const setupCamera = useCallback(async (facing: CameraFacing = 'user'): Promise<boolean> => {
    setStatus('Setting up camera...');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('Camera access not supported by your browser');
      return false;
    }

    // Stop any existing stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    try {
      // Set resolution based on processing power
      let width, height;
      switch (processingPower) {
        case 'low':
          width = { ideal: 320 };
          height = { ideal: 240 };
          break;
        case 'high':
          width = { ideal: 1280 };
          height = { ideal: 720 };
          break;
        case 'medium':
        default:
          width = { ideal: 640 };
          height = { ideal: 480 };
          break;
      }

      // Request camera access with selected facing mode
      const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: facing,
            width,
            height
          }
        });
        
        // Store the stream for later access
        streamRef.current = stream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          return new Promise<boolean>((resolve) => {
            if (videoRef.current) {
              videoRef.current.onloadedmetadata = () => {
                // Only play if camera is active
                if (cameraActive) {
                  videoRef.current?.play();
                }
                resolve(true);
              };
            } else {
              resolve(false);
            }
          });
        }
        return false;
      } catch (error) {
        setStatus(`Camera error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return false;
      }
    }, [cameraActive, processingPower, setStatus, streamRef, videoRef]);

  const loadModels = useCallback(async (): Promise<boolean> => {
    try {
      debug.log('Setting up TensorFlow.js backend...');
      setStatus('Setting up TensorFlow.js backend...');
      
      // Check if TensorFlow.js is properly loaded
      if (!tf) {
        debug.error('TensorFlow.js is not loaded properly');
        return false;
        }
        
        try {
          debug.log('Checking TensorFlow.js version');
          debug.log('TF.js version: ' + tf.version.tfjs);
        } catch (verError) {
          debug.error('Error checking TF.js version', verError);
        }
        
        try {
          debug.log('Setting WebGL backend');
          await tf.setBackend('webgl');
          debug.log('Current backend: ' + tf.getBackend());
        } catch (backendError) {
          debug.error('Failed to set WebGL backend, trying CPU fallback', backendError);
          try {
            await tf.setBackend('cpu');
            debug.log('Fallback to CPU backend: ' + tf.getBackend());
          } catch (cpuError) {
            debug.error('Failed to set CPU backend too', cpuError);
            return false;
          }
        }
        
        // Check for WebGL context using the backend-specific functions
        try {
          // Instead of using tf.webgl.isWebGLAvailable() which doesn't exist, 
          // check the current backend to determine WebGL availability
          const currentBackend = tf.getBackend();
          const webglInfo = {
            available: currentBackend === 'webgl',
            version: currentBackend === 'webgl' ? 'WebGL' : 'None'
          };
          debug.log('WebGL info', webglInfo);
          
          if (!webglInfo.available) {
            debug.error('WebGL is not available on this device/browser');
          }
        } catch (webglError) {
          debug.error('Error checking WebGL status', webglError);
        }
        
        // Apply memory settings based on processing power
        try {
          switch (processingPower) {
            case 'low':
              tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
              tf.env().set('WEBGL_PACK', false);
              break;
            case 'high':
              tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
              tf.env().set('WEBGL_PACK', true);
              break;
            case 'medium':
            default:
              // Default settings
              break;
          }
          debug.log('Applied TF.js memory settings for power level: ' + processingPower);
        } catch (settingsError) {
          debug.error('Error applying TF.js memory settings', settingsError);
        }
        
        try {
          debug.log('Waiting for TensorFlow.js to be ready...');
          await tf.ready();
          debug.log('TensorFlow.js is ready');
        } catch (readyError) {
          debug.error('TF.js ready error', readyError);
          return false;
        }
        
        setStatus('Loading object detection model...');
        debug.log('Loading COCO-SSD model...');
        
        // Choose model base based on processing power
        const modelBase = processingPower === 'low' ? 'lite_mobilenet_v2' : 'mobilenet_v2';
        debug.log('Using model base: ' + modelBase);
        
        try {
          debug.log('Starting COCO-SSD model load...');
          
          // Check if COCO-SSD module is available
          if (!cocossd) {
            debug.error('COCO-SSD module is not loaded properly');
            return false;
          }
          
          debug.log('COCO-SSD module found, calling load method');
          const detector = await cocossd.load({
            base: modelBase
          });
          
          if (!detector) {
            debug.error('COCO-SSD model loaded but returned null/undefined');
            return false;
          }
          
          setObjectDetector(detector);
          
          debug.log('COCO-SSD model loaded successfully');
        } catch (modelError) {
          debug.error('Failed to load COCO-SSD model', modelError);
          setStatus(`Error loading COCO-SSD model: ${modelError instanceof Error ? modelError.message : 'Unknown error'}`);
          return false;
        }
        
        setStatus('Loading face detection model...');
        debug.log('Loading BlazeFace model...');
        
        // Adjust model parameters based on processing power
        const faceModelConfig = {
          maxFaces: processingPower === 'high' ? 5 : (processingPower === 'medium' ? 3 : 1),
          inputWidth: processingPower === 'low' ? 128 : 256,
          inputHeight: processingPower === 'low' ? 128 : 256,
          scoreThreshold: processingPower === 'low' ? 0.7 : 0.5
        };
        
        try {
          debug.log('Starting BlazeFace model load with config', faceModelConfig);
          
          // Check if BlazeFace module is available
          if (!blazeface) {
            debug.error('BlazeFace module is not loaded properly');
            return false;
          }
          
          const detector = await blazeface.load(faceModelConfig);
          
          if (!detector) {
            debug.error('BlazeFace model loaded but returned null/undefined');
            return false;
          }
          
          setFaceDetector(detector);
          
          debug.log('BlazeFace model loaded successfully');
        } catch (faceError) {
          debug.error('Failed to load BlazeFace model', faceError);
          setStatus(`Error loading BlazeFace model: ${faceError instanceof Error ? faceError.message : 'Unknown error'}`);
          return false;
        }
        
        debug.log('All models loaded successfully');
        setStatus('Models loaded successfully');
        return true;
      } catch (error) {
        debug.error('Error during model loading process', error);
        setStatus(`Error loading models: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return false;
      }
    }, [blazeface, cocossd, debug, processingPower, setStatus, tf]);

    // Advanced color detection with k-means clustering
    function detectDominantColors(imageData: ImageData, numClusters = 3): ColorInfo {
      const data = imageData.data;
      const pixels: number[][] = [];
      
      // Sample pixels (every 4th pixel to improve performance)
      for (let i = 0; i < data.length; i += 16) {
        pixels.push([data[i], data[i + 1], data[i + 2]]);
      }
      
      // Use k-means clustering to find dominant colors
      const colorClusters = kMeansClustering(pixels, numClusters);
      
      // Find the most saturated/colorful cluster (likely to be the hat color)
      let dominantCluster = colorClusters[0];
      let highestSaturation = colorSaturation(dominantCluster.centroid);
      
      for (let i = 1; i < colorClusters.length; i++) {
        const saturation = colorSaturation(colorClusters[i].centroid);
        if (saturation > highestSaturation) {
          highestSaturation = saturation;
          dominantCluster = colorClusters[i];
        }
      }
      
      const [r, g, b] = dominantCluster.centroid;
      
      // Match to named color
      const colorName = findNearestNamedColor(r, g, b);
      
      return { 
        r: Math.round(r), 
        g: Math.round(g), 
        b: Math.round(b), 
        colorName
      };
    }

    // Calculate color saturation (higher for more vibrant colors)
    function colorSaturation(rgb: number[]): number {
      const [r, g, b] = rgb;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      
      // Avoid division by zero
      if (max === 0) return 0;
      
      return (max - min) / max;
    }

    // K-means clustering algorithm for color quantization
    function kMeansClustering(points: number[][], k: number, maxIterations = 10): {
      centroid: number[],
      points: number[][]
    }[] {
      // Initialize centroids randomly from the data points
      const centroids = points.length >= k 
        ? points.sort(() => 0.5 - Math.random()).slice(0, k) 
        : Array(k).fill(0).map(() => [
            Math.random() * 255,
            Math.random() * 255,
            Math.random() * 255
          ]);
        
      const clusters = Array(k).fill(null).map(() => ({ 
        centroid: [0, 0, 0], 
        points: [] as number[][] 
      }));
      
      for (let iteration = 0; iteration < maxIterations; iteration++) {
        // Reset clusters
        clusters.forEach(cluster => {
          cluster.points = [];
        });
        
        // Assign points to nearest centroid
        for (let i = 0; i < points.length; i++) {
          const point = points[i];
          let minDistance = Infinity;
          let closestClusterIndex = 0;
          
          for (let j = 0; j < centroids.length; j++) {
            const distance = euclideanDistance(point, centroids[j]);
            if (distance < minDistance) {
              minDistance = distance;
              closestClusterIndex = j;
            }
          }
          
          clusters[closestClusterIndex].points.push(point);
        }
        
        // Calculate new centroids
        let changed = false;
        
        for (let i = 0; i < k; i++) {
          if (clusters[i].points.length === 0) continue;
          
          const newCentroid = [0, 0, 0];
          
          for (let j = 0; j < clusters[i].points.length; j++) {
            newCentroid[0] += clusters[i].points[j][0];
            newCentroid[1] += clusters[i].points[j][1];
            newCentroid[2] += clusters[i].points[j][2];
          }
          
          newCentroid[0] /= clusters[i].points.length;
          newCentroid[1] /= clusters[i].points.length;
          newCentroid[2] /= clusters[i].points.length;
          
          if (euclideanDistance(newCentroid, centroids[i]) > 1) {
            changed = true;
          }
          
          centroids[i] = newCentroid;
          clusters[i].centroid = newCentroid;
        }
        
        // If centroids didn't change significantly, we're done
        if (!changed) break;
      }
      
      return clusters;
    }

    // Calculate Euclidean distance between two RGB points
    function euclideanDistance(p1: number[], p2: number[]): number {
      return Math.sqrt(
        Math.pow(p2[0] - p1[0], 2) +
        Math.pow(p2[1] - p1[1], 2) +
        Math.pow(p2[2] - p1[2], 2)
      );
    }

    // Map RGB to named color
    function findNearestNamedColor(r: number, g: number, b: number): string {
      let minDistance = Infinity;
      let closestColor = 'unknown';
      
      for (const color of namedColors) {
        const distance = euclideanDistance([r, g, b], color.rgb);
        if (distance < minDistance) {
          minDistance = distance;
          closestColor = color.name;
        }
      }
      
      return closestColor;
    }

    // Estimate distance based on face size
    function estimateDistance(faceSize: number): number {
      const maxFaceSize = 300;
      const scaleFactor = 100;
      return Math.round((scaleFactor * maxFaceSize) / faceSize);
    }

    // Get human-readable movement direction
    function getMovementDirection(vx: number, vy: number): string {
      if (Math.abs(vx) < 1 && Math.abs(vy) < 1) return 'Stationary';
      
      const angle = Math.atan2(vy, vx) * 180 / Math.PI;
      
      if (angle > -22.5 && angle <= 22.5) return 'Right';
      if (angle > 22.5 && angle <= 67.5) return 'Down-Right';
      if (angle > 67.5 && angle <= 112.5) return 'Down';
      if (angle > 112.5 && angle <= 157.5) return 'Down-Left';
      if (angle > 157.5 || angle <= -157.5) return 'Left';
      if (angle > -157.5 && angle <= -112.5) return 'Up-Left';
      if (angle > -112.5 && angle <= -67.5) return 'Up';
      if (angle > -67.5 && angle <= -22.5) return 'Up-Right';
      
      return 'Unknown';
    }

    // Check if object is likely a hat based on position and shape
    function isLikelyHat(
      prediction: cocossd.DetectedObject,
      faceDetections: blazeface.NormalizedFace[]
    ): boolean {
      // These are common classes that may represent hats or head items
      const possibleHatClasses = ['hat', 'cap', 'helmet', 'sports ball', 'frisbee', 'bowl'];
      
      // If the prediction includes "hat" or similar in its class name, consider it likely
      if (possibleHatClasses.some(cls => prediction.class.toLowerCase().includes(cls))) {
        return true;
      }
      
      // For other objects, check if they're positioned above a face
      for (const face of faceDetections) {
        const faceTop = face.topLeft[1];
        const faceWidth = face.bottomRight[0] - face.topLeft[0];
        const faceCenterX = (face.topLeft[0] + face.bottomRight[0]) / 2;
        
        const objBottom = prediction.bbox[1] + prediction.bbox[3];
        const objCenterX = prediction.bbox[0] + prediction.bbox[2] / 2;
        const objWidth = prediction.bbox[2];
        const objHeight = prediction.bbox[3];
        
        // Check if object is above and roughly centered with a face
        const isAboveFace = objBottom >= faceTop - 30 && objBottom <= faceTop + 40;
        const isNearFaceCenterX = Math.abs(objCenterX - faceCenterX) < faceWidth * 0.7;
        const hasReasonableSize = objWidth >= faceWidth * 0.4 && objWidth <= faceWidth * 2.2;
        const hasReasonableShape = objWidth / objHeight >= 1.0; // Hats tend to be wider than tall
        
        if (isAboveFace && isNearFaceCenterX && hasReasonableSize && hasReasonableShape) {
          return true;
        }
      }
      
      return false;
    }

    // Calculate hat-like confidence
    function calculateHatConfidence(
      prediction: cocossd.DetectedObject,
      faceDetections: blazeface.NormalizedFace[]
    ): number {
      // These are common classes that may represent hats or head items
      const directHatClasses = ['hat', 'cap', 'helmet'];
      const secondaryHatClasses = ['sports ball', 'frisbee', 'bowl'];
      
      // Start with the model's confidence
      let confidence = prediction.score;
      
      // Boost confidence for hat-related classes
      if (directHatClasses.some(cls => prediction.class.toLowerCase().includes(cls))) {
        confidence = Math.min(1.0, confidence * 1.2);
        return confidence;
      }
      
      if (secondaryHatClasses.some(cls => prediction.class.toLowerCase().includes(cls))) {
        confidence = Math.min(1.0, confidence * 1.1);
      }
      
      // For other objects, assess positional relationship with faces
      for (const face of faceDetections) {
        const faceTop = face.topLeft[1];
        const faceWidth = face.bottomRight[0] - face.topLeft[0];
        const faceCenterX = (face.topLeft[0] + face.bottomRight[0]) / 2;
        
        const objBottom = prediction.bbox[1] + prediction.bbox[3];
        const objCenterX = prediction.bbox[0] + prediction.bbox[2] / 2;
        const objWidth = prediction.bbox[2];
        const objHeight = prediction.bbox[3];
        
        // Assess position metrics and adjust confidence accordingly
        const verticalPosition = Math.abs(objBottom - faceTop) / faceWidth;
        const horizontalAlignment = Math.abs(objCenterX - faceCenterX) / faceWidth;
        const sizeRatio = objWidth / faceWidth;
        const aspectRatio = objWidth / objHeight;
        
        // Ideal hat is directly above face, centered, with reasonable size and aspect ratio
        let positionalConfidence = 0.5;
        
        // Vertical position score - highest when just above face
        if (verticalPosition < 0.2) positionalConfidence += 0.2;
        else if (verticalPosition < 0.4) positionalConfidence += 0.1;
        else positionalConfidence -= 0.1;
        
        // Horizontal alignment score - highest when centered above face
        if (horizontalAlignment < 0.2) positionalConfidence += 0.2;
        else if (horizontalAlignment < 0.5) positionalConfidence += 0.1;
        else positionalConfidence -= 0.1;
        
        // Size ratio score - highest when hat width is similar to face width
        if (sizeRatio > 0.7 && sizeRatio < 1.5) positionalConfidence += 0.2;
        else if (sizeRatio > 0.5 && sizeRatio < 2.0) positionalConfidence += 0.1;
        else positionalConfidence -= 0.1;
        
        // Aspect ratio score - hats tend to be wider than tall
        if (aspectRatio > 1.2 && aspectRatio < 3.0) positionalConfidence += 0.1;
        
        // Combine original and positional confidence
        confidence = (confidence + positionalConfidence) / 2;
        break; // Only use the best face for this calculation
      }
      
      return Math.max(0, Math.min(1, confidence));
    }

    // Apply non-maximum suppression to remove duplicate detections
    function nonMaximumSuppression(
      detections: {bbox: number[], confidence: number, id: string}[], 
      overlapThreshold = 0.5
    ): {bbox: number[], confidence: number, id: string}[] {
      // Sort detections by confidence score (highest first)
      const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);
      const selectedDetections: {bbox: number[], confidence: number, id: string}[] = [];
      
      while (sortedDetections.length > 0) {
        // Select the detection with highest confidence
        const currentDetection = sortedDetections.shift();
        if (!currentDetection) break;
        
        selectedDetections.push(currentDetection);
        
        // Filter remaining detections to remove overlapping ones
        const remainingDetections: {bbox: number[], confidence: number, id: string}[] = [];
        
        for (const detection of sortedDetections) {
          const overlap = calculateIoU(currentDetection.bbox, detection.bbox);
          
          if (overlap < overlapThreshold) {
            remainingDetections.push(detection);
          }
        }
        
        sortedDetections.length = 0;
        sortedDetections.push(...remainingDetections);
      }
      
      return selectedDetections;
    }

    // Calculate Intersection over Union for two bounding boxes
    function calculateIoU(bbox1: number[], bbox2: number[]): number {
      const [x1, y1, width1, height1] = bbox1;
      const [x2, y2, width2, height2] = bbox2;
      
      const xOverlap = Math.max(0, Math.min(x1 + width1, x2 + width2) - Math.max(x1, x2));
      const yOverlap = Math.max(0, Math.min(y1 + height1, y2 + height2) - Math.max(y1, y2));
      const intersectionArea = xOverlap * yOverlap;
      
      const bbox1Area = width1 * height1;
      const bbox2Area = width2 * height2;
      const unionArea = bbox1Area + bbox2Area - intersectionArea;
      
      return intersectionArea / unionArea;
    }

  const detectInRealTime = useCallback(async () => {
    if (!objectDetector || !faceDetector || !videoRef.current || !canvasRef.current || !cameraActive) return;
    
    const startTime = performance.now();
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Apply frame skipping based on processing power to improve performance on lower-end devices
      const frameCount = statsRef.current.frames;
      const shouldSkipFrame = 
        (processingPower === 'low' && frameCount % 3 !== 0) || 
        (processingPower === 'medium' && frameCount % 2 !== 0);
      
      // Skip frames if needed based on processing power
      if (shouldSkipFrame) {
        // Just draw the frame but don't process it
        if (isFullscreen && containerRef.current) {
          const container = containerRef.current;
          const containerAspect = container.clientWidth / container.clientHeight;
          const videoAspect = video.videoWidth / video.videoHeight;
          
          let drawWidth, drawHeight;
          
          if (containerAspect > videoAspect) {
            drawHeight = container.clientHeight;
            drawWidth = drawHeight * videoAspect;
          } else {
            drawWidth = container.clientWidth;
            drawHeight = drawWidth / videoAspect;
          }
          
          const x = (container.clientWidth - drawWidth) / 2;
          const y = (container.clientHeight - drawHeight) / 2;
          
          canvas.width = container.clientWidth;
          canvas.height = container.clientHeight;
          
          ctx.drawImage(video, x, y, drawWidth, drawHeight);
        } else {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
        
        // Continue without processing
        statsRef.current.frames++;
        setAnimationId(requestAnimationFrame(detectInRealTime));
        return;
      }
      
      // Perform parallel detection
      const [objectPredictions, faceDetections] = await Promise.all([
        objectDetector.detect(video),
        faceDetector.estimateFaces(video, false)
      ]);
      
      // Clear canvas with proper dimensions
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Adjust canvas scaling based on fullscreen state
      if (isFullscreen && containerRef.current) {
        const container = containerRef.current;
        const containerAspect = container.clientWidth / container.clientHeight;
        const videoAspect = video.videoWidth / video.videoHeight;
        
        let drawWidth, drawHeight;
        
        if (containerAspect > videoAspect) {
          // Container is wider than video
          drawHeight = container.clientHeight;
          drawWidth = drawHeight * videoAspect;
        } else {
          // Container is taller than video
          drawWidth = container.clientWidth;
          drawHeight = drawWidth / videoAspect;
        }
        
        // Center the video in the container
        const x = (container.clientWidth - drawWidth) / 2;
        const y = (container.clientHeight - drawHeight) / 2;
        
        // Update canvas size to match container
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        
        // Draw video with proper scaling
        ctx.drawImage(video, x, y, drawWidth, drawHeight);
      } else {
        // Standard rendering for non-fullscreen
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      
      const currentFrameHats: Record<string, TrackedHat> = {};
      const prelimHatDetections: {bbox: number[], confidence: number, id: string}[] = [];
      
      // First pass: identify all potential hats
      for (const prediction of objectPredictions) {
        if (isLikelyHat(prediction, faceDetections)) {
          const [x, y] = prediction.bbox;
          const hatConfidence = calculateHatConfidence(prediction, faceDetections);
          
          // Generate a temporary ID
          const tempId = `hat_${Math.round(x)}_${Math.round(y)}`;
          
          prelimHatDetections.push({
            bbox: prediction.bbox,
            confidence: hatConfidence,
            id: tempId
          });
        }
      }
      
      // Apply non-maximum suppression to remove overlapping detections
      const filteredHatDetections = nonMaximumSuppression(prelimHatDetections);
      
      // Second pass: process the filtered hat detections
      for (const detection of filteredHatDetections) {
        const [x, y, width, height] = detection.bbox;
        const centerX = x + width/2;
        const centerY = y + height/2;
        
        try {
          // Sample from center of object for color analysis
          const colorX = Math.floor(x + width / 4);
          const colorY = Math.floor(y + height / 4);
          const colorWidth = Math.floor(width / 2);
          const colorHeight = Math.floor(height / 2);
          
          const imageData = ctx.getImageData(colorX, colorY, colorWidth, colorHeight);
          const colorInfo = detectDominantColors(imageData);
          
          // Estimate distance
          let distance = "unknown";
          let distanceValue = null;
          let nearestFace = null;
          
          // Find the nearest face to this hat
          if (faceDetections.length > 0) {
            let minDistance = Infinity;
            
            for (const face of faceDetections) {
              const faceCenterX = (face.topLeft[0] + face.bottomRight[0]) / 2;
              const faceCenterY = (face.topLeft[1] + face.bottomRight[1]) / 2;
              const dx = centerX - faceCenterX;
              const dy = centerY - faceCenterY;
              const distSquared = dx*dx + dy*dy;
              
              if (distSquared < minDistance) {
                minDistance = distSquared;
                nearestFace = face;
              }
            }
            
            if (nearestFace) {
              const faceWidth = nearestFace.bottomRight[0] - nearestFace.topLeft[0];
              distanceValue = estimateDistance(faceWidth);
              distance = `~${distanceValue}cm`;
            }
          }
          
          // Generate a unique hat ID based on position and color
          const hatId = `hat_${Math.round(centerX)}_${Math.round(centerY)}_${colorInfo.colorName}`;
          
          // Store this hat in the current frame collection
          currentFrameHats[hatId] = {
            id: hatId,
            x, y, width, height,
            centerX, centerY,
            color: colorInfo.colorName,
            colorRGB: { r: colorInfo.r, g: colorInfo.g, b: colorInfo.b },
            distance: distanceValue,
            distanceLabel: distance,
            confidence: detection.confidence,
            timestamp: Date.now(),
            nearestFaceId: nearestFace ? `face_${nearestFace.topLeft.join('_')}` : null,
            velocityX: 0,
            velocityY: 0,
            distanceChange: 0,
            framesTracked: 1
          };
          
          // Check if this hat was previously tracked
          const existingHatKey = Object.keys(trackedHatsRef.current).find(key => {
            const existing = trackedHatsRef.current[key];
            // Check if it's approximately the same hat based on position and color
            const xDiff = Math.abs(existing.centerX - centerX);
            const yDiff = Math.abs(existing.centerY - centerY);
            const sameColor = existing.color === colorInfo.colorName;
            return xDiff < 50 && yDiff < 50 && sameColor;
          });
          
          if (existingHatKey) {
            // Update existing hat with new information
            const existingHat = trackedHatsRef.current[existingHatKey];
            currentFrameHats[existingHatKey] = {
              ...existingHat,
              x, y, width, height,
              centerX, centerY,
              distance: distanceValue,
              distanceLabel: distance,
              confidence: detection.confidence,
              timestamp: Date.now(),
              // Calculate velocity based on position change
              velocityX: centerX - existingHat.centerX,
              velocityY: centerY - existingHat.centerY,
              // Track distance change
              distanceChange: existingHat.distance ? distanceValue - existingHat.distance : 0,
              framesTracked: (existingHat.framesTracked || 0) + 1
            };
            
            // Remove the temporary ID entry since we're using the tracked one
            delete currentFrameHats[hatId];
          }
          
        } catch (e) {
          console.error("Error analyzing hat:", e);
        }
      }
      
      // Update tracked hats reference
      trackedHatsRef.current = {...currentFrameHats};
      
      // Expire old hats (not seen for more than 2 seconds)
      const currentTime = Date.now();
      Object.keys(trackedHatsRef.current).forEach(key => {
        if (currentTime - trackedHatsRef.current[key].timestamp > 2000) {
          delete trackedHatsRef.current[key];
        }
      });
      
      // Calculate trajectory prediction for each hat
      Object.values(trackedHatsRef.current).forEach(hat => {
        if (hat.framesTracked > 5 && (Math.abs(hat.velocityX) > 1 || Math.abs(hat.velocityY) > 1)) {
          // Simple linear prediction of future position (10 frames ahead)
          hat.predictedX = hat.centerX + hat.velocityX * 10;
          hat.predictedY = hat.centerY + hat.velocityY * 10;
        }
      });
      
      // Draw faces for reference (semi-transparent)
      faceDetections.forEach(face => {
        const x = face.topLeft[0];
        const y = face.topLeft[1];
        const width = face.bottomRight[0] - face.topLeft[0];
        const height = face.bottomRight[1] - face.topLeft[1];
        
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
      });
      
      // Draw all detected hats
      Object.values(trackedHatsRef.current).forEach(hat => {
        const {x, y, width, height, colorRGB, color, distanceLabel, framesTracked, confidence} = hat;
        
        // Draw bounding box - thicker for hats tracked longer
        const lineWidth = Math.min(5, 1 + Math.floor(framesTracked / 10));
        ctx.strokeStyle = `rgb(${colorRGB.r}, ${colorRGB.g}, ${colorRGB.b})`;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, width, height);
        
        // Draw label
        ctx.fillStyle = `rgba(${colorRGB.r}, ${colorRGB.g}, ${colorRGB.b}, 0.7)`;
        ctx.fillRect(x, y - 30, width, 30);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '16px Arial';
        ctx.fillText(`${color} hat (${Math.round(confidence * 100)}%)`, x + 5, y - 10);
        
        // Draw distance info
        ctx.font = '12px Arial';
        ctx.fillText(`Distance: ${distanceLabel}`, x + 5, y + height + 15);
        
        // Draw tracking ID
        ctx.fillText(`ID: ${hat.id.substring(0, 8)}`, x + 5, y + height + 30);
        
        // Draw movement vector if hat is moving
        if (Math.abs(hat.velocityX) > 1 || Math.abs(hat.velocityY) > 1) {
          ctx.beginPath();
          ctx.moveTo(hat.centerX, hat.centerY);
          ctx.lineTo(hat.centerX + hat.velocityX * 5, hat.centerY + hat.velocityY * 5);
          ctx.strokeStyle = 'yellow';
          ctx.lineWidth = 2;
          ctx.stroke();
          
          // Draw predicted future position
          if (hat.predictedX && hat.predictedY) {
            ctx.beginPath();
            ctx.arc(hat.predictedX, hat.predictedY, 5, 0, Math.PI * 2);
            ctx.fillStyle = 'yellow';
            ctx.fill();
          }
        }
      });
      
      // Calculate frame processing time
      const endTime = performance.now();
      const processingTime = endTime - startTime;
      
      // Update statistics
      statsRef.current.frames++;
      statsRef.current.detections += Object.keys(currentFrameHats).length;
      statsRef.current.performance.push(processingTime);
      
      // Keep only last 30 performance measurements
      if (statsRef.current.performance.length > 30) {
        statsRef.current.performance.shift();
      }
      
      // Draw performance metrics
      const avgPerformance = statsRef.current.performance.length > 0 
        ? statsRef.current.performance.reduce((a, b) => a + b, 0) / statsRef.current.performance.length 
        : 0;
        
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(5, 5, 200, 20);
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(`Processing: ${processingTime.toFixed(1)}ms | FPS: ${(1000 / (avgPerformance || 1)).toFixed(1)}`, 10, 20);
      
      // Update state with tracked hats
      setTrackedHats(trackedHatsRef.current);
      
      // Update results for display
      const results: HatDetection[] = Object.values(trackedHatsRef.current).map(hat => ({
        id: hat.id.substring(0, 8),
        type: 'Hat',
        color: hat.color,
        distance: hat.distanceLabel,
        confidence: hat.confidence.toFixed(2),
        framesTracked: hat.framesTracked,
        moving: Math.abs(hat.velocityX) > 1 || Math.abs(hat.velocityY) > 1 ? 'Yes' : 'No',
        direction: getMovementDirection(hat.velocityX, hat.velocityY)
      }));
      
      setDetections(results);
      setLoading(false);
      
      // Continue detection
      setAnimationId(requestAnimationFrame(detectInRealTime));
    }, [objectDetector, faceDetector, videoRef, canvasRef, cameraActive, isFullscreen, containerRef, processingPower, statsRef, trackedHatsRef]);

  const init = useCallback(async () => {
    debug.log('Initializing camera and models');
    try {
      debug.log('Setting up camera...');
      const cameraReady = await setupCamera(cameraFacing);
      if (!cameraReady) {
        debug.error('Camera setup failed');
        return;
      }
      debug.log('Camera ready');
      
      debug.log('Loading ML models...');
      const modelsReady = await loadModels();
      if (!modelsReady) {
        debug.error('Models loading failed');
        return;
      }
      debug.log('Models ready');
      
      setStatus('Ready! Starting detection...');
      if (cameraActive) {
        debug.log('Starting real-time detection');
        detectInRealTime();
      }
    } catch (error) {
      debug.error('Initialization error', error);
      setStatus(`Initialization error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [debug, cameraActive, cameraFacing, detectInRealTime, setupCamera, loadModels, setStatus]);
    
    // Camera and settings effects
    useEffect(() => {
      debug.log('Component mounted, starting initialization');
      init();
      
      // Cleanup on unmount
      return () => {
        debug.log('Component unmounting, cleaning up resources');
        if (animationId !== null) {
          cancelAnimationFrame(animationId);
        }
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
      };
    }, [debug, init, animationId]); // Initial setup
    
    // Handle camera facing mode changes
    useEffect(() => {
      if (loading) return; // Skip if initial loading
      
      setupCamera(cameraFacing)
        .then(success => {
          if (success && cameraActive) {
            detectInRealTime();
          }
        });
    }, [cameraFacing]);
    
    // Handle processing power changes
    useEffect(() => {
      if (loading) return; // Skip if initial loading
      
      // Reload models when processing power changes
      loadModels().then(success => {
        if (success && cameraActive) {
          detectInRealTime();
        }
      });
    }, [processingPower]);
    
    // Handle camera active toggle
    useEffect(() => {
      if (loading) return; // Skip if initial loading
      
      if (cameraActive) {
        if (videoRef.current && videoRef.current.paused) {
          videoRef.current.play();
          detectInRealTime();
        }
      } else {
        if (videoRef.current && !videoRef.current.paused) {
          videoRef.current.pause();
        }
        if (animationId) {
          cancelAnimationFrame(animationId);
        }
      }
    }, [cameraActive]);
  
  // Toggle camera on/off
  const toggleCamera = () => {
    if (cameraActive) {
      setCameraActive(false);
    } else {
      setCameraActive(true);
    }
  };
  
  // Switch camera facing mode
  const switchCamera = () => {
    const newFacing = cameraFacing === 'user' ? 'environment' : 'user';
    setCameraFacing(newFacing);
  };
  
  // Change processing power level
  const setProcessingLevel = (level: ProcessingPower) => {
    setProcessingPower(level);
    setShowSettings(false);
  };
  
  // Toggle settings panel
  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };
  
  // Fullscreen toggle function
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    
    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen()
          .then(() => setIsFullscreen(true))
          .catch(err => console.error(`Error attempting to enable fullscreen: ${err.message}`));
      } else if ((containerRef.current as any).webkitRequestFullscreen) {
        (containerRef.current as any).webkitRequestFullscreen();
        setIsFullscreen(true);
      } else if ((containerRef.current as any).msRequestFullscreen) {
        (containerRef.current as any).msRequestFullscreen();
        setIsFullscreen(true);
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
          .then(() => setIsFullscreen(false))
          .catch(err => console.error(`Error attempting to exit fullscreen: ${err.message}`));
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen();
        setIsFullscreen(false);
      } else if ((document as any).msExitFullscreen) {
        (document as any).msExitFullscreen();
        setIsFullscreen(false);
      }
    }
  };
  
  // Listen for fullscreen change events
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(
        document.fullscreenElement || 
        (document as any).webkitFullscreenElement || 
        (document as any).msFullscreenElement
      ));
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('msfullscreenchange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('msfullscreenchange', handleFullscreenChange);
    };
  }, []);

  return (
    <div 
      className={`flex flex-col items-center w-full mx-auto transition-all ${
        isFullscreen ? '' : 'max-w-4xl p-4'
      }`}
    >
      <div className={`flex items-center justify-between w-full ${
        isFullscreen ? 'bg-blue-800 text-white p-3' : 'mb-4'
      }`}>
        <div className="flex items-center gap-3">
          <Camera className={`w-8 h-8 ${isFullscreen ? 'text-white' : 'text-blue-600'}`} />
          <h1 className={`${isFullscreen ? 'text-xl' : 'text-2xl'} font-bold`}>
            Advanced Hat Detection & Tracking System
          </h1>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Camera on/off button */}
          <button 
            onClick={toggleCamera}
            className={`flex items-center gap-1 px-2 py-2 rounded-lg transition-colors ${
              isFullscreen 
                ? 'bg-blue-700 hover:bg-blue-600 text-white' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
            aria-label={cameraActive ? "Turn camera off" : "Turn camera on"}
          >
            {cameraActive ? (
              <Camera className="w-5 h-5" />
            ) : (
              <CameraOff className="w-5 h-5" />
            )}
          </button>
          
          {/* Flip camera button */}
          <button 
            onClick={switchCamera}
            className={`flex items-center gap-1 px-2 py-2 rounded-lg transition-colors ${
              isFullscreen 
                ? 'bg-blue-700 hover:bg-blue-600 text-white' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
            aria-label="Switch camera"
            disabled={!cameraActive}
          >
            <FlipHorizontal className="w-5 h-5" />
          </button>
          
          {/* Settings button */}
          <div className="relative">
            <button 
              onClick={toggleSettings}
              className={`flex items-center gap-1 px-2 py-2 rounded-lg transition-colors ${
                isFullscreen 
                  ? 'bg-blue-700 hover:bg-blue-600 text-white' 
                  : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
              } ${showSettings ? 'bg-blue-300' : ''}`}
              aria-label="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>
            
            {/* Settings dropdown */}
            {showSettings && (
              <div className={`absolute right-0 top-full mt-2 w-52 p-3 rounded-lg shadow-lg z-10 ${
                isFullscreen ? 'bg-blue-900 text-white' : 'bg-white'
              }`}>
                <h3 className="font-medium mb-2 flex items-center gap-1">
                  <Cpu className="w-4 h-4" />
                  Processing Power
                </h3>
                <div className="space-y-1">
                  <button 
                    onClick={() => setProcessingLevel('low')}
                    className={`w-full text-left px-2 py-1 rounded ${
                      processingPower === 'low' 
                        ? (isFullscreen ? 'bg-blue-700' : 'bg-blue-100') 
                        : 'hover:bg-gray-100 hover:bg-opacity-20'
                    }`}
                  >
                    Low (Better Performance)
                  </button>
                  <button 
                    onClick={() => setProcessingLevel('medium')}
                    className={`w-full text-left px-2 py-1 rounded ${
                      processingPower === 'medium' 
                        ? (isFullscreen ? 'bg-blue-700' : 'bg-blue-100') 
                        : 'hover:bg-gray-100 hover:bg-opacity-20'
                    }`}
                  >
                    Medium (Balanced)
                  </button>
                  <button 
                    onClick={() => setProcessingLevel('high')}
                    className={`w-full text-left px-2 py-1 rounded ${
                      processingPower === 'high' 
                        ? (isFullscreen ? 'bg-blue-700' : 'bg-blue-100') 
                        : 'hover:bg-gray-100 hover:bg-opacity-20'
                    }`}
                  >
                    High (Better Quality)
                  </button>
                </div>
              </div>
            )}
          </div>
          
          {/* Fullscreen button */}
          <button 
            onClick={toggleFullscreen}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
              isFullscreen 
                ? 'bg-blue-700 hover:bg-blue-600 text-white' 
                : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
            }`}
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? (
              <>
                <Minimize className="w-5 h-5" />
                <span className="hidden sm:inline">Exit</span>
              </>
            ) : (
              <>
                <Maximize className="w-5 h-5" />
                <span className="hidden sm:inline">Fullscreen</span>
              </>
            )}
          </button>
        </div>
      </div>
      
      {/* Status message */}
      {(loading || status.startsWith('[Debug]') || status.startsWith('[Error]')) && (
        <div className={`mb-4 p-4 ${
          status.startsWith('[Error]') 
            ? 'bg-red-100 text-red-700' 
            : (isFullscreen ? 'bg-blue-700 text-white' : 'bg-blue-50 text-blue-700')
        } rounded flex items-center gap-2`}>
          <Activity className="w-5 h-5 animate-pulse" />
          {status}
        </div>
      )}
      
      {/* Always show loading info and debug panel during loading */}
      {loading && (
        <div className="w-full mb-4 p-4 bg-blue-600 text-white rounded-lg shadow">
          <h3 className="font-bold text-lg mb-2 flex items-center">
            <Activity className="w-6 h-6 mr-2 animate-pulse" />
            Loading System Status
          </h3>
          
          <div className="bg-blue-700 p-3 rounded">
            <div className="mb-3 font-medium">Current Status: {status}</div>
            
            {errorInfo.length > 0 && (
              <div className="mb-4">
                <h4 className="font-semibold text-red-300 mb-1">Errors Detected:</h4>
                <pre className="bg-red-900 text-red-100 p-2 rounded text-sm overflow-auto max-h-40">
                  {errorInfo.map((err, i) => (
                    <div key={i} className="mb-1">{err}</div>
                  ))}
                </pre>
              </div>
            )}
            
            {debugInfo.length > 0 && (
              <div>
                <h4 className="font-semibold text-blue-200 mb-1">Loading Progress:</h4>
                <pre className="bg-blue-800 text-blue-100 p-2 rounded text-sm overflow-auto max-h-40">
                  {debugInfo.slice(-10).map((log, i) => (
                    <div key={i} className="mb-1">{log}</div>
                  ))}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
      
      <div 
        ref={containerRef}
        className={`relative ${
          isFullscreen ? 'w-full h-screen flex items-center justify-center' : 'w-full max-w-lg bg-white p-4 rounded-lg shadow-lg'
        }`}
      >
        <video
          ref={videoRef}
          className={`${isFullscreen ? '' : 'w-full'} rounded-lg`}
          playsInline
          style={{ display: 'none' }}
        />
        <canvas
          ref={canvasRef}
          className={`${isFullscreen ? 'max-h-full max-w-full' : 'w-full'} rounded-lg`}
        />
        
        {isFullscreen && (
          <div className="absolute top-16 right-4 z-10 bg-black bg-opacity-50 text-white p-2 rounded">
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                <span className="text-sm font-medium">
                  Tracking {Object.keys(trackedHats).length} hats
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-300">
                <FlipHorizontal className="w-3 h-3" />
                <span>Camera: {cameraFacing === 'user' ? 'Front' : 'Back'}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-300">
                <Cpu className="w-3 h-3" />
                <span>Processing: {processingPower.charAt(0).toUpperCase() + processingPower.slice(1)}</span>
              </div>
              {statsRef.current.performance.length > 0 && (
                <div className="flex items-center gap-2 text-xs text-gray-300">
                  <Zap className="w-3 h-3" />
                  <span>FPS: {Math.round(1000 / (statsRef.current.performance.reduce((a, b) => a + b, 0) / statsRef.current.performance.length || 1))}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {!isFullscreen && (
        <>
          <div className="mt-6 w-full bg-white p-6 rounded-lg shadow-lg">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-semibold">Detected Hats ({Object.keys(trackedHats).length})</h2>
            </div>
            
            {detections.length === 0 ? (
              <p className="text-gray-500 italic">No hats detected yet</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="p-3 text-left border">ID</th>
                      <th className="p-3 text-left border">Color</th>
                      <th className="p-3 text-left border">Distance</th>
                      <th className="p-3 text-left border">Confidence</th>
                      <th className="p-3 text-left border">Tracked Frames</th>
                      <th className="p-3 text-left border">Moving</th>
                      <th className="p-3 text-left border">Direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detections.map((item, index) => (
                      <tr key={index} className="hover:bg-gray-50 transition-colors">
                        <td className="p-3 border font-mono text-sm">{item.id}</td>
                        <td className="p-3 border">
                          <div className="flex items-center gap-2">
                            <span 
                              className="inline-block w-4 h-4 rounded-full" 
                              style={{backgroundColor: item.color}}
                            />
                            <span>{item.color}</span>
                          </div>
                        </td>
                        <td className="p-3 border">{item.distance}</td>
                        <td className="p-3 border">{item.confidence}</td>
                        <td className="p-3 border">{item.framesTracked}</td>
                        <td className="p-3 border">
                          <span className={`px-2 py-1 rounded-full text-sm ${
                            item.moving === 'Yes' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {item.moving}
                          </span>
                        </td>
                        <td className="p-3 border">{item.direction || 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          
          <div className="mt-8 p-6 bg-white rounded-lg shadow-lg w-full">
            <div className="flex items-center gap-2 mb-4">
              <Zap className="w-6 h-6 text-blue-600" />
              <h3 className="text-lg font-semibold">Advanced Detection Features</h3>
            </div>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2">Computer Vision Pipeline</h4>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full" />
                    Object detection with ML model classification
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full" />
                    Face-relative hat positioning analysis
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full" />
                    K-means clustering for color analysis
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full" />
                    Non-maximum suppression for overlapping detections
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">Tracking System</h4>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full" />
                    Multi-object tracking with persistence
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full" />
                    Trajectory prediction and visualization
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full" />
                    Distance estimation via face size reference
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full" />
                    Movement direction classification
                  </li>
                </ul>
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-600">
              This system uses TensorFlow.js with advanced computer vision algorithms for real-time hat detection and tracking.
              The system performs k-means clustering for color analysis, maintains object persistence across frames, and
              provides detailed analytics on each detected hat including distance estimation, movement vectors, and trajectory prediction.
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default HatDetector;