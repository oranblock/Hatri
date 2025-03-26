export interface TrackedHat {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  color: string;
  colorRGB: {
    r: number;
    g: number;
    b: number;
  };
  distance: number | null;
  distanceLabel: string;
  confidence: number;
  timestamp: number;
  nearestFaceId: string | null;
  velocityX: number;
  velocityY: number;
  distanceChange: number;
  framesTracked: number;
  predictedX?: number;
  predictedY?: number;
}

export interface HatDetection {
  id: string;
  type: string;
  color: string;
  distance: string;
  confidence: string;
  framesTracked: number;
  moving: string;
  direction?: string;
}

export interface ColorInfo {
  r: number;
  g: number;
  b: number;
  colorName: string;
}