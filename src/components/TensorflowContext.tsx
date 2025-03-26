import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Make these imports dynamic to avoid React hook errors
let tf: any = null;
let cocossd: any = null;
let blazeface: any = null;

// Interface for the TensorFlow context
interface TensorflowContextType {
  isLoading: boolean;
  error: string | null;
  tf: any;
  cocossd: any;
  blazeface: any;
}

// Create context with default values
const TensorflowContext = createContext<TensorflowContextType>({
  isLoading: true,
  error: null,
  tf: null,
  cocossd: null,
  blazeface: null
});

// Provider component
export const TensorflowProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tfInstance, setTfInstance] = useState<any>(null);
  const [cocossdInstance, setCocossdInstance] = useState<any>(null);
  const [blazefaceInstance, setBlazeface] = useState<any>(null);

  useEffect(() => {
    async function loadLibraries() {
      try {
        console.log('Loading TensorFlow.js...');
        const tfModule = await import('@tensorflow/tfjs');
        // We need to ensure the backend is loaded
        try {
          await tfModule.setBackend('webgl');
          console.log('Using WebGL backend');
        } catch (backendError) {
          console.warn('WebGL backend failed, falling back to CPU', backendError);
          await tfModule.setBackend('cpu');
        }
        
        await tfModule.ready();
        setTfInstance(tfModule);
        console.log('TensorFlow.js loaded successfully');

        console.log('Loading COCO-SSD model...');
        const cocossdModule = await import('@tensorflow-models/coco-ssd');
        setCocossdInstance(cocossdModule);
        console.log('COCO-SSD loaded successfully');

        console.log('Loading BlazeFace model...');
        const blazefaceModule = await import('@tensorflow-models/blazeface');
        setBlazeface(blazefaceModule);
        console.log('BlazeFace loaded successfully');

        setIsLoading(false);
      } catch (err) {
        console.error('Error loading TensorFlow libraries:', err);
        setError(err instanceof Error ? err.message : 'Unknown error loading machine learning libraries');
        setIsLoading(false);
      }
    }

    loadLibraries();
  }, []);

  return (
    <TensorflowContext.Provider 
      value={{ 
        isLoading, 
        error, 
        tf: tfInstance, 
        cocossd: cocossdInstance, 
        blazeface: blazefaceInstance 
      }}
    >
      {children}
    </TensorflowContext.Provider>
  );
};

// Custom hook to use the TensorFlow context
export const useTensorflow = () => useContext(TensorflowContext);