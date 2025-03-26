import React from 'react';
import HatDetector from './components/HatDetector';
import { useTensorflow } from './components/TensorflowContext';

function App() {
  const { isLoading, error } = useTensorflow();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center p-8 max-w-lg">
          <h1 className="text-2xl font-bold text-blue-600 mb-4">Initializing Hat Detection System</h1>
          <p className="mb-4">Loading TensorFlow.js and initializing models...</p>
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center p-8 max-w-lg bg-white rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Hat Detection System - Error</h1>
          <p className="mb-4">Failed to initialize the TensorFlow.js library:</p>
          <div className="bg-red-50 p-4 rounded-lg text-red-800 mb-6 text-left">
            <strong>Error:</strong> {error}
          </div>
          <p className="mb-4">This could be caused by:</p>
          <ul className="text-left list-disc pl-6 mb-6">
            <li>WebGL not being supported by your browser</li>
            <li>Hardware acceleration being disabled</li>
            <li>Low system resources or memory</li>
            <li>Browser privacy features blocking WebGL</li>
          </ul>
          <button 
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <HatDetector />
    </div>
  );
}

export default App;