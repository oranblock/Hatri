import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { TensorflowProvider } from './components/TensorflowContext';
import App from './App.tsx';
import './index.css';

// Add global error handler for better error reporting
window.addEventListener('error', (event) => {
  console.error('Global error caught:', event.error || event.message);
  
  // Don't show React hook errors in the UI since they're confusing to users
  if (!event.error?.message?.includes('Invalid hook call')) {
    // Add error to the DOM for visibility
    const errorDiv = document.createElement('div');
    errorDiv.style.backgroundColor = '#ffdddd';
    errorDiv.style.color = '#bb0000';
    errorDiv.style.padding = '15px';
    errorDiv.style.margin = '10px';
    errorDiv.style.border = '1px solid #ff0000';
    errorDiv.style.borderRadius = '5px';
    errorDiv.innerHTML = `<strong>Error detected:</strong> ${event.error?.message || event.message}`;
    
    const root = document.getElementById('root');
    if (root) {
      root.prepend(errorDiv);
    }
  }
});

// Simple render with the TensorflowProvider to manage library loading
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TensorflowProvider>
      <App />
    </TensorflowProvider>
  </StrictMode>
);