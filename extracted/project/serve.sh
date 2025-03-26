#!/bin/bash
# Run the production build using a simple Node.js server

# Kill any existing Node.js or Python servers
pkill -f "node" || true
pkill -f "python" || true

echo "Starting Hat Detection application server..."
echo "----------------------------------------"

# Check if dist directory exists, if not, build the application
if [ ! -d "dist" ]; then
  echo "Build directory not found. Running build process..."
  npm run build
fi

# Run the server
node simple-serve.js