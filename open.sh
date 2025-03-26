#!/bin/bash
# Open the demo in a browser

echo "Opening Hat Detection System demo..."
echo ""
echo "This is a simplified demo that shows how the interface works."
echo "It provides a functional UI with the camera controls."
echo ""
echo "NOTE: This demo shows simulated hat detection. For the full experience,"
echo "run the application with the TensorFlow.js backend through 'npm run dev'."
echo ""

# Use xdg-open on Linux to open the file in default browser
xdg-open ./demo.html || open ./demo.html || start ./demo.html || echo "Please open demo.html manually in your browser"