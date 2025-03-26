#!/bin/bash
# Main entry point for the Hat Detection application

# Kill any existing Node.js or Python servers
pkill -f "node" || true
pkill -f "python" || true

echo "Hat Detection System"
echo "===================="
echo "1) Run in development mode (with hot reloading)"
echo "2) Run in production mode (recommended)"
echo

read -p "Enter your choice (1-2): " choice

case $choice in
  1)
    echo "Starting development server..."
    ./dev.sh
    ;;
  2)
    echo "Starting production server..."
    ./serve.sh
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac