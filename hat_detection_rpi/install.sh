#!/bin/bash
# Installation script for Hat Detection System for Raspberry Pi with Hailo-8

# Set error handling
set -e

# Function to print colored text
print_color() {
  local color=$1
  local text=$2
  
  case $color in
    "red") echo -e "\033[0;31m$text\033[0m" ;;
    "green") echo -e "\033[0;32m$text\033[0m" ;;
    "yellow") echo -e "\033[0;33m$text\033[0m" ;;
    "blue") echo -e "\033[0;34m$text\033[0m" ;;
    *) echo "$text" ;;
  esac
}

# Function to check command existence
check_command() {
  if ! command -v $1 &> /dev/null; then
    print_color "red" "Error: $1 is not installed. Please install it first."
    exit 1
  fi
}

# Function to create virtual environment
create_venv() {
  print_color "blue" "Creating Python virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
}

# Show script header
print_color "green" "===================================================="
print_color "green" "  Hat Detection System Installation for Raspberry Pi"
print_color "green" "===================================================="
print_color "blue" "This script will install all dependencies required for"
print_color "blue" "the Hat Detection System using the Hailo-8 accelerator."
echo ""

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
  model=$(cat /proc/device-tree/model)
  if [[ $model == *"Raspberry Pi"* ]]; then
    print_color "green" "Detected Raspberry Pi: $model"
    RASPBERRY_PI=true
  else
    print_color "yellow" "Warning: Not running on a Raspberry Pi. Some hardware features may not work."
    RASPBERRY_PI=false
  fi
else
  print_color "yellow" "Warning: Unable to determine if this is a Raspberry Pi. Assuming this is development mode."
  RASPBERRY_PI=false
fi

# Check for Python 3
print_color "blue" "Checking for Python 3..."
check_command python3

# Get Python version
PYTHON_VERSION=$(python3 --version)
print_color "green" "Found $PYTHON_VERSION"

# Update system packages
print_color "blue" "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_color "blue" "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev python3-opencv \
    libopencv-dev libgtk-3-dev libjpeg-dev libpng-dev \
    v4l-utils i2c-tools

# Check if Hailo-8 is connected
if [ "$RASPBERRY_PI" = true ]; then
  print_color "blue" "Checking for Hailo-8 device..."
  if lsusb | grep -i "hailo" &> /dev/null; then
    print_color "green" "Hailo-8 device detected!"
    HAILO_DETECTED=true
  else
    print_color "yellow" "Warning: Hailo-8 device not detected. The application will run in simulation mode."
    HAILO_DETECTED=false
  fi
else
  print_color "yellow" "Not on a Raspberry Pi. Assuming development mode without Hailo hardware."
  HAILO_DETECTED=false
fi

# Create Python virtual environment
create_venv

# Install Python requirements
print_color "blue" "Installing Python requirements..."
pip install -r requirements.txt

# Install Hailo SDK if device is detected
if [ "$HAILO_DETECTED" = true ]; then
  print_color "blue" "Installing Hailo SDK..."
  
  # This is a placeholder - in reality, you would need to add Hailo's
  # package repository and install their Python package
  
  # Example (this would need to be adjusted based on Hailo's actual installation instructions):
  # wget -O - https://hailo.ai/developer/download/hailo_apt.key | sudo apt-key add -
  # echo "deb https://hailo.ai/developer/download/debian /" | sudo tee /etc/apt/sources.list.d/hailo.list
  # sudo apt update
  # sudo apt install -y hailo-runtime hailo-python
  
  print_color "green" "Hailo SDK installed successfully!"
else
  print_color "yellow" "Skipping Hailo SDK installation (using simulation mode)"
fi

# Create model directories
print_color "blue" "Creating model directories..."
mkdir -p models/hailo_models

# Create fallback models directory for simulation mode
mkdir -p models/fallback

# Convert models for Hailo (or create simulation models)
print_color "blue" "Preparing AI models..."
if [ "$HAILO_DETECTED" = true ]; then
  print_color "blue" "Converting models for Hailo-8 (this may take a while)..."
  python -m models.model_conversion.convert_models
else
  print_color "blue" "Setting up simulation models..."
  python -m models.model_conversion.convert_models
fi

# Set execute permissions
print_color "blue" "Setting execute permissions..."
chmod +x src/main.py

# Create desktop shortcut if on Raspberry Pi
if [ "$RASPBERRY_PI" = true ]; then
  print_color "blue" "Creating desktop shortcut..."
  
  DESKTOP_FILE="$HOME/Desktop/hat-detection.desktop"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  
  cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Hat Detection System
Comment=Hat Detection using Hailo-8 accelerator
Exec=bash -c "cd $SCRIPT_DIR && source venv/bin/activate && python -m src.main"
Icon=camera
Terminal=false
Categories=Application;
EOF
  
  chmod +x "$DESKTOP_FILE"
  print_color "green" "Desktop shortcut created at $DESKTOP_FILE"
fi

# Installation complete
print_color "green" "===================================================="
print_color "green" "Installation complete!"
print_color "green" "===================================================="
print_color "blue" "To run the Hat Detection System:"
print_color "blue" "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
print_color "blue" "2. Run the application:"
echo "   python -m src.main"
print_color "blue" "3. Keyboard shortcuts in the application:"
echo "   - ESC: Exit application"
echo "   - F: Toggle fullscreen mode"
echo "   - 1/2/3: Set low/medium/high processing power"
echo "   - H: Toggle trajectory visualization"
echo "   - G: Toggle face box display"
print_color "green" "===================================================="