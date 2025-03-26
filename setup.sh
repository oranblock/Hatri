#!/bin/bash
# Setup script to prepare the Hat Detection application

# Copy all project files to the main directory for easier access
cp -r extracted/project/* .

# Make scripts executable
chmod +x *.sh

echo "Hat Detection application is ready to use!"
echo "Run './start.sh' to launch the application"