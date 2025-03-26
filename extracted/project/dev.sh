#!/bin/bash
# Increase file descriptor limit for this script
ulimit -n 4096 2>/dev/null || echo "Could not increase file descriptor limit"
# Run development server
npx vite --force