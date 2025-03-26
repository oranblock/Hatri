#!/usr/bin/env python3
"""
Simple HTTP Server for the Hat Detection Demo
"""
import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse
import socket

def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return 8765  # Fallback port

# Find a free port
PORT = find_free_port()

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler that serves the demo.html file for root requests"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/' or parsed_path.path == '':
            # Redirect root to demo.html
            self.path = '/demo.html'
        
        return super().do_GET()

def run_server():
    """Start the HTTP server"""
    handler = SimpleHTTPRequestHandler
    
    # Create the server
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        os.system('clear' if os.name == 'posix' else 'cls')
        print("="*70)
        print(f"  Hat Detection Demo Server Running at http://localhost:{PORT}")
        print("="*70)
        print("\nInstructions:")
        print("1. Open a web browser and navigate to the URL above")
        print("2. Use the controls to start the camera and try different settings")
        print("3. Press Ctrl+C in this terminal when you're done to stop the server")
        print("\nNOTE: This is a simplified demo that shows the interface")
        print("      without the full TensorFlow.js functionality.")
        print("="*70)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    run_server()