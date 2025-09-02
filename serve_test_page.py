#!/usr/bin/env python3
"""
Simple HTTP Server for Email Generator Test Page

This script serves the frontend UI files on a local HTTP server
to avoid CORS issues when opening the file directly in a browser.

Usage:
    python serve_test_page.py

Then open: http://localhost:3000
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 3001  # Changed to 3001 to avoid conflict with Docker frontend (port 3000)
HOST = 'localhost'

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve the test page and handle CORS"""
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '':
            # Serve the main UI page as the root
            self.path = '/ui/index.html'
        elif self.path == '/history.html':
            # Direct access to history page
            self.path = '/ui/history.html'
        elif self.path == '/index.html':
            # Direct access to index page
            self.path = '/ui/index.html'
        
        return super().do_GET()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_response(200)
        self.end_headers()

def main():
    """Start the HTTP server"""
    
    # Change to the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if the UI files exist
    ui_dir = project_root / 'ui'
    index_file = ui_dir / 'index.html'
    if not index_file.exists():
        print(f"Error: ui/index.html not found in {project_root}")
        print("Make sure you're running this script from the project root directory.")
        return
    
    # Create the server
    try:
        with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
            print("Email Generator Test Page Server")
            print("=" * 50)
            print(f"Server running at: http://{HOST}:{PORT}")
            print(f"Serving from: {project_root}")
            print(f"Main page URL: http://{HOST}:{PORT}")
            print(f"History page URL: http://{HOST}:{PORT}/ui/history.html")
            print()
            print("Instructions:")
            print("1. Make sure your FastAPI server is running on http://localhost:8000")
            print("   - Docker: make up-alpine (API will be available)")
            print("   - Local: uvicorn src.main:app --reload")
            print(f"2. Open http://localhost:{PORT} in your browser")
            print("3. Upload a CSV file and watch real-time email generation!")
            print()
            print("Note: This Python server runs on port 3001 to avoid conflict with Docker frontend (port 3000)")
            print("Press Ctrl+C to stop the server")
            print("=" * 50)
            
            # Optionally open browser automatically
            try:
                webbrowser.open(f'http://{HOST}:{PORT}')
                print(f"Opened browser to http://{HOST}:{PORT}")
            except Exception:
                print("Could not auto-open browser. Please open manually.")
            
            # Start serving
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Error: Port {PORT} is already in use.")
            print("Try stopping other servers or choose a different port.")
        else:
            print(f"Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n[!] Server stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
