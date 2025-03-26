// Simple HTTP server to serve the application
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Try different ports
async function findAvailablePort(startPort) {
  for (let port = startPort; port < startPort + 10; port++) {
    try {
      const server = http.createServer();
      
      // Use a promise to check if port is available
      const available = await new Promise((resolve) => {
        server.once('error', (err) => {
          if (err.code === 'EADDRINUSE') {
            resolve(false);
          }
        });
        
        server.once('listening', () => {
          server.close();
          resolve(true);
        });
        
        server.listen(port);
      });
      
      if (available) {
        return port;
      }
    } catch (e) {
      continue;
    }
  }
  
  // If no ports are available, return a fallback
  return 8765;
}

// Find a port and start the server
async function startServer() {
  const PORT = await findAvailablePort(3000);
  const DIST_DIR = path.join(__dirname, 'dist');

  // MIME types for different file extensions
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav',
    '.mp4': 'video/mp4',
    '.woff': 'application/font-woff',
    '.ttf': 'application/font-ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.otf': 'application/font-otf',
    '.wasm': 'application/wasm'
  };

  const server = http.createServer((req, res) => {
    console.log(`Request: ${req.url}`);
    
    // Handle root request and SPA routes
    if (req.url === '/' || !req.url.includes('.')) {
      fs.readFile(path.join(DIST_DIR, 'index.html'), (err, content) => {
        if (err) {
          res.writeHead(500);
          res.end(`Server Error: ${err.code}`);
          return;
        }
        
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(content, 'utf-8');
      });
      return;
    }
    
    // Handle asset requests
    let filePath = path.join(DIST_DIR, req.url);
    
    // Get file extension
    const extname = String(path.extname(filePath)).toLowerCase();
    const contentType = mimeTypes[extname] || 'application/octet-stream';
    
    // Read the file
    fs.readFile(filePath, (error, content) => {
      if (error) {
        if (error.code === 'ENOENT') {
          // File not found - try index.html for SPA routes
          console.log(`File not found: ${filePath}, serving index.html instead`);
          fs.readFile(path.join(DIST_DIR, 'index.html'), (err, indexContent) => {
            if (err) {
              res.writeHead(404);
              res.end('404 - File Not Found');
              return;
            }
            
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(indexContent, 'utf-8');
          });
        } else {
          // Server error
          res.writeHead(500);
          res.end(`Server Error: ${error.code}`);
        }
      } else {
        // Success - add caching headers
        const headers = { 
          'Content-Type': contentType,
          'Cache-Control': 'max-age=86400' // Cache for 1 day
        };
        res.writeHead(200, headers);
        res.end(content, 'utf-8');
      }
    });
  });

  server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
    console.log(`Serving files from: ${DIST_DIR}`);
    console.log(`Open your browser to access the Hat Detection System`);
  });
}

// Start the server
startServer();