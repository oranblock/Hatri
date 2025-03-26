// Fix asset paths in the built index.html
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const indexPath = path.join(__dirname, 'dist', 'index.html');

try {
  // Read the index.html file
  let content = fs.readFileSync(indexPath, 'utf8');
  
  // Replace absolute paths with relative paths
  content = content.replace(/src="\/assets\//g, 'src="assets/');
  content = content.replace(/href="\/assets\//g, 'href="assets/');
  
  // Write the fixed content back
  fs.writeFileSync(indexPath, content);
  
  console.log('Asset paths fixed successfully');
} catch (error) {
  console.error('Error fixing asset paths:', error);
}