#!/bin/bash
# Alternative server using Node's http-server
cd "$(dirname "$0")"
echo "Serving built application on http://localhost:8080"
npx http-server dist