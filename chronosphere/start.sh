#!/bin/bash

echo "
╔═══════════════════════════════════════════╗
║          CHRONOSPHERE v1.0                ║
║      Historical Discovery Engine           ║
║           by Aeryn & Claude                ║
╚═══════════════════════════════════════════╝
"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the server
echo "🚀 Starting ChronoSphere..."
node src/backend/server.js
