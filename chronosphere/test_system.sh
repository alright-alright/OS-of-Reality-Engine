#!/bin/bash

# ChronoSphere Production Test Script
# Tests all real components are working

echo "╔═══════════════════════════════════════════╗"
echo "║     CHRONOSPHERE SYSTEM TEST              ║"
echo "║     Testing Real Components                ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from example..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys"
    echo ""
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/atlas
mkdir -p data/mpu
mkdir -p data/akasha

# Install dependencies if needed
if [ ! -d node_modules ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Test Wikidata connection
echo ""
echo "🔍 Testing Wikidata SPARQL endpoint..."
curl -s -X POST https://query.wikidata.org/sparql \
  -H "Content-Type: application/sparql-query" \
  -H "Accept: application/json" \
  --data-raw 'SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1' \
  > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Wikidata connection successful"
else
    echo "❌ Wikidata connection failed"
fi

# Check Atlas Framework
echo ""
echo "🔍 Checking Atlas Framework components..."
if [ -d ../atlas-framework/packages ]; then
    echo "✅ Atlas Framework found"
    ls -d ../atlas-framework/packages/*/ | head -5 | while read dir; do
        echo "   - $(basename $dir)"
    done
else
    echo "⚠️  Atlas Framework not found at ../atlas-framework"
fi

# Check LucianOS core
echo ""
echo "🔍 Checking LucianOS-core..."
if [ -d ../lucianos-core/protocols ]; then
    echo "✅ LucianOS-core found"
    ls -d ../lucianos-core/protocols/*/ 2>/dev/null | while read dir; do
        echo "   - $(basename $dir) protocol"
    done
else
    echo "⚠️  LucianOS-core not found at ../lucianos-core"
fi

# Start the server
echo ""
echo "🚀 Starting ChronoSphere server..."
echo "   API: http://localhost:8000"
echo "   WebSocket: ws://localhost:8001"
echo "   Frontend: Open src/frontend/index.html in browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start server with environment variables
node src/backend/server.js