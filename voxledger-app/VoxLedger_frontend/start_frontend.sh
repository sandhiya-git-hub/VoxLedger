#!/bin/bash
# VoxLedger Frontend Startup Script (Linux/macOS)

set -e

cd "$(dirname "$0")"

echo "=============================="
echo "  VoxLedger Frontend Startup"
echo "=============================="

# Install npm dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

echo ""
echo "Starting VoxLedger Frontend on http://localhost:5173"
echo ""
echo "Make sure the backend is running on http://localhost:8000"
echo ""

npm run dev
