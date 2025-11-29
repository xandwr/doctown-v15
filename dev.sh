#!/bin/bash
# Launch Doctown for demo/testing

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "Building frontend..."
cd "$PROJECT_DIR/doctown_website" && npm run build

echo ""
echo "================================"
echo "  Doctown running at:"
echo "  http://localhost:8000"
echo "  http://$LOCAL_IP:8000"
echo "================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PROJECT_DIR"
uv run docpack web --host 0.0.0.0 --port 8000 --no-browser
