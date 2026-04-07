#!/bin/bash
# Start FastAPI in background
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to be ready
echo "Waiting for FastAPI to start..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
        echo "FastAPI is ready!"
        break
    fi
    sleep 1
done

# Start Dash on port 7860 (HF Spaces expects this port)
python -c "
import sys, os
sys.path.insert(0, '/app')
os.chdir('/app')
from dashboards.dash_app.app import app
app.run(host='0.0.0.0', port=7860, debug=False)
"
