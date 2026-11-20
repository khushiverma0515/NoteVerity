import subprocess
import sys
import time
import os

# Start FastAPI backend in background
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Give backend time to load models (YOLO + ResNet can take ~15 s)
time.sleep(15)

# Start Streamlit frontend
os.execv(sys.executable, [
    sys.executable, "-m", "streamlit", "run", "website/app.py",
    "--server.port", "7860",
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
])
