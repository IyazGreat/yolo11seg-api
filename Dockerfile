# Dockerfile for Render (Docker environment)
# Assumes you run a Python web API (e.g., FastAPI) via Uvicorn.
# Update the CMD line at the bottom if your entrypoint/module is different.

FROM python:3.11-slim

WORKDIR /app

# Minimal OS deps commonly needed for OpenCV / image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Render provides PORT at runtime
ENV PYTHONUNBUFFERED=1

# --- START COMMAND ---
# Default: FastAPI file main.py with app variable named "app"
# If your file is api.py, change to: uvicorn api:app ...
# If your variable is "application", change to: uvicorn main:application ...
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
