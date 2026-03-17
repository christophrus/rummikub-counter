# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System-Dependencies für OpenCV (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU für Docker (schlanker als GPU-Version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Python-Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren (inkl. models/rummikub_cnn.pth)
COPY . .

# Port
EXPOSE 8000

# Start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
