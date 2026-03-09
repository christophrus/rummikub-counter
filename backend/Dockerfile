# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System-Dependencies für OpenCV (headless) und EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python-Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EasyOCR-Modell beim Build vorladen (spart Zeit beim Start)
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# App-Code kopieren
COPY . .

# Port
EXPOSE 8000

# Start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
