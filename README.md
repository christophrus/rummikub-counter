# 🎲 Rummikub Stein-Erkennung

Eine Web-App, die Rummikub-Steine auf Fotos erkennt und deren Punktzahl berechnet.
Nutzt **Deep Learning (EasyOCR)** für die Zahlenerkennung und **OpenCV** für die Stein-Segmentierung.

![Tech Stack](https://img.shields.io/badge/React-19-blue)
![Tech Stack](https://img.shields.io/badge/FastAPI-0.115-green)
![Tech Stack](https://img.shields.io/badge/EasyOCR-Deep%20Learning-orange)
![Tech Stack](https://img.shields.io/badge/Docker-Ready-blue)

## 🏗️ Architektur

```
┌──────────────────┐     HTTP/JSON     ┌──────────────────────────┐
│                  │  ◄──────────────► │                          │
│   React Frontend │                   │   FastAPI Backend        │
│   (Vite)         │                   │                          │
│   - Bild-Upload  │                   │   ┌──────────────────┐   │
│   - Ergebnisse   │                   │   │ OpenCV           │   │
│   - Punkte       │                   │   │ Stein-Erkennung  │   │
│                  │                   │   └────────┬─────────┘   │
└──────────────────┘                   │            │             │
                                       │   ┌────────▼─────────┐   │
                                       │   │ EasyOCR          │   │
                                       │   │ CNN + LSTM       │   │
                                       │   │ Zahlenerkennung  │   │
                                       │   └────────┬─────────┘   │
                                       │            │             │
                                       │   ┌────────▼─────────┐   │
                                       │   │ HSV-Analyse      │   │
                                       │   │ Farberkennung    │   │
                                       │   └──────────────────┘   │
                                       └──────────────────────────┘
```

## 🧠 Deep Learning Pipeline

EasyOCR verwendet ein mehrstufiges Deep-Learning-Modell:

1. **Feature Extraction (CNN):** ResNet/VGG extrahiert visuelle Merkmale
2. **Sequence Modeling (BiLSTM):** Bidirektionales LSTM verarbeitet die Feature-Sequenz
3. **Prediction (CTC):** Connectionist Temporal Classification dekodiert den Text

## 🚀 Schnellstart mit Docker

```bash
# Repository klonen
git clone <repo-url>
cd rummiKub-counter

# Mit Docker Compose starten
docker-compose up --build

# App öffnen
# → http://localhost:3000
```

## 💻 Lokale Entwicklung

### Backend

```bash
cd backend

# Virtual Environment erstellen
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Dependencies installieren
pip install -r requirements.txt

# Backend starten
uvicorn app.main:app --reload --port 8000
```

> ⚠️ Beim ersten Start wird das EasyOCR-Modell heruntergeladen (~100 MB).

### Frontend

```bash
cd frontend

# Dependencies installieren
npm install

# Dev-Server starten
npm run dev
```

Die App ist dann unter **http://localhost:5173** erreichbar.

## 📡 API-Endpunkte

| Methode | Pfad              | Beschreibung                        |
|---------|-------------------|-------------------------------------|
| `POST`  | `/api/analyze`       | Bild analysieren → Steine + Punkte  |
| `POST`  | `/api/analyze-debug` | Debug-Bild mit Markierungen         |
| `GET`   | `/health`            | Health Check                        |
| `GET`   | `/docs`              | Swagger UI (API-Dokumentation)      |

### Beispiel: Bild analysieren

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@mein_foto.jpg"
```

### Antwort

```json
{
  "tiles": [
    {"number": 7, "color": "rot", "confidence": 0.95, "is_joker": false},
    {"number": 12, "color": "blau", "confidence": 0.88, "is_joker": false},
    {"number": null, "color": null, "confidence": 0.80, "is_joker": true}
  ],
  "total_score": 49,
  "tile_count": 3,
  "processing_time_ms": 1234.56
}
```

## 📁 Projektstruktur

```
rummiKub-counter/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py                 # FastAPI App + CORS
│       ├── routers/
│       │   └── analyze.py          # API-Endpunkte
│       ├── services/
│       │   ├── tile_detector.py    # OpenCV Stein-Segmentierung
│       │   ├── ocr_service.py      # EasyOCR Zahlenerkennung
│       │   └── color_detector.py   # HSV Farberkennung
│       ├── models/
│       │   └── schemas.py          # Pydantic Datenmodelle
│       └── utils/
│           └── image_processing.py # Bildvorverarbeitung
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/
│       ├── App.jsx                 # Hauptkomponente
│       ├── components/
│       │   ├── ImageUpload.jsx     # Drag & Drop Upload
│       │   ├── ResultDisplay.jsx   # Ergebnis-Anzeige
│       │   └── TileCard.jsx        # Einzelner Stein
│       └── services/
│           └── api.js              # API-Client
└── README.md
```

## 📸 Tipps für beste Erkennung

- **Gute Beleuchtung** – Gleichmäßiges Licht, keine harten Schatten
- **Draufsicht** – Kamera direkt von oben auf die Steine richten
- **Hintergrund** – Einfarbiger, dunkler Hintergrund hilft bei der Segmentierung
- **Abstände** – Steine mit etwas Abstand zueinander legen
- **Schärfe** – Scharfes Foto, kein Verwackeln

## 🛠️ Technologien

- **Frontend:** React 19, Vite 6, Axios
- **Backend:** Python 3.11, FastAPI, Uvicorn
- **KI/ML:** EasyOCR (PyTorch, CNN + LSTM), OpenCV
- **Deployment:** Docker, Docker Compose, Nginx, Caddy

## 🌐 VPS-Deployment (Produktion)

### Voraussetzungen

- VPS mit mind. **2 GB RAM** (PyTorch + YOLO brauchen Speicher)
- Docker und Docker Compose installiert
- Eine Domain, die auf die VPS-IP zeigt (A-Record)

### 1. Repository auf den VPS klonen

```bash
ssh user@dein-server
git clone <repo-url>
cd rummiKub-counter
```

### 2. Umgebungsvariablen konfigurieren

```bash
cp .env.example .env
nano .env
```

Die `DOMAIN` auf deine echte Domain setzen (z.B. `rummikub.meinedomain.de`).  
Caddy holt sich automatisch ein Let's Encrypt SSL-Zertifikat.

### 3. Deployment starten

```bash
chmod +x deploy.sh
./deploy.sh
```

Oder manuell:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### 4. Überprüfen

```bash
# Container-Status
docker compose -f docker-compose.prod.yml ps

# Logs ansehen
docker compose -f docker-compose.prod.yml logs -f

# Health-Check
curl https://deine-domain.de/health
```

### Architektur in Produktion

```
Internet → Caddy (HTTPS/443) → Nginx (Frontend + /api Proxy) → FastAPI Backend
```

- **Caddy** terminiert SSL (automatisches Let's Encrypt)
- **Nginx** liefert die React-SPA und proxied `/api/` zum Backend
- **FastAPI** verarbeitet die Bilderkennung

### Updaten

```bash
cd rummiKub-counter
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

## 📝 Lizenz

MIT
