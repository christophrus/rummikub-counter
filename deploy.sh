#!/usr/bin/env bash
set -euo pipefail

# ─── Rummikub Counter – VPS Deployment Script ───
# Voraussetzung: Docker + Docker Compose auf dem VPS installiert

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# .env prüfen
if [ ! -f .env ]; then
    echo "❌ .env Datei fehlt! Kopiere .env.example und passe sie an:"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

# .env laden für Anzeige
source .env

echo "🚀 Deploying Rummikub Counter..."
echo "   Domain: ${DOMAIN:-localhost}"
echo ""

# Images bauen und Container starten
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "✅ Deployment fertig!"
echo ""
echo "   Frontend: https://${DOMAIN:-localhost}"
echo "   API Docs: https://${DOMAIN:-localhost}/api/docs (über Nginx Proxy)"
echo ""
echo "   Logs ansehen:    docker compose -f docker-compose.prod.yml logs -f"
echo "   Stoppen:         docker compose -f docker-compose.prod.yml down"
