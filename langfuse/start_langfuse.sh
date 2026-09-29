#!/bin/bash
# Start self-hosted LangFuse — run with: sudo bash start_langfuse.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting LangFuse (self-hosted)..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d

echo ""
echo "Waiting for LangFuse to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:3000/api/public/health | grep -q "ok"; then
        echo ""
        echo "LangFuse is ready!"
        echo ""
        echo "  UI      : http://localhost:3000"
        echo "  Email   : suganya.selvaraj04@infosys.com"
        echo "  Password: itops1234"
        echo "  Project : learningloop_agenticai"
        echo ""
        break
    fi
    printf "."
    sleep 2
done
