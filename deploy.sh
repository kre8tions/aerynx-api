#!/bin/bash
set -e

TIER="${1:-}"
if [ -z "$TIER" ]; then
    if systemctl is-active --quiet aerynx-cpu-lite; then
        TIER="cpu-lite"
    elif systemctl is-active --quiet aerynx@cpu-plus; then
        TIER="cpu-plus"
    elif systemctl is-active --quiet aerynx@gpu; then
        TIER="gpu"
    else
        echo "ERROR: Could not detect tier. Pass it explicitly: bash deploy.sh [cpu-lite|cpu-plus|gpu]"
        exit 1
    fi
fi

case "$TIER" in
    cpu-lite) SERVICE="aerynx-cpu-lite" ;;
    cpu-plus) SERVICE="aerynx@cpu-plus" ;;
    gpu)      SERVICE="aerynx@gpu" ;;
    *) echo "ERROR: Unknown tier '$TIER'. Use cpu-lite, cpu-plus, or gpu."; exit 1 ;;
esac

echo "=== Aerynx deploy — tier: $TIER ==="

cd /home/ubuntu/aerynx-api
git config --global --add safe.directory /home/ubuntu/aerynx-api 2>/dev/null || true
git fetch origin
git reset --hard origin/main
echo "Code updated to: $(git log --oneline -1)"

PATCH="/home/ubuntu/aerynx-api/patch_personality.py"
if [ -f "$PATCH" ]; then
    echo "Applying personality patch..."
    python3 "$PATCH" && echo "Personality patch applied." || echo "WARN: personality patch failed (non-fatal)"
else
    echo "WARN: patch_personality.py not found — skipping"
fi

echo "Restarting $SERVICE ..."
sudo systemctl restart "$SERVICE"
sleep 2
sudo systemctl status "$SERVICE" --no-pager | head -12

echo ""
echo "=== Done ==="
