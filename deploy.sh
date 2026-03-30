#!/bin/bash
TIER="${1:-}"
if [ -z "$TIER" ]; then
    if systemctl is-active --quiet aerynx-cpu-lite; then TIER="cpu-lite"
    elif systemctl is-active --quiet aerynx@cpu-plus; then TIER="cpu-plus"
    elif systemctl is-active --quiet aerynx@gpu; then TIER="gpu"
    else echo "ERROR: pass tier explicitly: bash deploy.sh [cpu-lite|cpu-plus|gpu]"; exit 1
    fi
fi
case "$TIER" in
    cpu-lite) SERVICE="aerynx-cpu-lite" ;;
    cpu-plus) SERVICE="aerynx@cpu-plus" ;;
    gpu)      SERVICE="aerynx@gpu" ;;
    *) echo "Unknown tier"; exit 1 ;;
esac
echo "=== Deploying $TIER ==="
cd /home/ubuntu/aerynx-api
git config --global --add safe.directory /home/ubuntu/aerynx-api 2>/dev/null || true
git fetch origin && git reset --hard origin/main
echo "Updated to: $(git log --oneline -1)"
sudo systemctl restart "$SERVICE" && sleep 2
sudo systemctl status "$SERVICE" --no-pager | head -10
echo "=== Done ==="
