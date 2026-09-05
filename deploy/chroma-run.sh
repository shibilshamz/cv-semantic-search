#!/usr/bin/env bash
# Start Chroma without Compose.
#
# docker-compose.yml is the portable path and works anywhere with Compose v2.
# This script exists because the target VPS has neither: Ubuntu's docker.io
# package ships no v2 plugin, and the v1 python package installed there fails
# against this Docker version with "Not supported URL scheme http+docker" (a
# requests/urllib3 incompatibility, not a config problem).
#
# Installing a compose plugin onto a box that serves a paying client's pipeline
# is a bigger change than this needs. A bare `docker run` is also how n8n is
# already started on this host, so this matches local practice.
#
# Keep the flags here in sync with docker-compose.yml.
set -euo pipefail

IMAGE="chromadb/chroma:1.5.9"   # pinned, and matching the client in requirements.txt
NAME="chroma"
DATA="/root/chroma-data"
PORT="127.0.0.1:8001"           # 8000 belongs to trading-agent on this host

mkdir -p "$DATA"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container '$NAME' already exists; starting it"
  docker start "$NAME"
else
  docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    -p "${PORT}:8000" \
    -v "${DATA}:/data" \
    -e IS_PERSISTENT=TRUE \
    -e PERSIST_DIRECTORY=/data \
    -e ANONYMIZED_TELEMETRY=FALSE \
    "$IMAGE"
fi

echo "waiting for heartbeat..."
for _ in $(seq 1 30); do
  if curl -sf -m 3 "http://127.0.0.1:8001/api/v2/heartbeat" >/dev/null; then
    echo "chroma is up on ${PORT}"
    exit 0
  fi
  sleep 2
done

echo "chroma did not answer within 60s; check: docker logs $NAME" >&2
exit 1
