#!/usr/bin/env bash
# Copy the shared engine into both deployable bot projects.
set -euo pipefail
cd "$(dirname "$0")"
for bot in vera-signal vera-planner; do
  rm -rf "$bot/vera"
  cp -R shared/vera "$bot/vera"
  find "$bot/vera" -name '__pycache__' -prune -exec rm -rf {} +
done
echo "synced shared/vera -> vera-signal/vera, vera-planner/vera"
