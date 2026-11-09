#!/usr/bin/env bash
# Rebuild all graphviz figures as svg.
set -euo pipefail
cd "$(dirname "$0")"
for f in *.dot; do
  dot -Tsvg "$f" -o "${f%.dot}.svg"
  echo "wrote ${f%.dot}.svg"
done
