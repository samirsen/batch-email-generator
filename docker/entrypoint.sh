#!/bin/sh
set -eu

API_BASE_URL=${API_BASE_URL:-http://localhost:8000}
echo "Configuring frontend with API_BASE_URL: $API_BASE_URL"

# Replace placeholder or default value across HTML/JS
find /usr/share/nginx/html -type f \( -name '*.html' -o -name '*.js' \) -print0 | while IFS= read -r -d '' f; do
  if grep -q "__API_BASE_URL__" "$f"; then
    sed -i "s|__API_BASE_URL__|$API_BASE_URL|g" "$f"
  else
    sed -i "s|const API_BASE_URL = 'http://localhost:8000';|const API_BASE_URL = '$API_BASE_URL';|g" "$f" || true
  fi
done

exec nginx -g 'daemon off;'
