#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/admin/Guangxi-autodata-prediect}"
APP_USER="${APP_USER:-admin}"
BRANCH="${BRANCH:-main}"
SERVICE_NAME="${SERVICE_NAME:-guangxi-power}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

run_in_app_dir() {
  local command="$1"
  if [ "$(id -u)" -eq 0 ]; then
    runuser -u "$APP_USER" -- bash -lc "cd '$APP_DIR' && $command"
  else
    bash -lc "cd '$APP_DIR' && $command"
  fi
}

if [ ! -d "$APP_DIR/.git" ]; then
  echo "Repository not found: $APP_DIR" >&2
  exit 1
fi

echo "Checking GitHub updates for $APP_DIR on branch $BRANCH..."
run_in_app_dir "git fetch origin '$BRANCH'"

LOCAL_COMMIT="$(run_in_app_dir "git rev-parse HEAD")"
REMOTE_COMMIT="$(run_in_app_dir "git rev-parse 'origin/$BRANCH'")"

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
  echo "Already up to date: $LOCAL_COMMIT"
  exit 0
fi

echo "Updating $LOCAL_COMMIT -> $REMOTE_COMMIT"
run_in_app_dir "git pull --ff-only origin '$BRANCH'"
run_in_app_dir ".venv/bin/pip install -r backend/requirements.txt -i '$PIP_INDEX_URL' --trusted-host '$PIP_TRUSTED_HOST'"

if command -v systemctl >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    systemctl restart "$SERVICE_NAME"
  else
    sudo systemctl restart "$SERVICE_NAME"
  fi
  echo "Restarted service: $SERVICE_NAME"
else
  echo "systemctl not found; please restart the app manually." >&2
fi
