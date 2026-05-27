#!/usr/bin/env bash

set -euo pipefail

# File extension to restrict
EXTENSION="parquet"

# Allowed directory
ALLOWED_DIR="tests/fixtures/"

# Get staged files matching extension
FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep "\.${EXTENSION}$" || true)

if [ -z "$FILES" ]; then
  exit 0
fi

INVALID_FILES=()

while IFS= read -r file; do
  if [[ "$file" != ${ALLOWED_DIR}* ]]; then
    INVALID_FILES+=("$file")
  fi
done <<< "$FILES"

if [ ${#INVALID_FILES[@]} -gt 0 ]; then
  echo "Error: .${EXTENSION} files are only allowed in ${ALLOWED_DIR}"
  echo
  echo "Invalid files:"
  for file in "${INVALID_FILES[@]}"; do
    echo "  - $file"
  done
  exit 1
fi
