#!/usr/bin/env bash
# post-start.sh — runs every time the dev container starts.
# Checks GitHub CLI auth and prints a one-liner hint if not authenticated.
set -euo pipefail

if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
        echo "✓ gh CLI authenticated as $(gh api user -q .login 2>/dev/null || echo 'unknown')"
    else
        echo "⚠ gh CLI is not authenticated."
        echo "  Option 1 (recommended): Run 'gh auth login' and follow the prompts."
        echo "  Option 2: Export GH_TOKEN on the host before opening the container."
    fi
fi
