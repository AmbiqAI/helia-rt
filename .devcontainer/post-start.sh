#!/usr/bin/env bash
# post-start.sh — runs every time the dev container starts.
# Checks GitHub CLI auth and prints a one-liner hint if not authenticated.
set -euo pipefail

if command -v gh >/dev/null 2>&1; then
    # Capture once: avoids an extra `gh api user` network call on every start
    # (which would also fail offline). Parse the username out of the status
    # output if available; otherwise just confirm auth is configured.
    if status_out="$(gh auth status 2>&1)"; then
        user="$(printf '%s\n' "$status_out" | sed -n 's/.*account \([^ ]*\) .*/\1/p' | head -1)"
        if [ -n "$user" ]; then
            echo "✓ gh CLI authenticated as $user"
        else
            echo "✓ gh CLI authenticated"
        fi
    else
        echo "⚠ gh CLI is not authenticated."
        echo "  Option 1 (recommended): Run 'gh auth login' and follow the prompts."
        echo "  Option 2: Export GH_TOKEN on the host before opening the container."
    fi
fi
