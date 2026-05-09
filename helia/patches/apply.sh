#!/usr/bin/env bash
# helia/patches/apply.sh
#
# Idempotently applies every patch in helia/patches/applied/*.patch to the
# working tree. Called automatically by tools/ci_build/build_helia.sh and
# tools/ci_build/test_helia.sh before the make build runs.
#
# Each patch must be a unified diff producible by `git diff` from the repo
# root. Patches are applied in lexical filename order; use a 4-digit ordinal
# prefix (NNNN-<name>.patch) to control ordering.
#
# Idempotency: a patch that is already fully applied is detected via
# `git apply --check --reverse` and skipped silently. A partially-applied
# patch (which means a patch needs an update) aborts the script with a clear
# error.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APPLIED_DIR="${SCRIPT_DIR}/applied"

cd "${ROOT_DIR}"

if [[ ! -d "${APPLIED_DIR}" ]]; then
  exit 0
fi

shopt -s nullglob
patches=( "${APPLIED_DIR}"/*.patch )
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
  exit 0
fi

echo "==> helia/patches: applying ${#patches[@]} patch(es)"

for p in "${patches[@]}"; do
  name="$(basename "${p}")"
  # Already applied?
  if git apply --check --reverse "${p}" >/dev/null 2>&1; then
    echo "    [skip] ${name} (already applied)"
    continue
  fi
  # Cleanly applies?
  if git apply --check "${p}" >/dev/null 2>&1; then
    git apply "${p}"
    echo "    [ok]   ${name}"
    continue
  fi
  echo "    [fail] ${name}" >&2
  echo "" >&2
  echo "    Patch does not apply cleanly and is not already applied." >&2
  echo "    This usually means upstream has changed near the patch hunk." >&2
  echo "    Inspect with:" >&2
  echo "      git apply --3way ${p}" >&2
  echo "    and either rebase the patch or delete it (and remove its entry" >&2
  echo "    from helia/patches/inline_drift.md) if upstream now subsumes it." >&2
  exit 2
done

echo "==> helia/patches: done"
