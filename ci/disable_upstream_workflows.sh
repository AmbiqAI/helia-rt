#!/usr/bin/env bash
# Disable upstream-only GitHub Actions workflows that we never want
# firing in this fork.
#
# Why this script (vs editing the workflow YAML files directly):
#   The workflows listed below are vendored from upstream tflm/main and
#   are refreshed verbatim on every upstream replant (see
#   `feat(helia): replant ...` PRs). Editing their `on:` blocks creates
#   merge conflicts on every sync. The GitHub API "disable" state is
#   server-side, persists across pushes, and survives upstream syncs --
#   so disabling at the API level keeps blast radius zero.
#
#   This script is idempotent: re-run it any time after an upstream
#   sync to re-disable anything a refresh might have re-activated.
#
# Requires: gh CLI authenticated with `repo` scope (or a workflow_token
#           with `actions:write`).
#
# Usage:
#   ./ci/disable_upstream_workflows.sh                    # AmbiqAI/helia-rt
#   ./ci/disable_upstream_workflows.sh owner/repo         # other fork
set -euo pipefail

REPO="${1:-AmbiqAI/helia-rt}"

# List of upstream-only workflow file paths (relative to repo root) that
# we never want firing. Anything firing on schedule, pull_request_target,
# or workflow_dispatch from this list should land here.
#
# Keep this list alphabetised to ease auditing on future syncs.
UPSTREAM_ONLY=(
  ".github/workflows/cortex_m.yml"
  ".github/workflows/cortex_m_arm_compiler.yml"
  ".github/workflows/cortex_m_virtual_hardware.yml"
  ".github/workflows/log_binary_size_pr.yml"
  ".github/workflows/run_ci.yml"
  ".github/workflows/stale_handler.yml"
  ".github/workflows/sync.yml"
)

echo "Disabling upstream-only workflows in ${REPO}..."

# Fetch one snapshot of all workflows up front to avoid one API call per
# workflow.
mapfile -t WORKFLOWS < <(
  gh api --paginate "repos/${REPO}/actions/workflows" \
    --jq '.workflows[] | "\(.id)\t\(.state)\t\(.path)"'
)

declare -i changed=0 already=0 missing=0

for path in "${UPSTREAM_ONLY[@]}"; do
  match=""
  for row in "${WORKFLOWS[@]}"; do
    IFS=$'\t' read -r id state wf_path <<<"${row}"
    if [[ "${wf_path}" == "${path}" ]]; then
      match="${id}|${state}"
      break
    fi
  done

  if [[ -z "${match}" ]]; then
    echo "  [skip] ${path}  (not found)"
    missing+=1
    continue
  fi

  id="${match%%|*}"
  state="${match##*|}"

  if [[ "${state}" == "disabled_manually" ]]; then
    echo "  [ok]   ${path}  (already disabled)"
    already+=1
    continue
  fi

  echo "  [fix]  ${path}  (was: ${state}) -> disabling"
  gh api --silent --method PUT \
    "repos/${REPO}/actions/workflows/${id}/disable"
  changed+=1
done

echo
echo "Summary: ${changed} disabled, ${already} already-disabled, ${missing} missing"
