#!/usr/bin/env bash
# Copyright 2026 Ambiq
# SPDX-License-Identifier: Apache-2.0
#
# run_helia_ci_local.sh — run the helia_test.yml PR gate on this machine.
#
# Runs the image-independent source checks, then every matrix leg
# (arch x toolchain x SPEED/SIZE) of helia_test.yml through test_helia.sh,
# in parallel, inside the CI image pinned in helia_test.yml, with each leg's
# executed-case floors read from that workflow. The working tree is tested as
# it stands, uncommitted and untracked files included.
#
# Usage:
#   run_helia_ci_local.sh [options]
#
# Options:
#   --src <dir>        Checkout to test (default: the one holding this script).
#   --legs <text>      Run only units whose "<arch>-<toolchain>-<opt>" name
#                      contains <text> (e.g. 'cortex-m4+fp', 'atfe-SPEED').
#   --opt <SPEED|SIZE|BOTH>
#                      Kernel optimization profiles (default: BOTH).
#   --parallel <n>     Units run at once (default: 2).
#   --cpus <n>         CPUs shared by the running units (default: half of
#                      nproc). Each unit gets cpus/parallel, and its make -j
#                      follows: test_helia.sh sizes -j from nproc, which
#                      honours OMP_NUM_THREADS.
#   --mem <size>       Memory cap per unit in the image, in docker syntax
#                      (default: 8g). A unit over it fails on its own instead
#                      of pressuring the host.
#   --no-floors        Do not pass the workflow's executed-case floors.
#   --skip-checks      Skip the source checks.
#   --native           Use the host toolchains and FVP instead of the image:
#                      ATFE_PREINSTALL and an FVP_Corstone_SSE-300_Ethos-U55
#                      on PATH must be set up by the caller.
#   --list             Print the selected units and exit.
#   -h, --help         Show this help.
#
# Environment:
#   HELIA_LOCAL_CI_DIR  Cache root for per-unit trees, shared downloads,
#                       ccache and logs (default: ~/.cache/helia-rt-local-ci).
#
# Exit status is 0 only when every source check and every unit passes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CACHE_DIR="${HELIA_LOCAL_CI_DIR:-${HOME}/.cache/helia-rt-local-ci}"

LEGS_RE=""
OPT_CHOICE="BOTH"
PARALLEL=2
CPUS=$(( $(nproc) / 2 ))
MEM=8g
USE_FLOORS=1
RUN_CHECKS=1
NATIVE=0
LIST_ONLY=0

usage() { sed -n '/^# Usage:/,/^# Exit status/p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) ROOT_DIR="$(cd "${2:?--src needs a value}" && pwd)"; shift 2 ;;
    --legs) LEGS_RE="${2:?--legs needs a value}"; shift 2 ;;
    --opt) OPT_CHOICE="${2:?--opt needs a value}"; shift 2 ;;
    --parallel) PARALLEL="${2:?--parallel needs a value}"; shift 2 ;;
    --cpus) CPUS="${2:?--cpus needs a value}"; shift 2 ;;
    --mem) MEM="${2:?--mem needs a value}"; shift 2 ;;
    --no-floors) USE_FLOORS=0; shift ;;
    --skip-checks) RUN_CHECKS=0; shift ;;
    --native) NATIVE=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

WORKFLOW="${ROOT_DIR}/.github/workflows/helia_test.yml"
[[ -f "${WORKFLOW}" ]] || { echo "No ${WORKFLOW}" >&2; exit 2; }

case "${OPT_CHOICE}" in
  SPEED|SIZE) OPTS=("${OPT_CHOICE}") ;;
  BOTH) OPTS=(SPEED SIZE) ;;
  *) echo "Invalid --opt '${OPT_CHOICE}'." >&2; exit 2 ;;
esac

# The matrix, its excludes and its floors come from helia_test.yml so this
# script cannot drift from the gate. Prints "arch toolchain min_bins min_cases"
# per leg; a leg without floors prints "-" for both.
read_matrix() {
  python3 -I - "${WORKFLOW}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"^\s+matrix:\n(.*?)^\s+include:\n(.*?)(?=^\S|^\s{0,4}\S)", text,
              re.S | re.M)
if not m:
    sys.exit("helia_test.yml: matrix block not found")
head, include = m.group(1), m.group(2)
def items(key):
    v = re.search(r"^\s+%s:\s*\[(.*?)\]" % key, head, re.M)
    if not v:
        sys.exit("helia_test.yml: matrix %s not found" % key)
    return [s.strip().strip('"') for s in v.group(1).split(",")]
excludes = set(re.findall(
    r'-\s*arch:\s*[\'"]?([^\'"\n]+?)[\'"]?\s*\n\s*toolchain:\s*[\'"]?([^\'"\n]+?)[\'"]?\s*$',
    head.split("exclude:", 1)[1] if "exclude:" in head else "", re.M))
floors = {}
for entry in re.split(r"\n\s*-\s*(?=arch:)", "\n" + include)[1:]:
    get = lambda k: (re.search(r'%s:\s*"?([^"\n]+)"?' % k, entry) or [None, "-"])[1]
    floors[(get("arch"), get("toolchain"))] = (get("min-binaries"),
                                               get("min-cases"))
for arch in items("arch"):
    for tc in items("toolchain"):
        if (arch, tc) not in excludes:
            print(arch, tc, *floors.get((arch, tc), ("-", "-")))
PY
}

IMAGE="$(grep -oE 'ghcr\.io/ambiqai/helia-rt-ci@sha256:[0-9a-f]{64}' \
  "${WORKFLOW}" | head -1)"
[[ -n "${IMAGE}" ]] || { echo "No pinned CI image in ${WORKFLOW}" >&2; exit 1; }

UNITS=()
while read -r arch tc min_bins min_cases; do
  for opt in "${OPTS[@]}"; do
    name="${arch}-${tc}-${opt}"
    if [[ -z "${LEGS_RE}" || "${name}" == *"${LEGS_RE}"* ]]; then
      UNITS+=("${name} ${arch} ${tc} ${opt} ${min_bins} ${min_cases}")
    fi
  done
done < <(read_matrix)

# A workflow edit the parser does not follow would drop every floor and let a
# run pass with none enforced, so finding no floor at all is an error.
if [[ ${USE_FLOORS} -eq 1 ]] && ! printf '%s\n' "${UNITS[@]}" |
    awk '$5 != "-" && $6 != "-" {found = 1} END {exit !found}'; then
  echo "No executed-case floors parsed from ${WORKFLOW}; pass --no-floors to run anyway." >&2
  exit 2
fi

if [[ ${#UNITS[@]} -eq 0 ]]; then
  echo "No matrix unit matches --legs '${LEGS_RE}'." >&2
  exit 2
fi
if [[ ${LIST_ONLY} -eq 1 ]]; then
  printf '%s\n' "${UNITS[@]}" | awk '{print $1, "floors:", $5, $6}'
  exit 0
fi
[[ ${PARALLEL} -ge 1 && ${CPUS} -ge 1 ]] || { echo "--parallel and --cpus must be >= 1" >&2; exit 2; }
[[ ${PARALLEL} -le ${#UNITS[@]} ]] || PARALLEL=${#UNITS[@]}
UNIT_CPUS=$(( CPUS / PARALLEL ))
[[ ${UNIT_CPUS} -ge 1 ]] || UNIT_CPUS=1

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CACHE_DIR}/logs/${RUN_ID}"
DOWNLOADS="${CACHE_DIR}/downloads"
mkdir -p "${LOG_DIR}" "${DOWNLOADS}" "${CACHE_DIR}/ccache" "${CACHE_DIR}/trees"
echo "==> logs: ${LOG_DIR}"
[[ ${NATIVE} -eq 1 ]] && echo "==> native toolchains" || echo "==> image: ${IMAGE}"

check_status=0
if [[ ${RUN_CHECKS} -eq 1 ]]; then
  # Same list as the ci-image job in helia_test.yml.
  for check in \
      check_helia_kernel_runner_temporaries.sh \
      check_helia_ci_image_pins.sh check_helia_ci_image_pins_test.sh \
      check_helia_cmsis_nn_status_checks_test.sh \
      check_helia_workflow_secrets.sh check_helia_workflow_secrets_test.sh; do
    if (cd "${ROOT_DIR}" && "./tensorflow/lite/micro/tools/ci_build/${check}") \
        > "${LOG_DIR}/check-${check%.sh}.log" 2>&1; then
      echo "PASS  ${check}"
    else
      echo "FAIL  ${check}  (${LOG_DIR}/check-${check%.sh}.log)"
      check_status=1
    fi
  done
fi

# Each unit builds in its own copy of the tree: test_helia.sh runs
# `make clean`, which removes the whole gen/ directory. In the image every
# copy shares one downloads directory, bind-mounted at its in-tree path: the
# download scripts reach patches through ../ from it, so it cannot be a
# symlink. Native units keep their own downloads in their copy.
DOWNLOADS_REL=tensorflow/lite/micro/tools/make/downloads
sync_tree() {
  local tree="$1"
  mkdir -p "${tree}/${DOWNLOADS_REL}"
  rsync -a --delete \
    --exclude=/.git --exclude=/gen --exclude=/astro-site/node_modules \
    --exclude="/${DOWNLOADS_REL}" \
    "${ROOT_DIR}/" "${tree}/"
}

in_unit() {
  local tree="$1"; shift
  if [[ ${NATIVE} -eq 1 ]]; then
    (cd "${tree}" && OMP_NUM_THREADS="${UNIT_CPUS}" nice -n 10 "$@")
  else
    DOCKER_RUN_OPTS="--cpus=${UNIT_CPUS} --memory=${MEM} --memory-swap=${MEM} \
      -e OMP_NUM_THREADS=${UNIT_CPUS} \
      -v ${DOWNLOADS}:/workspace/${DOWNLOADS_REL} \
      -v ${CACHE_DIR}/ccache:/ccache -e CCACHE_DIR=/ccache \
      -e CCACHE_MAXSIZE=20G -e CCACHE_BASEDIR=/workspace \
      -e CCACHE_WRAP_DIR=/tmp/ccwrap -e FVP_TIMEOUT_SECONDS=120 \
      ${UNIT_ENV:-}" \
      "${tree}/tensorflow/lite/micro/tools/ci_build/run_in_docker.sh" \
      --image "${IMAGE}" "$@"
  fi
}

# Downloads run once per toolchain before the units start, so concurrent
# units find them in place instead of racing to unpack the same directory.
# Native units each own their downloads, so each fetches its own.
declare -A fetched=()
for unit in "${UNITS[@]}"; do
  read -r name arch tc opt _ _ <<<"${unit}"
  sync_tree "${CACHE_DIR}/trees/${name}"
  key="${arch}-${tc}"
  [[ ${NATIVE} -eq 1 ]] && key="${name}"
  if [[ -z "${fetched[${key}]:-}" ]]; then
    echo "==> downloads for ${arch}/${tc}"
    atfe_args=()
    [[ "${tc}" == atfe && ${NATIVE} -eq 1 ]] && \
      atfe_args=("TARGET_TOOLCHAIN_ROOT=${ATFE_PREINSTALL:?}/bin/")
    [[ "${tc}" == atfe && ${NATIVE} -eq 0 ]] && \
      atfe_args=("TARGET_TOOLCHAIN_ROOT=/opt/atfe/bin/")
    if ! UNIT_ENV="" in_unit "${CACHE_DIR}/trees/${name}" \
        make -f tensorflow/lite/micro/tools/make/Makefile \
        TARGET=cortex_m_corstone_300 TARGET_ARCH="${arch}" TOOLCHAIN="${tc}" \
        OPTIMIZED_KERNEL_DIR=helia "${atfe_args[@]}" third_party_downloads \
        > "${LOG_DIR}/downloads-${arch}-${tc}.log" 2>&1; then
      echo "FAIL  downloads for ${arch}/${tc} (${LOG_DIR}/downloads-${arch}-${tc}.log)"
      exit 1
    fi
    fetched[${key}]=1
  fi
done

run_unit() {
  local name="$1" arch="$2" tc="$3" opt="$4" min_bins="$5" min_cases="$6"
  local env_opts=""
  if [[ ${USE_FLOORS} -eq 1 && "${min_bins}" != "-" ]]; then
    env_opts="-e HELIA_MIN_TEST_BINARIES=${min_bins} -e HELIA_MIN_EXECUTED_CASES=${min_cases}"
    if [[ ${NATIVE} -eq 1 ]]; then
      export HELIA_MIN_TEST_BINARIES="${min_bins}" HELIA_MIN_EXECUTED_CASES="${min_cases}"
    fi
  fi
  local start=${SECONDS}
  [[ ${NATIVE} -eq 1 ]] || env_opts+=" --name helia-local-ci-${RUN_ID}-${name//+/p}"
  if UNIT_ENV="${env_opts}" in_unit "${CACHE_DIR}/trees/${name}" \
      tensorflow/lite/micro/tools/ci_build/test_helia.sh \
      -a "${arch}" -t "${tc}" -O "${opt}" > "${LOG_DIR}/${name}.log" 2>&1; then
    echo "PASS ${name} $((SECONDS - start))s"
  else
    echo "FAIL ${name} $((SECONDS - start))s"
  fi
}

echo "==> ${#UNITS[@]} units, ${PARALLEL} at a time, ${UNIT_CPUS} CPUs each"

# Stop every unit, and the containers they started, if the run is interrupted.
stop_units() {
  trap - INT TERM
  jobs -p | xargs -r kill 2>/dev/null || true
  if [[ ${NATIVE} -eq 0 ]]; then
    docker ps -q --filter "name=helia-local-ci-${RUN_ID}-" | xargs -r docker kill >/dev/null 2>&1 || true
  fi
  echo "==> interrupted; units stopped. logs: ${LOG_DIR}" >&2
  exit 130
}
trap stop_units INT TERM
results="${LOG_DIR}/results.txt"
: > "${results}"
running=0
for unit in "${UNITS[@]}"; do
  read -r name arch tc opt min_bins min_cases <<<"${unit}"
  run_unit "${name}" "${arch}" "${tc}" "${opt}" "${min_bins}" "${min_cases}" \
    >> "${results}" &
  running=$((running + 1))
  if [[ ${running} -ge ${PARALLEL} ]]; then
    wait -n || true
    running=$((running - 1))
  fi
done
wait || true

echo
echo "unit                          result  time   tally"
status=${check_status}
for unit in "${UNITS[@]}"; do
  read -r name _ <<<"${unit}"
  # Unit names contain '+', so match the field exactly rather than as a regex.
  line="$(awk -v n="${name}" '$2 == n' "${results}")"
  [[ -n "${line}" ]] || line="FAIL ${name} ?"
  read -r verdict _ took <<<"${line}"
  # -a: ATfE logs carry non-text bytes, which turn grep -o silent.
  tally="$(grep -aoE '[0-9]+ binaries .*, [0-9]+ test cases' \
    "${LOG_DIR}/${name}.log" 2>/dev/null | tail -1 || true)"
  printf '%-29s %-7s %-6s %s\n' "${name}" "${verdict}" "${took}" "${tally}"
  if [[ "${verdict}" != PASS ]]; then
    status=1
    grep -a -A20 -E '^==> Failing test targets' "${LOG_DIR}/${name}.log" 2>/dev/null \
      | sed -n '2,20p' | grep -v '^\s*(none)' | sed 's/^/    /' || true
    grep -a -E '::error ::' "${LOG_DIR}/${name}.log" 2>/dev/null | sed 's/^/    /' || true
  fi
done
echo "==> logs: ${LOG_DIR}"
partial=()
[[ -n "${LEGS_RE}" ]] && partial+=("--legs ${LEGS_RE}")
[[ "${OPT_CHOICE}" != BOTH ]] && partial+=("--opt ${OPT_CHOICE}")
[[ ${USE_FLOORS} -eq 0 ]] && partial+=("--no-floors")
[[ ${RUN_CHECKS} -eq 0 ]] && partial+=("--skip-checks")
if [[ ${#partial[@]} -gt 0 ]]; then
  echo "==> NOTE: partial run (${partial[*]}): this is not the full helia_test.yml gate."
fi
exit ${status}
