#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Execute the selected release FP tests against an unchanged generic archive.
# SPEED consumes the release workflow's archive; SIZE builds a qualification
# archive that is not included in the release bundle. See AmbiqAI/helia-rt#234.

set -Eeuo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_command() {
  printf 'RUN:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

usage() {
  cat <<'USAGE'
Usage: test_helia_release_fp.sh --profile <SPEED|SIZE> [options]

Options:
  --archive <path>  Exact shipped archive; required for SPEED, rejected for SIZE
  --out-dir <path>  Fresh directory for generated objects and receipts
  -h, --help        Show this help
USAGE
}

require_map_members() {  # <map> <absolute-archive> <member>...
  local map_file="$1"
  local archive="$2"
  shift 2

  [[ -f "${map_file}" ]] || die "link map not found: ${map_file}"
  local member match
  for member in "$@"; do
    match="${archive}(${member})"
    if ! grep -Fq "${match}" "${map_file}"; then
      die "${map_file} does not attribute ${member} to ${archive}"
    fi
    grep -Fm1 "${match}" "${map_file}"
  done
}

validate_tally() {  # <tally-file>
  local tally_file="$1"
  [[ -f "${tally_file}" ]] || die "test tally not found: ${tally_file}"

  awk -F '\t' '
    BEGIN {
      expected["kernel_float_activation_edge_test"] = 12
      expected["kernel_float_elementwise_edge_test"] = 5
      expected["kernel_float_lstm_tail_lane_test"] = 2
    }
    NF != 3 {
      printf "ERROR: malformed tally row %d\n", NR > "/dev/stderr"
      bad = 1
      next
    }
    !($1 in expected) {
      printf "ERROR: unexpected tally binary %s\n", $1 > "/dev/stderr"
      bad = 1
      next
    }
    seen[$1]++ {
      printf "ERROR: duplicate tally binary %s\n", $1 > "/dev/stderr"
      bad = 1
      next
    }
    $2 !~ /^[0-9]+$/ || $2 != expected[$1] {
      printf "ERROR: %s executed %s cases; expected %d\n", \
             $1, $2, expected[$1] > "/dev/stderr"
      bad = 1
    }
    $3 != "counted" {
      printf "ERROR: %s has tally kind %s; expected counted\n", \
             $1, $3 > "/dev/stderr"
      bad = 1
    }
    { rows++; total += $2 }
    END {
      for (name in expected) {
        if (seen[name] != 1) {
          printf "ERROR: expected exactly one tally row for %s\n", \
                 name > "/dev/stderr"
          bad = 1
        }
      }
      if (rows != 3 || total != 19) {
        printf "ERROR: tally has %d rows and %d cases; expected 3 and 19\n", \
               rows, total > "/dev/stderr"
        bad = 1
      }
      exit bad
    }
  ' "${tally_file}" || die "release FP test tally failed"

  echo "release FP tally: 3 binaries, 19 test cases"
  cat "${tally_file}"
}

assert_archive_unchanged() {  # <archive> <resolved-path> <sha256>
  local archive="$1"
  local expected_path="$2"
  local expected_hash="$3"

  [[ -f "${archive}" ]] || die "archive disappeared after execution: ${archive}"
  local actual_path actual_hash
  actual_path="$(realpath "${archive}")"
  actual_hash="$(sha256sum "${archive}" | awk '{print $1}')"
  [[ "${actual_path}" == "${expected_path}" ]] || \
    die "archive path changed: ${expected_path} -> ${actual_path}"
  [[ "${actual_hash}" == "${expected_hash}" ]] || \
    die "archive changed: ${expected_hash} -> ${actual_hash}"
  echo "archive SHA-256 after execution: ${actual_hash}  ${actual_path}"
}

main() {
  local profile=""
  local input_archive=""
  local out_dir=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        [[ $# -ge 2 ]] || { usage; exit 2; }
        profile="$2"
        shift 2
        ;;
      --archive)
        [[ $# -ge 2 ]] || { usage; exit 2; }
        input_archive="$2"
        shift 2
        ;;
      --out-dir)
        [[ $# -ge 2 ]] || { usage; exit 2; }
        out_dir="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage
        exit 2
        ;;
    esac
  done

  case "${profile}" in
    SPEED)
      [[ -n "${input_archive}" ]] || \
        die "SPEED requires --archive from the release build job"
      ;;
    SIZE)
      [[ -z "${input_archive}" ]] || \
        die "SIZE builds its qualification archive; --archive is not accepted"
      ;;
    *)
      echo "Invalid --profile '${profile}'. Use SPEED or SIZE." >&2
      exit 2
      ;;
  esac

  local invocation_dir
  invocation_dir="${PWD}"
  if [[ "${profile}" == "SPEED" ]]; then
    [[ -f "${input_archive}" ]] || die "SPEED archive not found: ${input_archive}"
    [[ "$(basename "${input_archive}")" == \
       "libhelia-rt-cm55-gcc-release-with-logs.a" ]] || \
      die "unexpected SPEED archive name: $(basename "${input_archive}")"
    input_archive="$(realpath "${input_archive}")"
  fi
  if [[ -n "${out_dir}" && "${out_dir}" != /* ]]; then
    out_dir="${invocation_dir}/${out_dir}"
  fi

  local script_dir root_dir makefile jobs
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd "${script_dir}/../../../../.." && pwd)"
  makefile="${root_dir}/tensorflow/lite/micro/tools/make/Makefile"
  jobs="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
  cd "${root_dir}"

  if [[ -z "${out_dir}" ]]; then
    out_dir="$(mktemp -d "${TMPDIR:-/tmp}/helia-release-fp-${profile,,}.XXXXXX")"
  else
    if [[ -e "${out_dir}" ]] && \
       [[ -n "$(find "${out_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      die "--out-dir must be absent or empty: ${out_dir}"
    fi
    mkdir -p "${out_dir}"
  fi
  out_dir="$(realpath "${out_dir}")"

  local -a corstone_base=(
    -f "${makefile}"
    TARGET=cortex_m_corstone_300
    TARGET_ARCH=cortex-m55
    TOOLCHAIN=gcc
    OPTIMIZED_KERNEL_DIR=helia
    BUILD_TYPE=release_with_logs
    "GLOBAL_KERNEL_OPTIMIZE=${profile}"
    CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=1
  )

  run_command make "${corstone_base[@]}" third_party_downloads

  local expected_core actual_core compiler compiler_version
  expected_core="$(sed -n 's/^NS_CMSIS_NN_COMMIT ?= //p' \
    tensorflow/lite/micro/tools/make/ext_libs/helia.inc)"
  actual_core="$(git -C tensorflow/lite/micro/tools/make/downloads/ns_cmsis_nn \
    rev-parse HEAD)"
  [[ "${actual_core}" == "${expected_core}" ]] || \
    die "CORE HEAD ${actual_core} does not match pin ${expected_core}"
  compiler="tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-g++"

  echo "heliaRT head: $(git rev-parse HEAD)"
  echo "CORE head: ${actual_core}"
  echo "kernel profile: ${profile}"
  compiler_version="$("${compiler}" --version)"
  sed -n '1p' <<<"${compiler_version}"

  local archive
  if [[ "${profile}" == "SPEED" ]]; then
    archive="${input_archive}"
  else
    local generic_gen generic_gendir built_archive archive_dir
    generic_gen="${out_dir}/gen-generic-size"
    local -a generic_args=(
      -f "${makefile}"
      TARGET=cortex_m_generic
      TARGET_ARCH=cortex-m55
      TOOLCHAIN=gcc
      OPTIMIZED_KERNEL_DIR=helia
      BUILD_TYPE=release_with_logs
      GLOBAL_KERNEL_OPTIMIZE=SIZE
      CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=1
      "BASE_GENDIR=${generic_gen}"
    )
    run_command make -j"${jobs}" "${generic_args[@]}" microlite
    generic_gendir="$(make "${generic_args[@]}" list_gendir 2>/dev/null | tail -n 1)"
    built_archive="${generic_gendir}lib/libtensorflow-microlite.a"
    [[ -f "${built_archive}" ]] || die "SIZE archive not found: ${built_archive}"
    archive_dir="${out_dir}/archive"
    mkdir -p "${archive_dir}"
    archive="${archive_dir}/libhelia-rt-cm55-gcc-release-with-logs-size.a"
    cp "${built_archive}" "${archive}"
    archive="$(realpath "${archive}")"
  fi

  local archive_path_before archive_hash_before
  archive_path_before="$(realpath "${archive}")"
  archive_hash_before="$(sha256sum "${archive}" | awk '{print $1}')"
  echo "archive SHA-256 before linking: ${archive_hash_before}  ${archive_path_before}"

  local runtime_sources
  runtime_sources="tensorflow/lite/micro/debug_log.cc"
  runtime_sources+=" tensorflow/lite/micro/cortex_m_corstone_300/micro_time.cc"
  runtime_sources+=" tensorflow/lite/micro/cortex_m_corstone_300/system_setup.cc"
  runtime_sources+=" tensorflow/lite/micro/cortex_m_corstone_300/fault_handlers.cc"
  runtime_sources+=" tensorflow/lite/micro/tools/make/downloads/ethos_u_core_platform/targets/corstone-300/uart.c"
  runtime_sources+=" tensorflow/lite/micro/tools/make/downloads/ethos_u_core_platform/targets/corstone-300/retarget.c"
  runtime_sources+=" tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/ARMCM55/Source/system_ARMCM55.c"
  runtime_sources+=" tensorflow/lite/micro/tools/make/downloads/cmsis/Cortex_DFP/Device/ARMCM55/Source/startup_ARMCM55.c"

  local exact_gen tally_file receipt_dir corstone_gendir live_map
  exact_gen="${out_dir}/gen-exact-${profile,,}"
  tally_file="${out_dir}/test-tally.tsv"
  receipt_dir="${out_dir}/maps"
  mkdir -p "${receipt_dir}"
  : > "${tally_file}"
  export HELIA_TEST_TALLY_FILE="${tally_file}"
  export FVP_TIMEOUT_SECONDS="${FVP_TIMEOUT_SECONDS:-120}"

  local -a link_args=(
    "${corstone_base[@]}"
    V=1
    "BASE_GENDIR=${exact_gen}"
    "MICROLITE_LIB_PATH=${archive}"
    "MICROLITE_TEST_RUNTIME_SRCS=${runtime_sources}"
  )
  corstone_gendir="$(make "${link_args[@]}" list_gendir 2>/dev/null | tail -n 1)"
  live_map="${corstone_gendir}cortex_m_corstone_300.map"

  local spec binary expected_count members map_copy
  local -a required_members
  local -a test_specs=(
    "kernel_float_activation_edge_test:12:logistic.o,tanh.o,arm_nn_activation_f32.o,arm_nn_activation_f16.o"
    "kernel_float_elementwise_edge_test:5:add.o,mul.o,arm_elementwise_add_f32.o,arm_elementwise_add_f16.o,arm_elementwise_mul_f32.o,arm_elementwise_mul_f16.o"
    "kernel_float_lstm_tail_lane_test:2:unidirectional_sequence_lstm.o,arm_lstm_unidirectional_f32.o,arm_lstm_unidirectional_f16.o"
  )

  for spec in "${test_specs[@]}"; do
    IFS=: read -r binary expected_count members <<<"${spec}"
    run_command make --old-file="${archive}" -j"${jobs}" \
      "${link_args[@]}" "${binary}"
    [[ -f "${live_map}" ]] || die "link did not produce map: ${live_map}"
    map_copy="${receipt_dir}/${binary}.map"
    cp "${live_map}" "${map_copy}"
    IFS=, read -r -a required_members <<<"${members}"
    require_map_members "${map_copy}" "${archive}" "${required_members[@]}"
    run_command make --old-file="${archive}" \
      "${link_args[@]}" "test_${binary}"
    echo "expected ${binary} tally: ${expected_count}"
  done

  validate_tally "${tally_file}"
  assert_archive_unchanged \
    "${archive}" "${archive_path_before}" "${archive_hash_before}"
  echo "release FP execution receipt: PASS (${profile})"
  echo "receipt directory: ${out_dir}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
