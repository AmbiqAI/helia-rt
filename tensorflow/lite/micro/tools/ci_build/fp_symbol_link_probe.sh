#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# fp_symbol_link_probe.sh
#
# Floating-point symbol link-probe (helia-rt, issue #227 item 2).
#
# Builds a tiny program that takes the address of every ns-cmsis-nn floating
# point entry point the helia kernels call, links it against a built heliaRT
# static library, and fails if the linker cannot resolve anything.
#
# Why this exists: our FP coverage today is "the kernels compiled" plus "the
# archive contains an object file with the right name". Neither statement
# survives a library whose FP entry point is absent, renamed, or itself
# references a symbol that does not exist (AmbiqAI/ns-cmsis-nn#305 is exactly
# that class: a float kernel emitting a call to `__ARM_undef`). A real link is
# the cheapest check that catches all three, and unlike an executed test it
# needs no FVP, so it can run on every leg of every matrix.
#
# The symbol list is HARVESTED, never hand-maintained: a checked-in list goes
# stale the first time a kernel starts calling a new FP entry point, and a
# stale list is a guard that reports green about code it no longer covers.
#
# Per-config surface: the f16 entry points only exist when the library was
# built with ARM_NN_ENABLE_F16, which ext_libs/helia.inc gates on
# TARGET_ARCH=cortex-m55. On every other arch helia.inc also filters the
# `%_f16.c` / `%_fp16.c` sources out of the build, so probing f16 there would
# fail correctly-built libraries. `--arch` therefore selects the probe list.
#
# Usage:
#   fp_symbol_link_probe.sh --lib <archive.a> --arch <arch> \
#                           [--toolchain gcc|armclang|atfe] \
#                           [--target <make TARGET>] [--label <name>] \
#                           [--workdir <dir>]
#
# Examples:
#   # Against a test-matrix build (gen/ tree):
#   fp_symbol_link_probe.sh \
#     --lib gen/cortex_m_corstone_300_cortex-m55_helia_gcc/lib/libtensorflow-microlite.a \
#     --arch cortex-m55 --toolchain gcc --target cortex_m_corstone_300
#
#   # Against a release artifact:
#   fp_symbol_link_probe.sh --lib out/.../lib/libhelia-rt-cm55-armclang-release.a \
#     --arch cortex-m55 --toolchain armclang --target cortex_m_generic
#
# Exit codes: 0 probe linked, 2 usage error, 3 no symbols harvested,
#             4 compile failed, 5 link failed (this is the real finding).
# ------------------------------------------------------------------------------

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${ROOT_DIR}"

MAKEFILE="tensorflow/lite/micro/tools/make/Makefile"
KERNELS_DIR="tensorflow/lite/micro/kernels/helia"

LIB=""
ARCH="cortex-m55"
TOOLCHAIN="gcc"
TARGET="cortex_m_generic"
LABEL=""
WORKDIR=""

usage() {
  # Print the header comment block (everything up to the closing rule).
  sed -n '2,/^# -\{20,\}$/p' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)       LIB="${2:?missing value for --lib}"; shift 2 ;;
    -a|--arch)   ARCH="${2:?missing value for --arch}"; shift 2 ;;
    -t|--toolchain) TOOLCHAIN="${2:?missing value for --toolchain}"; shift 2 ;;
    --target)    TARGET="${2:?missing value for --target}"; shift 2 ;;
    --label)     LABEL="${2:?missing value for --label}"; shift 2 ;;
    --workdir)   WORKDIR="${2:?missing value for --workdir}"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "fp-probe: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${LIB}" ]] || { echo "fp-probe: --lib is required" >&2; exit 2; }
[[ -f "${LIB}" ]] || { echo "fp-probe: no such library: ${LIB}" >&2; exit 2; }

case "${TOOLCHAIN}" in
  gcc|armclang|atfe) ;;
  *) echo "fp-probe: invalid --toolchain '${TOOLCHAIN}' (gcc|armclang|atfe)" >&2; exit 2 ;;
esac

# ATfE: prefer the copy baked into the CI image, exactly as build_helia.sh and
# test_helia.sh do. Without this the make query below would try to download a
# second ATfE just to answer a question about compiler flags.
if [[ "${TOOLCHAIN}" == "atfe" && -z "${TARGET_TOOLCHAIN_ROOT:-}" ]]; then
  ATFE_PREINSTALL="${ATFE_PREINSTALL:-/opt/atfe}"
  if [[ -x "${ATFE_PREINSTALL}/bin/clang" ]]; then
    TARGET_TOOLCHAIN_ROOT="${ATFE_PREINSTALL}/bin/"
  fi
fi

LABEL="${LABEL:-${ARCH}-${TOOLCHAIN}}"
WORKDIR="${WORKDIR:-${ROOT_DIR}/gen/fp_link_probe/${LABEL}}"
mkdir -p "${WORKDIR}"

LIB_ABS="$(cd "$(dirname "${LIB}")" && pwd)/$(basename "${LIB}")"

# ------------------------- 1. Harvest the symbol list --------------------------
# Match call sites: an `arm_..._f32` / `arm_..._f16` identifier immediately
# followed by an open paren. Declarations match too, which is fine -- helia
# only declares what it calls -- but comments and doc prose do not, which is
# the point of requiring the paren.
#
# `-H` keeps the filename, so each symbol is tracked back to the helia kernel(s)
# that call it (SYMBOL_FILES). A legitimate MISSING report then names the
# referring kernel, not just a bare `arm_..._f16` -- the ns#329 class (a
# removed fp16 pair) reports exactly such a bare symbol, and "which kernel
# breaks?" is the first triage question.
harvest_pairs() {
  # Emits "<symbol><TAB><file>" per call site. awk splits on the FIRST colon
  # only (repo paths carry no colon), then strips the trailing "(" and any
  # leading space off the matched token.
  grep -rHoE '\barm_[a-z0-9_]+_f(16|32)[[:space:]]*\(' \
       --include='*.cc' --include='*.h' --include='*.c' \
       "${KERNELS_DIR}" \
    | awk -F: '{ file=$1; sym=$2; sub(/[[:space:]]*\(.*/, "", sym); print sym "\t" file }' \
    | sort -u
}

declare -A SYMBOL_FILES=()
ALL_SYMBOLS=()
while IFS=$'\t' read -r _sym _file; do
  [[ -n "${_sym}" ]] || continue
  if [[ -z "${SYMBOL_FILES[${_sym}]:-}" ]]; then
    ALL_SYMBOLS+=("${_sym}")
    SYMBOL_FILES[${_sym}]="${_file}"
  else
    # De-dup files per symbol; a symbol called from several kernels lists all.
    case " ${SYMBOL_FILES[${_sym}]} " in
      *" ${_file} "*) : ;;
      *) SYMBOL_FILES[${_sym}]+=" ${_file}" ;;
    esac
  fi
done < <(harvest_pairs)

if [[ "${#ALL_SYMBOLS[@]}" -eq 0 ]]; then
  echo "fp-probe: harvested no FP symbols from ${KERNELS_DIR} -- the harvest" >&2
  echo "fp-probe: pattern or the kernel layout changed. Refusing to report" >&2
  echo "fp-probe: green on an empty probe." >&2
  exit 3
fi

# ------------------------- 2. Ask make for the leg's flags ---------------------
# The probe must be compiled with the same arch/FPU/ABI flags as the library or
# the linker rejects the objects on build-attribute grounds. Rather than
# duplicating that flag logic here (where it would drift), ask the Makefile
# that produced the library. `--eval` injects a print rule without editing the
# upstream Makefile; pattern rules never become the default goal, so this is
# side-effect free.
make_var() {
  # shellcheck disable=SC2016  # `$($*)` is make syntax, not shell; must not expand
  make -f "${MAKEFILE}" \
    --eval='helia_fp_probe_print_%: ; @echo "$($*)"' \
    TARGET="${TARGET}" \
    TARGET_ARCH="${ARCH}" \
    TOOLCHAIN="${TOOLCHAIN}" \
    OPTIMIZED_KERNEL_DIR=helia \
    ${TARGET_TOOLCHAIN_ROOT:+TARGET_TOOLCHAIN_ROOT="${TARGET_TOOLCHAIN_ROOT}"} \
    "helia_fp_probe_print_$1" 2>/dev/null | tail -1
}

PROBE_CC="$(make_var CC)"
PROBE_CCFLAGS="$(make_var CCFLAGS)"
# MICROLITE_LIBS is queried, not hardcoded, on purpose: it is `-lm` for gcc and
# ATfE but EMPTY for armclang, because the upstream Makefile filters `-lm` out
# under armclang (armlink provides libm implementations itself and rejects the
# flag with `L6450U: Cannot find library m` -- see cortex_m_generic_makefile.inc).
# Passing a fixed `-lm` here would make the armclang link fail with an error
# that names no probe symbol, which the classifier below would (correctly)
# treat as "guard could not be evaluated" and exit non-zero -- turning a sound
# library into a red release leg. Inheriting the value from make is what keeps
# the armclang branch honest. Do not replace this with a literal.
PROBE_LIBS="$(make_var MICROLITE_LIBS)"

[[ -n "${PROBE_CC}" ]] || { echo "fp-probe: could not resolve CC from ${MAKEFILE}" >&2; exit 2; }

# -Werror is dropped: the probe is generated code and a warning in it is not a
# finding about the library. Nothing else is filtered -- the arch/FPU/ABI flags
# are the whole point of querying make.
CCFLAGS_FILTERED=()
for flag in ${PROBE_CCFLAGS}; do
  [[ "${flag}" == "-Werror" ]] && continue
  CCFLAGS_FILTERED+=("${flag}")
done

echo "==> fp-probe ${LABEL}: CC=${PROBE_CC}"

# ------------------------- 3. Per-config filter --------------------------------
# f16 entry points only exist where ARM_NN_ENABLE_F16 is set. Read that off the
# flags make just reported rather than re-testing `ARCH == cortex-m55` here:
# ext_libs/helia.inc owns the condition (it gates both the -D and the `%_f16.c`
# source filter on it), and a second copy of it would silently drop 18 symbols
# and still report green the day a new f16-capable target is added.
ENABLE_F16=0
case " ${PROBE_CCFLAGS} " in
  *" -DARM_NN_ENABLE_F16=1 "*) ENABLE_F16=1 ;;
esac

SYMBOLS=()
for sym in "${ALL_SYMBOLS[@]}"; do
  if [[ "${sym}" == *_f16 && "${ENABLE_F16}" -eq 0 ]]; then
    continue
  fi
  SYMBOLS+=("${sym}")
done

echo "==> fp-probe ${LABEL}: ${#SYMBOLS[@]} symbols (ARM_NN_ENABLE_F16 $([[ ${ENABLE_F16} -eq 1 ]] && echo set || echo unset), per make)"
printf '    %s\n' "${SYMBOLS[@]}"

# ------------------------- 4. Generate the probe -------------------------------
PROBE_C="${WORKDIR}/fp_link_probe.c"
{
  echo "/* Generated by tools/ci_build/fp_symbol_link_probe.sh -- do not edit. */"
  echo "/* arch=${ARCH} toolchain=${TOOLCHAIN} lib=$(basename "${LIB_ABS}") */"
  echo
  echo "typedef void (*helia_fp_probe_fn)(void);"
  echo
  # Deliberately declared as `void (void)`: the probe only takes addresses, so
  # the real prototypes (and the ns-cmsis-nn headers they need) are irrelevant,
  # and not including the headers keeps the probe independent of whichever
  # ns-cmsis-nn tree happens to be checked out.
  for sym in "${SYMBOLS[@]}"; do
    echo "extern void ${sym}(void);"
  done
  echo
  echo "helia_fp_probe_fn helia_fp_probe_table[] = {"
  for sym in "${SYMBOLS[@]}"; do
    echo "  ${sym},"
  done
  echo "};"
  echo
  echo "int main(void) { return helia_fp_probe_table[0] != 0 ? 0 : 1; }"
} > "${PROBE_C}"


# ------------------------- 5. Compile ------------------------------------------
PROBE_O="${WORKDIR}/fp_link_probe.o"
if ! "${PROBE_CC}" "${CCFLAGS_FILTERED[@]}" -c "${PROBE_C}" -o "${PROBE_O}"; then
  echo "::error ::fp-probe ${LABEL}: probe failed to COMPILE (not a library finding)" >&2
  exit 4
fi

# ------------------------- 6. Link ---------------------------------------------
# The probe is a freestanding executable, so it needs an entry point and no
# start files; the toolchain's default libraries stay in, because ns-cmsis-nn's
# float kernels legitimately reference libm and the compiler runtime.
#
# Link options are tried in order and the first success wins. A genuinely
# unresolved FP symbol is unresolved under every variant, so the fallback list
# cannot mask a real finding -- it only absorbs the differences between GNU ld,
# LLD and armlink defaults.
LINK_VARIANTS=()
case "${TOOLCHAIN}" in
  armclang)
    # armlink links a bare executable with the Arm C library and no scatter
    # file, and reports unresolved references as L6218E errors.
    LINK_VARIANTS=("" "-Wl,--entry=main")
    ;;
  *)
    LINK_VARIANTS=("-nostartfiles -Wl,--entry=main" "--specs=nosys.specs" "")
    ;;
esac


# A linker naming ONE OF OUR PROBE SYMBOLS as unresolved is a conclusive
# finding: stop there, because the remaining variants only bury the useful line
# under their own noise.
#
# Deciding that requires the UNDEFINED symbol specifically, not "this line
# mentions one of our names somewhere". Two distinct traps:
#
#   * "undefined reference" alone is not enough -- the fallback variants
#     legitimately report undefined libc stubs (`_sbrk`, `_exit`, ...) on some
#     toolchain/target pairings. Treating that as conclusive would abort the
#     fallback chain and misreport an environment problem as a library defect.
#   * armlink's message also names the REFERRING object, and ns-cmsis-nn
#     members are named after the very symbols we probe. So
#         Error: L6218E: Undefined symbol __missing_libc_helper
#                        (referred from arm_softmax_f32.o).
#     mentions "arm_softmax_f32" on the wrong half of the line. Matching the
#     whole message -- by substring or word boundary -- cannot tell the two
#     halves apart.
#
# So: extract the undefined symbol NAME from each diagnostic, then test exact
# membership in the probe list. `undef_symbols` handles the three linkers we
# use (GNU ld, LLD, armlink) and stops at the symbol token, which discards
# armlink's "(referred from ...)" tail.
UNDEF_RE='undefined (reference|symbol)|L6218E'

undef_symbols() {  # $1 = link log -> undefined symbol names, one per line
  sed -E -n \
    -e "s/.*undefined reference to \`([A-Za-z0-9_.]+)'.*/\1/p" \
    -e "s/.*[Uu]ndefined symbol:?[[:space:]]+([A-Za-z0-9_.]+).*/\1/p" \
    "$1" | sort -u
}

# Which of the probe's OWN symbols does this log report as undefined?
missing_probe_symbols() {  # $1 = link log
  local sym want
  while IFS= read -r sym; do
    for want in "${SYMBOLS[@]}"; do
      if [[ "${sym}" == "${want}" ]]; then
        printf '%s\n' "${sym}"
        break
      fi
    done
  done < <(undef_symbols "$1")
}

PROBE_ELF="${WORKDIR}/fp_link_probe.elf"
link_ok=0
conclusive=""
attempt=0
for variant in "${LINK_VARIANTS[@]}"; do
  attempt=$((attempt + 1))
  log="${WORKDIR}/link_attempt_${attempt}.log"
  # shellcheck disable=SC2206  # deliberate word splitting: variant/libs are flag lists
  extra=(${variant} ${PROBE_LIBS})
  if "${PROBE_CC}" "${CCFLAGS_FILTERED[@]}" -o "${PROBE_ELF}" \
       "${PROBE_O}" "${LIB_ABS}" "${extra[@]}" > "${log}" 2>&1; then
    echo "==> fp-probe ${LABEL}: linked OK (variant ${attempt}: ${variant:-<driver defaults>})"
    link_ok=1
    break
  fi
  mapfile -t MISSING < <(missing_probe_symbols "${log}")
  if [[ "${#MISSING[@]}" -gt 0 ]]; then
    conclusive="${log}"
    break
  fi
done

if [[ "${link_ok}" -ne 1 ]]; then
  echo "::error ::fp-probe ${LABEL}: probe failed to LINK against $(basename "${LIB_ABS}")" >&2
  if [[ -n "${conclusive}" ]]; then
    echo "fp-probe: the linker could not resolve FP entry points that helia" >&2
    echo "fp-probe: calls -- a real defect in the library, not a probe" >&2
    echo "fp-probe: configuration issue. Unresolved probe symbols:" >&2
    for _m in "${MISSING[@]}"; do
      printf 'fp-probe:   MISSING %s (called from %s)\n' \
        "${_m}" "${SYMBOL_FILES[${_m}]:-unknown}" >&2
    done
    echo "fp-probe: linker diagnostics:" >&2
    grep -E "${UNDEF_RE}" "${conclusive}" >&2
    echo "fp-probe: full log: ${conclusive}" >&2
  else
    echo "fp-probe: every link variant failed WITHOUT naming any probed FP" >&2
    echo "fp-probe: symbol as unresolved. That points at the probe's link" >&2
    echo "fp-probe: environment rather than at the library, but it is still a" >&2
    echo "fp-probe: failure: the guard could not be evaluated. Full logs:" >&2
    for log in "${WORKDIR}"/link_attempt_*.log; do
      echo "----- ${log} -----" >&2
      cat "${log}" >&2
    done
  fi
  exit 5
fi

echo "==> fp-probe ${LABEL}: PASS (${#SYMBOLS[@]} FP entry points resolved)"
