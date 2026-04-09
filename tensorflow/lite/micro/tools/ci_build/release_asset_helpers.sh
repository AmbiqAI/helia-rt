#!/usr/bin/env bash
# Shared helpers for release asset workflows and packaging scripts.

set -euo pipefail

die() {
  echo "::error ::$*" >&2
  exit 1
}

sanitize_tag() {
  local tag="$1"
  printf '%s' "${tag//\//-}"
}

emit_release_meta() {
  local ref="$1"
  local tag="$2"
  local upload="${3:-}"
  local sanitize="${4:-false}"
  local output_path="${GITHUB_OUTPUT:-/dev/stdout}"
  local resolved_tag="$tag"

  [[ -n "${ref}" ]] || die "Input ref is required."
  if [[ -z "${resolved_tag}" ]]; then
    resolved_tag="${ref}"
  fi
  if [[ "${sanitize}" == "true" ]]; then
    resolved_tag="$(sanitize_tag "${resolved_tag}")"
  fi
  [[ -n "${resolved_tag}" ]] || die "Resolved tag is empty."

  {
    echo "ref=${ref}"
    echo "tag=${resolved_tag}"
    if [[ -n "${upload}" ]]; then
      echo "upload=${upload}"
    fi
  } >> "${output_path}"
}

prune_cmsis_nn_to_headers() {
  local bundle_root="$1"
  local cmsis_dir="${bundle_root}/third_party/cmsis_nn"

  if [[ ! -d "${cmsis_dir}" ]]; then
    return 0
  fi

  find "${cmsis_dir}" -type f \
    ! \( \
      \( -path "${cmsis_dir}/Include/*" -a \( -name '*.h' -o -name '*.hpp' \) \) \
      -o -name 'LICENSE' \
    \) \
    -delete
  find "${cmsis_dir}" -depth -type d -empty -delete
}

copy_archives_from_artifacts() {
  local artifacts_dir="$1"
  local dest_dir="$2"

  mkdir -p "${dest_dir}"
  find "${artifacts_dir}" -type f -path "*/lib/*.a" -print -exec cp -v {} "${dest_dir}/" \;
}

copy_first_tflm_tree() {
  local artifacts_dir="$1"
  local dest_dir="$2"
  local candidate

  candidate="$(find "${artifacts_dir}" -type d -path '*/tflm' | sort | head -n1 || true)"
  [[ -n "${candidate}" ]] || die "No tflm tree found in artifacts."

  cp -a "${candidate}/." "${dest_dir}/"
}

write_manifest() {
  local bundle_name="$1"
  local tag="$2"
  local sha="$3"
  local libs_dir="$4"
  local output_file="$5"
  local extra="${6:-}"

  {
    echo "${bundle_name} ${tag}"
    echo "Commit: ${sha}"
    if [[ -n "${extra}" ]]; then
      echo "${extra}"
    fi
    echo
    echo "Libraries:"
    ls -1 "${libs_dir}"
  } > "${output_file}"
}

zip_bundle_into_upload_dir() {
  local bundle_dir="$1"
  local zip_name="$2"
  local upload_dir="$3"

  mkdir -p "${upload_dir}"
  rm -f "${zip_name}"
  zip -r "${zip_name}" "${bundle_dir}"
  mv "${zip_name}" "${upload_dir}/"
}

upload_bundle_to_release() {
  local tag="$1"
  local zip_path="$2"
  local repo="$3"

  [[ -f "${zip_path}" ]] || die "Zip artifact ${zip_path} not found."

  echo "Attempting upload of ${zip_path} to release ${tag}"
  if gh release upload "${tag}" "${zip_path}" --repo "${repo}" --clobber; then
    echo "Upload succeeded."
  else
    echo "::warning ::Upload failed (release may not exist). Create the release tag ${tag} then re-run or trigger a release event."
  fi
}
