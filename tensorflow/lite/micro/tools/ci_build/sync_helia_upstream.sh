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
#
# Applies upstream tflite-micro from the commit in helia/UPSTREAM to <ref>,
# path by path, into the index of a clean checkout. Nothing is committed.
# see AmbiqAI/helia-rt#143
#
# Usage: sync_helia_upstream.sh [--no-record] <upstream branch, tag or sha>
#
# For each path upstream changed between the two commits:
#   heliaRT equals the base   -> take upstream (adds and deletes included);
#   both sides changed        -> 3-way merge, staged when clean;
#   otherwise                 -> conflict: heliaRT's version stays staged, and
#                                a modify/modify gets diff3 markers in the
#                                working tree only. A .github path heliaRT
#                                does not carry is skipped.
# --no-record leaves helia/UPSTREAM unchanged.

set -euo pipefail

usage() {
  echo "usage: $0 [--no-record] <upstream branch, tag or sha>" >&2
  exit 2
}

record=1
if [[ ${1:-} == --no-record ]]; then
  record=0
  shift
fi
[[ $# -eq 1 ]] || usage
ref_arg=$1

cd "$(git rev-parse --show-toplevel)"
upstream_file=helia/UPSTREAM
base=$(sed -n 's/^commit: *//p' "$upstream_file")
repo=$(sed -n 's/^repository: *//p' "$upstream_file")
if [[ -z $base || -z $repo ]]; then
  echo "error: no commit or repository line in $upstream_file" >&2
  exit 1
fi
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  echo "error: the checkout has uncommitted changes" >&2
  exit 1
fi

# A branch or tag name is always fetched, so a heliaRT branch of the same
# name (main) is never taken for upstream's.
have_commit() { git cat-file -e "$1^{commit}" 2>/dev/null; }
have_commit "$base" || git fetch -q "$repo" "$base"
if [[ $ref_arg =~ ^[0-9a-f]{7,40}$ ]] && have_commit "$ref_arg"; then
  ref=$(git rev-parse "$ref_arg^{commit}")
else
  git fetch -q "$repo" "$ref_arg"
  ref=$(git rev-parse FETCH_HEAD)
fi
if ! git merge-base --is-ancestor "$base" "$ref"; then
  echo "error: $ref does not descend from $base" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Prints "<mode> blob <sha>" for a path at a commit, or nothing if absent.
entry() { git ls-tree -z "$1" -- "$2" | cut -f1; }
sha_of() { echo "${1##* }"; }

took=0 added=0 deleted=0 merged=0 skipped=0 unchanged=0
conflicts=()
while IFS= read -r -d '' _status && IFS= read -r -d '' path; do
  h=$(entry HEAD "$path")
  b=$(entry "$base" "$path")
  r=$(entry "$ref" "$path")
  if [[ -z $h && $path == .github/* ]]; then
    skipped=$((skipped + 1))
  elif [[ $h == "$r" ]]; then
    unchanged=$((unchanged + 1))
  elif [[ $h == "$b" ]]; then
    if [[ -z $r ]]; then
      git rm -q -- "$path"
      deleted=$((deleted + 1))
    else
      git checkout -q "$ref" -- "$path"
      if [[ -z $h ]]; then added=$((added + 1)); else took=$((took + 1)); fi
    fi
  elif [[ -z $h ]]; then
    conflicts+=("deleted in heliaRT, changed upstream: $path")
  elif [[ -z $r ]]; then
    conflicts+=("changed in heliaRT, deleted upstream: $path")
  elif [[ -z $b ]]; then
    conflicts+=("added on both sides: $path")
  else
    git cat-file blob "$(sha_of "$h")" >"$tmp/heliart"
    git cat-file blob "$(sha_of "$b")" >"$tmp/base"
    git cat-file blob "$(sha_of "$r")" >"$tmp/upstream"
    rc=0
    git merge-file -q -p "$tmp/heliart" "$tmp/base" "$tmp/upstream" \
      >"$tmp/out" || rc=$?
    if [[ $rc -eq 0 ]]; then
      sha=$(git hash-object -w --no-filters "$tmp/out")
      git update-index --cacheinfo "${h%% *},$sha,$path"
      git checkout -q -- "$path"
      merged=$((merged + 1))
    else
      # rc above 127 is a merge-file error (a binary file): no markers.
      if [[ $rc -lt 128 ]]; then
        git merge-file -q -p --diff3 -L heliaRT -L "upstream ${base:0:8}" \
          -L "upstream ${ref:0:8}" "$tmp/heliart" "$tmp/base" \
          "$tmp/upstream" >"$path" || true
      fi
      conflicts+=("changed on both sides: $path")
    fi
  fi
done < <(git diff -z --no-renames --name-status "$base" "$ref")

if [[ $record -eq 1 ]]; then
  date=$(git log -1 --format=%cs "$ref")
  sed -i.bak -e "s/^commit: .*/commit: $ref/" -e "s/^date: .*/date: $date/" \
    "$upstream_file"
  rm -f "$upstream_file.bak"
  git add -- "$upstream_file"
fi

echo "upstream ${base:0:8}..${ref:0:8}"
echo "  took upstream:          $took"
echo "  added:                  $added"
echo "  deleted:                $deleted"
echo "  merged cleanly:         $merged"
echo "  already equal:          $unchanged"
echo "  skipped (.github):      $skipped"
echo "  conflicts:              ${#conflicts[@]}"
for c in ${conflicts[@]+"${conflicts[@]}"}; do
  echo "    $c"
done
