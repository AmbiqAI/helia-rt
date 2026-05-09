#!/usr/bin/env bash
# Refresh `third_party_static/` from the canonical downloaded copies under
# `tensorflow/lite/micro/tools/make/downloads/`.
#
# The committed `third_party_static/` tree exists so that downstream Zephyr
# projects can `west`-import this repo directly without running a `make`
# step before CMake configure. It must therefore stay in sync with the
# upstream third-party versions pinned by the various `*_download.sh`
# scripts under `tensorflow/lite/micro/tools/make/`.
#
# Run this script after any upstream sync that touches a third-party pin
# (flatbuffers, gemmlowp, ruy, kissfft). The companion CI job
# `static_export_drift_check` in `.github/workflows/ci.yml` will fail the
# PR if the committed tree is out of date.
#
# This script does NOT `git add` or `git commit` -- review the diff first.
set -euo pipefail

MAKE_FOLDER=./tensorflow/lite/micro/tools/make
DL_FOLDER=$MAKE_FOLDER/downloads

# Only fetch the third-party downloads, not a full microlite build.
make -C "$MAKE_FOLDER" TENSORFLOW_ROOT="$(pwd)/" third_party_downloads

# Wipe destinations before copying so:
#   1. Files removed upstream don't linger as zombies.
#   2. `cp -r src dst` doesn't nest into an existing dst directory.
for d in flatbuffers gemmlowp "ruy/ruy" kissfft; do
  rm -rf "third_party_static/$d"
  mkdir -p "third_party_static/$d"
done

cp "$DL_FOLDER/flatbuffers/LICENSE"     third_party_static/flatbuffers/LICENSE
cp -r "$DL_FOLDER/flatbuffers/include"  third_party_static/flatbuffers/include

cp "$DL_FOLDER/gemmlowp/LICENSE"        third_party_static/gemmlowp/LICENSE
cp -r "$DL_FOLDER/gemmlowp/fixedpoint"  third_party_static/gemmlowp/fixedpoint
cp -r "$DL_FOLDER/gemmlowp/internal"    third_party_static/gemmlowp/internal

cp "$DL_FOLDER/ruy/LICENSE"             third_party_static/ruy/LICENSE
cp -r "$DL_FOLDER/ruy/ruy/profiler"     third_party_static/ruy/ruy/profiler

cp "$DL_FOLDER/kissfft/COPYING"         third_party_static/kissfft/COPYING
cp "$DL_FOLDER/kissfft/_kiss_fft_guts.h" third_party_static/kissfft/_kiss_fft_guts.h
cp "$DL_FOLDER/kissfft/kiss_fft.c"      third_party_static/kissfft/kiss_fft.c
cp "$DL_FOLDER/kissfft/kiss_fft.h"      third_party_static/kissfft/kiss_fft.h
cp "$DL_FOLDER/kissfft/kissfft.hh"      third_party_static/kissfft/kissfft.hh
cp -r "$DL_FOLDER/kissfft/tools"        third_party_static/kissfft/tools

# Some upstream tarballs ship sources with the executable bit set. Strip it
# from regular files so the committed tree has stable, reproducible modes.
find third_party_static -type f -exec chmod a-x {} +

echo
echo "Static export complete. Review the diff and commit if intentional:"
echo "  git status third_party_static/"
echo "  git diff --stat third_party_static/"
