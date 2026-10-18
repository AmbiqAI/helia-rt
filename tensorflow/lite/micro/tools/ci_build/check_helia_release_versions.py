#!/usr/bin/env python3
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
"""Checks that every release version and ns-cmsis-nn pin site agrees.

release-please bumps only the files listed in release-please-config.json
`extra-files`, and only on lines carrying its version marker. A versioned
file missing from that list, or a list entry whose marked line holds another
version, ships stale. The ns-cmsis-nn tag and commit are written by hand in
several files, and nothing bumps them together.
see AmbiqAI/helia-rt#445, AmbiqAI/helia-rt#402

Rules:
  * .release-please-manifest.json holds one package, ".", at X.Y.Z.
  * Every extra-files entry is a generic file that exists and has at least
    one marked line or start/end block, each holding an X.Y.Z. Every X.Y.Z
    there is the manifest version: release-please rewrites them all.
  * Every tracked file that carries a marker, and every nsx-module.yaml, is
    an extra-files entry.
  * The ns-cmsis-nn tag and commit in ext_libs/helia.inc match the copies in
    PIN_SITES, each of which appears exactly once in its file.

Usage: check_helia_release_versions.py [repo-root]
  repo-root defaults to the repository root inferred from this file.

Exit codes: 0 consistent, 1 inconsistent, 2 usage error.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Split so this file does not carry the marker itself.
MARKER = "x-release-please-" + "version"
BLOCK_START = "x-release-please-start-" + "version"
BLOCK_END = "x-release-please-" + "end"

# An X.Y.Z not inside a longer dotted number, so 2.1.0 in helia-rt.2.1.0.pack
# counts and 1.2.3.4 does not.
SEMVER = re.compile(r"(?<!\d)(?<!\d\.)(\d+\.\d+\.\d+)(?!\d|\.\d)")

# Outside a git checkout, the tree scan skips VCS state, fetched third-party
# code and build or tool output, none of which release-please edits.
SKIP_DIRS = {
  ".git",
  "node_modules",
  "downloads",
  "gen",
  "__pycache__",
  ".venv",
  "dist",
  ".astro",
}

HELIA_INC = "tensorflow/lite/micro/tools/make/ext_libs/helia.inc"
PIN_TAG = re.compile(
  r"^# Tracks AmbiqAI/ns-cmsis-nn tag (v\d+\.\d+\.\d+)\.", re.M
)
PIN_COMMIT = re.compile(r"^NS_CMSIS_NN_COMMIT \?= ([0-9a-f]{40})$", re.M)

# (path, pattern, "tag" or "commit"). Group 1 must equal the helia.inc value.
PIN_SITES = (
  (
    "tensorflow/lite/micro/tools/make/ext_libs/ns_cmsis_nn_download.sh",
    r"^# Default tracks AmbiqAI/ns-cmsis-nn tag (v\d+\.\d+\.\d+)\.",
    "tag",
  ),
  (
    "tensorflow/lite/micro/tools/make/ext_libs/ns_cmsis_nn_download.sh",
    r"^GIT_COMMIT=\$\{NS_CMSIS_NN_COMMIT:-([0-9a-f]{40})\}$",
    "commit",
  ),
  (
    ".github/workflows/ns_cmsis_nn_canary.yml",
    r"^\s*# Current repository pin: AmbiqAI/ns-cmsis-nn (v\d+\.\d+\.\d+)\.",
    "tag",
  ),
  (
    ".github/workflows/ns_cmsis_nn_canary.yml",
    r"^\s*default: \"([0-9a-f]{40})\"$",
    "commit",
  ),
  (
    ".github/workflows/smoke_cmake.yml",
    r"^\s*NS_CMSIS_NN_REF: (v\d+\.\d+\.\d+)$",
    "tag",
  ),
  (
    "astro-site/src/content/docs/getting-started/zephyr.md",
    r"^- name: ns-cmsis-nn\n(?:  .*\n)*?  revision: (\S+)$",
    "tag",
  ),
)


class Checker:
  def __init__(self, root):
    self.root = root
    self.errors = []

  def error(self, message):
    self.errors.append(message)

  def read(self, rel):
    try:
      return (self.root / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
      self.error(f"{rel}: cannot read ({e})")
      return None

  def manifest_version(self):
    text = self.read(".release-please-manifest.json")
    if text is None:
      return None
    try:
      manifest = json.loads(text)
    except json.JSONDecodeError as e:
      self.error(f".release-please-manifest.json: invalid JSON ({e})")
      return None
    if (
      not isinstance(manifest, dict)
      or list(manifest) != ["."]
      or not isinstance(manifest["."], str)
      or not re.fullmatch(r"\d+\.\d+\.\d+", manifest["."])
    ):
      self.error(
        ".release-please-manifest.json: expected exactly {\".\": \"X.Y.Z\"}"
      )
      return None
    return manifest["."]

  def extra_files(self):
    text = self.read("release-please-config.json")
    if text is None:
      return None
    try:
      config = json.loads(text)
      entries = config["packages"]["."]["extra-files"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
      self.error(
        f"release-please-config.json: no packages.\".\".extra-files ({e})"
      )
      return None
    paths = []
    for entry in entries:
      if isinstance(entry, str):
        paths.append(entry)
      elif (
        isinstance(entry, dict)
        and entry.get("type") == "generic"
        and isinstance(entry.get("path"), str)
        and set(entry) <= {"type", "path"}
      ):
        paths.append(entry["path"])
      else:
        self.error(
          "release-please-config.json: extra-files entry "
          f"{json.dumps(entry)} is not a generic file; this check "
          "reads only marker lines"
        )
    return paths

  def marked_units(self, rel, text):
    """Yields (first line number, lines) for each unit release-please rewrites:
    a line carrying the marker, or the lines of a start/end block."""
    block = None
    for number, line in enumerate(text.splitlines(), 1):
      if BLOCK_START in line:
        if block is not None:
          self.error(f"{rel}:{number}: version block opened inside another")
        block = (number + 1, [])
      elif block is not None and BLOCK_END in line:
        yield block
        block = None
      elif block is not None:
        block[1].append(line)
      elif MARKER in line:
        yield number, [line]
    if block is not None:
      self.error(f"{rel}:{block[0] - 1}: version block is never closed")

  def check_extra_file(self, rel, version):
    if not (self.root / rel).is_file():
      self.error(f"{rel}: listed in extra-files but missing")
      return
    text = self.read(rel)
    if text is None:
      return
    units = list(self.marked_units(rel, text))
    if not units:
      self.error(
        f"{rel}: listed in extra-files but has no {MARKER} line or "
        "block, so release-please never bumps it"
      )
    for number, lines in units:
      found = [
        (number + i, v)
        for i, line in enumerate(lines)
        for v in SEMVER.findall(line)
      ]
      if not found:
        self.error(
          f"{rel}:{number}: marked line or block holds no X.Y.Z version"
        )
      for at, value in found:
        if value != version:
          self.error(
            f"{rel}:{at}: version {value} where release-please "
            f"writes the release; the manifest is at {version}"
          )

  def tree_files(self):
    """Yields repository-relative paths: the tracked files of a git checkout,
    otherwise every file outside SKIP_DIRS."""
    if (self.root / ".git").exists():
      listed = subprocess.run(
        ["git", "-C", str(self.root), "ls-files", "-z"],
        check=True,
        capture_output=True,
      ).stdout
      yield from (p.decode() for p in listed.split(b"\0") if p)
      return
    for dirpath, dirnames, filenames in os.walk(self.root):
      dirnames[:] = [
        d for d in dirnames if d not in SKIP_DIRS and not d.startswith("bazel-")
      ]
      for name in filenames:
        yield (Path(dirpath) / name).relative_to(self.root).as_posix()

  def scan_tree(self):
    """Returns (files carrying a marker, nsx-module.yaml files)."""
    marked, modules = set(), set()
    for rel in self.tree_files():
      if Path(rel).name == "nsx-module.yaml":
        modules.add(rel)
      try:
        data = (self.root / rel).read_bytes()
      except OSError:
        continue
      if MARKER.encode() in data or BLOCK_START.encode() in data:
        marked.add(rel)
    return marked, modules

  def check_release_version(self):
    version = self.manifest_version()
    listed = self.extra_files()
    if version is None or listed is None:
      return
    for rel in listed:
      self.check_extra_file(rel, version)
    marked, modules = self.scan_tree()
    for rel in sorted(marked - set(listed)):
      self.error(
        f"{rel}: carries {MARKER} but is not in extra-files, so "
        "release-please never bumps it"
      )
    for rel in sorted(modules - set(listed)):
      self.error(
        f"{rel}: NSX module manifest not in extra-files, so its "
        "version never follows the release"
      )
    if not self.errors:
      print(
        f"check_helia_release_versions: {len(listed)} extra-files at {version}"
      )

  def find_one(self, rel, pattern, text):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
      self.error(
        f"{rel}: expected one match for /{pattern}/, found {len(matches)}"
      )
      return None
    return matches[0]

  def check_ns_cmsis_nn_pin(self):
    inc = self.read(HELIA_INC)
    if inc is None:
      return
    pin = {
      "tag": self.find_one(HELIA_INC, PIN_TAG.pattern, inc),
      "commit": self.find_one(HELIA_INC, PIN_COMMIT.pattern, inc),
    }
    if None in pin.values():
      return
    before = len(self.errors)
    for rel, pattern, kind in PIN_SITES:
      text = self.read(rel)
      if text is None:
        continue
      value = self.find_one(rel, pattern, text)
      if value is not None and value != pin[kind]:
        self.error(
          f"{rel}: ns-cmsis-nn {kind} {value}; {HELIA_INC} pins {pin[kind]}"
        )
    if len(self.errors) == before:
      print(
        f"check_helia_release_versions: {len(PIN_SITES)} ns-cmsis-nn "
        f"pin sites at {pin['tag']} ({pin['commit'][:12]})"
      )


def main(argv):
  if len(argv) > 2:
    print(f"usage: {Path(argv[0]).name} [repo-root]", file=sys.stderr)
    return 2
  root = (
    Path(argv[1]) if len(argv) == 2 else Path(__file__).resolve().parents[5]
  )
  if not (root / ".release-please-manifest.json").is_file():
    print(
      f"error: '{root}' has no .release-please-manifest.json", file=sys.stderr
    )
    return 2
  checker = Checker(root)
  checker.check_release_version()
  checker.check_ns_cmsis_nn_pin()
  for message in checker.errors:
    print(f"error: {message}", file=sys.stderr)
  return 1 if checker.errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
