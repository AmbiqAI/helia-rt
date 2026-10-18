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
`extra-files`, and in each only the first version on a marked line or on a
line inside a marked block. A versioned file missing from that list, or a
marked line whose first version is not the release, ships stale. The
ns-cmsis-nn tag and commit are written by hand in several files, and nothing
bumps them together.
see AmbiqAI/helia-rt#445, AmbiqAI/helia-rt#402

Rules:
  * .release-please-manifest.json holds one package, ".", at a version.
  * Every extra-files entry is {"type": "generic", "path": ...} for a file
    that exists and has a version-scope marked line or block. A marked line
    holds exactly one version, a block line at most one, each block at least
    one and its start line none, and every one is the manifest version.
  * Every tracked file that carries a release-please marker, and every
    nsx-module.yaml, is an extra-files entry.
  * The ns-cmsis-nn tag and commit in ext_libs/helia.inc match the copies in
    PIN_SITES, each of which appears exactly once in its file. With --remote,
    the tag must also resolve to the commit in AmbiqAI/ns-cmsis-nn.

Usage: check_helia_release_versions.py [--remote] [repo-root]
  repo-root defaults to the repository root inferred from this file.

Exit codes: 0 consistent, 1 inconsistent, 2 usage error.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# The marker regexes and VERSION_RE are release-please's own; it applies
# VERSION_RE once per line, not globally. Each pattern is split so this file
# carries no marker itself. see googleapis/release-please
# src/updaters/generic.ts
MARKER = "x-release-please-" + "version"
SCOPES = "major|minor|patch|version-date|version|date"
INLINE_RE = re.compile("x-release-please-" + f"(?P<scope>{SCOPES})", re.A)
BLOCK_START_RE = re.compile(
  "x-release-please-start-" + f"(?P<scope>{SCOPES})", re.A
)
BLOCK_END_RE = re.compile("x-release-please-" + "end")
ANY_MARKER_RE = re.compile(
  ("x-release-please-" + f"(?:start-)?(?:{SCOPES})").encode()
)
VERSION_RE = re.compile(
  r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
  r"(-(?P<preRelease>[\w.]+))?(\+(?P<build>[-\w.]+))?",
  re.A,
)

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

NS_CMSIS_NN_URL = "https://github.com/AmbiqAI/ns-cmsis-nn"
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
  (
    "tensorflow/lite/micro/kernels/helia/README.md",
    r"provided by the pinned\n`AmbiqAI/ns-cmsis-nn` (v\d+\.\d+\.\d+) release",
    "tag",
  ),
)


def remote_tag_commit(tag, attempts=3):
  """Returns the commit `tag` names in AmbiqAI/ns-cmsis-nn, or None."""
  command = [
    "git",
    "ls-remote",
    NS_CMSIS_NN_URL,
    f"refs/tags/{tag}",
    f"refs/tags/{tag}^{{}}",
  ]
  for attempt in range(attempts):
    try:
      out = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=60
      ).stdout
      break
    except (OSError, subprocess.SubprocessError):
      if attempt + 1 == attempts:
        return None
      time.sleep(5 * (attempt + 1))
  refs = {}
  for line in out.splitlines():
    sha, _, ref = line.partition("\t")
    if re.fullmatch(r"[0-9a-f]{40}", sha) and ref:
      refs[ref] = sha
  # An annotated tag lists its peeled commit as <ref>^{}.
  return refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")


class Checker:
  def __init__(self, root):
    self.root = root
    self.errors = []

  def error(self, message):
    self.errors.append(message)

  def read(self, rel):
    try:
      # Bytes, so a lone \r stays inside its line as release-please sees it.
      return (self.root / rel).read_bytes().decode("utf-8")
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
      or not VERSION_RE.fullmatch(manifest["."])
    ):
      self.error(
        '.release-please-manifest.json: expected exactly {".": "<version>"}'
      )
      return None
    return manifest["."]

  def extra_files(self):
    text = self.read("release-please-config.json")
    if text is None:
      return None
    try:
      entries = json.loads(text)["packages"]["."]["extra-files"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
      self.error(
        f'release-please-config.json: no packages.".".extra-files ({e!r})'
      )
      return None
    if not isinstance(entries, list):
      self.error("release-please-config.json: extra-files is not a list")
      return None
    paths = []
    for entry in entries:
      path = entry.get("path") if isinstance(entry, dict) else None
      # A leading / is the repository root; the package is ".", so the same.
      rel = path.lstrip("/") if isinstance(path, str) else ""
      if (
        not isinstance(entry, dict)
        or set(entry) != {"type", "path"}
        or entry["type"] != "generic"
        or not rel
        or ".." in Path(rel).parts
      ):
        self.error(
          "release-please-config.json: extra-files entry "
          f'{json.dumps(entry)} is not {{"type": "generic", "path": ...}} '
          "for a repository file; this check models only that updater"
        )
        continue
      paths.append(rel)
    return paths

  def marked_lines(self, rel, text):
    """Yields (line number, line, kind) for each line release-please
    rewrites, kind "inline" or "block", and (number, None, "end") after each
    block. Mirrors Generic.updateContent."""
    block_start = None
    for number, line in enumerate(re.split(r"\r?\n", text), 1):
      inline = INLINE_RE.search(line)
      if inline:
        if inline.group("scope") != "version":
          self.error(f"{rel}:{number}: unsupported marker scope")
        yield number, line, "inline"
      elif block_start is not None:
        if BLOCK_START_RE.search(line):
          self.error(f"{rel}:{number}: version block opened inside another")
        yield number, line, "block"
        if BLOCK_END_RE.search(line):
          yield block_start, None, "end"
          block_start = None
      else:
        start = BLOCK_START_RE.search(line)
        if start:
          if start.group("scope") != "version":
            self.error(f"{rel}:{number}: unsupported marker scope")
          if VERSION_RE.search(line):
            self.error(
              f"{rel}:{number}: version on a block-start line, which "
              "release-please never rewrites"
            )
          block_start = number
    if block_start is not None:
      self.error(f"{rel}:{block_start}: version block is never closed")

  def check_extra_file(self, rel, version):
    if not (self.root / rel).is_file():
      self.error(f"{rel}: listed in extra-files but missing")
      return
    text = self.read(rel)
    if text is None:
      return
    units = 0
    block_versions = 0
    for number, line, kind in self.marked_lines(rel, text):
      if kind == "end":
        units += 1
        if block_versions == 0:
          self.error(f"{rel}:{number}: version block holds no version")
        block_versions = 0
        continue
      found = [m.group(0) for m in VERSION_RE.finditer(line)]
      if kind == "inline":
        units += 1
        if not found:
          self.error(f"{rel}:{number}: marked line holds no version")
      else:
        block_versions += len(found)
      if len(found) > 1:
        self.error(
          f"{rel}:{number}: {len(found)} versions on one line; "
          "release-please rewrites only the first"
        )
      for value in found:
        if value != version:
          self.error(
            f"{rel}:{number}: version {value} where release-please "
            f"writes the release; the manifest is at {version}"
          )
    if units == 0:
      self.error(
        f"{rel}: listed in extra-files but has no {MARKER} line or "
        "block, so release-please never bumps it"
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
      yield from (os.fsdecode(p) for p in listed.split(b"\0") if p)
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
      if ANY_MARKER_RE.search(data):
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
        f"{rel}: carries a release-please marker but is not in extra-files, "
        "so release-please never bumps it"
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

  def check_ns_cmsis_nn_pin(self, remote):
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
    if remote:
      resolved = remote_tag_commit(pin["tag"])
      if resolved is None:
        self.error(f"could not resolve {pin['tag']} in {NS_CMSIS_NN_URL}")
      elif resolved != pin["commit"]:
        self.error(
          f"{HELIA_INC}: {pin['tag']} is {resolved} in {NS_CMSIS_NN_URL}, "
          f"but the pinned commit is {pin['commit']}"
        )
    if len(self.errors) == before:
      print(
        f"check_helia_release_versions: {len(PIN_SITES)} ns-cmsis-nn "
        f"pin sites at {pin['tag']} ({pin['commit'][:12]})"
        + (", tag resolved upstream" if remote else "")
      )


def main(argv):
  args = argv[1:]
  remote = "--remote" in args
  args = [a for a in args if a != "--remote"]
  if len(args) > 1 or any(a.startswith("-") for a in args):
    print(
      f"usage: {Path(argv[0]).name} [--remote] [repo-root]", file=sys.stderr
    )
    return 2
  root = Path(args[0]) if args else Path(__file__).resolve().parents[5]
  if not (root / ".release-please-manifest.json").is_file():
    print(
      f"error: '{root}' has no .release-please-manifest.json", file=sys.stderr
    )
    return 2
  checker = Checker(root)
  checker.check_release_version()
  checker.check_ns_cmsis_nn_pin(remote)
  for message in checker.errors:
    print(f"error: {message}", file=sys.stderr)
  return 1 if checker.errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
