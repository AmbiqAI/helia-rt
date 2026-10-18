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
"""Self-test for check_helia_release_versions.py.

Each case copies the files the check reads from this repository into a
scratch tree, edits it, and expects the check's exit status and, for an
inconsistent tree, a fragment of the error it prints. Versions and pins are
read from the copy, so the cases follow every release and pin bump.
see AmbiqAI/helia-rt#445, AmbiqAI/helia-rt#402
"""

import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_helia_release_versions as check  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[5]
CONFIG = "release-please-config.json"
MANIFEST = ".release-please-manifest.json"
ZEPHYR = "astro-site/src/content/docs/getting-started/zephyr.md"
CMSIS_PACK = "astro-site/src/content/docs/getting-started/cmsis-pack.md"
DOWNLOAD = "tensorflow/lite/micro/tools/make/ext_libs/ns_cmsis_nn_download.sh"
CANARY = ".github/workflows/ns_cmsis_nn_canary.yml"
SMOKE = ".github/workflows/smoke_cmake.yml"
NSX = "nsx/nsx-module.yaml"
NSX_SOURCE = "nsx-source/nsx-module.yaml"


def extra_files(root):
  config = json.loads((root / CONFIG).read_text())
  return [
    e if isinstance(e, str) else e["path"]
    for e in config["packages"]["."]["extra-files"]
  ]


class ReleaseVersionsTest(unittest.TestCase):
  def setUp(self):
    self.scratch = tempfile.TemporaryDirectory(prefix="helia-rt-versions-")
    self.root = Path(self.scratch.name)
    files = {CONFIG, MANIFEST, check.HELIA_INC, NSX_SOURCE}
    files.update(extra_files(ROOT))
    files.update(path for path, _, _ in check.PIN_SITES)
    for rel in files:
      (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(ROOT / rel, self.root / rel)
    self.version = json.loads((self.root / MANIFEST).read_text())["."]
    inc = (self.root / check.HELIA_INC).read_text()
    self.tag = check.PIN_TAG.search(inc).group(1)
    self.commit = check.PIN_COMMIT.search(inc).group(1)

  def tearDown(self):
    self.scratch.cleanup()

  def edit(self, rel, old, new, count=1):
    """Replaces `count` occurrences (0 = all) of the regex `old` in `rel`."""
    path = self.root / rel
    text = path.read_text()
    edited, n = re.subn(old, new, text, count=count, flags=re.M)
    self.assertGreater(n, 0, f"edit /{old}/ matched nothing in {rel}")
    path.write_text(edited)

  def write(self, rel, text):
    (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
    (self.root / rel).write_text(text)

  def set_extra_files(self, entries):
    config = json.loads((self.root / CONFIG).read_text())
    config["packages"]["."]["extra-files"] = entries
    self.write(CONFIG, json.dumps(config, indent=2))

  def run_check(self):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
      status = check.main(["check", str(self.root)])
    return status, err.getvalue()

  def assert_consistent(self):
    status, err = self.run_check()
    self.assertEqual(0, status, err)

  def assert_inconsistent(self, fragment):
    status, err = self.run_check()
    self.assertEqual(1, status, err)
    self.assertIn(fragment, err)

  def bump_release(self, new):
    old = re.escape(self.version)
    self.edit(MANIFEST, old, new)
    for rel in extra_files(self.root):
      self.edit(rel, old, new, count=0)

  # Consistent trees.

  def test_unchanged_tree(self):
    self.assert_consistent()

  def test_release_bump_everywhere(self):
    self.bump_release("9.10.11")
    self.assert_consistent()

  def test_pin_bump_everywhere(self):
    for rel in {check.HELIA_INC, ZEPHYR, DOWNLOAD, CANARY, SMOKE}:
      self.edit(rel, re.escape(self.tag) + r"\b", "v9.10.11", count=0)
    for rel in {check.HELIA_INC, DOWNLOAD, CANARY}:
      self.edit(rel, self.commit, "f" * 40, count=0)
    self.assert_consistent()

  def test_untracked_marked_file_in_a_git_checkout(self):
    if shutil.which("git") is None:
      self.skipTest("git not installed")
    git = ["git", "-C", str(self.root)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    self.write(
      ".scratch/vendored/nsx-module.yaml",
      f'version: "0.0.1" # {check.MARKER}\n',
    )
    self.assert_consistent()

  # Release version.

  def test_manifest_bumped_alone(self):
    self.edit(MANIFEST, re.escape(self.version), "9.10.11")
    self.assert_inconsistent("the manifest is at 9.10.11")

  def test_stale_marked_line(self):
    self.edit(NSX, re.escape(self.version), "1.17.0")
    self.assert_inconsistent(f"{NSX}:6: version 1.17.0")

  def test_stale_version_inside_block(self):
    self.edit(
      CMSIS_PACK,
      r"(dist/Ambiq\.helia-rt\.)" + re.escape(self.version) + r"(\.pack)",
      r"\g<1>1.17.0\g<2>",
    )
    self.assert_inconsistent("version 1.17.0 where release-please writes")

  def test_marker_removed_from_listed_file(self):
    self.edit(NSX_SOURCE, r" # " + check.MARKER, "")
    self.assert_inconsistent(f"{NSX_SOURCE}: listed in extra-files but has no")

  def test_marked_line_without_version(self):
    self.edit(NSX, r'"' + re.escape(self.version) + r'"', '"latest"')
    self.assert_inconsistent(f"{NSX}:6: marked line or block holds no X.Y.Z")

  def test_unclosed_block(self):
    self.edit(CMSIS_PACK, r"^<!-- " + check.BLOCK_END + r" -->\n", "", count=0)
    self.assert_inconsistent("version block is never closed")

  def test_nsx_module_dropped_from_extra_files(self):
    # The AmbiqAI/helia-rt#445 tree: the module manifest is not bumped.
    self.set_extra_files(
      [
        e
        for e in json.loads((self.root / CONFIG).read_text())["packages"]["."][
          "extra-files"
        ]
        if e["path"] != NSX_SOURCE
      ]
    )
    self.edit(NSX_SOURCE, r" # " + check.MARKER, "")
    self.assert_inconsistent(
      f"{NSX_SOURCE}: NSX module manifest not in extra-files"
    )

  def test_new_nsx_module_unlisted(self):
    self.write("nsx-extra/nsx-module.yaml", 'version: "1.0.0"\n')
    self.assert_inconsistent(
      "nsx-extra/nsx-module.yaml: NSX module manifest not in extra-files"
    )

  def test_marked_file_unlisted(self):
    self.write("docs/new.md", f"v{self.version} <!-- {check.MARKER} -->\n")
    self.assert_inconsistent("docs/new.md: carries")

  def test_listed_file_missing(self):
    (self.root / NSX).unlink()
    self.assert_inconsistent(f"{NSX}: listed in extra-files but missing")

  def test_non_generic_entry(self):
    self.set_extra_files(
      [{"type": "json", "path": "package.json", "jsonpath": "$.version"}]
    )
    self.assert_inconsistent("is not a generic file")

  def test_manifest_with_two_packages(self):
    self.write(MANIFEST, json.dumps({".": self.version, "x": "1.0.0"}))
    self.assert_inconsistent("expected exactly")

  # ns-cmsis-nn pin (AmbiqAI/helia-rt#402).

  def test_zephyr_guide_revision_differs(self):
    self.edit(
      ZEPHYR, r"^(  revision: )" + re.escape(self.tag) + "$", r"\g<1>v7.0.0"
    )
    self.assert_inconsistent(f"{ZEPHYR}: ns-cmsis-nn tag v7.0.0")

  def test_smoke_ref_differs(self):
    self.edit(SMOKE, re.escape(self.tag) + "$", "v7.0.0")
    self.assert_inconsistent(f"{SMOKE}: ns-cmsis-nn tag v7.0.0")

  def test_download_default_commit_differs(self):
    self.edit(DOWNLOAD, self.commit, "0" * 40)
    self.assert_inconsistent(f"{DOWNLOAD}: ns-cmsis-nn commit " + "0" * 40)

  def test_canary_default_commit_differs(self):
    self.edit(CANARY, self.commit, "0" * 40)
    self.assert_inconsistent(f"{CANARY}: ns-cmsis-nn commit " + "0" * 40)

  def test_pin_bumped_in_helia_inc_alone(self):
    self.edit(check.HELIA_INC, self.commit, "f" * 40)
    self.assert_inconsistent(f"pins {'f' * 40}")

  def test_pin_site_duplicated(self):
    self.edit(SMOKE, r"^(\s*NS_CMSIS_NN_REF: .*\n)", r"\g<1>\g<1>")
    self.assert_inconsistent(f"{SMOKE}: expected one match")

  def test_pin_tag_comment_missing(self):
    self.edit(check.HELIA_INC, r"^# Tracks AmbiqAI/ns-cmsis-nn tag ", "# ")
    self.assert_inconsistent(f"{check.HELIA_INC}: expected one match")


if __name__ == "__main__":
  unittest.main()
