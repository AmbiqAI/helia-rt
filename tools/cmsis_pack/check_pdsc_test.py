"""Regression checks for selected-source and packaged HELIA dependency floors."""

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest import mock

import build_pack
import check_pdsc


class MetadataComplete(Exception):
  pass


class DependencyFloorTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Retain the small metadata fixtures for inspection; no directory deletion.
    cls.scratch = Path(tempfile.mkdtemp(prefix="helia-rt-floor-"))
    source = Path(__file__).resolve().parents[2]
    guard = (source / build_pack.CORE_VERSION_GUARD).read_text()
    cls.roots = {}
    for floor, numeric in (("7.41.0", "7041000"), ("7.42.0", "7042000")):
      root = cls.scratch / floor
      (root / "cmake").mkdir(parents=True)
      (root / build_pack.CORE_VERSION_GUARD).write_text(
        guard.replace("7041000", numeric).replace("7.41.0", floor)
      )
      (root / "cmake/helia_rt_sources.cmake").write_text(
        "# Metadata fixture only\n"
      )
      cls.roots[floor] = root
    print(f"Retained metadata fixtures: {cls.scratch}")

  def manifests(self):
    return {
      backend: build_pack.BackendManifest(
        backend=backend,
        version="2.0.0",
        include_dirs=(),
        common_sources=(),
        kernel_sources=(build_pack.CORE_VERSION_GUARD,)
        if backend == "helia"
        else (),
        backend_defines=(),
      )
      for backend, _, _ in build_pack.BACKENDS
    }

  def make_pdsc(self, floor="7.41.0"):
    return build_pack.build_pdsc(
      version="2.0.0",
      core_min_version=floor,
      manifests=self.manifests(),
      sources=set(),
      headers=set(),
      include_dirs=set(),
    ).getroot()

  def test_current_generated_contract(self):
    floor = build_pack.core_source_min_version(
      Path(__file__).resolve().parents[2]
    )
    self.assertEqual(
      [], check_pdsc.check_contract(self.make_pdsc(floor), source_floor=floor)
    )

  def test_old_matching_advertised_metadata_rejected(self):
    pdsc = self.make_pdsc("7.39.2")
    self.assertEqual(
      "7.39.2", pdsc.find("conditions/condition/require").get("Cversion")
    )
    failures = check_pdsc.check_contract(pdsc, source_floor="7.41.0")
    self.assertTrue(any("compiled HELIA source floor" in f for f in failures))

  def test_build_pack_follows_both_selected_source_roots(self):
    real_build_pdsc = build_pack.build_pdsc
    for floor, root in self.roots.items():
      with self.subTest(floor=floor):
        stage = self.scratch / ("stage-" + floor)
        stage.mkdir()
        captured = []

        def stage_guard(selected_root, manifests, destination):
          target = destination / build_pack.CORE_VERSION_GUARD
          target.parent.mkdir()
          shutil.copyfile(selected_root / build_pack.CORE_VERSION_GUARD, target)
          return {build_pack.CORE_VERSION_GUARD}, set(), set()

        def capture_pdsc(**kwargs):
          pdsc = real_build_pdsc(**kwargs).getroot()
          captured.append(pdsc)
          ET.ElementTree(pdsc).write(stage / "probe.pdsc")
          raise MetadataComplete

        with (
          mock.patch.object(
            build_pack,
            "dump_manifest",
            side_effect=lambda selected, backend: self.manifests()[backend],
          ),
          mock.patch.object(
            build_pack, "stage_pack_files", side_effect=stage_guard
          ),
          mock.patch.object(
            build_pack.tempfile,
            "TemporaryDirectory",
            return_value=contextlib.nullcontext(str(stage)),
          ),
          mock.patch.object(build_pack, "build_pdsc", side_effect=capture_pdsc),
          self.assertRaises(MetadataComplete),
        ):
          build_pack.build_pack(root, self.scratch / ("output-" + floor))
        source_floor = build_pack.core_source_min_version(stage)
        self.assertEqual(floor, source_floor)
        self.assertEqual(
          floor,
          captured[0].find("conditions/condition/require").get("Cversion"),
        )
        self.assertEqual(
          [], check_pdsc.check_contract(captured[0], source_floor=source_floor)
        )

  def check_cli(self, *args):
    with (
      mock.patch.object(sys, "argv", ["check_pdsc.py", *map(str, args)]),
      contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      return check_pdsc.main()

  def test_plain_pdsc_uses_explicit_or_staged_source_root(self):
    for floor, root in self.roots.items():
      pdsc = root / "probe.pdsc"
      ET.ElementTree(self.make_pdsc(floor)).write(pdsc)
      self.assertEqual(0, self.check_cli(pdsc))
      self.assertEqual(0, self.check_cli(pdsc, "--repo-root", root))
    wrong = self.scratch / "wrong.pdsc"
    ET.ElementTree(self.make_pdsc("7.41.0")).write(wrong)
    self.assertEqual(
      1, self.check_cli(wrong, "--repo-root", self.roots["7.42.0"])
    )
    self.assertEqual(1, self.check_cli(wrong))

  def test_pack_uses_its_own_guard_and_rejects_wrong_tool_floor(self):
    for advertised, expected in (("7.42.0", 0), ("7.41.0", 1)):
      pack = self.scratch / (advertised + ".pack")
      with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr(
          "Ambiq.helia-rt.pdsc", ET.tostring(self.make_pdsc(advertised))
        )
        archive.write(
          self.roots["7.42.0"] / build_pack.CORE_VERSION_GUARD,
          build_pack.CORE_VERSION_GUARD,
        )
      self.assertEqual(expected, self.check_cli(pack))
    self.assertEqual(
      1, self.check_cli(pack, "--repo-root", self.roots["7.41.0"])
    )

  def test_generator_cli_preserves_repo_root_selection(self):
    result = self.roots["7.42.0"] / build_pack.CORE_VERSION_GUARD
    with (
      mock.patch.object(build_pack, "build_pack", return_value=result) as build,
      contextlib.redirect_stdout(io.StringIO()),
    ):
      self.assertEqual(
        0,
        build_pack.main(
          [
            "--repo-root",
            str(self.roots["7.42.0"]),
            "--output",
            str(self.scratch),
          ]
        ),
      )
    self.assertEqual(self.roots["7.42.0"], build.call_args.args[0])


if __name__ == "__main__":
  unittest.main()
