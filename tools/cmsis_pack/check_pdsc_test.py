"""Regression checks for the advertised HELIA source dependency floor."""

import unittest
from unittest import mock

import build_pack
import check_pdsc


class DependencyFloorTest(unittest.TestCase):
  def make_pdsc(self):
    manifests = {
      backend: build_pack.BackendManifest(
        backend=backend,
        version="2.0.0",
        include_dirs=(),
        common_sources=(),
        kernel_sources=("cmake/helia_rt_core_version.cc",)
        if backend == "helia"
        else (),
        backend_defines=(),
      )
      for backend, _, _ in build_pack.BACKENDS
    }
    return build_pack.build_pdsc(
      version="2.0.0",
      manifests=manifests,
      sources=set(),
      headers=set(),
      include_dirs=set(),
    ).getroot()

  def test_current_generated_contract(self):
    self.assertEqual([], check_pdsc.check_contract(self.make_pdsc()))

  def test_old_floor_rejected_even_when_generator_and_xml_agree(self):
    with (
      mock.patch.object(build_pack, "NS_CMSIS_NN_MIN_VERSION", "7.39.2"),
      mock.patch.object(check_pdsc, "NS_CMSIS_NN_MIN_VERSION", "7.39.2"),
    ):
      pdsc = self.make_pdsc()
      self.assertEqual(
        "7.39.2", pdsc.find("conditions/condition/require").get("Cversion")
      )
      failures = check_pdsc.check_contract(pdsc)
    self.assertTrue(any("compiled HELIA source floor" in f for f in failures))


if __name__ == "__main__":
  unittest.main()
