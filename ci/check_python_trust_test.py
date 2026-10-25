"""Regression checks for package floors and revoked-root rejection."""

from pathlib import Path
import unittest

import certifi
import check_python_trust as trust


class PythonTrustTest(unittest.TestCase):
  def test_rejects_revoked_root_even_with_current_packages(self):
    with self.assertRaisesRegex(ValueError, "Revoked emSign"):
      trust.check_bundle(Path(__file__).parent / "testdata/emsign_g3.pem")

  def test_accepts_current_bundle(self):
    trust.check_bundle(Path(certifi.where()))

  def test_rejects_old_and_prerelease_tools(self):
    for version in ["26.2.1", "26.2.2rc1"]:
      with self.subTest(version=version):
        with self.assertRaises(ValueError):
          trust.check_version("pip", version, "26.2.2")
    self.assertEqual("26.2.2", trust.check_version("pip", "26.2.2", "26.2.2"))


if __name__ == "__main__":
  unittest.main()
