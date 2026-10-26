"""Regression checks for package floors and revoked-root rejection."""

from pathlib import Path
import os
import tempfile
import ssl
import unittest
from unittest.mock import patch

import certifi
import check_python_trust as trust


class PythonTrustTest(unittest.TestCase):
  def test_rejects_revoked_root_even_with_current_packages(self):
    with self.assertRaisesRegex(ValueError, "Revoked emSign"):
      trust.check_bundle(Path(__file__).parent / "testdata/emsign_g3.pem")

  def test_rejects_openssl_trusted_certificate_encoding(self):
    bundle = Path(__file__).parent / "testdata/emsign_g3_trusted.pem"
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=str(bundle))
    self.assertEqual(context.cert_store_stats()["x509_ca"], 1)
    with self.assertRaisesRegex(ValueError, "Revoked emSign"):
      trust.check_bundle(bundle)

  def test_scans_all_custom_certificate_directories(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      first.mkdir()
      second.mkdir()
      bad = second / "revoked.pem"
      bad.write_bytes(
        (Path(__file__).parent / "testdata/emsign_g3.pem").read_bytes()
      )
      with patch.dict(
        os.environ, {"SSL_CERT_DIR": os.pathsep.join([str(first), str(second)])}
      ):
        bundles = trust.system_bundles()
        self.assertIn(bad, bundles)
        with self.assertRaisesRegex(ValueError, "Revoked emSign"):
          trust.check_bundle(bad)

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
