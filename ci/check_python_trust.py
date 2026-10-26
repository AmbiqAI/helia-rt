#!/usr/bin/env python3
"""Check Python package and certificate stores without making network requests."""

import hashlib
import os
import json
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys

import pip
from pip._vendor import certifi as pip_certifi
from pip._vendor.packaging.version import Version

try:
  import certifi
except ModuleNotFoundError:
  certifi = None

# see AmbiqAI/helia-rt#509
REVOKED_ROOT = (
  "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
)


def check_bundle(path):
  context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
  context.load_verify_locations(cafile=str(path))
  for certificate in context.get_ca_certs(binary_form=True):
    if hashlib.sha256(certificate).hexdigest() == REVOKED_ROOT:
      raise ValueError(f"Revoked emSign ECC Root CA - G3 in {path}")


def check_version(name, actual, minimum):
  if Version(actual) < Version(minimum):
    raise ValueError(f"{name} {actual} is below {minimum}")
  return actual


def system_bundles():
  paths = ssl.get_default_verify_paths()
  bundles = set()
  if paths.cafile:
    bundles.add(Path(paths.cafile))
  capath = os.environ.get(paths.openssl_capath_env, paths.openssl_capath) or ""
  for directory in capath.split(os.pathsep):
    if not directory or not Path(directory).is_dir():
      continue
    for candidate in Path(directory).iterdir():
      if candidate.suffix == ".pem" or re.fullmatch(
        r"[0-9a-f]{8}\.\d+", candidate.name
      ):
        bundles.add(candidate.resolve())
  return bundles


def main():
  report = {
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "pip": check_version("pip", pip.__version__, "26.2.2"),
    "certifi": check_version("certifi", certifi.__version__, "2026.10.19")
    if certifi
    else None,
    "pip_certifi": check_version(
      "pip.certifi", pip_certifi.__version__, "2026.10.19"
    ),
    "uv": None,
  }
  uv = shutil.which("uv")
  if uv:
    output = subprocess.check_output([uv, "--version"], text=True)
    report["uv"] = check_version("uv", output.split()[1], "0.13.4")
  paths = ssl.get_default_verify_paths()
  bundles = {Path(pip_certifi.where())}
  if certifi:
    bundles.add(Path(certifi.where()))
  bundles.update(system_bundles())
  for bundle in bundles:
    check_bundle(bundle)
  report["certificate_bundles_checked"] = len(bundles)
  report["openssl"] = ssl.OPENSSL_VERSION
  report["default_verify_paths"] = paths._asdict()
  if shutil.which("dpkg-query"):
    report["ca_certificates"] = subprocess.check_output(
      ["dpkg-query", "-W", "-f=${Version}", "ca-certificates"], text=True
    ).strip()
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
