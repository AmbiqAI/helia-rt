#!/usr/bin/env python3
"""Audit declared Bazel Python runtimes and fresh venvs without a full build."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
  bazel = [
    sys.argv[1] if len(sys.argv) > 1 else "bazel",
    "--batch",
    "--host_jvm_args=-Xmx2048m",
  ]
  if len(sys.argv) > 2:
    bazel.append("--output_user_root=" + sys.argv[2])
  versions = ["3_10", "3_11", "3_12", "3_13", "3_14"]
  targets = ["@python_" + version + "//:bin/python3" for version in versions]
  subprocess.run(
    bazel
    + [
      "build",
      "--jobs=4",
      "--local_resources=cpu=4",
      "--local_resources=memory=2048",
      "--loading_phase_threads=4",
      "--http_max_parallel_downloads=4",
    ]
    + targets,
    check=True,
    stdout=sys.stderr,
  )
  execution_root = Path(
    subprocess.check_output(
      bazel + ["info", "execution_root"], text=True
    ).strip()
  )
  checker = Path(__file__).with_name("check_python_trust.py").resolve()
  report = {}
  with tempfile.TemporaryDirectory() as directory:
    for version, target in zip(versions, targets):
      binary = subprocess.check_output(
        bazel
        + ["cquery", target, "--output=files", "--loading_phase_threads=4"],
        text=True,
      ).strip()
      python = str(execution_root / binary)
      runtime = json.loads(
        subprocess.check_output([python, "-I", str(checker)], text=True)
      )
      venv = Path(directory) / version
      subprocess.run([python, "-I", "-m", "venv", str(venv)], check=True)
      child = json.loads(
        subprocess.check_output(
          [str(venv / "bin/python"), "-I", str(checker)], text=True
        )
      )
      report[version] = {"runtime": runtime, "venv": child}
  subprocess.run(
    bazel
    + [
      "run",
      "--jobs=4",
      "--local_resources=cpu=4",
      "--local_resources=memory=2048",
      "--loading_phase_threads=4",
      "//python:requirements_3_10.update",
      "--",
      "--help",
    ],
    check=True,
    stdout=sys.stderr,
  )
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
