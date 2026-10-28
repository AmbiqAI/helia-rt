#!/usr/bin/env python3
"""Verify that a PR records an advancing upstream revision.

Usage: check_upstream_sync.py BASE HEAD
The comparison uses BASE's merge base with HEAD. The recorded revision must
advance the old pin and belong to the configured upstream main branch. Invalid
metadata, remote-query errors, and non-descendant revisions fail the check.
"""

import datetime
import pathlib
import re
import subprocess
import sys


def git(root, *args):
  return subprocess.run(
    ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
  ).stdout.strip()


def read_upstream(text):
  fields = {}
  for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    key, separator, value = line.partition(":")
    if (
      not separator
      or key not in ("repository", "commit", "date")
      or key in fields
    ):
      raise ValueError("invalid helia/UPSTREAM fields")
    fields[key] = value.strip()
  if set(fields) != {"repository", "commit", "date"}:
    raise ValueError("incomplete helia/UPSTREAM metadata")
  if fields["repository"] != "https://github.com/tensorflow/tflite-micro":
    raise ValueError("unexpected upstream repository")
  if not re.fullmatch(r"[0-9a-f]{40}", fields["commit"]):
    raise ValueError("upstream commit must be a full SHA")
  if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["date"]):
    raise ValueError("invalid upstream date")
  datetime.date.fromisoformat(fields["date"])
  return fields


def fetch_upstream_main(root, upstream):
  advertised = git(
    root, "ls-remote", "--exit-code", upstream, "refs/heads/main"
  )
  match = re.fullmatch(r"([0-9a-f]{40})\trefs/heads/main", advertised)
  if not match:
    raise ValueError("invalid upstream main advertisement")
  tip = match.group(1)
  git(
    root,
    "fetch",
    "--no-tags",
    "--no-recurse-submodules",
    "--filter=blob:none",
    upstream,
    tip,
  )
  return tip


def check_sync(root, base, head):
  ancestor = git(root, "merge-base", base, head)
  before = read_upstream(git(root, "show", f"{ancestor}:helia/UPSTREAM"))
  after = read_upstream(git(root, "show", f"{head}:helia/UPSTREAM"))
  if before["commit"] == after["commit"]:
    raise ValueError("helia/UPSTREAM does not advance the upstream commit")
  upstream_main = fetch_upstream_main(root, before["repository"])
  git(root, "merge-base", "--is-ancestor", before["commit"], after["commit"])
  git(root, "merge-base", "--is-ancestor", after["commit"], upstream_main)
  actual_date = git(root, "show", "-s", "--format=%cs", after["commit"])
  if after["date"] != actual_date:
    raise ValueError("helia/UPSTREAM date does not match the upstream commit")
  return (
    f"Recorded upstream sync: {before['commit'][:8]}..{after['commit'][:8]}"
  )


def main():
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  try:
    print(check_sync(pathlib.Path.cwd(), *sys.argv[1:]))
  except (ValueError, subprocess.CalledProcessError) as error:
    print(f"Not a verified upstream sync: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
