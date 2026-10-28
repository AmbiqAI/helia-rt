"""Regression checks for the upstream-sync scope exception."""

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import check_upstream_sync as scope


class UpstreamSyncTest(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = pathlib.Path(self.directory.name)
    self.git("init", "-q")
    self.git("config", "user.name", "Scope test")
    self.git("config", "user.email", "scope-test@example.invalid")
    self.old = self.commit("old upstream")
    self.new = self.commit("new upstream")
    self.date = self.git("show", "-s", "--format=%cs", self.new)
    self.git("checkout", "-q", "--detach", self.old)
    (self.root / "helia").mkdir()
    self.metadata(self.old)
    self.base = self.commit("fork base")

  def git(self, *args):
    return scope.git(self.root, *args)

  def commit(self, message):
    self.git("add", ".")
    self.git("commit", "-q", "--allow-empty", "-m", message)
    return self.git("rev-parse", "HEAD")

  def metadata(self, commit, repository=None, date=None):
    repository = repository or "https://github.com/tensorflow/tflite-micro"
    (self.root / "helia/UPSTREAM").write_text(
      f"repository: {repository}\ncommit: {commit}\ndate: {date or self.date}\n"
    )

  def verify(self, head, base=None):
    with mock.patch.object(scope, "fetch_commits") as fetch:
      result = scope.check_sync(self.root, base or self.base, head)
    fetch.assert_called_once_with(
      self.root,
      "https://github.com/tensorflow/tflite-micro",
      self.old,
      self.new,
    )
    return result

  def test_recorded_advance(self):
    self.metadata(self.new)
    head = self.commit("record sync")
    self.assertIn(f"{self.old[:8]}..{self.new[:8]}", self.verify(head))

  def test_ordinary_mixed_change_is_not_exempt(self):
    (self.root / "CMakeLists.txt").write_text("# changed manifest\n")
    (self.root / "tensorflow").mkdir()
    (self.root / "tensorflow/kernel.cc").write_text("// changed upstream\n")
    head = self.commit("mixed change")
    with mock.patch.object(scope, "fetch_commits") as fetch:
      with self.assertRaisesRegex(ValueError, "does not advance"):
        scope.check_sync(self.root, self.base, head)
    fetch.assert_not_called()

  def test_metadata_only_edit_is_not_exempt(self):
    self.metadata(self.old, date="2000-01-01")
    head = self.commit("edit date")
    with self.assertRaisesRegex(ValueError, "does not advance"):
      scope.check_sync(self.root, self.base, head)

  def test_changed_repository_is_not_exempt(self):
    self.metadata(self.new, repository="https://github.com/example/fork")
    head = self.commit("replace repository")
    with self.assertRaisesRegex(ValueError, "unexpected upstream repository"):
      scope.check_sync(self.root, self.base, head)

  def test_short_sha_is_not_exempt(self):
    self.metadata(self.new[:8])
    head = self.commit("short SHA")
    with self.assertRaisesRegex(ValueError, "full SHA"):
      scope.check_sync(self.root, self.base, head)

  def test_duplicate_metadata_is_not_exempt(self):
    self.metadata(self.new)
    with (self.root / "helia/UPSTREAM").open("a") as metadata:
      metadata.write(f"commit: {self.old}\n")
    head = self.commit("duplicate field")
    with self.assertRaisesRegex(ValueError, "invalid.*fields"):
      scope.check_sync(self.root, self.base, head)

  def test_backwards_revision_is_not_exempt(self):
    self.metadata(self.new)
    base = self.commit("record newer upstream")
    self.metadata(self.old)
    head = self.commit("move backwards")
    with mock.patch.object(scope, "fetch_commits"):
      with self.assertRaises(subprocess.CalledProcessError):
        scope.check_sync(self.root, base, head)

  def test_unknown_revision_is_not_exempt(self):
    self.metadata("f" * 40)
    head = self.commit("unknown SHA")
    with mock.patch.object(scope, "fetch_commits"):
      with self.assertRaises(subprocess.CalledProcessError):
        scope.check_sync(self.root, self.base, head)

  def test_unrelated_revision_is_not_exempt(self):
    self.git("checkout", "-q", "--orphan", "unrelated")
    unrelated = self.commit("unrelated history")
    self.git("checkout", "-q", "--detach", self.base)
    self.metadata(unrelated)
    head = self.commit("record unrelated revision")
    with mock.patch.object(scope, "fetch_commits"):
      with self.assertRaises(subprocess.CalledProcessError):
        scope.check_sync(self.root, self.base, head)

  def test_fetch_failure_is_not_exempt(self):
    self.metadata(self.new)
    head = self.commit("record sync")
    error = subprocess.CalledProcessError(1, ["git", "fetch"])
    with mock.patch.object(scope, "fetch_commits", side_effect=error):
      with self.assertRaises(subprocess.CalledProcessError):
        scope.check_sync(self.root, self.base, head)

  def test_wrong_commit_date_is_not_exempt(self):
    self.metadata(self.new, date="2000-01-01")
    head = self.commit("wrong date")
    with mock.patch.object(scope, "fetch_commits"):
      with self.assertRaisesRegex(ValueError, "date does not match"):
        scope.check_sync(self.root, self.base, head)

  def test_removed_metadata_is_not_exempt(self):
    (self.root / "helia/UPSTREAM").unlink()
    head = self.commit("remove record")
    with self.assertRaises(subprocess.CalledProcessError):
      scope.check_sync(self.root, self.base, head)

  def test_uses_merge_base_when_base_branch_advances(self):
    self.metadata(self.new)
    head = self.commit("record sync")
    self.git("checkout", "-q", "--detach", self.base)
    self.metadata(self.new)
    (self.root / "unrelated.txt").write_text("base branch change\n")
    advanced_base = self.commit("base branch advances")
    self.assertIn("Recorded upstream sync", self.verify(head, advanced_base))


if __name__ == "__main__":
  unittest.main()
