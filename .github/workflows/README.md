# CI Image Pinning

The CI Docker image (`ghcr.io/ambiqai/helia-rt-ci`) is referenced by every
workflow that uses `container:` or runs `docker run` against it.

## Why pinning matters

Pinning to `:latest` is non-reproducible: two runs of the same commit can pull
different images if `:latest` is repushed in between, and there's no way to
roll back a bad image. We pin to an immutable tag instead and bump it through
PRs so image changes go through code review and CI.

## How it's wired

Each consumer workflow declares a workflow-level env var:

```yaml
# Pinned CI image. Bump this on Dockerfile changes.
# See .github/workflows/README.md for the bump procedure.
env:
  CI_IMAGE: ghcr.io/ambiqai/helia-rt-ci:<TAG>
```

…and references it in `container.image` (or inline `docker run`) as
`${{ env.CI_IMAGE }}`.

The image build workflow ([helia_build_docker_image.yml](helia_build_docker_image.yml))
already publishes immutable `sha-<short>` tags on every Dockerfile change, plus
a floating `:latest` tag on `main` (used only for local dev / manual debugging).

## Bumping the pin

When you change `.devcontainer/Dockerfile` (or otherwise want a new image):

1. Merge the Dockerfile change to `main` (or push to your branch). The image
   build workflow runs and publishes `ghcr.io/ambiqai/helia-rt-ci:sha-abc1234`.
2. Open a one-line PR that updates the `CI_IMAGE:` value in **all five**
   workflow files to the new `sha-...` tag:
   - [ci.yml](ci.yml)
   - [check_tflite_files.yml](check_tflite_files.yml)
   - [helia_build.yml](helia_build.yml)
   - [helia_release.yml](helia_release.yml)
   - [helia_test.yml](helia_test.yml)
3. CI runs against the new image *as a PR check*. If anything breaks, the
   PR fails — the bad image never reaches `main`.
4. Merge the bump PR.

A quick mechanical bump:

```sh
OLD=ghcr.io/ambiqai/helia-rt-ci:latest
NEW=ghcr.io/ambiqai/helia-rt-ci:sha-abc1234
grep -rl "$OLD" .github/workflows/ | xargs sed -i "s|$OLD|$NEW|g"
```

## Local dev

The dev container ([.devcontainer/](../../.devcontainer/)) builds its own image
from `.devcontainer/Dockerfile` and is unaffected by this pinning scheme.
