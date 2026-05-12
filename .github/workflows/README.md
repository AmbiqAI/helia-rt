# CI Image Pinning

The CI Docker image (`ghcr.io/ambiqai/helia-rt-ci`) is referenced by every
workflow that uses `container:` or runs `docker run` against it.

## Why pinning matters

Pinning to `:latest` is non-reproducible: two runs of the same commit can pull
different images if `:latest` is repushed in between, and there's no way to
roll back a bad image. The mechanism below lets us move to immutable
`sha-<short>` tags (or release tags) through reviewable one-line PRs while
keeping `:latest` as a fallback for local debugging.

## Current state

The initial value remains `ghcr.io/ambiqai/helia-rt-ci:latest` in every
workflow. Today CI still tracks `:latest` — the next step is to flip the
value in each pin point to a `sha-<short>` tag (see "Bumping the pin"
below). `:latest` then becomes the local-dev / manual-debug fallback only.

## How it's wired

GitHub Actions does **not** allow the `${{ env.* }}` context inside
`jobs.<job_id>.container.image`, so each workflow that uses `container:`
declares a tiny `ci-image` setup job that emits the pinned tag as an output.
Downstream jobs depend on it and resolve the image via
`${{ needs.ci-image.outputs.image }}`:

```yaml
jobs:
  ci-image:
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.image.outputs.image }}
    steps:
      - id: image
        run: echo "image=ghcr.io/ambiqai/helia-rt-ci:<TAG>" >> "$GITHUB_OUTPUT"

  some_job:
    needs: ci-image
    runs-on: ubuntu-latest-md
    container:
      image: ${{ needs.ci-image.outputs.image }}
```

`check_tflite_files.yml` is the one exception: it invokes the image inside
an inline `docker run` (a `run:` step, where the `env` context *is*
allowed), so it keeps a workflow-level `env.CI_IMAGE` instead.

The image build workflow ([helia_build_docker_image.yml](helia_build_docker_image.yml))
already publishes immutable `sha-<short>` tags on every Dockerfile change,
plus a floating `:latest` tag on `main`.

## Bumping the pin

When you change `.devcontainer/Dockerfile` (or otherwise want a new image):

1. Merge the Dockerfile change to `main` (or push to a branch). The image
   build workflow runs and publishes `ghcr.io/ambiqai/helia-rt-ci:sha-abc1234`.
2. Open a one-line PR that updates the pinned tag in each pin point:
   - 4× `ci-image` setup jobs (one per file): [ci.yml](ci.yml), [helia_build.yml](helia_build.yml), [helia_release.yml](helia_release.yml), [helia_test.yml](helia_test.yml)
   - 1× `env.CI_IMAGE` value: [check_tflite_files.yml](check_tflite_files.yml)
3. CI runs against the new image *as a PR check*. If anything breaks, the
   PR fails — the bad image never reaches `main`.
4. Merge the bump PR.

A quick mechanical bump (limited to YAML files so it doesn't touch this
README or other docs):

```sh
OLD=ghcr.io/ambiqai/helia-rt-ci:latest
NEW=ghcr.io/ambiqai/helia-rt-ci:sha-abc1234
grep -rl --include='*.yml' "$OLD" .github/workflows/ | xargs sed -i "s|$OLD|$NEW|g"
```

## Local dev

The dev container ([.devcontainer/](../../.devcontainer/)) builds its own
image from `.devcontainer/Dockerfile` and is unaffected by this pinning
scheme.
