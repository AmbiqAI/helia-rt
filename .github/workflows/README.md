# CI Image Pinning

The CI Docker image (`ghcr.io/ambiqai/helia-rt-ci`) is referenced by every
workflow that uses `container:` or runs `docker run` against it.

## Why pinning matters

Pinning to `:latest` is non-reproducible: two runs of the same commit can pull
different images if `:latest` is repushed in between, and there's no way to
roll back a bad image. Worse, `:latest` is rebuilt by *manual dispatch* of the
image build workflow and is not a `needs:` of the workflows that consume it,
so the published image can drift from `.devcontainer/Dockerfile` with no
commit recording the change and no run recording which image it used.

## Current state

**Every pin point carries a digest**, not a tag:

```
ghcr.io/ambiqai/helia-rt-ci@sha256:<64 hex>  # ghcr.io/ambiqai/helia-rt-ci:latest as of <date>
```

The human-readable tag survives as a trailing comment on the same line, so the
line stays greppable and reviewers can see which tag the digest came from.
A digest is immutable, so the image becomes a reviewable part of the diff.

Two rules follow from this and both matter:

- **All five pin points must carry the same digest.** They are separate
  literals in separate files. If a bump updates some and not others, the test
  matrix and the release build run on *different* images — release artifacts
  built by an image nothing tested. The pin check below fails such a PR; bump
  them together, in one PR.
- **Pin the image index digest, not a per-architecture manifest digest.**
  The recipe below returns the index digest, which preserves multi-arch
  resolution exactly as the tag did.

`:latest` remains the local-dev / manual-debug convenience tag.

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
        run: echo "image=ghcr.io/ambiqai/helia-rt-ci@sha256:<DIGEST>" >> "$GITHUB_OUTPUT"  # :latest as of <date>

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
publishes immutable `sha-<short>` tags on every Dockerfile change, plus a
floating `:latest` tag on `main`.

## The five pin points

| # | File | Form |
| - | ---- | ---- |
| 1 | [ci.yml](ci.yml) | `ci-image` setup job |
| 2 | [helia_build.yml](helia_build.yml) | `ci-image` setup job |
| 3 | [helia_release.yml](helia_release.yml) | `ci-image` setup job |
| 4 | [helia_test.yml](helia_test.yml) | `ci-image` setup job |
| 5 | [check_tflite_files.yml](check_tflite_files.yml) | workflow-level `env.CI_IMAGE` |

`tensorflow/lite/micro/tools/ci_build/check_helia_ci_image_pins.sh` enforces
this on every PR, from the `ci-image` job of [helia_test.yml](helia_test.yml).
Outside YAML comments it fails when a listed pin point does not reference the
image exactly once as `helia-rt-ci@sha256:<64 lowercase hex>`, when the pin
points disagree, or when any other workflow references the image by digest,
tag or bare name (the bare repository name in
[helia_build_docker_image.yml](helia_build_docker_image.yml) is the one
exception). Comments are recognised line by line, so a ` #` inside a shell or
quoted string before the image can hide a reference there. It checks the PR
head, not the merge result, so two PRs that each pass can still combine
unevenly; a later PR based on that `main` then fails. Run it locally before
opening a bump PR:

```sh
./tensorflow/lite/micro/tools/ci_build/check_helia_ci_image_pins.sh
```

Adding a pin point means adding it to `PIN_POINTS` in that script and to the
table above.

## Resolving a digest

The package is public, so the tag's current digest resolves anonymously with
no `read:packages` scope and without pulling the image:

```sh
TOKEN=$(curl -sSL \
  "https://ghcr.io/token?scope=repository:ambiqai/helia-rt-ci:pull&service=ghcr.io" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -sSI -H "Authorization: Bearer ${TOKEN}" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
  https://ghcr.io/v2/ambiqai/helia-rt-ci/manifests/latest \
  | grep -i docker-content-digest
```

Substitute any tag (`sha-abc1234`, a release tag) for `latest`. Both media
types are offered so the multi-arch manifest still resolves if the image is
ever republished in Docker rather than OCI format — requesting only the OCI
index would 404 against a Docker-format manifest list.

`docker buildx imagetools inspect ghcr.io/ambiqai/helia-rt-ci:latest` works
too if you have Docker to hand; the curl form is here because CI containers
and review checkouts generally do not.

## Bumping the pin

When you change `.devcontainer/Dockerfile` (or otherwise want a new image):

1. Merge the Dockerfile change to `main` (or push to a branch). The image
   build workflow runs and publishes `ghcr.io/ambiqai/helia-rt-ci:sha-abc1234`
   and, on `main`, repoints `:latest`.
2. Resolve the new digest with the recipe above.
3. Open a PR that updates **all five** pin points to that one digest, keeping
   each trailing `# ghcr.io/ambiqai/helia-rt-ci:<tag> as of <date>` comment in
   step so the provenance of the digest stays visible.
4. CI runs against the new image *as a PR check*. If anything breaks, the
   PR fails — the bad image never reaches `main`.
5. Merge the bump PR.

A quick mechanical bump (limited to YAML files so it doesn't touch this
README or other docs), which by construction cannot update some pin points
and miss others:

```sh
OLD=$(grep -rho 'helia-rt-ci@sha256:[0-9a-f]*' .github/workflows/*.yml | sort -u)
# Exactly one distinct digest, or refuse. `grep -c .` counts non-empty lines,
# so this rejects "none found" as well as "more than one" -- `wc -l` on an
# unterminated string reports 0 for both cases and would wave the first through.
test "$(printf '%s\n' "${OLD}" | grep -c .)" -eq 1 \
  || { echo "pin points disagree, or no digest found"; exit 1; }
NEW=helia-rt-ci@sha256:<new digest>
# `sed -i.bak` is the portable spelling: BSD sed (macOS) requires an argument
# to -i, GNU sed requires it be attached. `sed -i ''` works only on BSD -- GNU
# reads the '' as an empty script, errors per file, and edits nothing.
grep -rl --include='*.yml' "${OLD}" .github/workflows/ \
  | xargs sed -i.bak "s|${OLD}|${NEW}|g"
rm -f .github/workflows/*.yml.bak
```

Then run `check_helia_ci_image_pins.sh` (above), and update each trailing
tag comment's date by hand.

## Local dev

The dev container ([.devcontainer/](../../.devcontainer/)) builds its own
image from `.devcontainer/Dockerfile` and is unaffected by this pinning
scheme.
