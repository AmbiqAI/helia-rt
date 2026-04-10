# Continuous Integration

This page describes the CI workflows that exist in this repository today and how they are typically used.

## CI Overview

The repository uses a mix of:

- PR-triggered workflows for routine validation
- label-gated workflows for heavier test coverage
- manually dispatched workflows for ad hoc runs
- release and documentation workflows

The main GitHub Actions workflow files live under `.github/workflows/`.

## Pull Request Validation

The main PR entry point is `tests_entry.yml`.

Current behavior:

- `ambiq_test.yml` runs for pull requests through [tests_entry.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/tests_entry.yml)
- the broader `ci.yml` suite is only invoked on PRs when the PR has the `ci:run_full` label

Relevant workflows:

- [tests_entry.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/tests_entry.yml)
- [ambiq_test.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/ambiq_test.yml)
- [ci.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/ci.yml)

## Label-Gated Full CI

The label used in the current repository is `ci:run_full`.

When a pull request has `ci:run_full`:

- `tests_entry.yml` calls the broader `ci.yml` workflow
- additional Cortex-M workflows are also wired to run on labeled PR events

Relevant workflows:

- [cortex_m.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/cortex_m.yml)
- [cortex_m_arm_compiler.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/cortex_m_arm_compiler.yml)

`ci:run` still appears in a few automation paths for bot-created PRs, but it is not the main user-facing full-CI label described by the active PR entry workflows.

## Post-Merge Validation

The post-merge entry point is `tests_post.yml`.

This workflow runs on closed pull requests and is intended for longer-running validation after merge. It currently fans out into:

- Docker image build updates
- Ambiq test runs
- Cortex-M validation

Relevant workflow:

- [tests_post.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/tests_post.yml)

## Manual Workflow Entry Points

Several workflows can be triggered manually with `workflow_dispatch`.

Common manual entry points:

- [run_ci.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/run_ci.yml): manual entry point for the reusable `ci.yml` workflow
- [run_ambiq.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/run_ambiq.yml): manual entry point for Ambiq build and test flows
- [cortex_m.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/cortex_m.yml): manual Cortex-M validation
- [cortex_m_arm_compiler.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/cortex_m_arm_compiler.yml): manual Arm Compiler 6 Cortex-M validation
- [docs.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/docs.yml): documentation build and publish workflow

## Release and Asset Workflows

Release automation and asset packaging are handled separately from routine presubmit CI.

Relevant workflows:

- [release-please.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/release-please.yml)
- [helia_release.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml)
- [zephyr_tflm_rt_assets.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/zephyr_tflm_rt_assets.yml) (manual dispatch)

## Sync Workflow

This repository also keeps a sync workflow for pulling shared changes from TensorFlow-related upstream sources.

Relevant workflow:

- [sync.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/sync.yml)

## Error Reporting

Several workflows call a shared issue-reporting workflow when a run fails in the main repository.

Relevant workflow:

- [issue_on_error.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/issue_on_error.yml)

## Local Validation

For local verification, the exact command depends on the workflow you are trying to mirror. In practice, the most useful approach is to inspect the referenced workflow file and then run the underlying script or build command locally.

For example, the reusable CI workflow delegates to scripts under:

- `tensorflow/lite/micro/tools/ci_build/`

## Notes

- This page documents the workflows that are present in the repository now.
- If a workflow file changes, this page should be updated to match the current trigger and entry-point model.
- GitHub Actions file links here intentionally point to GitHub rather than relative local paths, so the published docs site does not produce broken links.
