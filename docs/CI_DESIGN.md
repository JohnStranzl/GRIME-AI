# GRIME AI CI Design

## 1. Purpose

This document describes the design of the GRIME AI continuous integration
pipeline: how conda packages and container images are versioned, built, and
published. It records the reasoning behind the current structure so that future
changes do not reintroduce defects that have already been fixed.

The operational procedure for cutting a release is documented separately in
[`RELEASING.md`](RELEASING.md).

## 2. Components

| File | Role |
| --- | --- |
| `src/GRIME_AI/version.py` | Holds `SW_VERSION`, `RELEASE`, `SHA`, and `BUILD_DATE`. `SW_VERSION` is the only value edited by hand. |
| `recipe/grime-ai/meta.yaml` | Conda recipe for the application package. |
| `recipe/grime-ai-post/meta.yaml` | Conda recipe for the post-install helper scripts. Independently versioned; rarely changes. |
| `.github/workflows/nightly.yml` | Scheduled and manual development builds. Publishes to the `dev` label. |
| `.github/workflows/release.yml` | Tag-triggered release builds. Publishes to the `main` label. |
| `.github/workflows/update_sha.yml` | Writes `SHA` and `BUILD_DATE` into `version.py` on pushes to `main`. |

## 3. Version model

The package version is decided by the workflow and passed to `conda build`
through the `GRIME_PKG_VERSION` environment variable. The recipe consumes that
value and performs no derivation of its own:

{% raw %}
```jinja
{% set sw_version = version_file.get('SW_VERSION') %}
{% set pkg_version = environ.get('GRIME_PKG_VERSION', sw_version) %}
```
{% endraw %}

The fallback to `SW_VERSION` covers local builds run outside CI, where the
environment variable is absent.

### Version strings by build type

| Build | Version string | Label |
| --- | --- | --- |
| Release | `2.1.7.0` | `main` |
| Nightly | `2.1.7.0.dev202608091530` | `dev` |

Release versions are taken from the git tag, with the leading `v` removed.
Nightly versions append a UTC timestamp accurate to the minute, which guarantees
that no two builds can produce the same version string. This permits a manual
rebuild on a day that has already produced a nightly, without overwriting the
earlier artifact.

> **Immutability.** No published artifact is ever overwritten. A given version
> string always identifies the same bits, which is a prerequisite for
> reproducible citation of a specific build.

### Build string

Conda appends a *build string* to every package filename, derived from
`build: number: 0` in `meta.yaml`. A release therefore appears on anaconda.org
as `grime_ai-2.1.7.0-py_0.conda`. The `py_0` is not part of the version. Users
never specify it; `conda install grime-ai=2.1.7.0` resolves it automatically,
with the solver preferring the highest build number when several exist for one
version. The build number exists to permit republishing a corrected package
under an unchanged version. Because this pipeline never republishes a version,
the build number is permanently `0` and every artifact, release or nightly,
carries `_0`.

## 4. Nightly pipeline

Nightly runs on a 06:00 UTC schedule and on manual dispatch. It consists of two
jobs.

### Job: `nightly`

- Clones the repository.
- Determines whether a build is warranted. A scheduled run whose most recent
  commit predates the current UTC day is skipped, because the source tree is
  unchanged and rebuilding would produce identical bits. A manual dispatch
  always builds.
- Stamps the current date into `BUILD_DATE` in the working copy. The change is
  never committed.
- Computes the timestamped version string and exposes it as a job output.
- Builds and uploads to the `dev` label with `GRIME_PKG_VERSION` set.

### Job: `nightly-container`

- Gated on the `built` output of the preceding job, so a skipped nightly does not
  leave the container job searching for a package that was never published.
- Consumes the conda job's `package_version` output rather than re-deriving it.
- Takes the commit hash from `GITHUB_SHA` on the runner.
- Tags the image with the version, the short commit hash, and `nightly`.

## 5. Release pipeline

The release workflow runs only on tags matching `v*`. Publishing a GitHub
release creates the tag, and nothing else starts the workflow.

### Job: `release`

- Clones the repository at the tag.
- Verifies that the tag matches `SW_VERSION` and fails the build if they
  disagree. This guard is the primary defence against a mislabelled release.
- Stamps `RELEASE` to `stable` in the working copy and emits the tag-derived
  version as a job output. The change is never pushed.
- Builds and uploads to the `main` label with `GRIME_PKG_VERSION` set.

### Job: `release-container`

- Builds the container image against the released conda package.
- Takes the commit hash from `GITHUB_SHA` and the build date from the event
  payload, rather than from `version.py`, whose `SHA` and `BUILD_DATE` fields are
  refreshed only on pushes to `main` and are therefore stale during a tag build.
- Tags the image with the version, the short commit hash, and `stable`.

## 6. Labels

Packages are published to the `grimelab` channel on anaconda.org under two
labels:

- **`main`** — released versions. This is conda's conventional name for the
  default bucket, which is what allows users to install with a plain
  `-c grimelab`. The label name is unrelated to the value of `RELEASE`.
- **`dev`** — nightly builds. Installed explicitly with
  `-c grimelab/label/dev`.

## 7. Design constraints

### Single derivation of the version

The defect that produced version `2.1.6.0.dev20260807` under the `main` label
arose because the recipe and the container job each derived the version string
independently. Any future change must preserve the property that exactly one
component computes the version and every other component consumes that result.

### Release state is never committed

`RELEASE` remains `nightly` in the repository at all times. The release workflow
overwrites it in the runner's working copy, and the runner is discarded when the
job ends. This removes the need to revert `version.py` after a release and
eliminates the window in which `main` could describe itself as a stable build.

### Guards fail loudly

The tag-versus-`SW_VERSION` check exists because a mismatch would otherwise
publish a package whose version does not correspond to any tag in the
repository. A failed build is recoverable; a mislabelled published artifact is
not.

### Skipped builds report success

A nightly that is skipped for want of new commits exits successfully rather than
failing. A red indicator that appears routinely trains maintainers to disregard
it, which defeats the purpose of the notification.

## 8. Known characteristics

- `update_sha.yml` commits to `main` automatically. It preserves the
  `SW_VERSION` and `RELEASE` lines verbatim and appends fresh `SHA` and
  `BUILD_DATE` values. It rewrites `version.py` to exactly four lines, so any
  comment or additional definition placed in that file will be removed on the
  next push to `main`.
- `SHA` and `BUILD_DATE` in `version.py` are refreshed only on pushes to `main`.
  Consumers that require accurate values during a tag build must read them from
  the runner environment instead.
- Nightly artifacts accumulate under the `dev` label, since every build produces
  a distinct version. Periodic pruning is expected.
- `grime-ai-post` is versioned independently and is not rebuilt by either
  workflow. Its published version must remain resolvable under the `main` label
  for release installations to succeed.

## 9. Revision history

| Date | Change |
| --- | --- |
| 2026-08-09 | Initial version. Records the pipeline as restructured following the v2.1.6.0 labelling defect. |
| 2026-08-09 | Added build-string explanation to the version model. |
