# GRIME AI — Release Process

This procedure creates a stable release of the `grime-ai` conda package and the
accompanying container image. Every step can be completed on the GitHub website;
no local clone is required.

> **Before you begin.** Everything you want included in the release must already
> be merged to `main`.

---

## Step 1 — Set the version

1. Open `src/GRIME_AI/version.py` on github.com.
2. Click the pencil icon to edit.
3. Change `SW_VERSION` only. Leave `RELEASE` alone.

   ```python
   SW_VERSION = "2.1.7.0"
   ```

4. Click **Commit changes**.
5. Choose "Create a new branch and start a pull request." Name the branch
   `release/2.1.7.0`.
6. Open the pull request, obtain approval, and merge to `main`.

> **Do not edit `RELEASE`.** It stays `"nightly"` in the repository permanently.
> CI overwrites it during the build.

## Step 2 — Tag the release

1. Go to the **Releases** page and click **Draft a new release**.
2. In the tag dropdown, type `v2.1.7.0` and select
   "Create new tag: v2.1.7.0 on publish."
3. Confirm that **Target** is set to `main`.
4. Add a title and notes, or click **Generate release notes**.
5. Click **Publish release**.

Publishing creates the tag, and the tag is what starts the Official Release
workflow. Nothing else starts it.

> **Tag format.** The tag must match `SW_VERSION` exactly, with a leading `v`.
> A mismatch fails the build at the verification step.

## Step 3 — Watch the build

Open the **Actions** tab and follow the Official Release workflow. It performs
the following:

```
Checkout
Verify tag matches SW_VERSION
Stamp RELEASE = "stable"
Build and upload to main label
Build and push container
```

If the verification step fails, the tag and `SW_VERSION` disagree. Correct
`version.py` — not the guard — then redo the tag.

## Step 4 — Verify the package

On anaconda.org, under the `grimelab` channel, the `main` label should read:

```
2.1.7.0
```

There must be no `.dev` suffix. If one appears, the stamp step did not run.
Delete any leftover `.dev` package carrying the same version number.

## Step 5 — Bump for the next cycle

Repeat Step 1 with the next version number, for example `2.1.8.0`. Nightly
builds will then produce `2.1.8.0.devYYYYMMDD` and will no longer collide with
the released number.

---

## Troubleshooting — redoing a tag

Do this only if nobody has downloaded the package yet.

1. On the Releases page, open the release and click **Delete**.
2. Go to the tags list and delete the tag as well. Deleting a release does not
   delete its tag.
3. Repeat Step 2.

Once anyone has downloaded the package, do not reuse the number. Bump to the
next version instead.

---

## How it works

`version.py` holds two values:

- **`SW_VERSION`** — the version number. You edit this. The About box reads it.
- **`RELEASE`** — the build channel. Always `"nightly"` in the repository.

Nightly builds read `RELEASE` as committed and receive a `.dev` suffix from
`meta.yaml`. Release builds overwrite `RELEASE` to `"stable"` on the runner's own
copy of the files, so `meta.yaml` sees a non-nightly value and omits the suffix.
The runner is discarded afterward, which is why `main` is never modified.

The Anaconda label is always `main`. That is conda's name for the default bucket
and is unrelated to the `RELEASE` value. It is what allows users to install with
a plain `-c grimelab`.

---

*John E. Stranzl Jr., PhD — August 8, 2026*
