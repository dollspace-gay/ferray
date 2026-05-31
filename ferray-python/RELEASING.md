# Releasing ferray-python to PyPI

The CI workflow at `.github/workflows/ferray-python-wheels.yml` builds
wheels for **20 platform × Python-version cells** plus an sdist on
every push to `main` and every PR that touches the relevant code paths.

Publishing to PyPI happens automatically when a release tag is pushed.

## Coverage

| Platform | Architectures | Python versions |
|----------|---------------|-----------------|
| Linux    | x86_64, aarch64 | 3.10, 3.11, 3.12, 3.13 |
| macOS    | x86_64 (Intel), aarch64 (Apple Silicon) | 3.10, 3.11, 3.12, 3.13 |
| Windows  | x64 | 3.10, 3.11, 3.12, 3.13 |
| sdist    | (universal source) | — |

Cross-compiled wheels (linux aarch64, macos x86_64 on arm runners) skip
the post-build smoke test; native wheels run a small import + numerical
check before being uploaded as artifacts.

## One-time setup: PyPI trusted publishing

Before the first tag push, configure trusted publishing on PyPI so the
workflow can upload without an API token:

1. Sign in to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Project name: `ferray`
4. Owner: `dollspace-gay`
5. Repository name: `ferray`
6. Workflow filename: `ferray-python-wheels.yml`
7. Environment name: `pypi`

The `environment: pypi` declaration in the workflow plus the
`id-token: write` permission lets PyPI verify the workflow's identity
via OIDC. No API token ever lives in the repo or its secrets.

For the first release, PyPI registers the project name on first upload;
subsequent uploads only need to match the trusted-publisher record.

## Cutting a release

ferray is a multi-crate workspace, so we use a prefixed tag scheme to
avoid clashing with future Rust-crate-only releases. For ferray-python:

1. **Bump the version** in the workspace root `Cargo.toml` (the
   `[workspace.package] version` field — `ferray-python` inherits it).
2. **Update the changelog** in `CHANGELOG.md`.
3. **Open a release PR**, get review, merge to `main`.
4. **Tag** the merge commit:
   ```bash
   git tag ferray-python-v0.4.0
   git push origin ferray-python-v0.4.0
   ```
5. **Watch the workflow run** at the Actions tab. The `publish` job
   only runs on tag pushes; it downloads every wheel/sdist artifact
   from the matrix and runs `maturin upload` against PyPI.
6. **Verify on PyPI** that the new version appears at
   https://pypi.org/project/ferray/ with all 20 wheels + sdist
   visible under "Download files".

## Re-running a failed release

If the publish job fails (network blip, transient PyPI error), don't
re-tag. Re-run the failed job from the Actions UI. The
`--skip-existing` flag on `maturin upload` handles re-runs cleanly:
already-uploaded wheels are skipped, only the missing ones go up.

## Manually triggering a wheel build

Use the `workflow_dispatch` trigger from the Actions UI to build the
full matrix on demand without tagging — useful for verifying a branch
before merge or building a release-candidate wheel for internal
testing.

## Wheel naming

maturin produces wheels named like:

```
ferray-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
ferray-0.4.0-cp310-cp310-macosx_11_0_arm64.whl
ferray-0.4.0-cp310-cp310-win_amd64.whl
ferray-0.4.0.tar.gz                  # sdist
```

Per-version (not abi3) wheels keep the per-call overhead minimal, which
matters for a numerical library.

## Local pre-flight

Before tagging, sanity-check that maturin can build a wheel locally:

```bash
cd ferray-python
source .venv/bin/activate
maturin build --release --out dist
ls dist  # should contain a single matching wheel
```

A successful local build doesn't guarantee the matrix passes (the
manylinux container has stricter glibc constraints), but a local
failure is a fast signal something is wrong before burning CI minutes.
