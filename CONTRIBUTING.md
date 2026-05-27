# CONTRIBUTING.md

# Contributing to core-lens

Thank you for contributing to `core-lens`.

This document defines the development workflow, branching strategy, commit conventions, testing expectations, release process, and hotfix procedures for the project.

---

# 1. Development Philosophy

`core-lens` is a geospatial library focused on reproducibility, correctness, and long-term API stability.

The repository is designed to ensure:

* reproducible environments
* deterministic CI
* safe releases
* strict plugin contracts
* zero production data in the repository

All contributors are expected to follow the workflows documented here.

---

# 2. Repository Rules

## 2.1 No Real Data

Real parquet files must never be committed to the repository.

Allowed:

* synthetic fixture parquet files under `tests/fixtures/`

Forbidden:

* production datasets
* client datasets
* experimental local datasets
* parquet files outside `tests/fixtures/`

A pre-commit hook enforces this automatically.

---

# 3. Repository Structure

```text
core-lens/
├── src/
├── tests/
├── docs/
├── scripts/
├── .github/
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
└── uv.lock
```

Important directories:

| Directory                | Purpose                                |
| ------------------------ | -------------------------------------- |
| `src/lib/`               | Library source code                    |
| `tests/unit/`            | Pure logic tests                       |
| `tests/plugin_contract/` | Plugin API validation                  |
| `tests/stats/`           | Statistical method tests               |
| `tests/fixtures/`        | Synthetic parquet fixtures only        |
| `docs/`                  | Sphinx documentation                   |
| `scripts/`               | Local utilities                        |

---

# 4. Development Environment

## 4.1 Tooling

The project uses:

* `uv` for dependency management and environments
* `ruff` for linting and formatting
* `mypy` for type checking
* `pytest` for testing
* `pre-commit` for local quality gates

Do not use `pip install` or unmanaged virtual environments.

---

## 4.2 Initial Setup

Clone the repository:

```bash
git clone <repo-url>
cd core-lens
```

Install dependencies:

```bash
uv sync --all-extras
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Install the git commit template:

```bash
git config commit.template .gitmessage
```

---

# 5. Branching Strategy

## 5.1 Main Branches

| Branch | Purpose                            |
| ------ | ---------------------------------- |
| `main` | Always releasable                  |
| `dev`  | Integration branch for normal work |

Both branches are protected.

Direct pushes are forbidden.

---

## 5.2 Feature Branches

All normal work branches from `dev`.

Examples:

```text
feature/stats-similarity
feature/plugin-validation
fix/version-clash-fortnightly
```

Workflow:

```text
dev
 └── feature/my-feature
       └── PR → dev
```

Rules:

* keep branches short-lived
* rebase frequently
* one logical change per PR

---

## 5.3 Hotfix Branches

Hotfixes branch from `main`, never `dev`.

Example:

```text
hotfix/critical-bug
```

Workflow:

```text
main
 └── hotfix/critical-bug
       ├── PR → main
       └── PR → dev
```

Hotfixes require:

* passing CI
* review approval
* changelog entry
* patch release

---

# 6. Commit Conventions

The project follows Conventional Commits.

Format:

```text
<type>[optional scope]: <description>
```

Examples:

```text
feat: add stats.similarity()
fix: resolve VersionClashError on fortnightly
docs: add plugin guide
test: add aggregate validation fixtures
chore: update ruff
```

---

## 6.1 Breaking Changes

Breaking changes must use:

```text
feat!: rename BaseResult.df() to BaseResult.frame()
```

Breaking changes affect semantic versioning and release generation.

---

# 7. Running Quality Checks

## 7.1 Linting

```bash
uv run ruff check .
```

---

## 7.2 Formatting

```bash
uv run ruff format .
```

To verify formatting only:

```bash
uv run ruff format --check .
```

---

## 7.3 Type Checking

```bash
uv run mypy src/lib/
```

---

## 7.4 Running Tests

Run all tests:

```bash
uv run pytest
```

Run unit tests only:

```bash
uv run pytest tests/unit/
```

Run plugin contract tests:

```bash
uv run pytest tests/plugin_contract/
```

Run stats tests:

```bash
uv run pytest tests/stats/
```

---

# 8. Dependency Management

## 8.1 Lockfile Rules

`uv.lock` is committed and required.

If dependencies change:

```bash
uv lock
```

CI will fail if:

* `pyproject.toml` changes
* but `uv.lock` is not updated

Verify locally:

```bash
uv lock --check
```

---

## 8.2 Dependency Groups

Examples:

```bash
uv sync --extra core
uv sync --extra spatial
uv sync --all-extras
```

---

# 9. Pre-commit Hooks

Hooks run automatically before commits.

Included checks:

* `ruff check`
* `ruff format --check`
* YAML validation
* TOML validation
* trailing whitespace
* EOF fixer
* parquet file guard

If hooks fail, fix the issue before committing.

---

# 10. Pull Requests

## 10.1 PR Requirements

All PRs must:

* pass CI
* include tests where appropriate
* update documentation where appropriate
* update `CHANGELOG.md` for source changes
* use conventional commits

---

## 10.2 CHANGELOG Requirement

If a PR modifies files under:

```text
src/
```

then `CHANGELOG.md` must also be updated.

CI enforces this rule for PRs to `dev`.

Exceptions:

* docs-only changes
* test-only changes
* chore-only changes

---

## 10.3 PR Review Expectations

PRs to `main` require:

* passing CI
* at least one reviewer approval

PRs to `dev` require:

* passing CI

---

# 11. Fixture Data

Fixtures are already generated and should not be manually edited.

---

# 12. CI Pipeline Overview

CI stages:

1. Lockfile verification
2. Linting
3. Formatting
4. Type checking
5. CHANGELOG validation
6. Security audit
7. Test matrix
8. Coverage
9. Build verification

The CI pipeline is authoritative.

A locally passing test run does not override CI failures.

---

# 13. Coverage Policy

Coverage gates:

| Area             | Requirement           |
| ---------------- | --------------------- |
| `src/lib/core/`  | 80% minimum           |
| `src/lib/stats/` | 70% warning threshold |

New functionality should include:

* unit tests

---

# 14. Documentation

Documentation uses:

* Sphinx
* sphinx-autoapi
* MyST
* Shibuya

Build docs locally:

```bash
uv run sphinx-build -b html docs/source docs/build/
```

Documentation should accompany:

* new public APIs
* new plugin interfaces
* behavioral changes
* breaking changes

---

# 15. Release Process

Releases are automated using:

* GitHub Actions
* python-semantic-release
* PyPI trusted publishing via OIDC

Contributors do not publish manually.

---

## 15.1 Normal Release Flow

1. `python-semantic-release` opens a release PR
2. Team reviews PR
3. PR merged into `main`
4. Tag pushed
5. CI re-runs
6. Build validated
7. Manual approval gate
8. Publish to PyPI
9. GitHub Release created

---

## 15.2 Pre-releases

Pre-release tags publish to TestPyPI.

Examples:

```text
v0.3.0-alpha.1
v0.3.0-beta.1
v0.3.0-rc.1
```

---

# 16. Hotfix Procedure

## 16.1 Creating a Hotfix

Branch from `main`:

```bash
git checkout main
git pull
git checkout -b hotfix/my-fix
```

Apply the fix and commit.

---

## 16.2 Triggering the Hotfix Workflow

Use the GitHub Actions workflow:

```text
hotfix-bump.yml
```

Inputs:

* patch version number

The workflow:

* updates `pyproject.toml`
* generates changelog content
* opens a PR to `main`

---

## 16.3 Completing the Hotfix

After merge to `main`:

1. push patch tag
2. verify CI
3. approve release
4. verify PyPI publish
5. open PR from hotfix branch → `dev`

Both PRs are required.

---

# 18. Platform Support

Official CI support:

* Linux only

macOS and Windows are best-effort.

---

# 19. Things Intentionally Not Included

The project intentionally does not include:

* Docker images
* benchmark CI jobs
* production datasets
* commitlint enforcement
* Windows/macOS CI matrices

These may be revisited later.

---

# 20. Contributor Expectations

Contributors are expected to:

* keep PRs focused
* write tests
* maintain type safety
* preserve API consistency
* document breaking changes clearly
* follow the established workflow

When uncertain:

* open a draft PR early
* ask design questions before implementation
* prefer explicitness over cleverness
