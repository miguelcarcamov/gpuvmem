# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where practical. Earlier history is preserved in git and in the
[Astronomy & Computing paper](https://doi.org/10.1016/j.ascom.2017.11.003).

## [Unreleased]

### Added

- `CONTRIBUTING.md` — contribution and issue guidelines.
- `CHANGELOG.md` — this file.
- **Optimization projections (Pyralysis-style):** `include/optimization/projection.hh` with implementations in `src/optimization/projection.cu` — `NoProjection`, `PositivityProjection` (MEM `η` + per-image reference levels), scalar masks (`EqualTo`, `NotEqualTo`, `GreaterThan`, `GreaterThanEqualTo`, `LessThan`, `LessThanEqualTo`), and `CompositeProjection`. Device scalar pass runs after trial updates where wired.
- **`Optimizer::setProjection`** / **`LineSearcher`** hold `std::unique_ptr<Projection>`; **`applyProjectionToImagePlane`** in `linesearch_utils` applies **`Projection::applyToImagePlane`** after `newP` / `evaluateXt` (and in `f1dim` for the no-positivity `evaluateXtNoPositivity` path).

### Changed

- **Positivity / line search wiring:** MEM positivity parameters and XT references are no longer on **`Image`**; **`MFS::setDevice`** configures **`Optimizer::setProjection`** with `NoProjection` or `PositivityProjection` instead of `Image::setLineSearchEta` / XT helpers.
- **Projection API naming:** replaced line-search–specific names with **`positivityEta()`**, **`referenceValue(int)`**, and **`applyToImagePlane(...)`** on **`Projection`**.
- **`docs/extern_inventory.md`** — documents projection ownership and the new API.
- **`CMakeLists.txt`** — `gpuvmem_linesearch_static` includes `src/optimization/*.cu` (e.g. `projection.cu`); optimizer glob uses **`CONFIGURE_DEPENDS`** so new `src/optimizers/*.cu` files are picked up without a manual reconfigure.
- **`LineSearcher`** (`linesearcher.cuh` / `linesearcher.cu`) — declarations-only in the header; default ctor, accessors, **`getProjection()`**, **`setProjection`**, **`releaseProjection`**, and related methods defined in the `.cu`.
- **`Optimizer`** (`optimizer.cuh` / `src/optimizers/optimizer.cu`) — same split for ctors, destructor, getters/setters, and default **`setLineSearcher`** / **`setProjection`**.
- README rewritten for clarity: badges, quick start, tables, consolidated build/install/docs, star-history chart, links to `CONTRIBUTING.md` / `CHANGELOG.md`.
- README positioning: project described as an **RML-style C++/CUDA framework** (data term + modular regularizers) while keeping **MEM** origins and the 2018 paper in context.
- README: extra badges (stars, commits, issues, PRs, contributors, repo size, pre-commit, C++/CUDA, Pyralysis); **Pyralysis** roadmap note (GPU gap today; deprecation when Pyralysis is CPU+GPU+Dask).

### Fixed

- **CTest / CLI:** Restored legacy short/long flags in `options.cu` (`-X/-Y/-V`, `-t` iterations, `-T` threshold, `-Z` regularization, `-G` gpus, `-g` gridding, boolean flags, etc.) so `tests/*/test.sh` matches `getopt_long` again.
- **CTest paths:** `freq78` / `m87` test directories are `FREQ78` / `M87` on case-sensitive filesystems — `CMakeLists.txt` updated.
- **Configure:** Skip `stoi` on GPU list when `gpus` is the sentinel `"NULL"` (`mfs.cu`).
- **Flags:** Avoid `Var(..., '\\0', ...)` with the current `Flags` implementation (it still appends `:` to `optionStr`, which broke parsing); use real short flags for stokes / optimization mode.

---

## Earlier releases

Tagged releases and detailed change notes before this changelog may be incomplete.
See **git log**, **GitHub Releases**, and the paper for the scientific lineage of gpuvmem.
