# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where practical. Earlier history is preserved in git and in the
[Astronomy & Computing paper](https://doi.org/10.1016/j.ascom.2017.11.003).

## [Unreleased]

### Added

- **`ImagingHeader`** (`include/classes/imaging_header.hh`) — in-memory image astrometry with **0-based** `reference_column` / `reference_row` and WCS-related fields; **`imaging_header_from_fits_wire`** (FITS read) and **`fits_wire_header_from_imaging`** (FITS write, **CRPIX = reference + 1**) keep **`FitsHeader`** as wire-only for CFITSIO.
- `CONTRIBUTING.md` — contribution and issue guidelines.
- `CHANGELOG.md` — this file.
- **Line-search projections (Pyralysis-style):** `include/projection/projection.hh` with implementations in `src/projection/projection.cu` — `NoProjection` and `PositivityProjection` (MEM `η`, per-image reference levels, minimal floors). **`makeLineSearchProjectionFromCli`** (`include/projection/projection_from_cli.hh`, `src/mfs/projection_from_cli.cc`) builds the active projection from **`GpuvmemCliConfig`** only.
- **`Optimizer::setProjection`** / **`LineSearcher`** hold `std::unique_ptr<Projection>`; **`applyProjectionToImagePlane`** in `linesearch_utils` applies **`Projection::applyToImagePlane`** after linear `newP` / `evaluateXt` host wrappers (with **`ProjectionApplyContext`** for iterate vs 1D line sample).

### Changed

- **Default objective (`src/main.cu`):** The executable’s wired-in **Fi** terms are **χ²**, **Entropy**, **L1-Norm**, and **TotalSquaredVariation** only — **L2ConstantPrior** is no longer registered there (the class remains in-tree for reuse). **`-Z` / `--regularization_factors`:** fewer than **five** comma-separated weights keep **χ² at λ = 1** and map values to entropy, L1, TSV in order (extra list entries are ignored); **five or more** put **χ²’s weight in the first slot**, then entropy, L1, TSV (same “extras ignored” rule). Removed the old **0.05** fallback that was applied to L2CP when its index fell past the end of the `-Z` list.
- **CLI help (`src/options.cu`):** Clarified **`-Z`** layout; **`-L`** (built-in **Brent** when omitted — optimizers construct with Brent and `main` does not replace an empty `-L`); **`-B`** (no step seeder unless set; **`-B` without `-L`** forces **Brent** for that run so a seeder can attach).
- **`Image`:** Replaced embedded **`FitsHeader`** with **`ImagingHeader`** — **`setImagingHeader`**, **`imagingHeader()`**, **`hasImagingHeader()`**, **`clearImagingHeader()`**; **`setImagingHeader`** syncs legacy **`pixel_scale_*_deg`** from header CDELTs. **`imaging_geometry()`** reads the reference pixel from the header when present.
- **FITS writers:** **`WriteFitsImageOptions`** / **`WriteFitsComplexImageOptions`** use **`const ImagingHeader* inline_primary_header`**; **`fits_io.cu`** builds a temporary **`FitsHeader`** for **`write_primary_wcs_from_header`**.
- **Io / IoFITS / MFS:** **`readHeader`**, **`setModelFitsGeometry`**, **`inlineHeaderForWrites`**, and **`MFS::resolved_model_header_`** use **`ImagingHeader`**; synthetic grid (**no `-m`**) fills **`ImagingHeader`** directly (including **`width_columns` / `height_rows`**).
- **Geometry / visibility API:** **`ImagingGeometry`** exposes **`reference_column` / `reference_row`** (0-based); **`phase_rotate`** / **`computeFrequenciesAndPhaseCenter`** take those names; bilinear DC-at-center path uses the same naming for the grid origin.
- **Layout:** `include/optimization/` and `src/optimization/` renamed to **`projection/`** (line-search feasibility / positivity projections only; avoids conflating with the full optimization stack). Legacy shared regularizer CUDA/host layer: **`include/regularizers/`** and **`src/regularizers/`** renamed to **`regularizer_kernels/`** to distinguish from **`objective_function/terms/regularizers/`** (Fi term classes). Includes and CMake include paths updated accordingly.
- **χ² (chunked MS) / measurement operator:** Forward model and adjoint run **per baseline**: for each channel we recompute **`device_V`** with **`computeImageToVisibilityGridBaseline`**, applying **√(PB_ant1 · PB_ant2)** on the sky (`apply_baseline_beam2I`), then gather only that baseline’s visibilities per correlation; **`DChi2Baseline`** matches that PB in the gradient imaging step. **`do_degridding`** still uses the single-antenna **`computeImageToVisibilityGrid`** until updated the same way.
- **Projection API naming:** replaced line-search–specific names with **`positivityEta()`**, **`referenceValue(int)`**, and **`applyToImagePlane(...)`** on **`Projection`**.
- **`docs/extern_inventory.md`** — documents projection ownership and the new API.
- **`CMakeLists.txt`** — `gpuvmem_linesearch_static` includes `src/projection/*.cu` (e.g. `projection.cu`); optimizer glob uses **`CONFIGURE_DEPENDS`** so new `src/optimizers/*.cu` files are picked up without a manual reconfigure.
- **`LineSearcher`** (`linesearcher.cuh` / `linesearcher.cu`) — declarations-only in the header; default ctor, accessors, **`getProjection()`**, **`setProjection`**, **`releaseProjection`**, and related methods defined in the `.cu`.
- **`Optimizer`** (`optimizer.cuh` / `src/optimizers/optimizer.cu`) — same split for ctors, destructor, getters/setters, and default **`setLineSearcher`** / **`setProjection`**.
- README rewritten for clarity: badges, quick start, tables, consolidated build/install/docs, star-history chart, links to `CONTRIBUTING.md` / `CHANGELOG.md`.
- README positioning: project described as an **RML-style C++/CUDA framework** (data term + modular regularizers) while keeping **MEM** origins and the 2018 paper in context.
- README: extra badges (stars, commits, issues, PRs, contributors, repo size, pre-commit, C++/CUDA, Pyralysis); **Pyralysis** roadmap note (GPU gap today; deprecation when Pyralysis is CPU+GPU+Dask).
- **README (usage):** Cheat sheet and defaults for **`-Z`** (weight layout; default **Fi** in `main.cu`), **`-L`** / **`-B`** (Brent; no seeder), behavior when **`-Z`** is omitted (χ² + TSV only), **`--help`** empty **`[default: ]`**, and short-option parsing for **`-O`** / **`-m`**.

### Fixed

- **Gridded MS metadata:** After the "Checking frames" loop fills **`FieldMetadata`** on the native **`ms`**, copy it onto **`gridded_ms`** so **`host_field_for_gpu_field`** (used by χ²) does not keep stale **zero** **`ref_*` / `phs_*`** from grid construction.
- **Line search / LBFGS:** dropped fused **`particularNewP` / `particularEvaluateXt`**; all planes use linear kernels plus **`Projection`** (same as **`updatePoint`**). **`lineSearch1dEval`** uses **`imageMap::evaluateXt`** only. **LBFGS** **`initializeOptimizationState`** now sets **`ObjectiveFunction`** CUDA grids after **`configure`**, matching **ConjugateGradient** (avoids invalid launch configs in **`updatePoint`**).
- **CTest / CLI:** Restored legacy short/long flags in `options.cu` (`-X/-Y/-V`, `-t` iterations, `-T` threshold, `-Z` regularization, `-G` gpus, `-g` gridding, boolean flags, etc.) so `tests/*/test.sh` matches `getopt_long` again.
- **CTest paths:** `freq78` / `m87` test directories are `FREQ78` / `M87` on case-sensitive filesystems — `CMakeLists.txt` updated.
- **Configure:** Skip `stoi` on GPU list when `gpus` is the sentinel `"NULL"` (`mfs.cu`).
- **Flags / getopt (`include/classes/flags.cuh`):** **`Var`** only appends **`:`** to the short-option string when a **short letter** is present. Long-only options used to append orphan colons, corrupting the optstring (e.g. **`m::::`**, **`O::`**) so **GNU getopt** mis-parsed flags — notably **`-O path`** (parsed like an optional argument glued to **`-O`**) and **`-m`** could be ignored.

### Removed

- Temporary **`[gpuvmem geometry]`** **`std::cerr`** traces from **`chi2_host.cu`** and **`measurement_operator_host.cu`**.

---

## Earlier releases

Tagged releases and detailed change notes before this changelog may be incomplete.
See **git log**, **GitHub Releases**, and the paper for the scientific lineage of gpuvmem.
