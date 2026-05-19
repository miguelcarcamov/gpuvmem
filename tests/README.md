# gpuvmem tests

## Layout (Google Test pyramid)

| Layer | Directory | Label | Convention |
|-------|-----------|-------|------------|
| Unit | `tests/unit/<module>/test_<Class>.cc` | `unit` | One class or small module per executable |
| Integration | `tests/integration/<module>/test_<A>_<B>.cc` | `integration` | Two or more classes wired together |
| E2E | `tests/<dataset>/test.sh` | `e2e` | Full binary on real MS (bash, not GTest) |

CMake helpers live in `cmake/GpuvmemGTest.cmake` (`gpuvmem_add_unit_test`, `gpuvmem_add_integration_test`, CUDA variants).

### Geometry presets (pixel scale, Fourier cells, ν₀)

Shared types in `tests/common/imaging_geometry.hh`:

- `ImagingGeometryParam` — `n`, `m`, `images`, sky pixel scales (deg), `nu_0_hz`, and derived `uv_cell_u` / `uv_cell_v` (same formulas as `Image` in `include/classes/image.cuh`).
- `dataset_geometry_presets()` — smoke, M87-like, co65 spectral, FREQ78, etc.
- `apply_legacy_imaging_globals(geom)` — sets legacy `deltau`, `deltav`, `DELTAX`, `DELTAY`, `nu_0` for CUDA tests.

Integration matrix: `gpuvmem_integration_objective_geometry` runs every light regularizer × every preset.

### Current GTest executables

**Unit (`ctest -L unit`)**

- `gpuvmem_unit_cli_observer` — `ConsoleRunObserver`, `NullRunObserver`
- `gpuvmem_unit_cli_parse` — `parse_gpuvmem_cli`
- `gpuvmem_unit_math_utils` — `median`, `iDivUp`, power-of-two helpers
- `gpuvmem_unit_ms_metadata` — `MeasurementSetMetadata`, `SpectralWindow`, …
- `gpuvmem_unit_flags` — `Flags` parser
- `gpuvmem_unit_cuda_grid` — `CudaGrid` launch geometry

**Integration (`ctest -L integration`)**

- `gpuvmem_integration_cli` — CLI parse + observer factory
- `gpuvmem_integration_ms` — `MeasurementSet` + `Field` + shared metadata
- `gpuvmem_integration_objective_regularizers` — each regularizer `calcFi` (default geometry)
- `gpuvmem_integration_objective_geometry` — regularizer × dataset geometry (N,M, pixel scale, ν₀, deltau/deltav)
- `gpuvmem_integration_objective_reg_combos` — production regularizer subset power set (Entropy/L1/TSV; Chi2 in E2E)
- `gpuvmem_integration_optimizer_stack` — optimizer × line search × seeder wiring matrix (336 cases)
- `gpuvmem_integration_optimizer_objective` — each optimizer runs 1 iteration on L2 prior + Fixed line search (**gpu**)

## Build and run

```bash
cmake -B build -DGPUVMEM_BUILD_TESTS=ON
cmake --build build
cd build && ctest -L unit --output-on-failure
cd build && ctest -L integration --output-on-failure
```

### CI vs local GPU

GitHub-hosted runners usually have the **CUDA toolkit** (build works) but **no GPU device**.

- Tests that allocate/run CUDA kernels are labeled **`gpu`** in CTest (`REQUIRES_GPU` in CMake).
- **CI (no GPU):** `ctest -L unit -LE gpu` and `ctest -L integration -LE gpu`
- **Local / GPU runner:** `ctest -L gpu` (or full `ctest -L unit` / `ctest -L integration`)

Inside tests, `require_cuda_device()` skips individual cases when `cudaSetDevice(0)` fails (e.g. `CUDA_VISIBLE_DEVICES=""`).

Disable GTest targets: `-DGPUVMEM_BUILD_TESTS=OFF`.

## End-to-end tests (per dataset)

Each dataset directory has a `test.sh` that runs **several scenarios** (verbose, quiet, weighting, metrics file, etc.) via `tests/common/e2e_lib.sh`.

```bash
cmake --build build
# Quick smoke (25 iterations per scenario by default):
ctest -L e2e --output-on-failure

# Longer run (closer to regression benchmarks):
GPUVMEM_E2E_MAX_ITER=500000000 ctest -L e2e -R e2e_M87
```

Requirements: CUDA GPU, Casacore datasets (often Git LFS), same as before.

### Adding a scenario

In `tests/<dataset>/test.sh`:

```bash
ARGS=("${COMMON[@]}" --your --flags)
e2e_append_iter_flag ARGS   # adds -t unless you already set -t
e2e_gpuvmem_run "scenario_name" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"
```

### Adding a unit or integration test

1. Add `tests/unit/<module>/test_foo.cc` (or `tests/integration/...`).
2. Register the executable in `tests/unit/CMakeLists.txt` or `tests/integration/CMakeLists.txt`.
3. Prefer minimal link dependencies (do not link `gpuvmem_mfs_static` unless necessary).
