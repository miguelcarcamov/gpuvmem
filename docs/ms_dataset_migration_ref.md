# Dataset usage and migration reference

This document lists where the legacy dataset (`MSDataset`, `Field`, `HVis`, `DVis`, `numVisibilitiesPerFreqPerStoke`, etc.) is used, so we can later feed the new MS model (MeasurementSet, ChunkedVisibilityGPU) into the same pipeline.

---

## Migration status (summary)

### Done

| Item | Notes |
|------|--------|
| **mfs.cu compiles** | Uses `std::vector<MSWithGPU> datasets`; read via MSReader; weighting, gridding, beam/noise use new MS. |
| **Legacy view removed** | `MSWithGPU` no longer has `data`, `fields`, `antennas`. chi2/dchi2/errors/IO use chunked path (ChunkedVisibilityGPU + metadata). `build_legacy_from_ms()` and `src/ms/legacy_adapter.cu` removed. |
| **functions.cu** | Global `extern std::vector<MSWithGPU>* g_datasets`; chi2/dchi2/calculateErrors use chunked path only (chi2_chunked, dchi2_chunked, calculateErrors_chunked). Legacy `degridding(fields, data, ..., dataset)` removed; only `do_degridding(ms, gpu, ...)` used. `precomputeNeff()` is no-op when `use_chi2_chunked_path()`. |
| **Polarization metadata (Section 14)** | Reader reads CORR_TYPE from POLARIZATION; `Polarization` class and table in `MeasurementSetMetadata`; per-DD npol from polarization. |
| **Image geometry / Chi2** | Image has pixel_scale, uv_cell, nx/ny; Chi2 uses `configure(Image*)`; legacy `configure(int)` removed. |
| **Beam/noise** | `MSWithGPU::computeNoiseAndBeamContribution` + `beamNoiseFromSums`; legacy `calculateNoiseAndBeam(MSDataset)` removed. |
| **Gridder** | `grid(std::vector<MSWithGPU>&)` and `degrid(MSWithGPU&, I, ip)` use native path (`do_gridding` / `do_degridding` on ms + gpu). |

### TODO / remaining

| Item | Notes |
|------|--------|
| **Legacy type cleanup** | Remove MSDataset, MSData, Field, HVis, DVis, MSAntenna from MSFITSIO.cuh and legacy IO paths when no longer referenced. |
| **do_gridding / do_degridding** | **Done.** Gridding: CPU preprocessing on host MeasurementSet; upload to ChunkedVisibilityGPU; gridded_ms for synthesis. Degridding: GPU via `do_degridding(ms, gpu, ...)`. |
| **Stokes (Section 14)** | Polarization helper, MSReadOptions, and conversion **done** (`correlations_to_stokes` / `stokes_to_correlations`; writer requires correlation mode to write). |
| **Legacy type cleanup** | After full native path: remove MSDataset, MSData, Field, HVis, DVis, MSAntenna from MSFITSIO.cuh and legacy IO paths. |
| **Image geometry globals** | Migrate gridder/chi2/degridding to use `image->uv_cell_u()`, `pixel_scale_ra_deg()` etc. and remove DELTAX, DELTAY, deltau, deltav globals. |

### Next steps (recommended order)

1. **do_gridding / do_degridding (native path)** — **Done.** Native do_gridding(ms, gpu) on CPU (iterate host MS, grid, upload to ChunkedVisibilityGPU, return gridded_ms). MSWithGPU.gridded_ms + build_legacy_from_ms(dw, true) after Gridder::grid for synthesis; writeResiduals: upload(ms), Gridder::degrid, download, build_legacy_from_ms(dw, false). Legacy degridding path no longer used when gridding is enabled.
2. **Chi2 / dchi2 on new structure** — **Done.** chi2_chunked and dchi2_chunked iterate over ChunkedVisibilityGPU (field → baseline → chunk) and metadata (dd_id → spw, pol); gather (chan, pol) visibilities from chunks, degrid/bilinear, residual, chi2Vector/DChi2. chi2() and dchi2() branch to chunked path when use_chi2_chunked_path() (all datasets have gpu.num_fields() > 0). mfs uploads to gpu when gridding is disabled so chunked path is used.
3. **Errors (calculateErrorImage) on new structure** — **Done.** — `calculateErrors_chunked` iterates over ChunkedVisibilityGPU (field → baseline → chunk) and metadata (dd_id → spw, pol); for each (dd_id, chan, pol) with RR/LL/XX/YY, gathers weights from chunks, deviceReduce for sum_weights, then runs I_nu_0_Noise, alpha_Noise, covariance_Noise and noise_reduction. `calculateErrors()` branches to chunked path when `use_chi2_chunked_path()` (all datasets have gpu.num_fields() > 0). **Intended:** errors should be calculated **per n_images** and **per stokes** (see §4.1).
4. **IO fully on new structure** — **Done.** When all datasets have `gpu.num_fields() > 0`, `MFS::writeResiduals()` uses the native path: for each d `download()` (sync Vm/Vr from ChunkedVisibilityGPU to ms), then copy(name, oname), then if `ms.storage_mode() == Stokes` call `stokes_to_correlations(ms)`, then `MSWriter::write(oname, ms, opts)` with MODEL_DATA and RESIDUAL_DATA. Legacy path (modelToHost + writeResidualsAndModel) used when gpu not populated. `CasacoreMSWriter::write_residual_column` implemented to write Vr from ms to the residual column.
5. **Remove legacy view** — **Done.** Dropped `data`, `fields`, `antennas` from `MSWithGPU`; removed `build_legacy_from_ms()` and `src/ms/legacy_adapter.cu`; removed legacy `degridding(fields, data, ..., dataset)`; `precomputeNeff()` no-op when chunked path. Remaining: delete legacy types from MSFITSIO.cuh when no longer referenced.
6. **Stokes integration (Section 14.3.5)** — Use “Stokes mode” in chi2/gridding/degridding when the user has converted to Stokes (e.g. image Stokes I only from Stokes I visibilities).
7. **Image geometry cleanup** — Use `image->uv_cell_u()`, `image->pixel_scale_ra_deg()` (etc.) in gridder/chi2/degridding and remove globals `DELTAX`, `DELTAY`, `deltau`, `deltav`.
8. **ImageProcessor / Chi2 configure** — Pass `Image*` into Chi2::configure and call `ip->configure(image)`; remove `VirtualImageProcessor::configure(int)` and `ImageProcessor::configure(int)` and extern M, N usage in ImageProcessor.

---

## Legacy detection – calls and redundant declarations

Use this section to find legacy call sites and declarations that can be removed or migrated once the new MS model and Image geometry are fully in use.

### 1. Legacy dataset access (MSWithGPU vs MSDataset)

**Status:** mfs.cu and functions.cu use the **native path** only. `MSWithGPU` has no legacy `data`/`fields`/`antennas`. chi2, dchi2, errors use chunked path (`chi2_chunked`, `dchi2_chunked`, `calculateErrors_chunked`). Degridding uses `do_degridding(ms, gpu, ...)` only. IO uses MSWriter native path.

| Pattern | Status |
|--------|--------|
| `datasets[d].ms.*`, `.ms.field(f).*`, `.ms.metadata().antenna()` | **Used**; legacy view removed. |
| `Gridder::degrid(datasets[d], I, ip)` | **Used**; calls `do_degridding(dataset.ms, &dataset.gpu, ...)`. |

**functions.cu:**

| Pattern | Status |
|--------|--------|
| `extern std::vector<MSWithGPU>* g_datasets` | **Done**; chi2/dchi2/errors use chunked path and `(*g_datasets)[d].ms`, `.gpu`. |
| `degridding(..., MSWithGPU& dataset)` | **Removed**; only `do_degridding(ms, gpu, ...)` exists. |
| `calculateNoiseAndBeam(std::vector<MSDataset>&, ...)` | **Removed**; mfs uses `MSWithGPU::computeNoiseAndBeamContribution` + `beamNoiseFromSums`. |

### 2. Legacy declarations (headers) – remove after migration

| File | Declaration | Remove when |
|------|-------------|-------------|
| **MSFITSIO.cuh** | `MSData`, `MSDataset`, `Field` (typedef struct field), `HVis`, `DVis`, `MSAntenna` | All call sites use `MeasurementSet` / `MSWithGPU` / `gpuvmem::ms::Field` |
| **MSFITSIO.cuh** | `modelToHost(std::vector<Field>&, MSData, ...)`, read/write taking `std::vector<Field>&`, `MSData` | IoMS and mfs use MSWriter / new model |
| **include/functions.cuh** | `do_gridding(std::vector<Field>&, MSData*, ...)` (legacy gridding); `degridding` and `calculateNoiseAndBeam` removed. New path: `do_gridding(ms, gpu, ...)`, `do_degridding(ms, gpu, ...)`. | New MS path only |
| **include/classes/io.cuh**, **ioms.cuh** | Virtuals taking `std::vector<Field>&`, `MSData data` (read, writeResidualsAndModel, writeModelVisibilities, etc.) | New IO uses MSReader/MSWriter and `MeasurementSet` |

### 3. ImageProcessor: `configure(int)` vs `configure(Image*)`

| Call site | Current | Action |
|-----------|---------|--------|
| **mfs.cu** (degridding block) | `ip->configure(image)` | Already using Image*; no change. |
| **chi2/chi2.cu** | `this->ip->configure(image_count)` | Only remaining call to `configure(int)`. Pass `Image*` into Chi2::configure (e.g. from synthesizer/optimizer) and call `ip->configure(image)`; then **remove** `VirtualImageProcessor::configure(int)` and `ImageProcessor::configure(int)`, and the `extern M, N` usage in ImageProcessor. |

### 4. Image geometry: getM/getN vs nx/ny, globals vs Image

| Item | Status | Action |
|------|--------|--------|
| **getM() / getN()** | Used in optimizers, line searchers, objective_function, chi2, functions.cu (~30+ call sites). | Keep for now. Optionally migrate to `image->nx()`, `image->ny()` for clarity; then deprecate getM/getN if desired. |
| **nx() / ny()** | Defined on Image; not yet used in .cu. | No redundancy; use when migrating call sites to “new” names. |
| **pixel_scale_ra_deg() / uv_cell_u()** etc. | Set on Image in mfs; not yet read elsewhere. | After migration, gridder/chi2/degridding can use `image->uv_cell_u()`, `image->pixel_scale_ra_deg()` and **remove** globals `DELTAX`, `DELTAY`, `deltau`, `deltav`. |
| **ObjectiveFunction::getM() / getN()** | Used by seeders, line searchers, optimizers. | Forwards to internal M, N (set by configure(N, M, I)). Not redundant; keep until/unless ObjectiveFunction takes Image* and delegates to it. |

### 5. Quick grep patterns to find remaining legacy

- Legacy dataset: `\.data\.`, `\.fields\[`, `\.antennas`, `numVisibilitiesPerFreqPerStoke`, `device_visibilities`, `backup_`, `MSDataset`, `MSData`, `HVis`, `DVis`
- Legacy IO/gridding: `std::vector<Field>&`, `MSData`, `modelToHost`, `writeResidualsAndModel`
- ImageProcessor legacy: `configure(image_count)` or `configure(\s*[0-9])`

---

## Weights in the new MS model (imaging_weights, weight_spectrum)

- **Weight priority from MS:** 1) **WEIGHT_SPECTRUM** if present (per chan, pol), 2) **WEIGHT** broadcast to all channels (per pol).
- **Per visibility:** `VisSample.weight` = value from MS (weight_spectrum or WEIGHT); `VisSample.imaging_weight` = value used for imaging (chi2, gridding). Initially imaging_weight = weight.
- **Weighting schemes** (Briggs, natural, etc.) should modify **imaging_weight** only; **restoreWeights** resets imaging_weight = weight (e.g. `TimeSample::restore_imaging_weights()`).
- **GPU upload:** Device weight buffers are filled from **imaging_weight** (so synthesis uses imaging weights). When writing back, **update_weights** writes imaging_weights (averaged per pol for the WEIGHT column).

---

## 1. **mfs.cu** – main synthesis driver

**Status:** mfs.cu uses `std::vector<MSWithGPU> datasets`. Read, weighting, gridding, setDevice, chi2, degridding, errors, and write use the **native path** only (ms + gpu). Legacy view removed. **Build is green.**

| Location | What uses dataset | Status |
|----------|-------------------|--------|
| **Global** | `std::vector<MSWithGPU> datasets`; `g_datasets = &datasets` | Done |
| **Read** | `reader->read(datasets[d].name, datasets[d].ms, read_opts)` | Done (MSReader) |
| **Weighting** | `scheme->apply(datasets)` | Done |
| **Gridding** | `gridder.grid(datasets)` (native) | Done |
| **Noise/beam** | `dw.computeNoiseAndBeamContribution(...)`; `beamNoiseFromSums(...)` | Done |
| **Device alloc** | Per field: `datasets[d].atten_image`; `datasets[d].gpu` (ChunkedVisibilityGPU) | Done (native) |
| **Field coords** | From `ms.field(f).metadata()`, `ms.field(f).reference_dir()` / `phase_dir()` | Done (native) |
| **Apply beam** | `apply_beam2I(..., datasets[d].atten_image[f], ms.metadata().antenna(0).*)` | Done (native) |
| **Chi2 / optimizer** | chi2/dchi2 use chunked path (`chi2_chunked`, `dchi2_chunked`) | Done (native) |
| **Degridding** | `Gridder::degrid(datasets[d], I, ip)` → `do_degridding(ms, &gpu, ...)` | Done (native) |
| **Write** | Native path: `download()`, copy, `stokes_to_correlations` if Stokes, `MSWriter::write` | Done (native) |
| **Cleanup** | `datasets[d].gpu.zero_model_and_residual()` / `clear()`; `atten_image` freed | Done (native) |
| **Restore weights** | `scheme->restoreWeights(*getDatasets())` | Done |
| **Error** | `error->calculateErrorImage(image, *getDatasets())` → `calculateErrors_chunked` | Done (native) |

**Summary mfs.cu:** Pipeline uses **native path** only; legacy view and `build_legacy_from_ms()` removed.

---

## 2. **chi2 and dchi2** – functions.cu (and chi2 Fi)

| Function / Fi | What uses dataset | Legacy access |
|---------------|-------------------|----------------|
| **chi2** (host) | Loops `d`, `f`, `i`, `s`; uses `datasets[d].data.nfields`, `total_frequencies`, `nstokes`, `fields[f].nu[i]`, `ref_xobs_pix`, `phs_xobs_pix`, `antennas[0].*`, `fields[f].device_visibilities[i][s].*`, `numVisibilitiesPerFreqPerStoke[i][s]` (4992–5126) | Global `datasets` (extern MSDataset*), `nMeasurementSets` |
| **dchi2** (host) | Same loop; uses `datasets[d].fields[f].device_visibilities[i][s].Vr, uvw, weight`, `N_eff_perFreqPerStoke`, `numVisibilitiesPerFreqPerStoke`, `fields[f].nu[i]`, `antennas[0].*` (5146–5264) | Same |
| **Chi2 Fi** (chi2.cu) | Calls `chi2(p, ip, ...)` and `dchi2(p, xi, result_dchi2, ip, ...)`; no direct dataset access | Indirect via chi2/dchi2 |

**Summary chi2/dchi2:** **New path:** when all datasets have `gpu.num_fields() > 0`, chi2/dchi2 use chi2_chunked/dchi2_chunked (field → baseline → chunk, gather per (dd_id, chan, pol), degrid, residual, chi2Vector/DChi2). Legacy path still used when gpu not populated.

### 2.1 Design notes (chi2/dchi2 refactor)

**N_eff (effective number of samples):**
- **Responsibility:** The dataset (e.g. `MSWithGPU` or `ChunkedVisibilityGPU`) should own N_eff. Expose a function such as `computeEffectiveSamples()` or `getNeff(field_id, dd_id, chan, pol)` that returns (Σw)² / Σw² for the relevant visibilities. Chi2/dchi2 then call into the dataset instead of using precomputed `N_eff_perFreqPerStoke` on the legacy view. Today `precomputeNeff()` fills legacy `fields[f].N_eff_perFreqPerStoke[i][s]`; the new path should use a dataset method that can iterate the new layout (field → baseline → chunk) and compute N_eff per (dd_id, chan, pol) or per group.

**Chi2 without joined/gathered arrays:**
- **Loop only:** Chi2 should be “loop over the dataset” and accumulate χ²; it does not need to build a single contiguous “chunked array” of all visibilities.
- **Residuals in the dataset:** Vm and Vr should live in the dataset (chunk buffers), not in temporary host/device buffers. Flow: (1) Compute Vm (degrid) and **write into** each chunk’s `Vm` at the right (chan, pol) offset; (2) Compute Vr = Vo − Vm **in place** in the chunk. Then chi2 is a reduction over all (chunk, offset) using the dataset’s Vr and weight (e.g. a kernel that takes pointer arrays `Vr_ptrs[]`, `weight_ptrs[]`, offset, and does a reduction). Temp buffers for “gathered” Vo/Vm/Vr/weight/uvw can be avoided if we have a “scatter degrid” kernel (write Vm into chunk.Vm from grid) and a “reduction over scattered Vr/weight” kernel.
- **Stokes/correlations agnostic:** Chi2 should be agnostic to Stokes vs correlations. Do not filter by RR/LL/XX/YY only; loop over all polarizations (or all that the dataset exposes). The same formula χ² = Σ w|Vr|² applies regardless of whether the data are in Stokes or correlation space.

**dchi2 dimensions:**
- **Current layout:** `result_dchi2` is allocated as `(image_count) * M * N` with layout `(n_images, M, N)`. For MFS, image 0 = I_nu_0, image 1 = alpha. `DChi2_total_I_nu_0` writes to `dchi2_total[N*i+j]` (first image); `DChi2_total_alpha` writes to `dchi2_total[N*M + N*i + j]` (second image). So today: **dimensions = (n_images, M, N)**.
- **Desired layout:** dchi2 should have dimensions **(stokes, n_images, M, N)** — stokes (or polarization index), image index (e.g. I_nu_0, alpha), and pixel (M, N). This allows gradients per Stokes and per image. Verify and refactor `result_dchi2`, `addToDphi`, and the DChi2_total_* kernels so that indexing is (stokes, n_images, M, N) and all call sites (Chi2 Fi, optimizers) use the new layout.

---

## 3. **MFS for I_nu_0 and alpha** – mfs.cu + functions.cu

| Location | What uses dataset | Notes |
|----------|-------------------|--------|
| **mfs.cu** | `chi2->setFgScale`, `setCKernel`; optimizer runs on image; chi2/dchi2 compute gradient w.r.t. I_nu_0 and alpha (1037–1047) | Same dataset source as above |
| **functions.cu** | `DChi2_total_I_nu_0`, `DChi2_total_alpha` use `vars_gpu[].device_dchi2`, `datasets[d].fields[f].nu[i]`, `nu_0` (5235–5255) | Gradient accumulation over frequencies; needs field and frequency info |

**Summary MFS:** I_nu_0/alpha optimization uses the same chi2/dchi2 and thus the same dataset layout. Supporting the new model means providing the same per-frequency and per-field information (e.g. from MeasurementSet metadata + ChunkedVisibilityGPU chunks).

---

## 4. **Errors (I_nu_0, alpha)** – calculateErrors / noise_reduction

| Location | What uses dataset | Legacy access |
|----------|-------------------|----------------|
| **calculateErrors** (functions.cu) | When chunked path: `calculateErrors_chunked` loops over ChunkedVisibilityGPU (field → baseline → chunk) and metadata; gathers weights per (dd_id, chan, pol), deviceReduce for sum_weights, I_nu_0_Noise/alpha_Noise/covariance_Noise, noise_reduction. Otherwise legacy loop over `datasets`, `fields`, `data.total_frequencies`, `nstokes`, `device_visibilities`, `antennas[0].*`. | Branches via `use_chi2_chunked_path()`; legacy path uses global `datasets` |
| **noise_reduction** (kernel) | Operates on image-sized buffers (noise_I, N, M); no direct visibility access | Uses pre-filled error image from host loop |
| **SecondDerivateError** (objective_function) | `calculateErrorImage(I, v)` → `calculateErrors(I, fg_scale)` (secondderivateerror.cu) | Gets dataset via Visibilities held by synthesizer |
| **mfs.cu** | `error->calculateErrorImage(this->image, this->visibilities)` | Visibilities holds datasets |

**Summary errors:** **New path done.** When all datasets have `gpu.num_fields() > 0`, `calculateErrors()` uses `calculateErrors_chunked` (ChunkedVisibilityGPU + metadata). Legacy path still used when gpu not populated.

### 4.1 Design notes (errors: per n_images, per stokes)

- **Per n_images:** Error maps (σ, Cov, ρ) should be produced **per image** (e.g. per I_nu_0, per alpha in MFS). Today the error array has fixed layout (σ(I_nu_0), σ(alpha), Cov, ρ) for 2 images; the intended design is a layout that scales with `image->getImageCount()` and optionally separates variance/covariance per image index.
- **Per stokes:** Errors should be accumulated **per Stokes** (or per correlation) so that each Stokes component has its own Fisher accumulation and resulting σ/Cov/ρ. That implies: (1) loop over polarizations and accumulate Fisher terms into buffers indexed by (stokes, image_index, pixel), not a single combined buffer; (2) error array layout **(stokes, n_images, …)** or equivalent (e.g. σ and Cov per (stokes, image)); (3) `noise_reduction` (or a variant) applied per (stokes, image) to produce per-Stokes, per-image error maps. Refactor `calculateErrors_chunked`, `I_nu_0_Noise` / `alpha_Noise` / `covariance_Noise` indexing, and `noise_reduction` (and any Io/error writers) to support this layout.

---

## 5. **Gridding and degridding** – functions.cu

| Function | Signature / dataset use | Legacy types |
|----------|--------------------------|--------------|
| **do_gridding** | `do_gridding(std::vector<Field>& fields, MSData* data, deltau, deltav, M, N, ckernel, gridding)` (1554–1561) | Iterates `data->nfields`, `data->total_frequencies`, `data->nstokes`; uses `fields[f].nu[i]`, `fields[f].numVisibilitiesPerFreqPerStoke[i][s]`, `fields[f].visibilities[i][s].uvw`, `.weight`, `.Vo`, backup copies | `Field` = legacy (HVis, backup_*, numVisibilitiesPerFreqPerStoke) |
| **degridding** | **Removed** (legacy `degridding(fields, data, ..., dataset)`). Native path: `do_degridding(MeasurementSet& ms, ChunkedVisibilityGPU* gpu, ...)` (GPU: image + FFT, write Vm into ChunkedVisibilityGPU). | Same |
| **Gridding::applyCriteria** (gridding.cu) | `do_gridding(v->getMSDataset()[d].fields, &v->getMSDataset()[d].data, ...)` | Visibilities → MSDataset → fields, data |

**Summary gridding/degridding:** Fully tied to legacy `Field` layout (per-field, per-frequency-index, per-stokes arrays of uvw, weight, Vo, Vm, and backup). To use the new model you’d either (a) implement gridding/degridding that take MeasurementSet + ChunkedVisibilityGPU (or host chunks) and loop field → baseline → chunk by data_desc_id, or (b) adapt from new MS to legacy `Field`/`MSData` before calling existing do_gridding/degridding.

### 5.1 Design: gridding on CPU, degridding on GPU

- **Gridding** in the legacy path runs entirely on the **CPU** (host vectors, OpenMP over fields/frequencies/stokes). The intended design for the native path is to **keep gridding as a CPU preprocessing step**: implement `do_gridding(MeasurementSet& ms, ChunkedVisibilityGPU* gpu, ...)` so it iterates the **host** MeasurementSet (field → baseline → time_sample → vis), grids into host buffers (same logic as legacy), then uploads to ChunkedVisibilityGPU for synthesis. No need to implement gridding on GPU for the main pipeline.
- **Degridding** in the legacy path runs on the **GPU** (image on device, FFT and `degriddingGPU` kernel on device, model visibilities Vm on device). The native path should keep **degridding on GPU**: `do_degridding(MeasurementSet& ms, ChunkedVisibilityGPU* gpu, ...)` uses the image and FFT on device and writes Vm into ChunkedVisibilityGPU. So: gridding = CPU preprocessing before upload; degridding = GPU using ChunkedVisibilityGPU and device image.

---

## 6. **Weighting schemes** – Natural, Briggs, Uniform, Radial

| File | What uses dataset | Legacy access |
|------|-------------------|----------------|
| **naturalweightingscheme.cu** | `apply(std::vector<MSDataset>& d)`; loops `d[j].fields[f]`, `numVisibilitiesPerFreqPerStoke[i][s]`, `visibilities[i][s].weight` (9–55) | `MSDataset`, `Field::visibilities`, `numVisibilitiesPerFreqPerStoke` |
| **briggsweightingscheme.cu** | Same; uses `d[j].fields[f].visibilities[i][s].uvw`, `.weight`, `.Vo` for robust weighting (30–155) | Same |
| **uniformweightingscheme.cu** | Same pattern (9–66) | Same |
| **radialweightingscheme.cu** | Same (8–48) | Same |
| **weightingscheme.cuh** | `virtual void apply(std::vector<MSDataset>& d)` | Interface takes `std::vector<MSDataset>&` |
| **mfs.cu** | `scheme->apply(datasets)`, `scheme->restoreWeights(datasets)` (549, 1247) | Passes full datasets |

**Summary weighting:** All schemes iterate dataset → field → frequency → stokes and touch `visibilities[i][s].uvw`, `.weight`, `.Vo`. New model support = new overload or implementation that takes MeasurementSet (and optionally ChunkedVisibilityGPU) and iterates field → baseline → time_sample (chunk), updating weight (and optionally Vo) in the new structure.

---

## 7. **Other call sites**

| Component | File | Dataset use |
|-----------|------|-------------|
| **Visibilities** | include/classes/visibilities.cuh | Holds `std::vector<MSDataset> datasets`; `setMSDataset`, `getMSDataset`, `applyWeightingScheme(scheme)` → `scheme->apply(datasets)` |
| **Gridding** | src/gridding.cu | `applyCriteria(Visibilities* v)` → `do_gridding(v->getMSDataset()[d].fields, &v->getMSDataset()[d].data, ...)` |
| **IO** | IoMS (ioms.cu, MSFITSIO.cu) | read/write use `fields`, `MSData`, `antennas`; writeResidualsAndModel(fields, data) |
| **framework.cuh** | Vars, globals | `nMeasurementSets`, gridding, etc. |

---

## 8. **Legacy data layout (reminder)**

- **MSDataset**: name, oname, `std::vector<Field> fields`, `std::vector<MSAntenna> antennas`, MSData (nfields, total_frequencies, nstokes, …).
- **Field**: ref_ra/dec, phs_ra/dec, ref_xobs_pix, phs_xobs_pix, `nu` (freq per channel), `visibilities[total_frequencies][nstokes]` (each = HVis: uvw, weight, Vo, Vm, Vr, S), `device_visibilities` (DVis), `numVisibilitiesPerFreqPerStoke`, backup_*, atten_image.
- **MSData**: nfields, total_frequencies, nstokes, n_internal_frequencies, channels[], etc.
- **Loop order**: dataset d → field f → frequency index i (0..total_frequencies-1) → stokes s.

New model loop: field → baseline → time_sample (each has data_desc_id → spw/freq); one time_sample = one (baseline, time, data_desc) with a vector of (chan, pol) visibilities.

---

## 9. **Suggested migration strategy**

1. **Adapter path (minimal change):** ✅ **Implemented.** `MSWithGPU` holds both new MS (ms, gpu) and legacy view (data, fields, antennas). `build_legacy_from_ms(dw)` fills the legacy view from `MeasurementSet` after read and weighting. chi2, dchi2, degridding, errors, IO use the legacy view; read, weighting, gridding, beam/noise use the new MS. Build is green.
2. **Native path (full):** **TODO.** Replace legacy view usage: reimplement chi2, dchi2, degridding (on GPU), calculateErrors, and IO on the new structure (field → baseline → chunk). Keep gridding as CPU preprocessing on MeasurementSet then upload. Then remove legacy `data`/`fields`/`antennas` and `build_legacy_from_ms`.

---

## 10. **Native path – progress**

**Done:**
- **MSWithGPU** (`include/ms/ms_with_gpu.h`): One MS + ChunkedVisibilityGPU + optional gridded_ms + atten_image. **Legacy view removed** (no data, fields, antennas).
- **build_legacy_from_ms()** and **src/ms/legacy_adapter.cu**: **Removed.**
- **Synthesizer**: Holds `std::vector<MSWithGPU>*` via `setDatasets`/`getDatasets()`. mfs sets `g_datasets = &datasets`.
- **Filter / Error**: `applyCriteria(std::vector<MSWithGPU>&)` and `calculateErrorImage(Image*, std::vector<MSWithGPU>&)` (native path).
- **WeightingScheme**: **apply(std::vector<MSWithGPU>&)** and **restoreWeights(std::vector<MSWithGPU>&)**; schemes work on new MS structure.
- **Natural, Briggs, Uniform, Radial** weighting: Implemented for the new structure.
- **Gridder**: **grid** and **degrid** use native path (`do_gridding(ms, gpu, ...)`, `do_degridding(ms, gpu, ...)`). Legacy `degridding(fields, data, ..., dataset)` **removed.**
- **do_gridding / do_degridding**: Full implementation; gridding = CPU preprocessing then upload; degridding = GPU (image + FFT, write Vm into ChunkedVisibilityGPU).
- **Beam/noise**: `MSWithGPU::computeNoiseAndBeamContribution` + `beamNoiseFromSums`.
- **mfs.cu**: Native path only; setDevice uses ms + gpu + atten_image; writeResiduals uses download, copy, stokes_to_correlations if needed, MSWriter::write; cleanup uses gpu.clear() and atten_image.
- **functions.cu**: chi2/dchi2/calculateErrors use chunked path only; **precomputeNeff()** is no-op when `use_chi2_chunked_path()`.

**Remaining (optional cleanup):**
- **Legacy types**: Remove MSDataset, MSData, Field, HVis, DVis, MSAntenna from MSFITSIO.cuh when no longer referenced by legacy IO or other code.

---

## 11. **Design: GPU/CPU and gridded/non-gridded**

- **No reflection type-erasure**: We use explicit types instead of a single “reflection” handle for GPU vs CPU or gridded vs non-gridded. **MSWithGPU** is the canonical unit: it holds both host (`MeasurementSet ms`) and device (`ChunkedVisibilityGPU gpu`) so the pipeline can use either as needed (e.g. weighting on host, synthesis on device). Operations take `std::vector<MSWithGPU>&` or `MSWithGPU&` and can iterate the same way for one or multiple datasets.
- **Gridded vs non-gridded**: “Non-gridded” = raw visibilities in `MeasurementSet` (and on device in `ChunkedVisibilityGPU`). “Gridded” = representation in `GriddedVisibilities` / `GriddedVisibilitySet` (e.g. produced by Gridder). The pipeline can work with either representation by passing the appropriate container; we do not type-erase them behind one interface. A future abstraction (e.g. `IVisibilityDataset` with host/device/gridded views) could be added if needed.

This file is a reference; code changes are tracked in git and in this section.

---

## 12. **Dataset abstraction (CPU / GPU / gridded) – recommended approach**

So we don’t forget: use **one concrete type first, then a thin abstraction**, and treat gridded as gridder output.

### Phase 1 (now): One concrete type only

- **Use `MSWithGPU` as the single dataset type** for the whole pipeline.
- Synthesizer, Gridder, WeightingScheme, beam/noise, and (after migration) chi2/degridding all take **`std::vector<MSWithGPU>&`** (or `std::vector<MSWithGPU>*`).
- **Do not** introduce an `IDataset` or variant yet. Finish migrating off `MSDataset` and get the build green.

Reasons: (1) We’re already close to “one type everywhere” with `MSWithGPU`; adding polymorphism now would touch the same call sites we’re still fixing. (2) A single concrete type keeps the migration predictable. (3) `MSWithGPU` already embodies “this dataset has both CPU and GPU”; upload/download are the sync points.

### Phase 2 (later): Thin abstraction for CPU / GPU / gridded

Once the refactor is stable:

- **Introduce a small `IDataset`** (or similar) with a minimal API: e.g. `name()`, `metadata()` (or equivalent), `num_visibilities()`, and optionally “visit visibilities” (e.g. `visit_visibilities(callback)` or `accept(IVisitor&)`) so weighting and beam/noise can stay backend-agnostic.
- **Implement `IDataset` for `MSWithGPU`** (adapter or make `MSWithGPU` implement it). The pipeline type becomes e.g. `std::vector<std::unique_ptr<IDataset>>` or `std::vector<IDataset*>`.
- **Gridded:** treat it as **output of the gridder**, not as another “dataset” in the same vector. So: input = list of datasets (CPU or GPU via `IDataset`); gridder produces a gridded representation (e.g. `GriddedVisibilitySet` or a `GriddedDataset`); degridder / imaging take that gridded representation + image and write back into the same (or same-shaped) datasets.

So the “reflection-style” idea is: **one interface for sample-based datasets (CPU/GPU), and a separate concept for gridded** that the gridder creates and the rest consumes.

### Interface vs variant

- **Prefer a small abstract interface (`IDataset`)** if we want to pass “any dataset” through the same pointer type and add more backends later (e.g. distributed, compressed) without changing call sites.
- **`std::variant<CPUDataset, GPUDataset, …>`** is an alternative if we prefer no inheritance and are okay with `std::visit` at the boundaries.

Recommendation: **interface-first** (`IDataset` + `MSWithGPU` as first implementation), with gridded as a separate product of the gridder.

---

## 13. **Image geometry and ImageProcessor**

**Design:** Put all image geometry in **Image** (single source of truth). Keep **ImageProcessor** for multi-image MFS logic (calculateInu, apply_beam, chainRule, clipWNoise) and kernel (getCKernel); when it needs dimensions, take **Image\*** and use its getters. Pipeline (gridder, chi2, degridding) reads geometry from Image; remove geometry globals.

### Naming for image geometry (use these when implementing)

| Current (globals / legacy) | Preferred name | Meaning |
|----------------------------|----------------|---------|
| `M` | `ny` or `n_rows` | Number of pixels in first axis (image rows / v direction). |
| `N` | `nx` or `n_cols` | Number of pixels in second axis (image columns / u direction). |
| `DELTAX` | `pixel_scale_ra_deg` or `cell_ra_deg` | Sky pixel size in RA (degrees), e.g. CDELT1. |
| `DELTAY` | `pixel_scale_dec_deg` or `cell_dec_deg` | Sky pixel size in Dec (degrees), e.g. CDELT2. |
| `deltau` | `uv_cell_u` or `cell_u_lambda` | UV grid cell size in u (wavelengths). `uv_cell_u = 1.0 / (ny * pixel_scale_ra_rad)` (current: `1/(M*DELTAX_rad)`). |
| `deltav` | `uv_cell_v` or `cell_v_lambda` | UV grid cell size in v (wavelengths). `uv_cell_v = 1.0 / (nx * pixel_scale_dec_rad)` (current: `1/(N*DELTAY_rad)`). |

**Suggested Image API (geometry):**

- **Dimensions:** `nx()`, `ny()` (or `get_nx()`, `get_ny()`), `set_nx()`, `set_ny()` — number of pixels per axis.
- **Sky pixel scale (degrees):** `pixel_scale_ra_deg()`, `pixel_scale_dec_deg()` (and setters). Optionally store in radians for internal use.
- **UV cell size (lambda):** `uv_cell_u()`, `uv_cell_v()` — computed from `1.0 / (ny * pixel_scale_ra_rad)` and `1.0 / (nx * pixel_scale_dec_rad)` (matching current deltau/deltav), or stored and updated when nx/ny/pixel_scale change.
- **Phase center (if on Image):** `crpix1()`, `crpix2()`, `ra_deg()`, `dec_deg()` (or similar) instead of global `ra`, `dec`, `crpix1`, `crpix2`.

**ImageProcessor:** Keep `VirtualImageProcessor` / `ImageProcessor` for multi-image ops and kernel. Change `configure(int i)` to e.g. `configure(Image* img)` and use `img->nx()`, `img->ny()` (and `img->uv_cell_u()`, `img->uv_cell_v()` if needed) instead of extern M, N.

---

## 14. **Stokes selection in MSReader and correlation ↔ Stokes conversion**

Design so gpuvmem can (1) read only requested Stokes or correlations and detect what is present, and (2) convert between correlations and Stokes (with weight propagation).

### 14.1 MSReader: Stokes / correlation selection

**Goals:**

- Allow the user to specify which **Stokes parameter(s)** to read (e.g. `["I"]`, `["I","Q"]`, `["I","Q","U","V"]`) or which **correlations** to read.
- **Detect** from the MS whether the requested Stokes (or correlations) are present:
  - Read **CORR_TYPE** from the POLARIZATION table (per POLARIZATION_ID used in DATA_DESCRIPTION).
  - Map CORR_TYPE to correlation names (e.g. 5→XX, 6→YY, 9→XY, 10→YX for linear; 1→RR, 2→LL, 3→RL, 4→LR for circular).
  - Determine **feed type** (linear / circular / mixed) from the set of correlation types.
  - **Available Stokes** from correlations:
    - Circular: I,V from RR,LL; Q,U from RL,LR.
    - Linear: I,Q from XX,YY; U,V from XY,YX.
  - If the user requests Stokes (e.g. `["I"]`), check that the required correlations exist (e.g. I requires RR+LL or XX+YY); otherwise error or warn and optionally read what is possible.

**MSReadOptions extension (suggested):**

- `std::vector<std::string> requested_stokes` — e.g. `{"I"}`, `{"I","Q"}`. Empty = read all correlations (current behaviour).
- Or `std::vector<std::string> requested_correlations` — e.g. `{"XX","YY"}` to read only those.
- Reader logic: when `requested_stokes` is non-empty, (1) read POLARIZATION CORR_TYPE for each pol_id used, (2) compute available Stokes, (3) intersect with requested, (4) read only the correlation columns (pol indices) needed for those Stokes, or read all and mark which pol index corresponds to which correlation/Stokes.

**Metadata to add:**

- **Polarization info** per DATA_DESCRIPTION (or per POLARIZATION_ID): store `CORR_TYPE` (e.g. vector of ints or enum) and derived correlation names (e.g. `["XX","YY"]`) and feed type (linear / circular). This is needed both for “which Stokes can we form?” and for the conversion matrices below.

### 14.2 Dataset: “turn to Stokes” / “turn to correlations”

**Goals:**

- **Correlations → Stokes:** Given visibility data stored as correlations (current layout: per (chan, pol) with pol = correlation index), compute Stokes I, Q, U, V via the standard linear combinations (matrix multiply).
- **Stokes → Correlations:** Given Stokes, reconstruct correlation visibilities (e.g. for writing back to MS or for algorithms that expect correlations).
- **Weight / error propagation** both ways:
  - Correlations → Stokes: for each Stokes parameter S = Σ a_i C_i, variance σ²_S = Σ |a_i|² σ²_Ci; weights w = 1/σ², so new weight after combination. Prefer SIGMA or SIGMA_SPECTRUM when present; else use σ² = 1/w from WEIGHT.
  - Stokes → Correlations: same idea in reverse; backup original weights when converting to Stokes so they can be restored when converting back.

**Stokes–correlation relationships (same as in the Python Polarization class):**

- **Circular:**  
  I = (RR + LL)/2, V = (RR − LL)/2, Q + iU = RL, Q − iU = LR.  
  RR = I + V, LL = I − V, RL = Q + iU, LR = Q − iU.
- **Linear:**  
  I = (XX + YY)/2, Q = (XX − YY)/2, U + iV = XY, U − iV = YX.  
  XX = I + Q, YY = I − Q, XY = U + iV, YX = U − iV.

**Where it can live:**

- **Option A – Polarization helper class** (e.g. `include/ms/polarization.h`, `src/ms/polarization.cu`): Holds CORR_TYPE-derived state per data_desc (or per pol_id): correlation names, feed type, conversion matrices (Stokes↔correlations). Provides:
  - `get_available_stokes(data_desc_id)` (or pol_id),
  - `get_stokes_to_correlation_matrix(data_desc_id, stokes_list)`,
  - `get_correlation_to_stokes_matrix(data_desc_id, stokes_list)`,
  - `correlations_to_stokes(MeasurementSet& ms, stokes_list)` — in-place or into a “Stokes view” structure,
  - `stokes_to_correlations(MeasurementSet& ms)` — inverse,
  - and weight/sigma propagation in both directions (formulas above).
- **Option B – Methods on MeasurementSet or MSWithGPU:** e.g. `ms.to_stokes(stokes_list)` / `ms.to_correlations()` that call into a shared Polarization helper and rewrite visibility buffers (and weights) accordingly.

**VisSample / layout:**

- Today each `VisSample` has `chan`, `pol`, and complex values; `pol` is an index 0..npol-1. To support “Stokes space” as first-class, either:
  - Keep the same layout and **interpret** `pol` as either “correlation index” or “Stokes index” (and store in metadata which mode we are in), or
  - Add a small enum or flag per data_desc: “storage_order = correlations | stokes” and then pol index maps to correlation or to Stokes (I=0, Q=1, U=2, V=3) according to that flag. Conversion then rewrites the buffers and updates the flag (and npol if needed).

**Weight propagation (summary):**

- **Correlations → Stokes:** For S = Σ a_i C_i, use σ²_S = Σ |a_i|² σ²_i with σ²_i = 1/w_i (or SIGMA² when present). Set new weight w_S = 1/σ²_S. Zero weights → σ² = 0; skip or set result weight to 0. Handle NaN.
- **Stokes → Correlations:** Same formula in reverse when expressing each correlation as a linear combination of Stokes; compute σ² for each correlation and set weight = 1/σ². When converting back after having converted to Stokes, can optionally restore original correlation weights from a backup if they were stored.

### 14.3 Implementation order (suggested)

1. **Metadata:** ✅ **Done.** Read and store CORR_TYPE (and NUM_CORR) per POLARIZATION_ID in the reader; attach to a new **Polarization** table in MeasurementSetMetadata. `Polarization` class in `include/ms/metadata.h`; reader in `src/ms/ms_reader.cu` reads CORR_TYPE and adds polarizations; per-DD npol from polarization.
2. **Polarization helper:** **Done.** `include/ms/polarization.h` and `src/ms/polarization.cu`: `PolarizationHelper(metadata)`; `feed_type(pol_id)` / `feed_type_for_data_desc(data_desc_id)`; `correlation_names`, `available_stokes`; `get_correlation_to_stokes_matrix` and `get_stokes_to_correlation_matrix` (complex, per pol_id or data_desc_id).
3. **MSReadOptions:** **Done.** `requested_stokes` and `requested_correlations` in MSReadOptions; reader calls `apply_pol_selection()` after metadata; per-DD `selected_pol_indices` and npol set; only selected pols read and stored (pol = 0..selected-1). DataDescription has `selected_pol_indices` / `has_pol_selection()`.
4. **Conversion:** **Done.** `correlations_to_stokes(ms, stokes_list)` and `stokes_to_correlations(ms)` in `polarization.cu` (in-place, weight propagation σ²_S = Σ |a_i|² σ²_i). MeasurementSet has `storage_mode()` (Correlations | Stokes) and `stored_stokes_list()`. **Writer requires correlation mode:** call `stokes_to_correlations(ms)` before writing to MS.
5. **Integration:** **TODO.**
6. **Stokes imaging (Section 14.4):** Image dimension and validation — see 14.4 below.

Use "Stokes mode" or "correlation mode" in chi2/gridding/degridding as needed (e.g. image Stokes I only from Stokes I visibilities).

### 14.4 Stokes imaging: image dimension and validation

**Goals:**

- When instantiating an image, support a **Stokes dimension** so the user can image one or more Stokes parameters (e.g. "I" by default, or "I,Q,U,V" for full polarimetry).
- The **size of the Stokes dimension** (number of requested Stokes) is the **image plane count** (`image_count`): e.g. ["I"] → 1, ["I","Q","U","V"] → 4. That dimension drives buffer sizes (image, gradients) everywhere: `device_Image`, `dphi`, `result_dchi2`, `xi` are all `M * N * image_count`, so **gradient dimensions follow automatically** from the image's Stokes dimension.
- **Validation:** Before accepting a requested Stokes list, the program must **cross-check with the dataset(s)**. For each measurement set, every data description must be able to form all requested Stokes from its correlations (e.g. I,V from RR,LL; Q,U from RL,LR for circular). Use `PolarizationHelper::available_stokes_for_data_desc()` (or a helper `stokes_supported_by_metadata(meta, requested_stokes)`) and require that for **all** datasets, **all** data descriptions support the full requested list. If the user asks for "I,Q,U,V" but a dataset only has RR,LL (only I,V), fail at configure with a clear error.

**Suggested flow:**

1. **Config:** Add an option (e.g. `stokes` or `requested_stokes`) with default "I". Allowed values: "I", "I,Q", "I,Q,U,V", etc. Parse to a `std::vector<std::string> requested_stokes`; its size = `image_count` for Stokes imaging.
2. **Initial values:** Keep `initial_values` as one numeric value per image plane (one per Stokes), e.g. one value for I, or four for I,Q,U,V.
3. **After MS read:** For each dataset, call `stokes_supported_by_metadata(datasets[d].ms.metadata(), requested_stokes)`. If any returns false, error with a message listing which Stokes are requested and which are available per dataset.
4. **Image / objective function:** Create `Image(..., image_count, M, N)` with `image_count = requested_stokes.size()`. Configure objective function and optimizers with the same `image_count`; gradient buffers (`dphi`, `result_dchi2`) are already `M*N*image_count`, so **no change** to gradient layout is required when the Stokes dimension is used as the image dimension.

**Summary:** Default Stokes = "I" (one image plane). Multi-Stokes (e.g. "I,Q,U,V") sets the image (and gradient) dimension; validation ensures the dataset(s) can form those Stokes before proceeding.

**Clip by noise (clipWNoise):** For MFS (`image_count == 2`), `ImageProcessor::clipWNoise` uses `linkClipWNoise2I` (I_nu_0 and alpha). For Stokes or single-plane imaging (`image_count != 2`), it uses `linkClipStokesWNoise`: only **Stokes I (plane 0)** is clipped by the noise map (where noise > noise_cut, set to 1e-10 or -η·MINPIX). **Stokes Q, U, V (planes 1..) are left unchanged** because they can and must take negative values; clipping them to a positive floor would corrupt polarization.

**Error maps (calculateErrors_chunked):** When `image_count == 2` (MFS), error layout is unchanged: I_nu_0, alpha, covariance, ρ (Fisher 2×2 inversion). When `image_count != 2` (Stokes or single-plane), one σ (standard deviation) per image plane is computed: inverse-variance is accumulated per (chan, pol) as inv_var(S_p) += fg_scale²·atten²·sum_weights, then `stokes_noise_reduction` converts to σ = 1/√inv_var per plane. Error images are written as `error_stokes_0.fits`, … when in Stokes/single mode; MFS still writes `error_Inu_0.fits`, `error_alpha_0.fits`, etc.
