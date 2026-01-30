# FITS I/O refactor: C++ library and interface design

## Current state

- **CFITSIO** (C, `fitsio.h`) is used via:
  - **MSFITSIO.cuh / MSFITSIO.cu**: low-level wrappers (`openFITS`, `closeFITS`, `OCopyFITS`, `readFITSHeader`, `open_fits`, etc.). The header also pulls in Casacore, CUDA, Boost and defines legacy MS types (MSData, Field, etc.), which mixes FITS I/O with unrelated concerns.
  - **Io** base class + **IoFITS**: many overloads (`printImage`, `printNormalizedImage`, `printNotNormalizedImage`, `printImageIteration`, …) that funnel into `OCopyFITS` with 15+ parameters.
- **Problems**:
  - No RAII: manual open/close, errors often handled with `exit()`.
  - Parameter-heavy, repetitive APIs; multi-plane (Stokes, error planes) is expressed as an `index` into a flat buffer rather than a first-class concept.
  - FITS types and legacy MS types live in the same header; hard to test or reuse FITS code alone.
  - Writing a slice is always “copy header from template + write one 2D image”; no single place that describes “write N planes with shared header” in one call.

## Goal

1. Use a **C++ FITS library** (better encapsulation, RAII, exceptions or consistent error handling).
2. Expose interfaces that match current use cases: **multi-plane images** (Stokes I/Q/U/V, MFS I_nu_0/alpha, error maps), **header template**, **units**, **iteration metadata**, and optional **GPU buffer** (copy to host then write).

## Recommended library: CCfits

**CCfits** (HEASARC) is the standard C++ wrapper around CFITSIO:

- Object-oriented: `FITS`, `HDU`, image/table extensions, exceptions (`FitsException`).
- Requires CFITSIO; no new low-level dependency, just a C++ layer on top.
- Documentation: <https://heasarc.gsfc.nasa.gov/fitsio/CCfits/>
- Build: usually `find_package(CCfits)` or manual include/link against `CCfits` and CFITSIO.

**Alternative**: Keep CFITSIO only and add a **thin in-tree C++ wrapper** (RAII, options structs, multi-plane–aware helpers). No new dependency; later swap the implementation to CCfits if desired.

## Target API shape

Design the public API so that:

1. **Header and metadata** are separate from legacy MS/Casacore types. Use a `FitsHeader` struct with FITS-aligned names: `naxis1`/`naxis2`, `cdelt1`/`cdelt2`, `crval1`/`crval2`, `crpix1`/`crpix2`, `beam_maj`/`beam_min`/`beam_pa`, `radesys`, `equinox`, `bitpix`, etc. (no `M`/`N` or `ra`/`dec` to avoid ambiguity with FITS keywords.)
2. **Read**:
   - `FitsHeader read_fits_header(const std::string& path);`
   - `std::vector<float> read_fits_image_float(const std::string& path);`  
   Optional: support for multi-HDU or specified HDU.
3. **Write**:
   - Single entry point for “write one 2D slice from a buffer (possibly GPU)” with options:
     - `header_template` (copy header from this FITS file),
     - `output_path`, `data` (host or device), `naxis1`/`naxis2`, `plane_index`,
     - `bunit`, `niter`, `normalization_factor`, `radesys`, `equinox`, `crval1`/`crval2`,
     - `normalize` (apply normalization_factor to pixels),
     - `data_on_device` (if true, copy device → host before writing).
   - Struct + free function: `WriteFitsImageOptions` and `void write_fits_image_slice(const WriteFitsImageOptions& opts);`
   - Optional: `write_fits_image_planes(opts, nplanes)` that writes multiple slices (e.g. Stokes or error planes) reusing the same header and avoiding repeated template open/close.
4. **Complex images** (e.g. MEM): same pattern with a `cufftComplex*` or `std::complex<float>*` buffer and an option for amplitude/phase/real/imag.

Then:

- **IoFITS** (or a new `FitsImageHandler`) can call `read_fits_header`, `read_fits_image_float`, `write_fits_image_slice` with options built from its member state (path, M, N, frame, equinox, fg_scale, etc.), reducing the number of overloads and parameters at the Io boundary.
- **mfs.cu** and other call sites pass a single options struct or a small set of arguments instead of 15 positional parameters.
- **Stokes / error maps**: naturally expressed as multiple planes (e.g. loop over `plane_index` or call `write_fits_image_planes` with `nplanes` and a base filename pattern).

## Implementation options

| Option | Pros | Cons |
|--------|------|------|
| **A. Add CCfits** | Standard C++ API, exceptions, good long-term fit | New dependency, packaging/build may need updates |
| **B. Thin in-tree C++ wrapper over CFITSIO** | No new dependency, full control over API | More code to maintain; later swap to CCfits possible |
| **C. Hybrid** | Introduce the new API (options struct, read/write helpers) implemented with CFITSIO (Option B); later replace implementation with CCfits (Option A) | Two-phase migration |

**Recommendation**: **C. Hybrid.** Introduce a small `fits` namespace (or `gpuvmem::fits`) with:

- `FitsHeader` (or reuse a cleaned `headerValues` in a FITS-only header),
- `read_fits_header()`, `read_fits_image_float()`,
- `WriteFitsImageOptions` + `write_fits_image_slice()`,

implemented first with CFITSIO behind RAII (e.g. a `FitsFile` handle that closes in the destructor). Migrate IoFITS and call sites to this API. Once stable, the implementation can be switched to CCfits without changing call sites.

## Migration steps

1. **Extract FITS-only types** from MSFITSIO.cuh into a dedicated header (e.g. `include/fits/fits_io.h`): at least the header struct used for read/write (no Casacore/CUDA in that header).
2. **Implement** `read_fits_header`, `read_fits_image_float`, `write_fits_image_slice` (and optionally `write_fits_image_planes`) in `src/fits/fits_io.cu` using CFITSIO with RAII; keep CUDA copy (device → host) inside this module if needed.
3. **Refactor IoFITS** to use the new API: build `WriteFitsImageOptions` from Io state and call `write_fits_image_slice` (and read helpers). Deprecate or remove the many `printImage`/`printNormalizedImage` overloads in favor of a smaller set that fill the options struct.
4. **Optionally add CCfits** as a dependency and reimplement the `fits_io` module using CCfits; keep the same public API so callers do not change.
5. **Clean up** MSFITSIO.cuh: move legacy MS types to an MS-specific header; keep only what is still needed for FITS (if anything) or remove FITS from MSFITSIO entirely once migration is done.

## File layout (suggested)

- `include/fits/fits_io.h` – FitsHeader, WriteFitsImageOptions, read/write declarations.
- `src/fits/fits_io.cu` – implementation (CFITSIO + RAII; or CCfits).
- `cmake/FindCCfits.cmake` – optional, when moving to CCfits.
- Io and mfs continue to use the new API; MSFITSIO.cuh/cu are simplified or relegated to legacy-only use.

This keeps FITS I/O in one place, makes multi-plane and Stokes/error outputs natural, and allows a later switch to CCfits without disrupting the rest of the codebase.
