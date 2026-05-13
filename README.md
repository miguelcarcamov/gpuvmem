<h1 align="center">
  <img src="https://github.com/miguelcarcamov/gpuvmem/wiki/images/logos/logo2.png" height="320" alt="gpuvmem logo">
</h1>

<p align="center">
  <strong>Multi-GPU regularized imaging for radio astronomy</strong><br>
  <strong>C++/CUDA framework (RML-style: data term + regularizers)</strong><br>
  <em>Originally maximum-entropy (MEM) synthesis — see paper below.</em>
</p>

<p align="center">
  <em>In one line:</em> turn measurement sets and a sky grid into an image by minimizing <strong>data fidelity + priors</strong> on the GPU, with optimizers and regularizers you can swap like LEGO bricks.
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.ascom.2017.11.003"><img src="https://img.shields.io/badge/A%26C-Paper-006599?logo=academia&logoColor=white" alt="Paper"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://cmake.org/"><img src="https://img.shields.io/badge/CMake-%E2%89%A53.18-064F8C?logo=cmake&logoColor=white" alt="CMake"></a>
  <a href="https://en.cppreference.com/w/cpp/17"><img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white" alt="C++17"></a>
  <a href="https://developer.nvidia.com/cuda-zone"><img src="https://img.shields.io/badge/CUDA-GPU-76B900?logo=nvidia&logoColor=white" alt="CUDA GPU"></a>
</p>
<p align="center">
  <a href="https://github.com/miguelcarcamov/gpuvmem/actions/workflows/workflow.yml"><img src="https://github.com/miguelcarcamov/gpuvmem/actions/workflows/workflow.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/miguelcarcamov/gpuvmem/pkgs/container/gpuvmem"><img src="https://img.shields.io/badge/ghcr.io-container-2496ED?logo=github" alt="Container"></a>
  <a href="https://github.com/miguelcarcamov/gpuvmem/stargazers"><img src="https://img.shields.io/github/stars/miguelcarcamov/gpuvmem?style=social&logo=github" alt="GitHub stars"></a>
  <a href="https://github.com/miguelcarcamov/gpuvmem/commits/master"><img src="https://img.shields.io/github/last-commit/miguelcarcamov/gpuvmem?logo=github&logoColor=white&label=last%20commit" alt="Last commit"></a>
</p>
<p align="center">
  <a href="https://github.com/miguelcarcamov/gpuvmem/issues"><img src="https://img.shields.io/github/issues/miguelcarcamov/gpuvmem?logo=github" alt="Issues"></a>
  <a href="https://github.com/miguelcarcamov/gpuvmem/pulls"><img src="https://img.shields.io/github/issues-pr/miguelcarcamov/gpuvmem?logo=github" alt="Pull requests"></a>
  <a href="https://github.com/miguelcarcamov/gpuvmem/graphs/contributors"><img src="https://img.shields.io/github/contributors/miguelcarcamov/gpuvmem?logo=github" alt="Contributors"></a>
  <a href="https://github.com/miguelcarcamov/gpuvmem"><img src="https://img.shields.io/github/repo-size/miguelcarcamov/gpuvmem?logo=github&label=repo%20size" alt="Repo size"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
  <a href="https://gitlab.com/clirai/pyralysis"><img src="https://img.shields.io/badge/Pyralysis-GitLab-FC6D26?logo=gitlab&logoColor=white" alt="Pyralysis on GitLab"></a>
</p>

---

## What is gpuvmem?

**gpuvmem** began as a CUDA implementation of **maximum-entropy (MEM)** image synthesis for radio interferometry, described in the **2018 *Astronomy & Computing* paper** linked below. Since then it has grown into a broader **regularized imaging** tool: a **C++/CUDA framework** in the spirit of **regularized maximum likelihood (RML)**. You minimize **data fidelity** (χ² on visibilities) plus **priors and penalties** (entropy, L1, TV, …), and you can swap **optimizers**, **line searches**, **seeders**, **gridding kernels**, and **weighting schemes** without rewriting the whole pipeline.

The **name and paper** still say MEM; the **architecture** is a pragmatic **multi-GPU RML workbench** on **measurement sets** and **FITS**, built on **casacore**, **CCfits**, and friends.


|                        |                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Paper**              | [doi:10.1016/j.ascom.2017.11.003](https://doi.org/10.1016/j.ascom.2017.11.003)                               |
| **Wiki**               | [github.com/miguelcarcamov/gpuvmem/wiki](https://github.com/miguelcarcamov/gpuvmem/wiki)                     |
| **Contributing**       | [CONTRIBUTING.md](CONTRIBUTING.md)                                                                           |
| **Changelog**          | [CHANGELOG.md](CHANGELOG.md)                                                                                 |
| **Pyralysis (Python)** | [gitlab.com/clirai/pyralysis](https://gitlab.com/clirai/pyralysis) — see [below](#pyralysis-and-the-roadmap) |


---

## Pyralysis and the roadmap

A more **modular, flexible, and extensible** Python stack for radio astronomy analysis and image synthesis — **[Pyralysis](https://gitlab.com/clirai/pyralysis)** (*PYthon Radio Astronomy anaLYSis and Image Synthesis*) — is under active development on GitLab.

**Today:** Pyralysis does **not** yet ship a **GPU** imaging backend comparable to gpuvmem’s CUDA path, so **this repository remains the reference for high-performance GPU regularized imaging** in that line of work.

**Future:** Once Pyralysis offers a **full CPU + GPU backend** with **Dask**-scale execution end-to-end, **gpuvmem is expected to be discontinued or deprecated** in favour of that stack. Until then, both projects can coexist: Pyralysis for Python-first workflows and experimentation, gpuvmem for CUDA-accelerated production runs tied to the current C++/CUDA architecture.

---

## Table of contents

- [Pyralysis and the roadmap](#pyralysis-and-the-roadmap)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Dependencies (detailed)](#dependencies-detailed)
- [Build and run](#build-and-run)
- [Install on the system](#install-on-the-system)
- [Usage (CLI)](#usage-cli)
- [Extending the framework](#extending-the-framework)
- [Image restoration](#image-restoration)
- [Tests](#tests)
- [Docker](#docker)
- [Citing](#citing)
- [Contributors](#contributors)
- [Star history](#star-history)

---

## Quick start

```bash
git clone https://github.com/miguelcarcamov/gpuvmem.git
cd gpuvmem
git lfs install && git lfs pull    # large test assets; needed for ctest
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
./../bin/gpuvmem --help            # lists every flag; works before CUDA sees a GPU
```

After build, the **executable** lives in **`bin/`** at the **repository root**, and the internal **static libraries** in **`lib/`** (see [Build and run](#build-and-run)). From there, jump to [Usage (CLI)](#usage-cli) when you are ready to wire a real dataset.

---

## Requirements


| Component            | Notes                                                                            |
| -------------------- | -------------------------------------------------------------------------------- |
| **CMake**            | ≥ **3.18** (see `cmake_minimum_required` in `CMakeLists.txt`)                    |
| **CUDA**             | Toolkit on **PATH**; GPU compute capability detected or set with `-DCUDA_ARCH=…` |
| **C++**              | **C++17** by default (CUDA 13+ needs it)                                         |
| **casacore**         | Build from source recommended; distro packages often lag features gpuvmem needs  |
| **CFITSIO / CCfits** | FITS I/O                                                                         |
| **Boost**            | Headers + libraries used by the project                                          |
| **OpenMP**           | Used for host parallel sections                                                  |
| **git-lfs**          | For test measurement sets and FITS data                                          |


---

## Dependencies (detailed)

### git-lfs

```bash
sudo apt-get install git-lfs
```

### casacore (example: v3.2.1 from source)

```bash
git clone --single-branch --branch v3.2.1 https://github.com/casacore/casacore.git
cd casacore
# Install build deps for your distro (example for Debian/Ubuntu):
sudo apt-get install -y build-essential cmake gfortran g++ libncurses5-dev libreadline-dev \
  flex bison libblas-dev liblapacke-dev libcfitsio-dev wcslib-dev libhdf5-serial-dev \
  libfftw3-dev libboost-all-dev
mkdir build && cd build
cmake -DUSE_FFTW3=ON -DUSE_OPENMP=ON -DUSE_HDF5=ON -DUSE_THREADS=ON ..
make -j$(nproc)
sudo make install
```

Use a **casacore** version compatible with your workflows (≥ ~3.1.2 has been referenced historically; prefer a maintained release).

### Boost and CFITSIO (Debian/Ubuntu examples)

   ```bash
sudo apt-get install -y libboost-all-dev libcfitsio-dev
```

### CUDA

Install the NVIDIA CUDA toolkit matching your driver. Put **`nvcc`** and the CUDA libraries on your **`PATH`** / **`LD_LIBRARY_PATH`** (or use environment modules).

---

## Build and run

   ```bash
   cd gpuvmem
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

**Outputs (outside `build/` only):**


| Path                           | Content                                         |
| ------------------------------ | ----------------------------------------------- |
| `../bin/gpuvmem`               | Main program (`src/main.cu`)                    |
| `../lib/libgpuvmem_*_static.a` | Internal static archives linked into the binary |


Optional: pass **`-DPREFIX=/path`** at configure time to change where the **runtime** binary is written (see `CMakeLists.txt`).

**Antenna configurations** are read from the measurement set; no separate antenna file is required for that part of the pipeline.

---

## Install on the system

CMake generates install rules. With **Makefiles**, **`make install`** is the same idea as **`cmake --install .`** from the build directory.

```bash
   cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build . -j$(nproc)
sudo cmake --install . --prefix /usr/local
# or: sudo make install
```

**Staged install** (packaging):

```bash
make install DESTDIR=/tmp/stage
# → /tmp/stage/usr/local/... when CMAKE_INSTALL_PREFIX=/usr/local
```

### CMake install switches


| Option                        | Default | Role                                                             |
| ----------------------------- | ------- | ---------------------------------------------------------------- |
| `GPUVMEM_INSTALL_EXECUTABLE`  | **ON**  | Install `gpuvmem` to `CMAKE_INSTALL_BINDIR` (e.g. `prefix/bin`). |
| `GPUVMEM_INSTALL_STATIC_LIBS` | **OFF** | Install internal `.a` libraries to `CMAKE_INSTALL_LIBDIR`.       |
| `GPUVMEM_INSTALL_HEADERS`     | **OFF** | Install `include/` under `CMAKE_INSTALL_INCLUDEDIR/gpuvmem`.     |


**End user (binary only):** defaults are enough.

**Developer (custom `main` against installed libs):**

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/gpuvmem \
      -DGPUVMEM_INSTALL_STATIC_LIBS=ON \
      -DGPUVMEM_INSTALL_HEADERS=ON
cmake --build . -j$(nproc) && sudo cmake --install . --prefix /opt/gpuvmem
```

Re-run **`cmake ..`** after changing install options, then rebuild and install again.

---

## Usage (CLI)

You describe **where the data lives**, **what pixel grid to use**, and **how to start the image**. gpuvmem does the rest on the GPU: χ² on visibilities plus whichever regularizers you turn on, with your choice of optimizer and line search.

### The mental model (three pieces)

1. **Visibilities** — Measurement sets in with **`-i`**, products out with **`-o`** (comma-separated lists, same length).
2. **Grid on the sky** — Either a **FITS template** with WCS (**`-m` / `--model_input`**, e.g. a dirty image from **CASA tclean**), *or* a **synthetic grid**: **`--imsize M,N`**, **`--cellsize`** (arcseconds), **`--phase-center RA,DEC`** in degrees (ICRS). If you pass both a FITS and synthetic flags, **the FITS wins** (synthetic flags are ignored on purpose).
3. **Starting pixels** — **`-z`** gives **constant** values per image plane: the **first number is MINPIX** (positivity floor); extra numbers set planes like spectral index. Optional **`-I` / `--initial-model`** loads **plane 0** from a **2D float FITS** (same `M,N` as the grid); other planes still follow **`-z`**.

```bash
./bin/gpuvmem [options]
```

### Help, warranty, and “no GPU yet”

You can run **`-h` / `--help`** before CUDA is touched, so listing flags works even on a login node without a GPU. **`-w` / `--warranty`** and **`-c` / `--copyright`** print GPL text and exit cleanly when you are **not** starting a real run (no **`-i`**). If **`-i`** is set, those flags are **ignored** so a stray **`-w`** never skips validation.

Authoritative list of every flag lives in **`src/options.cu`**; **`--help`** tracks it automatically. **Note:** empty-string defaults print as **`[default: ]`** in the generated help; the **`-L`** / **`-B`** descriptions spell out the real runtime behavior (**Brent**, no seeder). **Parsing:** a **`Flags` / getopt** fix ensures spaced **`-O out.fits`** no longer corrupts short-option parsing for later flags such as **`-m`**.

### Cheat sheet by topic

**Data & I/O**

| Flag | Remember it as… |
|------|------------------|
| `-i` / `--input` | Input MS path(s). |
| `-o` / `--output` | Output MS path(s). |
| `-p` / `--path` | Where FITS cubes land (e.g. `mem/`). |
| `-O` / `--output_image` | Basename for images. |
| `-G` / `--gpus` | Which GPUs (`0`, `0,1`, …). |
| `-s` | Write the model to the **MODEL** column. |

**Grid & weights**

| Flag | Remember it as… |
|------|------------------|
| `-m` / `--model_input` | FITS = geometry + WCS template. |
| `--imsize`, `--cellsize`, `--phase-center` | Headless grid when you skip `-m`. |
| `-W` / `--weighting` | `natural`, `uniform`, `radial`, `briggs`, `robust` (pair **`-R`** with Briggs-style robustness). |

**Starting image & Stokes**

| Flag | Remember it as… |
|------|------------------|
| `-z` / `--initial_values` | Constants per plane; **first = MINPIX**. |
| `-I` / `--initial-model` | Optional FITS for **plane 0** only. |
| `-S` / `--stokes` | e.g. `I` or `I,Q,U,V` (multi-Stokes runs). |

**Optimization stack**

| Flag | Remember it as… |
|------|------------------|
| `-C` / `--optimizer` | Factory name (default **LBFGS**). |
| `-L` / `--linesearch` | Line search factory id; if omitted, **LBFGS** / **CG** use built-in **Brent** (`main` does not replace an empty `-L`). |
| `-B` / `--seeder` | Step-size seeder; if omitted, **none**. If you pass **`-B`** without **`-L`**, the driver sets **`-L` Brent** so the seeder attaches. |
| `-K` / `--lbfgs-m` | L-BFGS memory (ignored for other optimizers). |
| `-J` / `--optimization_mode` | `joint`, `block`, `one`, `alpha_static`. |
| `-Z` / `--regularization_factors` | Comma-separated **λ** weights (`regularization_weights` in code). **Fewer than five** values: χ² stays at **λ = 1**, then **entropy, L1, total squared variation (TSV)** (extra commas ignored). **Five or more:** first value is **χ²**, then entropy, L1, TSV. Default **Fi** wired in `main.cu`: those four terms only (no **L2ConstantPrior** in the stock binary). |

**Physics & tuning (short list)**

| Flag | Role |
|------|------|
| `-t` iterations | How long to iterate (default **1000**). |
| `-g` gridding | CPU gridding / thread count (default **1**). |
| `-e` eta | Entropy / positivity coupling (default **-1**). |
| `-F` ref_frequency | Reference ν in Hz. |
| `-n`, `-N`, `-r` | Noise, noise cut, random subsampling of data. |
| `-T`, `-A` | Thresholding / alpha masking knobs. |
| `-X`, `-Y`, `-V` | GPU block sizes (**`-1`** = pick for you). |

**Long-only & toggles**

| Long / flag | Role |
|-------------|------|
| `--metrics-file` | Same text as stdout after the run: **φ total** from one `calcFunction` pass, then **each active Fi** (only terms with non-zero λ are registered — with **`-Z`** omitted that is usually **χ²** and **TSV**; see [Defaults worth knowing](#defaults-worth-knowing)): `lambda`, raw **value**, and `lambda_times_value`. No reduced χ² or ad hoc normalizations; CPU and wall time at the end. |
| `--normalize` | Normalize χ² / effective samples. |
| `--modify-weights` | Experimental weight tweaks. |
| `-v` / `--verbose` | Extra **one-line run context** before the final summary (grid, GPU, weighting, optimizer, line search, seeder, L-BFGS **m**). |
| `-x`, `-a`, `-P`, `-E`, `-M` | No positivity, noise on data, FITS every iteration, error maps, radius mask. |

### Defaults worth knowing

Out of the box you get **1000** iterations, **natural** weighting, **LBFGS**, **joint** mode, **L-BFGS memory 10**, and **α mask 5σ**. **`--help`** prints **`[default: ]`** for options whose registered default is an empty string (**`-L`**, **`-B`**); at runtime **LBFGS** / **CG** still construct with **Brent**, and no **seeder** is attached unless you pass **`-B`**. **`-Z`:** if you omit it, **`main.cu`** leaves entropy and L1 at **λ = 0** (those terms are **not** added — **`ObjectiveFunction::addFi`** skips **λ == 0**) and applies a legacy **λ = 0.05** to **TSV** only, beside **χ²** at **λ = 1**. Pass **`-Z`** to supply comma-separated weights (layout in the table above). Full literals live in **`setDefaultVars`** (`src/options.cu`).

### What we might add next

Multi-plane **`-I`** (e.g. one FITS per Stokes), a small **config file** beside argv, and richer **MS-specific** flags are natural extensions; today everything is explicit on the command line.

---

## Extending the framework

Think **pipeline, not monolith**: a **synthesizer** (data + coordinates), an **optimizer** (CG family, L-BFGS, …), an **objective function**, and **Fi** terms you register (χ² plus any regularizer). Factories pick concrete classes at runtime. Weights for those terms come from **`-Z`**. If you have ever added a penalty to a likelihood, you already know the pattern — gpuvmem just does it with CUDA and radio datasets.

---

## Image restoration

For a **restored** image (model convolved with the CLEAN beam plus residuals in Jy/beam), follow **Cárcamo et al. (2018)**. A helper script lives in `scripts/restore.py`:

```bash
python scripts/restore.py residual_folder.ms mem_model.fits restored_output 2.0
```

The last argument is the **robust** parameter used when cleaning residuals.

---

## Tests

After **`git lfs pull`**, from the build directory:

   ```bash
ctest --output-on-failure
```

Individual cases live under `tests/` (e.g. `antennae`, `co65`, …).

---

## Docker

```bash
docker pull ghcr.io/miguelcarcamov/gpuvmem:latest
```

See GitHub Container Registry and workflow files under `.github/workflows/` for build details.

---

## Citing

If you use gpuvmem in research, please cite **Cárcamo et al.**:

```bibtex
@article{CARCAMO201816,
  title   = "Multi-GPU maximum entropy image synthesis for radio astronomy",
  journal = "Astronomy and Computing",
  volume  = "22",
  pages   = "16 - 27",
  year    = "2018",
  issn    = "2213-1337",
  doi     = "https://doi.org/10.1016/j.ascom.2017.11.003",
  url     = "http://www.sciencedirect.com/science/article/pii/S2213133717300094",
  author  = "M. Cárcamo and P.E. Román and S. Casassus and V. Moral and F.R. Rannou",
  keywords = "Maximum entropy, GPU, ALMA, Inverse problem, Radio interferometry, Image synthesis"
}
```

---

## Contributors

- **Miguel Cárcamo** — Universidad de Santiago de Chile (USACH) — [github.com/miguelcarcamov](https://github.com/miguelcarcamov/) — [miguel.carcamo@usach.cl](mailto:miguel.carcamo@usach.cl)  
- **Nicolás Muñoz** — Universidad de Santiago de Chile  
- **Fernando Rannou** — Universidad de Santiago de Chile  
- **Pablo Román** — Universidad de Santiago de Chile  
- **Simón Casassus** — Universidad de Chile  
- **Axel Osses** — Universidad de Chile  
- **Victor Moral** — Universidad de Chile

**Bugs and features:** use [GitHub Issues](https://github.com/miguelcarcamov/gpuvmem/issues) and [CONTRIBUTING.md](CONTRIBUTING.md).

**License:** [GNU General Public License v3.0](LICENSE.txt).

---

## Star history

