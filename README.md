<h1 align="center">
  <img src="https://github.com/miguelcarcamov/gpuvmem/wiki/images/logos/logo2.png" height="400" alt="gpuvmem logo">
</h1>

<p align="center">
  <strong>Multi-GPU regularized imaging for radio astronomy</strong><br>
  <strong>C++/CUDA framework (RML-style: data term + regularizers)</strong><br>
  <em>Originally maximum-entropy (MEM) synthesis — see paper below.</em>
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

**gpuvmem** began as a CUDA implementation of **maximum-entropy (MEM)** image synthesis for radio interferometry, described in the **2018 *Astronomy & Computing* paper** linked below. Since then the codebase has grown into a broader **regularized imaging** tool: a **C++/CUDA framework** in the spirit of **regularized maximum likelihood (RML)** — you minimize a **sum of a data fidelity term** (e.g. χ² on visibilities) and **multiple regularization / prior terms** (entropy, L1, total variation, etc.), with **modular optimizers**, **line search**, **seeders**, **synthesizers**, and **measurement-operator / gridding** pieces you can recombine.

So: the **name and paper** reflect the **MEM** roots; the **current architecture** is an **RML-style, multi-term, multi-GPU optimization framework** on top of **measurement sets** and **FITS** (**casacore**, **CCfits**, and related dependencies).

| | |
|--|--|
| **Paper** | [doi:10.1016/j.ascom.2017.11.003](https://doi.org/10.1016/j.ascom.2017.11.003) |
| **Wiki** | [github.com/miguelcarcamov/gpuvmem/wiki](https://github.com/miguelcarcamov/gpuvmem/wiki) |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |
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
./../bin/gpuvmem --help            # or: path/to/build/../bin/gpuvmem
```

After build, the **executable** is under **`bin/`** at the **repository root**, and internal **static libraries** under **`lib/`** (see [Build and run](#build-and-run)).

---

## Requirements

| Component | Notes |
|-----------|--------|
| **CMake** | ≥ **3.18** (see `cmake_minimum_required` in `CMakeLists.txt`) |
| **CUDA** | Toolkit on **PATH**; GPU compute capability detected or set with `-DCUDA_ARCH=…` |
| **C++** | **C++17** by default (CUDA 13+ needs it) |
| **casacore** | Build from source recommended; distro packages often lag features gpuvmem needs |
| **CFITSIO / CCfits** | FITS I/O |
| **Boost** | Headers + libraries used by the project |
| **OpenMP** | Used for host parallel sections |
| **git-lfs** | For test measurement sets and FITS data |

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

Install the NVIDIA CUDA toolkit matching your driver. Ensure **`nvcc`** and CUDA libraries are on **`PATH`** / **`LD_LIBRARY_PATH`** (or use environment modules).

---

## Build and run

```bash
cd gpuvmem
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

**Outputs (outside `build/` only):**

| Path | Content |
|------|---------|
| `../bin/gpuvmem` | Main program (`src/main.cu`) |
| `../lib/libgpuvmem_*_static.a` | Internal static archives linked into the binary |

Optional: set **`-DPREFIX=/path`** at configure time to change where the **runtime** binary is written (see `CMakeLists.txt`).

**Antenna configurations** are read from the measurement set; no separate antenna file is required for that part of the pipeline.

---

## Install on the system

CMake generates install rules. With **Makefiles**, **`make install`** is equivalent to **`cmake --install .`** from the build directory.

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

| Option | Default | Role |
|--------|---------|------|
| `GPUVMEM_INSTALL_EXECUTABLE` | **ON** | Install `gpuvmem` to `CMAKE_INSTALL_BINDIR` (e.g. `prefix/bin`). |
| `GPUVMEM_INSTALL_STATIC_LIBS` | **OFF** | Install internal `.a` libraries to `CMAKE_INSTALL_LIBDIR`. |
| `GPUVMEM_INSTALL_HEADERS` | **OFF** | Install `include/` under `CMAKE_INSTALL_INCLUDEDIR/gpuvmem`. |

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

Prepare a **FITS model** with correct astrometry in the header (often the dirty image from CASA **tclean**).

Run from the repo (after build):

```bash
./bin/gpuvmem [options]
```

Help excerpt (see also **`--help`** on the binary):

```text
  -O --output_image [default: mod_out.fits]
      Output image name
  -e --eta [default: -1]
      Controls minimum image value in the entropy prior
  -T --threshold [default: 0]
      Threshold for spectral-index image (sigmas on I_nu_0)
  -p --path [default: mem/]
      Directory for FITS outputs (include trailing /)
  -G --gpus [default: 0]
      Comma-separated GPU indices
  -R --robust_parameter [default: 2]
      Robust weighting: -2 uniform, 2 natural, 0 tradeoff
  -X --blockSizeX [default: -1]   -Y --blockSizeY   -V --blockSizeV
      GPU block sizes (-1 = auto)
  -t --iterations [default: 500]
      Optimization iterations
  -g --gridding [default: 0]
      Gridded visibilities (CPU gridding; set thread count)
  -z --initial_values [default: NULL]
      Comma-separated initial values per image
  -Z --regularization_factors [default: NULL]
      Comma-separated regularization weights

  Flags: -v --verbose  -x --nopositivity  -a --apply-noise  -P --print-images
         -E --print-errors  -s --save_modelcolumn  -M --use-radius-mask

  Help: -h --help  -w --warranty  -c --copyright

  Mandatory:
  -i --input   Input MS path(s), comma-separated
  -o --output  Output MS path(s), comma-separated
  -m --model_input [default: mod_in_0.fits]
               FITS model with full astrometry header

  Optional:
  -n --noise  -N --noise_cut  -F --ref_frequency  -r --random_sampling  -f --output_file
```

---

## Extending the framework

The code is organized as an **RML-style pipeline**: build a **synthesizer** (data + coordinates), an **optimizer** (e.g. conjugate gradient variants, L-BFGS), an **objective function**, and attach **Fi terms** (χ² plus any regularizers you register). You can swap **gridding kernels**, **weighting schemes**, **line searchers**, and **step-size seeders**. Objects are created via **factories** or constructors. Each term is configured with weights (`-Z`), and image indices for where gradients are read and written — matching how you would extend any regularized likelihood problem, not only the original MEM setup.

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

- **Miguel Cárcamo** — The University of Manchester — miguel.carcamo@postgrad.manchester.ac.uk  
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

<a href="https://www.star-history.com/?repos=miguelcarcamov%2Fgpuvmem&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=miguelcarcamov/gpuvmem&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=miguelcarcamov/gpuvmem&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=miguelcarcamov/gpuvmem&type=date&legend=top-left" />
 </picture>
</a>
