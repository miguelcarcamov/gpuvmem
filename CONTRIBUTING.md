# Contributing to gpuvmem

Thanks for your interest in improving gpuvmem. This document explains how to report issues, propose changes, and what we expect from pull requests.

## Ways to contribute

- **Bug reports** — reproducible steps, environment, and logs help a lot.
- **Feature ideas** — open an issue to discuss before large refactors.
- **Code / docs / tests** — pull requests are welcome (see below).

## Before you start

1. **Search existing issues** on GitHub to avoid duplicates.
2. For **build or runtime problems**, include:
   - OS and version
   - CUDA toolkit version (`nvcc --version`)
   - CMake version (`cmake --version`)
   - GPU model (if relevant) and `nvidia-smi` output when useful
   - Full configure and build log (or the first error block)

## Development setup

```bash
git clone https://github.com/miguelcarcamov/gpuvmem.git
cd gpuvmem
git lfs install
git lfs pull   # test data (measurement sets, FITS) for ctest
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
ctest --output-on-failure   # optional; needs LFS data
```

Useful CMake flags (see `CMakeLists.txt` and README):

- `-DCUDA_ARCH=80` — override auto-detected GPU architecture
- `-DCMAKE_BUILD_TYPE=Debug` — with `-DMEMORY_DEBUG=ON` for CUDA device debugging
- `-DGPUVMEM_INSTALL_STATIC_LIBS=ON` — if you work on installable static libs

## Pull requests

- **Keep changes focused** — one logical fix or feature per PR is easier to review.
- **Match existing style** — naming, headers, and CMake patterns used in the repo.
- **Do not reformat unrelated code** unless the project agrees on a formatting pass.
- **Update docs** when you change user-visible behavior (README, options, install paths).

If your change is large, open an issue first to agree on direction.

## Bug report template

Use this structure in GitHub issues:

**Describe the bug**  
What went wrong?

**To reproduce**  
Minimal commands, dataset type (synthetic vs real), and options.

**Expected behavior**  
What you wanted to happen.

**Environment**  
OS, CUDA, CMake, gpuvmem commit or tag.

**Logs**  
Configure/build output or runtime error text.

## Feature request template

**Problem / motivation**  
What limitation or pain point does this address?

**Proposed solution**  
What would you like to see?

**Alternatives**  
Other approaches you considered.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (**GNU General Public License v3.0** — see `LICENSE.txt`).
