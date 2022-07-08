<h1 align="center">
   <img src="https://github.com/miguelcarcamov/gpuvmem/wiki/images/logos/logo2.png" height="400">
</h1>

# Papers and documentation

- Paper: <https://doi.org/10.1016/j.ascom.2017.11.003>
- Wiki: <https://github.com/miguelcarcamov/gpuvmem/wiki>

# Citing

If you use GPUVMEM for your research please do not forget to cite Cárcamo et al.

   ```tex
   @article{CARCAMO201816,
   title = "Multi-GPU maximum entropy image synthesis for radio astronomy",
   journal = "Astronomy and Computing",
   volume = "22",
   pages = "16 - 27",
   year = "2018",
   issn = "2213-1337",
   doi = "https://doi.org/10.1016/j.ascom.2017.11.003",
   url = "http://www.sciencedirect.com/science/article/pii/S2213133717300094",
   author = "M. Cárcamo and P.E. Román and S. Casassus and V. Moral and F.R. Rannou",
   keywords = "Maximum entropy, GPU, ALMA, Inverse problem, Radio interferometry, Image synthesis"
   }
   ```

# Installation

1. Install git-lfs

    a. `sudo apt-get install git-lfs`

2. Install casacore latest stable version v3.2.1

    a. `git clone --single-branch --branch v3.2.1 https://github.com/casacore/casacore.git`

    b. `sudo apt-get install -y build-essential cmake gfortran g++ libncurses5-dev libreadline-dev flex bison libblas-dev liblapacke-dev libcfitsio-dev wcslib-dev libhdf5-serial-dev libfftw3-dev python-numpy libboost-python-dev libpython2.7-dev`

    b. `cd casacore`

    c. `mkdir build`

    d. `cd build`

    e. `cmake -DUSE_FFTW3=ON -DUSE_OPENMP=ON -DUSE_HDF5=ON -DUSE_THREADS=ON ..`

    f. `make -j`

    g. `sudo make install`

3. Install Boost

    a. `sudo apt-get -y install libboost-all-dev`

4. Install cfitsio

    a. `sudo apt-get -y install libcfitsio-dev`

5. Download or clone gpuvmem.

6. To compile GPUVMEM you will need:

   - cfitsio - Usually the package is called `libcfitsio-dev`.
   - cmake >= 3.8
   - git-lfs - `git-lfs`
   - casacore >= v3.1.2 (<https://github.com/casacore/casacore> - branch v3.1.2. please make sure you have installed the github version, Ubuntu package doesn't work well since doesn't have the `put()` function).
   - CUDA 9, 9.1, 9.2, 10.0 and 11.0. Remember to add binaries and libraries to the **PATH** and **LD_LIBRARY_PATH** environment variables, respectively.
   - OpenMP

7. To run the cmake tests you need to run `git lfs install` if not installled and then `git-lfs pull` to pull the measurement sets and model input FITS images.

# Installation using Docker

   ```bash
   docker pull ghcr.io/miguelcarcamov/gpuvmem:latest
   ```

# Compiling

   ```bash
   cd gpuvmem
   mkdir build
   cd build
   cmake ..
   make -j
   ```

## Now antenna configurations are read directly from the MS file

# Usage

Create your FITS model input astrometry data on the header, typically we use the resulting dirty image from CASA's tclean.

# Use GPUVMEM

Usage: `./bin/gpuvmem [options]`

   ```text
      -O --output_image [default: mod_out.fits]
          Name of the output visibility file/s (separated by a comma)
      -e --eta [default: -1]
          Variable that controls the minimum image value in the entropy prior
      -T --threshold [default: 0]
          Threshold to calculate the spectral index image above a certain number of
          sigmas in I_nu_0
      -p --path [default: mem/]
          Path to save FITS images. With last trail / included. (Example ./../mem/)
      -G --gpus [default: 0]
          Index of the GPU/s you are going to use separated by a comma
      -R --robust_parameter [default: 2]
          Robust weighting parameter when gridding. -2.0 for uniform weighting, 2.0
          for natural weighting and 0.0 for a tradeoff between these two.
      -X --blockSizeX [default: -1]
          GPU block X Size for image/Fourier plane (Needs to be pow of 2)
      -Y --blockSizeY [default: -1]
          GPU block Y Size for image/Fourier plane (Needs to be pow of 2)
      -V --blockSizeV [default: -1]
          GPU block V Size for visibilities (Needs to be pow of 2)
      -t --iterations [default: 500]
          Number of iterations for optimization
      -g --gridding [default: 0]
          Use gridded visibilities. This is done in CPU (Need to select the CPU thre
          ads that will grid the input visibilities)
      -z --initial_values [default: NULL]
          Initial values for image/s
      -Z --regularization_factors [default: NULL]
          Regularization factors for each regularization (separated by a comma)

    Flags:
      -v --verbose [default: (unset)]
          Shows information through all the execution
      -x --nopositivity [default: (unset)]
          Runs gpuvmem with no positivity restrictions on the images
      -a --apply-noise [default: (unset)]
          Applies random gaussian noise to visibilities
      -P --print-images [default: (unset)]
          Prints images per iteration
      -E --print-errors [default: (unset)]
          Prints final error maps
      -s --save_modelcolumn [default: (unset)]
          Saves the model visibilities on the model column of the input MS
      -M --use-radius-mask [default: (unset)]
          Use a mask based on a radius instead of the noise estimation

    Help:
      -h --help [default: (unset)]
          Shows this help
      -w --warranty [default: (unset)]
          Shows warranty details
      -c --copyright [default: (unset)]
          Shows copyright conditions

    Mandatory:
      -i --input [default: NULL]
          Name of the input visibility file/s (separated by a comma)
      -o --output [default: NULL]
          Name of the output visibility file/s (separated by a comma)
      -m --model_input [default: mod_in_0.fits]
          FITS file including a complete header for astrometry

    Optional:
      -n --noise [default: -1]
          Noise factor parameter
      -N --noise_cut [default: 10]
          Noise-cut Parameter
      -F --ref_frequency [default: -1]
          Reference frequency in Hz (if alpha is not zero). It will be calculated fr
          om the measurement set if not set
      -r --random_sampling [default: 1]
          Percentage of data used when random sampling
      -f --output_file [default: NULL]
          Output file where final objective function values are saved
   ```

# Framework usage

- The normal flow of the program starts by creating a synthesizer, creating an optimizer, creating an objective function, and adding the terms to the objective function. It is also possible to add a convolution kernel for gridding and a weighting scheme.

- Objects can be created by their respective factory or by their constructors.

- The configuration of each objective function term is parameterized by the penalty factor (-Z), the index of the image from where data will be calculated and the index of the image where results are going to be applied.

# TO RESTORE YOUR IMAGE PLEASE SEE CARCAMO ET AL. 2018 FOR MORE INFORMATION

- This will return a restored image: A convolution of the model image with the CLEAN BEAM + residuals (JY/BEAM)
- Residuals (JY/BEAM)
- The script file is on the scripts folder and it is named restore_continuum_ms_SC.py

Restoring usage:

   ```bash
   casa --nologger -c restore_continuum_ms_SC.py residual_folder.ms mem_model.fits restored_output 2.0
   ```

The last parameter, is the robust parameter that you want to use to clean the residuals.

# CONTRIBUTORS

- Miguel Cárcamo - The University of Manchester - miguel.carcamo@postgrad.manchester.ac.uk
- Nicolás Muñoz - Universidad de Santiago de Chile
- Fernando Rannou - Universidad de Santiago de Chile
- Pablo Román - Universidad de Santiago de Chile
- Simón Casassus - Universidad de Chile
- Axel Osses - Universidad de Chile
- Victor Moral - Universidad de Chile

# CONTRIBUTION AND BUG REPORTS

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1\. Go to '...'
2\. Click on '....'
3\. Scroll down to '....'
4\. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Desktop (please complete the following information):**

- OS: [e.g. Ubuntu 16.04]
- CUDA version [e.g. 9]
- gpuvmem Version [e.g. 22]

**Additional context**
Add any other context about the problem here.

# FEATURE REQUEST

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
