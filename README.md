<p align="center">
  <img src="https://github.com/miguelcarcamov/gpuvmem/wiki/images/logos/logo2.png" height="400">
</p>

# Papers and documentation

-   <https://doi.org/10.1016/j.ascom.2017.11.003>

# Citing

If you use GPUVMEM for your research please do not forget to cite Cárcamo et al.

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

# Installation

1.  Install git-lfs

  a. `sudo apt-get install git-lfs`

2.  Install casacore latest stable version v3.1.2

  a. `git clone --single-branch --branch v3.1.2 https://github.com/casacore/casacore.git`

  b. `sudo apt-get install -y build-essential cmake gfortran g++ libncurses5-dev libreadline-dev flex bison libblas-dev liblapacke-dev libcfitsio-dev wcslib-dev libhdf5-serial-dev libfftw3-dev python-numpy libboost-python-dev libpython2.7-dev`

  b. `cd casacore`

  c. `mkdir build`

  d. `cd build`

  e. `cmake -DUSE_FFTW3=ON -DUSE_OPENMP=ON -DUSE_HDF5=ON -DUSE_THREADS=ON ..`

  f. `make -j`

  g. `sudo make install`

3.  Install cfitsio

  a. `sudo apt-get -y install libcfitsio-dev`

3.  Download or clone gpuvmem.

4.  To compile GPUVMEM you will need:

-   cfitsio - Usually the package is called `libcfitsio-dev`.
-   cmake >= 3.8
-   git-lfs - `git-lfs`
-   casacore >= v3.1.2 (<https://github.com/casacore/casacore> - branch v3.1.2. please make sure you have installed the github version, Ubuntu package doesn't work well since doesn't have the `put()` function).
-   CUDA 9, 9.1, 9.2, 10.0 or 10.1 **Update 2**. Remember to add binaries and libraries to the **PATH** and **LD_LIBRARY_PATH** environment variables, respectively.
-   OpenMP

# Installation using a Singularity container

We have created a recipe to use Singularity without worrying about installing CUDA and installing libraries. The container has gpuvmem compiled and all you need to work. Additionally, you can edit the recipe and add applications that you are interested in. To use the Singularity container, follow the next steps:

```bash
# Install Singularity, in this case we will follow the Linux instructions, but to install it on MAC see: https://singularity.lbl.gov/install-mac

sudo apt install squashfs-tools
VERSION=2.5.2
wget https://github.com/singularityware/singularity/releases/download/$VERSION/singularity-$VERSION.tar.gz
tar xvf singularity-$VERSION.tar.gz
cd singularity-$VERSION
./configure --prefix=/usr/local
make
sudo make install

#Now we can use Singularity to shell into the container
mkdir container # Make a directory in your workspace
singularity shell --nv shub://miguelcarcamov/container_docker:casacore.gpuvmem.9.2 #Use this command for CUDA 9.2
# singularity shell --nv shub://miguelcarcamov/container_docker:casacore.gpuvmem.10.0 #Use this command for CUDA 10.0
```

The gpuvmem binary should be in `/gpuvmem/bin`.

# Compiling

```bash
cd gpuvmem
mkdir build
cd build
cmake ..
make -j
```

# t_telescope references the following telescopes:

1.  CBI2
2.  ALMA
3.  ATCA
4.  VLA
5.  SZA
6.  CBI
7.  EHT

# Usage

Create your canvas or mod_in_0.fits with the image data in the header, typically we use a FITS CASA CLEAN dirty image of the desired object.

Create your input.dat file with the following data and change the parameters as you need. For example if you want to reconstruct a single ALMA dataset file:

    noise_cut	100.5
    ftol		1.0e-12
    random_probability 0.0
    t_telescope        2

However, if you want to reconstruct a multi-antenna dataset. For example, CBI and CBI2.

    noise_cut	100.5
    ftol		1.0e-12
    random_probability 0.0
    t_telescope        6,1

# Use GPUVMEM:

Example: `./bin/gpuvmem options [ arguments ...]`

    -h  --help             Shows this
    -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2)
    -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2)
    -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2)
    -i  --input            The name of the input file/s of visibilities(MS) separated by a comma.
    -o  --output           The name of the output file/s of residuals visibilities(MS) separated by a comma.
    -O  --output-image     The name of the output image FITS file
    -I  --inputdat         The name of the input file of parameters
    -m  --model_input      FITS file including a complete header for astrometry
    -n  --noise            Noise Parameter in Jy (Optional)
    -N  --noise-cut        Noise-cut Parameter (Optional)
    -r  --randoms          Percentage of data used when random sampling (Default = 1.0, optional)
    -e  --eta              Variable that controls the minimum image value (Default eta = -1.0)
    -p  --path             MEM path to save FITS images. With last / included. (Example ./../mem/)
    -f  --file             Output file where final objective function values are saved (Optional)
    -M  --multigpu         Number of GPUs to use multiGPU image synthesis (Default OFF => 0)
    -s  --select           If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)
    -t  --iterations       Number of iterations for optimization (Default = 500)
    -g  --gridding         Use count in cell gridding to decrease the number of visibilities. This is done in CPU (Need to select the CPU threads that will grid the input visibilities)
    -F  --nu_0             Reference frequency in Hz (if alpha is not zero)
    -z  --initial_values   Initial values for image/s
    -Z  --penalizators     penalizators for prior terms
    -R  --robust-parameter Robust weighting parameter when gridding. -2.0 for uniform weighting, 2.0 for natural weighting and 0.0 for a tradeoff between these two. (Default R = 2.0).\n
    -T  --threshold        Threshold to calculate the spectral index image from above a certain number of sigmas in I_nu_0
    -c  --copyright        Shows copyright conditions
    -w  --warranty         Shows no warranty details
    --savemodel-input      Saves the model visibilities on the model column of the input MS
    --nopositivity         Run gpuvmem using chi2 with no posititivy restriction
    --apply-noise          Apply random gaussian noise to visibilities
    --clipping             Clips the image to positive values
    --print-images         Prints images per iteration
    --print-errors         Prints final error images
    --verbose              Shows information through all the execution

# Framework usage

The normal flow of the program starts by creating a synthesizer, creating an optimizer, creating an objective function, and adding your terms to the objective function

All the objects must be created by their respective factory

The number of available images is determined by the -z command

Gridding can be applied both as a filter and as an input parameter

All the specializations of objects are listed in the enums

All filters can only be applied before using Synthesizer->setDevice()

The configuration of a Fi has as parameters, the index of its penalty factor (-Z), the index of the image from where the data will be calculated, and finally the index of the image where the results of the calculations will be applied.

# IMPORTANT

Remember to create the mem folder to save the FITS images

# Fixes

We have fixed the Makefile and now you can compile gpuvmem using the new version of casacore.

# TO RESTORE YOUR IMAGE PLEASE SEE CARCAMO ET AL. 2018 FOR MORE INFORMATION

-   This will return a restored image: A convolution of the model image with the CLEAN BEAM + residuals (JY/BEAM)
-   Residuals (JY/BEAM)

Restoring usage:

```bash
casapy --log2term --nogui -c restore_continuum_ms.py residual_folder.ms mem_model.fits restored_output
```

# CONTRIBUTORS

-   Miguel Cárcamo - The University of Manchester - miguel.carcamo@postgrad.manchester.ac.uk
-   Nicolás Muñoz - Universidad de Santiago de Chile
-   Fernando Rannou - Universidad de Santiago de Chile
-   Pablo Román - Universidad de Santiago de Chile
-   Simón Casassus - Universidad de Chile
-   Axel Osses - Universidad de Chile
-   Victor Moral - Universidad de Chile

# CONTRIBUTION AND BUG REPORTS

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Desktop (please complete the following information):**

-   OS: [e.g. Ubuntu 16.04]
-   CUDA version [e.g. 9]
-   gpuvmem Version [e.g. 22]

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
