FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
    DEBIAN_FRONTEND=noninteractive

# libccfits-dev is in universe on Ubuntu 22.04
RUN sed -i '/^Components:/ s/ main$/ main universe/' /etc/apt/sources.list \
    && sed -i '/^Components:/ s/ main restricted$/ main restricted universe/' /etc/apt/sources.list || true

# Install system dependencies and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        build-essential \
        cmake \
        gfortran \
        g++ \
        libncurses5-dev \
        libreadline-dev \
        flex \
        bison \
        libblas-dev \
        liblapacke-dev \
        libcfitsio-dev \
        libccfits-dev \
        wcslib-dev \
        libfftw3-dev \
        libhdf5-serial-dev \
        python3-numpy \
        libboost-all-dev \
        libgsl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install casacore
RUN git clone --single-branch --branch v3.5.0 --depth 1 \
        https://github.com/casacore/casacore.git /tmp/casacore && \
    cd /tmp/casacore && \
    mkdir build && cd build && \
    cmake -DUSE_FFTW3=ON \
          -DUSE_OPENMP=ON \
          -DUSE_HDF5=ON \
          -DBUILD_PYTHON3=ON \
          -DUSE_THREADS=ON \
          -DBUILD_PYTHON=OFF \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/casacore

# Install CUDA samples (required for common headers)
RUN cd /usr/local/cuda && \
    git clone --single-branch --branch v12.4 --depth 1 \
        https://github.com/NVIDIA/cuda-samples.git samples && \
    cd samples && \
    mv Common common && \
    mv Samples samples && \
    cd common && \
    mkdir -p inc && \
    mv *.h inc/ 2>/dev/null || true

# Verify CUDA installation and CCfits (gpuvmem requires find_package(CCfits))
RUN nvcc --version && \
    test -f /usr/local/cuda/bin/nvcc && \
    test -d /usr/local/cuda/lib64 && \
    test -f /usr/include/CCfits/FITS.h && \
    test -e /usr/lib/x86_64-linux-gnu/libCCfits.so && \
    test -f /usr/local/cuda/samples/common/inc/helper_cuda.h && \
    echo "CUDA + CCfits + cuda-samples headers verified"

LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/gpuvmem" \
      org.opencontainers.image.description="Base image for gpuvmem with CUDA 12.4.1, casacore, and CCfits"
