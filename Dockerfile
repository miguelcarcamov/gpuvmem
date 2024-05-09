FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV PATH /usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Update
RUN apt-get update

# Install git
RUN apt-get install -y git

RUN apt-get install -y python3-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-wheel && \
    apt-get install -y python3-setuptools

# Install dependencies
RUN apt-get install -y build-essential && \
    apt-get install -y cmake && \
    apt-get install -y gfortran && \
    apt-get install -y g++ && \
    apt-get install -y libncurses5-dev && \
    apt-get install -y libreadline-dev && \
    apt-get install -y flex && \
    apt-get install -y bison  && \
    apt-get install -y libblas-dev  && \
    apt-get install -y liblapacke-dev  && \
    apt-get install -y libcfitsio-dev  && \
    apt-get install -y wcslib-dev  && \
    apt-get install -y libfftw3-dev && \
    apt-get install -y libhdf5-serial-dev && \
    apt-get install -y python3-numpy && \
    apt-get install -y libboost-all-dev && \
    apt-get install -y libgsl-dev

# Install casacore
RUN git clone --single-branch --branch v3.5.0 https://github.com/casacore/casacore.git && \
    cd casacore && \
    mkdir build && \
    cd build && \
    cmake -DUSE_FFTW3=ON -DUSE_OPENMP=ON -DUSE_HDF5=ON -DBUILD_PYTHON3=ON -DUSE_THREADS=ON -DBUILD_PYTHON=OFF .. && \
    make -j2 && \
    make install

# Install CUDA samples
RUN cd /usr/local/cuda && \
    git clone --single-branch --branch v11.6 https://github.com/NVIDIA/cuda-samples.git samples && \
    cd samples && \
    mv Common common && \
    mv Samples samples && \
    cd common && \
    mkdir inc && \
    mv *.h inc/

RUN echo "Hello there from gpuvmem base image"
LABEL org.opencontainers.image.source="https://github.com/miguelcarcamov/gpuvmem"
