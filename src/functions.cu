/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */
#include "functions.cuh"

namespace cg = cooperative_groups;

extern long M, N;
extern int iterations, iter, image_count, status_mod_in, flag_opt, num_gpus,
    multigpu, firstgpu, reg_term;

extern cufftHandle plan1GPU;
extern cufftComplex *device_V, *device_fg_image, *device_I_nu;
extern float* device_I;

extern float *device_dphi, *device_S, *device_dchi2_total, *device_dS,
    *device_noise_image;
extern float noise_jypix, noise_cut, MINPIX, minpix, random_probability, eta;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern double beam_bmaj, beam_bmin, beam_bpa;
extern float *initial_values, *penalizators, robust_param;
extern double ra, dec, DELTAX, DELTAY, deltau, deltav, crpix1, crpix2;
extern float threshold;
extern float nu_0;
extern int nPenalizators, nMeasurementSets, max_number_vis;

extern char *mempath, *out_image;

extern fitsfile* mod_in;

extern MSDataset* datasets;

extern varsPerGPU* vars_gpu;

extern Vars variables;

extern bool verbose_flag, nopositivity, apply_noise, print_images, print_errors,
    save_model_input, radius_mask, modify_weights;

extern Flags flags;

typedef float (*FnPtr)(float, float, float, float);

__device__ FnPtr beam_maps[2] = {AiryDiskBeam, GaussianBeam};

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T*() {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }

  __device__ inline operator const T*() const {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double*() {
    extern __shared__ double __smem_d[];
    return (double*)__smem_d;
  }

  __device__ inline operator const double*() const {
    extern __shared__ double __smem_d[];
    return (double*)__smem_d;
  }
};

__host__ void goToError() {
  if (num_gpus > 1) {
    for (int i = firstgpu + 1; i < firstgpu + num_gpus; i++) {
      cudaSetDevice(firstgpu);
      cudaDeviceDisablePeerAccess(i);
      cudaSetDevice(i);
      cudaDeviceDisablePeerAccess(firstgpu);
    }

    for (int i = 0; i < num_gpus; i++) {
      cudaSetDevice((i % num_gpus) + firstgpu);
      cudaDeviceReset();
    }
  }

  printf("An error has ocurred, exiting\n");
  exit(0);
}

__host__ float median(std::vector<float> v) {
  size_t elements = v.size();
  size_t n = elements / 2;

  if (elements == 1) {
    return v[0];
  } else {
    std::nth_element(v.begin(), v.begin() + n, v.end());
    int vn = v[n];
    if (v.size() % 2 == 1) {
      return vn;
    } else {
      std::nth_element(v.begin(), v.begin() + n - 1, v.end());
      return 0.5 * (vn + v[n - 1]);
    }
  }
}

__host__ int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__host__ unsigned int NearestPowerOf2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

__host__ bool isPow2(unsigned int x) {
  return ((x & (x - 1)) == 0);
}

__host__ void print_help() {
  flags.PrintHelp();
}

__host__ char* strip(const char* string, const char* chars) {
  char* newstr = (char*)malloc(strlen(string) + 1);
  int counter = 0;

  for (; *string; string++) {
    if (!strchr(chars, *string)) {
      newstr[counter] = *string;
      ++counter;
    }
  }

  newstr[counter] = 0;
  return newstr;
}

__host__ Vars getOptions(int argc, char** argv) {
  Vars variables;
  bool help, copyright, warranty;

  flags.Var(variables.input, 'i', "input", std::string("NULL"),
            "Name of the input visibility file/s (separated by a comma)",
            "Mandatory");
  flags.Var(variables.output, 'o', "output", std::string("NULL"),
            "Name of the output visibility file/s (separated by a comma)",
            "Mandatory");
  flags.Var(variables.output_image, 'O', "output_image",
            std::string("mod_out.fits"),
            "Name of the output visibility file/s (separated by a comma)");
  flags.Var(variables.modin, 'm', "model_input", std::string("mod_in_0.fits"),
            "FITS file including a complete header for astrometry",
            "Mandatory");
  flags.Var(variables.noise, 'n', "noise", -1.0f, "Noise factor parameter",
            "Optional");
  flags.Var(
      variables.eta, 'e', "eta", -1.0f,
      "Variable that controls the minimum image value in the entropy prior");
  flags.Var(variables.noise_cut, 'N', "noise_cut", 10.0f, "Noise-cut Parameter",
            "Optional");
  flags.Var(variables.nu_0, 'F', "ref_frequency", -1.0f,
            "Reference frequency in Hz (if alpha is not zero). It will be "
            "calculated from the measurement set if not set",
            "Optional");
  flags.Var(variables.threshold, 'T', "threshold", 0.0f,
            "Threshold to calculate the spectral index image above a certain "
            "number of sigmas in I_nu_0");
  flags.Var(variables.path, 'p', "path", std::string("mem/"),
            "Path to save FITS images. With last trail / included. (Example "
            "./../mem/)");
  flags.Var(variables.gpus, 'G', "gpus", std::string("0"),
            "Index of the GPU/s you are going to use separated by a comma");
  flags.Var(variables.randoms, 'r', "random_sampling", 1.0f,
            "Percentage of data used when random sampling", "Optional");
  flags.Var(
      variables.robust_param, 'R', "robust_parameter", 2.0f,
      "Robust weighting parameter when gridding. -2.0 for uniform weighting, "
      "2.0 for natural weighting and 0.0 for a tradeoff between these two.");
  flags.Var(variables.ofile, 'f', "output_file", std::string("NULL"),
            "Output file where final objective function values are saved",
            "Optional");
  flags.Var(variables.blockSizeX, 'X', "blockSizeX", int32_t(-1),
            "GPU block X Size for image/Fourier plane (Needs to be pow of 2)");
  flags.Var(variables.blockSizeY, 'Y', "blockSizeY", int32_t(-1),
            "GPU block Y Size for image/Fourier plane (Needs to be pow of 2)");
  flags.Var(variables.blockSizeV, 'V', "blockSizeV", int32_t(-1),
            "GPU block V Size for visibilities (Needs to be pow of 2)");
  flags.Var(variables.it_max, 't', "iterations", int32_t(500),
            "Number of iterations for optimization");
  flags.Var(variables.gridding, 'g', "gridding", int32_t(0),
            "Use gridded visibilities. This is done in CPU (Need to select the "
            "CPU threads that will grid the input visibilities)");
  flags.Var(variables.initial_values, 'z', "initial_values",
            std::string("NULL"), "Initial values for image/s");
  flags.Var(
      variables.penalization_factors, 'Z', "regularization_factors",
      std::string("NULL"),
      "Regularization factors for each regularization (separated by a comma)");
  flags.Var(variables.user_mask, 'U', "user-mask", std::string("NULL"),
            "Use a user created mask instead of using the noise mask");
  flags.Bool(verbose_flag, 'v', "verbose",
             "Shows information through all the execution", "Flags");
  flags.Bool(nopositivity, 'x', "nopositivity",
             "Runs gpuvmem with no positivity restrictions on the images",
             "Flags");
  flags.Bool(apply_noise, 'a', "apply-noise",
             "Applies random gaussian noise to visibilities", "Flags");
  flags.Bool(print_images, 'P', "print-images", "Prints images per iteration",
             "Flags");
  flags.Bool(print_errors, 'E', "print-errors", "Prints final error maps",
             "Flags");
  flags.Bool(save_model_input, 's', "save_modelcolumn",
             "Saves the model visibilities on the model column of the input MS",
             "Flags");
  flags.Bool(radius_mask, 'M', "use-radius-mask",
             "Use a mask based on a radius instead of the noise estimation",
             "Flags");
  flags.Bool(modify_weights, 'W', "modify-weights",
             "Modify Measurement Set WEIGHT column with gpuvmem weights",
             "Flags");
  flags.Bool(help, 'h', "help", "Shows this help", "Help");
  flags.Bool(warranty, 'w', "warranty", "Shows warranty details", "Help");
  flags.Bool(copyright, 'c', "copyright", "Shows copyright conditions", "Help");

  if (!flags.Parse(argc, argv)) {
    print_help();
    exit(EXIT_SUCCESS);
  } else if (help) {
    print_help();
    exit(EXIT_SUCCESS);
  }

  if (warranty) {
    print_help();
    exit(EXIT_SUCCESS);
  }

  if (copyright) {
    print_help();
    exit(EXIT_SUCCESS);
  }

  if (variables.randoms > 1.0 || variables.randoms < 0.0) {
    print_help();
    exit(EXIT_FAILURE);
  }

  if (variables.gridding < 0) {
    print_help();
    exit(EXIT_FAILURE);
  }

  if (variables.user_mask != "NULL") {
    variables.noise_cut = 1.0f;
  }

  return variables;
}

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

__host__ void getNumBlocksAndThreads(int n,
                                     int maxBlocks,
                                     int maxThreads,
                                     int& blocks,
                                     int& threads,
                                     bool reduction) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  if (reduction) {
    threads = (n < maxThreads * 2) ? NearestPowerOf2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  } else {
    threads = (n < maxThreads) ? NearestPowerOf2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  }
  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }

  if (reduction) {
    blocks = MIN(maxBlocks, blocks);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! Uses Neumaier "improved Kahan–Babuška algorithm" for an accurate sum of
//! large arrays. http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <class T>
__host__ T reduceCPU(T* data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T t = sum + data[i];
    if (fabs(sum) >= fabs(data[i]))
      c += (sum - t) + data[i];
    else
      c += (data[i] - t) + sum;
    sum = t;
  }
  return sum;
}

template <class T, int blockSize, bool nIsPow2>
__global__ void reduceSumKernel(T* g_idata, T* g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T* sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  T mySum = (T)0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64)
      mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0)
    g_odata[blockIdx.x] = mySum;
}

template <int blockSize, bool nIsPow2>
__global__ void reduceMinKernel(float* g_idata,
                                float* g_odata,
                                unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  float myMin = FLT_MAX;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      myMin = fminf(myMin, g_idata[i]);
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        myMin = fminf(myMin, g_idata[i + blockSize]);
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      myMin = fminf(myMin, g_idata[i]);
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMin;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = myMin = fminf(myMin, sdata[tid + 256]);
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = myMin = fminf(myMin, sdata[tid + 128]);
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = myMin = fminf(myMin, sdata[tid + 64]);
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64)
      myMin = fminf(myMin, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      myMin = fminf(myMin, tile32.shfl_down(myMin, offset));
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0)
    g_odata[blockIdx.x] = myMin;
}

template <int blockSize, bool nIsPow2>
__global__ void reduceMaxKernel(float* g_idata,
                                float* g_odata,
                                unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  float myMax = FLT_MIN;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      myMax = fmaxf(myMax, g_idata[i]);
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        myMax = fmaxf(myMax, g_idata[i + blockSize]);
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      myMax = fmaxf(myMax, g_idata[i]);
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMax;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 256]);
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 128]);
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = myMax = fmaxf(myMax, sdata[tid + 64]);
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64)
      myMax = fmaxf(myMax, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      myMax = fmaxf(myMax, tile32.shfl_down(myMax, offset));
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0)
    g_odata[blockIdx.x] = myMax;
}

template <class T>
__host__ T deviceReduce(T* in, long N, int input_threads) {
  T sum = (T)0;
  T* d_odata = NULL;
  int maxThreads = input_threads;
  int maxBlocks = iDivUp(N, maxThreads);

  int threads = 0;
  int blocks = 0;

  getNumBlocksAndThreads(N, maxBlocks, maxThreads, blocks, threads, true);

  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  T* h_odata = (T*)malloc(blocks * sizeof(T));
  checkCudaErrors(cudaMalloc((void**)&d_odata, blocks * sizeof(T)));

  if (isPow2(N)) {
    switch (threads) {
      case 512:
        reduceSumKernel<T, 512, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 256:
        reduceSumKernel<T, 256, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 128:
        reduceSumKernel<T, 128, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 64:
        reduceSumKernel<T, 64, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 32:
        reduceSumKernel<T, 32, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 16:
        reduceSumKernel<T, 16, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 8:
        reduceSumKernel<T, 8, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 4:
        reduceSumKernel<T, 4, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 2:
        reduceSumKernel<T, 2, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 1:
        reduceSumKernel<T, 1, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        reduceSumKernel<T, 512, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 256:
        reduceSumKernel<T, 256, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 128:
        reduceSumKernel<T, 128, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 64:
        reduceSumKernel<T, 64, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 32:
        reduceSumKernel<T, 32, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 16:
        reduceSumKernel<T, 16, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 8:
        reduceSumKernel<T, 8, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 4:
        reduceSumKernel<T, 4, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 2:
        reduceSumKernel<T, 2, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 1:
        reduceSumKernel<T, 1, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(
      cudaMemcpy(h_odata, d_odata, blocks * sizeof(T), cudaMemcpyDeviceToHost));

  for (int i = 0; i < blocks; i++) {
    sum += h_odata[i];
  }

  cudaFree(d_odata);
  free(h_odata);
  return sum;
}

__host__ float deviceMaxReduce(float* in, long N, int input_threads) {
  float max = FLT_MIN;
  float* d_odata = NULL;
  int maxThreads = input_threads;
  int maxBlocks = iDivUp(N, maxThreads);

  int threads = 0;
  int blocks = 0;

  getNumBlocksAndThreads(N, maxBlocks, maxThreads, blocks, threads, true);

  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  float* h_odata = (float*)malloc(blocks * sizeof(float));
  checkCudaErrors(cudaMalloc((void**)&d_odata, blocks * sizeof(float)));

  if (isPow2(N)) {
    switch (threads) {
      case 512:
        reduceMaxKernel<512, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 256:
        reduceMaxKernel<256, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 128:
        reduceMaxKernel<128, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 64:
        reduceMaxKernel<64, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 32:
        reduceMaxKernel<32, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 16:
        reduceMaxKernel<16, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 8:
        reduceMaxKernel<8, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 4:
        reduceMaxKernel<4, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 2:
        reduceMaxKernel<2, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 1:
        reduceMaxKernel<1, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        reduceMaxKernel<512, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 256:
        reduceMaxKernel<256, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 128:
        reduceMaxKernel<128, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 64:
        reduceMaxKernel<64, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 32:
        reduceMaxKernel<32, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 16:
        reduceMaxKernel<16, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 8:
        reduceMaxKernel<8, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 4:
        reduceMaxKernel<4, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 2:
        reduceMaxKernel<2, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 1:
        reduceMaxKernel<1, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(h_odata, d_odata, blocks * sizeof(float),
                             cudaMemcpyDeviceToHost));

  for (int i = 0; i < blocks; i++) {
    max = std::max(max, h_odata[i]);
  }

  cudaFree(d_odata);
  free(h_odata);
  return max;
}

__host__ float deviceMinReduce(float* in, long N, int input_threads) {
  float min = FLT_MAX;
  float* d_odata = NULL;
  int maxThreads = input_threads;
  int maxBlocks = iDivUp(N, maxThreads);

  int threads = 0;
  int blocks = 0;

  getNumBlocksAndThreads(N, maxBlocks, maxThreads, blocks, threads, true);

  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  float* h_odata = (float*)malloc(blocks * sizeof(float));
  checkCudaErrors(cudaMalloc((void**)&d_odata, blocks * sizeof(float)));

  if (isPow2(N)) {
    switch (threads) {
      case 512:
        reduceMinKernel<512, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 256:
        reduceMinKernel<256, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 128:
        reduceMinKernel<128, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 64:
        reduceMinKernel<64, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 32:
        reduceMinKernel<32, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 16:
        reduceMinKernel<16, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 8:
        reduceMinKernel<8, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 4:
        reduceMinKernel<4, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 2:
        reduceMinKernel<2, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 1:
        reduceMinKernel<1, true>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        reduceMinKernel<512, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 256:
        reduceMinKernel<256, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 128:
        reduceMinKernel<128, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 64:
        reduceMinKernel<64, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 32:
        reduceMinKernel<32, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 16:
        reduceMinKernel<16, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 8:
        reduceMinKernel<8, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 4:
        reduceMinKernel<4, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 2:
        reduceMinKernel<2, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;

      case 1:
        reduceMinKernel<1, false>
            <<<dimGrid, dimBlock, smemSize>>>(in, d_odata, N);
        break;
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(h_odata, d_odata, blocks * sizeof(float),
                             cudaMemcpyDeviceToHost));

  for (int i = 0; i < blocks; i++) {
    min = std::min(min, h_odata[i]);
  }

  cudaFree(d_odata);
  free(h_odata);
  return min;
}

__global__ void fftshift_2D(cufftComplex* data, int N1, int N2) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < N1 && j < N2) {
    float a = 1 - 2 * ((i + j) & 1);

    data[N2 * i + j].x *= a;
    data[N2 * i + j].y *= a;
  }
}

/*
   This padding assumes that your input data has already a size power of 2.
    - new_image must be initialized with zeros
 */
template <class T>
__global__ void paddingKernel(T* dest,
                              T* src,
                              int fft_M,
                              int fft_N,
                              int m,
                              int n,
                              int centerKernel_x,
                              int centerKernel_y) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < m && j < n) {
    int ky = i - centerKernel_y;
    int kx = j - centerKernel_x;

    if (ky < 0) {
      ky += fft_M;
    }

    if (kx < 0) {
      kx += fft_N;
    }

    dest[fft_N * ky + kx] = src[n * i + j];
  }
}

template <class T>
__global__ void paddingData(T* dest,
                            T* src,
                            int fft_M,
                            int fft_N,
                            int M,
                            int N,
                            int m,
                            int n,
                            int centerKernel_x,
                            int centerKernel_y) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int borderM = M + centerKernel_y;
  const int borderN = N + centerKernel_x;

  if (i < fft_M && j < fft_N) {
    int dy, dx;

    if (i < M)
      dy = i;

    if (j < N)
      dx = j;

    if (i >= M && i < borderM)
      dy = M - 1;

    if (j >= N && j < borderN)
      dx = N - 1;

    if (i >= borderM)
      dy = 0;

    if (j >= borderN)
      dx = 0;

    dest[fft_N * i + j] = src[N * dy + dx];
  }
}

template <class T>
__global__ void depaddingData(T* dest,
                              T* src,
                              int fft_M,
                              int fft_N,
                              int M,
                              int N,
                              int centerKernel_x,
                              int centerKernel_y) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;

  const int offset_x = (fft_M - M - centerKernel_x) / 2;
  const int offset_y = (fft_N - N - centerKernel_y) / 2;
  const int border_x = offset_x + centerKernel_x;
  const int border_y = offset_y + centerKernel_y;
  const int x = j + border_x;
  const int y = i + border_y;

  if (y < N && x < M) {
    dest[N * i + j] = src[fft_N * y + x];
  }
}

template <class TD, class T>
__host__ TD* convolutionComplexRealFFT(TD* data,
                                       T* kernel,
                                       int M,
                                       int N,
                                       int m,
                                       int n,
                                       bool isDataOnFourier,
                                       bool isDataOnDevice,
                                       bool isKernelOnDevice) {
  T* kernel_device;
  TD* kernel_complex_device;
  TD* data_device;
  TD* padded_kernel_complex;
  TD* padded_data_complex;
  TD* data_spectrum_device;
  TD* kernel_spectrum_device;
  TD* result_host;

  cufftHandle fftPlan;
  int fftfwd, fftinv;
  if (isDataOnFourier) {
    fftfwd = CUFFT_INVERSE;
    fftinv = CUFFT_FORWARD;

  } else {
    fftfwd = CUFFT_FORWARD;
    fftinv = CUFFT_INVERSE;
  }

  if (!isKernelOnDevice) {
    checkCudaErrors(cudaMalloc((void**)&kernel_device, sizeof(T) * M * N));
    checkCudaErrors(cudaMemcpy(kernel_device, kernel, sizeof(T) * M * N,
                               cudaMemcpyHostToDevice));
  } else {
    kernel_device = kernel;
  }

  if (!isDataOnDevice) {
    checkCudaErrors(cudaMalloc((void**)&data_device, sizeof(TD) * M * N));
    checkCudaErrors(cudaMemcpy(data_device, data, sizeof(TD) * M * N,
                               cudaMemcpyHostToDevice));
  } else {
    data_device = data;
  }

  const int ckernel_x = ceil(m / 2);
  const int ckernel_y = ceil(n / 2);

  const int padding_M = NearestPowerOf2(M + m - 1);
  const int padding_N = NearestPowerOf2(N + n - 1);

  // Copy the float kernel to a complex array
  checkCudaErrors(
      cudaMalloc((void**)&kernel_complex_device, sizeof(TD) * M * N));
  checkCudaErrors(cudaMemset(kernel_complex_device, 0, sizeof(TD) * M * N));
  checkCudaErrors(cudaMemcpy(kernel_complex_device, kernel_device,
                             sizeof(T) * M * N, cudaMemcpyDeviceToDevice));

  // Allocate memory for padded arrays
  // Kernel
  checkCudaErrors(cudaMalloc((void**)&padded_kernel_complex,
                             sizeof(TD) * padding_M * padding_N));
  checkCudaErrors(
      cudaMemset(padded_kernel_complex, 0, sizeof(TD) * padding_M * padding_N));
  // Data
  checkCudaErrors(cudaMalloc((void**)&padded_data_complex,
                             sizeof(TD) * padding_M * padding_N));
  checkCudaErrors(
      cudaMemset(padded_data_complex, 0, sizeof(TD) * padding_M * padding_N));

  // Calculate thread blocks to execute kernel
  dim3 threads(variables.blockSizeX, variables.blockSizeY);

  dim3 blocks_kernel(iDivUp(m, threads.x), iDivUp(n, threads.y));

  dim3 blocks_data(iDivUp(padding_M, threads.x), iDivUp(padding_N, threads.y));

  // Padding the kernel
  paddingKernel<TD><<<blocks_kernel, threads>>>(
      padded_kernel_complex, kernel_complex_device, padding_M, padding_N, m, n,
      ckernel_x, ckernel_y);
  checkCudaErrors(cudaDeviceSynchronize());

  // Padding the data
  paddingData<TD><<<blocks_data, threads>>>(padded_data_complex, data_device,
                                            padding_M, padding_N, M, N, m, n,
                                            ckernel_x, ckernel_y);
  checkCudaErrors(cudaDeviceSynchronize());

  // Allocating memory for FFT results
  checkCudaErrors(cudaMalloc((void**)&data_spectrum_device,
                             sizeof(TD) * padding_M * padding_N));
  checkCudaErrors(cudaMalloc((void**)&kernel_spectrum_device,
                             sizeof(TD) * padding_M * padding_N));

  checkCudaErrors(cufftPlan2d(&fftPlan, padding_M, padding_N, CUFFT_R2C));

  FFT2D(data_spectrum_device, padded_data_complex, fftPlan, padding_M,
        padding_N, fftfwd, false);

  FFT2D(kernel_spectrum_device, padded_kernel_complex, fftPlan, padding_M,
        padding_N, fftfwd, false);

  mulArrayComplexComplex<<<blocks_data, threads>>>(
      data_spectrum_device, kernel_spectrum_device, padding_M, padding_N);
  checkCudaErrors(cudaDeviceSynchronize());

  FFT2D(padded_data_complex, data_spectrum_device, fftPlan, padding_M,
        padding_N, fftinv, false);

  depaddingData<TD><<<numBlocksNN, threadsPerBlockNN>>>(
      data_device, padded_data_complex, padding_M, padding_N, M, N, ckernel_x,
      ckernel_y);
  checkCudaErrors(cudaDeviceSynchronize());

  if (isDataOnDevice) {
    checkCudaErrors(cudaMemcpy(data, data_device, sizeof(TD) * M * N,
                               cudaMemcpyDeviceToHost));
  } else {
    result_host = (TD*)malloc(M * N * sizeof(TD));
    checkCudaErrors(cudaMemcpy(result_host, data_device, sizeof(TD) * M * N,
                               cudaMemcpyDeviceToHost));
  }

  // Free GPU MEMORY
  checkCudaErrors(cudaFree(kernel_device));
  checkCudaErrors(cudaFree(kernel_complex_device));
  checkCudaErrors(cudaFree(padded_kernel_complex));
  checkCudaErrors(cudaFree(padded_data_complex));
  checkCudaErrors(cudaFree(data_spectrum_device));
  checkCudaErrors(cudaFree(kernel_spectrum_device));
  checkCudaErrors(cudaFree(padded_data_complex));
  checkCudaErrors(cudaFree(data_device));
  checkCudaErrors(cufftDestroy(fftPlan));

  if (isDataOnDevice) {
    return data;
  } else {
    return result_host;
  }
}

__global__ void DFT2D(cufftComplex* Vm,
                      cufftComplex* I,
                      double3* UVW,
                      float* noise,
                      float noise_cut,
                      float xobs,
                      float yobs,
                      double DELTAX,
                      double DELTAY,
                      int M,
                      int N,
                      int numVisibilities) {
  const int v = threadIdx.x + blockDim.x * blockIdx.x;

  if (v < numVisibilities) {
    int x0, y0;
    double x, y, z;
    float cosk, sink, Ukv, Vkv, Wkv, I_sky;
    cufftComplex Vmodel;
    Vmodel.x = 0.0f;
    Vmodel.y = 0.0f;
    double3 uvw = UVW[v];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        x0 = xobs;
        y0 = yobs;
        x = (j - x0) * DELTAX * RPDEG_D;
        y = (i - y0) * DELTAY * RPDEG_D;
        z = sqrtf(1 - x * x - y * y) - 1;
        I_sky = I[N * i + j].x;
        if (noise[N * i + j] > noise_cut) {
          Ukv = x * uvw.x;
          Vkv = y * uvw.y;
          Wkv = z * uvw.z;
#if (__CUDA_ARCH__ >= 300)
          sincospif(2.0 * (Ukv + Vkv + Wkv), &sink, &cosk);
#else
          cosk = cospif(2.0 * (Ukv + Vkv + Wkv));
          sink = sinpif(2.0 * (Ukv + Vkv + Wkv));
#endif
          Vmodel = make_cuFloatComplex(Vmodel.x + I_sky * cosk,
                                       Vmodel.y + I_sky * sink);
        }
      }
    }
    Vm[v] = Vmodel;
  }
}

__host__ void do_gridding(std::vector<Field>& fields,
                          MSData* data,
                          double deltau,
                          double deltav,
                          int M,
                          int N,
                          CKernel* ckernel,
                          int gridding) {
  std::vector<float> g_weights(M * N);
  std::vector<float> g_weights_aux(M * N);
  std::vector<cufftComplex> g_Vo(M * N);
  std::vector<double3> g_uvw(M * N);
  cufftComplex complex_zero = floatComplexZero();

  double3 double3_zero;
  double3_zero.x = 0.0;
  double3_zero.y = 0.0;
  double3_zero.z = 0.0;

  int local_max = 0;
  int max = 0;
  float pow2_factor, S2, w_avg;

  /*
     Private variables - parallel for loop
   */
  int j, k;
  int grid_pos_x, grid_pos_y;
  double3 uvw;
  float w;
  cufftComplex Vo;
  int herm_j, herm_k;
  int shifted_j, shifted_k;
  int kernel_i, kernel_j;
  int visCounterPerFreq = 0;
  float ckernel_result = 1.0;
  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      visCounterPerFreq = 0;
      for (int s = 0; s < data->nstokes; s++) {
        fields[f].backup_visibilities[i][s].uvw.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].backup_visibilities[i][s].Vo.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].backup_visibilities[i][s].weight.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].backup_visibilities[i][s].uvw.assign(
            fields[f].visibilities[i][s].uvw.begin(),
            fields[f].visibilities[i][s].uvw.end());
        fields[f].backup_visibilities[i][s].weight.assign(
            fields[f].visibilities[i][s].weight.begin(),
            fields[f].visibilities[i][s].weight.end());
        fields[f].backup_visibilities[i][s].Vo.assign(
            fields[f].visibilities[i][s].Vo.begin(),
            fields[f].visibilities[i][s].Vo.end());
#pragma omp parallel for schedule(static, 1) num_threads(gridding)          \
    shared(g_weights, g_weights_aux, g_Vo) private(                         \
            j, k, grid_pos_x, grid_pos_y, uvw, w, Vo, shifted_j, shifted_k, \
                kernel_i, kernel_j, herm_j, herm_k, ckernel_result) ordered
        for (int z = 0; z < 2 * fields[f].numVisibilitiesPerFreqPerStoke[i][s];
             z++) {
          // Loop here is done twice since dataset only has half of the data
          int index = z;
          if (z >= fields[f].numVisibilitiesPerFreqPerStoke[i][s])
            index = z - fields[f].numVisibilitiesPerFreqPerStoke[i][s];

          uvw = fields[f].visibilities[i][s].uvw[index];
          w = fields[f].visibilities[i][s].weight[index];
          Vo = fields[f].visibilities[i][s].Vo[index];

          // The second half of the data must be the Hermitian symmetric
          // visibilities
          if (z >= fields[f].numVisibilitiesPerFreqPerStoke[i][s]) {
            uvw.x *= -1.0;
            uvw.y *= -1.0;
            uvw.z *= -1.0;
            Vo.y *= -1.0f;
          }

          // Visibilities from metres to klambda
          uvw.x = metres_to_lambda(uvw.x, fields[f].nu[i]);
          uvw.y = metres_to_lambda(uvw.y, fields[f].nu[i]);
          uvw.z = metres_to_lambda(uvw.z, fields[f].nu[i]);

          grid_pos_x = uvw.x / deltau;
          grid_pos_y = uvw.y / deltav;
          j = grid_pos_x + int(floor(N / 2)) + 0.5;
          k = grid_pos_y + int(floor(M / 2)) + 0.5;

          for (int m = -ckernel->getSupportY(); m <= ckernel->getSupportY();
               m++) {
            for (int n = -ckernel->getSupportX(); n <= ckernel->getSupportX();
                 n++) {
              shifted_j = j + n;
              shifted_k = k + m;
              kernel_j = n + ckernel->getSupportX();
              kernel_i = m + ckernel->getSupportY();
#pragma omp ordered
              {
#pragma omp critical
                {
                  if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
                      shifted_j < N) {
                    ckernel_result =
                        ckernel->getKernelValue(kernel_i, kernel_j);
                    g_Vo[N * shifted_k + shifted_j].x +=
                        w * Vo.x * ckernel_result;
                    g_Vo[N * shifted_k + shifted_j].y +=
                        w * Vo.y * ckernel_result;
                    g_weights[N * shifted_k + shifted_j] += w * ckernel_result;
                    g_weights_aux[N * shifted_k + shifted_j] +=
                        w * ckernel_result * ckernel_result;
                  }
                }
              }
            }
          }
        }

// fitsOutputCufftComplex(g_Vo.data(), mod_in, "gridfft_beforedividing.fits",
// "./", 0, 1.0, M, N, 0, false); OFITS(g_weights.data(), mod_in, "./",
// "weights_grid_beforedividing.fits", "JY/PIXEL", 0, 0, 1.0f, M, N, false);
// OFITS(g_weights_aux.data(), mod_in, "./",
// "weights_grid_aux_beforedividing.fits", "JY/PIXEL", 0, 0, 1.0f, M, N, false);
//  Normalize visibilities and weights
#pragma omp parallel for schedule(static, 1) \
    shared(g_weights, g_weights_aux, g_Vo, g_uvw)
        for (int k = 0; k < M; k++) {
          for (int j = 0; j < N; j++) {
            double u_lambdas = (j - int(floor(N / 2))) * deltau;
            double v_lambdas = (k - int(floor(M / 2))) * deltav;

            double u_meters = u_lambdas * freq_to_wavelength(data->max_freq);
            double v_meters = v_lambdas * freq_to_wavelength(data->max_freq);

            g_uvw[N * k + j].x = u_meters;
            g_uvw[N * k + j].y = v_meters;
            float ws = g_weights[N * k + j];
            float aux_ws = g_weights_aux[N * k + j];
            float weight;
            if (aux_ws != 0.0f && ws != 0.0f) {
              weight = ws * ws / aux_ws;
              g_Vo[N * k + j].x /= ws;
              g_Vo[N * k + j].y /= ws;
              g_weights[N * k + j] = weight;
            } else {
              g_weights[N * k + j] = 0.0f;
            }
          }
        }

        // The following lines are to create images with the resulting (u,v)
        // grid and weights
        // fitsOutputCufftComplex(g_Vo.data(), mod_in,
        // "gridfft_afterdividing.fits", "./", 0, 1.0, M, N, 0, false);
        // OFITS(g_weights.data(), mod_in, "./",
        // "weights_grid_afterdividing.fits", "JY/PIXEL", 0, 0, 1.0f, M, N,
        // false);

        // We already know that in quadrants < N/2 there are only zeros
        // Therefore, we start j from N/2
        int visCounter = 0;
#pragma omp parallel for shared(g_weights) reduction(+ : visCounter)
        for (int k = 0; k < M; k++) {
          for (int j = 0; j < N; j++) {
            float weight = g_weights[N * k + j];
            if (weight > 0.0f) {
              visCounter++;
            }
          }
        }

        fields[f].visibilities[i][s].uvw.resize(visCounter);
        fields[f].visibilities[i][s].Vo.resize(visCounter);

        fields[f].visibilities[i][s].Vm.resize(visCounter);
        memset(&fields[f].visibilities[i][s].Vm[0], 0,
               fields[f].visibilities[i][s].Vm.size() * sizeof(cufftComplex));

        fields[f].visibilities[i][s].weight.resize(visCounter);

        int l = 0;
        float weight;
        for (int k = 0; k < M; k++) {
          for (int j = 0; j < N; j++) {
            weight = g_weights[N * k + j];
            if (weight > 0.0f) {
              fields[f].visibilities[i][s].uvw[l].x = g_uvw[N * k + j].x;
              fields[f].visibilities[i][s].uvw[l].y = g_uvw[N * k + j].y;
              fields[f].visibilities[i][s].uvw[l].z = 0.0;
              fields[f].visibilities[i][s].Vo[l] =
                  make_cuFloatComplex(g_Vo[N * k + j].x, g_Vo[N * k + j].y);
              fields[f].visibilities[i][s].weight[l] = g_weights[N * k + j];
              l++;
            }
          }
        }

        fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s] =
            fields[f].numVisibilitiesPerFreqPerStoke[i][s];

        if (fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
          fields[f].numVisibilitiesPerFreqPerStoke[i][s] = visCounter;
          visCounterPerFreq += visCounter;
        } else {
          fields[f].numVisibilitiesPerFreqPerStoke[i][s] = 0;
        }

        std::fill_n(g_weights_aux.begin(), M * N, 0.0f);
        std::fill_n(g_weights.begin(), M * N, 0.0f);
        std::fill_n(g_uvw.begin(), M * N, double3_zero);
        std::fill_n(g_Vo.begin(), M * N, complex_zero);
      }

      local_max =
          *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i].begin(),
                            fields[f].numVisibilitiesPerFreqPerStoke[i].end());
      if (local_max > max) {
        max = local_max;
      }

      fields[f].backup_numVisibilitiesPerFreq[i] =
          fields[f].numVisibilitiesPerFreq[i];
      fields[f].numVisibilitiesPerFreq[i] = visCounterPerFreq;
    }
  }

  data->max_number_visibilities_in_channel_and_stokes = max;
}

__host__ void calc_sBeam(std::vector<double3> uvw,
                         std::vector<float> weight,
                         float nu,
                         double* s_uu,
                         double* s_vv,
                         double* s_uv) {
  double u_lambda, v_lambda;
#pragma omp parallel for shared(uvw, weight) private(u_lambda, v_lambda)
  for (int i = 0; i < uvw.size(); i++) {
    u_lambda = metres_to_lambda(uvw[i].x, nu);
    v_lambda = metres_to_lambda(uvw[i].y, nu);
#pragma omp critical
    {
      *s_uu += u_lambda * u_lambda * weight[i];
      *s_vv += v_lambda * v_lambda * weight[i];
      *s_uv += u_lambda * v_lambda * weight[i];
    }
  }
}

__host__ double3 calc_beamSize(double s_uu, double s_vv, double s_uv) {
  double3 beam_size;
  double uv_square = s_uv * s_uv;
  double uu_minus_vv = s_uu - s_vv;
  double uu_plus_vv = s_uu + s_vv;
  double sqrt_in = sqrt((uu_minus_vv * uu_minus_vv) + 4.0 * uv_square);
  beam_size.x = 1.0 / sqrt(2.0) / PI_D /
                sqrt(uu_plus_vv - sqrt_in);  // Major axis in radians
  beam_size.y = 1.0 / sqrt(2.0) / PI_D /
                sqrt(uu_plus_vv + sqrt_in);             // Minor axis in radians
  beam_size.z = -0.5 * atan2(2.0 * s_uv, uu_minus_vv);  // Angle in radians

  return beam_size;
}

__host__ float calculateNoiseAndBeam(std::vector<MSDataset>& datasets,
                                     int* total_visibilities,
                                     int blockSizeV,
                                     double* bmaj,
                                     double* bmin,
                                     double* bpa,
                                     float* noise) {
  // Declaring block size and number of blocks for visibilities
  float variance;
  float sum_weights = 0.0f;
  long UVpow2;
  double s_uu = 0.0;
  double s_vv = 0.0;
  double s_uv = 0.0;

  int device = -1;
  cudaDeviceProp dprop;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&dprop, device));
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              // sum_inverse_weight += 1 /
              // fields[f].visibilities[i][s].weight[j]; sum_weights +=
              // std::accumulate(datasets[d].fields[f].visibilities[i][s].weight.begin(),
              // datasets[d].fields[f].visibilities[i][s].weight.end(), 0.0f);
              calc_sBeam(datasets[d].fields[f].visibilities[i][s].uvw,
                         datasets[d].fields[f].visibilities[i][s].weight,
                         datasets[d].fields[f].nu[i], &s_uu, &s_vv, &s_uv);
              sum_weights += reduceCPU<float>(
                  datasets[d].fields[f].visibilities[i][s].weight.data(),
                  datasets[d].fields[f].visibilities[i][s].weight.size());
              *total_visibilities +=
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s];
            }
          }
          UVpow2 = NearestPowerOf2(
              datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
          if (blockSizeV == -1) {
            int threads1D, blocks1D;
            int threadsV, blocksV;
            threads1D = 512;
            blocks1D = iDivUp(UVpow2, threads1D);
            if (UVpow2 != 0) {
              getNumBlocksAndThreads(UVpow2, blocks1D, threads1D, blocksV,
                                     threadsV, false);
              datasets[d]
                  .fields[f]
                  .device_visibilities[i][s]
                  .threadsPerBlockUV = threadsV;
              datasets[d].fields[f].device_visibilities[i][s].numBlocksUV =
                  blocksV;
            } else {
              datasets[d]
                  .fields[f]
                  .device_visibilities[i][s]
                  .threadsPerBlockUV = threads1D;
              datasets[d].fields[f].device_visibilities[i][s].numBlocksUV =
                  blocks1D;
            }

          } else {
            datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV =
                blockSizeV;
            datasets[d].fields[f].device_visibilities[i][s].numBlocksUV =
                iDivUp(UVpow2, blockSizeV);
          }
        }
      }
    }
  }

  // We have calculate the running means so we divide by the sum of the weights

  if (sum_weights > 0.0f) {
    s_uu /= sum_weights;
    s_vv /= sum_weights;
    s_uv /= sum_weights;
    variance = 1.0f / sum_weights;
  } else {
    printf("Error: The sum of the visibility weights cannot be zero\n");
    exit(-1);
  }

  double3 beam_size_rad = calc_beamSize(s_uu, s_vv, s_uv);

  *bmaj = beam_size_rad.x / RPDEG_D;  // Major axis to degrees
  *bmin = beam_size_rad.y / RPDEG_D;  // Minor axis to degrees
  *bpa = beam_size_rad.z / RPDEG_D;   // Angle to degrees

  if (verbose_flag) {
    float aux_noise = 0.5f * sqrtf(variance);
    printf("Calculated NOISE %e\n", aux_noise);
  }

  if (*noise <= 0.0) {
    *noise = 0.5f * sqrtf(variance);
    if (verbose_flag) {
      printf("No NOISE keyword entered or detected in header\n");
      printf("Using NOISE: %e ...\n", *noise);
    }
  } else {
    printf("Using header keyword or entered NOISE...\n");
    printf("NOISE = %e\n", *noise);
  }

  return sum_weights;
}

__host__ void griddedTogrid(std::vector<cufftComplex>& Vm_gridded,
                            std::vector<cufftComplex> Vm_gridded_sp,
                            std::vector<double3> uvw_gridded_sp,
                            double deltau,
                            double deltav,
                            float freq,
                            long M,
                            long N,
                            int numvis) {
  float lambda = freq_to_wavelength(freq);
  double deltau_meters = deltau * lambda;
  double deltav_meters = deltav * lambda;

  cufftComplex complex_zero = floatComplexZero();

  std::fill_n(Vm_gridded.begin(), M * N, complex_zero);

  int j, k;
  int grid_pos_x, grid_pos_y;
  for (int i = 0; i < numvis; i++) {
    grid_pos_x = uvw_gridded_sp[i].x / deltau_meters;
    grid_pos_y = uvw_gridded_sp[i].y / deltav_meters;
    j = grid_pos_x + int(floor(N / 2));
    k = grid_pos_y + int(floor(M / 2));
    Vm_gridded[N * k + j] = Vm_gridded_sp[i];
  }
}

__host__ void getOriginalVisibilitiesBack(std::vector<Field>& fields,
                                          MSData data,
                                          int num_gpus,
                                          int firstgpu,
                                          int blockSizeV) {
  int local_max = 0;
  int max = 0;
  for (int f = 0; f < data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
    for (int i = 0; i < data.total_frequencies; i++) {
      unsigned int j = omp_get_thread_num();
      unsigned int num_cpu_threads = omp_get_num_threads();
      int gpu_idx = i % num_gpus;
      cudaSetDevice(gpu_idx + firstgpu);
      int gpu_id = -1;
      cudaGetDevice(&gpu_id);

      for (int s = 0; s < data.nstokes; s++) {
        // Now the number of visibilities will be the original one.

        fields[f].numVisibilitiesPerFreqPerStoke[i][s] =
            fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s];

        fields[f].visibilities[i][s].uvw.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].weight.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].Vm.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s], floatComplexZero());
        fields[f].visibilities[i][s].Vo.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vr,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMemset(fields[f].device_visibilities[i][s].Vr, 0,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

        checkCudaErrors(cudaMalloc(
            &fields[f].device_visibilities[i][s].uvw,
            sizeof(double3) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(cudaMalloc(
            &fields[f].device_visibilities[i][s].weight,
            sizeof(float) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

        // Copy original Vo visibilities to host
        fields[f].visibilities[i][s].Vo.assign(
            fields[f].backup_visibilities[i][s].Vo.begin(),
            fields[f].backup_visibilities[i][s].Vo.end());

        // Copy original (u,v) positions and weights to host and device

        fields[f].visibilities[i][s].uvw.assign(
            fields[f].backup_visibilities[i][s].uvw.begin(),
            fields[f].backup_visibilities[i][s].uvw.end());

        fields[f].visibilities[i][s].weight.assign(
            fields[f].backup_visibilities[i][s].weight.begin(),
            fields[f].backup_visibilities[i][s].weight.end());

        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].visibilities[i][s].uvw.data(),
            sizeof(double3) * fields[f].visibilities[i][s].uvw.size(),
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].Vo,
            fields[f].visibilities[i][s].Vo.data(),
            sizeof(cufftComplex) * fields[f].visibilities[i][s].Vo.size(),
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].weight,
            fields[f].visibilities[i][s].weight.data(),
            sizeof(float) * fields[f].visibilities[i][s].weight.size(),
            cudaMemcpyHostToDevice));

        long UVpow2 =
            NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        if (blockSizeV == -1) {
          int threads1D, blocks1D;
          int threadsV, blocksV;
          threads1D = 512;
          blocks1D = iDivUp(UVpow2, threads1D);
          getNumBlocksAndThreads(UVpow2, blocks1D, threads1D, blocksV, threadsV,
                                 false);
          fields[f].device_visibilities[i][s].threadsPerBlockUV = threadsV;
          fields[f].device_visibilities[i][s].numBlocksUV = blocksV;
        } else {
          fields[f].device_visibilities[i][s].threadsPerBlockUV = blockSizeV;
          fields[f].device_visibilities[i][s].numBlocksUV =
              iDivUp(UVpow2, blockSizeV);
        }

        hermitianSymmetry<<<
            fields[f].device_visibilities[i][s].numBlocksUV,
            fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].device_visibilities[i][s].Vo, fields[f].nu[i],
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        checkCudaErrors(cudaDeviceSynchronize());
      }
    }
  }

  for (int f = 0; f < data.nfields; f++) {
    for (int i = 0; i < data.total_frequencies; i++) {
      local_max =
          *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i].begin(),
                            fields[f].numVisibilitiesPerFreqPerStoke[i].end());
      if (local_max > max) {
        max = local_max;
      }
    }
  }

  data.max_number_visibilities_in_channel_and_stokes = max;
  max_number_vis = max;

  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    checkCudaErrors(
        cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float) * max_number_vis));
    checkCudaErrors(
        cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float) * max_number_vis));
  }

  cudaSetDevice(firstgpu);
}

__host__ void degridding(std::vector<Field>& fields,
                         MSData data,
                         double deltau,
                         double deltav,
                         int num_gpus,
                         int firstgpu,
                         int blockSizeV,
                         long M,
                         long N,
                         CKernel* ckernel) {
  long UVpow2;

  modelToHost(fields, data, num_gpus, firstgpu);

  std::vector<std::vector<cufftComplex>> gridded_visibilities(
      num_gpus, std::vector<cufftComplex>(M * N));

  for (int f = 0; f < data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
    for (int i = 0; i < data.total_frequencies; i++) {
      unsigned int j = omp_get_thread_num();
      unsigned int num_cpu_threads = omp_get_num_threads();
      int gpu_idx = i % num_gpus;
      cudaSetDevice(gpu_idx + firstgpu);
      int gpu_id = -1;
      cudaGetDevice(&gpu_id);

      for (int s = 0; s < data.nstokes; s++) {
        // Put gridded visibilities in a M*N grid
        griddedTogrid(
            gridded_visibilities[gpu_idx], fields[f].visibilities[i][s].Vm,
            fields[f].visibilities[i][s].uvw, deltau, deltav, fields[f].nu[i],
            M, N, fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        /*
        Model visibilities and original (u,v) positions to GPU.
        */
        // Now the number of visibilities will be the original one.
        fields[f].numVisibilitiesPerFreqPerStoke[i][s] =
            fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s];

        fields[f].visibilities[i][s].uvw.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].weight.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].Vm.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].Vo.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

        checkCudaErrors(cudaMalloc(
            &fields[f].device_visibilities[i][s].uvw,
            sizeof(double3) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(cudaMalloc(
            &fields[f].device_visibilities[i][s].weight,
            sizeof(float) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

        // Copy original Vo visibilities to host
        fields[f].visibilities[i][s].Vo.assign(
            fields[f].backup_visibilities[i][s].Vo.begin(),
            fields[f].backup_visibilities[i][s].Vo.end());

        // Copy gridded model visibilities to device
        checkCudaErrors(cudaMemcpy(
            vars_gpu[gpu_idx].device_V, gridded_visibilities[gpu_idx].data(),
            sizeof(cufftComplex) * M * N, cudaMemcpyHostToDevice));

        // Copy original (u,v) positions and weights to host and device

        fields[f].visibilities[i][s].uvw.assign(
            fields[f].backup_visibilities[i][s].uvw.begin(),
            fields[f].backup_visibilities[i][s].uvw.end());
        fields[f].visibilities[i][s].weight.assign(
            fields[f].backup_visibilities[i][s].weight.begin(),
            fields[f].backup_visibilities[i][s].weight.end());

        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].backup_visibilities[i][s].uvw.data(),
            sizeof(double3) * fields[f].backup_visibilities[i][s].uvw.size(),
            cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(fields[f].device_visibilities[i][s].Vo,
                       fields[f].backup_visibilities[i][s].Vo.data(),
                       sizeof(cufftComplex) *
                           fields[f].backup_visibilities[i][s].Vo.size(),
                       cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].weight,
            fields[f].backup_visibilities[i][s].weight.data(),
            sizeof(float) * fields[f].backup_visibilities[i][s].weight.size(),
            cudaMemcpyHostToDevice));

        if (blockSizeV == -1) {
          int threads1D, blocks1D;
          int threadsV, blocksV;
          long UVpow2 =
              NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
          threads1D = 512;
          blocks1D = iDivUp(UVpow2, threads1D);
          getNumBlocksAndThreads(UVpow2, blocks1D, threads1D, blocksV, threadsV,
                                 false);
          fields[f].device_visibilities[i][s].threadsPerBlockUV = threadsV;
          fields[f].device_visibilities[i][s].numBlocksUV = blocksV;
        } else {
          fields[f].device_visibilities[i][s].threadsPerBlockUV = blockSizeV;
          fields[f].device_visibilities[i][s].numBlocksUV = iDivUp(
              NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]),
              blockSizeV);
        }

        /*hermitianSymmetry<<<
            fields[f].device_visibilities[i][s].numBlocksUV,
            fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].device_visibilities[i][s].Vo, fields[f].nu[i],
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        checkCudaErrors(cudaDeviceSynchronize());*/

        // Interpolation / Degridding
        vis_mod2<<<fields[f].device_visibilities[i][s].numBlocksUV,
                   fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].Vm, vars_gpu[gpu_idx].device_V,
            fields[f].device_visibilities[i][s].uvw,
            fields[f].device_visibilities[i][s].weight, deltau, deltav,
            fields[f].numVisibilitiesPerFreqPerStoke[i][s], N);
        // degriddingGPU<<< fields[f].device_visibilities[i][s].numBlocksUV,
        //             fields[f].device_visibilities[i][s].threadsPerBlockUV
        //             >>>(fields[f].device_visibilities[i][s].uvw,
        //             fields[f].device_visibilities[i][s].Vm,
        //             vars_gpu[gpu_idx].device_V, ckernel->getGPUKernel(),
        //             deltau, deltav,
        //             fields[f].numVisibilitiesPerFreqPerStoke[i][s], M, N,
        //             ckernel->getm(), ckernel->getn(), ckernel->getSupportX(),
        //             ckernel->getSupportY());
        checkCudaErrors(cudaDeviceSynchronize());
      }
    }
  }
}

__host__ void initFFT(varsPerGPU* vars_gpu,
                      long M,
                      long N,
                      int firstgpu,
                      int num_gpus) {
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice((g % num_gpus) + firstgpu);
    checkCudaErrors(cufftPlan2d(&vars_gpu[g].plan, N, M, CUFFT_C2C));
  }
}

__host__ void FFT2D(cufftComplex* output_data,
                    cufftComplex* input_data,
                    cufftHandle plan,
                    int M,
                    int N,
                    int direction,
                    bool shift) {
  if (shift) {
    fftshift_2D<<<numBlocksNN, threadsPerBlockNN>>>(input_data, M, N);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)input_data,
                               (cufftComplex*)output_data, direction));

  if (shift) {
    fftshift_2D<<<numBlocksNN, threadsPerBlockNN>>>(output_data, M, N);
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

__global__ void do_griddingGPU(float3* uvw,
                               cufftComplex* Vo,
                               cufftComplex* Vo_g,
                               float* w,
                               float* w_g,
                               int* count,
                               double deltau,
                               double deltav,
                               int visibilities,
                               int M,
                               int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k, j;
  if (i < visibilities) {
    j = uvw[i].x / deltau + int(floorf(M / 2)) + 0.5;
    k = uvw[i].y / deltav + int(floorf(N / 2)) + 0.5;

    if (k < M && j < N) {
      atomicAdd(&Vo_g[N * k + j].x, w[i] * Vo[i].x);
      atomicAdd(&Vo_g[N * k + j].y, w[i] * Vo[i].y);
      atomicAdd(&w_g[N * k + j], w[i]);
    }
  }
}

__global__ void degriddingGPU(double3* uvw,
                              cufftComplex* Vm,
                              cufftComplex* Vm_g,
                              float* kernel,
                              double deltau,
                              double deltav,
                              int visibilities,
                              int M,
                              int N,
                              int kernel_m,
                              int kernel_n,
                              int supportX,
                              int supportY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k, j;
  int shifted_k, shifted_j;
  int kernel_i, kernel_j;
  int herm_j, herm_k;
  cufftComplex degrid_val = floatComplexZero();
  float ckernel_result;

  if (i < visibilities) {
    j = uvw[i].x / deltau + int(floorf(N / 2)) + 0.5;
    k = uvw[i].y / deltav + int(floorf(M / 2)) + 0.5;

    for (int m = -supportY; m <= supportY; m++) {
      for (int n = -supportX; n <= supportX; n++) {
        shifted_j = j + n;
        shifted_k = k + m;
        kernel_j = n + supportX;
        kernel_i = m + supportY;
        if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
            shifted_j < N) {
          ckernel_result = kernel[kernel_n * kernel_i + kernel_j];
          if (shifted_j >= N / 2) {
            degrid_val.x += ckernel_result * Vm_g[N * shifted_k + shifted_j].x;
            degrid_val.y += ckernel_result * Vm_g[N * shifted_k + shifted_j].y;
          } else {
            herm_j = N - shifted_j;
            herm_k = M - shifted_k;
            degrid_val.x += ckernel_result * Vm_g[N * herm_k + herm_j].x;
            degrid_val.y -= ckernel_result * Vm_g[N * herm_k + herm_j].y;
          }
        }
      }
    }
    Vm[i].x = degrid_val.x;
    Vm[i].y = degrid_val.y;
  }
}

__global__ void hermitianSymmetry(double3* UVW,
                                  cufftComplex* Vo,
                                  float freq,
                                  int numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    if (UVW[i].x > 0.0) {
      UVW[i].x *= -1.0;
      UVW[i].y *= -1.0;
      // UVW[i].z *= -1.0;
      Vo[i].y *= -1.0f;
    }
    UVW[i].x = metres_to_lambda(UVW[i].x, freq);
    UVW[i].y = metres_to_lambda(UVW[i].y, freq);
    UVW[i].z = metres_to_lambda(UVW[i].z, freq);
  }
}

__device__ float AiryDiskBeam(float distance,
                              float lambda,
                              float antenna_diameter,
                              float pb_factor) {
  float atten = 1.0f;
  if (distance != 0.0) {
    float r = pb_factor * lambda / antenna_diameter;
    float bessel_arg = PI * distance / (r / RZ);
    float bessel_func = j1f(bessel_arg);
    atten = 4.0f * (bessel_func / bessel_arg) * (bessel_func / bessel_arg);
  }

  return atten;
}

__device__ float GaussianBeam(float distance,
                              float lambda,
                              float antenna_diameter,
                              float pb_factor) {
  float fwhm = pb_factor * lambda / antenna_diameter;
  float c = 4.0f * logf(2.0f);
  float r = distance / fwhm;
  float atten = expf(-c * r * r);
  return atten;
}

__device__ float attenuation(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float freq,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten_result, atten;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG_D;
  float y = (i - y0) * DELTAY * RPDEG_D;

  float arc = distance(x, y, 0.0, 0.0);
  float lambda = freq_to_wavelength(freq);
  atten = (*beam_maps[primary_beam])(arc, lambda, antenna_diameter, pb_factor);
  if (arc <= pb_cutoff) {
    atten_result = atten;
  } else {
    atten_result = 0.0f;
  }

  return atten_result;
}

__device__ cufftComplex
WKernel(double w, float xobs, float yobs, double DELTAX, double DELTAY) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  cufftComplex Wk;
  float cosk, sink;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG_D;
  float y = (i - y0) * DELTAY * RPDEG_D;
  float z = sqrtf(1 - x * x - y * y) - 1;
  float arg = 2.0f * w * z;

#if (__CUDA_ARCH__ >= 300)
  sincospif(arg, &sink, &cosk);
#else
  cosk = cospif(arg);
  sink = sinpif(arg);
#endif

  Wk = make_cuFloatComplex(cosk, -sink);
}

__global__ void distance_image(float* distance_image,
                               float xobs,
                               float yobs,
                               float dist_arcsec,
                               double DELTAX,
                               double DELTAY,
                               long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * 3600.0;
  float y = (i - y0) * DELTAY * 3600.0;

  float dist = distance(x, y, 0.0, 0.0);
  distance_image[N * i + j] = 1.0f;

  if (dist < dist_arcsec)
    distance_image[N * i + j] = 0.0f;
}

__global__ void total_attenuation(float* total_atten,
                                  float antenna_diameter,
                                  float pb_factor,
                                  float pb_cutoff,
                                  float freq,
                                  float xobs,
                                  float yobs,
                                  double DELTAX,
                                  double DELTAY,
                                  long N,
                                  int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float attenPerFreq = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq,
                                   xobs, yobs, DELTAX, DELTAY, primary_beam);
  total_atten[N * i + j] += attenPerFreq;
}

__global__ void weight_image(float* weight_image, float* total_atten, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = total_atten[N * i + j];
  weight_image[N * i + j] += atten * atten;
}

__global__ void noise_image(float* noise_image,
                            float* weight_image,
                            float max_weight,
                            float noise_jypix,
                            long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float noise_squared = noise_jypix * noise_jypix;
  float normalized_weight =
      (weight_image[N * i + j] / max_weight) / noise_squared;
  float noiseval = sqrtf(1.0f / normalized_weight);
  noise_image[N * i + j] = noiseval;
}

__global__ void apply_beam2I(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             cufftComplex* image,
                             long N,
                             float xobs,
                             float yobs,
                             float fg_scale,
                             float freq,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs,
                            yobs, DELTAX, DELTAY, primary_beam);

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * atten * fg_scale, 0.0f);
}

__global__ void apply_beam2I(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float* gcf,
                             cufftComplex* image,
                             long N,
                             float xobs,
                             float yobs,
                             float freq,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs,
                            yobs, DELTAX, DELTAY, primary_beam);

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * gcf[N * i + j] * atten, 0.0f);
}

__global__ void apply_GCF(cufftComplex* image, float* gcf, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * gcf[N * i + j], 0.0f);
}

/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex* data,
                             long M,
                             long N,
                             double xphs,
                             double yphs) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float u, v, phase, c, s;
  double upix = xphs / (double)M;
  double vpix = yphs / (double)N;
  cufftComplex exp_phase;

  if (j < M / 2) {
    u = upix * j;
  } else {
    u = upix * (j - M);
  }

  if (i < N / 2) {
    v = vpix * i;
  } else {
    v = vpix * (i - N);
  }

  phase = -2.0f * (u + v);
#if (__CUDA_ARCH__ >= 300)
  sincospif(phase, &s, &c);
#else
  c = cospif(phase);
  s = sinpif(phase);
#endif
  exp_phase = make_cuFloatComplex(c, s);  // Create the complex cos + i sin
  data[N * i + j] =
      cuCmulf(data[N * i + j], exp_phase);  // Complex multiplication
}

__global__ void getGriddedVisFromPix(cufftComplex* Vm,
                                     cufftComplex* V,
                                     double3* UVW,
                                     float* weight,
                                     double deltau,
                                     double deltav,
                                     long numVisibilities,
                                     long N) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  int i1, j1;
  double du, dv;
  double2 uv;

  if (i < numVisibilities) {
    uv.x = UVW[i].x / deltau;
    uv.y = UVW[i].y / deltav;

    if (uv.x < 0.0)
      uv.x += N;

    if (uv.y < 0.0)
      uv.y += N;

    j1 = uv.x;
    i1 = uv.y;

    if (i1 >= 0 && i1 < N && j1 >= 0 && j1 < N)
      Vm[i] = V[N * i1 + j1];
    else
      weight[i] = 0.0f;
  }
}

/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod(cufftComplex* Vm,
                        cufftComplex* V,
                        double3* UVW,
                        float* weight,
                        double deltau,
                        double deltav,
                        long numVisibilities,
                        long N) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  int i1, i2, j1, j2;
  double du, dv;
  double2 uv;
  cufftComplex v11, v12, v21, v22;
  float Zreal;
  float Zimag;

  if (i < numVisibilities) {
    uv.x = UVW[i].x / deltau;
    uv.y = UVW[i].y / deltav;

    if (uv.x < 0.0)
      uv.x += N;

    if (uv.y < 0.0)
      uv.y += N;

    i1 = uv.x;
    i2 = (i1 + 1) % N;
    du = uv.x - i1;

    j1 = uv.y;
    j2 = (j1 + 1) % N;
    dv = uv.y - j1;

    if (i1 >= 0 && i1 < N && i2 >= 0 && i2 < N && j1 >= 0 && j1 < N &&
        j2 >= 0 && j2 < N) {
      /* Bilinear interpolation */
      v11 = V[N * j1 + i1]; /* [i1, j1] */
      v12 = V[N * j2 + i1]; /* [i1, j2] */
      v21 = V[N * j1 + i2]; /* [i2, j1] */
      v22 = V[N * j2 + i2]; /* [i2, j2] */

      Zreal = (1 - du) * (1 - dv) * v11.x + (1 - du) * dv * v12.x +
              du * (1 - dv) * v21.x + du * dv * v22.x;
      Zimag = (1 - du) * (1 - dv) * v11.y + (1 - du) * dv * v12.y +
              du * (1 - dv) * v21.y + du * dv * v22.y;

      Vm[i] = make_cuFloatComplex(Zreal, Zimag);
    } else {
      weight[i] = 0.0f;
    }
  }
}

/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod2(cufftComplex* Vm,
                         cufftComplex* V,
                         double3* UVW,
                         float* weight,
                         double deltau,
                         double deltav,
                         long numVisibilities,
                         long N) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  double f_j, f_k;
  int j1, j2, k1, k2;
  double2 uv;
  cufftComplex Z;

  if (i < numVisibilities) {
    uv.x = UVW[i].x / deltau;
    uv.y = UVW[i].y / deltav;

    f_j = uv.x + int(floorf(N / 2));
    j1 = f_j;
    j2 = j1 + 1;
    f_j = f_j - j1;

    f_k = uv.y + int(floorf(N / 2));
    k1 = f_k;
    k2 = k1 + 1;
    f_k = f_k - k1;

    if (j1 < N && k1 < N && j2 < N && k2 < N) {
      /* Bilinear interpolation */
      // Real part
      Z.x = (1 - f_j) * (1 - f_k) * V[N * k1 + j1].x +
            f_j * (1 - f_k) * V[N * k1 + j2].x +
            (1 - f_j) * f_k * V[N * k2 + j1].x + f_j * f_k * V[N * k2 + j2].x;
      // Imaginary part
      Z.y = (1 - f_j) * (1 - f_k) * V[N * k1 + j1].y +
            f_j * (1 - f_k) * V[N * k1 + j2].y +
            (1 - f_j) * f_k * V[N * k2 + j1].y + f_j * f_k * V[N * k2 + j2].y;

      Vm[i] = Z;
    } else {
      weight[i] = 0.0f;
    }
  }
}

__global__ void residual(cufftComplex* Vr,
                         cufftComplex* Vm,
                         cufftComplex* Vo,
                         long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < numVisibilities) {
    Vr[i] = cuCsubf(Vo[i], Vm[i]);
  }
}

__global__ void clipWNoise(cufftComplex* fg_image,
                           float* noise,
                           float* I,
                           long N,
                           float noise_cut,
                           float MINPIX,
                           float eta) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (noise[N * i + j] > noise_cut) {
    if (eta > 0.0) {
      I[N * i + j] = 0.0;
    } else {
      I[N * i + j] = -1.0f * eta * MINPIX;
    }
  }

  fg_image[N * i + j] = make_cuFloatComplex(I[N * i + j], 0.0f);
}

__global__ void clip2IWNoise(float* noise,
                             float* I,
                             long N,
                             long M,
                             float noise_cut,
                             float MINPIX,
                             float alpha_start,
                             float eta,
                             float threshold,
                             int schedule) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (noise[N * i + j] > noise_cut) {
    if (eta > 0.0f) {
      I[N * i + j] = 0.0f;
    } else {
      I[N * i + j] = -1.0 * eta * MINPIX;
    }
    I[N * M + N * i + j] = 0.0f;
  } else {
    if (I[N * i + j] < threshold && schedule > 0) {
      I[N * M + N * i + j] = 0.0f;
    }
  }
}

__global__ void getGandDGG(float* gg, float* dgg, float* xi, float* g, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  gg[N * i + j] = g[N * i + j] * g[N * i + j];
  dgg[N * i + j] = (xi[N * i + j] + g[N * i + j]) * xi[N * i + j];
}

__global__ void getGGandDGG(float* gg,
                            float* dgg,
                            float* xi,
                            float* g,
                            long N,
                            long M,
                            int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float gg_temp;
  float dgg_temp;

  gg_temp = g[M * N * image + N * i + j] * g[M * N * image + N * i + j];

  dgg_temp = (xi[M * N * image + N * i + j] + g[M * N * image + N * i + j]) *
             xi[M * N * image + N * i + j];

  gg[N * i + j] += gg_temp;
  dgg[N * i + j] += dgg_temp;
}

__global__ void clip2I(float* I, long N, float MINPIX) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (I[N * i + j] < MINPIX && MINPIX >= 0.0) {
    I[N * i + j] = MINPIX;
  }
}

__global__ void newP(float* p,
                     float* xi,
                     float xmin,
                     float MINPIX,
                     float eta,
                     long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N * i + j] *= xmin;
  if (p[N * i + j] + xi[N * i + j] > -1.0f * eta * MINPIX) {
    p[N * i + j] += xi[N * i + j];
  } else {
    p[N * i + j] = -1.0f * eta * MINPIX;
    xi[N * i + j] = 0.0f;
  }
  // p[N*i+j].y = 0.0;
}

__global__ void newP(float* p,
                     float* xi,
                     float xmin,
                     long N,
                     long M,
                     float MINPIX,
                     float eta,
                     int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N * M * image + N * i + j] *= xmin;

  if (p[N * M * image + N * i + j] + xi[N * M * image + N * i + j] >
      -1.0 * eta * MINPIX) {
    p[N * M * image + N * i + j] += xi[N * M * image + N * i + j];
  } else {
    p[N * M * image + N * i + j] = -1.0f * eta * MINPIX;
    xi[N * M * image + N * i + j] = 0.0f;
  }
}

__global__ void newPNoPositivity(float* p,
                                 float* xi,
                                 float xmin,
                                 long N,
                                 long M,
                                 int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N * M * image + N * i + j] *= xmin;
  p[N * M * image + N * i + j] += xi[N * M * image + N * i + j];
}

__global__ void evaluateXt(float* xt,
                           float* pcom,
                           float* xicom,
                           float x,
                           float MINPIX,
                           float eta,
                           long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (pcom[N * i + j] + x * xicom[N * i + j] > -1.0f * eta * MINPIX) {
    xt[N * i + j] = pcom[N * i + j] + x * xicom[N * i + j];
  } else {
    xt[N * i + j] = -1.0f * eta * MINPIX;
  }
  // xt[N*i+j].y = 0.0;
}

__global__ void evaluateXt(float* xt,
                           float* pcom,
                           float* xicom,
                           float x,
                           long N,
                           long M,
                           float MINPIX,
                           float eta,
                           int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j] >
      -1.0 * eta * MINPIX) {
    xt[N * M * image + N * i + j] =
        pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j];
  } else {
    xt[N * M * image + N * i + j] = -1.0f * eta * MINPIX;
  }
}

__global__ void evaluateXtNoPositivity(float* xt,
                                       float* pcom,
                                       float* xicom,
                                       float x,
                                       long N,
                                       long M,
                                       int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xt[N * M * image + N * i + j] =
      pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j];
}

__global__ void chi2Vector(float* chi2,
                           cufftComplex* Vr,
                           float* w,
                           long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    chi2[i] = w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
  }
}

__host__ __device__ float approxAbs(float val, float epsilon) {
  return sqrtf(val * val + epsilon);
}

__device__ float calculateL1norm(float* I,
                                 float epsilon,
                                 float noise,
                                 float noise_cut,
                                 int index,
                                 int M,
                                 int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float c = I[N * M * index + N * i + j];

  float l1 = 0.0f;

  if (noise < noise_cut) {
    l1 = approxAbs(c, epsilon);
  }

  return l1;
}

__global__ void L1Vector(float* L1,
                         float* noise,
                         float* I,
                         long N,
                         long M,
                         float epsilon,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  L1[N * i + j] =
      calculateL1norm(I, epsilon, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateDNormL1(float* I,
                                  float lambda,
                                  float noise,
                                  float epsilon,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float den;
  float dL1 = 0.0f;
  float c = I[N * M * index + N * i + j];

  if (noise < noise_cut)
    dL1 = c / approxAbs(c, epsilon);

  dL1 *= lambda;
  return dL1;
}

__global__ void DL1NormK(float* dL1,
                         float* I,
                         float* noise,
                         float epsilon,
                         float noise_cut,
                         float lambda,
                         long N,
                         long M,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dL1[N * i + j] = calculateDNormL1(I, lambda, noise[N * i + j], epsilon,
                                    noise_cut, index, M, N);
}

__device__ float calculateGL1norm(float* I,
                                  float prior,
                                  float epsilon_a,
                                  float epsilon_b,
                                  float noise,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float c = I[N * M * index + N * i + j];

  float l1 = 0.0f;

  if (noise < noise_cut) {
    l1 = approxAbs(c, epsilon_a) / (approxAbs(prior, epsilon_a) + epsilon_b);
  }

  return l1;
}

__global__ void GL1Vector(float* L1,
                          float* noise,
                          float* I,
                          float* prior,
                          long N,
                          long M,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  L1[N * i + j] = calculateGL1norm(I, prior[N * i + j], epsilon_a, epsilon_b,
                                   noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateDGNormL1(float* I,
                                   float prior,
                                   float lambda,
                                   float noise,
                                   float epsilon_a,
                                   float epsilon_b,
                                   float noise_cut,
                                   int index,
                                   int M,
                                   int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float den;
  float dL1 = 0.0f;
  float c = I[N * M * index + N * i + j];

  if (noise < noise_cut)
    dL1 = c /
          (approxAbs(c, epsilon_a) * (approxAbs(prior, epsilon_a) + epsilon_b));

  dL1 *= lambda;
  return dL1;
}

__global__ void DGL1NormK(float* dL1,
                          float* I,
                          float* prior,
                          float* noise,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          float lambda,
                          long N,
                          long M,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dL1[N * i + j] =
      calculateDGNormL1(I, prior[N * i + j], lambda, noise[N * i + j],
                        epsilon_a, epsilon_b, noise_cut, index, M, N);
}

__device__ float calculateS(float* I,
                            float G,
                            float eta,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float c = I[N * M * index + N * i + j];

  float S = 0.0f;

  if (noise < noise_cut) {
    S = c * logf((c / G) + (eta + 1.0f));
  }

  return S;
}

__device__ float calculateDS(float* I,
                             float G,
                             float eta,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dS = 0.0f;

  float c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    dS =
        logf((c / G) + (eta + 1.0f)) + 1.0f / (1.0f + (((eta + 1.0f) * G) / c));
  }

  dS *= lambda;
  return dS;
}

__global__ void SVector(float* S,
                        float* noise,
                        float* I,
                        long N,
                        long M,
                        float noise_cut,
                        float prior_value,
                        float eta,
                        int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  S[N * i + j] =
      calculateS(I, prior_value, eta, noise[N * i + j], noise_cut, index, M, N);
}

__global__ void DS(float* dS,
                   float* I,
                   float* noise,
                   float noise_cut,
                   float lambda,
                   float prior_value,
                   float eta,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dS[N * i + j] = calculateDS(I, prior_value, eta, lambda, noise[N * i + j],
                              noise_cut, index, M, N);
}

__global__ void SGVector(float* S,
                         float* noise,
                         float* I,
                         long N,
                         long M,
                         float noise_cut,
                         float* prior,
                         float eta,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  S[N * i + j] = calculateS(I, prior[N * i + j], eta, noise[N * i + j],
                            noise_cut, index, M, N);
}

__global__ void DSG(float* dS,
                    float* I,
                    float* noise,
                    float noise_cut,
                    float lambda,
                    float* prior,
                    float eta,
                    long N,
                    long M,
                    int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dS[N * i + j] = calculateDS(I, prior[N * i + j], eta, lambda,
                              noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateQP(float* I,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float c, l, r, d, u;

  float qp = 0.0f;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1)) {
      l = I[N * M * index + N * i + (j - 1)];
      r = I[N * M * index + N * i + (j + 1)];
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];

      qp = (c - l) * (c - l) + (c - r) * (c - r) + (c - u) * (c - u) +
           (c - d) * (c - d);
      qp /= 2.0f;
    } else {
      qp = c;
    }
  }

  return qp;
}
__global__ void QPVector(float* Q,
                         float* noise,
                         float* I,
                         long N,
                         long M,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  Q[N * i + j] = calculateQP(I, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateDQ(float* I,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dQ = 0.0f;
  float c, d, u, r, l;

  c = I[N * M * index + N * i + j];

  if (noise < noise_cut) {
    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1)) {
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];
      r = I[N * M * index + N * i + (j + 1)];
      l = I[N * M * index + N * i + (j - 1)];

      dQ = 2.0f * (4.0f * c - d + u + r + l);
    } else {
      dQ = c;
    }
  }

  dQ *= lambda;

  return dQ;
}

__global__ void DQ(float* dQ,
                   float* I,
                   float* noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dQ[N * i + j] =
      calculateDQ(I, lambda, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateTV(float* I,
                             float epsilon,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float c, r, d;
  float tv = 0.0f;
  float dxy0, dxy1;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if (i < N - 1 && j < N - 1) {
      r = I[N * M * index + N * i + (j + 1)];
      d = I[N * M * index + N * (i + 1) + j];

      dxy0 = (r - c) * (r - c);
      dxy1 = (d - c) * (d - c);
      tv = sqrtf(dxy0 + dxy1 + epsilon);
    } else {
      tv = c;
    }
  }

  return tv;
}

__global__ void TVVector(float* TV,
                         float* noise,
                         float* I,
                         float epsilon,
                         long N,
                         long M,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  TV[N * i + j] =
      calculateTV(I, epsilon, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateDTV(float* I,
                              float epsilon,
                              float lambda,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float c, d, u, r, l, dl_corner, ru_corner;

  float num0, num1, num2;
  float den0, den1, den2;
  float den_arg0, den_arg1, den_arg2;
  float dtv = 0.0f;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1)) {
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];
      r = I[N * M * index + N * i + (j + 1)];
      l = I[N * M * index + N * i + (j - 1)];
      dl_corner = I[N * M * index + N * (i + 1) + (j - 1)];
      ru_corner = I[N * M * index + N * (i - 1) + (j + 1)];

      num0 = 2.0f * c - r - d;
      num1 = c - l;
      num2 = c - u;

      den_arg0 = (c - r) * (c - r) + (c - d) * (c - d) + epsilon;

      den_arg1 =
          (l - c) * (l - c) + (l - dl_corner) * (l - dl_corner) + epsilon;

      den_arg2 =
          (u - ru_corner) * (u - ru_corner) + (u - c) * (u - c) + epsilon;

      den0 = sqrtf(den_arg0);
      den1 = sqrtf(den_arg1);
      den2 = sqrtf(den_arg2);

      dtv = num0 / den0 + num1 / den1 + num2 / den2;
    } else {
      dtv = c;
    }
  }

  dtv *= lambda;

  return dtv;
}
__global__ void DTV(float* dTV,
                    float* I,
                    float* noise,
                    float epsilon,
                    float noise_cut,
                    float lambda,
                    long N,
                    long M,
                    int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dTV[N * i + j] = calculateDTV(I, epsilon, lambda, noise[N * i + j], noise_cut,
                                index, M, N);
}

__device__ float calculateTSV(float* I,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float c, r, d;
  float tv = 0.0f;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if (i < N - 1 && j < N - 1) {
      r = I[N * M * index + N * i + (j + 1)];
      d = I[N * M * index + N * (i + 1) + j];

      float dx = c - r;
      float dy = c - d;
      tv = dx * dx + dy * dy;
    } else {
      tv = c;
    }
  }

  return tv;
}

__global__ void TSVVector(float* STV,
                          float* noise,
                          float* I,
                          long N,
                          long M,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  STV[N * i + j] = calculateTSV(I, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateDTSV(float* I,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float c, d, u, r, l, dl_corner, ru_corner;

  float dstv = 0.0f;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1)) {
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];
      r = I[N * M * index + N * i + (j + 1)];
      l = I[N * M * index + N * i + (j - 1)];

      dstv = 8.0f * c - 2.0f * (u + l + d + r);
    } else {
      dstv = c;
    }
  }

  dstv *= lambda;

  return dstv;
}

__global__ void DTSV(float* dSTV,
                     float* I,
                     float* noise,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  float center, down, up, right, left, dl_corner, ru_corner;

  dSTV[N * i + j] =
      calculateDTSV(I, lambda, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateL(float* I,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float Dx, Dy;
  float L = 0.0f;
  float c, l, r, d, u;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if ((i > 0 && i < N - 1) && (j > 0 && j < N - 1)) {
      l = I[N * M * index + N * i + (j - 1)];
      r = I[N * M * index + N * i + (j + 1)];
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];

      Dx = l - 2.0f * c + r;
      Dy = u - 2.0f * c + d;
      L = 0.5f * (Dx + Dy) * (Dx + Dy);
    } else {
      L = c;
    }
  }

  return L;
}

__global__ void LVector(float* L,
                        float* noise,
                        float* I,
                        long N,
                        long M,
                        float noise_cut,
                        int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  L[N * i + j] = calculateL(I, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateDL(float* I,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float c, d, u, r, l, dl_corner, dr_corner, lu_corner, ru_corner, d2, u2, l2,
      r2;

  float dL = 0.0f;

  c = I[N * M * index + N * i + j];

  if (noise < noise_cut) {
    if ((i > 1 && i < N - 2) && (j > 1 && j < N - 2)) {
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];
      r = I[N * M * index + N * i + (j + 1)];
      l = I[N * M * index + N * i + (j - 1)];
      dl_corner = I[N * M * index + N * (i + 1) + (j - 1)];
      dr_corner = I[N * M * index + N * (i + 1) + (j + 1)];
      lu_corner = I[N * M * index + N * (i - 1) + (j - 1)];
      ru_corner = I[N * M * index + N * (i - 1) + (j + 1)];
      d2 = I[N * M * index + N * (i + 2) + j];
      u2 = I[N * M * index + N * (i - 2) + j];
      l2 = I[N * M * index + N * i + (j - 2)];
      r2 = I[N * M * index + N * i + (j + 2)];

      dL = 20.0f * c - 8.0f * (d - r - u - l) +
           2.0f * (dl_corner + dr_corner + lu_corner + ru_corner) + d2 + r2 +
           u2 + l2;
    } else
      dL = 0.0f;
  }

  dL *= lambda;

  return dL;
}

__global__ void DL(float* dL,
                   float* I,
                   float* noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dL[N * i + j] =
      calculateDL(I, lambda, noise[N * i + j], noise_cut, index, M, N);
}

__global__ void normalizeImageKernel(float* image,
                                     float normalization_factor,
                                     long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  image[N * i + j] /= normalization_factor;
}

__global__ void searchDirection(float* g, float* xi, float* h, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N * i + j] = -xi[N * i + j];
  xi[N * i + j] = h[N * i + j] = g[N * i + j];
}

__global__ void searchDirection_LBFGS(float* xi, long N, long M, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[M * N * image + N * i + j] *= -1.0f;
}

__global__ void getDot_LBFGS_ff(float* aux_vector,
                                float* vec_1,
                                float* vec_2,
                                int k,
                                int h,
                                int M,
                                int N,
                                int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  aux_vector[N * i + j] =
      vec_1[M * N * image * k + M * N * image + (N * i + j)] *
      vec_2[M * N * image * h + M * N * image + (N * i + j)];
}

__global__ void normArray(float* result,
                          float* array,
                          int M,
                          int N,
                          int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  result[M * N * image + (N * i + j)] =
      fabsf(array[M * N * image + (N * i + j)]);
}

__global__ void CGGradCondition(float* temp,
                                float* xi,
                                float* p,
                                float den,
                                int M,
                                int N,
                                int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  temp[M * N * image + (N * i + j)] =
      fabsf(xi[M * N * image + (N * i + j)]) *
      fmaxf(fabsf(p[M * N * image + (N * i + j)]), 1.0f) / den;
}

__global__ void
updateQ(float* d_q, float alpha, float* d_y, int k, int M, int N, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_q[M * N * image + N * i + j] +=
      alpha * d_y[M * N * image + M * N * k + (N * i + j)];
}

__global__ void getR(float* d_r,
                     float* d_q,
                     float scalar,
                     int M,
                     int N,
                     int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_r[M * N * image + N * i + j] = d_q[M * N * image + N * i + j] * scalar;
}

__global__ void calculateSandY(float* d_y,
                               float* d_s,
                               float* p,
                               float* xi,
                               float* p_old,
                               float* xi_old,
                               int iter,
                               int M,
                               int N,
                               int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_y[M * N * image * iter + M * N * image + (N * i + j)] =
      xi[M * N * image + N * i + j] -
      (-1.0f * xi_old[M * N * image + N * i + j]);
  d_s[M * N * image * iter + M * N * image + (N * i + j)] =
      p[M * N * image + N * i + j] - p_old[M * N * image + N * i + j];
}

__global__ void searchDirection(float* g,
                                float* xi,
                                float* h,
                                long N,
                                long M,
                                int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[M * N * image + N * i + j] = -xi[M * N * image + N * i + j];

  xi[M * N * image + N * i + j] = h[M * N * image + N * i + j] =
      g[M * N * image + N * i + j];
}

__global__ void newXi(float* g, float* xi, float* h, float gam, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N * i + j] = -xi[N * i + j];
  xi[N * i + j] = h[N * i + j] = g[N * i + j] + gam * h[N * i + j];
}

__global__ void
newXi(float* g, float* xi, float* h, float gam, long N, long M, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[M * N * image + N * i + j] = -xi[M * N * image + N * i + j];

  xi[M * N * image + N * i + j] = h[M * N * image + N * i + j] =
      g[M * N * image + N * i + j] + gam * h[M * N * image + N * i + j];
}

__global__ void restartDPhi(float* dphi, float* dChi2, float* dH, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N * i + j] = 0.0f;
  dChi2[N * i + j] = 0.0f;
  dH[N * i + j] = 0.0f;
}

__global__ void DChi2_SharedMemory(float* noise,
                                   float* dChi2,
                                   cufftComplex* Vr,
                                   double3* UVW,
                                   float* w,
                                   long N,
                                   long numVisibilities,
                                   float noise_cut,
                                   float ref_xobs,
                                   float ref_yobs,
                                   float phs_xobs,
                                   float phs_yobs,
                                   double DELTAX,
                                   double DELTAY,
                                   float antenna_diameter,
                                   float pb_factor,
                                   float pb_cutoff,
                                   float freq,
                                   int primary_beam,
                                   bool normalize) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ double s_array[];

  int x0 = phs_xobs;

  int y0 = phs_yobs;
  double x = (j - x0) * DELTAX * RPDEG_D;
  double y = (i - y0) * DELTAY * RPDEG_D;
  double z = sqrtf(1.0 - x * x - y * y);

  float Ukv, Vkv, Wkv, cosk, sink, atten;

  double* u_shared = s_array;
  double* v_shared = (double*)&u_shared[numVisibilities];
  double* w_shared = (double*)&v_shared[numVisibilities];
  float* weight_shared = (float*)&w_shared[numVisibilities];
  cufftComplex* Vr_shared = (cufftComplex*)&weight_shared[numVisibilities];
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int v = 0; v < numVisibilities; v++) {
      u_shared[v] = UVW[v].x;
      v_shared[v] = UVW[v].y;
      w_shared[v] = UVW[v].z;
      weight_shared[v] = w[v];
      Vr_shared[v] = Vr[v];
      printf("u: %f, v:%f, weight: %f, real: %f, imag: %f\n", u_shared[v],
             v_shared[v], w_shared[v], Vr_shared[v].x, Vr_shared[v].y);
    }
  }
  cg::sync(cta);

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                      ref_yobs, DELTAX, DELTAY, primary_beam);

  float dchi2 = 0.0f;
  if (noise[N * i + j] < noise_cut) {
    for (int v = 0; v < numVisibilities; v++) {
      Ukv = x * u_shared[v];
      Vkv = y * v_shared[v];
      Wkv = (z - 1.0) * w_shared[v];
#if (__CUDA_ARCH__ >= 300)
      sincospif(2.0 * (Ukv + Vkv), &sink, &cosk);
#else
      cosk = cospif(2.0 * (Ukv + Vkv + Wkv));
      sink = sinpif(2.0 * (Ukv + Vkv + Wkv));
#endif
      dchi2 += weight_shared[v] *
               ((Vr_shared[v].x * cosk) + (Vr_shared[v].y * sink));
    }

    dchi2 *= atten;

    if (normalize)
      dchi2 /= numVisibilities;

    dChi2[N * i + j] = -1.0f * dchi2;
  }
}

__global__ void DChi2(float* noise,
                      float* dChi2,
                      cufftComplex* Vr,
                      double3* UVW,
                      float* w,
                      long N,
                      long numVisibilities,
                      float fg_scale,
                      float noise_cut,
                      float ref_xobs,
                      float ref_yobs,
                      float phs_xobs,
                      float phs_yobs,
                      double DELTAX,
                      double DELTAY,
                      float antenna_diameter,
                      float pb_factor,
                      float pb_cutoff,
                      float freq,
                      int primary_beam,
                      bool normalize) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = phs_xobs;
  int y0 = phs_yobs;
  double x = (j - x0) * DELTAX * RPDEG_D;
  double y = (i - y0) * DELTAY * RPDEG_D;
  double z = sqrtf(1 - x * x - y * y);

  float Ukv, Vkv, Wkv, cosk, sink, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                      ref_yobs, DELTAX, DELTAY, primary_beam);

  float dchi2 = 0.0f;
  if (noise[N * i + j] < noise_cut) {
    for (int v = 0; v < numVisibilities; v++) {
      Ukv = x * UVW[v].x;
      Vkv = y * UVW[v].y;
      Wkv = (z - 1.0) * UVW[v].z;

#if (__CUDA_ARCH__ >= 300)
      sincospif(2.0 * (Ukv + Vkv + Wkv), &sink, &cosk);
#else
      cosk = cospif(2.0 * (Ukv + Vkv + Wkv));
      sink = sinpif(2.0 * (Ukv + Vkv + Wkv));
#endif
      dchi2 += w[v] * ((Vr[v].x * cosk) + (Vr[v].y * sink));
    }

    dchi2 *= fg_scale * atten;

    if (normalize)
      dchi2 /= numVisibilities;

    dChi2[N * i + j] = -1.0f * dchi2;
  }
}

__global__ void DChi2(float* noise,
                      float* gcf,
                      float* dChi2,
                      cufftComplex* Vr,
                      double3* UVW,
                      float* w,
                      long N,
                      long numVisibilities,
                      float fg_scale,
                      float noise_cut,
                      float ref_xobs,
                      float ref_yobs,
                      float phs_xobs,
                      float phs_yobs,
                      double DELTAX,
                      double DELTAY,
                      float antenna_diameter,
                      float pb_factor,
                      float pb_cutoff,
                      float freq,
                      int primary_beam,
                      bool normalize) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = phs_xobs;
  int y0 = phs_yobs;
  double x = (j - x0) * DELTAX * RPDEG_D;
  double y = (i - y0) * DELTAY * RPDEG_D;
  double z = sqrtf(1 - x * x - y * y);

  float Ukv, Vkv, Wkv, cosk, sink, atten, gcf_i;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                      ref_yobs, DELTAX, DELTAY, primary_beam);
  gcf_i = gcf[N * i + j];
  float dchi2 = 0.0f;
  if (noise[N * i + j] < noise_cut) {
    for (int v = 0; v < numVisibilities; v++) {
      Ukv = x * UVW[v].x;
      Vkv = y * UVW[v].y;
      Wkv = (z - 1.0) * UVW[v].z;
#if (__CUDA_ARCH__ >= 300)
      sincospif(2.0 * (Ukv + Vkv + Wkv), &sink, &cosk);
#else
      cosk = cospif(2.0 * (Ukv + Vkv + Wkv));
      sink = sinpif(2.0 * (Ukv + Vkv + Wkv));
#endif
      dchi2 += w[v] * ((Vr[v].x * cosk) + (Vr[v].y * sink));
    }

    dchi2 *= fg_scale * atten * gcf_i;

    if (normalize)
      dchi2 /= numVisibilities;

    dChi2[N * i + j] = -1.0f * dchi2;
  }
}

__global__ void AddToDPhi(float* dphi, float* dgi, long N, long M, int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N * M * index + N * i + j] += dgi[N * i + j];
}

__global__ void substraction(float* x,
                             cufftComplex* xc,
                             float* gc,
                             float lambda,
                             long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  x[N * i + j] = xc[N * i + j].x - lambda * gc[N * i + j];
}

__global__ void projection(float* px, float* x, float MINPIX, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (INFINITY < x[N * i + j]) {
    px[N * i + j] = INFINITY;
  } else {
    px[N * i + j] = x[N * i + j];
  }

  if (MINPIX > px[N * i + j]) {
    px[N * i + j] = MINPIX;
  } else {
    px[N * i + j] = px[N * i + j];
  }
}

__global__ void normVectorCalculation(float* normVector, float* gc, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  normVector[N * i + j] = gc[N * i + j] * gc[N * i + j];
}

__global__ void copyImage(cufftComplex* p, float* device_xt, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  p[N * i + j].x = device_xt[N * i + j];
}

__global__ void calculateInu(cufftComplex* I_nu,
                             float* I,
                             float nu,
                             float nu_0,
                             float MINPIX,
                             float eta,
                             long N,
                             long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, nudiv_pow_alpha, nudiv;

  nudiv = nu / nu_0;

  I_nu_0 = I[N * i + j];
  alpha = I[M * N + N * i + j];

  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu[N * i + j].x = I_nu_0 * nudiv_pow_alpha;

  if (I_nu[N * i + j].x < -1.0f * eta * MINPIX) {
    I_nu[N * i + j].x = -1.0f * eta * MINPIX;
  }

  I_nu[N * i + j].y = 0.0f;
}

__global__ void DChi2_total_alpha(float* noise,
                                  float* dchi2_total,
                                  float* dchi2,
                                  float* I,
                                  float nu,
                                  float nu_0,
                                  float noise_cut,
                                  float fg_scale,
                                  float threshold,
                                  long N,
                                  long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, dI_nu_0;
  float nudiv = nu / nu_0;

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];

  dI_nu_0 = powf(nudiv, alpha);
  dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

  if (noise[N * i + j] < noise_cut) {
    if (I_nu_0 > threshold) {
      dchi2_total[N * M + N * i + j] += dchi2[N * i + j] * dalpha;
    } else {
      dchi2_total[N * M + N * i + j] += 0.0f;
    }
  }
}

__global__ void DChi2_total_I_nu_0(float* noise,
                                   float* dchi2_total,
                                   float* dchi2,
                                   float* I,
                                   float nu,
                                   float nu_0,
                                   float noise_cut,
                                   float threshold,
                                   long N,
                                   long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dI_nu_0;
  float nudiv = nu / nu_0;

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];

  dI_nu_0 = powf(nudiv, alpha);
  // dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

  if (noise[N * i + j] < noise_cut)
    dchi2_total[N * i + j] += dchi2[N * i + j] * dI_nu_0;
}

__global__ void chainRule2I(float* chain,
                            float* noise,
                            float* I,
                            float nu,
                            float nu_0,
                            float noise_cut,
                            float fg_scale,
                            long N,
                            long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, dI_nu_0;
  float nudiv = nu / nu_0;

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];

  dI_nu_0 = powf(nudiv, alpha);
  dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

  chain[N * i + j] = dI_nu_0;
  chain[N * M + N * i + j] = dalpha;
}

__global__ void DChi2_2I(float* noise,
                         float* chain,
                         float* I,
                         float* dchi2,
                         float* dchi2_total,
                         float threshold,
                         float noise_cut,
                         int image,
                         long N,
                         long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (noise[N * i + j] < noise_cut && image) {
    if (I[N * i + j] > threshold) {
      dchi2_total[N * i + j] += dchi2[N * i + j] * chain[N * M + N * i + j];
    } else {
      dchi2_total[N * i + j] += 0.0f;
    }

  } else if (noise[N * i + j] < noise_cut) {
    dchi2_total[N * i + j] += dchi2[N * i + j] * chain[N * i + j];
  }
}

__global__ void I_nu_0_Noise(float* noise_I,
                             float* images,
                             float* noise,
                             float noise_cut,
                             float nu,
                             float nu_0,
                             float* w,
                             float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             float sum_weights,
                             long N,
                             long M,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float alpha, nudiv, nudiv_pow_alpha, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, 2.0f * alpha);

  if (noise[N * i + j] < noise_cut) {
    noise_I[N * i + j] += atten * atten * sum_weights * nudiv_pow_alpha;
  } else {
    noise_I[N * i + j] = 0.0f;
  }
}

__global__ void alpha_Noise(float* noise_I,
                            float* images,
                            float nu,
                            float nu_0,
                            float* w,
                            double3* UVW,
                            cufftComplex* Vr,
                            float* noise,
                            float noise_cut,
                            double DELTAX,
                            double DELTAY,
                            float xobs,
                            float yobs,
                            float antenna_diameter,
                            float pb_factor,
                            float pb_cutoff,
                            long numVisibilities,
                            long N,
                            long M,
                            int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, Ukv, Vkv, cosk,
      sink, x, y, dchi2, sum_noise, atten;
  int x0, y0;

  x0 = xobs;
  y0 = yobs;
  x = (j - x0) * DELTAX * RPDEG_D;
  y = (i - y0) * DELTAY * RPDEG_D;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  I_nu_0 = images[N * i + j];
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu = I_nu_0 * nudiv_pow_alpha;
  log_nu = logf(nudiv);

  sum_noise = 0.0f;
  if (noise[N * i + j] < noise_cut) {
    for (int v = 0; v < numVisibilities; v++) {
      Ukv = x * UVW[v].x;
      Vkv = y * UVW[v].y;
#if (__CUDA_ARCH__ >= 300)
      sincospif(2.0 * (Ukv + Vkv), &sink, &cosk);
#else
      cosk = cospif(2.0 * (Ukv + Vkv));
      sink = sinpif(2.0 * (Ukv + Vkv));
#endif
      dchi2 = ((Vr[v].x * cosk) - (Vr[v].y * sink));
      sum_noise += w[v] * (atten * I_nu + dchi2);
    }
    if (sum_noise <= 0.0f)
      noise_I[N * M + N * i + j] += 0.0f;
    else
      noise_I[N * M + N * i + j] += log_nu * log_nu * atten * I_nu * sum_noise;
  } else {
    noise_I[N * M + N * i + j] = 0.0f;
  }
}

__global__ void noise_reduction(float* noise_I, long N, long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (noise_I[N * i + j] > 0.0f)
    noise_I[N * i + j] = 1.0f / sqrt(noise_I[N * i + j]);
  else
    noise_I[N * i + j] = 0.0f;

  if (noise_I[N * M + N * i + j] > 0.0f)
    noise_I[N * M + N * i + j] = 1.0f / sqrt(noise_I[N * M + N * i + j]);
  else
    noise_I[N * M + N * i + j] = 0.0f;
}

__host__ float simulate(float* I, VirtualImageProcessor* ip, float fg_scale) {
  cudaSetDevice(firstgpu);

  float resultchi2 = 0.0f;

  ip->clipWNoise(I);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus) \
    reduction(+ : resultchi2)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        float result = 0.0;
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);

        ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, I,
                         datasets[d].fields[f].nu[i]);

        ip->apply_beam(vars_gpu[gpu_idx].device_I_nu,
                       datasets[d].antennas[0].antenna_diameter,
                       datasets[d].antennas[0].pb_factor,
                       datasets[d].antennas[0].pb_cutoff,
                       datasets[d].fields[f].ref_xobs_pix,
                       datasets[d].fields[f].ref_yobs_pix,
                       datasets[d].fields[f].nu[i],
                       datasets[d].antennas[0].primary_beam, fg_scale);

        if (NULL != ip->getCKernel()) {
          apply_GCF<<<numBlocksNN, threadsPerBlockNN>>>(
              vars_gpu[gpu_idx].device_I_nu, ip->getCKernel()->getGCFGPU(), N);
          checkCudaErrors(cudaDeviceSynchronize());
        }
        // FFT 2D
        FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
              vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, false);

        // PHASE_ROTATE
        phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(
            vars_gpu[gpu_idx].device_V, M, N,
            datasets[d].fields[f].phs_xobs_pix,
            datasets[d].fields[f].phs_yobs_pix);
        checkCudaErrors(cudaDeviceSynchronize());

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
                                         sizeof(float) * max_number_vis));

              // TODO: Here we could just use vis_mod and see what happens
              // Use always bilinear interpolation since we don't have
              // degridding yet
              vis_mod<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  datasets[d].fields[f].device_visibilities[i][s].Vm,
                  vars_gpu[gpu_idx].device_V,
                  datasets[d].fields[f].device_visibilities[i][s].uvw,
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  deltau, deltav,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  N);
              checkCudaErrors(cudaDeviceSynchronize());

              // RESIDUAL CALCULATION
              residual<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  datasets[d].fields[f].device_visibilities[i][s].Vr,
                  datasets[d].fields[f].device_visibilities[i][s].Vm,
                  datasets[d].fields[f].device_visibilities[i][s].Vo,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              ////chi2 VECTOR
              chi2Vector<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].device_visibilities[i][s].Vr,
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              result = deviceReduce<float>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);
              // REDUCTIONS
              // chi2
              resultchi2 +=
                  result /
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s];
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);

  return 0.5f * resultchi2;
};

__host__ float chi2(float* I,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  cudaSetDevice(firstgpu);

  float reduced_chi2 = 0.0f;

  ip->clipWNoise(I);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus) \
    reduction(+ : reduced_chi2)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        float result = 0.0;
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);

        ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, I,
                         datasets[d].fields[f].nu[i]);

        ip->apply_beam(vars_gpu[gpu_idx].device_I_nu,
                       datasets[d].antennas[0].antenna_diameter,
                       datasets[d].antennas[0].pb_factor,
                       datasets[d].antennas[0].pb_cutoff,
                       datasets[d].fields[f].ref_xobs_pix,
                       datasets[d].fields[f].ref_yobs_pix,
                       datasets[d].fields[f].nu[i],
                       datasets[d].antennas[0].primary_beam, fg_scale);

        if (NULL != ip->getCKernel()) {
          apply_GCF<<<numBlocksNN, threadsPerBlockNN>>>(
              vars_gpu[gpu_idx].device_I_nu, ip->getCKernel()->getGCFGPU(), N);
          checkCudaErrors(cudaDeviceSynchronize());
        }
        // FFT 2D
        FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
              vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, false);

        // PHASE_ROTATE
        phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(
            vars_gpu[gpu_idx].device_V, M, N,
            datasets[d].fields[f].phs_xobs_pix,
            datasets[d].fields[f].phs_yobs_pix);
        checkCudaErrors(cudaDeviceSynchronize());

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
                                         sizeof(float) * max_number_vis));

              // Use always bilinear interpolation since we don't have
              // degridding yet
              vis_mod<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  datasets[d].fields[f].device_visibilities[i][s].Vm,
                  vars_gpu[gpu_idx].device_V,
                  datasets[d].fields[f].device_visibilities[i][s].uvw,
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  deltau, deltav,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  N);
              checkCudaErrors(cudaDeviceSynchronize());

              // RESIDUAL CALCULATION
              residual<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  datasets[d].fields[f].device_visibilities[i][s].Vr,
                  datasets[d].fields[f].device_visibilities[i][s].Vm,
                  datasets[d].fields[f].device_visibilities[i][s].Vo,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              ////chi2 VECTOR
              chi2Vector<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].device_visibilities[i][s].Vr,
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              result = deviceReduce<float>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);
              // REDUCTIONS
              // chi2
              if (normalize)
                result /=
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s];

              reduced_chi2 += result;
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);

  return 0.5f * reduced_chi2;
};

__host__ void dchi2(float* I,
                    float* dxi2,
                    float* result_dchi2,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_dchi2, 0,
                                         sizeof(float) * M * N));
              // size_t shared_memory;
              // shared_memory =
              // 3*fields[f].numVisibilitiesPerFreq[i]*sizeof(float) +
              // fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex);
              if (NULL != ip->getCKernel()) {
                DChi2<<<numBlocksNN, threadsPerBlockNN>>>(
                    device_noise_image, ip->getCKernel()->getGCFGPU(),
                    vars_gpu[gpu_idx].device_dchi2,
                    datasets[d].fields[f].device_visibilities[i][s].Vr,
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].weight, N,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    fg_scale, noise_cut, datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].fields[f].phs_xobs_pix,
                    datasets[d].fields[f].phs_yobs_pix, DELTAX, DELTAY,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].nu[i],
                    datasets[d].antennas[0].primary_beam, normalize);
              } else {
                DChi2<<<numBlocksNN, threadsPerBlockNN>>>(
                    device_noise_image, vars_gpu[gpu_idx].device_dchi2,
                    datasets[d].fields[f].device_visibilities[i][s].Vr,
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].weight, N,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    fg_scale, noise_cut, datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].fields[f].phs_xobs_pix,
                    datasets[d].fields[f].phs_yobs_pix, DELTAX, DELTAY,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].nu[i],
                    datasets[d].antennas[0].primary_beam, normalize);
              }
              // DChi2<<<numBlocksNN, threadsPerBlockNN,
              // shared_memory>>>(device_noise_image,
              // vars_gpu[gpu_idx].device_dchi2,
              // fields[f].device_visibilities[i][s].Vr,
              // fields[f].device_visibilities[i][s].uvw,
              // fields[f].device_visibilities[i][s].weight, N,
              // fields[f].numVisibilitiesPerFreqPerStoke[i][s],
              // noise_cut, fields[f].ref_xobs, fields[f].ref_yobs,
              // fields[f].phs_xobs, fields[f].phs_yobs, DELTAX, DELTAY,
              // antenna_diameter, pb_factor, pb_cutoff, fields[f].nu[i]);
              checkCudaErrors(cudaDeviceSynchronize());

#pragma omp critical
              {
                if (flag_opt % 2 == 0)
                  DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                      device_noise_image, result_dchi2,
                      vars_gpu[gpu_idx].device_dchi2, I,
                      datasets[d].fields[f].nu[i], nu_0, noise_cut, threshold,
                      N, M);
                else
                  DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                      device_noise_image, result_dchi2,
                      vars_gpu[gpu_idx].device_dchi2, I,
                      datasets[d].fields[f].nu[i], nu_0, noise_cut, fg_scale,
                      threshold, N, M);
                checkCudaErrors(cudaDeviceSynchronize());
              }
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);
};

__host__ void linkAddToDPhi(float* dphi, float* dgi, int index) {
  cudaSetDevice(firstgpu);
  AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(dphi, dgi, N, M, index);
  checkCudaErrors(cudaDeviceSynchronize());
};

__host__ void defaultNewP(float* p, float* xi, float xmin, int image) {
  newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M,
                                                       image);
};

__host__ void particularNewP(float* p, float* xi, float xmin, int image) {
  newP<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M,
                                           initial_values[image], eta, image);
};

__host__ void defaultEvaluateXt(float* xt,
                                float* pcom,
                                float* xicom,
                                float x,
                                int image) {
  evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(xt, pcom, xicom, x,
                                                             N, M, image);
};

__host__ void particularEvaluateXt(float* xt,
                                   float* pcom,
                                   float* xicom,
                                   float x,
                                   int image) {
  evaluateXt<<<numBlocksNN, threadsPerBlockNN>>>(
      xt, pcom, xicom, x, N, M, initial_values[image], eta, image);
};

__host__ void linkClipWNoise2I(float* I) {
  clip2IWNoise<<<numBlocksNN, threadsPerBlockNN>>>(
      device_noise_image, I, N, M, noise_cut, initial_values[0],
      initial_values[1], eta, threshold, flag_opt);
  checkCudaErrors(cudaDeviceSynchronize());
};

__host__ void linkApplyBeam2I(cufftComplex* image,
                              float antenna_diameter,
                              float pb_factor,
                              float pb_cutoff,
                              float xobs,
                              float yobs,
                              float freq,
                              int primary_beam,
                              float fg_scale) {
  apply_beam2I<<<numBlocksNN, threadsPerBlockNN>>>(
      antenna_diameter, pb_factor, pb_cutoff, image, N, xobs, yobs, fg_scale,
      freq, DELTAX, DELTAY, primary_beam);
  checkCudaErrors(cudaDeviceSynchronize());
};

__host__ void linkCalculateInu2I(cufftComplex* image, float* I, float freq) {
  calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(
      image, I, freq, nu_0, initial_values[0], eta, N, M);
  checkCudaErrors(cudaDeviceSynchronize());
};

__host__ void linkChain2I(float* chain, float freq, float* I, float fg_scale) {
  chainRule2I<<<numBlocksNN, threadsPerBlockNN>>>(
      chain, device_noise_image, I, freq, nu_0, noise_cut, fg_scale, N, M);
  checkCudaErrors(cudaDeviceSynchronize());
};

__host__ void normalizeImage(float* image, float normalization_factor) {
  normalizeImageKernel<<<numBlocksNN, threadsPerBlockNN>>>(
      image, normalization_factor, N);
  checkCudaErrors(cudaDeviceSynchronize());
};
__host__ float L1Norm(float* I,
                      float* ds,
                      float penalization_factor,
                      float epsilon,
                      int mod,
                      int order,
                      int index,
                      int iter) {
  cudaSetDevice(firstgpu);

  float resultL1norm = 0.0f;
  if (iter > 0 && penalization_factor) {
    L1Vector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N,
                                                 M, epsilon, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultL1norm = deviceReduce<float>(
        ds, M * N, threadsPerBlockNN.x * threadsPerBlockNN.y);
  }

  return resultL1norm;
};

__host__ void DL1Norm(float* I,
                      float* dgi,
                      float penalization_factor,
                      float epsilon,
                      int mod,
                      int order,
                      int index,
                      int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DL1NormK<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, device_noise_image, epsilon, noise_cut, penalization_factor,
          N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float GL1NormK(float* I,
                        float* prior,
                        float* ds,
                        float penalization_factor,
                        float epsilon_a,
                        float epsilon_b,
                        int mod,
                        int order,
                        int index,
                        int iter) {
  cudaSetDevice(firstgpu);

  float resultL1norm = 0.0f;
  if (iter > 0 && penalization_factor) {
    GL1Vector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I,
                                                  prior, N, M, epsilon_a,
                                                  epsilon_b, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultL1norm = deviceReduce<float>(
        ds, M * N, threadsPerBlockNN.x * threadsPerBlockNN.y);
  }

  return resultL1norm;
};

__host__ void DGL1Norm(float* I,
                       float* prior,
                       float* dgi,
                       float penalization_factor,
                       float epsilon_a,
                       float epsilon_b,
                       int mod,
                       int order,
                       int index,
                       int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DGL1NormK<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, prior, device_noise_image, epsilon_a, epsilon_b, noise_cut,
          penalization_factor, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float SEntropy(float* I,
                        float* ds,
                        float prior_value,
                        float eta,
                        float penalization_factor,
                        int mod,
                        int order,
                        int index,
                        int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    SVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, N, M, noise_cut, prior_value, eta, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DEntropy(float* I,
                       float* dgi,
                       float prior_value,
                       float eta,
                       float penalization_factor,
                       int mod,
                       int order,
                       int index,
                       int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DS<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                             noise_cut, penalization_factor,
                                             prior_value, eta, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float SGEntropy(float* I,
                         float* ds,
                         float* prior,
                         float eta,
                         float penalization_factor,
                         int mod,
                         int order,
                         int index,
                         int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    SGVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, N, M, noise_cut, prior, eta, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DGEntropy(float* I,
                        float* dgi,
                        float* prior,
                        float eta,
                        float penalization_factor,
                        int mod,
                        int order,
                        int index,
                        int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DSG<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                              noise_cut, penalization_factor,
                                              prior, eta, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float laplacian(float* I,
                         float* ds,
                         float penalization_factor,
                         int mod,
                         int order,
                         int imageIndex,
                         int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    LVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M,
                                                noise_cut, imageIndex);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DLaplacian(float* I,
                         float* dgi,
                         float penalization_factor,
                         float mod,
                         float order,
                         float index,
                         int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DL<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                             noise_cut, penalization_factor, N,
                                             M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float quadraticP(float* I,
                          float* ds,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    QPVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N,
                                                 M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DQuadraticP(float* I,
                          float* dgi,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DQ<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                             noise_cut, penalization_factor, N,
                                             M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float totalvariation(float* I,
                              float* ds,
                              float epsilon,
                              float penalization_factor,
                              int mod,
                              int order,
                              int index,
                              int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    TVVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, epsilon, N, M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DTVariation(float* I,
                          float* dgi,
                          float epsilon,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DTV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                              epsilon, noise_cut,
                                              penalization_factor, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ float TotalSquaredVariation(float* I,
                                     float* ds,
                                     float penalization_factor,
                                     int mod,
                                     int order,
                                     int index,
                                     int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    TSVVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N,
                                                  M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DTSVariation(float* I,
                           float* dgi,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt % 2 == index) {
      DTSV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                               noise_cut, penalization_factor,
                                               N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
};

__host__ void calculateErrors(Image* image) {
  float* errors = image->getErrorImage();

  cudaSetDevice(firstgpu);

  checkCudaErrors(cudaMalloc((void**)&errors,
                             sizeof(float) * M * N * image->getImageCount()));
  checkCudaErrors(
      cudaMemset(errors, 0, sizeof(float) * M * N * image->getImageCount()));
  float sum_weights;
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
#pragma omp parallel for private(sum_weights) num_threads(num_gpus) \
    schedule(static, 1)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreq[i] > 0) {
              sum_weights = deviceReduce<float>(
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);

#pragma omp critical
              {
                I_nu_0_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), device_noise_image, noise_cut,
                    datasets[d].fields[f].nu[i], nu_0,
                    datasets[d].fields[f].device_visibilities[i][s].weight,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix, DELTAX, DELTAY,
                    sum_weights, N, M, datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());
                alpha_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), datasets[d].fields[f].nu[i],
                    nu_0,
                    datasets[d].fields[f].device_visibilities[i][s].weight,
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].Vr,
                    device_noise_image, noise_cut, DELTAX, DELTAY,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    N, M, datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());
              }
            }
          }
        }
      }
    }
  }

  noise_reduction<<<numBlocksNN, threadsPerBlockNN>>>(errors, N, M);
  checkCudaErrors(cudaDeviceSynchronize());

  image->setErrorImage(errors);
}
