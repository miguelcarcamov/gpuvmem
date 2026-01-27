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
#include "pillBox2D.cuh"

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
  flags.Bool(variables.normalize, 'l', "normalize",
             "Normalize chi-squared by effective number of samples", "Flags");
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

// Device function: Proper fftshift using quadrant swapping
// Works correctly for both even and odd dimensions
// For even N: ceil(N/2) = N/2, swaps first N/2 with second N/2
// For odd N: ceil(N/2) = (N+1)/2, swaps first (N+1)/2 with last (N-1)/2
// fftshift swaps: Q1(top-left)↔Q3(bottom-right), Q2(top-right)↔Q4(bottom-left)
__device__ __forceinline__ void fftshift_swap(cufftComplex* data,
                                              int i,
                                              int j,
                                              int N1,
                                              int N2) {
  const int h2 = (N1 + 1) / 2;  // ceil(N1/2) for fftshift
  const int w2 = (N2 + 1) / 2;  // ceil(N2/2) for fftshift

  // Process only elements in first half to avoid double-swapping
  if (i < h2) {
    int i2 = i + (N1 - h2);  // Target row: shift by (N1 - ceil(N1/2))
    // For even: N1 - N1/2 = N1/2, for odd: N1 - (N1+1)/2 = (N1-1)/2

    if (j < w2) {
      // Top-left quadrant (Q1): swap with bottom-right (Q3)
      int j2 = j + (N2 - w2);  // Target column: shift right by (N2 - w2)
      if (i2 < N1 && j2 < N2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    } else if (j >= w2 && j < N2) {
      // Top-right quadrant (Q2): swap with bottom-left (Q4)
      int j2 = j - w2;  // Target column: shift left by w2
      if (i2 < N1 && j2 >= 0 && j2 < w2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    }
  }
}

// Device function: Proper ifftshift using quadrant swapping
// Works correctly for both even and odd dimensions
// For even N: floor(N/2) = N/2, swaps first N/2 with second N/2 (same as
// fftshift) For odd N: floor(N/2) = (N-1)/2, swaps first (N-1)/2 with last
// (N+1)/2 ifftshift swaps: Q1(top-left)↔Q3(bottom-right),
// Q2(top-right)↔Q4(bottom-left)
__device__ __forceinline__ void ifftshift_swap(cufftComplex* data,
                                               int i,
                                               int j,
                                               int N1,
                                               int N2) {
  const int h2 = N1 / 2;  // floor(N1/2) for ifftshift
  const int w2 = N2 / 2;  // floor(N2/2) for ifftshift

  // Process only elements in first half to avoid double-swapping
  if (i < h2) {
    int i2 = i + (N1 - h2);  // Target row: shift by (N1 - floor(N1/2))
    // For even: N1 - N1/2 = N1/2, for odd: N1 - (N1-1)/2 = (N1+1)/2

    if (j < w2) {
      // Top-left quadrant (Q1): swap with bottom-right (Q3)
      int j2 = j + (N2 - w2);  // Target column: shift right by (N2 - w2)
      if (i2 < N1 && j2 < N2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    } else if (j >= w2 && j < N2) {
      // Top-right quadrant (Q2): swap with bottom-left (Q4)
      // For even N: w2 = N/2, so Q2 has j in [w2, N-1] = [N/2, N-1] (N/2
      // elements)
      //            and Q4 has j in [0, w2-1] = [0, N/2-1] (N/2 elements)
      //            They match perfectly, so j2 = j - w2 maps all of Q2 to Q4
      // For odd N: Q2 has ceil(N/2) elements, Q4 has floor(N/2) elements
      //            Only first floor(N/2) elements of Q2 swap with Q4
      bool should_swap = true;
      int j2 = j - w2;

      if (N2 % 2 != 0) {
        // Odd N: only first w2 elements of Q2 swap with Q4
        if (j >= w2 + w2) {
          // Last element of Q2 (for odd N) doesn't swap with Q4, skip it
          should_swap = false;
        }
      }

      if (should_swap && i2 < N1 && j2 >= 0 && j2 < w2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    }
  }
}

// Generic fftshift: uses quadrant swapping for all dimensions
// For even dimensions, fftshift and ifftshift are identical
// For odd dimensions, they differ by one sample
// Optimized fftshift: computes target index directly (more efficient for even
// dimensions) For even dimensions: shift by N/2, for odd: shift by ceil(N/2)
__global__ void fftshift_2D(cufftComplex* __restrict__ data, int N1, int N2) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N1 && j < N2) {
    // Compute target indices directly (avoids quadrant checking overhead)
    const int h2 = (N1 + 1) / 2;  // ceil(N1/2) for fftshift
    const int w2 = (N2 + 1) / 2;  // ceil(N2/2) for fftshift

    // Only process first half to avoid double-swapping
    if (i < h2) {
      int i2 = i + (N1 - h2);  // Target row

      if (j < w2) {
        // Q1 (top-left): swap with Q3 (bottom-right)
        int j2 = j + (N2 - w2);  // Target column
        if (i2 < N1 && j2 < N2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      } else if (j >= w2 && j < N2) {
        // Q2 (top-right): swap with Q4 (bottom-left)
        int j2 = j - w2;  // Target column
        if (i2 < N1 && j2 >= 0 && j2 < w2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      }
    }
  }
}

// Optimized ifftshift: computes target index directly (more efficient for even
// dimensions) For even dimensions: same as fftshift, for odd: shift by
// floor(N/2)
__global__ void ifftshift_2D(cufftComplex* __restrict__ data, int N1, int N2) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N1 && j < N2) {
    // Compute target indices directly (avoids quadrant checking overhead)
    const int h2 = N1 / 2;  // floor(N1/2) for ifftshift
    const int w2 = N2 / 2;  // floor(N2/2) for ifftshift

    // Only process first half to avoid double-swapping
    if (i < h2) {
      int i2 = i + (N1 - h2);  // Target row

      if (j < w2) {
        // Q1 (top-left): swap with Q3 (bottom-right)
        int j2 = j + (N2 - w2);  // Target column
        if (i2 < N1 && j2 < N2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      } else if (j >= w2 && j < N2) {
        // Q2 (top-right): swap with Q4 (bottom-left)
        // For even N: all of Q2 swaps; for odd N: only first w2 elements
        bool should_swap = true;
        int j2 = j - w2;

        if (N2 % 2 != 0 && j >= w2 + w2) {
          // Odd N: skip last element of Q2
          should_swap = false;
        }

        if (should_swap && i2 < N1 && j2 >= 0 && j2 < w2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      }
    }
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

/**
 * Grids irregular visibilities from a measurement set onto a regular grid.
 *
 * This function performs convolution gridding of visibility data:
 * 1. Creates Hermitian symmetric visibilities (MS only contains half the UV
 * plane)
 * 2. Converts UVW coordinates from meters to lambda units
 * 3. Grids visibilities using a convolution kernel (with 0.5 weight factor)
 * 4. Normalizes gridded visibilities and calculates effective weights
 * 5. Extracts non-zero gridded visibilities back to sparse format
 *
 * Note: Measurement sets store only half the UV plane data. We process each
 * visibility twice (original + Hermitian conjugate) to fill the full grid,
 * with weights multiplied by 0.5 to account for the duplication.
 *
 * @param fields Vector of Field structures containing visibility data
 * @param data MSData structure with measurement set metadata
 * @param deltau Grid spacing in u direction (lambda units)
 * @param deltav Grid spacing in v direction (lambda units)
 * @param M Grid size in v direction (rows)
 * @param N Grid size in u direction (columns)
 * @param ckernel Convolution kernel for gridding
 * @param gridding Number of OpenMP threads for parallel gridding
 */
__host__ void do_gridding(std::vector<Field>& fields,
                          MSData* data,
                          double deltau,
                          double deltav,
                          int M,
                          int N,
                          CKernel* ckernel,
                          int gridding) {
  // ========================================================================
  // Initialize grid arrays (reused for each frequency/stokes combination)
  // ========================================================================
  std::vector<float> g_weights(M * N);  // Accumulated weights
  std::vector<float> g_weights_aux(
      M * N);  // Sum of weight^2 (for effective weight calc)
  std::vector<cufftComplex> g_Vo(M * N);  // Gridded visibilities
  std::vector<double3> g_uvw(M * N);      // Grid UVW coordinates in meters

  // Zero-initialization constants
  cufftComplex complex_zero = floatComplexZero();
  double3 double3_zero = {0.0, 0.0, 0.0};

  // Track maximum number of visibilities across all channels/stokes
  int local_max = 0;
  int max = 0;

  // ========================================================================
  // Private variables for parallel gridding loop
  // ========================================================================
  double j_fp, k_fp;              // Floating-point grid coordinates
  int j, k;                       // Integer grid pixel indices
  double grid_pos_x, grid_pos_y;  // Grid positions in lambda units
  double3 uvw;                    // UVW coordinates
  float w;                        // Visibility weight
  cufftComplex Vo;                // Visibility value
  int shifted_j, shifted_k;    // Shifted grid indices (for kernel convolution)
  int kernel_i, kernel_j;      // Kernel array indices
  int visCounterPerFreq = 0;   // Counter for visibilities per frequency
  float ckernel_result = 1.0;  // Kernel value at current position
  // ========================================================================
  // Main loop: Process each field, frequency, and stokes parameter
  // ========================================================================
  // Pre-calculate constants that don't change per frequency/stokes
  double center_j = floor(N / 2.0);
  double center_k = floor(M / 2.0);
  int support_x = ckernel->getSupportX();
  int support_y = ckernel->getSupportY();

  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      visCounterPerFreq = 0;

      // Pre-calculate frequency-dependent constants (same for all stokes)
      float freq = fields[f].nu[i];
      float lambda = freq_to_wavelength(freq);

      for (int s = 0; s < data->nstokes; s++) {
        // ====================================================================
        // Step 1: Backup original visibility data before gridding
        // ====================================================================
        int num_visibilities = fields[f].numVisibilitiesPerFreqPerStoke[i][s];
        fields[f].backup_visibilities[i][s].uvw.resize(num_visibilities);
        fields[f].backup_visibilities[i][s].Vo.resize(num_visibilities);
        fields[f].backup_visibilities[i][s].weight.resize(num_visibilities);

        // Copy original data to backup
        fields[f].backup_visibilities[i][s].uvw.assign(
            fields[f].visibilities[i][s].uvw.begin(),
            fields[f].visibilities[i][s].uvw.end());
        fields[f].backup_visibilities[i][s].weight.assign(
            fields[f].visibilities[i][s].weight.begin(),
            fields[f].visibilities[i][s].weight.end());
        fields[f].backup_visibilities[i][s].Vo.assign(
            fields[f].visibilities[i][s].Vo.begin(),
            fields[f].visibilities[i][s].Vo.end());

        // ====================================================================
        // Step 2: Grid visibilities using convolution kernel
        // ====================================================================
        // For each visibility, grid both V(u,v) and its Hermitian conjugate
        // V(-u,-v) = V*(u,v) Measurement sets only contain half the UV plane
        // data, so we need both to fill the full grid. We multiply weights by
        // 0.5 to account for each visibility being gridded at two positions.
#pragma omp parallel for schedule(static) num_threads(gridding) shared(       \
        g_weights, g_weights_aux, g_Vo, freq, center_j, center_k, support_x,  \
            support_y) private(j_fp, k_fp, j, k, grid_pos_x, grid_pos_y, uvw, \
                                   w, Vo, shifted_j, shifted_k, kernel_i,     \
                                   kernel_j, ckernel_result)
        for (int z = 0; z < num_visibilities; z++) {
          // Load visibility data once (cache-friendly)
          uvw = fields[f].visibilities[i][s].uvw[z];
          w = fields[f].visibilities[i][s].weight[z] *
              0.5f;  // Half weight since gridded at two positions
          Vo = fields[f].visibilities[i][s].Vo[z];

          // ================================================================
          // Step 2a: Convert UVW coordinates from meters to lambda units
          // ================================================================
          double u_lambda = metres_to_lambda(uvw.x, freq);
          double v_lambda = metres_to_lambda(uvw.y, freq);

          // ================================================================
          // Step 2b: Grid both the original visibility and its Hermitian
          // conjugate
          // ================================================================
          // Loop over both: h=0 for original V(u,v), h=1 for Hermitian
          // V(-u,-v)=V*(u,v)
          for (int h = 0; h < 2; h++) {
            double u_pos = (h == 0) ? u_lambda : -u_lambda;
            double v_pos = (h == 0) ? v_lambda : -v_lambda;
            float Vo_imag = (h == 0) ? Vo.y : -Vo.y;  // Conjugate for Hermitian

            // Calculate grid pixel coordinates
            grid_pos_x = u_pos / deltau;
            grid_pos_y = v_pos / deltav;

            // Center the grid: center pixel is at floor(N/2) for both even and
            // odd N
            j_fp = grid_pos_x + center_j + 0.5;
            k_fp = grid_pos_y + center_k + 0.5;
            j = int(j_fp);
            k = int(k_fp);

            // ================================================================
            // Step 2c: Apply convolution kernel to grid this visibility
            // ================================================================
            for (int m = -support_y; m <= support_y; m++) {
              for (int n = -support_x; n <= support_x; n++) {
                shifted_j = j + n;
                shifted_k = k + m;
                kernel_j = n + support_x;
                kernel_i = m + support_y;

                if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
                    shifted_j < N && kernel_i >= 0 &&
                    kernel_i < ckernel->getm() && kernel_j >= 0 &&
                    kernel_j < ckernel->getn()) {
                  ckernel_result = ckernel->getKernelValue(kernel_i, kernel_j);
                  float ckernel_result_sq = ckernel_result * ckernel_result;
                  int grid_idx = N * shifted_k + shifted_j;

#pragma omp atomic
                  g_weights[grid_idx] += w * ckernel_result;

#pragma omp atomic
                  g_weights_aux[grid_idx] += w * ckernel_result_sq;

#pragma omp critical
                  {
                    g_Vo[grid_idx].x += w * Vo.x * ckernel_result;
                    g_Vo[grid_idx].y += w * Vo_imag * ckernel_result;
                  }
                }
              }
            }
          }
        }

        // ====================================================================
        // Step 3: Normalize gridded visibilities and calculate effective
        // weights
        // ====================================================================
        // Convert grid coordinates from lambda back to meters and normalize
#pragma omp parallel for schedule(static) \
    shared(g_weights, g_weights_aux, g_Vo, g_uvw, lambda, center_j, center_k)
        for (int grid_k = 0; grid_k < M; grid_k++) {
          for (int grid_j = 0; grid_j < N; grid_j++) {
            int grid_idx = N * grid_k + grid_j;

            // ================================================================
            // Step 3a: Calculate UVW coordinates in meters for this grid cell
            // ================================================================
            // Use same centering formula as forward calculation
            double u_lambdas = (grid_j - center_j) * deltau;
            double v_lambdas = (grid_k - center_k) * deltav;

            // Convert from lambda units back to meters using the frequency for
            // this channel
            double u_meters = u_lambdas * lambda;
            double v_meters = v_lambdas * lambda;

            g_uvw[grid_idx].x = u_meters;
            g_uvw[grid_idx].y = v_meters;

            // ================================================================
            // Step 3b: Normalize visibilities and calculate effective weights
            // ================================================================
            float ws = g_weights[grid_idx];  // Sum of weights: Σ(w * kernel)
            float aux_ws =
                g_weights_aux[grid_idx];  // Sum of weight^2: Σ(w * kernel^2)

            if (aux_ws != 0.0f && ws != 0.0f) {
              // Effective weight accounts for kernel convolution:
              //   weight_eff = (Σ w_i * k_i)^2 / Σ (w_i * k_i)^2
              // This gives the equivalent weight if all visibilities were at
              // the same point
              float weight_eff = ws * ws / aux_ws;

              // Normalize visibility: divide by accumulated weight sum
              // This gives the weighted average visibility at this grid point
              g_Vo[grid_idx].x /= ws;
              g_Vo[grid_idx].y /= ws;

              // Store effective weight
              g_weights[grid_idx] = weight_eff;
            } else {
              // No valid data at this grid point (no visibilities contributed)
              g_weights[grid_idx] = 0.0f;
              g_Vo[grid_idx].x = 0.0f;
              g_Vo[grid_idx].y = 0.0f;
            }
          }
        }

        // ====================================================================
        // Step 4: Extract non-zero gridded visibilities back to sparse format
        // ====================================================================
        // Count how many grid cells have valid (non-zero) visibilities
        int visCounter = 0;
#pragma omp parallel for schedule(static) shared(g_weights) \
    reduction(+ : visCounter)
        for (int grid_k = 0; grid_k < M; grid_k++) {
          for (int grid_j = 0; grid_j < N; grid_j++) {
            int grid_idx = N * grid_k + grid_j;
            if (g_weights[grid_idx] > 0.0f) {
              visCounter++;
            }
          }
        }

        // Resize output arrays to hold only non-zero visibilities
        fields[f].visibilities[i][s].uvw.resize(visCounter);
        fields[f].visibilities[i][s].Vo.resize(visCounter);
        fields[f].visibilities[i][s].Vm.resize(visCounter);
        fields[f].visibilities[i][s].weight.resize(visCounter);

        // Initialize Vm (model visibilities) to zero
        if (visCounter > 0) {
          memset(&fields[f].visibilities[i][s].Vm[0], 0,
                 visCounter * sizeof(cufftComplex));
        }

        // Copy non-zero gridded visibilities to output arrays
        int output_idx = 0;
        for (int grid_k = 0; grid_k < M; grid_k++) {
          for (int grid_j = 0; grid_j < N; grid_j++) {
            int grid_idx = N * grid_k + grid_j;
            float weight = g_weights[grid_idx];

            if (weight > 0.0f) {
              // Copy UVW coordinates (w is set to 0 for gridded data)
              fields[f].visibilities[i][s].uvw[output_idx].x =
                  g_uvw[grid_idx].x;
              fields[f].visibilities[i][s].uvw[output_idx].y =
                  g_uvw[grid_idx].y;
              fields[f].visibilities[i][s].uvw[output_idx].z = 0.0;

              // Copy normalized visibility
              fields[f].visibilities[i][s].Vo[output_idx] =
                  make_cuFloatComplex(g_Vo[grid_idx].x, g_Vo[grid_idx].y);

              // Copy effective weight
              fields[f].visibilities[i][s].weight[output_idx] = weight;

              output_idx++;
            }
          }
        }

        // ====================================================================
        // Step 5: Update visibility counts and backup
        // ====================================================================
        // Backup old count before updating
        fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s] =
            fields[f].numVisibilitiesPerFreqPerStoke[i][s];

        // Update to actual gridded visibility count
        fields[f].numVisibilitiesPerFreqPerStoke[i][s] = visCounter;
        if (visCounter > 0) {
          visCounterPerFreq += visCounter;
        }

        // ====================================================================
        // Step 6: Clear grid arrays for next stokes parameter
        // ====================================================================
        std::fill_n(g_weights_aux.begin(), M * N, 0.0f);
        std::fill_n(g_weights.begin(), M * N, 0.0f);
        std::fill_n(g_uvw.begin(), M * N, double3_zero);
        std::fill_n(g_Vo.begin(), M * N, complex_zero);
      }  // End stokes loop

      // Track maximum number of visibilities across all stokes for this
      // frequency
      local_max =
          *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i].begin(),
                            fields[f].numVisibilitiesPerFreqPerStoke[i].end());
      if (local_max > max) {
        max = local_max;
      }

      // Update total visibilities per frequency
      fields[f].backup_numVisibilitiesPerFreq[i] =
          fields[f].numVisibilitiesPerFreq[i];
      fields[f].numVisibilitiesPerFreq[i] = visCounterPerFreq;
    }  // End frequency loop
  }  // End field loop

  // Store global maximum across all fields/frequencies/stokes
  data->max_number_visibilities_in_channel_and_stokes = max;
}

__host__ void calc_sBeam(std::vector<double3> uvw,
                         std::vector<float> weight,
                         float nu,
                         double* s_uu,
                         double* s_vv,
                         double* s_uv) {
  double u_lambda, v_lambda;
  double local_s_uu = 0.0;
  double local_s_vv = 0.0;
  double local_s_uv = 0.0;

  // Use reduction for efficient parallel accumulation
#pragma omp parallel for shared(uvw, weight) private(u_lambda, v_lambda) \
    reduction(+ : local_s_uu, local_s_vv, local_s_uv)
  for (int i = 0; i < uvw.size(); i++) {
    u_lambda = metres_to_lambda(uvw[i].x, nu);
    v_lambda = metres_to_lambda(uvw[i].y, nu);

    // Accumulate into reduction variables (no synchronization needed)
    local_s_uu += u_lambda * u_lambda * weight[i];
    local_s_vv += v_lambda * v_lambda * weight[i];
    local_s_uv += u_lambda * v_lambda * weight[i];
  }

  // Update output pointers after parallel reduction
  *s_uu += local_s_uu;
  *s_vv += local_s_vv;
  *s_uv += local_s_uv;
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

  double center_j = floor(N / 2.0);
  double center_k = floor(M / 2.0);

  // Parallelize the loop with protection against race conditions
  // In theory, each visibility maps to a unique grid cell, but floating-point
  // rounding could cause collisions, so we protect the write with a critical
  // section
  int j, k;
  double grid_pos_x, grid_pos_y;
#pragma omp parallel for schedule(static)                            \
    shared(Vm_gridded, uvw_gridded_sp, Vm_gridded_sp, deltau_meters, \
               deltav_meters, center_j, center_k, M,                 \
               N) private(j, k, grid_pos_x, grid_pos_y)
  for (int i = 0; i < numvis; i++) {
    grid_pos_x = uvw_gridded_sp[i].x / deltau_meters;
    grid_pos_y = uvw_gridded_sp[i].y / deltav_meters;
    // Match the gridding coordinate calculation exactly:
    // j_fp = grid_pos_x + center_j + 0.5; j = int(j_fp)
    j = int(grid_pos_x + center_j + 0.5);
    k = int(grid_pos_y + center_k + 0.5);
    if (j >= 0 && j < N && k >= 0 && k < M) {
      // Critical section protects against potential collisions (should be rare)
      // Each visibility should map to a unique grid cell after gridding
#pragma omp critical
      {
        Vm_gridded[N * k + j] = Vm_gridded_sp[i];
      }
    }
  }
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
                         CKernel* ckernel,
                         float* I,
                         VirtualImageProcessor* ip,
                         MSDataset& dataset) {
  long UVpow2;
  bool fft_shift = true;  // fft_shift=false (DC at corner 0,0)

  // Instead of using sparse gridded model visibilities (computed with bilinear
  // interpolation), we recompute the FFT of the final image for each
  // frequency/channel and use the full grid directly. This is more accurate
  // than reconstructing from sparse samples.

  for (int f = 0; f < data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
    for (int i = 0; i < data.total_frequencies; i++) {
      unsigned int j = omp_get_thread_num();
      unsigned int num_cpu_threads = omp_get_num_threads();
      int gpu_idx = i % num_gpus;
      cudaSetDevice(gpu_idx + firstgpu);
      int gpu_id = -1;
      cudaGetDevice(&gpu_id);

      // Compute visibility grid from image using common pipeline
      // Use shift=true to move DC component to center, matching the gridding
      // coordinate system which uses center_j = floor(N/2.0)
      if (dataset.antennas.size() > 0) {
        computeImageToVisibilityGrid(
            I, ip, vars_gpu, gpu_idx, M, N, fields[f].nu[i],
            fields[f].ref_xobs_pix, fields[f].ref_yobs_pix,
            fields[f].phs_xobs_pix, fields[f].phs_yobs_pix,
            dataset.antennas[0].antenna_diameter, dataset.antennas[0].pb_factor,
            dataset.antennas[0].pb_cutoff, dataset.antennas[0].primary_beam,
            1.0f, ckernel, fft_shift);
      }

      for (int s = 0; s < data.nstokes; s++) {
        // Now the number of visibilities will be the original one (restore from
        // backup)
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

        // Note: vars_gpu[gpu_idx].device_V already contains the full FFT grid
        // (computed above), so we don't need to copy from gridded_visibilities

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

        // Convert UVW coordinates from meters to lambda units (required for
        // degriddingGPU) We sample at original (u,v) coordinates - no Hermitian
        // symmetry manipulation needed since the FFT grid from ifft2 has full
        // complex values at all positions
        convertUVWToLambda<<<
            fields[f].device_visibilities[i][s].numBlocksUV,
            fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].uvw, fields[f].nu[i],
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        checkCudaErrors(cudaDeviceSynchronize());

        // Degridding: Use proper convolution kernel degridding (works for both
        // 1x1 PillBox and larger kernels). For 1x1 PillBox, support=0 so it
        // reduces to nearest neighbor, matching the gridding procedure exactly.
        degriddingGPU<<<
            fields[f].device_visibilities[i][s].numBlocksUV,
            fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].device_visibilities[i][s].Vm, vars_gpu[gpu_idx].device_V,
            ckernel->getGPUKernel(), deltau, deltav,
            fields[f].numVisibilitiesPerFreqPerStoke[i][s], M, N,
            ckernel->getm(), ckernel->getn(), ckernel->getSupportX(),
            ckernel->getSupportY());
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

// Helper function to compute visibility grid from image (common pipeline)
// This encapsulates the repeated pattern: calculateInu -> apply_beam ->
// apply_GCF -> FFT2D -> phase_rotate
__host__ void computeImageToVisibilityGrid(float* I,
                                           VirtualImageProcessor* ip,
                                           varsPerGPU* vars_gpu,
                                           int gpu_idx,
                                           long M,
                                           long N,
                                           float nu,
                                           float ref_xobs_pix,
                                           float ref_yobs_pix,
                                           float phs_xobs_pix,
                                           float phs_yobs_pix,
                                           float antenna_diameter,
                                           float pb_factor,
                                           float pb_cutoff,
                                           int primary_beam,
                                           float fg_scale,
                                           CKernel* ckernel,
                                           bool fft_shift) {
  // Recompute FFT of final image for this frequency/channel
  ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, I, nu);

  // Apply primary beam
  ip->apply_beam(vars_gpu[gpu_idx].device_I_nu, antenna_diameter, pb_factor,
                 pb_cutoff, ref_xobs_pix, ref_yobs_pix, nu, primary_beam,
                 fg_scale);

  // Apply Gridding Correction Function (GCF) if using convolution kernel
  if (ckernel != NULL && ckernel->getGCFGPU() != NULL) {
    apply_GCF<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[gpu_idx].device_I_nu,
                                                  ckernel->getGCFGPU(), N);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // FFT 2D: Transform image to visibility grid
  FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
        vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, fft_shift);

  // Phase rotate to correct phase center
  // Pass fft_shift as dc_at_center since they match (fft_shift=true means DC at
  // center) Pass crpix1 and crpix2 from FITS header (already declared as
  // extern)
  phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(
      vars_gpu[gpu_idx].device_V, M, N, phs_xobs_pix, phs_yobs_pix, crpix1,
      crpix2, fft_shift);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void FFT2D(cufftComplex* output_data,
                    cufftComplex* input_data,
                    cufftHandle plan,
                    int M,
                    int N,
                    int direction,
                    bool shift) {
  if (shift) {
    // Before FFT/IFFT: use ifftshift to move DC from center to (0,0)
    // cuFFT expects DC component at (0,0) for proper FFT computation
    // This is the same for both CUFFT_FORWARD and CUFFT_INVERSE
    // In-place operation on input_data
    ifftshift_2D<<<numBlocksNN, threadsPerBlockNN>>>(input_data, M, N);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)input_data,
                               (cufftComplex*)output_data, direction));
  // Note: cufftExecC2C is asynchronous, but we rely on implicit synchronization
  // when the next kernel launches. However, if shift=false, we need explicit
  // sync before the next operation uses output_data. This is handled by syncs
  // after FFT2D calls in the calling code.

  if (shift) {
    // After FFT/IFFT: use fftshift to move DC from (0,0) to center
    // cuFFT always outputs with DC at (0,0) regardless of direction
    // (FORWARD/INVERSE) We restore DC to center to match our gridding
    // coordinate system (center_j = floor(N/2.0)) In-place operation on
    // output_data
    fftshift_2D<<<numBlocksNN, threadsPerBlockNN>>>(output_data, M, N);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  // When shift=false, the sync is deferred to the caller (which is correct
  // since phase_rotate needs device_V to be ready)
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
    // Precompute center values once
    const double center_j = floor(M / 2.0);
    const double center_k = floor(N / 2.0);

    // Use __ldg for read-only cached access to input data
    const float3 uvw_val = uvw[i];  // float3 is 12 bytes, __ldg doesn't apply
                                    // but compiler may optimize
    const cufftComplex Vo_val = __ldg(&Vo[i]);
    const float w_val = __ldg(&w[i]);

    j = (int)(uvw_val.x / deltau + center_j + 0.5);
    k = (int)(uvw_val.y / deltav + center_k + 0.5);

    if (k < M && j < N) {
      const int grid_idx = N * k + j;
      atomicAdd(&Vo_g[grid_idx].x, w_val * Vo_val.x);
      atomicAdd(&Vo_g[grid_idx].y, w_val * Vo_val.y);
      atomicAdd(&w_g[grid_idx], w_val);
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
  cufftComplex degrid_val = floatComplexZero();
  float ckernel_result;

  if (i < visibilities) {
    // Match gridding coordinate calculation exactly:
    // grid_pos = uvw / deltau, j = int(grid_pos + center + 0.5)
    // Use double precision for center calculation to match gridding
    double center_j = floor(N / 2.0);
    double center_k = floor(M / 2.0);
    double grid_pos_x = uvw[i].x / deltau;
    double grid_pos_y = uvw[i].y / deltav;
    j = int(grid_pos_x + center_j + 0.5);
    k = int(grid_pos_y + center_k + 0.5);

    for (int m = -supportY; m <= supportY; m++) {
      for (int n = -supportX; n <= supportX; n++) {
        shifted_j = j + n;
        shifted_k = k + m;
        kernel_j = n + supportX;
        kernel_i = m + supportY;
        // Check bounds: grid must be valid AND kernel indices must be valid
        if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
            shifted_j < N && kernel_i >= 0 && kernel_i < kernel_m &&
            kernel_j >= 0 && kernel_j < kernel_n) {
          // Use __ldg for read-only cached access to kernel (read-only)
          ckernel_result = __ldg(&kernel[kernel_n * kernel_i + kernel_j]);
          // Direct grid access - no Hermitian symmetry handling needed
          // The FFT grid from ifft2 of a real image already has full complex
          // values at all positions, so we can sample directly at any (u,v)
          // coordinate
          const cufftComplex grid_val = __ldg(&Vm_g[N * shifted_k + shifted_j]);
          degrid_val.x += ckernel_result * grid_val.x;
          degrid_val.y += ckernel_result * grid_val.y;
        }
      }
    }
    Vm[i].x = degrid_val.x;
    Vm[i].y = degrid_val.y;
  }
}

// Apply Hermitian symmetry only (flip u,v,w and conjugate visibility when u >
// 0) Hermitian symmetry: V(u,v,w) = V*(-u,-v,-w)* where * denotes complex
// conjugate Does NOT convert to lambda units - use convertUVWToLambda
// separately
__global__ void applyHermitianSymmetry(double3* UVW,
                                       cufftComplex* Vo,
                                       int numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    if (UVW[i].x > 0.0) {
      UVW[i].x *= -1.0;  // Negate u coordinate
      UVW[i].y *= -1.0;  // Negate v coordinate
      UVW[i].z *= -1.0;  // Negate w coordinate (for consistency with gridding)
      Vo[i].y *= -1.0f;  // Conjugate: flip imaginary part
    }
  }
}

// Convert UVW coordinates from meters to lambda units without applying
// Hermitian symmetry
__global__ void convertUVWToLambda(double3* UVW,
                                   float freq,
                                   int numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
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
    // Airy disk formula: [2*J1(π*D*θ/λ) / (π*D*θ/λ)]²
    // where D is diameter, θ is angle, λ is wavelength
    // pb_factor scales where the first null occurs (standard is RZ ≈ 1.22)
    // Scale the argument so first null occurs at pb_factor * λ / D
    float bessel_arg =
        PI * distance * antenna_diameter / lambda * (RZ / pb_factor);
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

__global__ void apply_GCF(cufftComplex* __restrict__ image,
                          const float* __restrict__ gcf,
                          long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * gcf[N * i + j], 0.0f);
}

/*--------------------------------------------------------------------
 * Device functions for phase rotation
 *--------------------------------------------------------------------*/

// Compute frequencies and relative phase coordinates for DC at center case
__device__ void computeFrequenciesAndPhaseCenter(int j,
                                                 int k,
                                                 long M,
                                                 long N,
                                                 double xphs,
                                                 double yphs,
                                                 double crpix1,
                                                 double crpix2,
                                                 double& u_freq,
                                                 double& v_freq,
                                                 double& xphs_relative,
                                                 double& yphs_relative) {
  // DC at center: frequencies are centered after fftshift
  // Frequencies: [-N/2, ..., -1, 0, 1, ..., N/2-1] / N
  // Use actual center pixel from FITS header (crpix1, crpix2)
  // FITS uses 1-indexed coordinates, so subtract 1 to get 0-indexed
  const double center_j = crpix1 - 1.0;  // Column center (u direction, width N)
  const double center_k = crpix2 - 1.0;  // Row center (v direction, height M)

  // u frequencies (column direction, width N) - u maps to j (columns)
  u_freq = ((double)j - center_j) / (double)N;

  // v frequencies (row direction, height M) - v maps to k (rows)
  v_freq = ((double)k - center_k) / (double)M;

  // FFT grid center is at (crpix1-1, crpix2-1), so convert from absolute to
  // relative to center
  xphs_relative =
      xphs - center_j;  // Convert to cartesian (relative to FFT grid center)
  yphs_relative = yphs - center_k;
}

// Compute frequencies and relative phase coordinates for DC at corner case
__device__ void computeFrequenciesAndPhaseCorner(int j,
                                                 int k,
                                                 long M,
                                                 long N,
                                                 double xphs,
                                                 double yphs,
                                                 double& u_freq,
                                                 double& v_freq,
                                                 double& xphs_relative,
                                                 double& yphs_relative) {
  // DC at corner: frequencies go from 0 to N-1 (standard fftfreq)
  // u frequencies (column direction)
  if (j < (N + 1) / 2) {
    u_freq = (double)j / (double)N;  // Positive frequencies
  } else {
    u_freq = ((double)j - (double)N) / (double)N;  // Negative frequencies
  }

  // v frequencies (row direction)
  if (k < (M + 1) / 2) {
    v_freq = (double)k / (double)M;  // Positive frequencies
  } else {
    v_freq = ((double)k - (double)M) / (double)M;  // Negative frequencies
  }

  // FFT grid corner is at (0, 0), so coordinates are already relative to corner
  xphs_relative = xphs;  // Already relative to corner (0,0)
  yphs_relative = yphs;
}

/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(-2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex* __restrict__ data,
                             long M,
                             long N,
                             double xphs,
                             double yphs,
                             double crpix1,
                             double crpix2,
                             bool dc_at_center) {
  // cuFFT uses row-major layout: data[N * row + column]
  // Match gridding convention: j = column (u direction), k = row (v direction)
  // Array indexing: V[N * row + column] = V[N * k + j]
  const int j =
      threadIdx.x + blockDim.x * blockIdx.x;  // Column index (u direction)
  const int k =
      threadIdx.y + blockDim.y * blockIdx.y;  // Row index (v direction)

  if (j < N && k < M) {
    double u_freq, v_freq;
    double xphs_relative, yphs_relative;

    if (dc_at_center) {
      computeFrequenciesAndPhaseCenter(j, k, M, N, xphs, yphs, crpix1, crpix2,
                                       u_freq, v_freq, xphs_relative,
                                       yphs_relative);
    } else {
      computeFrequenciesAndPhaseCorner(j, k, M, N, xphs, yphs, u_freq, v_freq,
                                       xphs_relative, yphs_relative);
    }

    double phase =
        -2.0 * CUDART_PI * (u_freq * xphs_relative + v_freq * yphs_relative);

    float c, s;
#if (__CUDA_ARCH__ >= 300)
    sincosf((float)phase, &s, &c);
#else
    c = cosf((float)phase);
    s = sinf((float)phase);
#endif
    cufftComplex exp_phase =
        make_cuFloatComplex(c, s);  // e^(-2πi*(u*x0 + v*y0))
    // Array indexing matches cuFFT and gridding: V[N * row + column] = V[N * k
    // + j]
    data[N * k + j] =
        cuCmulf(data[N * k + j], exp_phase);  // Complex multiplication
  }
}

/*--------------------------------------------------------------------
 * Device functions for bilinear interpolation
 *--------------------------------------------------------------------*/

// Bilinear interpolation for DC at center case
__device__ bool interpolateVisibilityCenter(const double3& uvw,
                                            const double deltau,
                                            const double deltav,
                                            const long M,
                                            const long N,
                                            const cufftComplex* __restrict__ V,
                                            cufftComplex& result) {
  // DC at center: standard bilinear interpolation
  // Use floor(N/2.0) and floor(M/2.0) to match gridding coordinate system
  // IMPORTANT: Match gridding convention: j = column (u), k = row (v)
  const double center_j =
      floor(N / 2.0);  // Column center (u direction, width N)
  const double center_k = floor(M / 2.0);  // Row center (v direction, height M)

  // Standard bilinear interpolation: continuous coordinate without +0.5
  // The +0.5 is only for rounding to nearest pixel (used in gridding), not for
  // interpolation Match Python implementation: u_pix = u/deltau + center (no
  // +0.5 for bilinear) Match gridding: j = column (u), k = row (v)
  double grid_pos_x = uvw.x / deltau;  // u -> j (column)
  double grid_pos_y = uvw.y / deltav;  // v -> k (row)

  // Continuous coordinate in grid space (centered) - NO +0.5 for bilinear
  // interpolation Gridding uses +0.5 for rounding to nearest pixel center, but
  // for bilinear interpolation we need to interpolate between pixel centers (at
  // integer positions), so we use floor
  double j_cont =
      grid_pos_x + center_j;  // Column (u direction) - continuous coordinate
  double k_cont =
      grid_pos_y + center_k;  // Row (v direction) - continuous coordinate

  // Get integer parts using floor (standard bilinear interpolation)
  // Note: floor gives us the lower-left corner of the interpolation cell
  int j1 = __double2int_rd(j_cont);  // Column index (u) - floor
  int j2 = j1 + 1;
  double du = j_cont - j1;  // Fractional part for interpolation

  int i1 = __double2int_rd(
      k_cont);  // Row index (v) - using i1 for row to match array indexing
  int i2 = i1 + 1;
  double dv = k_cont - i1;  // Fractional part for interpolation

  // Check all boundaries explicitly (no wrapping)
  // Note: j1, j2 are column indices (u), i1, i2 are row indices (v)
  // Array indexing must match gridding: V[N * row + column] = V[N * k + j] =
  // V[N * i + j] where i = row (v direction), j = column (u direction) For
  // bilinear interpolation, we need all four corners to be valid Allow i1, j1
  // to be -1 (for coordinates just below 0 after centering) but ensure i2, j2
  // are within bounds
  if (j1 >= -1 && j1 < N && i1 >= -1 && i1 < M && j2 >= 0 && j2 < N &&
      i2 >= 0 && i2 < M) {
    // Clamp negative indices to 0 for array access
    int j1_safe = (j1 < 0) ? 0 : j1;
    int i1_safe = (i1 < 0) ? 0 : i1;
    // Use regular global memory with __ldg for read-only cached access
    // Array layout matches gridding: V[N * row + column] = V[N * i + j]
    // where i = row index (v), j = column index (u)
    // Use safe indices to handle edge cases where i1 or j1 might be -1
    const cufftComplex v11 =
        __ldg(&V[N * i1_safe + j1_safe]);  // (i1, j1) = (row, column)
    const cufftComplex v12 = __ldg(&V[N * i1_safe + j2]);  // (i1, j2)
    const cufftComplex v21 = __ldg(&V[N * i2 + j1_safe]);  // (i2, j1)
    const cufftComplex v22 = __ldg(&V[N * i2 + j2]);       // (i2, j2)

    // Optimized bilinear interpolation weights
    const float w11 = (1.0f - du) * (1.0f - dv);
    const float w12 = du * (1.0f - dv);
    const float w21 = (1.0f - du) * dv;
    const float w22 = du * dv;

    result = make_cuFloatComplex(
        w11 * v11.x + w12 * v12.x + w21 * v21.x + w22 * v22.x,
        w11 * v11.y + w12 * v12.y + w21 * v21.y + w22 * v22.y);
    return true;
  }
  return false;
}

// Bilinear interpolation for DC at corner case
__device__ bool interpolateVisibilityCorner(const double3& uvw,
                                            const double deltau,
                                            const double deltav,
                                            const long M,
                                            const long N,
                                            const cufftComplex* __restrict__ V,
                                            cufftComplex& result) {
  // DC at corner: handle negative coordinates with wrapping
  // Match gridding convention: j = column (u), k = row (v)
  double grid_pos_x = uvw.x / deltau;  // u -> j (column)
  double grid_pos_y = uvw.y / deltav;  // v -> k (row)

  // Handle negative coordinates by wrapping
  if (grid_pos_x < 0.0)
    grid_pos_x += N;  // Wrap u (columns, width N)
  if (grid_pos_y < 0.0)
    grid_pos_y += M;  // Wrap v (rows, height M)

  // Get integer parts (floor)
  int j1 = __double2int_rd(grid_pos_x);  // Column index (u)
  int j2 = (j1 + 1) % N;                 // Wrap around for columns
  double du = grid_pos_x - j1;           // Fractional part

  int i1 = __double2int_rd(grid_pos_y);  // Row index (v) - using i1 for row
  int i2 = (i1 + 1) % M;                 // Wrap around for rows (height M)
  double dv = grid_pos_y - i1;           // Fractional part

  // Boundary check: j1, j2 are columns (u), i1, i2 are rows (v)
  // Array indexing must match gridding: V[N * row + column] = V[N * i + j]
  // where i = row (v direction, height M), j = column (u direction, width N)
  if (j1 >= 0 && j1 < N && i1 >= 0 && i1 < M) {
    // Use regular global memory with __ldg for read-only cached access
    // Note: i2 and j2 are wrapped, so they're always in range due to modulo
    const cufftComplex v11 =
        __ldg(&V[N * i1 + j1]);  // (i1, j1) = (row, column)
    const cufftComplex v12 = __ldg(&V[N * i1 + j2]);  // (i1, j2)
    const cufftComplex v21 = __ldg(&V[N * i2 + j1]);  // (i2, j1)
    const cufftComplex v22 = __ldg(&V[N * i2 + j2]);  // (i2, j2)

    // Optimized bilinear interpolation weights (same as dc_at_center path)
    const float w11 =
        (1.0f - du) * (1.0f - dv);       // Weight for (i1, j1) = lower-left
    const float w12 = du * (1.0f - dv);  // Weight for (i1, j2) = lower-right
    const float w21 = (1.0f - du) * dv;  // Weight for (i2, j1) = upper-left
    const float w22 = du * dv;           // Weight for (i2, j2) = upper-right

    const float Zreal = w11 * v11.x + w12 * v12.x + w21 * v21.x + w22 * v22.x;
    const float Zimag = w11 * v11.y + w12 * v12.y + w21 * v21.y + w22 * v22.y;

    result = make_cuFloatComplex(Zreal, Zimag);
    return true;
  }
  return false;
}

/*
 * Bilinear interpolation of visibilities from gridded visibility plane
 * dc_at_center: true if DC component is at center (N/2, M/2), false if at
 * corner (0,0) This unified function replaces vis_mod (DC at corner) and
 * vis_mod2 (DC at center)
 */
__global__ void bilinearInterpolateVisibility(
    cufftComplex* __restrict__ Vm,
    const cufftComplex* __restrict__ V,
    const double3* __restrict__ UVW,
    float* __restrict__ weight,
    const double deltau,
    const double deltav,
    const long numVisibilities,
    const long M,
    const long N,
    const bool dc_at_center) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    // Load UVW once (double3 is 64-bit, so __ldg doesn't apply)
    const double3 uvw = UVW[i];
    cufftComplex result;
    bool success;

    if (dc_at_center) {
      success =
          interpolateVisibilityCenter(uvw, deltau, deltav, M, N, V, result);
    } else {
      success =
          interpolateVisibilityCorner(uvw, deltau, deltav, M, N, V, result);
    }

    if (success) {
      Vm[i] = result;
    } else {
      weight[i] = 0.0f;
    }
  }
}

__global__ void residual(cufftComplex* __restrict__ Vr,
                         const cufftComplex* __restrict__ Vm,
                         const cufftComplex* __restrict__ Vo,
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

__global__ void chi2Vector(float* __restrict__ chi2,
                           const cufftComplex* __restrict__ Vr,
                           const float* __restrict__ w,
                           long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    chi2[i] = w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
  }
}

// Compute squared weights for effective number of samples calculation
__global__ void weightsSquaredVector(float* __restrict__ w_squared,
                                     const float* __restrict__ w,
                                     long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    w_squared[i] = w[i] * w[i];
  }
}

__host__ __device__ float approxAbs(float val, float epsilon) {
  return sqrtf(val * val + epsilon);
}

__device__ float calculateL1norm(const float* __restrict__ I,
                                 float epsilon,
                                 float noise,
                                 float noise_cut,
                                 int index,
                                 int M,
                                 int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float l1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    l1 = approxAbs(c, epsilon);
  }

  return l1;
}

__global__ void L1Vector(float* __restrict__ L1,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float epsilon,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  L1[N * i + j] =
      calculateL1norm(I, epsilon, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDNormL1(const float* __restrict__ I,
                                  float lambda,
                                  float noise,
                                  float epsilon,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dL1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    dL1 = c / approxAbs(c, epsilon);
  }

  dL1 *= lambda;
  return dL1;
}

__global__ void DL1NormK(float* __restrict__ dL1,
                         const float* __restrict__ I,
                         const float* __restrict__ noise,
                         float epsilon,
                         float noise_cut,
                         float lambda,
                         long N,
                         long M,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dL1[N * i + j] =
      calculateDNormL1(I, lambda, noise_val, epsilon, noise_cut, index, M, N);
}

__device__ float calculateGL1norm(const float* __restrict__ I,
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

  float l1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    l1 = approxAbs(c, epsilon_a) / (approxAbs(prior, epsilon_a) + epsilon_b);
  }

  return l1;
}

__global__ void GL1Vector(float* __restrict__ L1,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          const float* __restrict__ prior,
                          long N,
                          long M,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  L1[N * i + j] = calculateGL1norm(I, prior_val, epsilon_a, epsilon_b,
                                   noise_val, noise_cut, index, M, N);
}

__device__ float calculateDGNormL1(const float* __restrict__ I,
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

  float dL1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    const float prior_abs = approxAbs(prior, epsilon_a);
    dL1 = c / (approxAbs(c, epsilon_a) * (prior_abs + epsilon_b));
  }

  dL1 *= lambda;
  return dL1;
}

__global__ void DGL1NormK(float* __restrict__ dL1,
                          const float* __restrict__ I,
                          const float* __restrict__ prior,
                          const float* __restrict__ noise,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          float lambda,
                          long N,
                          long M,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  dL1[N * i + j] = calculateDGNormL1(I, prior_val, lambda, noise_val, epsilon_a,
                                     epsilon_b, noise_cut, index, M, N);
}

__device__ float calculateS(const float* __restrict__ I,
                            float G,
                            float eta,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float S = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    S = c * logf((c / G) + (eta + 1.0f));
  }

  return S;
}

__device__ float calculateDS(const float* __restrict__ I,
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
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    const float c_over_G_plus_eta = (c / G) + (eta + 1.0f);
    dS = logf(c_over_G_plus_eta) + 1.0f / (1.0f + (((eta + 1.0f) * G) / c));
  }

  dS *= lambda;
  return dS;
}

__global__ void SVector(float* __restrict__ S,
                        const float* __restrict__ noise,
                        float* __restrict__ I,
                        long N,
                        long M,
                        float noise_cut,
                        float prior_value,
                        float eta,
                        int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  S[N * i + j] =
      calculateS(I, prior_value, eta, noise_val, noise_cut, index, M, N);
}

__global__ void DS(float* __restrict__ dS,
                   float* __restrict__ I,
                   const float* __restrict__ noise,
                   float noise_cut,
                   float lambda,
                   float prior_value,
                   float eta,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dS[N * i + j] = calculateDS(I, prior_value, eta, lambda, noise_val, noise_cut,
                              index, M, N);
}

__global__ void SGVector(float* __restrict__ S,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float noise_cut,
                         const float* __restrict__ prior,
                         float eta,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  S[N * i + j] =
      calculateS(I, prior_val, eta, noise_val, noise_cut, index, M, N);
}

__global__ void DSG(float* __restrict__ dS,
                    const float* __restrict__ I,
                    const float* __restrict__ noise,
                    float noise_cut,
                    float lambda,
                    const float* __restrict__ prior,
                    float eta,
                    long N,
                    long M,
                    int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  dS[N * i + j] =
      calculateDS(I, prior_val, eta, lambda, noise_val, noise_cut, index, M, N);
}

__device__ float calculateQP(const float* __restrict__ I,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float qp = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float l = I[N * M * index + N * i + (j - 1)];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];

      qp = (c - l) * (c - l) + (c - r) * (c - r) + (c - u) * (c - u) +
           (c - d) * (c - d);
      qp *= 0.5f;  // Use multiply instead of divide
    } else {
      qp = I[N * M * index + N * i + j];
    }
  }

  return qp;
}
__global__ void QPVector(float* __restrict__ Q,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  Q[N * i + j] = calculateQP(I, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDQ(const float* __restrict__ I,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dQ = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];

      dQ = 2.0f * (4.0f * c - d + u + r + l);
    } else {
      dQ = I[N * M * index + N * i + j];
    }
  }

  dQ *= lambda;
  return dQ;
}

__global__ void DQ(float* __restrict__ dQ,
                   const float* __restrict__ I,
                   const float* __restrict__ noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dQ[N * i + j] = calculateDQ(I, lambda, noise_val, noise_cut, index, M, N);
}

__device__ float calculateTV(const float* __restrict__ I,
                             float epsilon,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0f;
  if (noise < noise_cut) {
    if (i < M - 1 && j < N - 1) {
      const float c = I[N * M * index + N * i + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];

      const float dxy0 = (r - c) * (r - c);
      const float dxy1 = (d - c) * (d - c);
      tv = sqrtf(dxy0 + dxy1 + epsilon);
    } else {
      tv = I[N * M * index + N * i + j];
    }
  }

  return tv;
}

__global__ void TVVector(float* __restrict__ TV,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         float epsilon,
                         long N,
                         long M,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  TV[N * i + j] = calculateTV(I, epsilon, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDTV(const float* __restrict__ I,
                              float epsilon,
                              float lambda,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dtv = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];
      const float dl_corner = I[N * M * index + N * (i + 1) + (j - 1)];
      const float ru_corner = I[N * M * index + N * (i - 1) + (j + 1)];

      const float num0 = 2.0f * c - r - d;
      const float num1 = c - l;
      const float num2 = c - u;

      const float den_arg0 = (c - r) * (c - r) + (c - d) * (c - d) + epsilon;
      const float den_arg1 =
          (l - c) * (l - c) + (l - dl_corner) * (l - dl_corner) + epsilon;
      const float den_arg2 =
          (u - ru_corner) * (u - ru_corner) + (u - c) * (u - c) + epsilon;

      const float den0 = sqrtf(den_arg0);
      const float den1 = sqrtf(den_arg1);
      const float den2 = sqrtf(den_arg2);

      dtv = num0 / den0 + num1 / den1 + num2 / den2;
    } else {
      dtv = I[N * M * index + N * i + j];
    }
  }

  dtv *= lambda;
  return dtv;
}
__global__ void DTV(float* __restrict__ dTV,
                    const float* __restrict__ I,
                    const float* __restrict__ noise,
                    float epsilon,
                    float noise_cut,
                    float lambda,
                    long N,
                    long M,
                    int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dTV[N * i + j] =
      calculateDTV(I, epsilon, lambda, noise_val, noise_cut, index, M, N);
}

// Anisotropic Total Variation kernels
__device__ float calculateATV(const float* __restrict__ I,
                              float epsilon,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atv = 0.0f;
  if (noise < noise_cut) {
    if (i < M - 1 && j < N - 1) {
      const float c = I[N * M * index + N * i + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];

      // Anisotropic TV: |dx| + |dy| + epsilon (for numerical stability)
      const float dx = fabsf(r - c);
      const float dy = fabsf(d - c);
      atv = dx + dy + epsilon;
    } else {
      atv = I[N * M * index + N * i + j];
    }
  }

  return atv;
}

__global__ void ATVVector(float* __restrict__ ATV,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          float epsilon,
                          long N,
                          long M,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  ATV[N * i + j] = calculateATV(I, epsilon, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDATV(const float* __restrict__ I,
                               float epsilon,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float datv = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];

      // Anisotropic TV derivative: sign(dx) + sign(dy)
      // For numerical stability, use smoothed sign: x / (|x| + epsilon)
      const float dx_right = c - r;
      const float dx_left = l - c;
      const float dy_down = c - d;
      const float dy_up = u - c;

      const float sign_dx_right = dx_right / (fabsf(dx_right) + epsilon);
      const float sign_dx_left = dx_left / (fabsf(dx_left) + epsilon);
      const float sign_dy_down = dy_down / (fabsf(dy_down) + epsilon);
      const float sign_dy_up = dy_up / (fabsf(dy_up) + epsilon);

      // Sum of contributions from all neighbors
      datv = sign_dx_right + sign_dx_left + sign_dy_down + sign_dy_up;
    } else {
      datv = I[N * M * index + N * i + j];
    }
  }

  datv *= lambda;
  return datv;
}

__global__ void DATV(float* __restrict__ dATV,
                     const float* __restrict__ I,
                     const float* __restrict__ noise,
                     float epsilon,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dATV[N * i + j] =
      calculateDATV(I, epsilon, lambda, noise_val, noise_cut, index, M, N);
}

__device__ float calculateTSV(const float* __restrict__ I,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0f;
  if (noise < noise_cut) {
    if (i < M - 1 && j < N - 1) {
      const float c = I[N * M * index + N * i + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];

      const float dx = c - r;
      const float dy = c - d;
      tv = dx * dx + dy * dy;
    } else {
      tv = I[N * M * index + N * i + j];
    }
  }

  return tv;
}

__global__ void TSVVector(float* __restrict__ STV,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          long N,
                          long M,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  STV[N * i + j] = calculateTSV(I, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDTSV(const float* __restrict__ I,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dstv = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];

      dstv = 8.0f * c - 2.0f * (u + l + d + r);
    } else {
      dstv = I[N * M * index + N * i + j];
    }
  }

  dstv *= lambda;
  return dstv;
}

__global__ void DTSV(float* __restrict__ dSTV,
                     const float* __restrict__ I,
                     const float* __restrict__ noise,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dSTV[N * i + j] = calculateDTSV(I, lambda, noise_val, noise_cut, index, M, N);
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
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
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
    if ((i > 1 && i < M - 2) && (j > 1 && j < N - 2)) {
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
      alpha * d_y[M * N * image * k + M * N * image + (N * i + j)];
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
                      bool normalize,
                      float N_eff) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Early exit: check noise threshold first to avoid unnecessary computation
  const float noise_val = noise[N * i + j];
  if (noise_val >= noise_cut) {
    return;
  }

  // Compute pixel coordinates and direction cosines
  const int x0 = phs_xobs;
  const int y0 = phs_yobs;
  const double x = (j - x0) * DELTAX * RPDEG_D;
  const double y = (i - y0) * DELTAY * RPDEG_D;
  const double z = sqrt(1.0 - x * x - y * y);

  // Compute attenuation only if we pass the noise check
  const float atten =
      attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                  ref_yobs, DELTAX, DELTAY, primary_beam);
  const float scale_factor = fg_scale * atten;

  // Accumulate chi-squared derivative
  float dchi2 = 0.0f;

  // Precompute constants for the loop
  const double z_minus_one = z - 1.0;
  const double two = 2.0;

// Unroll loop for better performance (compiler hint)
#pragma unroll 4
  for (int v = 0; v < numVisibilities; v++) {
    // Load UVW values - compiler will optimize memory access
    // For structs, regular access is fine as arrays are contiguous
    const double uvw_x = UVW[v].x;
    const double uvw_y = UVW[v].y;
    const double uvw_z = UVW[v].z;

    // Compute phase components (FMA-friendly operations)
    const double Ukv = x * uvw_x;
    const double Vkv = y * uvw_y;
    const double Wkv = z_minus_one * uvw_z;
    const double phase = two * (Ukv + Vkv + Wkv);

    // Compute sin and cos of phase
    float cosk, sink;
#if (__CUDA_ARCH__ >= 300)
    sincospif(phase, &sink, &cosk);
#else
    cosk = cospif(phase);
    sink = sinpif(phase);
#endif

    // Load visibility and weight - use __ldg for read-only scalar values
    const float weight = __ldg(&w[v]);
    const float vr_real = __ldg(&Vr[v].x);
    const float vr_imag = __ldg(&Vr[v].y);

    // Accumulate weighted contribution using FMA for better precision and
    // performance
    dchi2 = fmaf(weight, vr_real * cosk + vr_imag * sink, dchi2);
  }

  // Apply scaling factors
  dchi2 *= scale_factor;

  if (normalize) {
    // Normalize by effective number of samples: N_eff = (Σw_k)² / (Σw_k²)
    // This accounts for varying weights and represents effective degrees of
    // freedom. When all weights are equal, N_eff = N. When weights vary,
    // N_eff < N and properly accounts for the reduced effective sample size.
    if (N_eff > 0.0f) {
      dchi2 /= N_eff;
    } else {
      // Fallback to numVisibilities if N_eff is zero
      dchi2 /= numVisibilities;
    }
  }

  // Write result
  dChi2[N * i + j] = -dchi2;
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
                      bool normalize,
                      float N_eff) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Early exit: check noise threshold first to avoid unnecessary computation
  const float noise_val = noise[N * i + j];
  if (noise_val >= noise_cut) {
    return;
  }

  // Compute pixel coordinates and direction cosines
  const int x0 = phs_xobs;
  const int y0 = phs_yobs;
  const double x = (j - x0) * DELTAX * RPDEG_D;
  const double y = (i - y0) * DELTAY * RPDEG_D;
  const double z = sqrt(1.0 - x * x - y * y);

  // Compute attenuation and GCF only if we pass the noise check
  const float atten =
      attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                  ref_yobs, DELTAX, DELTAY, primary_beam);
  const float gcf_i = gcf[N * i + j];
  const float scale_factor = fg_scale * atten * gcf_i;

  // Accumulate chi-squared derivative
  float dchi2 = 0.0f;

  // Precompute constants for the loop
  const double z_minus_one = z - 1.0;
  const double two = 2.0;

// Unroll loop for better performance (compiler hint)
#pragma unroll 4
  for (int v = 0; v < numVisibilities; v++) {
    // Load UVW values - compiler will optimize memory access
    // For structs, regular access is fine as arrays are contiguous
    const double uvw_x = UVW[v].x;
    const double uvw_y = UVW[v].y;
    const double uvw_z = UVW[v].z;

    // Compute phase components (FMA-friendly operations)
    const double Ukv = x * uvw_x;
    const double Vkv = y * uvw_y;
    const double Wkv = z_minus_one * uvw_z;
    const double phase = two * (Ukv + Vkv + Wkv);

    // Compute sin and cos of phase
    float cosk, sink;
#if (__CUDA_ARCH__ >= 300)
    sincospif(phase, &sink, &cosk);
#else
    cosk = cospif(phase);
    sink = sinpif(phase);
#endif

    // Load visibility and weight - use __ldg for read-only scalar values
    const float weight = __ldg(&w[v]);
    const float vr_real = __ldg(&Vr[v].x);
    const float vr_imag = __ldg(&Vr[v].y);

    // Accumulate weighted contribution using FMA for better precision and
    // performance
    dchi2 = fmaf(weight, vr_real * cosk + vr_imag * sink, dchi2);
  }

  // Apply scaling factors
  dchi2 *= scale_factor;

  if (normalize) {
    // Normalize by effective number of samples: N_eff = (Σw_k)² / (Σw_k²)
    // This accounts for varying weights and represents effective degrees of
    // freedom. When all weights are equal, N_eff = N. When weights vary,
    // N_eff < N and properly accounts for the reduced effective sample size.
    if (N_eff > 0.0f) {
      dchi2 /= N_eff;
    } else {
      // Fallback to numVisibilities if N_eff is zero
      dchi2 /= numVisibilities;
    }
  }

  // Write result
  dChi2[N * i + j] = -dchi2;
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
  // Chain rule derivative for alpha (without fg_scale * atten):
  // ∂I_ν/∂α = fg_scale · atten · I_ν₀ · (ν/ν₀)^α · log(ν/ν₀)
  // Since dchi2 already includes fg_scale * atten, we only compute:
  // dalpha = I_ν₀ · (ν/ν₀)^α · log(ν/ν₀)
  // Final gradient: ∂χ²/∂α = dchi2 · dalpha
  dalpha = I_nu_0 * dI_nu_0 * logf(nudiv);

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
  // Chain rule derivative for I_ν₀ (without fg_scale * atten):
  // ∂I_ν/∂I_ν₀ = fg_scale · atten · (ν/ν₀)^α
  // Since dchi2 already includes fg_scale * atten, we only compute:
  // dI_nu_0 = (ν/ν₀)^α
  // Final gradient: ∂χ²/∂I_ν₀ = dchi2 · dI_nu_0

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
  // Chain rule derivatives (without fg_scale * atten, which is already in
  // dchi2): ∂I_ν/∂I_ν₀ = fg_scale · atten · (ν/ν₀)^α ∂I_ν/∂α = fg_scale · atten
  // · I_ν₀ · (ν/ν₀)^α · log(ν/ν₀)
  //
  // Since dchi2 (from DChi2 kernel) already includes fg_scale * atten,
  // we only store the part without fg_scale * atten here:
  // chain[I_ν₀] = (ν/ν₀)^α
  // chain[α] = I_ν₀ · (ν/ν₀)^α · log(ν/ν₀)
  //
  // Final gradient: ∂χ²/∂I_ν₀ = dchi2 · chain[I_ν₀] = (∂χ²/∂I_ν with
  // fg_scale*atten) · (ν/ν₀)^α
  //                 ∂χ²/∂α = dchi2 · chain[α] = (∂χ²/∂I_ν with fg_scale*atten)
  //                 · I_ν₀ · (ν/ν₀)^α · log(ν/ν₀)
  dalpha = I_nu_0 * dI_nu_0 * logf(nudiv);

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
                             float fg_scale,
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

  // Variance for I_nu_0: σ²(I_ν₀) ∝ (∂I_ν/∂I_ν₀)² * σ²(visibilities)
  // Where: ∂I_ν/∂I_ν₀ = fg_scale · atten · (ν/ν₀)^α
  // So: σ²(I_ν₀) ∝ fg_scale² · atten² · (ν/ν₀)^(2α) · σ²(visibilities)
  if (noise[N * i + j] < noise_cut) {
    float fg_scale_sq = fg_scale * fg_scale;
    noise_I[N * i + j] +=
        fg_scale_sq * atten * atten * sum_weights * nudiv_pow_alpha;
  } else {
    noise_I[N * i + j] = 0.0f;
  }
}

__global__ void alpha_Noise(float* noise_I,
                            float* images,
                            float nu,
                            float nu_0,
                            float* noise,
                            float noise_cut,
                            double DELTAX,
                            double DELTAY,
                            float xobs,
                            float yobs,
                            float antenna_diameter,
                            float pb_factor,
                            float pb_cutoff,
                            float sum_weights,
                            float fg_scale,
                            long N,
                            long M,
                            int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  I_nu_0 = images[N * i + j];
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu = I_nu_0 * nudiv_pow_alpha;
  log_nu = logf(nudiv);

  float fg_scale_sq = fg_scale * fg_scale;

  // First-order error propagation for alpha:
  // σ²(alpha) ∝ (∂I_ν/∂alpha)² * σ²(visibilities)
  // Where: ∂I_ν/∂alpha = fg_scale · atten · I_ν · log(ν/ν₀)
  // So: σ²(alpha) ∝ fg_scale² · atten² · I_ν² · log²(ν/ν₀) · σ²(visibilities)
  //
  // IMPORTANT: When ν = ν₀ (reference frequency), log(ν/ν₀)=0 so the alpha
  // variance contribution from that channel is zero. You need observation
  // frequencies away from ν₀ to get non-zero alpha errors. If you have only
  // one channel or all channels at ν₀, σ(alpha) will be 0.
  float first_order_term =
      fg_scale_sq * log_nu * log_nu * atten * atten * I_nu * I_nu * sum_weights;

  // Second-order correction term for variance of alpha:
  // In second-order error propagation, the correction accounts for curvature:
  // σ²_total = σ²_first_order + (1/2) * ∂²I_ν/∂alpha² * weight_contribution
  // Where: ∂²I_ν/∂alpha² = fg_scale · atten · I_ν · log²(ν/ν₀)
  // The second-order correction adds: (1/2) * fg_scale² · atten² · I_ν ·
  // log²(ν/ν₀) · sum_weights This is proportional to the curvature of I_ν as a
  // function of alpha
  float d2I_nu_dalpha2 =
      I_nu * log_nu * log_nu;  // ∂²I_ν/∂alpha² (without fg_scale · atten)
  float second_order_correction =
      fg_scale_sq * 0.5f * d2I_nu_dalpha2 * atten * atten * sum_weights;
  // = 0.5 * fg_scale² · I_ν · log²(ν/ν₀) · atten² · sum_weights

  if (noise[N * i + j] < noise_cut) {
    // Combine first-order and second-order terms
    noise_I[N * M + N * i + j] += first_order_term + second_order_correction;
  } else {
    noise_I[N * M + N * i + j] = 0.0f;
  }
}

// Compute covariance term for I_nu_0 and alpha noise correlation
// This accounts for the correlation between fitted parameters
// Full error propagation: σ²(I_nu) = (∂I_nu/∂I_nu_0)²σ²(I_nu_0) +
// (∂I_nu/∂alpha)²σ²(alpha)
//                         + 2(∂I_nu/∂I_nu_0)(∂I_nu/∂alpha)Cov(I_nu_0, alpha)
// The covariance term: Cov(I_nu_0, alpha) ∝ (∂I_nu/∂I_nu_0) × (∂I_nu/∂alpha) ×
// σ²(visibilities) Where: ∂I_nu/∂I_nu_0 = atten * (nu/nu_0)^alpha
//        ∂I_nu/∂alpha = I_nu * log(nu/nu_0)
// So Cov ∝ (ν/ν₀)^α · I_ν · log(ν/ν₀). When ν = ν₀, log(ν/ν₀)=0 and the
// covariance contribution from that channel is zero (same as σ(alpha)).
// Note: The covariance is stored at index 2 (after I_nu_0 variance at 0, alpha
// variance at 1) Can be used to compute:
// - Correlation coefficient: ρ = Cov(I_nu_0, alpha) / (σ(I_nu_0) * σ(alpha))
// - Total uncertainty in I_nu when both parameters vary
__global__ void covariance_Noise(float* noise_cov,
                                 float* images,
                                 float nu,
                                 float nu_0,
                                 float* noise,
                                 float noise_cut,
                                 double DELTAX,
                                 double DELTAY,
                                 float xobs,
                                 float yobs,
                                 float antenna_diameter,
                                 float pb_factor,
                                 float pb_cutoff,
                                 float sum_weights,
                                 float fg_scale,
                                 long N,
                                 long M,
                                 int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  I_nu_0 = images[N * i + j];
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu = I_nu_0 * nudiv_pow_alpha;
  log_nu = logf(nudiv);

  float fg_scale_sq = fg_scale * fg_scale;

  // First-order covariance contribution:
  // Cov(I_ν₀, alpha) ∝ (∂I_ν/∂I_ν₀) × (∂I_ν/∂alpha) × σ²(visibilities)
  // Where: ∂I_ν/∂I_ν₀ = fg_scale · atten · (ν/ν₀)^α
  //        ∂I_ν/∂alpha = fg_scale · atten · I_ν · log(ν/ν₀)
  // So: Cov(I_ν₀, alpha) ∝ fg_scale² · atten² · I_ν · log(ν/ν₀) ·
  // σ²(visibilities)
  float first_order_cov =
      fg_scale_sq * atten * atten * I_nu * log_nu * sum_weights;

  // Second-order correction for covariance:
  // The mixed second derivative term: ∂²I_ν/∂I_ν₀∂alpha * Cov contribution
  // Where: ∂²I_ν/∂I_ν₀∂alpha = fg_scale · atten · (I_ν/I_ν₀) · log(ν/ν₀)
  // This correction accounts for how the covariance affects the total
  // uncertainty The correction is: ∂²I_ν/∂I_ν₀∂alpha * (first-order covariance
  // contribution) = fg_scale² · atten² · (I_ν²/I_ν₀) · log²(ν/ν₀) · sum_weights
  float second_order_cov_correction = 0.0f;
  if (I_nu_0 != 0.0f && fabsf(I_nu_0) > 1e-10f) {
    float d2I_nu_dI_nu_0_dalpha =
        (I_nu / I_nu_0) *
        log_nu;  // ∂²I_ν/∂I_ν₀∂alpha (without fg_scale · atten)
    // The second-order correction to covariance accumulation
    second_order_cov_correction = fg_scale_sq * d2I_nu_dI_nu_0_dalpha * atten *
                                  atten * I_nu * log_nu * sum_weights;
    // = fg_scale² · (I_ν/I_ν₀ · log_nu) · atten² · I_ν · log_nu · sum_weights
    // = fg_scale² · (I_ν²/I_ν₀) · log²(ν/ν₀) · atten² · sum_weights
  }

  // Store at index 2: noise_cov[2 * M * N + N * i + j]
  // Store first-order covariance + second-order correction
  if (noise[N * i + j] < noise_cut) {
    noise_cov[2 * M * N + N * i + j] +=
        first_order_cov + second_order_cov_correction;
  } else {
    noise_cov[2 * M * N + N * i + j] = 0.0f;
  }
}

__global__ void noise_reduction(float* noise_I, long N, long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Indices 0 and 1: kernels store inverse-variance; convert to σ for output.
  // Index 0: σ(I_nu_0) [code units or Jy/pixel depending on fg_scale]
  if (noise_I[N * i + j] > 0.0f)
    noise_I[N * i + j] = 1.0f / sqrtf(noise_I[N * i + j]);
  else
    noise_I[N * i + j] = 0.0f;

  // Index 1: σ(alpha) [unitless]
  if (noise_I[N * M + N * i + j] > 0.0f)
    noise_I[N * M + N * i + j] = 1.0f / sqrtf(noise_I[N * M + N * i + j]);
  else
    noise_I[N * M + N * i + j] = 0.0f;

  // Index 2: Cov(I_nu_0, alpha) [code units or Jy/pixel depending on fg_scale]
  // — kept as covariance (not correlation). Correlation: ρ = Cov / (σ(I_nu_0) *
  // σ(alpha)). σ²(I_nu) = (∂I_nu/∂I_nu_0)²σ²(I_nu_0) +
  // (∂I_nu/∂alpha)²σ²(alpha)
  //          + 2(∂I_nu/∂I_nu_0)(∂I_nu/∂alpha)Cov(I_nu_0, alpha).
  // Covariance can be negative (anti-correlation).
}

// Convert error units from code units to Jy/pixel when normalize=False
// When normalize=False: fg_scale ≠ 1.0, so I_nu_0 is in code units
// We multiply error for I_nu_0 (index 0) and covariance (index 2) by fg_scale
// to convert to Jy/pixel. Alpha error (index 1) is unitless and unchanged.
__global__ void convertErrorUnits(float* errors,
                                  float fg_scale,
                                  long N,
                                  long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Index 0: σ(I_nu_0) — convert from code units to Jy/pixel
  errors[N * i + j] *= fg_scale;

  // Index 1: σ(alpha) — unitless, no conversion needed
  // (no change)

  // Index 2: Cov(I_nu_0, alpha) — convert from code units to Jy/pixel
  // (same units as I_nu_0)
  errors[2 * M * N + N * i + j] *= fg_scale;
}

__host__ float simulate(float* I, VirtualImageProcessor* ip, float fg_scale) {
  // simulate is equivalent to chi2 with normalize=true
  return chi2(I, ip, true, fg_scale);
}

__host__ float chi2(float* I,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  bool fft_shift = true;  // fft_shift=false (DC at corner 0,0)
  cudaSetDevice(firstgpu);

  float reduced_chi2 = 0.0f;

  // Create static 1x1 PillBox kernel for degridding when gridding is enabled
  static PillBox2D* degrid_kernel = NULL;
  static bool degrid_kernel_initialized = false;

  CKernel* ckernel = ip->getCKernel();
  bool use_gridding = (ckernel != NULL && ckernel->getGPUKernel() != NULL);

  if (use_gridding && !degrid_kernel_initialized) {
    degrid_kernel = new PillBox2D(1, 1);
    degrid_kernel->setGPUID(firstgpu);
    degrid_kernel->setSigmas(fabs(deltau), fabs(deltav));
    degrid_kernel->buildKernel();
    degrid_kernel_initialized = true;
  }

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

        // Compute visibility grid from image using common pipeline
        // Use fft_shift=false (DC at corner 0,0) for testing
        computeImageToVisibilityGrid(
            I, ip, vars_gpu, gpu_idx, M, N, datasets[d].fields[f].nu[i],
            datasets[d].fields[f].ref_xobs_pix,
            datasets[d].fields[f].ref_yobs_pix,
            datasets[d].fields[f].phs_xobs_pix,
            datasets[d].fields[f].phs_yobs_pix,
            datasets[d].antennas[0].antenna_diameter,
            datasets[d].antennas[0].pb_factor,
            datasets[d].antennas[0].pb_cutoff,
            datasets[d].antennas[0].primary_beam, fg_scale, ip->getCKernel(),
            fft_shift);  // fft_shift=false (DC at corner 0,0)

        // Texture memory removed - using regular global memory with __ldg()
        // instead

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
                                         sizeof(float) * max_number_vis));

              if (use_gridding) {
                degriddingGPU<<<
                    datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                    datasets[d]
                        .fields[f]
                        .device_visibilities[i][s]
                        .threadsPerBlockUV>>>(
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].Vm,
                    vars_gpu[gpu_idx].device_V, degrid_kernel->getGPUKernel(),
                    deltau, deltav,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    M, N, degrid_kernel->getm(), degrid_kernel->getn(),
                    degrid_kernel->getSupportX(), degrid_kernel->getSupportY());
                checkCudaErrors(cudaDeviceSynchronize());
              } else {
                // Gridding disabled: use bilinear interpolation
                bilinearInterpolateVisibility<<<
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
                    M, N, fft_shift);  // dc_at_center=false (DC at corner 0,0,
                                       // matching fft_shift=false)
                checkCudaErrors(cudaDeviceSynchronize());
              }

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

              // Use pre-computed effective number of samples (calculated once
              // before optimization)
              float N_eff = 0.0f;
              if (normalize) {
                N_eff = datasets[d].fields[f].N_eff_perFreqPerStoke[i][s];
                if (N_eff <= 0.0f) {
                  // Fallback to numVisibilities if N_eff was not pre-computed
                  N_eff = (float)datasets[d]
                              .fields[f]
                              .numVisibilitiesPerFreqPerStoke[i][s];
                }
              }

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
              if (normalize) {
                if (N_eff > 0.0f) {
                  result /= N_eff;
                } else {
                  // Fallback to numVisibilities if N_eff calculation fails
                  result /= datasets[d]
                                .fields[f]
                                .numVisibilitiesPerFreqPerStoke[i][s];
                }
              }

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

              // Use pre-computed effective number of samples (calculated once
              // before optimization)
              float N_eff = 0.0f;
              if (normalize) {
                N_eff = datasets[d].fields[f].N_eff_perFreqPerStoke[i][s];
                if (N_eff <= 0.0f) {
                  // Fallback to numVisibilities if N_eff was not pre-computed
                  N_eff = (float)datasets[d]
                              .fields[f]
                              .numVisibilitiesPerFreqPerStoke[i][s];
                }
              }

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
                    datasets[d].antennas[0].primary_beam, normalize, N_eff);
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
                    datasets[d].antennas[0].primary_beam, normalize, N_eff);
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

__host__ float isotropicTV(float* I,
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

__host__ void DIsotropicTV(float* I,
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

// Legacy function name for backward compatibility
__host__ float totalvariation(float* I,
                              float* ds,
                              float epsilon,
                              float penalization_factor,
                              int mod,
                              int order,
                              int index,
                              int iter) {
  return isotropicTV(I, ds, epsilon, penalization_factor, mod, order, index,
                     iter);
};

// Legacy function name for backward compatibility
__host__ void DTVariation(float* I,
                          float* dgi,
                          float epsilon,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  DIsotropicTV(I, dgi, epsilon, penalization_factor, mod, order, index, iter);
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

__host__ float anisotropicTV(float* I,
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
    ATVVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, epsilon, N, M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
};

__host__ void DAnisotropicTV(float* I,
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
      DATV<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, device_noise_image, epsilon, noise_cut, penalization_factor,
          N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
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

__host__ void calculateErrors(Image* image, float fg_scale) {
  float* errors = image->getErrorImage();

  cudaSetDevice(firstgpu);

  // Allocate error array: [I_nu_0, alpha, covariance] — 3 maps for 2 images.
  // Kernels accumulate inverse-variances (indices 0,1); noise_reduction
  // converts those to standard deviations. Output units:
  //   index 0: σ(I_nu_0)  [Jy/pixel] — uncertainty in flux at reference freq
  //   index 1: σ(alpha)    [unitless] — uncertainty in spectral index
  //   index 2: Cov(I_nu_0, alpha) [Jy/pixel] — covariance (same units as
  //   I_nu_0)
  int error_image_count = image->getImageCount() + 1;  // +1 for covariance
  checkCudaErrors(
      cudaMalloc((void**)&errors, sizeof(float) * M * N * error_image_count));
  checkCudaErrors(
      cudaMemset(errors, 0, sizeof(float) * M * N * error_image_count));
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
                // Compute variance for I_nu_0 (stored at index 0)
                I_nu_0_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), device_noise_image, noise_cut,
                    datasets[d].fields[f].nu[i], nu_0,
                    datasets[d].fields[f].device_visibilities[i][s].weight,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix, DELTAX, DELTAY,
                    sum_weights, fg_scale, N, M,
                    datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());

                // Compute variance for alpha (stored at index 1)
                alpha_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), datasets[d].fields[f].nu[i],
                    nu_0, device_noise_image, noise_cut, DELTAX, DELTAY,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff, sum_weights, fg_scale, N,
                    M, datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());

                // Compute covariance between I_nu_0 and alpha (stored at index
                // 2)
                covariance_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), datasets[d].fields[f].nu[i],
                    nu_0, device_noise_image, noise_cut, DELTAX, DELTAY,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff, sum_weights, fg_scale, N,
                    M, datasets[d].antennas[0].primary_beam);
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

  // Convert error units: if normalize=False (fg_scale ≠ 1.0), I_nu_0 is in code
  // units, so we need to multiply error by fg_scale to convert to Jy/pixel. If
  // normalize=True (fg_scale = 1.0), I_nu_0 is already in Jy/pixel, so no
  // conversion needed.
  if (fg_scale != 1.0f) {
    convertErrorUnits<<<numBlocksNN, threadsPerBlockNN>>>(errors, fg_scale, N,
                                                          M);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  image->setErrorImage(errors);
}

__host__ void precomputeNeff(bool normalize) {
  if (!normalize) {
    return;
  }

  cudaSetDevice(firstgpu);

  // Pre-compute N_eff for all datasets, fields, frequencies, and Stokes
  // parameters
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      // Initialize N_eff storage
      datasets[d].fields[f].N_eff_perFreqPerStoke.resize(
          datasets[d].data.total_frequencies,
          std::vector<float>(datasets[d].data.nstokes, 0.0f));

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
              // Compute sum of weights
              float sum_weights = deviceReduce<float>(
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);

              // Compute sum of squared weights (reuse device_chi2 as temp
              // storage)
              weightsSquaredVector<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  vars_gpu[gpu_idx].device_chi2,  // Temp storage
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              float sum_weights_squared = deviceReduce<float>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);

              // Calculate effective number of samples
              if (sum_weights_squared > 0.0f && sum_weights > 0.0f) {
                datasets[d].fields[f].N_eff_perFreqPerStoke[i][s] =
                    (sum_weights * sum_weights) / sum_weights_squared;
              } else {
                // Fallback to numVisibilities if calculation fails
                datasets[d].fields[f].N_eff_perFreqPerStoke[i][s] =
                    (float)datasets[d]
                        .fields[f]
                        .numVisibilitiesPerFreqPerStoke[i][s];
              }
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);
}
