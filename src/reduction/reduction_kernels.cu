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

#include "reduction/reduction_kernels.cuh"
#include <cooperative_groups.h>
#include <float.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace cg = cooperative_groups;

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

  float myMax = -CUDART_INF_F;

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

// Explicit template instantiations for reduction kernels
// These are needed so the linker can find the template instantiations

// reduceSumKernel instantiations for float
template __global__ void reduceSumKernel<float, 1, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 1, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 2, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 2, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 4, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 4, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 8, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 8, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 16, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 16, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 32, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 32, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 64, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 64, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 128, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 128, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 256, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 256, false>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 512, true>(float*, float*, unsigned int);
template __global__ void reduceSumKernel<float, 512, false>(float*, float*, unsigned int);

// reduceSumKernel instantiations for double
template __global__ void reduceSumKernel<double, 1, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 1, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 2, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 2, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 4, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 4, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 8, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 8, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 16, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 16, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 32, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 32, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 64, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 64, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 128, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 128, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 256, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 256, false>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 512, true>(double*, double*, unsigned int);
template __global__ void reduceSumKernel<double, 512, false>(double*, double*, unsigned int);

// reduceMinKernel instantiations
template __global__ void reduceMinKernel<1, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<1, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<2, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<2, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<4, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<4, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<8, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<8, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<16, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<16, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<32, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<32, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<64, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<64, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<128, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<128, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<256, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<256, false>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<512, true>(float*, float*, unsigned int);
template __global__ void reduceMinKernel<512, false>(float*, float*, unsigned int);

// reduceMaxKernel instantiations
template __global__ void reduceMaxKernel<1, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<1, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<2, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<2, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<4, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<4, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<8, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<8, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<16, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<16, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<32, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<32, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<64, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<64, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<128, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<128, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<256, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<256, false>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<512, true>(float*, float*, unsigned int);
template __global__ void reduceMaxKernel<512, false>(float*, float*, unsigned int);
