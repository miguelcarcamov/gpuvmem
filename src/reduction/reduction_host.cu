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

#include "reduction/reduction_host.cuh"
#include "reduction/reduction_kernels.cuh"
#include "utils/math_utils.hh"
#include "utils/cuda_utils.cuh"
#include "error.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <float.h>
#include <helper_cuda.h>
#include "reduction/reduction_kernels.cuh"
#include "utils/math_utils.hh"
#include "utils/cuda_utils.cuh"
#include "error.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>

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

// Explicit template instantiations
template __host__ float reduceCPU<float>(float* data, int size);
template __host__ double reduceCPU<double>(double* data, int size);
template __host__ float deviceReduce<float>(float* in, long N, int input_threads);
template __host__ double deviceReduce<double>(double* in, long N, int input_threads);
