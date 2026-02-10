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

#include "fft/padding.cuh"
#include "fft/fft_host.cuh"
#include "utils/math_utils.hh"
#include "utils/cuda_utils.cuh"
#include "utils/complexOps.cuh"
#include "framework.cuh"
#include "error.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// Extern variables
extern dim3 threadsPerBlockNN, numBlocksNN;
extern Vars variables;

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
