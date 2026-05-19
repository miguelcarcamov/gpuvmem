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

#include "fft/fft_host.cuh"
#include "fft/fft_kernels.cuh"
#include "framework.cuh"
#include "error.cuh"
#include <cufft.h>
#include <cuda_runtime.h>

// Extern variables
extern dim3 threadsPerBlockNN, numBlocksNN;

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
