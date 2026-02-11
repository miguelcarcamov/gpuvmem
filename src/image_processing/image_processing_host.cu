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

#include "image_processing/image_processing_host.cuh"
#include "image_processing/image_processing_kernels.cuh"
#include "error.cuh"
#include <helper_cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>

// Extern variables
extern dim3 threadsPerBlockNN, numBlocksNN;
extern long N, M;
extern float* device_noise_image;
extern float noise_cut, MINPIX, eta;

__host__ void normalizeImage(float* image, float normalization_factor) {
  normalizeImageKernel<<<numBlocksNN, threadsPerBlockNN>>>(
      image, normalization_factor, N);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkClipStokesWNoise(float* I, int nplanes) {
  clipStokesWNoise<<<numBlocksNN, threadsPerBlockNN>>>(
      I, nplanes, N, M, device_noise_image, noise_cut, MINPIX, eta);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkCopyItoInu(cufftComplex* image, float* I) {
  copyItoInu<<<numBlocksNN, threadsPerBlockNN>>>(image, I, M, N);
  checkCudaErrors(cudaDeviceSynchronize());
}
