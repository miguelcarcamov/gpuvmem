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

#include "optimizers/optimizer_host.cuh"
#include "optimizers/optimizer_kernels.cuh"
#include "linesearch/linesearch_kernels.cuh"
#include "error.cuh"
#include "framework.cuh"
#include <cuda_runtime.h>

// Extern variables
extern dim3 threadsPerBlockNN, numBlocksNN;
extern long N, M;
extern int firstgpu;
extern float eta, MINPIX;
extern float* initial_values;

// Forward declaration for accessing Image object from updatePoint
// This avoids needing extern global Image* I
__host__ Image* getCurrentImage();

__host__ void linkRestartDGi(float* dgi) {
  cudaSetDevice(firstgpu);
  // restartDPhi signature: restartDPhi(float* dphi, float* dChi2, float* dH, long N)
  // For restart, we set dphi = dgi, dChi2 = dgi, dH = dgi (all point to same array)
  restartDPhi<<<numBlocksNN, threadsPerBlockNN>>>(dgi, dgi, dgi, N);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkAddToDPhi(float* dphi, float* dgi, int index) {
  cudaSetDevice(firstgpu);
  AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(dphi, dgi, N, M, index);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void defaultNewP(float* p, float* xi, float xmin, int image) {
  // Ensure we're on firstgpu before launching kernel
  cudaSetDevice(firstgpu);
  newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M,
                                                       image);
}

__host__ void particularNewP(float* p, float* xi, float xmin, int image) {
  // Ensure we're on firstgpu before launching kernel
  cudaSetDevice(firstgpu);

  // Access Image object through thread-local variable set in updatePoint()
  // This avoids needing extern global variable
  Image* current_image = getCurrentImage();

  // Get dimensions and minimal pixel value from Image object instead of extern variables
  long M_local = current_image ? current_image->getM() : M;  // Fallback to extern M if not set
  long N_local = current_image ? current_image->getN() : N;   // Fallback to extern N if not set
  float min_pixel_value = current_image ? current_image->getMinimalPixelValue(image) : MINPIX;  // Use Image's minimal pixel value

  newP<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N_local, M_local,
                                           min_pixel_value, eta, image);
}

__host__ void defaultEvaluateXt(float* xt,
                                float* pcom,
                                float* xicom,
                                float x,
                                int image) {
  evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(xt, pcom, xicom, x,
                                                             N, M, image);
}

__host__ void particularEvaluateXt(float* xt,
                                   float* pcom,
                                   float* xicom,
                                   float x,
                                   int image) {
  evaluateXt<<<numBlocksNN, threadsPerBlockNN>>>(
      xt, pcom, xicom, x, N, M, initial_values[image], eta, image);
}
