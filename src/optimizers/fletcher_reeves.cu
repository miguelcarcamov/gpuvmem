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

#include "optimizers/conjugategradient.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"

float FletcherReeves::computeConjugateGradientParameter(
    float* grad, float* grad_prev, float* dir_prev, float norm2_grad_prev) {
  // β_k^{FR} = ||g_{k+1}||^2 / ||g_k||^2
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  dim3 tpb = of->getThreadsPerBlockNN();
  dim3 nblk = of->getNumBlocksNN();
  size_t vector_size = sizeof(float) * M_local * N_local * image_count_local;
  checkCudaErrors(cudaMemset(device_gg_vector, 0, vector_size));

  for (int i = 0; i < image_count_local; i++) {
    computeNorm2Gradient<<<nblk, tpb>>>(
        device_gg_vector, grad, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  float norm2_grad = deviceReduce<float>(
      device_gg_vector, M_local * N_local * image_count_local,
      tpb.x * tpb.y);

  return norm2_grad / norm2_grad_prev;
}

// Factory registration function
namespace {
Optimizer* CreateFletcherReeves() {
  return new FletcherReeves;
}

const std::string name_fr = "CG-FletcherReeves";
const bool RegisteredFR =
    registerCreationFunction<Optimizer, std::string>(name_fr, CreateFletcherReeves);
}  // namespace
