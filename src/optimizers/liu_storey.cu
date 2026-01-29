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

float LiuStorey::computeConjugateGradientParameter(
    float* grad, float* grad_prev, float* dir_prev, float norm2_grad_prev) {
  // β_k^{LS} = -(y_k^T g_{k+1}) / (g_k^T d_k), where y_k = g_{k+1} - g_k
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  dim3 tpb = of->getThreadsPerBlockNN();
  dim3 nblk = of->getNumBlocksNN();
  size_t vector_size = sizeof(float) * M_local * N_local * image_count_local;
  checkCudaErrors(cudaMemset(device_gg_vector, 0, vector_size));
  checkCudaErrors(cudaMemset(device_dgg_vector, 0, vector_size));

  for (int i = 0; i < image_count_local; i++) {
    computeGradientDifference<<<nblk, tpb>>>(
        device_dgg_vector, grad, grad_prev, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());

    computeDotProduct<<<nblk, tpb>>>(
        device_gg_vector, device_dgg_vector, grad, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  float numerator = -deviceReduce<float>(
      device_gg_vector, M_local * N_local * image_count_local,
      tpb.x * tpb.y);

  checkCudaErrors(cudaMemset(device_gg_vector, 0, vector_size));
  for (int i = 0; i < image_count_local; i++) {
    computeDotProduct<<<nblk, tpb>>>(
        device_gg_vector, dir_prev, grad_prev, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  float denominator = deviceReduce<float>(
      device_gg_vector, M_local * N_local * image_count_local,
      tpb.x * tpb.y);

  if (denominator == 0.0f) {
    return 0.0f;  // Fallback to steepest descent
  }

  return numerator / denominator;
}

// Factory registration function
namespace {
Optimizer* CreateLiuStorey() {
  return new LiuStorey;
}

const std::string name_ls = "CG-LiuStorey";
const bool RegisteredLS =
    registerCreationFunction<Optimizer, std::string>(name_ls, CreateLiuStorey);
}  // namespace
