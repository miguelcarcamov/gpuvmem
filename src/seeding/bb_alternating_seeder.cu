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

#include "seeding/bb_alternating_seeder.cuh"
#include "optimizers/conjugategradient.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include "classes/objectivefunction.cuh"
#include <cmath>

float BBAlternatingSeeder::seed(ObjectiveFunction* objective_function,
                                float* current_point, float* search_direction,
                                float* current_gradient, float prev_step_size,
                                float* prev_gradient, float* prev_point) {
  if (prev_gradient == nullptr || prev_point == nullptr) {
    return prev_step_size;
  }
  
  if (objective_function == nullptr) {
    return prev_step_size;
  }

  // Get dimensions from objective_function
  long M = objective_function->getM();
  long N = objective_function->getN();
  int image_count = objective_function->getImageCount();
  dim3 threadsPerBlockNN = objective_function->getThreadsPerBlockNN();
  dim3 numBlocksNN = objective_function->getNumBlocksNN();

  // Ensure we're on firstgpu before allocating memory and launching kernels
  // (prev_gradient and prev_point are allocated on firstgpu)
  extern int firstgpu;
  cudaSetDevice(firstgpu);

  // Barzilai-Borwein Alternating: alternate between BB1 and BB2
  
  float* s_k;
  float* y_k;
  float* dot_result;
  
  checkCudaErrors(
      cudaMalloc((void**)&s_k, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMalloc((void**)&y_k, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMalloc((void**)&dot_result, sizeof(float) * M * N * image_count));
  
  // Compute s_k and y_k
  for (int i = 0; i < image_count; i++) {
    computeGradientDifference<<<numBlocksNN, threadsPerBlockNN>>>(
        s_k, current_point, prev_point, N, M, i);
    computeGradientDifference<<<numBlocksNN, threadsPerBlockNN>>>(
        y_k, current_gradient, prev_gradient, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  // Compute s_k^T s_k
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, s_k, s_k, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  float sTs = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  // Compute s_k^T y_k
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, s_k, y_k, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  float sTy = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  // Compute y_k^T y_k
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, y_k, y_k, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  float yTy = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  // Cleanup
  cudaFree(dot_result);
  cudaFree(y_k);
  cudaFree(s_k);
  
  // Alternate between BB1 and BB2
  use_bb1 = !use_bb1;
  
  if (use_bb1) {
    // BB1: α = (s_k^T s_k) / (s_k^T y_k)
    if (fabsf(sTy) > 1e-10f) {
      return sTs / sTy;
    }
  } else {
    // BB2: α = (s_k^T y_k) / (y_k^T y_k)
    if (fabsf(yTy) > 1e-10f) {
      return sTy / yTy;
    }
  }
  
  return prev_step_size;
}

namespace {
StepSizeSeeder* CreateBBAlternatingSeeder() {
  return new BBAlternatingSeeder();
}

const std::string name = "BBAlternatingSeeder";
const bool RegisteredBBAlternatingSeeder =
    registerCreationFunction<StepSizeSeeder, std::string>(name, CreateBBAlternatingSeeder);
}  // namespace
