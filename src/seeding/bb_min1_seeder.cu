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

#include "seeding/bb_min1_seeder.cuh"
#include "optimizers/conjugategradient.cuh"  // For computeDotProduct and computeGradientDifference kernels
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include "classes/objectivefunction.cuh"
#include <cmath>

float BBMin1Seeder::seed(ObjectiveFunction* objective_function,
                         float* current_point, float* search_direction,
                         float* current_gradient, float prev_step_size,
                         float* prev_gradient, float* prev_point) {
  if (prev_gradient == nullptr || prev_point == nullptr) {
    return prev_step_size;  // Fallback to previous step size
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

  // Barzilai-Borwein Min1: α = min(α_BB1, α_BB2)
  // where α_BB1 = (s_k^T s_k) / (s_k^T y_k)
  //       α_BB2 = (s_k^T y_k) / (y_k^T y_k)
  // s_k = x_k - x_{k-1}, y_k = g_k - g_{k-1}
  
  // Allocate temporary memory
  float* s_k;  // Step: s_k = current_point - prev_point
  float* y_k;  // Gradient difference: y_k = current_gradient - prev_gradient
  float* dot_result;
  
  checkCudaErrors(
      cudaMalloc((void**)&s_k, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMalloc((void**)&y_k, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMalloc((void**)&dot_result, sizeof(float) * M * N * image_count));
  
  // Compute s_k = current_point - prev_point
  for (int i = 0; i < image_count; i++) {
    computeGradientDifference<<<numBlocksNN, threadsPerBlockNN>>>(
        s_k, current_point, prev_point, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  // Compute y_k = current_gradient - prev_gradient
  for (int i = 0; i < image_count; i++) {
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
  
  // Compute BB step sizes
  float alpha_bb1 = 0.0f;
  float alpha_bb2 = 0.0f;
  
  if (fabsf(sTy) > 1e-10f) {
    alpha_bb1 = sTs / sTy;
  }
  if (fabsf(yTy) > 1e-10f) {
    alpha_bb2 = sTy / yTy;
  }
  
  // Return minimum (Min1)
  if (alpha_bb1 > 0.0f && alpha_bb2 > 0.0f) {
    return std::min(alpha_bb1, alpha_bb2);
  } else if (alpha_bb1 > 0.0f) {
    return alpha_bb1;
  } else if (alpha_bb2 > 0.0f) {
    return alpha_bb2;
  }
  
  // Fallback
  return prev_step_size;
}

namespace {
StepSizeSeeder* CreateBBMin1Seeder() {
  return new BBMin1Seeder();
}

const std::string name = "BBMin1Seeder";
const bool RegisteredBBMin1Seeder =
    registerCreationFunction<StepSizeSeeder, std::string>(name, CreateBBMin1Seeder);
}  // namespace
