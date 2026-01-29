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

#include "seeding/cubic_interpolation_seeder.cuh"
#include "optimizers/conjugategradient.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include "classes/objectivefunction.cuh"
#include <cmath>
#include <algorithm>

float CubicInterpolationSeeder::seed(ObjectiveFunction* objective_function,
                                     float* current_point,
                                     float* search_direction,
                                     float* current_gradient,
                                     float prev_step_size,
                                     float* prev_gradient,
                                     float* prev_point) {
  // Cubic interpolation based on function values and gradients
  // Uses f(x), f'(x), f(x_prev), f'(x_prev) to fit cubic polynomial
  // and find its minimum
  
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
  
  // Compute directional derivatives
  // f'(x) = ∇f(x)^T * d
  // f'(x_prev) = ∇f(x_prev)^T * d
  
  float* dot_result;
  checkCudaErrors(cudaMalloc((void**)&dot_result, sizeof(float) * M * N * image_count));
  
  // Compute current directional derivative
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, current_gradient, search_direction, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  float df_current = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  // Compute previous directional derivative (need to evaluate at prev_point)
  // For simplicity, use prev_gradient if available
  // Note: This is an approximation - ideally we'd evaluate gradient at prev_point
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, prev_gradient, search_direction, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  float df_prev = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  cudaFree(dot_result);
  
  // Cubic interpolation: minimize cubic polynomial
  // Using simplified formula: α ≈ -df_current / (df_current - df_prev) * prev_step_size
  // This is a heuristic approximation
  float df_diff = df_current - df_prev;
  if (fabsf(df_diff) > 1e-10f && df_current < 0.0f) {
    float alpha = -df_current / df_diff * prev_step_size;
    // Clamp to reasonable range
    return std::max(0.1f * prev_step_size, std::min(10.0f * prev_step_size, alpha));
  }
  
  return prev_step_size;
}

namespace {
StepSizeSeeder* CreateCubicInterpolationSeeder() {
  return new CubicInterpolationSeeder();
}

const std::string name = "CubicInterpolationSeeder";
const bool RegisteredCubicInterpolationSeeder =
    registerCreationFunction<StepSizeSeeder, std::string>(name, CreateCubicInterpolationSeeder);
}  // namespace
