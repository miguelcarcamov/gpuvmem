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

#include "linesearch/fista_backtracking.cuh"
#include "linesearch/linesearch_utils.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "optimizers/conjugategradient.cuh"  // For computeDotProduct kernel declaration
#include "factory.cuh"
#include <iostream>
#include <cmath>

extern long M;
extern long N;
extern int image_count;
extern float MINPIX, eta;
extern bool nopositivity;
extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern int verbose_flag;
extern Image* I;
extern ObjectiveFunction* testof;

// Global variables for f1dim
extern float* device_pcom;
extern float *device_xicom, (*nrfunc)(float*);

std::pair<float, float> FistaBacktracking::search(
    float* current_point, float* search_direction,
    ObjectiveFunction* objective_function,
    float* mask) {
  // FISTA backtracking line search
  // Similar to Armijo but with adaptive step size increase
  
  // Get initial step size from seeder/history/initial_step_size_value
  float alpha = computeInitialAlpha(objective_function, current_point, search_direction);
  if (alpha <= 0.0f) {
    alpha = initial_step_size_value;  // Fallback to initial_step_size_value
  }
  
  // Get dimensions from ObjectiveFunction (avoid extern variables)
  long M_local = objective_function->getM();
  long N_local = objective_function->getN();
  int image_count_local = objective_function->getImageCount();
  
  // Ensure we're on firstgpu before allocating memory
  extern int firstgpu;
  cudaSetDevice(firstgpu);
  
  // Allocate temporary memory using ObjectiveFunction dimensions
  float* local_device_pcom;
  float* local_device_xicom;
  float* temp_point;
  
  size_t array_size = sizeof(float) * M_local * N_local * image_count_local;
  checkCudaErrors(cudaMalloc((void**)&local_device_pcom, array_size));
  checkCudaErrors(cudaMalloc((void**)&local_device_xicom, array_size));
  checkCudaErrors(cudaMalloc((void**)&temp_point, array_size));
  
  checkCudaErrors(cudaMemcpy(local_device_pcom, current_point, array_size,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(local_device_xicom, search_direction, array_size,
                             cudaMemcpyDeviceToDevice));
  
  device_pcom = local_device_pcom;
  device_xicom = local_device_xicom;
  testof = objective_function;
  nrfunc = nullptr;
  
  // Compute function value at current point
  float f0 = objective_function->calcFunction(current_point);
  
  // Get gradient from ObjectiveFunction (dphi) - optimizer must call calcGradient first!
  float* gradient = objective_function->getCurrentGradient();
  if (gradient == nullptr) {
    std::cerr << "ERROR: FistaBacktracking requires gradient but ObjectiveFunction::getCurrentGradient() returned nullptr!" << std::endl;
    std::cerr << "The optimizer must call calcGradient before calling line search." << std::endl;
    return std::make_pair(f0, 0.0f);
  }
  
  // Compute directional derivative ∇f(x)^T*d
  float* dot_result;
  // computeDotProduct kernel accesses result[M * N * image + N * i + j]
  // So we need to allocate for all images, not just one
  checkCudaErrors(cudaMalloc((void**)&dot_result, sizeof(float) * M * N * image_count));
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, gradient, search_direction, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  // Reduce across all images: sum dot products from all images
  float dir_deriv = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  cudaFree(dot_result);
  
  // FISTA backtracking loop with adaptive step size
  int max_iterations = 50;
  float alpha_prev = alpha;
  
  for (int iter = 0; iter < max_iterations; iter++) {
    // Evaluate function at new point using base class method (uses f1dim internally)
    float f_alpha = this->evaluateLineFunction(alpha);
    
    // Armijo condition: f(x + αd) ≤ f(x) + c*α*∇f(x)^T*d
    if (f_alpha <= f0 + c * alpha * dir_deriv) {
      // Condition satisfied - update point and return
      checkCudaErrors(cudaMemcpy(temp_point, current_point,
                                 sizeof(float) * M * N * image_count,
                                 cudaMemcpyDeviceToDevice));
      // Use Image object from line searcher member instead of extern Image* I
      updatePoint(objective_function, this->image, temp_point, search_direction, alpha);
      checkCudaErrors(cudaMemcpy(current_point, temp_point,
                                 sizeof(float) * M * N * image_count,
                                 cudaMemcpyDeviceToDevice));
      
      // Update history for seeder (if seeder is set)
      updateHistory(objective_function, current_point, gradient, alpha);
      
      // Cleanup
      cudaFree(temp_point);
      cudaFree(local_device_xicom);
      cudaFree(local_device_pcom);
      // Don't free gradient - it's owned by the optimizer, not allocated here
      device_pcom = nullptr;
      device_xicom = nullptr;
      
      if (verbose_flag) {
        printf("Alpha for linear minimization = %f\n\n", alpha);
      }
      
      return std::make_pair(f_alpha, alpha);
    }
    
    // Backtrack: reduce step size
    alpha_prev = alpha;
    alpha *= 0.5f;  // Standard backtracking factor
    
    // FISTA can also increase step size if previous step was too small
    // This is a simplified version - full FISTA has more sophisticated logic
  }
  
  // If we get here, backtracking failed - use minimum alpha found
  alpha = 1.0f * std::pow(0.5f, max_iterations - 1);
  float f_final = this->evaluateLineFunction(alpha);
  checkCudaErrors(cudaMemcpy(temp_point, current_point, array_size,
                             cudaMemcpyDeviceToDevice));
  updatePoint(objective_function, I, temp_point, search_direction, alpha);
  checkCudaErrors(cudaMemcpy(current_point, temp_point, array_size,
                             cudaMemcpyDeviceToDevice));
  
  // Update history for seeder (if seeder is set)
  // Use the gradient that was computed after line search (set by optimizer)
  // The optimizer will call calcGradient after line search and update it
  // For now, use the gradient we already have (at the starting point)
  updateHistory(objective_function, current_point, gradient, alpha);
  
  // Cleanup
  cudaFree(temp_point);
  cudaFree(local_device_xicom);
  cudaFree(local_device_pcom);
  device_pcom = nullptr;
  device_xicom = nullptr;
  
  if (verbose_flag) {
    printf("Alpha for linear minimization = %f\n\n", alpha);
  }
  
  return std::make_pair(f_final, alpha);
}

namespace {
LineSearcher* CreateFistaBacktracking() {
  return new FistaBacktracking();
}

const std::string name = "FistaBacktracking";
const bool RegisteredFistaBacktracking =
    registerCreationFunction<LineSearcher, std::string>(name, CreateFistaBacktracking);
}  // namespace
