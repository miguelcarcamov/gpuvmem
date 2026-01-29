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

#include "linesearcher.cuh"
#include "f1dim.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "optimizers/conjugategradient.cuh"  // For computeDotProduct kernel

extern long M;
extern long N;
extern int image_count;

LineSearcher::~LineSearcher() {
  if (seeder_ptr != nullptr) {
    delete seeder_ptr;
    seeder_ptr = nullptr;
  }
  if (prev_point != nullptr) {
    cudaFree(prev_point);
    prev_point = nullptr;
  }
  if (prev_gradient != nullptr) {
    cudaFree(prev_gradient);
    prev_gradient = nullptr;
  }
}

void LineSearcher::setStepSizeSeeder(std::unique_ptr<StepSizeSeeder> seeder) {
  if (seeder_ptr != nullptr) {
    delete seeder_ptr;
  }
  seeder_ptr = seeder.release();
  
  // Note: Memory allocation for history is deferred until updateHistory is called
  // with valid ObjectiveFunction dimensions, since M, N, image_count are not available here
}

float LineSearcher::computeInitialAlpha(ObjectiveFunction* objective_function,
                                       float* current_point, float* search_direction) {
  // Check if we have history from a previous iteration (not just initialization)
  // History exists if prev_point and prev_gradient are set AND prev_step_size has been
  // updated from a real line search (not just the initialization in initializeOptimizationState)
  bool has_real_history = (prev_point != nullptr && prev_gradient != nullptr && 
                           prev_step_size != 1.0f);
  
  if (has_real_history) {
    // We have history from a previous iteration - use seeder or prev_step_size
    
    // First try to use seeder if available
    if (seeder_ptr != nullptr && objective_function != nullptr) {
      // Get current gradient (should be computed before line search)
      float* current_gradient = objective_function->getCurrentGradient();
      if (current_gradient != nullptr) {
        // Call seeder to get initial step size estimate based on history
        float initial_alpha = seeder_ptr->seed(objective_function, current_point, search_direction,
                                               current_gradient, prev_step_size,
                                               prev_gradient, prev_point);
        if (initial_alpha > 0.0f) {
          return initial_alpha;
        }
      }
    }
    
    // Seeder not available or failed - use previous step size from history
    return prev_step_size;
  }
  
  // No real history yet (first iteration):
  // Use initial_step_size_value (set via setInitialStepSize, defaults to 1.0f)
  return initial_step_size_value;
}

float LineSearcher::computeNextInitialAlpha(ObjectiveFunction* objective_function,
                                            float* current_point, float* search_direction,
                                            float* current_gradient) {
  // If no seeder, use prev_step_size (the step size from current iteration)
  if (seeder_ptr == nullptr) {
    return prev_step_size > 0.0f ? prev_step_size : initial_step_size_value;
  }
  
  if (objective_function == nullptr) {
    return prev_step_size > 0.0f ? prev_step_size : initial_step_size_value;
  }
  
  // Check if M, N, image_count are valid (get from objective_function)
  if (objective_function->getM() <= 0 || objective_function->getN() <= 0 || 
      objective_function->getImageCount() <= 0) {
    return prev_step_size > 0.0f ? prev_step_size : initial_step_size_value;
  }
  
  // If no history yet, use prev_step_size or initial_step_size_value
  // Note: prev_point/prev_gradient should be initialized by initializeOptimizationState
  // or updateHistory. If they're nullptr here, history hasn't been set up yet.
  if (prev_point == nullptr || prev_gradient == nullptr) {
    return prev_step_size > 0.0f ? prev_step_size : initial_step_size_value;
  }
  
  // Call seeder with current and previous gradients/history
  // prev_step_size contains the step size from the current iteration
  float initial_alpha = seeder_ptr->seed(objective_function, current_point, search_direction,
                                         current_gradient, prev_step_size,
                                         prev_gradient, prev_point);
  
  // Ensure positive step size
  if (initial_alpha <= 0.0f) {
    return prev_step_size > 0.0f ? prev_step_size : initial_step_size_value;
  }
  
  return initial_alpha;
}

void LineSearcher::updateHistory(ObjectiveFunction* objective_function,
                                 float* current_point, float* current_gradient, float step_size) {
  // Only update history if seeder is set
  if (seeder_ptr == nullptr) {
    return;
  }
  
  if (objective_function == nullptr) {
    return;
  }
  
  // Get dimensions from objective_function
  long M = objective_function->getM();
  long N = objective_function->getN();
  int image_count = objective_function->getImageCount();
  
  // Ensure we're on firstgpu before allocating memory and copying
  // (same device context as where dphi and xi are allocated)
  extern int firstgpu;
  cudaSetDevice(firstgpu);
  
  // Ensure memory is allocated
  if (prev_point == nullptr) {
    checkCudaErrors(cudaMalloc((void**)&prev_point, sizeof(float) * M * N * image_count));
  }
  if (prev_gradient == nullptr) {
    checkCudaErrors(cudaMalloc((void**)&prev_gradient, sizeof(float) * M * N * image_count));
  }
  
  // Update history
  checkCudaErrors(cudaMemcpy(prev_point, current_point,
                             sizeof(float) * M * N * image_count,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(prev_gradient, current_gradient,
                             sizeof(float) * M * N * image_count,
                             cudaMemcpyDeviceToDevice));
  prev_step_size = step_size;
}

float LineSearcher::evaluateLineFunction(float alpha) {
  // Use f1dim internally for consistency with Brent and other Numerical Recipes methods
  return f1dim(alpha);
}

// Global pointer for wrapper function (used by Brent and other Numerical Recipes routines)
// that need a C-style function pointer
LineSearcher* current_line_searcher = nullptr;

__host__ float evaluateLineFunctionWrapper(float alpha) {
  if (current_line_searcher != nullptr) {
    return current_line_searcher->evaluateLineFunction(alpha);
  }
  // Fallback to f1dim if no line searcher is set
  return f1dim(alpha);
}
