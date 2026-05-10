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

#include "linesearch/linesearcher.cuh"
#include "linesearch/line_search_1d_eval.cuh"
#include "optimization/projection.hh"
#include "error.cuh"
#include <iostream>
#include "framework.cuh"
#include "optimizers/conjugategradient.cuh"  // For computeDotProduct kernel

LineSearcher::LineSearcher()
    : tolerance(1.0e-7f),
      initial_step_size_value(1.0f),
      image(nullptr),
      objective_function_(nullptr),
      active_line_1d_eval_(nullptr),
      projection_(std::make_unique<NoProjection>()),
      seeder_ptr(nullptr),
      prev_point(nullptr),
      prev_gradient(nullptr),
      prev_step_size(1.0f) {}

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

void LineSearcher::setImage(Image* im) { image = im; }

Image* LineSearcher::getImage() const { return image; }

StepSizeSeeder* LineSearcher::getStepSizeSeeder() const { return seeder_ptr; }

void LineSearcher::setTolerance(float tol) { tolerance = tol; }

float LineSearcher::getTolerance() const { return tolerance; }

void LineSearcher::setInitialStepSize(float initial_step_size) {
  initial_step_size_value = initial_step_size;
}

float LineSearcher::getInitialStepSize() const { return initial_step_size_value; }

void LineSearcher::setObjectiveFunction(ObjectiveFunction* of) { objective_function_ = of; }

ObjectiveFunction* LineSearcher::getObjectiveFunction() const { return objective_function_; }

const Projection* LineSearcher::getProjection() const { return projection_.get(); }

void LineSearcher::setProjection(std::unique_ptr<Projection> projection) {
  projection_ = std::move(projection);
  if (projection_ == nullptr) {
    projection_ = std::make_unique<NoProjection>();
  }
}

std::unique_ptr<Projection> LineSearcher::releaseProjection() {
  return std::move(projection_);
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
  if (active_line_1d_eval_ != nullptr) {
    return lineSearch1dEval(active_line_1d_eval_, alpha);
  }
  std::cerr << "ERROR: evaluateLineFunction: no active LineSearch1dEval (missing "
               "ScopedSearchContext with line_eval in LineSearcher::search?)."
            << std::endl;
  return 0.0f;
}

// Global pointer for wrapper function (used by Brent and other Numerical Recipes routines)
// that need a C-style function pointer
namespace {
LineSearcher* current_line_searcher = nullptr;
}  // namespace

LineSearcher* lineSearchGetCurrent() {
  return current_line_searcher;
}

__host__ float evaluateLineFunctionWrapper(float alpha) {
  LineSearcher* cur = lineSearchGetCurrent();
  if (cur != nullptr) {
    return cur->evaluateLineFunction(alpha);
  }
  std::cerr << "ERROR: evaluateLineFunctionWrapper: no active LineSearcher." << std::endl;
  return 0.0f;
}

LineSearcher::ScopedSearchContext::ScopedSearchContext(LineSearcher* owner,
                                                      ObjectiveFunction* of,
                                                      const LineSearch1dEval* line_eval)
    : owner_(owner),
      prev_current_(current_line_searcher),
      prev_line_eval_(owner->active_line_1d_eval_) {
  current_line_searcher = owner_;
  owner_->setObjectiveFunction(of);
  owner_->active_line_1d_eval_ = line_eval;
}

LineSearcher::ScopedSearchContext::~ScopedSearchContext() {
  current_line_searcher = prev_current_;
  owner_->active_line_1d_eval_ = prev_line_eval_;
  owner_->setObjectiveFunction(nullptr);
}
