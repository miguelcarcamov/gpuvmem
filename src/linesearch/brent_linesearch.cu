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

#include "linesearch/brent.cuh"
#include "classes/image.cuh"
#include "linesearch/linesearch_utils.cuh"
#include "linesearch/line_search_1d_eval.cuh"
#include "linesearch/mnbrak.cuh"
#include "linesearch/nrutil.h"
#include "error.cuh"
#include "framework.cuh"
#include "factory.cuh"
#include "cli/gpuvmem_cli_config.hh"
#include <iostream>

std::pair<float, float> Brent::search(
    float* current_point, float* search_direction,
    ObjectiveFunction* objective_function,
    float* mask) {
  // Brent's method algorithm
  // Requires bracketed interval [a, b] with minimum inside
  
  const long M = objective_function->getM();
  const long N = objective_function->getN();
  const int image_count = objective_function->getImageCount();
  const size_t vec_bytes = sizeof(float) * static_cast<size_t>(M) * static_cast<size_t>(N) *
                           static_cast<size_t>(image_count);

  // Allocate temporary memory
  float* local_device_pcom;
  float* local_device_xicom;
  
  checkCudaErrors(cudaMalloc((void**)&local_device_pcom, vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&local_device_xicom, vec_bytes));
  
  checkCudaErrors(cudaMemcpy(local_device_pcom, current_point, vec_bytes,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(local_device_xicom, search_direction, vec_bytes,
                             cudaMemcpyDeviceToDevice));

  LineSearch1dEval line_eval{local_device_pcom, local_device_xicom, this->image,
                             objective_function};
  LineSearcher::ScopedSearchContext _ls_ctx(this, objective_function, &line_eval);
  
  // Get initial step size from seeder/history/initial_step_size_value
  float initial_alpha = computeInitialAlpha(objective_function, current_point, search_direction);
  if (initial_alpha <= 0.0f) {
    initial_alpha = initial_step_size_value;  // Fallback to initial_step_size_value
  }

  // Bracket the minimum first
  float ax = 0.0f;
  float xx = initial_alpha;
  float bx, fa, fx, fb;
  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, lineSearch1dEvalThunk, &line_eval);

  // Brent's method
  float xmin;
  float function_value =
      brent(ax, xx, bx, tolerance, &xmin, lineSearch1dEvalThunk, &line_eval);
  
  // Update point: p = p + xmin * d
  // Use Image object from line searcher member instead of extern Image* I
  updatePoint(objective_function, this->image, current_point, search_direction, xmin);
  
  // Clean up temporary memory
  cudaFree(local_device_pcom);
  cudaFree(local_device_xicom);
  
  if (gpuvmem_cli_verbose()) {
    printf("Alpha for linear minimization = %f\n\n", xmin);
  }
  
  return std::make_pair(function_value, xmin);
}

namespace {
LineSearcher* CreateBrent() {
  return new Brent();
}

const std::string name = "Brent";
const bool RegisteredBrent =
    registerCreationFunction<LineSearcher, std::string>(name, CreateBrent);
}  // namespace
