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
#include "linesearch/linesearch_utils.cuh"
#include "brent.cuh"
#include "mnbrak.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include <iostream>

// Forward declaration for wrapper function
extern LineSearcher* current_line_searcher;
__host__ float evaluateLineFunctionWrapper(float alpha);

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

std::pair<float, float> Brent::search(float* current_point,
                                      float* search_direction,
                                      ObjectiveFunction* objective_function,
                                      float* mask) {
  float xmin, fx, fb, fa, bx, ax;

  // Get Image object from line searcher member instead of extern Image* I
  Image* image_to_use = this->image;
  if (image_to_use == nullptr) {
    // Fallback to extern I for backward compatibility
    extern Image* I;
    image_to_use = I;
  }
  if (image_to_use == nullptr) {
    std::cerr << "ERROR: Brent::search: No Image object available!" << std::endl;
    return std::make_pair(0.0f, 0.0f);
  }
  
  long M_local = image_to_use->getM();
  long N_local = image_to_use->getN();
  int image_count_local = image_to_use->getImageCount();

  // Allocate device memory for line search (temporary, freed at end)
  float* local_device_pcom;
  float* local_device_xicom;
  
  checkCudaErrors(
      cudaMalloc((void**)&local_device_pcom, sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMalloc((void**)&local_device_xicom, sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(cudaMemset(local_device_pcom, 0,
                             sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(cudaMemset(local_device_xicom, 0,
                             sizeof(float) * M_local * N_local * image_count_local));
  
  // Set global pointers for f1dim to use
  device_pcom = local_device_pcom;
  device_xicom = local_device_xicom;

  // Copy current point and search direction
  checkCudaErrors(cudaMemcpy(device_pcom, current_point,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(device_xicom, search_direction,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));

  // Set global objective function for f1dim (used internally by evaluateLineFunction)
  testof = objective_function;
  nrfunc = nullptr;  // Not used, f1dim uses testof directly
  
  // Set current line searcher for wrapper function (uses evaluateLineFunction internally)
  extern LineSearcher* current_line_searcher;
  LineSearcher* prev_searcher = current_line_searcher;
  current_line_searcher = this;

  // Get initial step size from seeder/history/initial_step_size_value
  float initial_alpha = computeInitialAlpha(objective_function, current_point, search_direction);
  if (initial_alpha <= 0.0f) {
    initial_alpha = initial_step_size_value;  // Fallback to initial_step_size_value
  }

  // Bracket the minimum using wrapper function (calls evaluateLineFunction internally)
  ax = 0.0f;
  float xx = initial_alpha;
  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, evaluateLineFunctionWrapper);

  // Find minimum using Brent's method with wrapper function
  float fret = brent(ax, xx, bx, tolerance, &xmin, evaluateLineFunctionWrapper);
  
  // Restore the previous line searcher pointer (in case of nested calls)
  current_line_searcher = prev_searcher;

  if (verbose_flag) {
    printf("Alpha for linear minimization = %f\n\n", xmin);
  }

  // Update current point: p = p + xmin * search_direction
  // Reuse image_to_use and dimensions from earlier in function
  imageMap* auxPtr = image_to_use->getFunctionMapping();
  if (!nopositivity) {
    for (int i = 0; i < image_count_local; i++) {
      (auxPtr[i].newP)(current_point, search_direction, xmin, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  } else {
    for (int i = 0; i < image_count_local; i++) {
      newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(
          current_point, search_direction, xmin, N_local, M_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // Free temporary memory
  cudaFree(local_device_xicom);
  cudaFree(local_device_pcom);
  device_pcom = nullptr;
  device_xicom = nullptr;

  return std::make_pair(fret, xmin);
}

namespace {
LineSearcher* CreateBrent() {
  return new Brent();
}

const std::string name = "Brent";
const bool RegisteredBrent =
    registerCreationFunction<LineSearcher, std::string>(name, CreateBrent);
}  // namespace
