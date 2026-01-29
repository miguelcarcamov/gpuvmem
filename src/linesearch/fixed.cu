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

#include "linesearch/fixed.cuh"
#include "linesearch/linesearch_utils.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include <iostream>

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

// Global variables for f1dim (used by evaluateLineFunction)
extern float* device_pcom;
extern float *device_xicom, (*nrfunc)(float*);

std::pair<float, float> Fixed::search(float* current_point,
                                      float* search_direction,
                                      ObjectiveFunction* objective_function,
                                      float* mask) {
  float alpha = fixed_step_size;  // Fixed step size ignores initial_alpha
  
  // Get Image object from line searcher member instead of extern Image* I
  Image* image_to_use = this->image;
  if (image_to_use == nullptr) {
    // Fallback to extern I for backward compatibility
    extern Image* I;
    image_to_use = I;
  }
  if (image_to_use == nullptr) {
    std::cerr << "ERROR: Fixed::search: No Image object available!" << std::endl;
    return std::make_pair(0.0f, alpha);
  }
  
  long M_local = image_to_use->getM();
  long N_local = image_to_use->getN();
  int image_count_local = image_to_use->getImageCount();
  
  // Allocate temporary memory
  float* local_device_pcom;
  float* local_device_xicom;
  
  checkCudaErrors(
      cudaMalloc((void**)&local_device_pcom, sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMalloc((void**)&local_device_xicom, sizeof(float) * M_local * N_local * image_count_local));

  checkCudaErrors(cudaMemcpy(local_device_pcom, current_point,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(local_device_xicom, search_direction,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));

  // Set global pointers for f1dim (used by evaluateLineFunction)
  device_pcom = local_device_pcom;
  device_xicom = local_device_xicom;
  testof = objective_function;
  nrfunc = nullptr;

  float f_value = this->evaluateLineFunction(alpha);

  // Update current point
  // Reuse image_to_use and dimensions from earlier in function
  imageMap* auxPtr = image_to_use->getFunctionMapping();
  if (!nopositivity) {
    for (int i = 0; i < image_count_local; i++) {
      (auxPtr[i].newP)(current_point, search_direction, alpha, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  } else {
    for (int i = 0; i < image_count_local; i++) {
      newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(
          current_point, search_direction, alpha, N_local, M_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // Free temporary memory
  cudaFree(local_device_xicom);
  cudaFree(local_device_pcom);
  device_pcom = nullptr;
  device_xicom = nullptr;

  return std::make_pair(f_value, alpha);
}

namespace {
LineSearcher* CreateFixed() {
  return new Fixed();
}

const std::string name = "Fixed";
const bool RegisteredFixed =
    registerCreationFunction<LineSearcher, std::string>(name, CreateFixed);
}  // namespace
