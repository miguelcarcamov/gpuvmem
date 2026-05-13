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
#include "classes/objectivefunction.cuh"
#include "error.cuh"
#include "framework.cuh"
#include "linesearch/linesearch_kernels.cuh"
#include "factory.cuh"
#include "cli/gpuvmem_cli_config.hh"
#include <iostream>

#include "linesearch/line_search_1d_eval.cuh"

std::pair<float, float> Fixed::search(float* current_point,
                                      float* search_direction,
                                      ObjectiveFunction* objective_function,
                                      float* mask) {
  float alpha = fixed_step_size;  // Fixed step size ignores initial_alpha
  
  Image* image_to_use = this->image;
  if (image_to_use == nullptr) {
    std::cerr << "ERROR: Fixed::search: No Image object available!" << std::endl;
    return std::make_pair(0.0f, alpha);
  }
  
  long M_local = image_to_use->getM();
  long N_local = image_to_use->getN();
  int image_count_local = image_to_use->getImageCount();

  checkCudaErrors(cudaSetDevice(objective_function->getPrimaryCudaDevice()));

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

  LineSearch1dEval line_eval{local_device_pcom, local_device_xicom, image_to_use,
                             objective_function};
  LineSearcher::ScopedSearchContext _ls_ctx(this, objective_function, &line_eval);

  float f_value = this->evaluateLineFunction(alpha);

  // Update current point
  // Reuse image_to_use and dimensions from earlier in function
  imageMap* auxPtr = image_to_use->getFunctionMapping();
  if (!gpuvmem_cli_nopositivity()) {
    for (int i = 0; i < image_count_local; i++) {
      (auxPtr[i].newP)(current_point, search_direction, alpha, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  } else {
    dim3 threads_bb = objective_function->getThreadsPerBlockNN();
    dim3 blocks_bb = objective_function->getNumBlocksNN();
    for (int i = 0; i < image_count_local; i++) {
      newPNoPositivity<<<blocks_bb, threads_bb>>>(
          current_point, search_direction, alpha, N_local, M_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // Free temporary memory
  cudaFree(local_device_xicom);
  cudaFree(local_device_pcom);

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
