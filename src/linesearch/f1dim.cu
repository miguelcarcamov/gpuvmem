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

#include "linesearch/line_search_1d_eval.cuh"
#include "linesearch/linesearcher.cuh"
#include "linesearch/linesearch_utils.cuh"
#include "error.cuh"
#include "framework.cuh"
#include "classes/image.cuh"
#include "classes/objectivefunction.cuh"
#include <iostream>

__host__ float lineSearch1dEval(const LineSearch1dEval* ctx, float x) {
  if (ctx == nullptr || ctx->device_pcom == nullptr || ctx->device_xicom == nullptr ||
      ctx->objective_function == nullptr) {
    std::cerr << "ERROR: lineSearch1dEval: invalid LineSearch1dEval context." << std::endl;
    return 0.0f;
  }

  ObjectiveFunction* objective_function = ctx->objective_function;
  // Must match device used for the main image / line-search buffers (MFS may leave another
  // GPU selected after multi-device setup).
  checkCudaErrors(cudaSetDevice(objective_function->getPrimaryCudaDevice()));

  Image* image_to_use = ctx->image;
  if (image_to_use == nullptr) {
    std::cerr << "ERROR: lineSearch1dEval: No Image in context." << std::endl;
    return 0.0f;
  }

  long M_local = image_to_use->getM();
  long N_local = image_to_use->getN();
  int image_count_local = image_to_use->getImageCount();

  float* device_xt = nullptr;
  checkCudaErrors(
      cudaMalloc((void**)&device_xt, sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMemset(device_xt, 0, sizeof(float) * M_local * N_local * image_count_local));
  imageMap* auxPtr = image_to_use->getFunctionMapping();
  for (int i = 0; i < image_count_local; i++) {
    if (auxPtr == nullptr || auxPtr[i].evaluateXt == nullptr) {
      std::cerr << "ERROR: lineSearch1dEval: imageMap / evaluateXt not configured." << std::endl;
      cudaFree(device_xt);
      return 0.0f;
    }
    (auxPtr[i].evaluateXt)(device_xt, ctx->device_pcom, ctx->device_xicom, x, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  float f = objective_function->calcFunction(device_xt);
  checkCudaErrors(cudaFree(device_xt));
  return f;
}

__host__ float lineSearch1dEvalThunk(float x, void* user) {
  return lineSearch1dEval(static_cast<const LineSearch1dEval*>(user), x);
}
