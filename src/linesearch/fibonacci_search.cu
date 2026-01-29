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

#include "linesearch/fibonacci_search.cuh"
#include "linesearch/linesearch_utils.cuh"
#include "f1dim.cuh"
#include "mnbrak.cuh"
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include <iostream>
#include <vector>

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

std::pair<float, float> FibonacciSearch::search(
    float* current_point, float* search_direction,
    ObjectiveFunction* objective_function,
    float* mask) {
  // Fibonacci search algorithm
  // Requires bracketed interval [a, b] with minimum inside
  
  // Allocate temporary memory
  float* local_device_pcom;
  float* local_device_xicom;
  
  checkCudaErrors(
      cudaMalloc((void**)&local_device_pcom, sizeof(float) * M * N * image_count));
  checkCudaErrors(
      cudaMalloc((void**)&local_device_xicom, sizeof(float) * M * N * image_count));
  
  checkCudaErrors(cudaMemcpy(local_device_pcom, current_point,
                             sizeof(float) * M * N * image_count,
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(local_device_xicom, search_direction,
                             sizeof(float) * M * N * image_count,
                             cudaMemcpyDeviceToDevice));
  
  device_pcom = local_device_pcom;
  device_xicom = local_device_xicom;
  testof = objective_function;
  nrfunc = nullptr;
  
  // Get initial step size from seeder/history/initial_step_size_value
  float initial_alpha = computeInitialAlpha(objective_function, current_point, search_direction);
  if (initial_alpha <= 0.0f) {
    initial_alpha = initial_step_size_value;  // Fallback to initial_step_size_value
  }

  // Bracket the minimum first
  float ax = 0.0f;
  float xx = initial_alpha;
  float bx, fa, fx, fb;
  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, f1dim);
  
  // Generate Fibonacci numbers
  std::vector<long long> fib(max_iterations + 2);
  fib[0] = 1;
  fib[1] = 1;
  for (int i = 2; i < max_iterations + 2; i++) {
    fib[i] = fib[i-1] + fib[i-2];
  }
  
  // Find number of iterations needed
  float interval_length = bx - ax;
  int n = 0;
  for (int i = 0; i < max_iterations; i++) {
    if (fib[i+2] * tolerance >= interval_length) {
      n = i;
      break;
    }
  }
  if (n == 0) n = max_iterations;
  
  // Fibonacci search
  float a = ax;
  float b = bx;
  float c = a + (fib[n] / (float)fib[n+2]) * (b - a);
  float d = a + (fib[n+1] / (float)fib[n+2]) * (b - a);
  float fc = this->evaluateLineFunction(c);
  float fd = this->evaluateLineFunction(d);
  
  for (int i = n; i >= 1; i--) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = a + (fib[i-1] / (float)fib[i+1]) * (b - a);
      fc = this->evaluateLineFunction(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + (fib[i] / (float)fib[i+1]) * (b - a);
      fd = this->evaluateLineFunction(d);
    }
  }
  
  float xmin = (a + b) / 2.0f;
  float fret = this->evaluateLineFunction(xmin);
  
  // Update point
  // Use Image object from line searcher member instead of extern Image* I
  updatePoint(objective_function, this->image, current_point, search_direction, xmin);
  
  // Cleanup
  cudaFree(local_device_xicom);
  cudaFree(local_device_pcom);
  device_pcom = nullptr;
  device_xicom = nullptr;
  
  if (verbose_flag) {
    printf("Alpha for linear minimization = %f\n\n", xmin);
  }
  
  return std::make_pair(fret, xmin);
}

namespace {
LineSearcher* CreateFibonacciSearch() {
  return new FibonacciSearch();
}

const std::string name = "FibonacciSearch";
const bool RegisteredFibonacciSearch =
    registerCreationFunction<LineSearcher, std::string>(name, CreateFibonacciSearch);
}  // namespace
