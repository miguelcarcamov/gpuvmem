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

#include "optimizers/conjugategradient.cuh"
#include "linesearcher.cuh"  // Include here to avoid circular dependency
#include "linesearch/brent.cuh"  // For Brent class
#include "error.cuh"
#include "functions.cuh"
#include "factory.cuh"
#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>

// M, N, image_count are now accessed through Image object (image->getM(), getN(), getImageCount())
// Removed extern declarations to encourage using Image object

ObjectiveFunction* testof;
// Removed Image* I; - now using optimizer's image member instead

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int verbose_flag;
int flag_opt;

#define EPS 1.0e-10

#define FREEALL                \
  if (device_gg_vector) { cudaFree(device_gg_vector); device_gg_vector = nullptr; } \
  if (device_dgg_vector) { cudaFree(device_dgg_vector); device_dgg_vector = nullptr; } \
  if (xi) { cudaFree(xi); xi = nullptr; } \
  if (device_h) { cudaFree(device_h); device_h = nullptr; } \
  if (device_g) { cudaFree(device_g); device_g = nullptr; } \
  if (temp) { cudaFree(temp); temp = nullptr; }

// CUDA kernel implementations

__global__ void computeDotProduct(float* result, float* vec1, float* vec2,
                                  long N, long M, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < M && j < N) {
    result[M * N * image + N * i + j] =
        vec1[M * N * image + N * i + j] * vec2[M * N * image + N * i + j];
  }
}

__global__ void computeGradientDifference(float* grad_diff, float* grad,
                                          float* grad_prev, long N, long M,
                                          int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < M && j < N) {
    grad_diff[M * N * image + N * i + j] =
        grad[M * N * image + N * i + j] -
        grad_prev[M * N * image + N * i + j];
  }
}

__global__ void computeNorm2Gradient(float* result, float* grad, long N, long M,
                                    int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < M && j < N) {
    result[M * N * image + N * i + j] =
        grad[M * N * image + N * i + j] * grad[M * N * image + N * i + j];
  }
}

__global__ void updateSearchDirectionCG(float* search_dir, float* grad,
                                        float* prev_search_dir, float beta,
                                        long N, long M, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (i < M && j < N) {
    search_dir[M * N * image + N * i + j] =
        -grad[M * N * image + N * i + j] +
        beta * prev_search_dir[M * N * image + N * i + j];
  }
}

// Base class implementation

__host__ ConjugateGradient::ConjugateGradient() {
  // Default to Brent line search (current implementation)
  linesearcher_ptr = new Brent();
  prev_step_size = 1.0f;
}

__host__ void ConjugateGradient::setLineSearcher(std::unique_ptr<LineSearcher> searcher) {
  if (linesearcher_ptr != nullptr) {
    delete static_cast<LineSearcher*>(linesearcher_ptr);
  }
  linesearcher_ptr = searcher.release();
  // Set Image object in line searcher so it can use this->image instead of extern Image* I
  if (linesearcher_ptr != nullptr && image != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setImage(image);
  }
}

// setStepSizeSeeder removed - seeder is now owned by LineSearcher
// Use: linesearcher->setStepSizeSeeder(std::make_unique<BBMin1Seeder>());

__host__ void ConjugateGradient::allocateMemoryGpu() {
  // Free any existing memory first (in case optimize() is called multiple times)
  // This is important for multi-parameter optimization where optimize() may be called repeatedly
  if (device_g != nullptr || device_h != nullptr || xi != nullptr || 
      temp != nullptr || device_gg_vector != nullptr || device_dgg_vector != nullptr) {
    deallocateMemoryGpu();
  }
  
  // Configure ObjectiveFunction first if needed
  // Get dimensions from Image object instead of extern variables
  // This ensures dimensions are set before we try to allocate memory
  // Note: configured flag should be reset to 1 when switching between different optimization runs
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  if (configured) {
    of->configure(N_local, M_local, image_count_local);
    // Set CUDA launch configuration
    extern dim3 threadsPerBlockNN, numBlocksNN;
    of->setThreadsPerBlockNN(threadsPerBlockNN);
    of->setNumBlocksNN(numBlocksNN);
    configured = 0;
  }
  
  // Verify ObjectiveFunction dimensions match Image dimensions
  if (of->getM() != M_local || of->getN() != N_local || of->getImageCount() != image_count_local) {
    std::cerr << "WARNING: ObjectiveFunction dimensions don't match Image dimensions!" << std::endl;
    std::cerr << "  Image: M=" << M_local << ", N=" << N_local << ", image_count=" << image_count_local << std::endl;
    std::cerr << "  ObjectiveFunction: M=" << of->getM() << ", N=" << of->getN() << ", image_count=" << of->getImageCount() << std::endl;
  }
  
  // Validate dimensions before allocating
  if (M_local <= 0 || N_local <= 0 || image_count_local <= 0) {
    std::cerr << "ERROR: allocateMemoryGpu: Invalid dimensions from ObjectiveFunction!" << std::endl;
    std::cerr << "  M=" << M_local << ", N=" << N_local << ", image_count=" << image_count_local << std::endl;
    std::cerr << "  ObjectiveFunction must be configured before allocating memory!" << std::endl;
    return;
  }
  
  // Ensure we're on firstgpu before allocating memory
  // (dphi and other memory is allocated on firstgpu)
  extern int firstgpu;
  cudaSetDevice(firstgpu);
  
  size_t array_size = sizeof(float) * M_local * N_local * image_count_local;
  // device_gg_vector and device_dgg_vector are used by kernels that access
  // result[M * N * image + N * i + j], so they need size for all images
  size_t vector_size = sizeof(float) * M_local * N_local * image_count_local;
  
  checkCudaErrors(cudaMalloc((void**)&device_g, array_size));
  checkCudaErrors(cudaMemset(device_g, 0, array_size));
  checkCudaErrors(cudaMalloc((void**)&device_h, array_size));
  checkCudaErrors(cudaMemset(device_h, 0, array_size));
  checkCudaErrors(cudaMalloc((void**)&xi, array_size));
  checkCudaErrors(cudaMemset(xi, 0, array_size));
  checkCudaErrors(cudaMalloc((void**)&temp, array_size));
  checkCudaErrors(cudaMemset(temp, 0, array_size));
  
  checkCudaErrors(cudaMalloc((void**)&device_gg_vector, vector_size));
  checkCudaErrors(cudaMemset(device_gg_vector, 0, vector_size));
  
  checkCudaErrors(cudaMalloc((void**)&device_dgg_vector, vector_size));
  checkCudaErrors(cudaMemset(device_dgg_vector, 0, vector_size));
  
  // Verify xi was allocated successfully
  if (xi == nullptr) {
    std::cerr << "ERROR: allocateMemoryGpu: Failed to allocate xi!" << std::endl;
  }
}

__host__ void ConjugateGradient::deallocateMemoryGpu() {
  FREEALL
  // Note: Do NOT delete linesearcher_ptr here - it should persist across multiple
  // optimize() calls when switching between images (e.g., in optimizationOrder).
  // The line searcher is owned by ConjugateGradient and should only be deleted
  // in the destructor or when explicitly replaced via setLineSearcher().
}

__host__ float ConjugateGradient::initializeOptimizationState() {
  if (linesearcher_ptr != nullptr && image != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setImage(image);
  }
  flag_opt = this->flag;
  testof = of;

  float initial_function_value = of->calcFunction(image->getImage());
  if (verbose_flag) {
    std::cout << "Starting function value = " << std::setprecision(4)
              << std::fixed << initial_function_value << std::endl;
  }

  if (xi == nullptr) {
    std::cerr << "ERROR: initializeOptimizationState: xi is null! allocateMemoryGpu() must be called first!" << std::endl;
    return 0.0f;
  }
  of->calcGradient(image->getImage(), xi, 0);

  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  dim3 blocks = of->getNumBlocksNN();
  dim3 threads = of->getThreadsPerBlockNN();

  for (int i = 0; i < image_count_local; i++) {
    searchDirection<<<blocks, threads>>>(device_g, xi, device_h, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  prev_step_size = 1.0f;

  LineSearcher* searcher = static_cast<LineSearcher*>(linesearcher_ptr);
  if (searcher != nullptr) {
    searcher->setImage(image);
    if (searcher->getStepSizeSeeder() != nullptr) {
      searcher->updateHistory(of, image->getImage(), xi, 1.0f);
    }
  }

  return initial_function_value;
}

__host__ bool ConjugateGradient::checkFunctionConvergence(float new_value,
                                                           float prev_value) {
  return (2.0f * fabsf(new_value - prev_value) <=
          this->ftol * (fabsf(new_value) + fabsf(prev_value) + EPS));
}

__host__ bool ConjugateGradient::checkGradientConvergence(float* current_gradient,
                                                           float function_value) {
  long M_local = image->getM();
  long N_local = image->getN();
  int nimg = image->getImageCount();
  float den = std::max(function_value, 1.0f);
  dim3 blocks = of->getNumBlocksNN();
  dim3 threads = of->getThreadsPerBlockNN();

  for (int i = 0; i < nimg; i++) {
    CGGradCondition<<<blocks, threads>>>(
        temp, current_gradient, image->getImage(), den, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  float gmax = deviceMaxReduce(temp, M_local * N_local * nimg,
                               threads.x * threads.y);
  return (gmax < this->gtol);
}

__host__ float ConjugateGradient::conjugateGradientParameter(
    float* grad, float* grad_prev, float* dir_prev) {
  // Compute squared norm of previous gradient
  // Get dimensions from Image object instead of extern variables
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  size_t vector_size = sizeof(float) * M_local * N_local * image_count_local;
  checkCudaErrors(cudaMemset(device_gg_vector, 0, vector_size));
  
  dim3 blocks = of->getNumBlocksNN();
  dim3 threads = of->getThreadsPerBlockNN();
  for (int i = 0; i < image_count_local; i++) {
    computeNorm2Gradient<<<blocks, threads>>>(
        device_gg_vector, grad_prev, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  float norm2_grad_prev = deviceReduce<float>(
      device_gg_vector, M_local * N_local * image_count_local,
      threads.x * threads.y);

  if (norm2_grad_prev == 0.0f) {
    throw GradientNormError();
  }

  // Call method specialization
  return computeConjugateGradientParameter(grad, grad_prev, dir_prev,
                                          norm2_grad_prev);
}

__host__ float ConjugateGradient::performIteration(int iteration,
                                                   float prev_function_value) {
  double start = omp_get_wtime();
  this->current_iteration = iteration;

  if (verbose_flag) {
    std::cout << "\n\n********** Iteration " << iteration << " **********\n"
              << std::endl;
  }

  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  size_t grad_size = sizeof(float) * M_local * N_local * image_count_local;
  extern int firstgpu;
  cudaSetDevice(firstgpu);

  // Gradient at current point: only compute on first iteration (g_0). For iteration >= 2,
  // device_g already holds g_{k-1} from the end of the previous iteration (saved below).
  if (iteration == 1) {
    checkCudaErrors(cudaMemset(device_g, 0, grad_size));
    of->calcGradient(image->getImage(), device_g, iteration);
  }

  LineSearcher* searcher = static_cast<LineSearcher*>(linesearcher_ptr);
  if (searcher == nullptr) {
    std::cerr << "ERROR: performIteration: linesearcher_ptr is null!" << std::endl;
    return prev_function_value;
  }

  searcher->setImage(image);
  float* current_search_direction = device_h;

  auto result = searcher->search(image->getImage(), xi, of, nullptr);
  float new_function_value = result.first;
  float alpha_step = result.second;
  fret = new_function_value;

  if (verbose_flag) {
    std::cout << "Function value = " << std::setprecision(4) << std::fixed
              << new_function_value << std::endl;
  }

  // New gradient at the new point (after line search). Zero xi so no leftover
  // from search direction; cudaMemset is appropriate for zeroing.
  checkCudaErrors(cudaMemset(xi, 0, grad_size));
  cudaSetDevice(firstgpu);
  of->calcGradient(image->getImage(), xi, iteration);

  prev_step_size = searcher->computeNextInitialAlpha(of, image->getImage(),
                                                      current_search_direction,
                                                      xi);
  searcher->updateHistory(of, image->getImage(), xi, alpha_step);

  // Conjugate gradient parameter and search-direction update: use kernels
  // (conjugateGradientParameter uses reduce kernels; newXi does per-pixel math).
  float beta;
  try {
    beta = conjugateGradientParameter(xi, device_g, device_h);
  } catch (const GradientNormError&) {
    throw;
  }

  // Save g_{k+1} (in xi) to device_g for next iteration; then newXi overwrites xi and
  // needs a buffer for -g_{k+1}. Pass temp for that so device_g is not overwritten.
  for (int i = 0; i < image_count_local; i++) {
    checkCudaErrors(cudaMemcpy(&device_g[M_local * N_local * i], &xi[M_local * N_local * i],
                               sizeof(float) * M_local * N_local, cudaMemcpyDeviceToDevice));
  }
  dim3 blocks = of->getNumBlocksNN();
  dim3 threads = of->getThreadsPerBlockNN();
  for (int i = 0; i < image_count_local; i++) {
    newXi<<<blocks, threads>>>(temp, xi, device_h, beta, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  if (verbose_flag) {
    double end = omp_get_wtime();
    std::cout << "Time: " << std::setprecision(4) << (end - start)
              << " seconds" << std::endl;
  }

  return new_function_value;
}

__host__ void ConjugateGradient::optimize() {
  if (verbose_flag) {
    std::cout << "\n\nStarting " << methodName()
              << " method (Conj. Grad.)\n\n";
  }

  // Reset configuration flag to ensure proper setup for this optimization run
  // This is important for multi-parameter optimization where optimize() may be called multiple times
  configured = 1;
  
  allocateMemoryGpu();
  
  // Ensure line searcher has Image object set (use optimizer's image member)
  if (linesearcher_ptr != nullptr && image != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setImage(image);
  }

  float prev_function_value = initializeOptimizationState();

  for (int iteration = 1; iteration <= this->total_iterations; iteration++) {
    float new_function_value;
    try {
      new_function_value = performIteration(iteration, prev_function_value);
    } catch (const GradientNormError&) {
      // Zero gradient norm detected - optimization converged
      if (verbose_flag) {
        std::cout << methodName() << " converged due to zero gradient norm (gg = 0) after " 
                  << iteration << " iterations" << std::endl;
      }
      // Use optimizer's image member instead of extern Image* I
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    // Check for function convergence
    if (checkFunctionConvergence(new_function_value, prev_function_value)) {
      if (verbose_flag) {
        std::cout << methodName() << " converged after " << iteration
                  << " iterations" << std::endl;
      }
      // Use optimizer's image member instead of extern Image* I
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    // Check for gradient convergence (device_g holds current gradient; xi holds search direction)
    if (checkGradientConvergence(device_g, new_function_value)) {
      if (verbose_flag) {
        std::cout << methodName() << " converged due to gradient tolerance after " 
                  << iteration << " iterations" << std::endl;
      }
      // Use optimizer's image member instead of extern Image* I
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    prev_function_value = new_function_value;
  }

  if (verbose_flag) {
    std::cout << methodName() << " reached maximum iterations ("
              << this->total_iterations << ")" << std::endl;
  }

  // Use optimizer's image member instead of extern Image* I
  of->calcFunction(image->getImage());
  deallocateMemoryGpu();
}
