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

#include "optimizers/lbfgs.cuh"
#include "linesearcher.cuh"  // Include here to avoid circular dependency
#include "linesearch/brent.cuh"  // For Brent class
#include "error.cuh"
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cstring>

// M, N, image_count are now accessed through Image object (image->getM(), getN(), getImageCount())
// Removed extern declarations to encourage using Image object

extern ObjectiveFunction* testof;
extern Image* I;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int verbose_flag;
extern int flag_opt;

#define EPS 1.0e-10
#define MIN_Y_NORM 1e-10f  // Minimum value for ||y||^2 to avoid division by zero

#define FREEALL     \
  cudaFree(d_y);    \
  cudaFree(d_s);    \
  cudaFree(xi);     \
  cudaFree(xi_old); \
  cudaFree(p_old);  \
  cudaFree(norm_vector);

__host__ int LBFGS::getK() {
  return this->K;
}

__host__ void LBFGS::setK(int K) {
  this->K = K;
}

__host__ void LBFGS::allocateMemoryGpu() {
  // Get dimensions from Image object instead of extern variables
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  checkCudaErrors(cudaMalloc(
      (void**)&d_y, sizeof(float) * M_local * N_local * K * image_count_local));
  checkCudaErrors(
      cudaMemset(d_y, 0, sizeof(float) * M_local * N_local * K * image_count_local));

  checkCudaErrors(cudaMalloc(
      (void**)&d_s, sizeof(float) * M_local * N_local * K * image_count_local));
  checkCudaErrors(
      cudaMemset(d_s, 0, sizeof(float) * M_local * N_local * K * image_count_local));

  checkCudaErrors(cudaMalloc((void**)&p_old,
                             sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMemset(p_old, 0, sizeof(float) * M_local * N_local * image_count_local));

  checkCudaErrors(
      cudaMalloc((void**)&xi, sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMemset(xi, 0, sizeof(float) * M_local * N_local * image_count_local));

  checkCudaErrors(cudaMalloc((void**)&xi_old,
                             sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMemset(xi_old, 0, sizeof(float) * M_local * N_local * image_count_local));

  checkCudaErrors(cudaMalloc((void**)&norm_vector,
                             sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(cudaMemset(norm_vector, 0,
                             sizeof(float) * M_local * N_local * image_count_local));
}

__host__ LBFGS::LBFGS() {
  // Default to Brent line search (current implementation)
  linesearcher_ptr = new Brent();
  prev_step_size = 1.0f;
  // Note: Image object will be set in setLineSearcher() or performIteration()
  // when image member is available
}

__host__ void LBFGS::setLineSearcher(std::unique_ptr<LineSearcher> searcher) {
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

__host__ void LBFGS::deallocateMemoryGpu() {
  FREEALL
  // Cleanup line searcher (seeder is owned by LineSearcher, cleaned up automatically)
  if (linesearcher_ptr != nullptr) {
    delete static_cast<LineSearcher*>(linesearcher_ptr);
    linesearcher_ptr = nullptr;
  }
}

__host__ int LBFGS::mapToCircularBuffer(int k, int par_M, int lbfgs_it) {
  // Map logical index k to circular buffer index
  // Most recent is lbfgs_it, previous is (lbfgs_it-1+K)%K, etc.
  return (lbfgs_it - (par_M - 1 - k) + this->K) % this->K;
}

__host__ float LBFGS::initializeOptimizationState() {
  // Set Image object in line searcher so it can use this->image instead of extern Image* I
  if (linesearcher_ptr != nullptr && image != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setImage(image);
  }
  flag_opt = this->flag;
  testof = of;
  
  // Get dimensions from Image object instead of extern variables
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  if (configured) {
    of->configure(N_local, M_local, image_count_local);
    configured = 0;
  }

  // Note: prev_point is now managed by LineSearcher (if seeder is set)
  // LBFGS uses p_old for its own history (correction pairs)

  float initial_function_value = of->calcFunction(image->getImage());
  
  if (verbose_flag) {
    std::cout << "Starting function value = " << std::setprecision(4)
              << std::fixed << initial_function_value << std::endl;
  }

  // Compute initial gradient
  of->calcGradient(image->getImage(), xi, 0);

  // Initialize search direction as negative gradient (steepest descent)
  // Reuse dimensions from earlier in function (already declared at lines 149-151)
  for (int i = 0; i < image_count_local; i++) {
    searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(
        xi, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  prev_step_size = 1.0f;  // Initial step size

  return initial_function_value;
}

__host__ bool LBFGS::checkFunctionConvergence(float new_value, float prev_value) {
  return ((prev_value - new_value) / std::max({fabsf(new_value), fabsf(prev_value), 1.0f}) <= this->ftol);
}

__host__ bool LBFGS::checkGradientConvergence() {
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  for (int i = 0; i < image_count_local; i++) {
    normArray<<<numBlocksNN, threadsPerBlockNN>>>(norm_vector, xi, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  max_per_it = deviceMaxReduce(norm_vector, M_local * N_local * image_count_local,
                               threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  return (max_per_it <= this->gtol);
}

__host__ float LBFGS::computeScalingFactor(int par_M, int lbfgs_it) {
  // Use oldest iteration in history for scaling factor (standard LBFGS)
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  int oldest_idx = mapToCircularBuffer(0, par_M, lbfgs_it);
  float sy = 0.0f;
  float yy = 0.0f;
  float sy_yy = 0.0f;
  float* temp_aux;

  checkCudaErrors(cudaMalloc((void**)&temp_aux, sizeof(float) * M_local * N_local));
  checkCudaErrors(cudaMemset(temp_aux, 0, sizeof(float) * M_local * N_local));

  for (int i = 0; i < image_count_local; i++) {
    getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(
        temp_aux, d_y, d_s, oldest_idx, oldest_idx, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
    sy = deviceReduce<float>(temp_aux, M_local * N_local,
                             threadsPerBlockNN.x * threadsPerBlockNN.y);

    getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(
        temp_aux, d_y, d_y, oldest_idx, oldest_idx, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
    yy = deviceReduce<float>(temp_aux, M_local * N_local,
                             threadsPerBlockNN.x * threadsPerBlockNN.y);

    // Safety check: avoid division by very small numbers to prevent overflow
    if (fabsf(yy) > EPS) {
      float ratio = sy / yy;
      if (isfinite(ratio))
        sy_yy += ratio;
      // else skip if ratio is NaN/Inf
    }
    // else skip if yy is too small
  }

  cudaFree(temp_aux);
  return sy_yy;
}

__host__ void LBFGS::computeAlphaCoefficients(float* gradient, int par_M,
                                              int lbfgs_it, float** alpha) {
  // First loop: iterate backwards (newest to oldest)
  // Note: aux_vector and d_q are allocated in computeDirection
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  float rho = 0.0f;
  float rho_den;

  for (int i = 0; i < image_count_local; i++) {
    for (int k = par_M - 1; k >= 0; k--) {
      int hist_idx = mapToCircularBuffer(k, par_M, lbfgs_it);
      
      // Compute rho_k = 1.0 / (y_k^T s_k)
      getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s,
                                                          hist_idx, hist_idx, M_local, N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
      rho_den = deviceReduce<float>(aux_vector, M_local * N_local,
                                    threadsPerBlockNN.x * threadsPerBlockNN.y);
      // Safety check: avoid division by very small numbers to prevent overflow
      if (fabsf(rho_den) > EPS)
        rho = 1.0f / rho_den;
      else
        rho = 0.0f;
      
      // Compute alpha_k = rho_k * (s_k^T * q)
      getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_s, d_q,
                                                          hist_idx, 0, M_local, N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
      float dot_sq = deviceReduce<float>(aux_vector, M_local * N_local,
                                         threadsPerBlockNN.x * threadsPerBlockNN.y);
      alpha[i][k] = rho * dot_sq;
      
      // Safety check: ensure alpha is finite
      if (!isfinite(alpha[i][k])) {
        alpha[i][k] = 0.0f;
      }
      
      // Update q: q = q - alpha_k * y_k
      updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_q, -alpha[i][k], d_y, hist_idx, M_local,
                                                  N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ void LBFGS::computeBetaCoefficients(float* r, float** alpha, int par_M,
                                             int lbfgs_it) {
  // Second loop: iterate forwards (oldest to newest)
  // Note: aux_vector is allocated in computeDirection
  float rho = 0.0f;
  float rho_den;
  float beta = 0.0f;

  for (int i = 0; i < image->getImageCount(); i++) {
    for (int k = 0; k < par_M; k++) {
      int hist_idx = mapToCircularBuffer(k, par_M, lbfgs_it);
      
      // Compute rho_k = 1.0 / (y_k^T s_k)
      getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s,
                                                          hist_idx, hist_idx, M, N, i);
      checkCudaErrors(cudaDeviceSynchronize());
      rho_den = deviceReduce<float>(aux_vector, M * N,
                                    threadsPerBlockNN.x * threadsPerBlockNN.y);
      // Safety check: avoid division by very small numbers to prevent overflow
      if (fabsf(rho_den) > EPS)
        rho = 1.0f / rho_den;
      else
        rho = 0.0f;
      
      // Compute beta_k = rho_k * (y_k^T * r)
      getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, r,
                                                          hist_idx, 0, M, N, i);
      checkCudaErrors(cudaDeviceSynchronize());
      float dot_yr = deviceReduce<float>(aux_vector, M * N,
                                       threadsPerBlockNN.x * threadsPerBlockNN.y);
      beta = rho * dot_yr;
      
      // Safety check: ensure beta is finite
      if (!isfinite(beta)) {
        beta = 0.0f;
      }
      
      // Update r: r = r + s_k * (alpha_k - beta_k)
      updateQ<<<numBlocksNN, threadsPerBlockNN>>>(r, alpha[i][k] - beta, d_s,
                                                  hist_idx, M, N, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ void LBFGS::computeDirection(float* gradient) {
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  int par_M = std::min(this->K, this->current_iteration);
  
  if (par_M == 0) {
    // No history available - use steepest descent
    for (int i = 0; i < image_count_local; i++) {
      searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(
          gradient, M_local, N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    return;
  }

  int lbfgs_it = (this->current_iteration - 1) % this->K;

  // Allocate alpha array
  float** alpha = (float**)malloc(image_count_local * sizeof(float*));
  for (int i = 0; i < image_count_local; i++) {
    alpha[i] = (float*)malloc(par_M * sizeof(float));
    memset(alpha[i], 0, par_M * sizeof(float));
  }

  // Allocate temporary vectors
  checkCudaErrors(cudaMalloc((void**)&aux_vector, sizeof(float) * M_local * N_local));
  checkCudaErrors(cudaMalloc((void**)&d_q,
                             sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(cudaMalloc((void**)&d_r,
                             sizeof(float) * M_local * N_local * image_count_local));

  checkCudaErrors(cudaMemset(aux_vector, 0, sizeof(float) * M_local * N_local));
  checkCudaErrors(cudaMemcpy(d_q, gradient,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));

  // First loop: compute alpha coefficients
  computeAlphaCoefficients(gradient, par_M, lbfgs_it, alpha);

  // Compute gamma scaling factor
  float gamma = computeScalingFactor(par_M, lbfgs_it);

  // Scale q: r = gamma * q
  for (int i = 0; i < image->getImageCount(); i++) {
    getR<<<numBlocksNN, threadsPerBlockNN>>>(d_r, d_q, gamma, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // Second loop: compute beta coefficients and update r
  computeBetaCoefficients(d_r, alpha, par_M, lbfgs_it);

  // Set search direction to negative of r
  for (int i = 0; i < image->getImageCount(); i++) {
    searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(
        d_r, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // Copy result back to gradient/xi
  checkCudaErrors(cudaMemcpy(gradient, d_r,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));

  // Cleanup
  cudaFree(aux_vector);
  cudaFree(d_q);
  cudaFree(d_r);
  for (int i = 0; i < image->getImageCount(); i++) {
    free(alpha[i]);
  }
  free(alpha);
}

__host__ void LBFGS::updateHistory(int iteration) {
  // Compute correction pairs: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  int hist_idx = (iteration - 1) % this->K;
  
  for (int i = 0; i < image_count_local; i++) {
    calculateSandY<<<numBlocksNN, threadsPerBlockNN>>>(
        d_y, d_s, image->getImage(), xi, p_old, xi_old,
        hist_idx, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

__host__ float LBFGS::performIteration(int iteration, float prev_function_value) {
  double start = omp_get_wtime();
  this->current_iteration = iteration;
  this->max_per_it = 0.0f;

  if (verbose_flag) {
    std::cout << "\n\n********** Iteration " << iteration << " **********\n"
              << std::endl;
  }

  // Save previous state before line search
  checkCudaErrors(cudaMemcpy(p_old, image->getImage(),
                             sizeof(float) * M * N * image->getImageCount(),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(xi_old, xi,
                             sizeof(float) * M * N * image->getImageCount(),
                             cudaMemcpyDeviceToDevice));

  // Perform line search
  // LineSearcher now owns and manages its own seeder internally
  LineSearcher* searcher = static_cast<LineSearcher*>(linesearcher_ptr);
  
  // Ensure line searcher has Image object set (use optimizer's image member)
  searcher->setImage(image);
  
  // Line searcher manages its own initial step size internally
  auto result = searcher->search(image->getImage(), xi, of, nullptr);
  float new_function_value = result.first;
  float alpha_step = result.second;
  fret = new_function_value;  // Store for compatibility
  
  // Store step size for next iteration (as fallback initial_alpha)
  prev_step_size = alpha_step;

  // Check for function convergence
  
  if (verbose_flag) {
    std::cout << "Function value = " << std::setprecision(4) << std::fixed
              << new_function_value << std::endl;
  }

  // Compute new gradient
  of->calcGradient(image->getImage(), xi, iteration);

  // Update history with correction pairs
  updateHistory(iteration);

  // Compute new search direction using two-loop recursion
  computeDirection(xi);

  if (verbose_flag) {
    double end = omp_get_wtime();
    std::cout << "Time: " << std::setprecision(4) << (end - start)
              << " seconds" << std::endl;
  }

  return new_function_value;
}

__host__ void LBFGS::optimize() {
  if (verbose_flag) {
    std::cout << "\n\nStarting LBFGS method\n" << std::endl;
  }

  allocateMemoryGpu();
  
  // Ensure line searcher has Image object set (use optimizer's image member)
  if (linesearcher_ptr != nullptr && image != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setImage(image);
  }

  float prev_function_value = initializeOptimizationState();

  // Main optimization loop
  for (int iteration = 1; iteration <= this->total_iterations; iteration++) {
    float new_function_value = performIteration(iteration, prev_function_value);

    // Check for function convergence
    if (checkFunctionConvergence(new_function_value, prev_function_value)) {
      if (verbose_flag) {
        std::cout << "Exit due to tolerance" << std::endl;
      }
      // Use optimizer's image member instead of extern Image* I
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    // Check for gradient convergence
    if (checkGradientConvergence()) {
      if (verbose_flag) {
        std::cout << "Exit due to gnorm ~ 0" << std::endl;
      }
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    // Update state for next iteration
    prev_function_value = new_function_value;
  }

  if (verbose_flag) {
    std::cout << "Too many iterations in LBFGS" << std::endl;
  }

  of->calcFunction(image->getImage());
  deallocateMemoryGpu();
}

// Factory registration
namespace {
Optimizer* CreateLbfgs() {
  return new LBFGS;
}

const std::string name = "CG-LBFGS";
const bool RegisteredLbgs =
    registerCreationFunction<Optimizer, std::string>(name, CreateLbfgs);
}  // namespace
