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
#include "optimizers/optimizer_kernels.cuh"
#include "optimizers/conjugategradient.cuh"  // computeDotProduct
#include "reduction/reduction_host.cuh"
#include "linesearch/linesearcher.cuh"  // Include here to avoid circular dependency
#include "linesearch/brent.cuh"  // For Brent class
#include "error.cuh"
#include "cli/gpuvmem_cli_config.hh"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <vector>

// M, N, image_count are now accessed through Image object (image->getM(), getN(), getImageCount())
// Removed extern declarations to encourage using Image object

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int flag_opt;
extern int firstgpu;

#define EPS 1.0e-10
#define MIN_Y_NORM 1e-10f  // Minimum value for ||y||^2 to avoid division by zero

#define FREEALL              \
  cudaFree(lbfgs_scratch_y); \
  lbfgs_scratch_y = nullptr; \
  cudaFree(lbfgs_scratch_s); \
  lbfgs_scratch_s = nullptr; \
  cudaFree(d_y);             \
  cudaFree(d_s);             \
  cudaFree(xi);              \
  cudaFree(xi_old);          \
  cudaFree(p_old);           \
  cudaFree(norm_vector);     \
  cudaFree(aux_vector);     \
  aux_vector = nullptr;      \
  cudaFree(d_q);             \
  d_q = nullptr;             \
  cudaFree(d_r);             \
  d_r = nullptr;

__host__ int LBFGS::getK() {
  return this->K;
}

__host__ void LBFGS::setK(int K) {
  this->K = K;
}

__host__ void LBFGS::allocateMemoryGpu() {
  checkCudaErrors(cudaSetDevice(firstgpu));
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

  const size_t plane_bytes =
      sizeof(float) * static_cast<size_t>(M_local) * static_cast<size_t>(N_local);
  const size_t vol_bytes = plane_bytes * static_cast<size_t>(image_count_local);
  checkCudaErrors(cudaMalloc((void**)&aux_vector, plane_bytes));
  checkCudaErrors(cudaMemset(aux_vector, 0, plane_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_q, vol_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_r, vol_bytes));
  checkCudaErrors(cudaMemset(d_q, 0, vol_bytes));
  checkCudaErrors(cudaMemset(d_r, 0, vol_bytes));

  const size_t cube_bytes =
      sizeof(float) * static_cast<size_t>(M_local) * static_cast<size_t>(N_local) *
      static_cast<size_t>(image_count_local);
  checkCudaErrors(cudaMalloc((void**)&lbfgs_scratch_y, cube_bytes));
  checkCudaErrors(cudaMalloc((void**)&lbfgs_scratch_s, cube_bytes));
  checkCudaErrors(cudaMemset(lbfgs_scratch_y, 0, cube_bytes));
  checkCudaErrors(cudaMemset(lbfgs_scratch_s, 0, cube_bytes));
}

__host__ LBFGS::LBFGS() {
  // Default to Brent line search (current implementation)
  linesearcher_ptr = new Brent();
  prev_step_size = 1.0f;
  // Note: Image object will be set in setLineSearcher() or performIteration()
  // when image member is available
}

__host__ void LBFGS::setLineSearcher(std::unique_ptr<LineSearcher> searcher) {
  std::unique_ptr<Projection> saved;
  if (linesearcher_ptr != nullptr) {
    saved = static_cast<LineSearcher*>(linesearcher_ptr)->releaseProjection();
  }
  if (linesearcher_ptr != nullptr) {
    delete static_cast<LineSearcher*>(linesearcher_ptr);
  }
  linesearcher_ptr = searcher.release();
  // Set Image object in line searcher so it can use this->image instead of extern Image* I
  if (linesearcher_ptr != nullptr && image != nullptr) {
    LineSearcher* ls = static_cast<LineSearcher*>(linesearcher_ptr);
    ls->setImage(image);
    if (saved) {
      ls->setProjection(std::move(saved));
    } else {
      ls->setProjection(std::make_unique<NoProjection>());
    }
  }
}

__host__ void LBFGS::setProjection(std::unique_ptr<Projection> projection) {
  if (linesearcher_ptr != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setProjection(std::move(projection));
  }
}

// setStepSizeSeeder removed - seeder is now owned by LineSearcher
// Use: linesearcher->setStepSizeSeeder(std::make_unique<BBMin1Seeder>());

__host__ void LBFGS::deallocateMemoryGpu() {
  checkCudaErrors(cudaSetDevice(firstgpu));
  FREEALL
  // Do NOT delete linesearcher_ptr here - it must persist across multiple
  // optimize() calls (e.g. block mode: I_nu_0 then alpha). Same as ConjugateGradient.
}

__host__ int LBFGS::mapToCircularBuffer(int k, int par_M, int lbfgs_it) {
  // Map logical index k to circular buffer index
  // Most recent is lbfgs_it, previous is (lbfgs_it-1+K)%K, etc.
  return (lbfgs_it - (par_M - 1 - k) + this->K) % this->K;
}

__host__ float LBFGS::initializeOptimizationState() {
  checkCudaErrors(cudaSetDevice(firstgpu));
  // Set Image object in line searcher so it can use this->image instead of extern Image* I
  if (linesearcher_ptr != nullptr && image != nullptr) {
    static_cast<LineSearcher*>(linesearcher_ptr)->setImage(image);
  }
  flag_opt = this->flag;

  // Get dimensions from Image object instead of extern variables
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  if (configured) {
    of->configure(N_local, M_local, image_count_local);
    extern dim3 threadsPerBlockNN, numBlocksNN;
    of->setThreadsPerBlockNN(threadsPerBlockNN);
    of->setNumBlocksNN(numBlocksNN);
    configured = 0;
  }

  // Note: prev_point is now managed by LineSearcher (if seeder is set)
  // LBFGS uses p_old for its own history (correction pairs)

  float initial_function_value = of->calcFunction(image->getImage());
  
  if (gpuvmem_cli_verbose()) {
    std::cout << "  Initial objective f(x): " << std::setprecision(4) << std::fixed
              << initial_function_value << std::endl;
  }

  // Compute initial gradient
  of->calcGradient(image->getImage(), xi, 0);

  const size_t image_bytes_init =
      sizeof(float) * static_cast<size_t>(M_local) * static_cast<size_t>(N_local) *
      static_cast<size_t>(image_count_local);
  checkCudaErrors(cudaMemcpy(xi_old, xi, image_bytes_init, cudaMemcpyDeviceToDevice));

  // Initialize search direction as negative gradient (steepest descent)
  // Reuse dimensions from earlier in function (already declared at lines 149-151)
  for (int i = 0; i < image_count_local; i++) {
    searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(
        xi, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  prev_step_size = 1.0f;  // Initial step size
  lbfgs_stored_pairs = 0;

  return initial_function_value;
}

__host__ bool LBFGS::checkFunctionConvergence(float new_value, float prev_value) {
  return objectiveSequenceWithinTolerance(new_value, prev_value);
}

__host__ bool LBFGS::checkGradientConvergence() {
  checkCudaErrors(cudaSetDevice(firstgpu));
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();

  const size_t norm_elems = static_cast<size_t>(M_local) * static_cast<size_t>(N_local) *
                            static_cast<size_t>(image_count_local);
  checkCudaErrors(cudaMemset(norm_vector, 0, norm_elems * sizeof(float)));

  const int Mi = static_cast<int>(M_local);
  const int Ni = static_cast<int>(N_local);
  for (int i = 0; i < image_count_local; i++) {
    normArray<<<numBlocksNN, threadsPerBlockNN>>>(norm_vector, xi, Mi, Ni, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  max_per_it = deviceMaxReduce(norm_vector, M_local * N_local * image_count_local,
                               threadsPerBlockNN.x * threadsPerBlockNN.y);

  if (max_per_it <= this->gtol) {
    // If the objective is still O(1) or larger, an all-zero max|grad| is almost certainly
    // wrong (wrong CUDA device, incomplete norm fill, or a broken gradient path).
    if (max_per_it < 1e-30f && fabsf(this->last_objective_value_) > 1e-6f) {
      if (gpuvmem_cli_verbose()) {
        std::cerr << "L-BFGS: ignoring near-zero max|grad|=" << max_per_it << " at iteration "
                  << this->current_iteration << " while |f|=" << fabsf(this->last_objective_value_)
                  << " (likely numerical artefact).\n";
      }
      return false;
    }
    return true;
  }
  return false;
}

__host__ float LBFGS::computeScalingFactor(int par_M, int lbfgs_it) {
  // Initial Hessian scaling H0 = gamma * I with gamma = (s^T y) / (y^T y) on the
  // most recent correction pair — matches Nocedal & Wright and Pyralysis
  // (not the oldest pair in the L-BFGS window, and not sum_i (s_i^T y_i)/(y_i^T y_i)).
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();

  const int latest_idx = mapToCircularBuffer(par_M - 1, par_M, lbfgs_it);
  float total_sy = 0.0f;
  float total_yy = 0.0f;

  if (aux_vector == nullptr) {
    return 1.0f;
  }

  for (int i = 0; i < image_count_local; i++) {
    getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(
        aux_vector, d_y, d_s, latest_idx, latest_idx, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
    const float sy = deviceReduce<float>(aux_vector, M_local * N_local,
                                         threadsPerBlockNN.x * threadsPerBlockNN.y);

    getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(
        aux_vector, d_y, d_y, latest_idx, latest_idx, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
    const float yy = deviceReduce<float>(aux_vector, M_local * N_local,
                                         threadsPerBlockNN.x * threadsPerBlockNN.y);

    if (isfinite(sy)) {
      total_sy += sy;
    }
    if (isfinite(yy)) {
      total_yy += yy;
    }
  }

  if (!(total_yy > MIN_Y_NORM) || !isfinite(total_sy) || !isfinite(total_yy)) {
    return 1.0f;
  }
  const float gamma = total_sy / total_yy;
  return isfinite(gamma) ? gamma : 1.0f;
}

__host__ void LBFGS::computeAlphaCoefficients(float* gradient, int par_M,
                                              int lbfgs_it,
                                              std::vector<std::vector<float>>& alpha_coeffs) {
  // First loop: iterate backwards (newest to oldest). Uses aux_vector, d_q from
  // allocateMemoryGpu (one M×N scratch plane + full q vector).
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
      alpha_coeffs[i][k] = rho * dot_sq;
      
      // Safety check: ensure alpha is finite
      if (!isfinite(alpha_coeffs[i][k])) {
        alpha_coeffs[i][k] = 0.0f;
      }
      
      // Update q: q = q - alpha_k * y_k
      updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_q, -alpha_coeffs[i][k], d_y, hist_idx, M_local,
                                                  N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ void LBFGS::computeBetaCoefficients(float* r,
                                             std::vector<std::vector<float>>& alpha_coeffs,
                                             int par_M, int lbfgs_it) {
  // Second loop: iterate forwards (oldest to newest). Uses aux_vector from allocateMemoryGpu.
  const long M_local = image->getM();
  const long N_local = image->getN();
  float rho = 0.0f;
  float rho_den;
  float beta = 0.0f;

  for (int i = 0; i < image->getImageCount(); i++) {
    for (int k = 0; k < par_M; k++) {
      int hist_idx = mapToCircularBuffer(k, par_M, lbfgs_it);
      
      // Compute rho_k = 1.0 / (y_k^T s_k)
      getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s,
                                                          hist_idx, hist_idx, M_local,
                                                          N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
      rho_den = deviceReduce<float>(aux_vector, M_local * N_local,
                                    threadsPerBlockNN.x * threadsPerBlockNN.y);
      // Safety check: avoid division by very small numbers to prevent overflow
      if (fabsf(rho_den) > EPS)
        rho = 1.0f / rho_den;
      else
        rho = 0.0f;
      
      // Compute beta_k = rho_k * (y_k^T * r)
      getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, r,
                                                          hist_idx, 0, M_local, N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
      float dot_yr = deviceReduce<float>(aux_vector, M_local * N_local,
                                         threadsPerBlockNN.x * threadsPerBlockNN.y);
      beta = rho * dot_yr;
      
      // Safety check: ensure beta is finite
      if (!isfinite(beta)) {
        beta = 0.0f;
      }
      
      // Update r: r = r + s_k * (alpha_k - beta_k)
      updateQ<<<numBlocksNN, threadsPerBlockNN>>>(r, alpha_coeffs[i][k] - beta, d_s,
                                                  hist_idx, M_local, N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ void LBFGS::computeDirection(float* gradient) {
  // Get dimensions from Image object
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();
  
  int par_M = std::min(this->K, lbfgs_stored_pairs);

  if (par_M == 0) {
    // No history available - use steepest descent
    for (int i = 0; i < image_count_local; i++) {
      searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(
          gradient, M_local, N_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    return;
  }

  const int lbfgs_it =
      (lbfgs_stored_pairs > 0) ? ((lbfgs_stored_pairs - 1) % this->K) : 0;

  std::vector<std::vector<float>> alpha_coeffs(
      static_cast<size_t>(image_count_local),
      std::vector<float>(static_cast<size_t>(par_M), 0.0f));

  if (aux_vector == nullptr || d_q == nullptr || d_r == nullptr) {
    return;
  }

  checkCudaErrors(cudaMemset(aux_vector, 0, sizeof(float) * M_local * N_local));
  checkCudaErrors(cudaMemcpy(d_q, gradient,
                             sizeof(float) * M_local * N_local * image_count_local,
                             cudaMemcpyDeviceToDevice));

  // First loop: compute alpha coefficients
  computeAlphaCoefficients(gradient, par_M, lbfgs_it, alpha_coeffs);

  // Compute gamma scaling factor
  float gamma = computeScalingFactor(par_M, lbfgs_it);

  // Scale q: r = gamma * q
  for (int i = 0; i < image->getImageCount(); i++) {
    getR<<<numBlocksNN, threadsPerBlockNN>>>(d_r, d_q, gamma, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // Second loop: compute beta coefficients and update r
  computeBetaCoefficients(d_r, alpha_coeffs, par_M, lbfgs_it);

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
}

__host__ void LBFGS::updateHistory() {
  // Pyralysis-style: skip pair if curvature y^T s <= 0; optional full history clear.
  long M_local = image->getM();
  long N_local = image->getN();
  int image_count_local = image->getImageCount();

  if (lbfgs_scratch_s == nullptr || lbfgs_scratch_y == nullptr) {
    return;
  }

  for (int i = 0; i < image_count_local; i++) {
    calculateSandYScratch<<<numBlocksNN, threadsPerBlockNN>>>(
        lbfgs_scratch_y, lbfgs_scratch_s, image->getImage(), xi, p_old, xi_old, M_local,
        N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  checkCudaErrors(cudaMemset(norm_vector, 0,
                             sizeof(float) * M_local * N_local * image_count_local));
  for (int i = 0; i < image_count_local; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        norm_vector, lbfgs_scratch_s, lbfgs_scratch_y, N_local, M_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  const long dot_elems = M_local * N_local * static_cast<long>(image_count_local);
  float total_sy =
      deviceReduce<float>(norm_vector, dot_elems, threadsPerBlockNN.x * threadsPerBlockNN.y);

  if (!isfinite(total_sy) || total_sy <= 0.0f) {
    if (gpuvmem_cli_verbose()) {
      std::cout << "L-BFGS: curvature test y^T s <= 0; skipping this secant pair.\n";
    }
    if (clear_history_on_curvature_failure_) {
      const size_t hist_bytes =
          sizeof(float) * static_cast<size_t>(M_local) * static_cast<size_t>(N_local) *
          static_cast<size_t>(K) * static_cast<size_t>(image_count_local);
      checkCudaErrors(cudaMemset(d_s, 0, hist_bytes));
      checkCudaErrors(cudaMemset(d_y, 0, hist_bytes));
      lbfgs_stored_pairs = 0;
      if (gpuvmem_cli_verbose()) {
        std::cout << "L-BFGS: cleared stored secant pairs (clear_on_curvature_failure).\n";
      }
    }
    return;
  }

  const int hist_idx = lbfgs_stored_pairs % this->K;
  for (int i = 0; i < image_count_local; i++) {
    calculateSandY<<<numBlocksNN, threadsPerBlockNN>>>(
        d_y, d_s, image->getImage(), xi, p_old, xi_old, hist_idx, M_local, N_local, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  lbfgs_stored_pairs++;
}

__host__ float LBFGS::performIteration(int iteration, float prev_function_value) {
  double start = omp_get_wtime();
  this->current_iteration = iteration;
  this->max_per_it = 0.0f;
  gradient_tolerance_met_ = false;

  checkCudaErrors(cudaSetDevice(firstgpu));

  const long M_local = image->getM();
  const long N_local = image->getN();
  const int image_count_local = image->getImageCount();
  const size_t image_bytes =
      sizeof(float) * static_cast<size_t>(M_local) * static_cast<size_t>(N_local) *
      static_cast<size_t>(image_count_local);

  if (gpuvmem_cli_verbose()) {
    std::cout << "\n--- L-BFGS iteration " << iteration << " ---\n";
  }

  // Save previous state before line search
  checkCudaErrors(cudaMemcpy(p_old, image->getImage(), image_bytes,
                             cudaMemcpyDeviceToDevice));
  /* xi_old keeps g_k from the end of the previous iteration (or init); do not overwrite with
   * the search direction here — y must be g_{k+1} - g_k for a valid secant pair. */

  // Perform line search
  // LineSearcher now owns and manages its own seeder internally
  LineSearcher* searcher = static_cast<LineSearcher*>(linesearcher_ptr);

  // Ensure line searcher has Image object set (use optimizer's image member)
  searcher->setImage(image);

  // Line searcher manages its own initial step size internally
  auto result = searcher->search(image->getImage(), xi, of, nullptr);
  float new_function_value = result.first;
  float alpha_step = result.second;
  last_objective_value_ = new_function_value;

  // Store step size for next iteration (as fallback initial_alpha)
  prev_step_size = alpha_step;

  // Check for function convergence

  if (gpuvmem_cli_verbose()) {
    std::cout << "  Objective f(x_k): " << std::setprecision(4) << std::fixed << new_function_value
              << std::endl;
  }

  // Compute new gradient (must check ||g|| before computeDirection overwrites `xi`)
  of->calcGradient(image->getImage(), xi, iteration);
  gradient_tolerance_met_ = checkGradientConvergence();

  // Update history with correction pairs (curvature-gated)
  updateHistory();

  checkCudaErrors(cudaMemcpy(xi_old, xi, image_bytes, cudaMemcpyDeviceToDevice));

  // Compute new search direction using two-loop recursion (overwrites `xi`)
  computeDirection(xi);

  if (gpuvmem_cli_verbose()) {
    double end = omp_get_wtime();
    std::cout << "  Wall time this iteration: " << std::setprecision(4) << (end - start) << " s\n";
  }

  return new_function_value;
}

__host__ void LBFGS::optimize() {
  if (gpuvmem_cli_verbose()) {
    std::cout << "\n--- L-BFGS limited-memory quasi-Newton ---\n";
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

    // Check for function convergence (includes stagnation: |df| == 0 in float)
    if (checkFunctionConvergence(new_function_value, prev_function_value)) {
      if (gpuvmem_cli_verbose()) {
        const float df = fabsf(new_function_value - prev_function_value);
        if (!(df > 0.0f)) {
          std::cout << "L-BFGS stopped at iteration " << iteration
                    << ": objective unchanged at float precision (line search / plateau).\n";
        } else {
          std::cout << "L-BFGS converged: relative change in objective below ftol.\n";
        }
      }
      // Use optimizer's image member instead of extern Image* I
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    // Check for gradient convergence (uses true gradient before computeDirection)
    if (gradient_tolerance_met_) {
      if (gpuvmem_cli_verbose()) {
        std::cout << "L-BFGS converged: max|gradient| " << std::setprecision(6) << std::scientific
                  << max_per_it << " <= gtol " << this->gtol << std::fixed << ".\n"
                  << std::defaultfloat;
      }
      of->calcFunction(image->getImage());
      deallocateMemoryGpu();
      return;
    }

    // Update state for next iteration
    prev_function_value = new_function_value;
  }

  if (gpuvmem_cli_verbose()) {
    std::cout << "L-BFGS: reached maximum iteration budget without meeting tolerances.\n";
  }

  of->calcFunction(image->getImage());
  deallocateMemoryGpu();
}

// Factory registration
namespace {
Optimizer* CreateLbfgs() {
  return new LBFGS;
}

// L-BFGS (limited-memory BFGS quasi-Newton); factory id "LBFGS".
const bool RegisteredLbgs =
    registerCreationFunction<Optimizer, std::string>("LBFGS", CreateLbfgs);
}  // namespace
