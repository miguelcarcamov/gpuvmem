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

#include "linesearch/linesearch_utils.cuh"
#include "linesearch/linesearch_kernels.cuh"
#include "linesearch/linesearcher.cuh"  // For LineSearcher class definition
#include "projection/projection.hh"
#include "cli/gpuvmem_cli_config.hh"
#include "error.cuh"
#include "optimizers/conjugategradient.cuh"  // For computeDotProduct kernel
#include "reduction/reduction_host.cuh"
#include <iostream>

/** ObjectiveFunction passed to updatePoint; used by imageMap kernel wrappers. */
static thread_local const ObjectiveFunction* tls_linesearch_kernel_objective = nullptr;

struct LinesearchObjectiveTlsGuard {
  const ObjectiveFunction* prev_;
  explicit LinesearchObjectiveTlsGuard(const ObjectiveFunction* of) : prev_(tls_linesearch_kernel_objective) {
    tls_linesearch_kernel_objective = of;
  }
  ~LinesearchObjectiveTlsGuard() { tls_linesearch_kernel_objective = prev_; }
};

static const ObjectiveFunction* resolve_objective_for_linesearch_kernels() {
  if (tls_linesearch_kernel_objective != nullptr) return tls_linesearch_kernel_objective;
  LineSearcher* cur = lineSearchGetCurrent();
  if (cur != nullptr) return cur->getObjectiveFunction();
  return nullptr;
}

static const Projection* active_projection() {
  LineSearcher* cur = lineSearchGetCurrent();
  if (cur == nullptr) return nullptr;
  return cur->getProjection();
}

__host__ void applyProjectionToImagePlane(const Projection* proj, float* buffer, long N, long M,
                                          int image, unsigned blocks_x, unsigned blocks_y,
                                          unsigned threads_x, unsigned threads_y,
                                          ProjectionApplyContext ctx) {
  if (proj == nullptr || buffer == nullptr) return;
  proj->applyToImagePlane(buffer, N, M, image, blocks_x, blocks_y, threads_x, threads_y, ctx);
}

// Free helper: delegates to active LineSearcher (same path as member evaluateLineFunction).
__host__ float evaluateLineFunction(float alpha) {
  LineSearcher* cur = lineSearchGetCurrent();
  if (cur != nullptr) {
    return cur->evaluateLineFunction(alpha);
  }
  std::cerr << "ERROR: evaluateLineFunction: no active LineSearcher (ScopedSearchContext)." << std::endl;
  return 0.0f;
}

// Helper function to update point: p = p + alpha * d
__host__ void updatePoint(ObjectiveFunction* objective_function, Image* image,
                         float* p, float* d, float alpha) {
  if (objective_function == nullptr || image == nullptr) {
    return;
  }
  
  // Get dimensions from objective_function
  long M_local = objective_function->getM();
  long N_local = objective_function->getN();
  int image_count = image->getImageCount();
  dim3 threadsPerBlockNN_local = objective_function->getThreadsPerBlockNN();
  dim3 numBlocksNN_local = objective_function->getNumBlocksNN();
  
  // Validate dimensions
  if (M_local <= 0 || N_local <= 0 || image_count <= 0) {
    return;
  }
  
  // Use ObjectiveFunction values directly (avoid extern variables)
  // Note: Arrays (temp_point, search_direction) MUST be allocated with same dimensions!
  dim3 threadsPerBlockNN_use = threadsPerBlockNN_local;
  dim3 numBlocksNN_use = numBlocksNN_local;
  
  // Use ObjectiveFunction dimensions
  long M_use = M_local;
  long N_use = N_local;
  
  // Validate kernel launch configuration
  if (threadsPerBlockNN_use.x == 0 || threadsPerBlockNN_use.y == 0 ||
      numBlocksNN_use.x == 0 || numBlocksNN_use.y == 0) {
    std::cerr << "ERROR: updatePoint: Invalid kernel launch configuration!" << std::endl;
    return;
  }
  
  // Validate pointers
  if (p == nullptr || d == nullptr) {
    std::cerr << "ERROR: updatePoint: Null pointer!" << std::endl;
    return;
  }
  
  const int primary_dev = objective_function->getPrimaryCudaDevice();
  cudaSetDevice(primary_dev);

  // Verify device context is set correctly
  int current_device;
  cudaGetDevice(&current_device);
  if (current_device != primary_dev) {
    cudaSetDevice(primary_dev);
  }
  
  // Verify pointers are on the correct device (if they're device memory)
  cudaPointerAttributes p_attrs, d_attrs;
  cudaError_t err_p = cudaPointerGetAttributes(&p_attrs, p);
  cudaError_t err_d = cudaPointerGetAttributes(&d_attrs, d);
  
  // If pointers are valid device memory, check they're on firstgpu
  if (err_p == cudaSuccess && p_attrs.type == cudaMemoryTypeDevice) {
    if (p_attrs.device != primary_dev) {
      std::cerr << "ERROR: updatePoint: pointer p is on device " << p_attrs.device
                << " but should be on device " << primary_dev << std::endl;
      return;
    }
  } else if (err_p != cudaSuccess) {
    std::cerr << "ERROR: updatePoint: Failed to get pointer attributes for p: " 
              << cudaGetErrorString(err_p) << std::endl;
    return;
  }
  
  if (err_d == cudaSuccess && d_attrs.type == cudaMemoryTypeDevice) {
    if (d_attrs.device != primary_dev) {
      std::cerr << "ERROR: updatePoint: pointer d is on device " << d_attrs.device
                << " but should be on device " << primary_dev << std::endl;
      return;
    }
  } else if (err_d != cudaSuccess) {
    std::cerr << "ERROR: updatePoint: Failed to get pointer attributes for d: " 
              << cudaGetErrorString(err_d) << std::endl;
    return;
  }
  
  // Clear any previous CUDA errors from pointer attribute checks
  cudaGetLastError();
  
  imageMap* auxPtr = image->getFunctionMapping();
  if (auxPtr == nullptr) {
    std::cerr << "ERROR: updatePoint: image->getFunctionMapping() returned nullptr!" << std::endl;
    return;
  }

  LinesearchObjectiveTlsGuard _obj_kern_guard(objective_function);

  const Projection* line_proj = active_projection();

  // Ensure device context is set before kernel launches
  cudaGetDevice(&current_device);
  if (current_device != primary_dev) {
    cudaSetDevice(primary_dev);
  }

  for (int img_idx = 0; img_idx < image_count; img_idx++) {
    cudaSetDevice(primary_dev);
    cudaGetLastError();  // Clear any previous errors
    
    // Use the function pointer from imageMap - this respects the configuration
    // and ensures positivity is applied correctly even during multi-parameter optimization
    if (auxPtr[img_idx].newP != nullptr) {
      (auxPtr[img_idx].newP)(p, d, alpha, img_idx);
    } else {
      // Fallback: if function pointer is null, use default (no positivity)
      // This should not happen if imageMap is properly configured
      std::cerr << "WARNING: updatePoint: imageMap[" << img_idx << "].newP is null, using defaultNewP" << std::endl;
      newPNoPositivity<<<numBlocksNN_use, threadsPerBlockNN_use>>>(p, d, alpha, N_use, M_use, img_idx);
    }
    
    // Check for launch errors immediately
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      std::cerr << "ERROR: updatePoint: Kernel launch failed for image " << img_idx << ": " 
                << cudaGetErrorString(launch_err) << std::endl;
      return;
    }
    
    // Synchronize and check for execution errors
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
      std::cerr << "ERROR: updatePoint: Kernel execution failed for image " << img_idx << ": " 
                << cudaGetErrorString(sync_err) << std::endl;
      std::cerr << "  Parameters: M=" << M_use << ", N=" << N_use << ", image=" << img_idx << std::endl;
      std::cerr << "  Launch config: blocks(" << numBlocksNN_use.x << ", " << numBlocksNN_use.y 
                << "), threads(" << threadsPerBlockNN_use.x << ", " << threadsPerBlockNN_use.y << ")" << std::endl;
      return;
    }

    applyProjectionToImagePlane(line_proj, p, N_use, M_use, img_idx,
                                static_cast<unsigned>(numBlocksNN_use.x),
                                static_cast<unsigned>(numBlocksNN_use.y),
                                static_cast<unsigned>(threadsPerBlockNN_use.x),
                                static_cast<unsigned>(threadsPerBlockNN_use.y),
                                ProjectionApplyContext::kParameterStep);
  }
}

// Helper function to compute directional derivative: ∇f(x)^T*d
__host__ float computeDirectionalDerivative(float* gradient, float* search_direction) {
  const ObjectiveFunction* of = resolve_objective_for_linesearch_kernels();
  if (of == nullptr) {
    std::cerr << "ERROR: computeDirectionalDerivative: no ObjectiveFunction context." << std::endl;
    return 0.0f;
  }
  const long M = of->getM();
  const long N = of->getN();
  const int image_count = of->getImageCount();
  const dim3 threadsPerBlockNN = of->getThreadsPerBlockNN();
  const dim3 numBlocksNN = of->getNumBlocksNN();
  float* dot_result;
  checkCudaErrors(cudaSetDevice(of->getPrimaryCudaDevice()));
  // computeDotProduct kernel accesses result[M * N * image + N * i + j]
  // So we need to allocate for all images, not just one
  checkCudaErrors(cudaMalloc((void**)&dot_result, sizeof(float) * M * N * image_count));
  checkCudaErrors(cudaMemset(dot_result, 0, sizeof(float) * M * N * image_count));
  
  for (int i = 0; i < image_count; i++) {
    computeDotProduct<<<numBlocksNN, threadsPerBlockNN>>>(
        dot_result, gradient, search_direction, N, M, i);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  // Reduce across all images: sum dot products from all images
  float dir_deriv = deviceReduce<float>(
      dot_result, M * N * image_count, threadsPerBlockNN.x * threadsPerBlockNN.y);
  
  cudaFree(dot_result);
  return dir_deriv;
}

// Host wrappers for line search kernels (moved from functions.cu)

__host__ void defaultNewP(float* p, float* xi, float xmin, int image) {
  const ObjectiveFunction* of = resolve_objective_for_linesearch_kernels();
  if (of == nullptr) {
    std::cerr << "ERROR: defaultNewP: no ObjectiveFunction context." << std::endl;
    return;
  }
  cudaSetDevice(of->getPrimaryCudaDevice());
  newPNoPositivity<<<of->getNumBlocksNN(), of->getThreadsPerBlockNN()>>>(
      p, xi, xmin, of->getN(), of->getM(), image);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void defaultEvaluateXt(float* xt,
                                float* pcom,
                                float* xicom,
                                float x,
                                int image) {
  const ObjectiveFunction* of = resolve_objective_for_linesearch_kernels();
  if (of == nullptr) {
    std::cerr << "ERROR: defaultEvaluateXt: no ObjectiveFunction context." << std::endl;
    return;
  }
  cudaSetDevice(of->getPrimaryCudaDevice());
  evaluateXtNoPositivity<<<of->getNumBlocksNN(), of->getThreadsPerBlockNN()>>>(
      xt, pcom, xicom, x, of->getN(), of->getM(), image);
  checkCudaErrors(cudaDeviceSynchronize());
  const dim3 nb = of->getNumBlocksNN();
  const dim3 th = of->getThreadsPerBlockNN();
  applyProjectionToImagePlane(active_projection(), xt, of->getN(), of->getM(), image,
                              static_cast<unsigned>(nb.x), static_cast<unsigned>(nb.y),
                              static_cast<unsigned>(th.x), static_cast<unsigned>(th.y),
                              ProjectionApplyContext::kLineSearch1dSample);
}
