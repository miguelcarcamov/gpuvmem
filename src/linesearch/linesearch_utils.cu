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
#include "linesearcher.cuh"  // For LineSearcher class definition
#include "error.cuh"
#include "functions.cuh"
#include "optimizers/conjugategradient.cuh"  // For computeDotProduct kernel

extern bool nopositivity;
// Image object is accessed through LineSearcher member (this->image) instead of extern global
// For evaluateLineFunction, we use current_line_searcher->getImage() with fallback to extern I
extern Image* I;  // Fallback for backward compatibility
extern ObjectiveFunction* testof;
extern LineSearcher* current_line_searcher;  // Set by f1dim, used by evaluateLineFunction

// Thread-local storage for Image pointer (set in updatePoint, accessed by particularNewP)
static thread_local Image* current_image_for_newp = nullptr;

// Accessor function for particularNewP to get current Image object
__host__ Image* getCurrentImage() {
  return current_image_for_newp;
}
// Keep extern dim3 for fallback (will be removed once ObjectiveFunction always has them set)
// Note: These are declared in mfs.cu as threadsPerBlockNN and numBlocksNN
extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

// Global variables for f1dim (used by Brent and other methods)
extern float* device_pcom;
extern float *device_xicom, (*nrfunc)(float*);

// Helper function to evaluate function along line
// Uses current_line_searcher's image member instead of extern Image* I
__host__ float evaluateLineFunction(float alpha) {
  float* device_xt;
  float f;

  // Get Image object from current_line_searcher (set by f1dim)
  extern LineSearcher* current_line_searcher;
  Image* image_to_use = (current_line_searcher != nullptr) ? current_line_searcher->getImage() : I;
  
  // Fallback to extern I if line searcher doesn't have image set (for backward compatibility)
  if (image_to_use == nullptr) {
    image_to_use = I;
  }
  
  if (image_to_use == nullptr) {
    std::cerr << "ERROR: evaluateLineFunction: No Image object available!" << std::endl;
    return 0.0f;
  }

  long M_local = image_to_use->getM();
  long N_local = image_to_use->getN();
  int image_count_local = image_to_use->getImageCount();

  checkCudaErrors(
      cudaMalloc((void**)&device_xt, sizeof(float) * M_local * N_local * image_count_local));
  checkCudaErrors(
      cudaMemset(device_xt, 0, sizeof(float) * M_local * N_local * image_count_local));

  imageMap* auxPtr = image_to_use->getFunctionMapping();
  if (!nopositivity) {
    for (int i = 0; i < image_count_local; i++) {
      (auxPtr[i].evaluateXt)(device_xt, device_pcom, device_xicom, alpha, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  } else {
    for (int i = 0; i < image_count_local; i++) {
      evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(
          device_xt, device_pcom, device_xicom, alpha, N_local, M_local, i);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  f = testof->calcFunction(device_xt);
  cudaFree(device_xt);
  return f;
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
  dim3 threadsPerBlockNN_use, numBlocksNN_use;
  if (threadsPerBlockNN_local.x == 0 || threadsPerBlockNN_local.y == 0 ||
      numBlocksNN_local.x == 0 || numBlocksNN_local.y == 0) {
    // Fallback to extern if ObjectiveFunction values not set
    extern dim3 threadsPerBlockNN, numBlocksNN;
    threadsPerBlockNN_use = threadsPerBlockNN;
    numBlocksNN_use = numBlocksNN;
  } else {
    threadsPerBlockNN_use = threadsPerBlockNN_local;
    numBlocksNN_use = numBlocksNN_local;
  }
  
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
  
  // Ensure we're on firstgpu before launching kernels
  extern int firstgpu;
  cudaSetDevice(firstgpu);
  
  // Verify device context is set correctly
  int current_device;
  cudaGetDevice(&current_device);
  if (current_device != firstgpu) {
    cudaSetDevice(firstgpu);
  }
  
  // Verify pointers are on the correct device (if they're device memory)
  cudaPointerAttributes p_attrs, d_attrs;
  cudaError_t err_p = cudaPointerGetAttributes(&p_attrs, p);
  cudaError_t err_d = cudaPointerGetAttributes(&d_attrs, d);
  
  // If pointers are valid device memory, check they're on firstgpu
  if (err_p == cudaSuccess && p_attrs.type == cudaMemoryTypeDevice) {
    if (p_attrs.device != firstgpu) {
      std::cerr << "ERROR: updatePoint: pointer p is on device " << p_attrs.device 
                << " but should be on device " << firstgpu << std::endl;
      return;
    }
  } else if (err_p != cudaSuccess) {
    std::cerr << "ERROR: updatePoint: Failed to get pointer attributes for p: " 
              << cudaGetErrorString(err_p) << std::endl;
    return;
  }
  
  if (err_d == cudaSuccess && d_attrs.type == cudaMemoryTypeDevice) {
    if (d_attrs.device != firstgpu) {
      std::cerr << "ERROR: updatePoint: pointer d is on device " << d_attrs.device 
                << " but should be on device " << firstgpu << std::endl;
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
  
  // Ensure device context is set before kernel launches
  cudaGetDevice(&current_device);
  if (current_device != firstgpu) {
    cudaSetDevice(firstgpu);
  }
  
  // Use imageMap function pointers to determine which kernel to use
  // This ensures consistency with how other line searchers (Brent, Fixed) work
  // and respects the configuration set up in mfs.cu
  // The imageMap function pointers are already correctly configured:
  // - particularNewP for image 0 when positivity is enabled
  // - defaultNewP for other images or when positivity is disabled
  // This is critical for multi-parameter optimization where initial_values might
  // become invalid, but the imageMap configuration remains correct
  
  // Store Image pointer in thread-local variable so particularNewP can access it
  // This avoids needing extern global variable
  current_image_for_newp = image;
  
  for (int img_idx = 0; img_idx < image_count; img_idx++) {
    cudaSetDevice(firstgpu);
    cudaGetLastError();  // Clear any previous errors
    
    // Use the function pointer from imageMap - this respects the configuration
    // and ensures positivity is applied correctly even during multi-parameter optimization
    if (auxPtr[img_idx].newP != nullptr) {
      (auxPtr[img_idx].newP)(p, d, alpha, img_idx);
    } else {
      // Fallback: if function pointer is null, use default (no positivity)
      // This should not happen if imageMap is properly configured
      std::cerr << "WARNING: updatePoint: imageMap[" << img_idx << "].newP is null, using defaultNewP" << std::endl;
      extern dim3 threadsPerBlockNN, numBlocksNN;
      extern long M, N;
      newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, d, alpha, N, M, img_idx);
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
  }
}

// Helper function to compute directional derivative: ∇f(x)^T*d
__host__ float computeDirectionalDerivative(float* gradient, float* search_direction) {
  float* dot_result;
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
