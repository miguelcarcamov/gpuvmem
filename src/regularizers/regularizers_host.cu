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

#include "regularizers/regularizers_host.cuh"
#include "regularizers/regularizers_kernels.cuh"
#include "reduction/reduction_host.cuh"
#include "error.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Extern variables
extern dim3 threadsPerBlockNN, numBlocksNN;
extern long N, M;
extern int firstgpu, flag_opt, iter;
extern float* device_noise_image;
extern float noise_cut;

__host__ float L1Norm(float* I,
                      float* ds,
                      float penalization_factor,
                      float epsilon,
                      int mod,
                      int order,
                      int index,
                      int iter) {
  cudaSetDevice(firstgpu);

  float resultL1norm = 0.0f;
  if (iter > 0 && penalization_factor) {
    L1Vector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N,
                                                 M, epsilon, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultL1norm = deviceReduce<float>(
        ds, M * N, threadsPerBlockNN.x * threadsPerBlockNN.y);
  }

  return resultL1norm;
}

__host__ void DL1Norm(float* I,
                      float* dgi,
                      float penalization_factor,
                      float epsilon,
                      int mod,
                      int order,
                      int index,
                      int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DL1NormK<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, device_noise_image, epsilon, noise_cut, penalization_factor,
          N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float GL1NormK(float* I,
                        float* prior,
                        float* ds,
                        float penalization_factor,
                        float epsilon_a,
                        float epsilon_b,
                        int mod,
                        int order,
                        int index,
                        int iter) {
  cudaSetDevice(firstgpu);

  float resultL1norm = 0.0f;
  if (iter > 0 && penalization_factor) {
    GL1Vector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I,
                                                  prior, N, M, epsilon_a,
                                                  epsilon_b, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultL1norm = deviceReduce<float>(
        ds, M * N, threadsPerBlockNN.x * threadsPerBlockNN.y);
  }

  return resultL1norm;
}

__host__ void DGL1Norm(float* I,
                       float* prior,
                       float* dgi,
                       float penalization_factor,
                       float epsilon_a,
                       float epsilon_b,
                       int mod,
                       int order,
                       int index,
                       int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DGL1NormK<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, prior, device_noise_image, epsilon_a, epsilon_b, noise_cut,
          penalization_factor, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float SEntropy(float* I,
                        float* ds,
                        float prior_value,
                        float eta,
                        float penalization_factor,
                        int mod,
                        int order,
                        int index,
                        int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    SVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, N, M, noise_cut, prior_value, eta, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DEntropy(float* I,
                       float* dgi,
                       float prior_value,
                       float eta,
                       float penalization_factor,
                       int mod,
                       int order,
                       int index,
                       int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DS<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                             noise_cut, penalization_factor,
                                             prior_value, eta, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float SGEntropy(float* I,
                         float* ds,
                         float* prior,
                         float eta,
                         float penalization_factor,
                         int mod,
                         int order,
                         int index,
                         int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    SGVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, N, M, noise_cut, prior, eta, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DGEntropy(float* I,
                        float* dgi,
                        float* prior,
                        float eta,
                        float penalization_factor,
                        int mod,
                        int order,
                        int index,
                        int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DSG<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                              noise_cut, penalization_factor,
                                              prior, eta, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float laplacian(float* I,
                         float* ds,
                         float penalization_factor,
                         int mod,
                         int order,
                         int imageIndex,
                         int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    LVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M,
                                                noise_cut, imageIndex);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DLaplacian(float* I,
                         float* dgi,
                         float penalization_factor,
                         float mod,
                         float order,
                         float index,
                         int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DL<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                             noise_cut, penalization_factor, N,
                                             M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float quadraticP(float* I,
                          float* ds,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    QPVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N,
                                                 M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DQuadraticP(float* I,
                          float* dgi,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DQ<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                             noise_cut, penalization_factor, N,
                                             M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float l2ConstantPrior(float* I,
                               float* ds,
                               float prior_value,
                               float penalization_factor,
                               int mod,
                               int order,
                               int index,
                               int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    L2ConstantPriorVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, prior_value, N, M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DL2ConstantPrior(float* I,
                               float* dgi,
                               float prior_value,
                               float penalization_factor,
                               int mod,
                               int order,
                               int index,
                               int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DL2ConstantPriorKernel<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, device_noise_image, prior_value, noise_cut,
          penalization_factor, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ float isotropicTV(float* I,
                           float* ds,
                           float epsilon,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    TVVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, epsilon, N, M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DIsotropicTV(float* I,
                           float* dgi,
                           float epsilon,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DTV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                              epsilon, noise_cut,
                                              penalization_factor, N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

// Legacy function name for backward compatibility
__host__ float totalvariation(float* I,
                              float* ds,
                              float epsilon,
                              float penalization_factor,
                              int mod,
                              int order,
                              int index,
                              int iter) {
  return isotropicTV(I, ds, epsilon, penalization_factor, mod, order, index,
                     iter);
}

// Legacy function name for backward compatibility
__host__ void DTVariation(float* I,
                          float* dgi,
                          float epsilon,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter) {
  DIsotropicTV(I, dgi, epsilon, penalization_factor, mod, order, index, iter);
}

__host__ float TotalSquaredVariation(float* I,
                                     float* ds,
                                     float penalization_factor,
                                     int mod,
                                     int order,
                                     int index,
                                     int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    TSVVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N,
                                                  M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ float anisotropicTV(float* I,
                             float* ds,
                             float epsilon,
                             float penalization_factor,
                             int mod,
                             int order,
                             int index,
                             int iter) {
  cudaSetDevice(firstgpu);

  float resultS = 0.0f;
  if (iter > 0 && penalization_factor) {
    ATVVector<<<numBlocksNN, threadsPerBlockNN>>>(
        ds, device_noise_image, I, epsilon, N, M, noise_cut, index);
    checkCudaErrors(cudaDeviceSynchronize());
    resultS = deviceReduce<float>(ds, M * N,
                                  threadsPerBlockNN.x * threadsPerBlockNN.y);
  }
  return resultS;
}

__host__ void DAnisotropicTV(float* I,
                             float* dgi,
                             float epsilon,
                             float penalization_factor,
                             int mod,
                             int order,
                             int index,
                             int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DATV<<<numBlocksNN, threadsPerBlockNN>>>(
          dgi, I, device_noise_image, epsilon, noise_cut, penalization_factor,
          N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}

__host__ void DTSVariation(float* I,
                           float* dgi,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter) {
  cudaSetDevice(firstgpu);

  if (iter > 0 && penalization_factor) {
    if (flag_opt == -1 || flag_opt % 2 == index) {
      DTSV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image,
                                               noise_cut, penalization_factor,
                                               N, M, index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
}
