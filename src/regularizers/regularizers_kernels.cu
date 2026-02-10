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

#include "regularizers/regularizers_kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

__host__ __device__ float approxAbs(float val, float epsilon) {
  return sqrtf(val * val + epsilon);
}

__device__ float calculateL1norm(const float* __restrict__ I,
                                 float epsilon,
                                 float noise,
                                 float noise_cut,
                                 int index,
                                 int M,
                                 int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float l1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    l1 = approxAbs(c, epsilon);
  }

  return l1;
}

__global__ void L1Vector(float* __restrict__ L1,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float epsilon,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  L1[N * i + j] =
      calculateL1norm(I, epsilon, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDNormL1(const float* __restrict__ I,
                                  float lambda,
                                  float noise,
                                  float epsilon,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dL1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    dL1 = c / approxAbs(c, epsilon);
  }

  dL1 *= lambda;
  return dL1;
}

__global__ void DL1NormK(float* __restrict__ dL1,
                         const float* __restrict__ I,
                         const float* __restrict__ noise,
                         float epsilon,
                         float noise_cut,
                         float lambda,
                         long N,
                         long M,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dL1[N * i + j] =
      calculateDNormL1(I, lambda, noise_val, epsilon, noise_cut, index, M, N);
}

__device__ float calculateGL1norm(const float* __restrict__ I,
                                  float prior,
                                  float epsilon_a,
                                  float epsilon_b,
                                  float noise,
                                  float noise_cut,
                                  int index,
                                  int M,
                                  int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float l1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    l1 = approxAbs(c, epsilon_a) / (approxAbs(prior, epsilon_a) + epsilon_b);
  }

  return l1;
}

__global__ void GL1Vector(float* __restrict__ L1,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          const float* __restrict__ prior,
                          long N,
                          long M,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  L1[N * i + j] = calculateGL1norm(I, prior_val, epsilon_a, epsilon_b,
                                   noise_val, noise_cut, index, M, N);
}

__device__ float calculateDGNormL1(const float* __restrict__ I,
                                   float prior,
                                   float lambda,
                                   float noise,
                                   float epsilon_a,
                                   float epsilon_b,
                                   float noise_cut,
                                   int index,
                                   int M,
                                   int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dL1 = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    const float prior_abs = approxAbs(prior, epsilon_a);
    dL1 = c / (approxAbs(c, epsilon_a) * (prior_abs + epsilon_b));
  }

  dL1 *= lambda;
  return dL1;
}

__global__ void DGL1NormK(float* __restrict__ dL1,
                          const float* __restrict__ I,
                          const float* __restrict__ prior,
                          const float* __restrict__ noise,
                          float epsilon_a,
                          float epsilon_b,
                          float noise_cut,
                          float lambda,
                          long N,
                          long M,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  dL1[N * i + j] = calculateDGNormL1(I, prior_val, lambda, noise_val, epsilon_a,
                                     epsilon_b, noise_cut, index, M, N);
}

__device__ float calculateS(const float* __restrict__ I,
                            float G,
                            float eta,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float S = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    S = c * logf((c / G) + (eta + 1.0f));
  }

  return S;
}

__device__ float calculateDS(const float* __restrict__ I,
                             float G,
                             float eta,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dS = 0.0f;
  if (noise < noise_cut) {
    const float c = I[N * M * index + N * i + j];
    const float c_over_G_plus_eta = (c / G) + (eta + 1.0f);
    dS = logf(c_over_G_plus_eta) + 1.0f / (1.0f + (((eta + 1.0f) * G) / c));
  }

  dS *= lambda;
  return dS;
}

__global__ void SVector(float* __restrict__ S,
                        const float* __restrict__ noise,
                        float* __restrict__ I,
                        long N,
                        long M,
                        float noise_cut,
                        float prior_value,
                        float eta,
                        int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  S[N * i + j] =
      calculateS(I, prior_value, eta, noise_val, noise_cut, index, M, N);
}

__global__ void DS(float* __restrict__ dS,
                   float* __restrict__ I,
                   const float* __restrict__ noise,
                   float noise_cut,
                   float lambda,
                   float prior_value,
                   float eta,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dS[N * i + j] = calculateDS(I, prior_value, eta, lambda, noise_val, noise_cut,
                              index, M, N);
}

__global__ void SGVector(float* __restrict__ S,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float noise_cut,
                         const float* __restrict__ prior,
                         float eta,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  S[N * i + j] =
      calculateS(I, prior_val, eta, noise_val, noise_cut, index, M, N);
}

__global__ void DSG(float* __restrict__ dS,
                    const float* __restrict__ I,
                    const float* __restrict__ noise,
                    float noise_cut,
                    float lambda,
                    const float* __restrict__ prior,
                    float eta,
                    long N,
                    long M,
                    int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  const float prior_val = prior[N * i + j];
  dS[N * i + j] =
      calculateDS(I, prior_val, eta, lambda, noise_val, noise_cut, index, M, N);
}

__device__ float calculateQP(const float* __restrict__ I,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float qp = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float l = I[N * M * index + N * i + (j - 1)];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];

      qp = (c - l) * (c - l) + (c - r) * (c - r) + (c - u) * (c - u) +
           (c - d) * (c - d);
      qp *= 0.5f;  // Use multiply instead of divide
    } else {
      qp = I[N * M * index + N * i + j];
    }
  }

  return qp;
}

__global__ void QPVector(float* __restrict__ Q,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         long N,
                         long M,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  Q[N * i + j] = calculateQP(I, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDQ(const float* __restrict__ I,
                             float lambda,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dQ = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];

      dQ = 2.0f * (4.0f * c - d + u + r + l);
    } else {
      dQ = I[N * M * index + N * i + j];
    }
  }

  dQ *= lambda;
  return dQ;
}

__global__ void DQ(float* __restrict__ dQ,
                   const float* __restrict__ I,
                   const float* __restrict__ noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dQ[N * i + j] = calculateDQ(I, lambda, noise_val, noise_cut, index, M, N);
}

// L2 constant prior: R = sum_p (I_p - prior)^2. Used e.g. for spectral index
// (image index 1) to penalize deviation from a constant alpha_prior.
__device__ float calculateL2ConstantPrior(const float* __restrict__ I,
                                          float prior,
                                          float noise,
                                          float noise_cut,
                                          int index,
                                          int M,
                                          int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float val = 0.0f;
  if (noise < noise_cut) {
    float c = I[N * M * index + N * i + j];
    float d = c - prior;
    val = d * d;
  }
  return val;
}

__global__ void L2ConstantPriorVector(float* __restrict__ R,
                                      const float* __restrict__ noise,
                                      const float* __restrict__ I,
                                      float prior,
                                      long N,
                                      long M,
                                      float noise_cut,
                                      int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  R[N * i + j] =
      calculateL2ConstantPrior(I, prior, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDL2ConstantPrior(const float* __restrict__ I,
                                           float prior,
                                           float lambda,
                                           float noise,
                                           float noise_cut,
                                           int index,
                                           int M,
                                           int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dR = 0.0f;
  if (noise < noise_cut) {
    float c = I[N * M * index + N * i + j];
    dR = 2.0f * (c - prior);
  }
  dR *= lambda;
  return dR;
}

__global__ void DL2ConstantPriorKernel(float* __restrict__ dR,
                                       const float* __restrict__ I,
                                       const float* __restrict__ noise,
                                       float prior,
                                       float noise_cut,
                                       float lambda,
                                       long N,
                                       long M,
                                       int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dR[N * i + j] = calculateDL2ConstantPrior(I, prior, lambda, noise_val,
                                             noise_cut, index, M, N);
}

__device__ float calculateDL(float* I,
                            float lambda,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float c, d, u, r, l, dl_corner, dr_corner, lu_corner, ru_corner, d2, u2, l2,
      r2;

  float dL = 0.0f;

  c = I[N * M * index + N * i + j];

  if (noise < noise_cut) {
    if ((i > 1 && i < M - 2) && (j > 1 && j < N - 2)) {
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];
      r = I[N * M * index + N * i + (j + 1)];
      l = I[N * M * index + N * i + (j - 1)];
      dl_corner = I[N * M * index + N * (i + 1) + (j - 1)];
      dr_corner = I[N * M * index + N * (i + 1) + (j + 1)];
      lu_corner = I[N * M * index + N * (i - 1) + (j - 1)];
      ru_corner = I[N * M * index + N * (i - 1) + (j + 1)];
      d2 = I[N * M * index + N * (i + 2) + j];
      u2 = I[N * M * index + N * (i - 2) + j];
      l2 = I[N * M * index + N * i + (j - 2)];
      r2 = I[N * M * index + N * i + (j + 2)];

      dL = 20.0f * c - 8.0f * (d - r - u - l) +
           2.0f * (dl_corner + dr_corner + lu_corner + ru_corner) + d2 + r2 +
           u2 + l2;
    } else
      dL = 0.0f;
  }

  dL *= lambda;

  return dL;
}

__global__ void DL(float* dL,
                   float* I,
                   float* noise,
                   float noise_cut,
                   float lambda,
                   long N,
                   long M,
                   int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dL[N * i + j] =
      calculateDL(I, lambda, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateL(float* I,
                            float noise,
                            float noise_cut,
                            int index,
                            int M,
                            int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float Dx, Dy;
  float L = 0.0f;
  float c, l, r, d, u;

  c = I[N * M * index + N * i + j];
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      l = I[N * M * index + N * i + (j - 1)];
      r = I[N * M * index + N * i + (j + 1)];
      d = I[N * M * index + N * (i + 1) + j];
      u = I[N * M * index + N * (i - 1) + j];

      Dx = l - 2.0f * c + r;
      Dy = u - 2.0f * c + d;
      L = 0.5f * (Dx + Dy) * (Dx + Dy);
    } else {
      L = c;
    }
  }

  return L;
}

__global__ void LVector(float* L,
                        float* noise,
                        float* I,
                        long N,
                        long M,
                        float noise_cut,
                        int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  L[N * i + j] = calculateL(I, noise[N * i + j], noise_cut, index, M, N);
}

__device__ float calculateTV(const float* __restrict__ I,
                             float epsilon,
                             float noise,
                             float noise_cut,
                             int index,
                             int M,
                             int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0f;
  if (noise < noise_cut) {
    if (i < M - 1 && j < N - 1) {
      const float c = I[N * M * index + N * i + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];

      const float dxy0 = (r - c) * (r - c);
      const float dxy1 = (d - c) * (d - c);
      tv = sqrtf(dxy0 + dxy1 + epsilon);
    } else {
      tv = I[N * M * index + N * i + j];
    }
  }

  return tv;
}

__global__ void TVVector(float* __restrict__ TV,
                         const float* __restrict__ noise,
                         const float* __restrict__ I,
                         float epsilon,
                         long N,
                         long M,
                         float noise_cut,
                         int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  TV[N * i + j] = calculateTV(I, epsilon, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDTV(const float* __restrict__ I,
                              float epsilon,
                              float lambda,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dtv = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];
      const float dl_corner = I[N * M * index + N * (i + 1) + (j - 1)];
      const float ru_corner = I[N * M * index + N * (i - 1) + (j + 1)];

      const float num0 = 2.0f * c - r - d;
      const float num1 = c - l;
      const float num2 = c - u;

      const float den_arg0 = (c - r) * (c - r) + (c - d) * (c - d) + epsilon;
      const float den_arg1 =
          (l - c) * (l - c) + (l - dl_corner) * (l - dl_corner) + epsilon;
      const float den_arg2 =
          (u - ru_corner) * (u - ru_corner) + (u - c) * (u - c) + epsilon;

      const float den0 = sqrtf(den_arg0);
      const float den1 = sqrtf(den_arg1);
      const float den2 = sqrtf(den_arg2);

      dtv = num0 / den0 + num1 / den1 + num2 / den2;
    } else {
      dtv = I[N * M * index + N * i + j];
    }
  }

  dtv *= lambda;
  return dtv;
}

__global__ void DTV(float* __restrict__ dTV,
                    const float* __restrict__ I,
                    const float* __restrict__ noise,
                    float epsilon,
                    float noise_cut,
                    float lambda,
                    long N,
                    long M,
                    int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dTV[N * i + j] =
      calculateDTV(I, epsilon, lambda, noise_val, noise_cut, index, M, N);
}

// Anisotropic Total Variation kernels
__device__ float calculateATV(const float* __restrict__ I,
                              float epsilon,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atv = 0.0f;
  if (noise < noise_cut) {
    if (i < M - 1 && j < N - 1) {
      const float c = I[N * M * index + N * i + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];

      // Anisotropic TV: |dx| + |dy| + epsilon (for numerical stability)
      const float dx = fabsf(r - c);
      const float dy = fabsf(d - c);
      atv = dx + dy + epsilon;
    } else {
      atv = I[N * M * index + N * i + j];
    }
  }

  return atv;
}

__global__ void ATVVector(float* __restrict__ ATV,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          float epsilon,
                          long N,
                          long M,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  ATV[N * i + j] = calculateATV(I, epsilon, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDATV(const float* __restrict__ I,
                               float epsilon,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float datv = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];

      // Anisotropic TV derivative: sign(dx) + sign(dy)
      // For numerical stability, use smoothed sign: x / (|x| + epsilon)
      const float dx_right = c - r;
      const float dx_left = l - c;
      const float dy_down = c - d;
      const float dy_up = u - c;

      const float sign_dx_right = dx_right / (fabsf(dx_right) + epsilon);
      const float sign_dx_left = dx_left / (fabsf(dx_left) + epsilon);
      const float sign_dy_down = dy_down / (fabsf(dy_down) + epsilon);
      const float sign_dy_up = dy_up / (fabsf(dy_up) + epsilon);

      // Sum of contributions from all neighbors
      datv = sign_dx_right + sign_dx_left + sign_dy_down + sign_dy_up;
    } else {
      datv = I[N * M * index + N * i + j];
    }
  }

  datv *= lambda;
  return datv;
}

__global__ void DATV(float* __restrict__ dATV,
                     const float* __restrict__ I,
                     const float* __restrict__ noise,
                     float epsilon,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dATV[N * i + j] =
      calculateDATV(I, epsilon, lambda, noise_val, noise_cut, index, M, N);
}

__device__ float calculateTSV(const float* __restrict__ I,
                              float noise,
                              float noise_cut,
                              int index,
                              int M,
                              int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0f;
  if (noise < noise_cut) {
    if (i < M - 1 && j < N - 1) {
      const float c = I[N * M * index + N * i + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float d = I[N * M * index + N * (i + 1) + j];

      const float dx = c - r;
      const float dy = c - d;
      tv = dx * dx + dy * dy;
    } else {
      tv = I[N * M * index + N * i + j];
    }
  }

  return tv;
}

__global__ void TSVVector(float* __restrict__ STV,
                          const float* __restrict__ noise,
                          const float* __restrict__ I,
                          long N,
                          long M,
                          float noise_cut,
                          int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  STV[N * i + j] = calculateTSV(I, noise_val, noise_cut, index, M, N);
}

__device__ float calculateDTSV(const float* __restrict__ I,
                               float lambda,
                               float noise,
                               float noise_cut,
                               int index,
                               int M,
                               int N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dstv = 0.0f;
  if (noise < noise_cut) {
    if ((i > 0 && i < M - 1) && (j > 0 && j < N - 1)) {
      const float c = I[N * M * index + N * i + j];
      const float d = I[N * M * index + N * (i + 1) + j];
      const float u = I[N * M * index + N * (i - 1) + j];
      const float r = I[N * M * index + N * i + (j + 1)];
      const float l = I[N * M * index + N * i + (j - 1)];

      dstv = 8.0f * c - 2.0f * (u + l + d + r);
    } else {
      dstv = I[N * M * index + N * i + j];
    }
  }

  dstv *= lambda;
  return dstv;
}

__global__ void DTSV(float* __restrict__ dSTV,
                     const float* __restrict__ I,
                     const float* __restrict__ noise,
                     float noise_cut,
                     float lambda,
                     long N,
                     long M,
                     int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  dSTV[N * i + j] = calculateDTSV(I, lambda, noise_val, noise_cut, index, M, N);
}
