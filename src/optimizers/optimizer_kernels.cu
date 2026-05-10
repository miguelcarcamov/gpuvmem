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

#include "optimizers/optimizer_kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void searchDirection(float* g, float* xi, float* h, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N * i + j] = -xi[N * i + j];
  xi[N * i + j] = h[N * i + j] = g[N * i + j];
}

__global__ void searchDirection_LBFGS(float* xi, long N, long M, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[M * N * image + N * i + j] *= -1.0f;
}

__global__ void getDot_LBFGS_ff(float* aux_vector,
                                float* vec_1,
                                float* vec_2,
                                int k,
                                int h,
                                int M,
                                int N,
                                int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  aux_vector[N * i + j] =
      vec_1[M * N * image * k + M * N * image + (N * i + j)] *
      vec_2[M * N * image * h + M * N * image + (N * i + j)];
}

__global__ void normArray(float* result,
                          float* array,
                          int M,
                          int N,
                          int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  result[M * N * image + (N * i + j)] =
      fabsf(array[M * N * image + (N * i + j)]);
}

__global__ void CGGradCondition(float* temp,
                                float* xi,
                                float* p,
                                float den,
                                int M,
                                int N,
                                int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  temp[M * N * image + (N * i + j)] =
      fabsf(xi[M * N * image + (N * i + j)]) *
      fmaxf(fabsf(p[M * N * image + (N * i + j)]), 1.0f) / den;
}

__global__ void
updateQ(float* d_q, float alpha, float* d_y, int k, int M, int N, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_q[M * N * image + N * i + j] +=
      alpha * d_y[M * N * image * k + M * N * image + (N * i + j)];
}

__global__ void getR(float* d_r,
                     float* d_q,
                     float scalar,
                     int M,
                     int N,
                     int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_r[M * N * image + N * i + j] = d_q[M * N * image + N * i + j] * scalar;
}

__global__ void calculateSandY(float* d_y,
                               float* d_s,
                               float* p,
                               float* xi,
                               float* p_old,
                               float* xi_old,
                               int iter,
                               int M,
                               int N,
                               int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_y[M * N * image * iter + M * N * image + (N * i + j)] =
      xi[M * N * image + N * i + j] -
      (-1.0f * xi_old[M * N * image + N * i + j]);
  d_s[M * N * image * iter + M * N * image + (N * i + j)] =
      p[M * N * image + N * i + j] - p_old[M * N * image + N * i + j];
}

__global__ void calculateSandYScratch(float* scratch_y,
                                      float* scratch_s,
                                      float* p,
                                      float* xi,
                                      float* p_old,
                                      float* xi_old,
                                      int M,
                                      int N,
                                      int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  scratch_y[M * N * image + N * i + j] =
      xi[M * N * image + N * i + j] -
      (-1.0f * xi_old[M * N * image + N * i + j]);
  scratch_s[M * N * image + N * i + j] =
      p[M * N * image + N * i + j] - p_old[M * N * image + N * i + j];
}

__global__ void searchDirection(float* g,
                                float* xi,
                                float* h,
                                long N,
                                long M,
                                int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[M * N * image + N * i + j] = -xi[M * N * image + N * i + j];
  xi[M * N * image + N * i + j] = h[M * N * image + N * i + j] =
      g[M * N * image + N * i + j];
}

__global__ void newXi(float* g, float* xi, float* h, float gam, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N * i + j] = h[N * i + j] = g[N * i + j] + gam * h[N * i + j];
}

__global__ void newXi(float* g, float* xi, float* h, float gam, long N, long M, int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[M * N * image + N * i + j] = h[M * N * image + N * i + j] =
      g[M * N * image + N * i + j] + gam * h[M * N * image + N * i + j];
}

__global__ void getGandDGG(float* gg, float* dgg, float* xi, float* g, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  gg[N * i + j] = g[N * i + j] * g[N * i + j];
  dgg[N * i + j] = (xi[N * i + j] - g[N * i + j]) * xi[N * i + j];
}

__global__ void getGGandDGG(float* gg,
                            float* dgg,
                            float* xi,
                            float* g,
                            long N,
                            long M,
                            int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  gg[M * N * image + N * i + j] =
      g[M * N * image + N * i + j] * g[M * N * image + N * i + j];
  dgg[M * N * image + N * i + j] =
      (xi[M * N * image + N * i + j] - g[M * N * image + N * i + j]) *
      xi[M * N * image + N * i + j];
}

__global__ void restartDPhi(float* dphi, float* dChi2, float* dH, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N * i + j] = dChi2[N * i + j] + dH[N * i + j];
}

// AddToDPhi moved to chi2/chi2_kernels.cu (single-plane dgi version)
// If multi-plane version is needed, it should be added separately
