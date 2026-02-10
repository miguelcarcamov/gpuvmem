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

#include "linesearch/linesearch_kernels.cuh"
#include <cuda_runtime.h>

__global__ void newP(float* p,
                     float* xi,
                     float xmin,
                     float MINPIX,
                     float eta,
                     long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N * i + j] *= xmin;
  if (p[N * i + j] + xi[N * i + j] > -1.0f * eta * MINPIX) {
    p[N * i + j] += xi[N * i + j];
  } else {
    p[N * i + j] = -1.0f * eta * MINPIX;
    xi[N * i + j] = 0.0f;
  }
  // p[N*i+j].y = 0.0;
}

__global__ void newP(float* p,
                     float* xi,
                     float xmin,
                     long N,
                     long M,
                     float MINPIX,
                     float eta,
                     int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Bounds check to prevent illegal memory access
  if (i >= M || j >= N) {
    return;
  }

  const long idx = N * M * image + N * i + j;
  xi[idx] *= xmin;

  float min_threshold = -1.0f * eta * MINPIX;

  if (p[idx] + xi[idx] > min_threshold) {
    p[idx] += xi[idx];
  } else {
    p[idx] = min_threshold;
    xi[idx] = 0.0f;
  }
}

__global__ void newPNoPositivity(float* p,
                                 float* xi,
                                 float xmin,
                                 long N,
                                 long M,
                                 int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Bounds check to prevent illegal memory access
  if (i >= M || j >= N) {
    return;
  }

  const long idx = N * M * image + N * i + j;
  xi[idx] *= xmin;
  p[idx] += xi[idx];
}

__global__ void evaluateXt(float* xt,
                           float* pcom,
                           float* xicom,
                           float x,
                           float MINPIX,
                           float eta,
                           long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (pcom[N * i + j] + x * xicom[N * i + j] > -1.0f * eta * MINPIX) {
    xt[N * i + j] = pcom[N * i + j] + x * xicom[N * i + j];
  } else {
    xt[N * i + j] = -1.0f * eta * MINPIX;
  }
  // xt[N*i+j].y = 0.0;
}

__global__ void evaluateXt(float* xt,
                           float* pcom,
                           float* xicom,
                           float x,
                           long N,
                           long M,
                           float MINPIX,
                           float eta,
                           int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float min_threshold = -1.0f * eta * MINPIX;

  if (pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j] >
      min_threshold) {
    xt[N * M * image + N * i + j] =
        pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j];
  } else {
    xt[N * M * image + N * i + j] = min_threshold;
  }
}

__global__ void evaluateXtNoPositivity(float* xt,
                                       float* pcom,
                                       float* xicom,
                                       float x,
                                       long N,
                                       long M,
                                       int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  xt[N * M * image + N * i + j] =
      pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j];
}
