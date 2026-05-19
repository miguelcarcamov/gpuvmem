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

#include "image_processing/image_processing_kernels.cuh"
#include "utils/complexOps.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void clipWNoise(cufftComplex* fg_image,
                           float* noise,
                           float* I,
                           long N,
                           float noise_cut,
                           float MINPIX,
                           float eta) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (noise[N * i + j] > noise_cut) {
    if (eta > 0.0) {
      I[N * i + j] = 0.0;
    } else {
      I[N * i + j] = -1.0f * eta * MINPIX;
    }
  }

  fg_image[N * i + j] = make_cuFloatComplex(I[N * i + j], 0.0f);
}

__global__ void clip2IWNoise(float* noise,
                             float* I,
                             long N,
                             long M,
                             float noise_cut,
                             float MINPIX,
                             float alpha_start,
                             float eta,
                             float threshold,
                             float alpha_n_sigma,
                             int schedule) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (noise[N * i + j] > noise_cut) {
    if (eta > 0.0f) {
      I[N * i + j] = 1e-10f;
    } else {
      I[N * i + j] = -1.0f * eta * MINPIX;
    }
    I[N * M + N * i + j] = 0.0f;
  }
  // Alpha masking/clipping removed - alpha is no longer masked based on I_nu_0 threshold
}

__global__ void clip2I(float* I, long N, float MINPIX) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (I[N * i + j] < MINPIX && MINPIX >= 0.0) {
    I[N * i + j] = MINPIX;
  }
}

__global__ void normalizeImageKernel(float* image,
                                     float normalization_factor,
                                     long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  image[N * i + j] /= normalization_factor;
}

__global__ void substraction(float* x,
                             cufftComplex* xc,
                             float* gc,
                             float lambda,
                             long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  x[N * i + j] = xc[N * i + j].x - lambda * gc[N * i + j];
}

__global__ void projection(float* px, float* x, float MINPIX, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (INFINITY < x[N * i + j]) {
    px[N * i + j] = INFINITY;
  } else {
    px[N * i + j] = x[N * i + j];
  }

  if (MINPIX > px[N * i + j]) {
    px[N * i + j] = MINPIX;
  } else {
    px[N * i + j] = px[N * i + j];
  }
}

__global__ void normVectorCalculation(float* normVector, float* gc, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  normVector[N * i + j] = gc[N * i + j] * gc[N * i + j];
}

__global__ void copyImage(cufftComplex* p, float* device_xt, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  p[N * i + j].x = device_xt[N * i + j];
}

// Clip by noise: Stokes/single-plane (not MFS).
// Only Stokes I (plane 0) is clipped to a non-negative floor; Q,U,V (planes 1..) are
// left unchanged since they can and must be negative.
__global__ void clipStokesWNoise(float* I,
                                 int nplanes,
                                 long N,
                                 long M,
                                 float* noise,
                                 float noise_cut,
                                 float MINPIX,
                                 float eta) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= M || j >= N) return;
  const long idx = N * i + j;
  if (noise[idx] <= noise_cut) return;
  float clip_val = (eta > 0.0f) ? 1e-10f : -1.0f * eta * MINPIX;
  // Plane 0 = Stokes I: enforce non-negativity where noise is high.
  I[idx] = clip_val;
  // Planes 1.. = Q,U,V: do not clip (they can be negative).
}

/** Copy single-plane float image to complex (real = I, imag = 0). Used for Stokes/single-image mode. */
__global__ void copyItoInu(cufftComplex* image, const float* I, long M, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= M || j >= N) return;
  long idx = N * i + j;
  image[idx].x = I[idx];
  image[idx].y = 0.0f;
}
