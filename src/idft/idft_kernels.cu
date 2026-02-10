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
#include "idft/idft_kernels.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Device function to compute IDFT for a single pixel
// This can be called from both the idft kernel and DChi2 kernel
__device__ float computeIdftPixel(int i,
                                   int j,
                                   cufftComplex* visibilities,
                                   double3* uvw,
                                   float* weights,
                                   long N,
                                   long numVisibilities,
                                   float phs_xobs,
                                   float phs_yobs,
                                   double DELTAX,
                                   double DELTAY) {
  // Compute pixel coordinates and direction cosines
  const int x0 = phs_xobs;
  const int y0 = phs_yobs;
  const double x = (j - x0) * DELTAX * RPDEG_D;
  const double y = (i - y0) * DELTAY * RPDEG_D;
  const double z = sqrt(1.0 - x * x - y * y);

  // Accumulate IDFT result
  float idft_result = 0.0f;

  // Precompute constants for the loop
  const double z_minus_one = z - 1.0;
  const double two = 2.0;

  // Unroll loop for better performance (compiler hint)
#pragma unroll 4
  for (int v = 0; v < numVisibilities; v++) {
    // Load UVW values - compiler will optimize memory access
    const double uvw_x = uvw[v].x;
    const double uvw_y = uvw[v].y;
    const double uvw_z = uvw[v].z;

    // Compute phase components (FMA-friendly operations)
    const double Ukv = x * uvw_x;
    const double Vkv = y * uvw_y;
    const double Wkv = z_minus_one * uvw_z;
    const double phase = two * (Ukv + Vkv + Wkv);

    // Compute sin and cos of phase
    float cosk, sink;
#if (__CUDA_ARCH__ >= 300)
    sincospif(phase, &sink, &cosk);
#else
    cosk = cospif(phase);
    sink = sinpif(phase);
#endif

    // Load visibility and weight - use __ldg for read-only scalar values
    const float weight = __ldg(&weights[v]);
    const float vr_real = __ldg(&visibilities[v].x);
    const float vr_imag = __ldg(&visibilities[v].y);

    // Accumulate weighted contribution using FMA for better precision and
    // performance
    // IDFT: Re[exp(2πi*phase) * V] = cos(phase) * Re(V) - sin(phase) * Im(V)
    // But we accumulate: weight * (vr_real * cosk + vr_imag * sink)
    // This matches the adjoint operation
    idft_result = fmaf(weight, vr_real * cosk + vr_imag * sink, idft_result);
  }

  return idft_result;
}

// Inverse Direct Fourier Transform kernel
// Computes IDFT: I(x,y) = Σ_v w_v * V_r(v) * exp(2πi * (u*x + v*y + w*(z-1)))
// This is the adjoint of the forward DFT used in the measurement operator
__global__ void idft(float* output_image,
                     cufftComplex* visibilities,
                     double3* uvw,
                     float* weights,
                     long N,
                     long M,
                     long numVisibilities,
                     float phs_xobs,
                     float phs_yobs,
                     double DELTAX,
                     double DELTAY) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (j >= N || i >= M) {
    return;
  }

  // Compute IDFT for this pixel
  float idft_result = computeIdftPixel(i, j, visibilities, uvw, weights, N,
                                        numVisibilities, phs_xobs, phs_yobs,
                                        DELTAX, DELTAY);

  // Store IDFT result
  output_image[N * i + j] = idft_result;
}
