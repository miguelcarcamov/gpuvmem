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

#include "gridding/gridding_kernels.cuh"
#include "utils/complexOps.cuh"
#include "io/MSFITSIO.cuh"
#include <cufft.h>

__global__ void do_griddingGPU(float3* uvw,
                               cufftComplex* Vo,
                               cufftComplex* Vo_g,
                               float* w,
                               float* w_g,
                               int* count,
                               double deltau,
                               double deltav,
                               int visibilities,
                               int M,
                               int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k, j;
  if (i < visibilities) {
    // Precompute center values once
    const double center_j = floor(M / 2.0);
    const double center_k = floor(N / 2.0);

    // Use __ldg for read-only cached access to input data
    const float3 uvw_val = uvw[i];  // float3 is 12 bytes, __ldg doesn't apply
                                    // but compiler may optimize
    const cufftComplex Vo_val = __ldg(&Vo[i]);
    const float w_val = __ldg(&w[i]);

    j = (int)(uvw_val.x / deltau + center_j + 0.5);
    k = (int)(uvw_val.y / deltav + center_k + 0.5);

    if (k < M && j < N) {
      const int grid_idx = N * k + j;
      atomicAdd(&Vo_g[grid_idx].x, w_val * Vo_val.x);
      atomicAdd(&Vo_g[grid_idx].y, w_val * Vo_val.y);
      atomicAdd(&w_g[grid_idx], w_val);
    }
  }
}

__global__ void degriddingGPU(double3* uvw,
                              cufftComplex* Vm,
                              cufftComplex* Vm_g,
                              float* kernel,
                              double deltau,
                              double deltav,
                              int visibilities,
                              int M,
                              int N,
                              int kernel_m,
                              int kernel_n,
                              int supportX,
                              int supportY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k, j;
  int shifted_k, shifted_j;
  int kernel_i, kernel_j;
  cufftComplex degrid_val = floatComplexZero();
  float ckernel_result;

  if (i < visibilities) {
    // Match gridding coordinate calculation exactly:
    // grid_pos = uvw / deltau, j = int(grid_pos + center + 0.5)
    // Use double precision for center calculation to match gridding
    double center_j = floor(N / 2.0);
    double center_k = floor(M / 2.0);
    double grid_pos_x = uvw[i].x / deltau;
    double grid_pos_y = uvw[i].y / deltav;
    j = int(grid_pos_x + center_j + 0.5);
    k = int(grid_pos_y + center_k + 0.5);

    for (int m = -supportY; m <= supportY; m++) {
      for (int n = -supportX; n <= supportX; n++) {
        shifted_j = j + n;
        shifted_k = k + m;
        kernel_j = n + supportX;
        kernel_i = m + supportY;
        // Check bounds: grid must be valid AND kernel indices must be valid
        if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
            shifted_j < N && kernel_i >= 0 && kernel_i < kernel_m &&
            kernel_j >= 0 && kernel_j < kernel_n) {
          // Use __ldg for read-only cached access to kernel (read-only)
          ckernel_result = __ldg(&kernel[kernel_n * kernel_i + kernel_j]);
          // Direct grid access - no Hermitian symmetry handling needed
          // The FFT grid from ifft2 of a real image already has full complex
          // values at all positions, so we can sample directly at any (u,v)
          // coordinate
          const cufftComplex grid_val = __ldg(&Vm_g[N * shifted_k + shifted_j]);
          degrid_val.x += ckernel_result * grid_val.x;
          degrid_val.y += ckernel_result * grid_val.y;
        }
      }
    }
    Vm[i].x = degrid_val.x;
    Vm[i].y = degrid_val.y;
  }
}

// Apply Hermitian symmetry only (flip u,v,w and conjugate visibility when u >
// 0) Hermitian symmetry: V(u,v,w) = V*(-u,-v,-w)* where * denotes complex
// conjugate Does NOT convert to lambda units - use convertUVWToLambda
// separately
__global__ void applyHermitianSymmetry(double3* UVW,
                                       cufftComplex* Vo,
                                       int numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    if (UVW[i].x > 0.0) {
      UVW[i].x *= -1.0;  // Negate u coordinate
      UVW[i].y *= -1.0;  // Negate v coordinate
      UVW[i].z *= -1.0;  // Negate w coordinate (for consistency with gridding)
      Vo[i].y *= -1.0f;  // Conjugate: flip imaginary part
    }
  }
}

// Convert UVW coordinates from meters to lambda units without applying
// Hermitian symmetry
__global__ void convertUVWToLambda(double3* UVW,
                                   float freq,
                                   int numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    UVW[i].x = metres_to_lambda(UVW[i].x, freq);
    UVW[i].y = metres_to_lambda(UVW[i].y, freq);
    UVW[i].z = metres_to_lambda(UVW[i].z, freq);
  }
}
