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

#include "fft/fft_kernels.cuh"
#include <cufft.h>

// Device function: Proper fftshift using quadrant swapping
// Works correctly for both even and odd dimensions
// For even N: ceil(N/2) = N/2, swaps first N/2 with second N/2
// For odd N: ceil(N/2) = (N+1)/2, swaps first (N+1)/2 with last (N-1)/2
// fftshift swaps: Q1(top-left)↔Q3(bottom-right), Q2(top-right)↔Q4(bottom-left)
__device__ __forceinline__ void fftshift_swap(cufftComplex* data,
                                              int i,
                                              int j,
                                              int N1,
                                              int N2) {
  const int h2 = (N1 + 1) / 2;  // ceil(N1/2) for fftshift
  const int w2 = (N2 + 1) / 2;  // ceil(N2/2) for fftshift

  // Process only elements in first half to avoid double-swapping
  if (i < h2) {
    int i2 = i + (N1 - h2);  // Target row: shift by (N1 - ceil(N1/2))
    // For even: N1 - N1/2 = N1/2, for odd: N1 - (N1+1)/2 = (N1-1)/2

    if (j < w2) {
      // Top-left quadrant (Q1): swap with bottom-right (Q3)
      int j2 = j + (N2 - w2);  // Target column: shift right by (N2 - w2)
      if (i2 < N1 && j2 < N2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    } else if (j >= w2 && j < N2) {
      // Top-right quadrant (Q2): swap with bottom-left (Q4)
      int j2 = j - w2;  // Target column: shift left by w2
      if (i2 < N1 && j2 >= 0 && j2 < w2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    }
  }
}

// Device function: Proper ifftshift using quadrant swapping
// Works correctly for both even and odd dimensions
// For even N: floor(N/2) = N/2, swaps first N/2 with second N/2 (same as
// fftshift) For odd N: floor(N/2) = (N-1)/2, swaps first (N-1)/2 with last
// (N+1)/2 ifftshift swaps: Q1(top-left)↔Q3(bottom-right),
// Q2(top-right)↔Q4(bottom-left)
__device__ __forceinline__ void ifftshift_swap(cufftComplex* data,
                                               int i,
                                               int j,
                                               int N1,
                                               int N2) {
  const int h2 = N1 / 2;  // floor(N1/2) for ifftshift
  const int w2 = N2 / 2;  // floor(N2/2) for ifftshift

  // Process only elements in first half to avoid double-swapping
  if (i < h2) {
    int i2 = i + (N1 - h2);  // Target row: shift by (N1 - floor(N1/2))
    // For even: N1 - N1/2 = N1/2, for odd: N1 - (N1-1)/2 = (N1+1)/2

    if (j < w2) {
      // Top-left quadrant (Q1): swap with bottom-right (Q3)
      int j2 = j + (N2 - w2);  // Target column: shift right by (N2 - w2)
      if (i2 < N1 && j2 < N2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    } else if (j >= w2 && j < N2) {
      // Top-right quadrant (Q2): swap with bottom-left (Q4)
      // For even N: w2 = N/2, so Q2 has j in [w2, N-1] = [N/2, N-1] (N/2
      // elements)
      //            and Q4 has j in [0, w2-1] = [0, N/2-1] (N/2 elements)
      //            They match perfectly, so j2 = j - w2 maps all of Q2 to Q4
      // For odd N: Q2 has ceil(N/2) elements, Q4 has floor(N/2) elements
      //            Only first floor(N/2) elements of Q2 swap with Q4
      bool should_swap = true;
      int j2 = j - w2;

      if (N2 % 2 != 0) {
        // Odd N: only first w2 elements of Q2 swap with Q4
        if (j >= w2 + w2) {
          // Last element of Q2 (for odd N) doesn't swap with Q4, skip it
          should_swap = false;
        }
      }

      if (should_swap && i2 < N1 && j2 >= 0 && j2 < w2) {
        const int idx1 = N2 * i + j;
        const int idx2 = N2 * i2 + j2;
        if (idx1 != idx2) {
          const cufftComplex tmp = data[idx1];
          data[idx1] = data[idx2];
          data[idx2] = tmp;
        }
      }
    }
  }
}

// Generic fftshift: uses quadrant swapping for all dimensions
// For even dimensions, fftshift and ifftshift are identical
// For odd dimensions, they differ by one sample
// Optimized fftshift: computes target index directly (more efficient for even
// dimensions) For even dimensions: shift by N/2, for odd: shift by ceil(N/2)
__global__ void fftshift_2D(cufftComplex* __restrict__ data, int N1, int N2) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N1 && j < N2) {
    // Compute target indices directly (avoids quadrant checking overhead)
    const int h2 = (N1 + 1) / 2;  // ceil(N1/2) for fftshift
    const int w2 = (N2 + 1) / 2;  // ceil(N2/2) for fftshift

    // Only process first half to avoid double-swapping
    if (i < h2) {
      int i2 = i + (N1 - h2);  // Target row

      if (j < w2) {
        // Q1 (top-left): swap with Q3 (bottom-right)
        int j2 = j + (N2 - w2);  // Target column
        if (i2 < N1 && j2 < N2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      } else if (j >= w2 && j < N2) {
        // Q2 (top-right): swap with Q4 (bottom-left)
        int j2 = j - w2;  // Target column
        if (i2 < N1 && j2 >= 0 && j2 < w2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      }
    }
  }
}

// Optimized ifftshift: computes target index directly (more efficient for even
// dimensions) For even dimensions: same as fftshift, for odd: shift by
// floor(N/2)
__global__ void ifftshift_2D(cufftComplex* __restrict__ data, int N1, int N2) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N1 && j < N2) {
    // Compute target indices directly (avoids quadrant checking overhead)
    const int h2 = N1 / 2;  // floor(N1/2) for ifftshift
    const int w2 = N2 / 2;  // floor(N2/2) for ifftshift

    // Only process first half to avoid double-swapping
    if (i < h2) {
      int i2 = i + (N1 - h2);  // Target row

      if (j < w2) {
        // Q1 (top-left): swap with Q3 (bottom-right)
        int j2 = j + (N2 - w2);  // Target column
        if (i2 < N1 && j2 < N2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      } else if (j >= w2 && j < N2) {
        // Q2 (top-right): swap with Q4 (bottom-left)
        // For even N: all of Q2 swaps; for odd N: only first w2 elements
        bool should_swap = true;
        int j2 = j - w2;

        if (N2 % 2 != 0 && j >= w2 + w2) {
          // Odd N: skip last element of Q2
          should_swap = false;
        }

        if (should_swap && i2 < N1 && j2 >= 0 && j2 < w2) {
          const int idx1 = N2 * i + j;
          const int idx2 = N2 * i2 + j2;
          if (idx1 != idx2) {
            const cufftComplex tmp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = tmp;
          }
        }
      }
    }
  }
}
