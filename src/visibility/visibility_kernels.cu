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

#include "visibility/visibility_kernels.cuh"
#include "utils/complexOps.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math_constants.h>

/*--------------------------------------------------------------------
 * Device functions for phase rotation
 *--------------------------------------------------------------------*/

// Compute frequencies and relative phase coordinates for DC at center case
__device__ void computeFrequenciesAndPhaseCenter(int j,
                                                 int k,
                                                 long M,
                                                 long N,
                                                 double xphs,
                                                 double yphs,
                                                 double reference_column,
                                                 double reference_row,
                                                 double& u_freq,
                                                 double& v_freq,
                                                 double& xphs_relative,
                                                 double& yphs_relative) {
  // DC at center: frequencies are centered after fftshift
  // Frequencies: [-N/2, ..., -1, 0, 1, ..., N/2-1] / N
  // reference_column / reference_row are 0-based (same convention as thread indices j, k and
  // FieldMetadata phs_*), from Image::imaging_geometry().

  // u frequencies (column direction, width N) - u maps to j (columns)
  u_freq = ((double)j - reference_column) / (double)N;

  // v frequencies (row direction, height M) - v maps to k (rows)
  v_freq = ((double)k - reference_row) / (double)M;

  xphs_relative = xphs - reference_column;
  yphs_relative = yphs - reference_row;
}

// Compute frequencies and relative phase coordinates for DC at corner case
__device__ void computeFrequenciesAndPhaseCorner(int j,
                                                 int k,
                                                 long M,
                                                 long N,
                                                 double xphs,
                                                 double yphs,
                                                 double& u_freq,
                                                 double& v_freq,
                                                 double& xphs_relative,
                                                 double& yphs_relative) {
  // DC at corner: frequencies go from 0 to N-1 (standard fftfreq)
  // u frequencies (column direction)
  if (j < (N + 1) / 2) {
    u_freq = (double)j / (double)N;  // Positive frequencies
  } else {
    u_freq = ((double)j - (double)N) / (double)N;  // Negative frequencies
  }

  // v frequencies (row direction)
  if (k < (M + 1) / 2) {
    v_freq = (double)k / (double)M;  // Positive frequencies
  } else {
    v_freq = ((double)k - (double)M) / (double)M;  // Negative frequencies
  }

  // FFT grid corner is at (0, 0), so coordinates are already relative to corner
  xphs_relative = xphs;  // Already relative to corner (0,0)
  yphs_relative = yphs;
}

/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(-2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex* __restrict__ data,
                             long M,
                             long N,
                             double xphs,
                             double yphs,
                             double reference_column,
                             double reference_row,
                             bool dc_at_center) {
  // cuFFT uses row-major layout: data[N * row + column]
  // Match gridding convention: j = column (u direction), k = row (v direction)
  // Array indexing: V[N * row + column] = V[N * k + j]
  const int j =
      threadIdx.x + blockDim.x * blockIdx.x;  // Column index (u direction)
  const int k =
      threadIdx.y + blockDim.y * blockIdx.y;  // Row index (v direction)

  if (j < N && k < M) {
    double u_freq, v_freq;
    double xphs_relative, yphs_relative;

    if (dc_at_center) {
      computeFrequenciesAndPhaseCenter(j, k, M, N, xphs, yphs, reference_column,
                                       reference_row, u_freq, v_freq, xphs_relative,
                                       yphs_relative);
    } else {
      computeFrequenciesAndPhaseCorner(j, k, M, N, xphs, yphs, u_freq, v_freq,
                                       xphs_relative, yphs_relative);
    }

    double phase =
        -2.0 * CUDART_PI * (u_freq * xphs_relative + v_freq * yphs_relative);

    float c, s;
#if (__CUDA_ARCH__ >= 300)
    sincosf((float)phase, &s, &c);
#else
    c = cosf((float)phase);
    s = sinf((float)phase);
#endif
    cufftComplex exp_phase =
        make_cuFloatComplex(c, s);  // e^(-2πi*(u*x0 + v*y0))
    // Array indexing matches cuFFT and gridding: V[N * row + column] = V[N * k
    // + j]
    data[N * k + j] =
        cuCmulf(data[N * k + j], exp_phase);  // Complex multiplication
  }
}

/*--------------------------------------------------------------------
 * Device functions for bilinear interpolation
 *--------------------------------------------------------------------*/

// Bilinear interpolation for DC at center case
__device__ bool interpolateVisibilityCenter(const double3& uvw,
                                            const double deltau,
                                            const double deltav,
                                            const long M,
                                            const long N,
                                            const cufftComplex* __restrict__ V,
                                            cufftComplex& result) {
  // Default 0-based reference matches Image::imaging_geometry() when no FITS header: N/2, M/2.
  const double reference_column = floor(N / 2.0);
  const double reference_row = floor(M / 2.0);

  // Standard bilinear interpolation: continuous coordinate without +0.5
  // The +0.5 is only for rounding to nearest pixel (used in gridding), not for
  // interpolation Match Python implementation: u_pix = u/deltau + center (no
  // +0.5 for bilinear) Match gridding: j = column (u), k = row (v)
  double grid_pos_x = uvw.x / deltau;  // u -> j (column)
  double grid_pos_y = uvw.y / deltav;  // v -> k (row)

  // Continuous coordinate in grid space (centered) - NO +0.5 for bilinear
  // interpolation Gridding uses +0.5 for rounding to nearest pixel center, but
  // for bilinear interpolation we need to interpolate between pixel centers (at
  // integer positions), so we use floor
  double j_cont = grid_pos_x + reference_column;  // column (u), continuous
  double k_cont = grid_pos_y + reference_row;     // row (v), continuous

  // Get integer parts using floor (standard bilinear interpolation)
  // Note: floor gives us the lower-left corner of the interpolation cell
  int j1 = __double2int_rd(j_cont);  // Column index (u) - floor
  int j2 = j1 + 1;
  double du = j_cont - j1;  // Fractional part for interpolation

  int i1 = __double2int_rd(
      k_cont);  // Row index (v) - using i1 for row to match array indexing
  int i2 = i1 + 1;
  double dv = k_cont - i1;  // Fractional part for interpolation

  // Check all boundaries explicitly (no wrapping)
  // Note: j1, j2 are column indices (u), i1, i2 are row indices (v)
  // Array indexing must match gridding: V[N * row + column] = V[N * k + j] =
  // V[N * i + j] where i = row (v direction), j = column (u direction) For
  // bilinear interpolation, we need all four corners to be valid Allow i1, j1
  // to be -1 (for coordinates just below 0 after centering) but ensure i2, j2
  // are within bounds
  if (j1 >= -1 && j1 < N && i1 >= -1 && i1 < M && j2 >= 0 && j2 < N &&
      i2 >= 0 && i2 < M) {
    // Clamp negative indices to 0 for array access
    int j1_safe = (j1 < 0) ? 0 : j1;
    int i1_safe = (i1 < 0) ? 0 : i1;
    // Use regular global memory with __ldg for read-only cached access
    // Array layout matches gridding: V[N * row + column] = V[N * i + j]
    // where i = row index (v), j = column index (u)
    // Use safe indices to handle edge cases where i1 or j1 might be -1
    const cufftComplex v11 =
        __ldg(&V[N * i1_safe + j1_safe]);  // (i1, j1) = (row, column)
    const cufftComplex v12 = __ldg(&V[N * i1_safe + j2]);  // (i1, j2)
    const cufftComplex v21 = __ldg(&V[N * i2 + j1_safe]);  // (i2, j1)
    const cufftComplex v22 = __ldg(&V[N * i2 + j2]);       // (i2, j2)

    // Optimized bilinear interpolation weights
    const float w11 = (1.0f - du) * (1.0f - dv);
    const float w12 = du * (1.0f - dv);
    const float w21 = (1.0f - du) * dv;
    const float w22 = du * dv;

    result = make_cuFloatComplex(
        w11 * v11.x + w12 * v12.x + w21 * v21.x + w22 * v22.x,
        w11 * v11.y + w12 * v12.y + w21 * v21.y + w22 * v22.y);
    return true;
  }
  return false;
}

// Bilinear interpolation for DC at corner case
__device__ bool interpolateVisibilityCorner(const double3& uvw,
                                            const double deltau,
                                            const double deltav,
                                            const long M,
                                            const long N,
                                            const cufftComplex* __restrict__ V,
                                            cufftComplex& result) {
  // DC at corner: handle negative coordinates with wrapping
  // Match gridding convention: j = column (u), k = row (v)
  double grid_pos_x = uvw.x / deltau;  // u -> j (column)
  double grid_pos_y = uvw.y / deltav;  // v -> k (row)

  // Handle negative coordinates by wrapping
  if (grid_pos_x < 0.0)
    grid_pos_x += N;  // Wrap u (columns, width N)
  if (grid_pos_y < 0.0)
    grid_pos_y += M;  // Wrap v (rows, height M)

  // Get integer parts (floor)
  int j1 = __double2int_rd(grid_pos_x);  // Column index (u)
  int j2 = (j1 + 1) % N;                 // Wrap around for columns
  double du = grid_pos_x - j1;           // Fractional part

  int i1 = __double2int_rd(grid_pos_y);  // Row index (v) - using i1 for row
  int i2 = (i1 + 1) % M;                 // Wrap around for rows (height M)
  double dv = grid_pos_y - i1;           // Fractional part

  // Boundary check: j1, j2 are columns (u), i1, i2 are rows (v)
  // Array indexing must match gridding: V[N * row + column] = V[N * i + j]
  // where i = row (v direction, height M), j = column (u direction, width N)
  if (j1 >= 0 && j1 < N && i1 >= 0 && i1 < M) {
    // Use regular global memory with __ldg for read-only cached access
    // Note: i2 and j2 are wrapped, so they're always in range due to modulo
    const cufftComplex v11 =
        __ldg(&V[N * i1 + j1]);  // (i1, j1) = (row, column)
    const cufftComplex v12 = __ldg(&V[N * i1 + j2]);  // (i1, j2)
    const cufftComplex v21 = __ldg(&V[N * i2 + j1]);  // (i2, j1)
    const cufftComplex v22 = __ldg(&V[N * i2 + j2]);  // (i2, j2)

    // Optimized bilinear interpolation weights (same as dc_at_center path)
    const float w11 =
        (1.0f - du) * (1.0f - dv);       // Weight for (i1, j1) = lower-left
    const float w12 = du * (1.0f - dv);  // Weight for (i1, j2) = lower-right
    const float w21 = (1.0f - du) * dv;  // Weight for (i2, j1) = upper-left
    const float w22 = du * dv;           // Weight for (i2, j2) = upper-right

    const float Zreal = w11 * v11.x + w12 * v12.x + w21 * v21.x + w22 * v22.x;
    const float Zimag = w11 * v11.y + w12 * v12.y + w21 * v21.y + w22 * v22.y;

    result = make_cuFloatComplex(Zreal, Zimag);
    return true;
  }
  return false;
}

/*
 * Bilinear interpolation of visibilities from gridded visibility plane
 * dc_at_center: true if DC component is at center (N/2, M/2), false if at
 * corner (0,0) This unified function replaces vis_mod (DC at corner) and
 * vis_mod2 (DC at center)
 */
__global__ void bilinearInterpolateVisibility(
    cufftComplex* __restrict__ Vm,
    const cufftComplex* __restrict__ V,
    const double3* __restrict__ UVW,
    float* __restrict__ weight,
    const double deltau,
    const double deltav,
    const long numVisibilities,
    const long M,
    const long N,
    const bool dc_at_center) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    // Load UVW once (double3 is 64-bit, so __ldg doesn't apply)
    const double3 uvw = UVW[i];
    cufftComplex result;
    bool success;

    if (dc_at_center) {
      success =
          interpolateVisibilityCenter(uvw, deltau, deltav, M, N, V, result);
    } else {
      success =
          interpolateVisibilityCorner(uvw, deltau, deltav, M, N, V, result);
    }

    if (success) {
      Vm[i] = result;
    } else {
      weight[i] = 0.0f;
    }
  }
}

__global__ void residual(cufftComplex* __restrict__ Vr,
                         const cufftComplex* __restrict__ Vm,
                         const cufftComplex* __restrict__ Vo,
                         long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < numVisibilities) {
    Vr[i] = cuCsubf(Vo[i], Vm[i]);
  }
}
