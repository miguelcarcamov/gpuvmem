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

#include "chi2/chi2_kernels.cuh"
#include "beam/beam_kernels.cuh"
#include "idft/idft_kernels.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void chi2Vector(float* __restrict__ chi2,
                           const cufftComplex* __restrict__ Vr,
                           const float* __restrict__ w,
                           long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    chi2[i] = w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
  }
}

// Compute squared weights for effective number of samples calculation
__global__ void weightsSquaredVector(float* __restrict__ w_squared,
                                     const float* __restrict__ w,
                                     long numVisibilities) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities) {
    w_squared[i] = w[i] * w[i];
  }
}

__global__ void DChi2(float* noise,
                      float* dChi2,
                      cufftComplex* Vr,
                      double3* UVW,
                      float* w,
                      long N,
                      long numVisibilities,
                      float fg_scale,
                      float noise_cut,
                      float ref_xobs,
                      float ref_yobs,
                      float phs_xobs,
                      float phs_yobs,
                      double DELTAX,
                      double DELTAY,
                      float antenna_diameter,
                      float pb_factor,
                      float pb_cutoff,
                      float freq,
                      int primary_beam,
                      bool normalize,
                      float N_eff) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Mask: exclude pixels with noise map value >= noise_cut (when noise_cut > 0).
  // noise_cut <= 0 means "no cut" — otherwise noise_cut==0 with a non-negative
  // noise/distance map would skip every pixel and zero the χ² gradient.
  const float noise_val = noise[N * i + j];
  if (noise_cut > 0.0f && noise_val >= noise_cut) {
    return;
  }

  // Compute IDFT for this pixel using decoupled IDFT computation
  float idft_result = computeIdftPixel(i, j, Vr, UVW, w, N, numVisibilities,
                                        phs_xobs, phs_yobs, DELTAX, DELTAY);

  // Compute attenuation only if we pass the noise check
  const float atten =
      attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                  ref_yobs, DELTAX, DELTAY, primary_beam);
  const float scale_factor = fg_scale * atten;

  // Apply scaling factors to IDFT result
  float dchi2 = idft_result * scale_factor;

  if (normalize) {
    // Normalize by effective number of samples: N_eff = (Σw_k)² / (Σw_k²)
    // This accounts for varying weights and represents effective degrees of
    // freedom. When all weights are equal, N_eff = N. When weights vary,
    // N_eff < N and properly accounts for the reduced effective sample size.
    if (N_eff > 0.0f) {
      dchi2 /= N_eff;
    } else {
      // Fallback to numVisibilities if N_eff is zero
      dchi2 /= numVisibilities;
    }
  }

  // Sign: χ² = 0.5·Σ w|Vo−Vm|² ⇒ ∂χ²/∂Vm ∝ (Vm−Vo) = −Vr. Degrid(weight·Vr) ∝ −∂χ²/∂I_ν.
  // Store dChi2 = −dchi2 so dChi2 = +∂χ²/∂I_ν (optimizer then uses descent dir −gradient).
  dChi2[N * i + j] = -dchi2;
}

__global__ void DChi2(float* noise,
                      float* gcf,
                      float* dChi2,
                      cufftComplex* Vr,
                      double3* UVW,
                      float* w,
                      long N,
                      long numVisibilities,
                      float fg_scale,
                      float noise_cut,
                      float ref_xobs,
                      float ref_yobs,
                      float phs_xobs,
                      float phs_yobs,
                      double DELTAX,
                      double DELTAY,
                      float antenna_diameter,
                      float pb_factor,
                      float pb_cutoff,
                      float freq,
                      int primary_beam,
                      bool normalize,
                      float N_eff) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  if (noise_cut > 0.0f && noise_val >= noise_cut) {
    return;
  }

  // Compute IDFT for this pixel using decoupled IDFT computation
  float idft_result = computeIdftPixel(i, j, Vr, UVW, w, N, numVisibilities,
                                        phs_xobs, phs_yobs, DELTAX, DELTAY);

  // Compute attenuation and GCF only if we pass the noise check
  const float atten =
      attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs,
                  ref_yobs, DELTAX, DELTAY, primary_beam);
  const float gcf_i = gcf[N * i + j];
  const float scale_factor = fg_scale * atten * gcf_i;

  // Apply scaling factors to IDFT result
  float dchi2 = idft_result * scale_factor;

  if (normalize) {
    // Normalize by effective number of samples: N_eff = (Σw_k)² / (Σw_k²)
    // This accounts for varying weights and represents effective degrees of
    // freedom. When all weights are equal, N_eff = N. When weights vary,
    // N_eff < N and properly accounts for the reduced effective sample size.
    if (N_eff > 0.0f) {
      dchi2 /= N_eff;
    } else {
      // Fallback to numVisibilities if N_eff is zero
      dchi2 /= numVisibilities;
    }
  }

  // Sign: χ² = 0.5·Σ w|Vo−Vm|² ⇒ ∂χ²/∂Vm ∝ (Vm−Vo) = −Vr. Degrid(weight·Vr) ∝ −∂χ²/∂I_ν.
  // Store dChi2 = −dchi2 so dChi2 = +∂χ²/∂I_ν (optimizer then uses descent dir −gradient).
  dChi2[N * i + j] = -dchi2;
}

__global__ void DChi2Baseline(float* noise,
                              float* dChi2,
                              cufftComplex* Vr,
                              double3* UVW,
                              float* w,
                              long N,
                              long numVisibilities,
                              float fg_scale,
                              float noise_cut,
                              float ref_xobs,
                              float ref_yobs,
                              float phs_xobs,
                              float phs_yobs,
                              double DELTAX,
                              double DELTAY,
                              float ant1_diameter,
                              float ant1_pb_factor,
                              float ant1_pb_cutoff,
                              int ant1_primary_beam,
                              float ant2_diameter,
                              float ant2_pb_factor,
                              float ant2_pb_cutoff,
                              int ant2_primary_beam,
                              float freq,
                              bool normalize,
                              float N_eff) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  if (noise_cut > 0.0f && noise_val >= noise_cut) {
    return;
  }

  float idft_result = computeIdftPixel(i, j, Vr, UVW, w, N, numVisibilities,
                                        phs_xobs, phs_yobs, DELTAX, DELTAY);

  const float a1 = attenuation(ant1_diameter, ant1_pb_factor, ant1_pb_cutoff, freq,
                               ref_xobs, ref_yobs, DELTAX, DELTAY, ant1_primary_beam);
  const float a2 = attenuation(ant2_diameter, ant2_pb_factor, ant2_pb_cutoff, freq,
                               ref_xobs, ref_yobs, DELTAX, DELTAY, ant2_primary_beam);
  const float comb = sqrtf(fmaxf(a1 * a2, 0.0f));
  const float scale_factor = fg_scale * comb;

  float dchi2 = idft_result * scale_factor;

  if (normalize) {
    if (N_eff > 0.0f) {
      dchi2 /= N_eff;
    } else {
      dchi2 /= numVisibilities;
    }
  }

  dChi2[N * i + j] = -dchi2;
}

__global__ void DChi2Baseline(float* noise,
                              float* gcf,
                              float* dChi2,
                              cufftComplex* Vr,
                              double3* UVW,
                              float* w,
                              long N,
                              long numVisibilities,
                              float fg_scale,
                              float noise_cut,
                              float ref_xobs,
                              float ref_yobs,
                              float phs_xobs,
                              float phs_yobs,
                              double DELTAX,
                              double DELTAY,
                              float ant1_diameter,
                              float ant1_pb_factor,
                              float ant1_pb_cutoff,
                              int ant1_primary_beam,
                              float ant2_diameter,
                              float ant2_pb_factor,
                              float ant2_pb_cutoff,
                              int ant2_primary_beam,
                              float freq,
                              bool normalize,
                              float N_eff) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  const float noise_val = noise[N * i + j];
  if (noise_cut > 0.0f && noise_val >= noise_cut) {
    return;
  }

  float idft_result = computeIdftPixel(i, j, Vr, UVW, w, N, numVisibilities,
                                        phs_xobs, phs_yobs, DELTAX, DELTAY);

  const float a1 = attenuation(ant1_diameter, ant1_pb_factor, ant1_pb_cutoff, freq,
                               ref_xobs, ref_yobs, DELTAX, DELTAY, ant1_primary_beam);
  const float a2 = attenuation(ant2_diameter, ant2_pb_factor, ant2_pb_cutoff, freq,
                               ref_xobs, ref_yobs, DELTAX, DELTAY, ant2_primary_beam);
  const float comb = sqrtf(fmaxf(a1 * a2, 0.0f));
  const float gcf_i = gcf[N * i + j];
  const float scale_factor = fg_scale * comb * gcf_i;

  float dchi2 = idft_result * scale_factor;

  if (normalize) {
    if (N_eff > 0.0f) {
      dchi2 /= N_eff;
    } else {
      dchi2 /= numVisibilities;
    }
  }

  dChi2[N * i + j] = -dchi2;
}

// Gather one (chan, pol) visibility from each chunk at offset into contiguous buffers
__global__ void gatherChunkAtOffset(double3* uvw_out,
                                    cufftComplex* Vo_out,
                                    float* weight_out,
                                    double3 const* const* uvw_ptrs,
                                    cufftComplex const* const* Vo_ptrs,
                                    float const* const* weight_ptrs,
                                    int offset,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uvw_out[i] = uvw_ptrs[i][offset];
  Vo_out[i] = Vo_ptrs[i][offset];
  weight_out[i] = weight_ptrs[i][offset];
}

__global__ void gatherChunkAtOffsets(double3* uvw_out,
                                     cufftComplex* Vo_out,
                                     float* weight_out,
                                     double3 const* const* uvw_ptrs,
                                     cufftComplex const* const* Vo_ptrs,
                                     float const* const* weight_ptrs,
                                     const int* slots,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int o = slots[i];
  uvw_out[i] = uvw_ptrs[i][o];
  Vo_out[i] = Vo_ptrs[i][o];
  weight_out[i] = weight_ptrs[i][o];
}

// Add gradient contribution to multi-plane gradient array
__global__ void AddToDPhi(float* dphi, float* dgi, long N, long M, int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N * M * index + N * i + j] += dgi[N * i + j];
}

