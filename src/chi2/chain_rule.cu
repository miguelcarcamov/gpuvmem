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

#include "chi2/chain_rule.cuh"
#include "beam/beam_kernels.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void DChi2_total_alpha(float* noise,
                                  float* dchi2_total,
                                  float* dchi2,
                                  float* I,
                                  float nu,
                                  float nu_0,
                                  float noise_cut,
                                  long N,
                                  long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, I_nu_base;
  
  // Safety check: avoid division by zero
  if (nu_0 <= 0.0f || nu <= 0.0f) {
    return;
  }
  
  // Convention: nu/nu_0 (NOT nu_0/nu). So (ν/ν₀)^α; log(ν/ν₀) correct for ∂I_ν/∂α
  float nudiv = nu / nu_0;
  
  // Safety check: nudiv must be positive for logf and powf
  if (nudiv <= 0.0f) {
    return;
  }

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];
  
  // Compute I_ν_base = I_ν₀ · (ν/ν₀)^α
  I_nu_base = I_nu_0 * powf(nudiv, alpha);
  
  // Chain rule for gradient with respect to α:
  // Forward model: I_ν = fg_scale × atten × I_ν₀ × (ν/ν₀)^α
  // ∂I_ν/∂α = fg_scale × atten × I_ν₀ × (ν/ν₀)^α × log(ν/ν₀) = fg_scale × atten × I_ν_base × log(ν/ν₀)
  // ∂χ²/∂α = ∂χ²/∂I_ν × ∂I_ν/∂α = dchi2 × I_ν_base × log(ν/ν₀). dchi2 is +∂χ²/∂I_ν (see DChi2),
  // so we accumulate +∂χ²/∂α; optimizer uses descent direction −∂χ²/∂α.
  float log_nudiv = logf(nudiv);
  dalpha = I_nu_base * log_nudiv;

  // += accumulates over frequency channels (caller zeros dchi2_total before channel loop)
  if (noise[N * i + j] < noise_cut && isfinite(dalpha)) {
    dchi2_total[N * M + N * i + j] += dchi2[N * i + j] * dalpha;
  }
}

__global__ void DChi2_total_I_nu_0(float* noise,
                                   float* dchi2_total,
                                   float* dchi2,
                                   float* I,
                                   float nu,
                                   float nu_0,
                                   float noise_cut,
                                   long N,
                                   long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dI_nu_0, I_nu_base;
  
  // Safety check: avoid division by zero
  if (nu_0 <= 0.0f || nu <= 0.0f) {
    return;
  }
  
  float nudiv = nu / nu_0;
  
  // Safety check: nudiv must be positive for powf
  if (nudiv <= 0.0f) {
    return;
  }

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];
  
  // Compute I_ν_base = I_ν₀ · (ν/ν₀)^α
  I_nu_base = I_nu_0 * powf(nudiv, alpha);
  
  // Chain rule for gradient with respect to I_ν₀:
  // Forward model: I_ν = fg_scale × atten × I_ν₀ × (ν/ν₀)^α
  // ∂I_ν/∂I_ν₀ = fg_scale × atten × (ν/ν₀)^α
  // ∂χ²/∂I_ν₀ = ∂χ²/∂I_ν × ∂I_ν/∂I_ν₀ = dchi2 × fg_scale × atten × (ν/ν₀)^α
  // Since dchi2 (from DChi2 kernel) already includes fg_scale × atten (see DChi2 line 4287),
  // we have: ∂χ²/∂I_ν₀ = dchi2 × (ν/ν₀)^α
  dI_nu_0 = powf(nudiv, alpha);

  // += accumulates over frequency channels (caller zeros dchi2_total before loop)
  if (noise[N * i + j] < noise_cut && isfinite(dI_nu_0))
    dchi2_total[N * i + j] += dchi2[N * i + j] * dI_nu_0;
}

// Simplified version of chainRule2I that doesn't require attenuation parameters
// Used when attenuation parameters aren't available (e.g., in linkChain2I)
// Assumes atten ≈ 1.0, so I_ν ≈ fg_scale * I_ν_base
__global__ void chainRule2ISimplified(float* chain,
                                      float* noise,
                                      float* I,
                                      float nu,
                                      float nu_0,
                                      float noise_cut,
                                      float fg_scale,
                                      long N,
                                      long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, dI_nu_0, I_nu_base;
  
  // Safety check: avoid division by zero
  if (nu_0 <= 0.0f || nu <= 0.0f) {
    chain[N * i + j] = 0.0f;
    chain[N * M + N * i + j] = 0.0f;
    return;
  }
  
  float nudiv = nu / nu_0;
  
  // Safety check: nudiv must be positive for logf and powf
  if (nudiv <= 0.0f) {
    chain[N * i + j] = 0.0f;
    chain[N * M + N * i + j] = 0.0f;
    return;
  }

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];
  
  // Compute I_ν_base = I_ν₀ · (ν/ν₀)^α
  I_nu_base = I_nu_0 * powf(nudiv, alpha);
  
  // Chain rule derivatives for forward model: I_ν = fg_scale × atten × I_ν₀ × (ν/ν₀)^α
  // ∂I_ν/∂I_ν₀ = fg_scale × atten × (ν/ν₀)^α
  // ∂I_ν/∂α = fg_scale × atten × I_ν₀ × (ν/ν₀)^α × log(ν/ν₀) = fg_scale × atten × I_ν_base × log(ν/ν₀)
  //
  // Since dchi2 (from DChi2 kernel) already includes fg_scale × atten (see DChi2 line 4287),
  // we store the chain rule factors without fg_scale × atten:
  // chain[I_ν₀] = (ν/ν₀)^α
  // chain[α] = I_ν_base × log(ν/ν₀)
  //
  // Final gradients:
  //   ∂χ²/∂I_ν₀ = dchi2 × chain[I_ν₀] = dchi2 × (ν/ν₀)^α
  //   ∂χ²/∂α = dchi2 × chain[α] = dchi2 × I_ν_base × log(ν/ν₀)
  dI_nu_0 = powf(nudiv, alpha);
  float log_nudiv = logf(nudiv);
  dalpha = I_nu_base * log_nudiv;

  // Safety check: ensure values are finite before storing
  chain[N * i + j] = isfinite(dI_nu_0) ? dI_nu_0 : 0.0f;
  chain[N * M + N * i + j] = isfinite(dalpha) ? dalpha : 0.0f;
}

// Full version with attenuation parameters (used in main gradient path)
__global__ void chainRule2I(float* chain,
                            float* noise,
                            float* I,
                            float nu,
                            float nu_0,
                            float noise_cut,
                            float fg_scale,
                            float ref_xobs,
                            float ref_yobs,
                            double DELTAX,
                            double DELTAY,
                            float antenna_diameter,
                            float pb_factor,
                            float pb_cutoff,
                            int primary_beam,
                            long N,
                            long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, dalpha, dI_nu_0, I_nu_base;
  
  // Safety check: avoid division by zero
  if (nu_0 <= 0.0f || nu <= 0.0f) {
    chain[N * i + j] = 0.0f;
    chain[N * M + N * i + j] = 0.0f;
    return;
  }
  
  float nudiv = nu / nu_0;
  
  // Safety check: nudiv must be positive for logf and powf
  if (nudiv <= 0.0f) {
    chain[N * i + j] = 0.0f;
    chain[N * M + N * i + j] = 0.0f;
    return;
  }

  I_nu_0 = I[N * i + j];
  alpha = I[N * M + N * i + j];
  
  // Compute I_ν_base = I_ν₀ · (ν/ν₀)^α
  I_nu_base = I_nu_0 * powf(nudiv, alpha);
  
  // Chain rule derivatives for forward model: I_ν = fg_scale × atten × I_ν₀ × (ν/ν₀)^α
  // ∂I_ν/∂I_ν₀ = fg_scale × atten × (ν/ν₀)^α
  // ∂I_ν/∂α = fg_scale × atten × I_ν₀ × (ν/ν₀)^α × log(ν/ν₀) = fg_scale × atten × I_ν_base × log(ν/ν₀)
  //
  // Since dchi2 (from DChi2 kernel) already includes fg_scale × atten (see DChi2 line 4287),
  // we store the chain rule factors without fg_scale × atten:
  // chain[I_ν₀] = (ν/ν₀)^α
  // chain[α] = I_ν_base × log(ν/ν₀)
  //
  // Final gradients:
  //   ∂χ²/∂I_ν₀ = dchi2 × chain[I_ν₀] = dchi2 × (ν/ν₀)^α
  //   ∂χ²/∂α = dchi2 × chain[α] = dchi2 × I_ν_base × log(ν/ν₀)
  dI_nu_0 = powf(nudiv, alpha);
  float log_nudiv = logf(nudiv);
  dalpha = I_nu_base * log_nudiv;

  // Safety check: ensure values are finite before storing
  chain[N * i + j] = isfinite(dI_nu_0) ? dI_nu_0 : 0.0f;
  chain[N * M + N * i + j] = isfinite(dalpha) ? dalpha : 0.0f;
}

__global__ void DChi2_2I(float* noise,
                         float* chain,
                         float* I,
                         float* dchi2,
                         float* dchi2_total,
                         float threshold,
                         float alpha_n_sigma,
                         float noise_cut,
                         int image,
                         long N,
                         long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Alpha masking removed - always accumulate gradient regardless of I_nu_0 threshold
  // Slice 0 = I_nu_0 gradient, slice 1 = alpha gradient (layout: dchi2_total[0..N*M-1], dchi2_total[N*M..2*N*M-1])
  if (noise[N * i + j] < noise_cut && image) {
    dchi2_total[N * M + N * i + j] += dchi2[N * i + j] * chain[N * M + N * i + j];
  } else if (noise[N * i + j] < noise_cut) {
    dchi2_total[N * i + j] += dchi2[N * i + j] * chain[N * i + j];
  }
}
