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
#include "errors/errors_kernels.cuh"
#include "beam/beam_kernels.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void I_nu_0_Noise(float* noise_I,
                             float* images,
                             float* noise,
                             float noise_cut,
                             float nu,
                             float nu_0,
                             float* w,
                             float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             float sum_weights,
                             float fg_scale,
                             long N,
                             long M,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, nudiv, nudiv_pow_alpha, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  I_nu_0 = images[N * i + j];
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, 2.0f * alpha);

  // Variance for I_ν₀: σ²(I_ν₀) ∝ (∂I_ν/∂I_ν₀)² * σ²(visibilities)
  // Where: ∂I_ν/∂I_ν₀ = fg_scale × atten × (ν/ν₀)^α
  // Per-channel contribution to inverse-variance: 1/σ²_c = (∂I_ν/∂I_ν₀)² / σ²(I_ν,c)
  //   = fg_scale² · atten² · (ν/ν₀)^(2α) · sum_weights
  // We accumulate inverse-variance contributions: 1/σ²_total = Σ_c (1/σ²_c)
  // Then convert to variance: σ²_total = 1 / (Σ_c 1/σ²_c)
  if (noise[N * i + j] < noise_cut) {
    float fg_scale_sq = fg_scale * fg_scale;
    // Accumulate inverse-variance contribution for this channel
    noise_I[N * i + j] +=
        fg_scale_sq * atten * atten * sum_weights * nudiv_pow_alpha;
  }
  // else: do nothing — keep accumulation from other channels
}

// σ(alpha) from the linear flux model I_ν = I_ν0·(ν/ν0)^α (no log fit).
// Fisher information for α in flux space: I_α = Σ_c (∂I_ν/∂α)²/σ²(I_ν,c),
// with ∂I_ν/∂α = I_ν·ln(ν/ν0). Per-channel flux variance (same convention as
// I_nu_0_Noise): σ²(I_ν,c) = 1/(fg_scale²·atten²·sum_weights) [since
// σ²(I_ν)=(ν/ν0)^(2α)·σ²(I_ν0) and inv_var(I_ν0)_c =
// fg_scale²·atten²·sum_weights·(ν/ν0)^(2α)]. So inv_var(α) = 1/σ²(α) = Σ_c
// (∂I_ν/∂α)²·σ⁻²(I_ν,c)
//              = Σ_c I_ν²·ln²(ν/ν0)·fg_scale²·atten²·sum_weights_c.
// When ν=ν0, ln(ν/ν0)=0 so that channel contributes 0. α and σ(α) are
// unitless; image and sum_weights must be in consistent units so σ(α) ~ 0.1–2.
__global__ void alpha_Noise(float* noise_I,
                            float* images,
                            float nu,
                            float nu_0,
                            float* noise,
                            float noise_cut,
                            double DELTAX,
                            double DELTAY,
                            float xobs,
                            float yobs,
                            float antenna_diameter,
                            float pb_factor,
                            float pb_cutoff,
                            float sum_weights,
                            float fg_scale,
                            long N,
                            long M,
                            int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  I_nu_0 = images[N * i + j];
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu = I_nu_0 * nudiv_pow_alpha;
  log_nu = logf(nudiv);

  // Variance for α: σ²(α) from Fisher information
  // Per-channel contribution to inverse-variance: 1/σ²_c = (∂I_ν/∂α)² / σ²(I_ν,c)
  //   = fg_scale² · atten² · I_ν² · log²(ν/ν₀) · sum_weights
  // Where: ∂I_ν/∂α = fg_scale × atten × I_ν_base × log(ν/ν₀) = fg_scale × atten × I_ν × log(ν/ν₀)
  // We accumulate inverse-variance contributions: 1/σ²_total = Σ_c (1/σ²_c)
  // Then convert to variance: σ²_total = 1 / (Σ_c 1/σ²_c)
  // When ν = ν₀, log(ν/ν₀) = 0 so this channel contributes 0.
  float inv_var_alpha_c = fg_scale * fg_scale * atten * atten * I_nu * I_nu *
                          sum_weights * log_nu * log_nu;

  if (noise[N * i + j] < noise_cut) {
    // Accumulate inverse-variance contribution for this channel
    noise_I[N * M + N * i + j] += inv_var_alpha_c;
  }
  // else: do nothing — keep accumulation from other channels
}

// Compute covariance term for I_nu_0 and alpha noise correlation
// This accounts for the correlation between fitted parameters
// Full error propagation: σ²(I_nu) = (∂I_nu/∂I_nu_0)²σ²(I_nu_0) +
// (∂I_nu/∂alpha)²σ²(alpha)
//                         + 2(∂I_nu/∂I_nu_0)(∂I_nu/∂alpha)Cov(I_nu_0, alpha)
// The covariance term: Cov(I_nu_0, alpha) ∝ (∂I_nu/∂I_nu_0) × (∂I_nu/∂alpha) ×
// σ²(visibilities) Where: ∂I_nu/∂I_nu_0 = atten * (nu/nu_0)^alpha
//        ∂I_nu/∂alpha = I_nu * log(nu/nu_0)
// So Cov ∝ (ν/ν₀)^α · I_ν · log(ν/ν₀). When ν = ν₀, log(ν/ν₀)=0 and the
// covariance contribution from that channel is zero (same as σ(alpha)).
// Note: The covariance is stored at index 2 (after I_nu_0 variance at 0, alpha
// variance at 1) Can be used to compute:
// - Correlation coefficient: ρ = Cov(I_nu_0, alpha) / (σ(I_nu_0) * σ(alpha))
// - Total uncertainty in I_nu when both parameters vary
__global__ void covariance_Noise(float* noise_cov,
                                 float* images,
                                 float nu,
                                 float nu_0,
                                 float* noise,
                                 float noise_cut,
                                 double DELTAX,
                                 double DELTAY,
                                 float xobs,
                                 float yobs,
                                 float antenna_diameter,
                                 float pb_factor,
                                 float pb_cutoff,
                                 float sum_weights,
                                 float fg_scale,
                                 long N,
                                 long M,
                                 int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, atten;

  atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs,
                      DELTAX, DELTAY, primary_beam);

  nudiv = nu / nu_0;
  I_nu_0 = images[N * i + j];
  alpha = images[N * M + N * i + j];
  nudiv_pow_alpha = powf(nudiv, alpha);

  I_nu = I_nu_0 * nudiv_pow_alpha;
  log_nu = logf(nudiv);

  float fg_scale_sq = fg_scale * fg_scale;

  // Fisher off-diagonal H[0,1] for covariance (linear model I_ν = I_ν0·(ν/ν0)^α).
  // Per-channel: H[0,1]_c = (∂I_ν/∂I_ν₀)(∂I_ν/∂α)/σ²(I_ν); we accumulate H[0,1].
  // noise_reduction inverts the 2x2 Hessian to get Cov(I_nu_0, alpha) = -H[0,1]/det(H).
  // When ν=ν₀, ln(ν/ν₀)=0 so this channel contributes 0 (same as σ(α)).
  float cov_c = fg_scale_sq * atten * atten * nudiv_pow_alpha * I_nu * log_nu *
                sum_weights;

  if (noise[N * i + j] < noise_cut) {
    noise_cov[2 * M * N + N * i + j] += cov_c;
  }
  // else: do nothing — keep accumulation from other channels
}

__global__ void noise_reduction(float* noise_I, long N, long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Error array holds (before this kernel): H[0,0], H[1,1], H[0,1] (Fisher).
  // Add small diagonal prior to stabilize inversion in low-S/N pixels.
  const float prior_lam = 1.0e-12f;
  float H00 = noise_I[N * i + j] + prior_lam;
  float H11 = noise_I[N * M + N * i + j] + prior_lam;
  float H01 = noise_I[2 * M * N + N * i + j];

  // Full 2x2 inverse: C = H^{-1}. Marginal variances: C[0,0]=H11/det, C[1,1]=H00/det.
  float det = H00 * H11 - H01 * H01;
  const float det_eps = 1.0e-12f * (H00 * H11 + 1.0e-30f);
  float sigma_I_nu_0 = 0.0f;
  float sigma_alpha = 0.0f;
  float cov01 = 0.0f;
  float rho = 0.0f;

  if (fabsf(det) > det_eps) {
    float C00 = H11 / det;   // Var(I_nu_0)
    float C11 = H00 / det;   // Var(alpha)
    cov01 = -H01 / det;      // Cov(I_nu_0, alpha)
    sigma_I_nu_0 = sqrtf(fmaxf(C00, 0.0f));
    float sigma_alpha_raw = sqrtf(fmaxf(C11, 0.0f));
    const float sigma_alpha_max = 10000.0f;
    sigma_alpha = fminf(sigma_alpha_raw, sigma_alpha_max);
    // Correlation coefficient ρ = Cov / (σ(I_nu_0) * σ(alpha)), clamped to [-1,1]
    if (sigma_I_nu_0 > 1.0e-30f && sigma_alpha > 1.0e-30f) {
      rho = cov01 / (sigma_I_nu_0 * sigma_alpha);
      rho = fmaxf(-1.0f, fminf(1.0f, rho));
    }
  }

  // Output: index 0 = σ(I_nu_0), 1 = σ(alpha), 2 = Cov(I_nu_0, alpha), 3 = ρ.
  // Cov has same units as I_nu_0 (α is unitless); ρ = Cov/(σ_I·σ_α) in [-1,1].
  // High |ρ| (e.g. ~0.9) means I_nu_0 and α are degenerate (trade off along ridge).
  noise_I[N * i + j] = sigma_I_nu_0;
  noise_I[N * M + N * i + j] = sigma_alpha;
  noise_I[2 * M * N + N * i + j] = cov01;
  noise_I[3 * M * N + N * i + j] = rho;
}

__global__ void stokes_Noise(float* noise_I,
                             int pol_plane,
                             float nu,
                             float* noise,
                             float noise_cut,
                             float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             float sum_weights,
                             float fg_scale,
                             long N,
                             long M,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= M || j >= N) return;
  const long idx = N * i + j;
  float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs,
                            yobs, DELTAX, DELTAY, primary_beam);
  if (noise[idx] < noise_cut) {
    float fg_scale_sq = fg_scale * fg_scale;
    noise_I[(long)pol_plane * M * N + idx] +=
        fg_scale_sq * atten * atten * sum_weights;
  }
}

__global__ void stokes_noise_reduction(float* noise_I,
                                       int nplanes,
                                       long N,
                                       long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= M || j >= N) return;
  const float prior_lam = 1.0e-12f;
  for (int p = 0; p < nplanes; p++) {
    long idx = (long)p * M * N + N * i + j;
    float inv_var = noise_I[idx] + prior_lam;
    noise_I[idx] = (inv_var > 1.0e-30f) ? sqrtf(1.0f / inv_var) : 0.0f;
  }
}
