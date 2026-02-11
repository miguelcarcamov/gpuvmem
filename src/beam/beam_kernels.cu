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

#include "beam/beam_kernels.cuh"
#include "utils/physics_utils.cuh"
#include "utils/complexOps.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Device function pointer array for beam types
typedef float (*FnPtr)(float, float, float, float);
__device__ FnPtr beam_maps[2] = {AiryDiskBeam, GaussianBeam};

__device__ float AiryDiskBeam(float distance,
                              float lambda,
                              float antenna_diameter,
                              float pb_factor) {
  float atten = 1.0f;
  if (distance != 0.0) {
    // Airy disk formula: [2*J1(π*D*θ/λ) / (π*D*θ/λ)]²
    // where D is diameter, θ is angle, λ is wavelength
    // pb_factor scales where the first null occurs (standard is RZ ≈ 1.22)
    // Scale the argument so first null occurs at pb_factor * λ / D
    float bessel_arg =
        PI * distance * antenna_diameter / lambda * (RZ / pb_factor);
    float bessel_func = j1f(bessel_arg);
    atten = 4.0f * (bessel_func / bessel_arg) * (bessel_func / bessel_arg);
  }

  return atten;
}

__device__ float GaussianBeam(float distance,
                              float lambda,
                              float antenna_diameter,
                              float pb_factor) {
  float fwhm = pb_factor * lambda / antenna_diameter;
  float c = 4.0f * logf(2.0f);
  float r = distance / fwhm;
  float atten = expf(-c * r * r);
  return atten;
}

__device__ float attenuation(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float freq,
                             float xobs,
                             float yobs,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten_result, atten;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG_D;
  float y = (i - y0) * DELTAY * RPDEG_D;

  float arc = distance(x, y, 0.0, 0.0);
  float lambda = freq_to_wavelength(freq);
  atten = (*beam_maps[primary_beam])(arc, lambda, antenna_diameter, pb_factor);
  if (arc <= pb_cutoff) {
    atten_result = atten;
  } else {
    atten_result = 0.0f;
  }

  return atten_result;
}

__device__ cufftComplex
WKernel(double w, float xobs, float yobs, double DELTAX, double DELTAY) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  cufftComplex Wk;
  float cosk, sink;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG_D;
  float y = (i - y0) * DELTAY * RPDEG_D;
  float z = sqrtf(1 - x * x - y * y) - 1;
  float arg = 2.0f * w * z;

#if (__CUDA_ARCH__ >= 300)
  sincospif(arg, &sink, &cosk);
#else
  cosk = cospif(arg);
  sink = sinpif(arg);
#endif

  Wk = make_cuFloatComplex(cosk, -sink);
  return Wk;
}

__global__ void distance_image(float* distance_image,
                               float xobs,
                               float yobs,
                               float dist_arcsec,
                               double DELTAX,
                               double DELTAY,
                               long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * 3600.0;
  float y = (i - y0) * DELTAY * 3600.0;

  float dist = distance(x, y, 0.0, 0.0);
  distance_image[N * i + j] = 1.0f;

  if (dist < dist_arcsec)
    distance_image[N * i + j] = 0.0f;
}

__global__ void total_attenuation(float* total_atten,
                                  float antenna_diameter,
                                  float pb_factor,
                                  float pb_cutoff,
                                  float freq,
                                  float xobs,
                                  float yobs,
                                  double DELTAX,
                                  double DELTAY,
                                  long N,
                                  int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float attenPerFreq = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq,
                                   xobs, yobs, DELTAX, DELTAY, primary_beam);
  total_atten[N * i + j] += attenPerFreq;
}

__global__ void weight_image(float* weight_image, float* total_atten, long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = total_atten[N * i + j];
  weight_image[N * i + j] += atten * atten;
}

__global__ void noise_image(float* noise_image,
                            float* weight_image,
                            float max_weight,
                            float noise_jypix,
                            long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float noise_squared = noise_jypix * noise_jypix;
  float normalized_weight =
      (weight_image[N * i + j] / max_weight) / noise_squared;
  float noiseval = sqrtf(1.0f / normalized_weight);
  noise_image[N * i + j] = noiseval;
}

__global__ void apply_beam2I(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             cufftComplex* image,
                             long N,
                             float xobs,
                             float yobs,
                             float fg_scale,
                             float freq,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs,
                            yobs, DELTAX, DELTAY, primary_beam);

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * atten * fg_scale, 0.0f);
}

__global__ void apply_beam2I(float antenna_diameter,
                             float pb_factor,
                             float pb_cutoff,
                             float* gcf,
                             cufftComplex* image,
                             long N,
                             float xobs,
                             float yobs,
                             float freq,
                             double DELTAX,
                             double DELTAY,
                             int primary_beam) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs,
                            yobs, DELTAX, DELTAY, primary_beam);

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * gcf[N * i + j] * atten, 0.0f);
}

__global__ void apply_GCF(cufftComplex* __restrict__ image,
                          const float* __restrict__ gcf,
                          long N) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  image[N * i + j] =
      make_cuFloatComplex(image[N * i + j].x * gcf[N * i + j], 0.0f);
}

// calculateInu kernel moved from functions.cu
__global__ void calculateInu(cufftComplex* I_nu,
                             float* I,
                             float nu,
                             float nu_0,
                             float MINPIX,
                             float eta,
                             long N,
                             long M) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  float I_nu_0, alpha, nudiv_pow_alpha, nudiv;

  // Safety check: avoid division by zero
  if (nu_0 <= 0.0f || nu <= 0.0f) {
    I_nu[N * i + j].x = 0.0f;
    I_nu[N * i + j].y = 0.0f;
    return;
  }

  // Convention: nu/nu_0 (NOT nu_0/nu). Model I_ν = I_ν₀ × (ν/ν₀)^α
  nudiv = nu / nu_0;
  
  // Safety check: nudiv must be positive for powf
  if (nudiv <= 0.0f) {
    I_nu[N * i + j].x = 0.0f;
    I_nu[N * i + j].y = 0.0f;
    return;
  }

  I_nu_0 = I[N * i + j];
  alpha = I[M * N + N * i + j];
  nudiv_pow_alpha = powf(nudiv, alpha);

  // Base spectral model: I_ν_base = I_ν₀ × (ν/ν₀)^α
  // (attenuation and fg_scale applied later in apply_beam2I)
  I_nu[N * i + j].x = I_nu_0 * nudiv_pow_alpha;

  if (I_nu[N * i + j].x < -1.0f * eta * MINPIX) {
    I_nu[N * i + j].x = -1.0f * eta * MINPIX;
  }

  I_nu[N * i + j].y = 0.0f;
}

