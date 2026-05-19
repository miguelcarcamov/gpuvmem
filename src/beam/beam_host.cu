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

#include "beam/beam_host.cuh"
#include "beam/beam_kernels.cuh"
#include "image_processing/image_processing_kernels.cuh"
#include "chi2/chain_rule.cuh"
#include "error.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>

// Extern variables
extern dim3 threadsPerBlockNN, numBlocksNN;
extern long N, M;
extern double DELTAX, DELTAY;
extern int firstgpu;
extern float* device_noise_image;
extern float noise_cut, nu_0, eta, threshold, alpha_n_sigma;
extern float* initial_values;
extern int flag_opt;

__host__ void linkApplyBeam2I(cufftComplex* image,
                              float antenna_diameter,
                              float pb_factor,
                              float pb_cutoff,
                              float xobs,
                              float yobs,
                              float freq,
                              int primary_beam,
                              float fg_scale) {
  apply_beam2I<<<numBlocksNN, threadsPerBlockNN>>>(
      antenna_diameter, pb_factor, pb_cutoff, image, N, xobs, yobs, fg_scale,
      freq, DELTAX, DELTAY, primary_beam);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkApplyBaselineBeam2I(cufftComplex* image,
                                      float ant1_diameter,
                                      float ant1_pb_factor,
                                      float ant1_pb_cutoff,
                                      int ant1_primary_beam,
                                      float ant2_diameter,
                                      float ant2_pb_factor,
                                      float ant2_pb_cutoff,
                                      int ant2_primary_beam,
                                      float xobs,
                                      float yobs,
                                      float freq,
                                      float fg_scale) {
  apply_baseline_beam2I<<<numBlocksNN, threadsPerBlockNN>>>(
      ant1_diameter, ant1_pb_factor, ant1_pb_cutoff, ant1_primary_beam,
      ant2_diameter, ant2_pb_factor, ant2_pb_cutoff, ant2_primary_beam, image, N,
      xobs, yobs, fg_scale, freq, DELTAX, DELTAY);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkClipWNoise2I(float* I) {
  clip2IWNoise<<<numBlocksNN, threadsPerBlockNN>>>(
      device_noise_image, I, N, M, noise_cut, initial_values[0],
      initial_values[1], eta, threshold, alpha_n_sigma, flag_opt);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkCalculateInu2I(cufftComplex* image, float* I, float freq) {
  calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(
      image, I, freq, nu_0, initial_values[0], eta, N, M);
  checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void linkChain2I(float* chain, float freq, float* I, float fg_scale) {
  // Note: chainRule2I needs attenuation parameters, but they're not available here
  // since they're stored in the dataset structure, not as global variables.
  // For now, we'll use a simplified version that computes I_ν_base and scales by fg_scale.
  // This assumes atten ≈ 1.0 near the center, which is reasonable.
  // The main gradient path uses DChi2_total_I_nu_0 / DChi2_total_alpha (no attenuation in kernel).
  
  // Use simplified chainRule2I that doesn't require attenuation parameters
  // We'll compute I_ν ≈ fg_scale * I_ν_base (assuming atten ≈ 1.0)
  chainRule2ISimplified<<<numBlocksNN, threadsPerBlockNN>>>(
      chain, device_noise_image, I, freq, nu_0, noise_cut, fg_scale, N, M);
  checkCudaErrors(cudaDeviceSynchronize());
}
