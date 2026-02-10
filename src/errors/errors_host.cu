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
#include "errors/errors_host.cuh"
#include "errors/errors_kernels.cuh"
#include "chi2/chi2_kernels.cuh"
#include "reduction/reduction_host.cuh"
#include "utils/cuda_utils.cuh"
#include "framework.cuh"
#include <cuda_runtime.h>
#include <omp.h>

extern long M, N;
extern int num_gpus, firstgpu;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern float noise_cut, nu_0;
extern float* device_noise_image;
extern double DELTAX, DELTAY;
extern MSDataset* datasets;
extern varsPerGPU* vars_gpu;
extern int nMeasurementSets;

__host__ void calculateErrors(Image* image, float fg_scale) {
  float* errors = image->getErrorImage();

  cudaSetDevice(firstgpu);

  // Allocate error array: [σ(I_nu_0), σ(alpha), Cov, ρ] — 4 maps for 2 images.
  // Fisher H[i,j] = Σ (∂I_ν/∂θ_i)(∂I_ν/∂θ_j)/σ²; C = H^{-1} gives marginal variances
  // and Cov. Cov has same units as I_nu_0; ρ = Cov/(σ_I·σ_α) in [-1,1] (high |ρ| = degeneracy).
  int error_image_count = image->getImageCount() + 2;  // +1 covariance, +1 correlation ρ
  checkCudaErrors(
      cudaMalloc((void**)&errors, sizeof(float) * M * N * error_image_count));
  checkCudaErrors(
      cudaMemset(errors, 0, sizeof(float) * M * N * error_image_count));
  float sum_weights;
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      // ordered: accumulate Fisher terms in iteration order (i then s) so that
      // floating-point += order is deterministic and error maps are reproducible.
#pragma omp parallel for private(sum_weights) num_threads(num_gpus)     schedule(static, 1) ordered
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreq[i] > 0) {
              sum_weights = deviceReduce<float>(
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);

#pragma omp ordered
              {
                // Compute variance for I_nu_0 (stored at index 0)
                I_nu_0_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), device_noise_image, noise_cut,
                    datasets[d].fields[f].nu[i], nu_0,
                    datasets[d].fields[f].device_visibilities[i][s].weight,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix, DELTAX, DELTAY,
                    sum_weights, fg_scale, N, M,
                    datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());

                // Compute variance for alpha (stored at index 1)
                alpha_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), datasets[d].fields[f].nu[i],
                    nu_0, device_noise_image, noise_cut, DELTAX, DELTAY,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff, sum_weights, fg_scale, N,
                    M, datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());

                // Compute covariance between I_nu_0 and alpha (stored at index
                // 2)
                covariance_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, image->getImage(), datasets[d].fields[f].nu[i],
                    nu_0, device_noise_image, noise_cut, DELTAX, DELTAY,
                    datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff, sum_weights, fg_scale, N,
                    M, datasets[d].antennas[0].primary_beam);
                checkCudaErrors(cudaDeviceSynchronize());
              }
            }
          }
        }
      }
    }
  }

  noise_reduction<<<numBlocksNN, threadsPerBlockNN>>>(errors, N, M);
  checkCudaErrors(cudaDeviceSynchronize());

  // Error array holds σ(I_nu_0), σ(alpha), Cov, ρ. fg_scale applied when saving
  // slices 0 and 2 (Io classes).
  image->setErrorImage(errors);
}

__host__ void precomputeNeff(bool normalize) {
  if (!normalize) {
    return;
  }

  cudaSetDevice(firstgpu);

  // Pre-compute N_eff for all datasets, fields, frequencies, and Stokes
  // parameters
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      // Initialize N_eff storage
      datasets[d].fields[f].N_eff_perFreqPerStoke.resize(
          datasets[d].data.total_frequencies,
          std::vector<float>(datasets[d].data.nstokes, 0.0f));

#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              // Compute sum of weights
              float sum_weights = deviceReduce<float>(
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);

              // Compute sum of squared weights (reuse device_chi2 as temp
              // storage)
              weightsSquaredVector<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  vars_gpu[gpu_idx].device_chi2,  // Temp storage
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              float sum_weights_squared = deviceReduce<float>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);

              // Calculate effective number of samples
              if (sum_weights_squared > 0.0f && sum_weights > 0.0f) {
                datasets[d].fields[f].N_eff_perFreqPerStoke[i][s] =
                    (sum_weights * sum_weights) / sum_weights_squared;
              } else {
                // Fallback to numVisibilities if calculation fails
                datasets[d].fields[f].N_eff_perFreqPerStoke[i][s] =
                    (float)datasets[d]
                        .fields[f]
                        .numVisibilitiesPerFreqPerStoke[i][s];
              }
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);
}
