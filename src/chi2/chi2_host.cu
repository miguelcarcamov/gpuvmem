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

#include "chi2/chi2_host.cuh"
#include "chi2/chi2_kernels.cuh"
#include "chi2/chain_rule.cuh"
#include "gridding/gridding_kernels.cuh"
#include "visibility/visibility_kernels.cuh"
#include "reduction/reduction_host.cuh"
#include "framework.cuh"
#include "kernels/pillBox2D.cuh"
#include "error.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>

// Extern variables
extern varsPerGPU* vars_gpu;
extern int nMeasurementSets, num_gpus, firstgpu, max_number_vis, flag_opt;
extern long M, N;
extern double deltau, deltav, DELTAX, DELTAY;
extern MSDataset* datasets;
extern float noise_cut, nu_0;
extern float* device_noise_image;
extern dim3 threadsPerBlockNN, numBlocksNN;

// Include measurement operator header
#include "measurement_operator/measurement_operator_host.cuh"

__host__ float simulate(float* I, VirtualImageProcessor* ip, float fg_scale) {
  // simulate is equivalent to chi2 with normalize=true
  return chi2(I, ip, true, fg_scale);
}

__host__ float chi2(float* I,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  bool fft_shift = true;  // fft_shift=false (DC at corner 0,0)
  cudaSetDevice(firstgpu);

  float reduced_chi2 = 0.0f;

  // Create static 1x1 PillBox kernel for degridding when gridding is enabled
  static PillBox2D* degrid_kernel = NULL;
  static bool degrid_kernel_initialized = false;

  CKernel* ckernel = ip->getCKernel();
  bool use_gridding = (ckernel != NULL && ckernel->getGPUKernel() != NULL);

  if (use_gridding && !degrid_kernel_initialized) {
    degrid_kernel = new PillBox2D(1, 1);
    degrid_kernel->setGPUID(firstgpu);
    degrid_kernel->setSigmas(fabs(deltau), fabs(deltav));
    degrid_kernel->buildKernel();
    degrid_kernel_initialized = true;
  }

  ip->clipWNoise(I);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus) \
    reduction(+ : reduced_chi2)
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        float result = 0.0;
        unsigned int j = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_idx = i % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);
        int gpu_id = -1;
        cudaGetDevice(&gpu_id);

        // Compute visibility grid from image using common pipeline
        // Use fft_shift=false (DC at corner 0,0) for testing
        computeImageToVisibilityGrid(
            I, ip, vars_gpu, gpu_idx, M, N, datasets[d].fields[f].nu[i],
            datasets[d].fields[f].ref_xobs_pix,
            datasets[d].fields[f].ref_yobs_pix,
            datasets[d].fields[f].phs_xobs_pix,
            datasets[d].fields[f].phs_yobs_pix,
            datasets[d].antennas[0].antenna_diameter,
            datasets[d].antennas[0].pb_factor,
            datasets[d].antennas[0].pb_cutoff,
            datasets[d].antennas[0].primary_beam, fg_scale, ip->getCKernel(),
            fft_shift);  // fft_shift=false (DC at corner 0,0)

        // Texture memory removed - using regular global memory with __ldg()
        // instead

        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
                                         sizeof(float) * max_number_vis));

              if (use_gridding) {
                degriddingGPU<<<
                    datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                    datasets[d]
                        .fields[f]
                        .device_visibilities[i][s]
                        .threadsPerBlockUV>>>(
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].Vm,
                    vars_gpu[gpu_idx].device_V, degrid_kernel->getGPUKernel(),
                    deltau, deltav,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    M, N, degrid_kernel->getm(), degrid_kernel->getn(),
                    degrid_kernel->getSupportX(), degrid_kernel->getSupportY());
                checkCudaErrors(cudaDeviceSynchronize());
              } else {
                // Gridding disabled: use bilinear interpolation
                bilinearInterpolateVisibility<<<
                    datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                    datasets[d]
                        .fields[f]
                        .device_visibilities[i][s]
                        .threadsPerBlockUV>>>(
                    datasets[d].fields[f].device_visibilities[i][s].Vm,
                    vars_gpu[gpu_idx].device_V,
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].weight,
                    deltau, deltav,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    M, N, fft_shift);  // dc_at_center=false (DC at corner 0,0,
                                       // matching fft_shift=false)
                checkCudaErrors(cudaDeviceSynchronize());
              }

              // RESIDUAL CALCULATION
              residual<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  datasets[d].fields[f].device_visibilities[i][s].Vr,
                  datasets[d].fields[f].device_visibilities[i][s].Vm,
                  datasets[d].fields[f].device_visibilities[i][s].Vo,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              // Use pre-computed effective number of samples (calculated once
              // before optimization)
              float N_eff = 0.0f;
              if (normalize) {
                N_eff = datasets[d].fields[f].N_eff_perFreqPerStoke[i][s];
                if (N_eff <= 0.0f) {
                  // Fallback to numVisibilities if N_eff was not pre-computed
                  N_eff = (float)datasets[d]
                              .fields[f]
                              .numVisibilitiesPerFreqPerStoke[i][s];
                }
              }

              ////chi2 VECTOR
              chi2Vector<<<
                  datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV>>>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].device_visibilities[i][s].Vr,
                  datasets[d].fields[f].device_visibilities[i][s].weight,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
              checkCudaErrors(cudaDeviceSynchronize());

              result = deviceReduce<float>(
                  vars_gpu[gpu_idx].device_chi2,
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                  datasets[d]
                      .fields[f]
                      .device_visibilities[i][s]
                      .threadsPerBlockUV);
              // REDUCTIONS
              // chi2
              if (normalize) {
                if (N_eff > 0.0f) {
                  result /= N_eff;
                } else {
                  // Fallback to numVisibilities if N_eff calculation fails
                  result /= datasets[d]
                                .fields[f]
                                .numVisibilitiesPerFreqPerStoke[i][s];
                }
              }

              reduced_chi2 += result;
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);

  return 0.5f * reduced_chi2;
}

__host__ void dchi2(float* I,
                    float* dxi2,
                    float* result_dchi2,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  cudaSetDevice(firstgpu);

  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      // ordered: accumulate gradient in iteration order (i then s) so that
      // floating-point += order is deterministic and runs are reproducible.
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus) ordered
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
              checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_dchi2, 0,
                                         sizeof(float) * M * N));

              // Use pre-computed effective number of samples (calculated once
              // before optimization)
              float N_eff = 0.0f;
              if (normalize) {
                N_eff = datasets[d].fields[f].N_eff_perFreqPerStoke[i][s];
                if (N_eff <= 0.0f) {
                  // Fallback to numVisibilities if N_eff was not pre-computed
                  N_eff = (float)datasets[d]
                              .fields[f]
                              .numVisibilitiesPerFreqPerStoke[i][s];
                }
              }

              // size_t shared_memory;
              // shared_memory =
              // 3*fields[f].numVisibilitiesPerFreq[i]*sizeof(float) +
              // fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex);
              if (NULL != ip->getCKernel()) {
                DChi2<<<numBlocksNN, threadsPerBlockNN>>>(
                    device_noise_image, ip->getCKernel()->getGCFGPU(),
                    vars_gpu[gpu_idx].device_dchi2,
                    datasets[d].fields[f].device_visibilities[i][s].Vr,
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].weight, N,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    fg_scale, noise_cut, datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].fields[f].phs_xobs_pix,
                    datasets[d].fields[f].phs_yobs_pix, DELTAX, DELTAY,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].nu[i],
                    datasets[d].antennas[0].primary_beam, normalize, N_eff);
              } else {
                DChi2<<<numBlocksNN, threadsPerBlockNN>>>(
                    device_noise_image, vars_gpu[gpu_idx].device_dchi2,
                    datasets[d].fields[f].device_visibilities[i][s].Vr,
                    datasets[d].fields[f].device_visibilities[i][s].uvw,
                    datasets[d].fields[f].device_visibilities[i][s].weight, N,
                    datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                    fg_scale, noise_cut, datasets[d].fields[f].ref_xobs_pix,
                    datasets[d].fields[f].ref_yobs_pix,
                    datasets[d].fields[f].phs_xobs_pix,
                    datasets[d].fields[f].phs_yobs_pix, DELTAX, DELTAY,
                    datasets[d].antennas[0].antenna_diameter,
                    datasets[d].antennas[0].pb_factor,
                    datasets[d].antennas[0].pb_cutoff,
                    datasets[d].fields[f].nu[i],
                    datasets[d].antennas[0].primary_beam, normalize, N_eff);
              }
              checkCudaErrors(cudaDeviceSynchronize());

#pragma omp ordered
              {
                // += accumulates over channels into result_dchi2 (slice 0 =
                // I_nu_0, slice 1 = alpha). Ordered so FP accumulation order
                // is deterministic (i then s) for reproducible optimizer results.
                if (flag_opt == -1) {
                  DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                      device_noise_image, result_dchi2,
                      vars_gpu[gpu_idx].device_dchi2, I,
                      datasets[d].fields[f].nu[i], nu_0, noise_cut, N, M);
                  DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                      device_noise_image, result_dchi2,
                      vars_gpu[gpu_idx].device_dchi2, I,
                      datasets[d].fields[f].nu[i], nu_0, noise_cut, N, M);
                } else if (flag_opt % 2 == 0) {
                  DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                      device_noise_image, result_dchi2,
                      vars_gpu[gpu_idx].device_dchi2, I,
                      datasets[d].fields[f].nu[i], nu_0, noise_cut, N, M);
                } else {
                  DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                      device_noise_image, result_dchi2,
                      vars_gpu[gpu_idx].device_dchi2, I,
                      datasets[d].fields[f].nu[i], nu_0, noise_cut, N, M);
                }
                checkCudaErrors(cudaDeviceSynchronize());
              }
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);
}
