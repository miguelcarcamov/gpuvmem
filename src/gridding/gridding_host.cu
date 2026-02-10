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

#include "gridding/gridding_host.cuh"
#include "gridding/gridding_kernels.cuh"
#include "reduction/reduction_host.cuh"
#include "utils/math_utils.hh"
#include "utils/cuda_utils.cuh"
#include "framework.cuh"
#include "io/MSFITSIO.cuh"
#include "utils/complexOps.cuh"
#include "error.cuh"
#include "framework.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <omp.h>

// Include measurement operator header
#include "measurement_operator/measurement_operator_host.cuh"

// Extern variables
extern varsPerGPU* vars_gpu;
extern int nMeasurementSets;
extern bool verbose_flag;
extern long M, N;
extern double deltau, deltav;
extern double crpix1, crpix2;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern int num_gpus, firstgpu;

// Extern variables for gridding functions
extern varsPerGPU* vars_gpu;
extern long M, N;
extern double deltau, deltav;
extern double crpix1, crpix2;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern int num_gpus, firstgpu;

__host__ void do_gridding(std::vector<Field>& fields,
                          MSData* data,
                          double deltau,
                          double deltav,
                          int M,
                          int N,
                          CKernel* ckernel,
                          int gridding) {
  // ========================================================================
  // Initialize grid arrays (reused for each frequency/stokes combination)
  // ========================================================================
  std::vector<float> g_weights(M * N);  // Accumulated weights
  std::vector<float> g_weights_aux(
      M * N);  // Sum of weight^2 (for effective weight calc)
  std::vector<cufftComplex> g_Vo(M * N);  // Gridded visibilities
  std::vector<double3> g_uvw(M * N);      // Grid UVW coordinates in meters

  // Zero-initialization constants
  cufftComplex complex_zero = floatComplexZero();
  double3 double3_zero = {0.0, 0.0, 0.0};

  // Track maximum number of visibilities across all channels/stokes
  int local_max = 0;
  int max = 0;

  // ========================================================================
  // Private variables for parallel gridding loop
  // ========================================================================
  double j_fp, k_fp;              // Floating-point grid coordinates
  int j, k;                       // Integer grid pixel indices
  double grid_pos_x, grid_pos_y;  // Grid positions in lambda units
  double3 uvw;                    // UVW coordinates
  float w;                        // Visibility weight
  cufftComplex Vo;                // Visibility value
  int shifted_j, shifted_k;    // Shifted grid indices (for kernel convolution)
  int kernel_i, kernel_j;      // Kernel array indices
  int visCounterPerFreq = 0;   // Counter for visibilities per frequency
  float ckernel_result = 1.0;  // Kernel value at current position
  // ========================================================================
  // Main loop: Process each field, frequency, and stokes parameter
  // ========================================================================
  // Pre-calculate constants that don't change per frequency/stokes
  double center_j = floor(N / 2.0);
  double center_k = floor(M / 2.0);
  int support_x = ckernel->getSupportX();
  int support_y = ckernel->getSupportY();

  for (int f = 0; f < data->nfields; f++) {
    for (int i = 0; i < data->total_frequencies; i++) {
      visCounterPerFreq = 0;

      // Pre-calculate frequency-dependent constants (same for all stokes)
      float freq = fields[f].nu[i];
      float lambda = freq_to_wavelength(freq);

      for (int s = 0; s < data->nstokes; s++) {
        // ====================================================================
        // Step 1: Backup original visibility data before gridding
        // ====================================================================
        int num_visibilities = fields[f].numVisibilitiesPerFreqPerStoke[i][s];
        fields[f].backup_visibilities[i][s].uvw.resize(num_visibilities);
        fields[f].backup_visibilities[i][s].Vo.resize(num_visibilities);
        fields[f].backup_visibilities[i][s].weight.resize(num_visibilities);

        // Copy original data to backup
        fields[f].backup_visibilities[i][s].uvw.assign(
            fields[f].visibilities[i][s].uvw.begin(),
            fields[f].visibilities[i][s].uvw.end());
        fields[f].backup_visibilities[i][s].weight.assign(
            fields[f].visibilities[i][s].weight.begin(),
            fields[f].visibilities[i][s].weight.end());
        fields[f].backup_visibilities[i][s].Vo.assign(
            fields[f].visibilities[i][s].Vo.begin(),
            fields[f].visibilities[i][s].Vo.end());

        // ====================================================================
        // Step 2: Grid visibilities using convolution kernel
        // ====================================================================
        // For each visibility, grid both V(u,v) and its Hermitian conjugate
        // V(-u,-v) = V*(u,v) Measurement sets only contain half the UV plane
        // data, so we need both to fill the full grid. We multiply weights by
        // 0.5 to account for each visibility being gridded at two positions.
#pragma omp parallel for schedule(static) num_threads(gridding) shared(       \
        g_weights, g_weights_aux, g_Vo, freq, center_j, center_k, support_x,  \
            support_y) private(j_fp, k_fp, j, k, grid_pos_x, grid_pos_y, uvw, \
                                   w, Vo, shifted_j, shifted_k, kernel_i,     \
                                   kernel_j, ckernel_result)
        for (int z = 0; z < num_visibilities; z++) {
          // Load visibility data once (cache-friendly)
          uvw = fields[f].visibilities[i][s].uvw[z];
          w = fields[f].visibilities[i][s].weight[z] *
              0.5f;  // Half weight since gridded at two positions
          Vo = fields[f].visibilities[i][s].Vo[z];

          // ================================================================
          // Step 2a: Convert UVW coordinates from meters to lambda units
          // ================================================================
          double u_lambda = metres_to_lambda(uvw.x, freq);
          double v_lambda = metres_to_lambda(uvw.y, freq);

          // ================================================================
          // Step 2b: Grid both the original visibility and its Hermitian
          // conjugate
          // ================================================================
          // Loop over both: h=0 for original V(u,v), h=1 for Hermitian
          // V(-u,-v)=V*(u,v)
          for (int h = 0; h < 2; h++) {
            double u_pos = (h == 0) ? u_lambda : -u_lambda;
            double v_pos = (h == 0) ? v_lambda : -v_lambda;
            float Vo_imag = (h == 0) ? Vo.y : -Vo.y;  // Conjugate for Hermitian

            // Calculate grid pixel coordinates
            grid_pos_x = u_pos / deltau;
            grid_pos_y = v_pos / deltav;

            // Center the grid: center pixel is at floor(N/2) for both even and
            // odd N
            j_fp = grid_pos_x + center_j + 0.5;
            k_fp = grid_pos_y + center_k + 0.5;
            j = int(j_fp);
            k = int(k_fp);

            // ================================================================
            // Step 2c: Apply convolution kernel to grid this visibility
            // ================================================================
            for (int m = -support_y; m <= support_y; m++) {
              for (int n = -support_x; n <= support_x; n++) {
                shifted_j = j + n;
                shifted_k = k + m;
                kernel_j = n + support_x;
                kernel_i = m + support_y;

                if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
                    shifted_j < N && kernel_i >= 0 &&
                    kernel_i < ckernel->getm() && kernel_j >= 0 &&
                    kernel_j < ckernel->getn()) {
                  ckernel_result = ckernel->getKernelValue(kernel_i, kernel_j);
                  float ckernel_result_sq = ckernel_result * ckernel_result;
                  int grid_idx = N * shifted_k + shifted_j;

#pragma omp atomic
                  g_weights[grid_idx] += w * ckernel_result;

#pragma omp atomic
                  g_weights_aux[grid_idx] += w * ckernel_result_sq;

#pragma omp critical
                  {
                    g_Vo[grid_idx].x += w * Vo.x * ckernel_result;
                    g_Vo[grid_idx].y += w * Vo_imag * ckernel_result;
                  }
                }
              }
            }
          }
        }

        // ====================================================================
        // Step 3: Normalize gridded visibilities and calculate effective
        // weights
        // ====================================================================
        // Convert grid coordinates from lambda back to meters and normalize
#pragma omp parallel for schedule(static) \
    shared(g_weights, g_weights_aux, g_Vo, g_uvw, lambda, center_j, center_k)
        for (int grid_k = 0; grid_k < M; grid_k++) {
          for (int grid_j = 0; grid_j < N; grid_j++) {
            int grid_idx = N * grid_k + grid_j;

            // ================================================================
            // Step 3a: Calculate UVW coordinates in meters for this grid cell
            // ================================================================
            // Use same centering formula as forward calculation
            double u_lambdas = (grid_j - center_j) * deltau;
            double v_lambdas = (grid_k - center_k) * deltav;

            // Convert from lambda units back to meters using the frequency for
            // this channel
            double u_meters = u_lambdas * lambda;
            double v_meters = v_lambdas * lambda;

            g_uvw[grid_idx].x = u_meters;
            g_uvw[grid_idx].y = v_meters;

            // ================================================================
            // Step 3b: Normalize visibilities and calculate effective weights
            // ================================================================
            float ws = g_weights[grid_idx];  // Sum of weights: Σ(w * kernel)
            float aux_ws =
                g_weights_aux[grid_idx];  // Sum of weight^2: Σ(w * kernel^2)

            if (aux_ws != 0.0f && ws != 0.0f) {
              // Effective weight accounts for kernel convolution:
              //   weight_eff = (Σ w_i * k_i)^2 / Σ (w_i * k_i)^2
              // This gives the equivalent weight if all visibilities were at
              // the same point
              float weight_eff = ws * ws / aux_ws;

              // Normalize visibility: divide by accumulated weight sum
              // This gives the weighted average visibility at this grid point
              g_Vo[grid_idx].x /= ws;
              g_Vo[grid_idx].y /= ws;

              // Store effective weight
              g_weights[grid_idx] = weight_eff;
            } else {
              // No valid data at this grid point (no visibilities contributed)
              g_weights[grid_idx] = 0.0f;
              g_Vo[grid_idx].x = 0.0f;
              g_Vo[grid_idx].y = 0.0f;
            }
          }
        }

        // ====================================================================
        // Step 4: Extract non-zero gridded visibilities back to sparse format
        // ====================================================================
        // Count how many grid cells have valid (non-zero) visibilities
        int visCounter = 0;
#pragma omp parallel for schedule(static) shared(g_weights) \
    reduction(+ : visCounter)
        for (int grid_k = 0; grid_k < M; grid_k++) {
          for (int grid_j = 0; grid_j < N; grid_j++) {
            int grid_idx = N * grid_k + grid_j;
            if (g_weights[grid_idx] > 0.0f) {
              visCounter++;
            }
          }
        }

        // Resize output arrays to hold only non-zero visibilities
        fields[f].visibilities[i][s].uvw.resize(visCounter);
        fields[f].visibilities[i][s].Vo.resize(visCounter);
        fields[f].visibilities[i][s].Vm.resize(visCounter);
        fields[f].visibilities[i][s].weight.resize(visCounter);

        // Initialize Vm (model visibilities) to zero
        if (visCounter > 0) {
          memset(&fields[f].visibilities[i][s].Vm[0], 0,
                 visCounter * sizeof(cufftComplex));
        }

        // Copy non-zero gridded visibilities to output arrays
        int output_idx = 0;
        for (int grid_k = 0; grid_k < M; grid_k++) {
          for (int grid_j = 0; grid_j < N; grid_j++) {
            int grid_idx = N * grid_k + grid_j;
            float weight = g_weights[grid_idx];

            if (weight > 0.0f) {
              // Copy UVW coordinates (w is set to 0 for gridded data)
              fields[f].visibilities[i][s].uvw[output_idx].x =
                  g_uvw[grid_idx].x;
              fields[f].visibilities[i][s].uvw[output_idx].y =
                  g_uvw[grid_idx].y;
              fields[f].visibilities[i][s].uvw[output_idx].z = 0.0;

              // Copy normalized visibility
              fields[f].visibilities[i][s].Vo[output_idx] =
                  make_cuFloatComplex(g_Vo[grid_idx].x, g_Vo[grid_idx].y);

              // Copy effective weight
              fields[f].visibilities[i][s].weight[output_idx] = weight;

              output_idx++;
            }
          }
        }

        // ====================================================================
        // Step 5: Update visibility counts and backup
        // ====================================================================
        // Backup old count before updating
        fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s] =
            fields[f].numVisibilitiesPerFreqPerStoke[i][s];

        // Update to actual gridded visibility count
        fields[f].numVisibilitiesPerFreqPerStoke[i][s] = visCounter;
        if (visCounter > 0) {
          visCounterPerFreq += visCounter;
        }

        // ====================================================================
        // Step 6: Clear grid arrays for next stokes parameter
        // ====================================================================
        std::fill_n(g_weights_aux.begin(), M * N, 0.0f);
        std::fill_n(g_weights.begin(), M * N, 0.0f);
        std::fill_n(g_uvw.begin(), M * N, double3_zero);
        std::fill_n(g_Vo.begin(), M * N, complex_zero);
      }  // End stokes loop

      // Track maximum number of visibilities across all stokes for this
      // frequency
      local_max =
          *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i].begin(),
                            fields[f].numVisibilitiesPerFreqPerStoke[i].end());
      if (local_max > max) {
        max = local_max;
      }

      // Update total visibilities per frequency
      fields[f].backup_numVisibilitiesPerFreq[i] =
          fields[f].numVisibilitiesPerFreq[i];
      fields[f].numVisibilitiesPerFreq[i] = visCounterPerFreq;
    }  // End frequency loop
  }  // End field loop

  // Store global maximum across all fields/frequencies/stokes
  data->max_number_visibilities_in_channel_and_stokes = max;
}

__host__ void griddedTogrid(std::vector<cufftComplex>& Vm_gridded,
                            std::vector<cufftComplex> Vm_gridded_sp,
                            std::vector<double3> uvw_gridded_sp,
                            double deltau,
                            double deltav,
                            float freq,
                            long M,
                            long N,
                            int numvis) {
  float lambda = freq_to_wavelength(freq);
  double deltau_meters = deltau * lambda;
  double deltav_meters = deltav * lambda;

  cufftComplex complex_zero = floatComplexZero();

  std::fill_n(Vm_gridded.begin(), M * N, complex_zero);

  double center_j = floor(N / 2.0);
  double center_k = floor(M / 2.0);

  // Parallelize the loop with protection against race conditions
  // In theory, each visibility maps to a unique grid cell, but floating-point
  // rounding could cause collisions, so we protect the write with a critical
  // section
  int j, k;
  double grid_pos_x, grid_pos_y;
#pragma omp parallel for schedule(static)                            \
    shared(Vm_gridded, uvw_gridded_sp, Vm_gridded_sp, deltau_meters, \
               deltav_meters, center_j, center_k, M,                 \
               N) private(j, k, grid_pos_x, grid_pos_y)
  for (int i = 0; i < numvis; i++) {
    grid_pos_x = uvw_gridded_sp[i].x / deltau_meters;
    grid_pos_y = uvw_gridded_sp[i].y / deltav_meters;
    // Match the gridding coordinate calculation exactly:
    // j_fp = grid_pos_x + center_j + 0.5; j = int(j_fp)
    j = int(grid_pos_x + center_j + 0.5);
    k = int(grid_pos_y + center_k + 0.5);
    if (j >= 0 && j < N && k >= 0 && k < M) {
      // Critical section protects against potential collisions (should be rare)
      // Each visibility should map to a unique grid cell after gridding
#pragma omp critical
      {
        Vm_gridded[N * k + j] = Vm_gridded_sp[i];
      }
    }
  }
}

__host__ void degridding(std::vector<Field>& fields,
                         MSData data,
                         double deltau,
                         double deltav,
                         int num_gpus,
                         int firstgpu,
                         int blockSizeV,
                         long M,
                         long N,
                         CKernel* ckernel,
                         float* I,
                         VirtualImageProcessor* ip,
                         MSDataset& dataset) {
  long UVpow2;
  bool fft_shift = true;  // fft_shift=false (DC at corner 0,0)

  // Instead of using sparse gridded model visibilities (computed with bilinear
  // interpolation), we recompute the FFT of the final image for each
  // frequency/channel and use the full grid directly. This is more accurate
  // than reconstructing from sparse samples.

  for (int f = 0; f < data.nfields; f++) {
#pragma omp parallel for schedule(static, 1) num_threads(num_gpus)
    for (int i = 0; i < data.total_frequencies; i++) {
      unsigned int j = omp_get_thread_num();
      unsigned int num_cpu_threads = omp_get_num_threads();
      int gpu_idx = i % num_gpus;
      cudaSetDevice(gpu_idx + firstgpu);
      int gpu_id = -1;
      cudaGetDevice(&gpu_id);

      // Compute visibility grid from image using common pipeline
      // Use shift=true to move DC component to center, matching the gridding
      // coordinate system which uses center_j = floor(N/2.0)
      if (dataset.antennas.size() > 0) {
        computeImageToVisibilityGrid(
            I, ip, vars_gpu, gpu_idx, M, N, fields[f].nu[i],
            fields[f].ref_xobs_pix, fields[f].ref_yobs_pix,
            fields[f].phs_xobs_pix, fields[f].phs_yobs_pix,
            dataset.antennas[0].antenna_diameter, dataset.antennas[0].pb_factor,
            dataset.antennas[0].pb_cutoff, dataset.antennas[0].primary_beam,
            1.0f, ckernel, fft_shift);
      }

      for (int s = 0; s < data.nstokes; s++) {
        // Now the number of visibilities will be the original one (restore from
        // backup)
        fields[f].numVisibilitiesPerFreqPerStoke[i][s] =
            fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s];

        fields[f].visibilities[i][s].uvw.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].weight.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].Vm.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        fields[f].visibilities[i][s].Vo.resize(
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vr,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMemset(fields[f].device_visibilities[i][s].Vr, 0,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

        checkCudaErrors(cudaMalloc(
            &fields[f].device_visibilities[i][s].uvw,
            sizeof(double3) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(
            cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
                       sizeof(cufftComplex) *
                           fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
        checkCudaErrors(cudaMalloc(
            &fields[f].device_visibilities[i][s].weight,
            sizeof(float) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

        // Copy original Vo visibilities to host
        fields[f].visibilities[i][s].Vo.assign(
            fields[f].backup_visibilities[i][s].Vo.begin(),
            fields[f].backup_visibilities[i][s].Vo.end());

        // Note: vars_gpu[gpu_idx].device_V already contains the full FFT grid
        // (computed above), so we don't need to copy from gridded_visibilities

        // Copy original (u,v) positions and weights to host and device

        fields[f].visibilities[i][s].uvw.assign(
            fields[f].backup_visibilities[i][s].uvw.begin(),
            fields[f].backup_visibilities[i][s].uvw.end());
        fields[f].visibilities[i][s].weight.assign(
            fields[f].backup_visibilities[i][s].weight.begin(),
            fields[f].backup_visibilities[i][s].weight.end());

        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].backup_visibilities[i][s].uvw.data(),
            sizeof(double3) * fields[f].backup_visibilities[i][s].uvw.size(),
            cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(fields[f].device_visibilities[i][s].Vo,
                       fields[f].backup_visibilities[i][s].Vo.data(),
                       sizeof(cufftComplex) *
                           fields[f].backup_visibilities[i][s].Vo.size(),
                       cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            fields[f].device_visibilities[i][s].weight,
            fields[f].backup_visibilities[i][s].weight.data(),
            sizeof(float) * fields[f].backup_visibilities[i][s].weight.size(),
            cudaMemcpyHostToDevice));

        if (blockSizeV == -1) {
          int threads1D, blocks1D;
          int threadsV, blocksV;
          long UVpow2 =
              NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
          threads1D = 512;
          blocks1D = iDivUp(UVpow2, threads1D);
          getNumBlocksAndThreads(UVpow2, blocks1D, threads1D, blocksV, threadsV,
                                 false);
          fields[f].device_visibilities[i][s].threadsPerBlockUV = threadsV;
          fields[f].device_visibilities[i][s].numBlocksUV = blocksV;
        } else {
          fields[f].device_visibilities[i][s].threadsPerBlockUV = blockSizeV;
          fields[f].device_visibilities[i][s].numBlocksUV = iDivUp(
              NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]),
              blockSizeV);
        }

        // Convert UVW coordinates from meters to lambda units (required for
        // degriddingGPU) We sample at original (u,v) coordinates - no Hermitian
        // symmetry manipulation needed since the FFT grid from ifft2 has full
        // complex values at all positions
        convertUVWToLambda<<<
            fields[f].device_visibilities[i][s].numBlocksUV,
            fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].uvw, fields[f].nu[i],
            fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
        checkCudaErrors(cudaDeviceSynchronize());

        // Degridding: Use proper convolution kernel degridding (works for both
        // 1x1 PillBox and larger kernels). For 1x1 PillBox, support=0 so it
        // reduces to nearest neighbor, matching the gridding procedure exactly.
        degriddingGPU<<<
            fields[f].device_visibilities[i][s].numBlocksUV,
            fields[f].device_visibilities[i][s].threadsPerBlockUV>>>(
            fields[f].device_visibilities[i][s].uvw,
            fields[f].device_visibilities[i][s].Vm, vars_gpu[gpu_idx].device_V,
            ckernel->getGPUKernel(), deltau, deltav,
            fields[f].numVisibilitiesPerFreqPerStoke[i][s], M, N,
            ckernel->getm(), ckernel->getn(), ckernel->getSupportX(),
            ckernel->getSupportY());
        checkCudaErrors(cudaDeviceSynchronize());
      }
    }
  }
}
