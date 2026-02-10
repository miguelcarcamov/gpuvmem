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

#include "psf/psf_host.cuh"
#include "io/MSFITSIO.cuh"  // For PI_D, metres_to_lambda, stokes enum (LL, RR, XX, YY)
#include "framework.cuh"  // For RPDEG_D constant
#include "utils/math_utils.hh"
#include "utils/cuda_utils.cuh"
#include "reduction/reduction_host.cuh"
#include "error.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Extern variables
extern int nMeasurementSets;
extern bool verbose_flag;

__host__ void calc_sBeam(std::vector<double3> uvw,
                         std::vector<float> weight,
                         float nu,
                         double* s_uu,
                         double* s_vv,
                         double* s_uv) {
  double u_lambda, v_lambda;
  double local_s_uu = 0.0;
  double local_s_vv = 0.0;
  double local_s_uv = 0.0;

  // Use reduction for efficient parallel accumulation
#pragma omp parallel for shared(uvw, weight) private(u_lambda, v_lambda) \
    reduction(+ : local_s_uu, local_s_vv, local_s_uv)
  for (int i = 0; i < uvw.size(); i++) {
    u_lambda = metres_to_lambda(uvw[i].x, nu);
    v_lambda = metres_to_lambda(uvw[i].y, nu);

    // Accumulate into reduction variables (no synchronization needed)
    local_s_uu += u_lambda * u_lambda * weight[i];
    local_s_vv += v_lambda * v_lambda * weight[i];
    local_s_uv += u_lambda * v_lambda * weight[i];
  }

  // Update output pointers after parallel reduction
  *s_uu += local_s_uu;
  *s_vv += local_s_vv;
  *s_uv += local_s_uv;
}

__host__ double3 calc_beamSize(double s_uu, double s_vv, double s_uv) {
  double3 beam_size;
  double uv_square = s_uv * s_uv;
  double uu_minus_vv = s_uu - s_vv;
  double uu_plus_vv = s_uu + s_vv;
  double sqrt_in = sqrt((uu_minus_vv * uu_minus_vv) + 4.0 * uv_square);
  beam_size.x = 1.0 / sqrt(2.0) / PI_D /
                sqrt(uu_plus_vv - sqrt_in);  // Major axis in radians
  beam_size.y = 1.0 / sqrt(2.0) / PI_D /
                sqrt(uu_plus_vv + sqrt_in);             // Minor axis in radians
  beam_size.z = -0.5 * atan2(2.0 * s_uv, uu_minus_vv);  // Angle in radians

  return beam_size;
}

__host__ float calculateNoiseAndBeam(std::vector<MSDataset>& datasets,
                                     int* total_visibilities,
                                     int blockSizeV,
                                     double* bmaj,
                                     double* bmin,
                                     double* bpa,
                                     float* noise) {
  // Declaring block size and number of blocks for visibilities
  float variance;
  float sum_weights = 0.0f;
  long UVpow2;
  double s_uu = 0.0;
  double s_vv = 0.0;
  double s_uv = 0.0;

  int device = -1;
  cudaDeviceProp dprop;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&dprop, device));
  for (int d = 0; d < nMeasurementSets; d++) {
    for (int f = 0; f < datasets[d].data.nfields; f++) {
      for (int i = 0; i < datasets[d].data.total_frequencies; i++) {
        for (int s = 0; s < datasets[d].data.nstokes; s++) {
          if (datasets[d].data.corr_type[s] == LL ||
              datasets[d].data.corr_type[s] == RR ||
              datasets[d].data.corr_type[s] == XX ||
              datasets[d].data.corr_type[s] == YY) {
            if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] >
                0) {
              calc_sBeam(datasets[d].fields[f].visibilities[i][s].uvw,
                         datasets[d].fields[f].visibilities[i][s].weight,
                         datasets[d].fields[f].nu[i], &s_uu, &s_vv, &s_uv);
              sum_weights += reduceCPU<float>(
                  datasets[d].fields[f].visibilities[i][s].weight.data(),
                  datasets[d].fields[f].visibilities[i][s].weight.size());
              *total_visibilities +=
                  datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s];
            }
          }
          UVpow2 = NearestPowerOf2(
              datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
          if (blockSizeV == -1) {
            int threads1D, blocks1D;
            int threadsV, blocksV;
            threads1D = 512;
            blocks1D = iDivUp(UVpow2, threads1D);
            if (UVpow2 != 0) {
              getNumBlocksAndThreads(UVpow2, blocks1D, threads1D, blocksV,
                                     threadsV, false);
              datasets[d]
                  .fields[f]
                  .device_visibilities[i][s]
                  .threadsPerBlockUV = threadsV;
              datasets[d].fields[f].device_visibilities[i][s].numBlocksUV =
                  blocksV;
            } else {
              datasets[d]
                  .fields[f]
                  .device_visibilities[i][s]
                  .threadsPerBlockUV = threads1D;
              datasets[d].fields[f].device_visibilities[i][s].numBlocksUV =
                  blocks1D;
            }

          } else {
            datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV =
                blockSizeV;
            datasets[d].fields[f].device_visibilities[i][s].numBlocksUV =
                iDivUp(UVpow2, blockSizeV);
          }
        }
      }
    }
  }

  // We have calculate the running means so we divide by the sum of the weights

  if (sum_weights > 0.0f) {
    s_uu /= sum_weights;
    s_vv /= sum_weights;
    s_uv /= sum_weights;
    variance = 1.0f / sum_weights;
  } else {
    printf("Error: The sum of the visibility weights cannot be zero\n");
    exit(-1);
  }

  double3 beam_size_rad = calc_beamSize(s_uu, s_vv, s_uv);

  *bmaj = beam_size_rad.x / RPDEG_D;  // Major axis to degrees
  *bmin = beam_size_rad.y / RPDEG_D;  // Minor axis to degrees
  *bpa = beam_size_rad.z / RPDEG_D;   // Angle to degrees

  if (verbose_flag) {
    float aux_noise = 0.5f * sqrtf(variance);
    printf("Calculated NOISE %e\n", aux_noise);
  }

  if (*noise <= 0.0) {
    *noise = 0.5f * sqrtf(variance);
    if (verbose_flag) {
      printf("No NOISE keyword entered or detected in header\n");
      printf("Using NOISE: %e ...\n", *noise);
    }
  } else {
    printf("Using header keyword or entered NOISE...\n");
    printf("NOISE = %e\n", *noise);
  }

  return sum_weights;
}
