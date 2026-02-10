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
#include "measurement_operator/measurement_operator_host.cuh"
#include "fft/fft_host.cuh"
#include "beam/beam_kernels.cuh"
#include "visibility/visibility_kernels.cuh"
#include "utils/cuda_utils.cuh"
#include "framework.cuh"
#include <cuda_runtime.h>

extern dim3 threadsPerBlockNN, numBlocksNN;
extern double crpix1, crpix2;

// Measurement operator: transforms image to visibility grid
// This encapsulates the forward model pipeline:
//   calculateInu -> apply_beam -> apply_GCF -> FFT2D -> phase_rotate
__host__ void computeImageToVisibilityGrid(float* I,
                                           VirtualImageProcessor* ip,
                                           varsPerGPU* vars_gpu,
                                           int gpu_idx,
                                           long M,
                                           long N,
                                           float nu,
                                           float ref_xobs_pix,
                                           float ref_yobs_pix,
                                           float phs_xobs_pix,
                                           float phs_yobs_pix,
                                           float antenna_diameter,
                                           float pb_factor,
                                           float pb_cutoff,
                                           int primary_beam,
                                           float fg_scale,
                                           CKernel* ckernel,
                                           bool fft_shift) {
  // Recompute FFT of final image for this frequency/channel
  ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, I, nu);

  // Apply primary beam
  ip->apply_beam(vars_gpu[gpu_idx].device_I_nu, antenna_diameter, pb_factor,
                 pb_cutoff, ref_xobs_pix, ref_yobs_pix, nu, primary_beam,
                 fg_scale);

  // Apply Gridding Correction Function (GCF) if using convolution kernel
  if (ckernel != NULL && ckernel->getGCFGPU() != NULL) {
    apply_GCF<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[gpu_idx].device_I_nu,
                                                  ckernel->getGCFGPU(), N);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // FFT 2D: Transform image to visibility grid
  FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
        vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, fft_shift);

  // Phase rotate to correct phase center
  // Pass fft_shift as dc_at_center since they match (fft_shift=true means DC at
  // center) Pass crpix1 and crpix2 from FITS header (already declared as
  // extern)
  phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(
      vars_gpu[gpu_idx].device_V, M, N, phs_xobs_pix, phs_yobs_pix, crpix1,
      crpix2, fft_shift);
  checkCudaErrors(cudaDeviceSynchronize());
}
