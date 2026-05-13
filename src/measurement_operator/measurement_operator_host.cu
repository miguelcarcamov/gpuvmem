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

__host__ void computeImageToVisibilityGridBaseline(
    const MeasurementGridView& model,
    const gpuvmem::ms::FieldMetadata& field,
    varsPerGPU* vars_gpu,
    int gpu_idx,
    float nu,
    float ant1_diameter,
    float ant1_pb_factor,
    float ant1_pb_cutoff,
    int ant1_primary_beam,
    float ant2_diameter,
    float ant2_pb_factor,
    float ant2_pb_cutoff,
    int ant2_primary_beam,
    float fg_scale,
    CKernel* ckernel,
    bool fft_shift) {
  if (!model.I_plane || !model.grid_image || !model.ip) return;
  const long M = model.grid_image->getM();
  const long N = model.grid_image->getN();
  const ImagingGeometry geo = model.grid_image->imaging_geometry();
  model.ip->calculateInu(vars_gpu[gpu_idx].device_I_nu, model.I_plane, nu);

  model.ip->apply_baseline_beam(
      vars_gpu[gpu_idx].device_I_nu, ant1_diameter, ant1_pb_factor, ant1_pb_cutoff,
      ant1_primary_beam, ant2_diameter, ant2_pb_factor, ant2_pb_cutoff,
      ant2_primary_beam, field.ref_xobs_pix, field.ref_yobs_pix, nu, fg_scale);

  if (ckernel != NULL && ckernel->getGCFGPU() != NULL) {
    apply_GCF<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[gpu_idx].device_I_nu,
                                                  ckernel->getGCFGPU(), N);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  FFT2D(vars_gpu[gpu_idx].device_V, vars_gpu[gpu_idx].device_I_nu,
        vars_gpu[gpu_idx].plan, M, N, CUFFT_INVERSE, fft_shift);

  phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(
      vars_gpu[gpu_idx].device_V, M, N, field.phs_xobs_pix, field.phs_yobs_pix,
      geo.reference_column, geo.reference_row, fft_shift);
  checkCudaErrors(cudaDeviceSynchronize());
}
