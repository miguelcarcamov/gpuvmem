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
#include "measurement_operator/measurement_operator_host.cuh"
#include "framework.cuh"
#include "ms/ms_with_gpu.h"
#include "kernels/pillBox2D.cuh"
#include "utils/constants.hh"
#include "utils/physics_utils.cuh"
#include "error.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <set>
#include <cmath>

// Extern variables
extern varsPerGPU* vars_gpu;
extern int nMeasurementSets, num_gpus, firstgpu, max_number_vis, flag_opt, image_count;
extern long M, N;
extern double deltau, deltav, DELTAX, DELTAY;
extern float noise_cut, nu_0;
extern float* device_noise_image;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern std::vector<gpuvmem::ms::MSWithGPU>* g_datasets;
extern int iDivUp(int a, int b);
extern unsigned int NearestPowerOf2(unsigned int x);

// Shared gather buffers for chi2_chunked and calculateErrors_chunked.
namespace {
std::vector<double3*> d_uvw_gather;
std::vector<cufftComplex*> d_Vo_gather;
std::vector<cufftComplex*> d_Vm_gather;
std::vector<cufftComplex*> d_Vr_gather;
std::vector<float*> d_weight_gather;
std::vector<double3**> d_uvw_ptrs;
std::vector<cufftComplex**> d_Vo_ptrs;
std::vector<float**> d_weight_ptrs;
bool gather_buffers_initialized = false;
void ensure_gather_buffers() {
  if (gather_buffers_initialized || max_number_vis <= 0) return;
  d_uvw_gather.resize(num_gpus);
  d_Vo_gather.resize(num_gpus);
  d_Vm_gather.resize(num_gpus);
  d_Vr_gather.resize(num_gpus);
  d_weight_gather.resize(num_gpus);
  d_uvw_ptrs.resize(num_gpus);
  d_Vo_ptrs.resize(num_gpus);
  d_weight_ptrs.resize(num_gpus);
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice(g + firstgpu);
    checkCudaErrors(cudaMalloc(&d_uvw_gather[g], max_number_vis * sizeof(double3)));
    checkCudaErrors(cudaMalloc(&d_Vo_gather[g], max_number_vis * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_Vm_gather[g], max_number_vis * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_Vr_gather[g], max_number_vis * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_weight_gather[g], max_number_vis * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_uvw_ptrs[g], max_number_vis * sizeof(double3*)));
    checkCudaErrors(cudaMalloc(&d_Vo_ptrs[g], max_number_vis * sizeof(cufftComplex*)));
    checkCudaErrors(cudaMalloc(&d_weight_ptrs[g], max_number_vis * sizeof(float*)));
  }
  gather_buffers_initialized = true;
}
}  // namespace

static bool use_chi2_chunked_path() {
  if (!g_datasets || g_datasets->empty()) return false;
  for (size_t d = 0; d < g_datasets->size(); d++)
    if ((*g_datasets)[d].gpu.num_fields() == 0) return false;
  return true;
}

// Chi2 on new MS layout: iterate field -> baseline -> chunk, group by
// (data_desc_id, chan, pol), gather visibilities, degrid, residual, chi2Vector.
__host__ float chi2_chunked(float* I,
                            VirtualImageProcessor* ip,
                            bool normalize,
                            float fg_scale) {
  bool fft_shift = true;
  cudaSetDevice(firstgpu);
  float reduced_chi2 = 0.0f;

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

  ensure_gather_buffers();

  for (int d = 0; d < nMeasurementSets; d++) {
    gpuvmem::ms::MSWithGPU& dw = (*g_datasets)[d];
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    if (meta.num_antennas() == 0) continue;
    const gpuvmem::ms::Antenna& ant0 = meta.antenna(0);
    int primary_beam_int =
        (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0;

    for (size_t f = 0; f < dw.gpu.num_fields(); f++) {
      const gpuvmem::ms::GPUField& gpu_field = dw.gpu.fields()[f];
      const gpuvmem::ms::Field& host_field = dw.ms.field(f);
      const gpuvmem::ms::FieldMetadata& fmeta = host_field.metadata();
      float ref_xobs = fmeta.ref_xobs_pix;
      float ref_yobs = fmeta.ref_yobs_pix;
      float phs_xobs = fmeta.phs_xobs_pix;
      float phs_yobs = fmeta.phs_yobs_pix;

      std::set<int> dd_ids;
      for (const auto& bl : gpu_field.baselines)
        for (const auto& ch : bl.chunks) dd_ids.insert(ch.data_desc_id);

      for (int dd_id : dd_ids) {
        const gpuvmem::ms::DataDescription* dd =
            meta.find_data_description(dd_id);
        if (!dd) continue;
        const gpuvmem::ms::SpectralWindow* spw =
            meta.find_spectral_window(dd->spectral_window_id());
        if (!spw) continue;
        const gpuvmem::ms::Polarization* pol_info =
            meta.find_polarization(dd->polarization_id());
        if (!pol_info) continue;
        const int nchan = dd->nchan();
        const int npol = dd->npol();
        if (nchan <= 0 || npol <= 0) continue;
        const std::vector<int>& corr_type = pol_info->corr_type();

        std::vector<const gpuvmem::ms::GPUChunk*> chunks_with_dd;
        for (const auto& bl : gpu_field.baselines)
          for (const auto& ch : bl.chunks)
            if (ch.data_desc_id == dd_id && ch.count > 0)
              chunks_with_dd.push_back(&ch);

        for (int chan = 0; chan < nchan; chan++) {
          float nu = static_cast<float>(spw->frequency(chan));
          int gpu_idx = chan % num_gpus;
          cudaSetDevice(gpu_idx + firstgpu);

          const bool stokes_imaging = (image_count == npol);
          if (!stokes_imaging)
            computeImageToVisibilityGrid(
                I, ip, vars_gpu, gpu_idx, M, N, nu, ref_xobs, ref_yobs, phs_xobs,
                phs_yobs, ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                primary_beam_int, fg_scale, use_gridding ? degrid_kernel : nullptr,
                fft_shift);

          for (int pol = 0; pol < npol; pol++) {
            if (pol >= static_cast<int>(corr_type.size())) continue;
            int ct = corr_type[pol];
            if (ct != 1 && ct != 2 && ct != 5 && ct != 6) continue;  // RR, LL, XX, YY

            if (stokes_imaging) {
              float* I_slice = I + static_cast<ptrdiff_t>(pol) * M * N;
              computeImageToVisibilityGrid(
                  I_slice, ip, vars_gpu, gpu_idx, M, N, nu, ref_xobs, ref_yobs,
                  phs_xobs, phs_yobs, ant0.antenna_diameter, ant0.pb_factor,
                  ant0.pb_cutoff, primary_beam_int, fg_scale,
                  use_gridding ? degrid_kernel : nullptr, fft_shift);
            }

            const int offset = chan * npol + pol;
            int nch = 0;
            for (const auto* ch : chunks_with_dd) {
              if (offset >= static_cast<int>(ch->count)) continue;
              nch++;
            }
            if (nch == 0) continue;
            if (nch > max_number_vis) continue;

            std::vector<double3*> h_uvw_ptrs(nch);
            std::vector<cufftComplex*> h_Vo_ptrs(nch);
            std::vector<float*> h_weight_ptrs(nch);
            int idx = 0;
            for (const auto* ch : chunks_with_dd) {
              if (offset >= static_cast<int>(ch->count)) continue;
              h_uvw_ptrs[idx] = ch->uvw;
              h_Vo_ptrs[idx] = ch->Vo;
              h_weight_ptrs[idx] = ch->weight;
              idx++;
            }
            nch = idx;
            if (nch == 0) continue;

            checkCudaErrors(cudaMemcpy(d_uvw_ptrs[gpu_idx], h_uvw_ptrs.data(),
                                      nch * sizeof(double3*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_Vo_ptrs[gpu_idx], h_Vo_ptrs.data(),
                                      nch * sizeof(cufftComplex*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_weight_ptrs[gpu_idx], h_weight_ptrs.data(),
                                      nch * sizeof(float*),
                                      cudaMemcpyHostToDevice));

            gatherChunkAtOffset<<<(nch + 255) / 256, 256>>>(
                d_uvw_gather[gpu_idx], d_Vo_gather[gpu_idx],
                d_weight_gather[gpu_idx],
                const_cast<double3 const* const*>(d_uvw_ptrs[gpu_idx]),
                const_cast<cufftComplex const* const*>(d_Vo_ptrs[gpu_idx]),
                const_cast<float const* const*>(d_weight_ptrs[gpu_idx]),
                offset, nch);
            checkCudaErrors(cudaDeviceSynchronize());

            std::vector<double3> h_uvw_m(nch);
            checkCudaErrors(cudaMemcpy(h_uvw_m.data(), d_uvw_gather[gpu_idx],
                                      nch * sizeof(double3),
                                      cudaMemcpyDeviceToHost));
            for (int c = 0; c < nch; c++) {
              h_uvw_m[c].x = metres_to_lambda(h_uvw_m[c].x, nu);
              h_uvw_m[c].y = metres_to_lambda(h_uvw_m[c].y, nu);
              h_uvw_m[c].z = metres_to_lambda(h_uvw_m[c].z, nu);
            }
            checkCudaErrors(cudaMemcpy(d_uvw_gather[gpu_idx], h_uvw_m.data(),
                                      nch * sizeof(double3),
                                      cudaMemcpyHostToDevice));

            long UVpow2 = NearestPowerOf2(nch);
            int threadsV = 512;
            int blocksV = iDivUp(UVpow2, threadsV);
            if (use_gridding && degrid_kernel) {
              degriddingGPU<<<blocksV, threadsV>>>(
                  d_uvw_gather[gpu_idx], d_Vm_gather[gpu_idx],
                  vars_gpu[gpu_idx].device_V, degrid_kernel->getGPUKernel(),
                  deltau, deltav, nch, M, N, degrid_kernel->getm(),
                  degrid_kernel->getn(), degrid_kernel->getSupportX(),
                  degrid_kernel->getSupportY());
            } else {
              bilinearInterpolateVisibility<<<blocksV, threadsV>>>(
                  d_Vm_gather[gpu_idx], vars_gpu[gpu_idx].device_V,
                  d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx], deltau, deltav,
                  nch, M, N, fft_shift);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            residual<<<blocksV, threadsV>>>(
                d_Vr_gather[gpu_idx], d_Vm_gather[gpu_idx], d_Vo_gather[gpu_idx],
                static_cast<long>(nch));
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
                                      sizeof(float) * max_number_vis));
            chi2Vector<<<blocksV, threadsV>>>(
                vars_gpu[gpu_idx].device_chi2, d_Vr_gather[gpu_idx],
                d_weight_gather[gpu_idx], static_cast<long>(nch));
            checkCudaErrors(cudaDeviceSynchronize());

            float result = deviceReduce<float>(
                vars_gpu[gpu_idx].device_chi2, nch, threadsV);
            float N_eff = normalize ? static_cast<float>(nch) : 0.0f;
            if (normalize && N_eff > 0.0f) result /= N_eff;
            reduced_chi2 += result;
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);
  return 0.5f * reduced_chi2;
}

__host__ void dchi2_chunked(float* I,
                            float* dxi2,
                            float* result_dchi2,
                            VirtualImageProcessor* ip,
                            bool normalize,
                            float fg_scale) {
  bool fft_shift = true;
  cudaSetDevice(firstgpu);

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

  static std::vector<double3*> d_uvw_gather;
  static std::vector<cufftComplex*> d_Vo_gather;
  static std::vector<cufftComplex*> d_Vm_gather;
  static std::vector<cufftComplex*> d_Vr_gather;
  static std::vector<float*> d_weight_gather;
  static std::vector<double3**> d_uvw_ptrs;
  static std::vector<cufftComplex**> d_Vo_ptrs;
  static std::vector<float**> d_weight_ptrs;
  static bool dchi2_gather_initialized = false;
  if (!dchi2_gather_initialized && max_number_vis > 0) {
    d_uvw_gather.resize(num_gpus);
    d_Vo_gather.resize(num_gpus);
    d_Vm_gather.resize(num_gpus);
    d_Vr_gather.resize(num_gpus);
    d_weight_gather.resize(num_gpus);
    d_uvw_ptrs.resize(num_gpus);
    d_Vo_ptrs.resize(num_gpus);
    d_weight_ptrs.resize(num_gpus);
    for (int g = 0; g < num_gpus; g++) {
      cudaSetDevice(g + firstgpu);
      checkCudaErrors(cudaMalloc(&d_uvw_gather[g], max_number_vis * sizeof(double3)));
      checkCudaErrors(cudaMalloc(&d_Vo_gather[g], max_number_vis * sizeof(cufftComplex)));
      checkCudaErrors(cudaMalloc(&d_Vm_gather[g], max_number_vis * sizeof(cufftComplex)));
      checkCudaErrors(cudaMalloc(&d_Vr_gather[g], max_number_vis * sizeof(cufftComplex)));
      checkCudaErrors(cudaMalloc(&d_weight_gather[g], max_number_vis * sizeof(float)));
      checkCudaErrors(cudaMalloc(&d_uvw_ptrs[g], max_number_vis * sizeof(double3*)));
      checkCudaErrors(cudaMalloc(&d_Vo_ptrs[g], max_number_vis * sizeof(cufftComplex*)));
      checkCudaErrors(cudaMalloc(&d_weight_ptrs[g], max_number_vis * sizeof(float*)));
    }
    dchi2_gather_initialized = true;
  }

  for (int d = 0; d < nMeasurementSets; d++) {
    gpuvmem::ms::MSWithGPU& dw = (*g_datasets)[d];
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    if (meta.num_antennas() == 0) continue;
    const gpuvmem::ms::Antenna& ant0 = meta.antenna(0);

    for (size_t f = 0; f < dw.gpu.num_fields(); f++) {
      const gpuvmem::ms::GPUField& gpu_field = dw.gpu.fields()[f];
      const gpuvmem::ms::Field& host_field = dw.ms.field(f);
      const gpuvmem::ms::FieldMetadata& fmeta = host_field.metadata();

      std::set<int> dd_ids;
      for (const auto& bl : gpu_field.baselines)
        for (const auto& ch : bl.chunks) dd_ids.insert(ch.data_desc_id);

      for (int dd_id : dd_ids) {
        const gpuvmem::ms::DataDescription* dd =
            meta.find_data_description(dd_id);
        if (!dd) continue;
        const gpuvmem::ms::SpectralWindow* spw =
            meta.find_spectral_window(dd->spectral_window_id());
        if (!spw) continue;
        const gpuvmem::ms::Polarization* pol_info =
            meta.find_polarization(dd->polarization_id());
        if (!pol_info) continue;
        const int nchan = dd->nchan();
        const int npol = dd->npol();
        if (nchan <= 0 || npol <= 0) continue;
        const std::vector<int>& corr_type = pol_info->corr_type();

        std::vector<const gpuvmem::ms::GPUChunk*> chunks_with_dd;
        for (const auto& bl : gpu_field.baselines)
          for (const auto& ch : bl.chunks)
            if (ch.data_desc_id == dd_id && ch.count > 0)
              chunks_with_dd.push_back(&ch);

        const bool stokes_imaging = (image_count == npol);
        for (int chan = 0; chan < nchan; chan++) {
          float nu = static_cast<float>(spw->frequency(chan));
          int gpu_idx = chan % num_gpus;
          cudaSetDevice(gpu_idx + firstgpu);

          if (!stokes_imaging)
            computeImageToVisibilityGrid(
                I, ip, vars_gpu, gpu_idx, M, N, nu, fmeta.ref_xobs_pix,
                fmeta.ref_yobs_pix, fmeta.phs_xobs_pix, fmeta.phs_yobs_pix,
                ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0,
                fg_scale, use_gridding ? degrid_kernel : nullptr, fft_shift);

          for (int pol = 0; pol < npol; pol++) {
            if (pol >= static_cast<int>(corr_type.size())) continue;
            int ct = corr_type[pol];
            if (ct != 1 && ct != 2 && ct != 5 && ct != 6) continue;

            if (stokes_imaging) {
              float* I_slice = I + static_cast<ptrdiff_t>(pol) * M * N;
              computeImageToVisibilityGrid(
                  I_slice, ip, vars_gpu, gpu_idx, M, N, nu, fmeta.ref_xobs_pix,
                  fmeta.ref_yobs_pix, fmeta.phs_xobs_pix, fmeta.phs_yobs_pix,
                  ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                  (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0,
                  fg_scale, use_gridding ? degrid_kernel : nullptr, fft_shift);
            }

            const int offset = chan * npol + pol;
            int nch = 0;
            for (const auto* ch : chunks_with_dd) {
              if (offset < static_cast<int>(ch->count)) nch++;
            }
            if (nch == 0 || nch > max_number_vis) continue;

            std::vector<double3*> h_uvw_ptrs(nch);
            std::vector<cufftComplex*> h_Vo_ptrs(nch);
            std::vector<float*> h_weight_ptrs(nch);
            int idx = 0;
            for (const auto* ch : chunks_with_dd) {
              if (offset >= static_cast<int>(ch->count)) continue;
              h_uvw_ptrs[idx] = ch->uvw;
              h_Vo_ptrs[idx] = ch->Vo;
              h_weight_ptrs[idx] = ch->weight;
              idx++;
            }
            nch = idx;
            if (nch == 0) continue;

            checkCudaErrors(cudaMemcpy(d_uvw_ptrs[gpu_idx], h_uvw_ptrs.data(),
                                      nch * sizeof(double3*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_Vo_ptrs[gpu_idx], h_Vo_ptrs.data(),
                                      nch * sizeof(cufftComplex*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_weight_ptrs[gpu_idx], h_weight_ptrs.data(),
                                      nch * sizeof(float*),
                                      cudaMemcpyHostToDevice));

            gatherChunkAtOffset<<<(nch + 255) / 256, 256>>>(
                d_uvw_gather[gpu_idx], d_Vo_gather[gpu_idx],
                d_weight_gather[gpu_idx],
                const_cast<double3 const* const*>(d_uvw_ptrs[gpu_idx]),
                const_cast<cufftComplex const* const*>(d_Vo_ptrs[gpu_idx]),
                const_cast<float const* const*>(d_weight_ptrs[gpu_idx]),
                offset, nch);
            checkCudaErrors(cudaDeviceSynchronize());

            std::vector<double3> h_uvw_m(nch);
            checkCudaErrors(cudaMemcpy(h_uvw_m.data(), d_uvw_gather[gpu_idx],
                                      nch * sizeof(double3),
                                      cudaMemcpyDeviceToHost));
            for (int c = 0; c < nch; c++) {
              h_uvw_m[c].x = metres_to_lambda(h_uvw_m[c].x, nu);
              h_uvw_m[c].y = metres_to_lambda(h_uvw_m[c].y, nu);
              h_uvw_m[c].z = metres_to_lambda(h_uvw_m[c].z, nu);
            }
            checkCudaErrors(cudaMemcpy(d_uvw_gather[gpu_idx], h_uvw_m.data(),
                                      nch * sizeof(double3),
                                      cudaMemcpyHostToDevice));

            long UVpow2 = NearestPowerOf2(nch);
            int threadsV = 512;
            int blocksV = iDivUp(UVpow2, threadsV);
            if (use_gridding && degrid_kernel) {
              degriddingGPU<<<blocksV, threadsV>>>(
                  d_uvw_gather[gpu_idx], d_Vm_gather[gpu_idx],
                  vars_gpu[gpu_idx].device_V, degrid_kernel->getGPUKernel(),
                  deltau, deltav, nch, M, N, degrid_kernel->getm(),
                  degrid_kernel->getn(), degrid_kernel->getSupportX(),
                  degrid_kernel->getSupportY());
            } else {
              bilinearInterpolateVisibility<<<blocksV, threadsV>>>(
                  d_Vm_gather[gpu_idx], vars_gpu[gpu_idx].device_V,
                  d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx], deltau, deltav,
                  nch, M, N, fft_shift);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            residual<<<blocksV, threadsV>>>(d_Vr_gather[gpu_idx],
                                            d_Vm_gather[gpu_idx],
                                            d_Vo_gather[gpu_idx],
                                            static_cast<long>(nch));
            checkCudaErrors(cudaDeviceSynchronize());

            float N_eff = normalize ? static_cast<float>(nch) : 0.0f;
            if (normalize && N_eff <= 0.0f) N_eff = static_cast<float>(nch);

            checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_dchi2, 0,
                                       sizeof(float) * M * N));
            if (ckernel && ckernel->getGCFGPU()) {
              DChi2<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, ckernel->getGCFGPU(),
                  vars_gpu[gpu_idx].device_dchi2, d_Vr_gather[gpu_idx],
                  d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx], N, nch,
                  fg_scale, noise_cut, fmeta.ref_xobs_pix, fmeta.ref_yobs_pix,
                  fmeta.phs_xobs_pix, fmeta.phs_yobs_pix, DELTAX, DELTAY,
                  ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff, nu,
                  (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0,
                  normalize, N_eff);
            } else {
              DChi2<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, vars_gpu[gpu_idx].device_dchi2,
                  d_Vr_gather[gpu_idx], d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx],
                  N, nch, fg_scale, noise_cut, fmeta.ref_xobs_pix, fmeta.ref_yobs_pix,
                  fmeta.phs_xobs_pix, fmeta.phs_yobs_pix, DELTAX, DELTAY,
                  ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff, nu,
                  (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0,
                  normalize, N_eff);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            if (stokes_imaging) {
              AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(
                  result_dchi2, vars_gpu[gpu_idx].device_dchi2, N, M, pol);
            } else if (flag_opt == -1) {
              DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, N, M);
              DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, N, M);
            } else if (flag_opt % 2 == 0) {
              DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, N, M);
            } else {
              DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, N, M);
            }
            checkCudaErrors(cudaDeviceSynchronize());
          }
        }
      }
    }
  }
  cudaSetDevice(firstgpu);
}

__host__ float chi2(float* I,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  if (!use_chi2_chunked_path())
    return 0.0f;
  return chi2_chunked(I, ip, normalize, fg_scale);
}

__host__ void dchi2(float* I,
                    float* dxi2,
                    float* result_dchi2,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  if (!use_chi2_chunked_path())
    return;
  dchi2_chunked(I, dxi2, result_dchi2, ip, normalize, fg_scale);
}

// Add gradient contribution from single-plane dgi to multi-plane dphi
__host__ void linkAddToDPhi(float* dphi, float* dgi, int index) {
  extern int firstgpu;
  extern long N, M;
  extern dim3 numBlocksNN, threadsPerBlockNN;
  cudaSetDevice(firstgpu);
  AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(dphi, dgi, N, M, index);
  checkCudaErrors(cudaDeviceSynchronize());
}
