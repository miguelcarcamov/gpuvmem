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
#include "classes/image.cuh"
#include "ms/ms_with_gpu.h"
#include "kernels/pillBox2D.cuh"
#include "utils/constants.hh"
#include "utils/physics_utils.cuh"
#include "error.cuh"
#include "framework/cuda_grid.cuh"
#include "cli/gpuvmem_cli_config.hh"
#include "ms/polarization.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <set>
#include <cmath>

// Extern variables
extern Vars variables;
extern varsPerGPU* vars_gpu;
extern int nMeasurementSets, num_gpus, firstgpu, max_number_vis, flag_opt, image_count;
extern long M, N;
extern float noise_cut, nu_0;
extern float* device_noise_image;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern std::vector<gpuvmem::ms::MSWithGPU>* g_datasets;
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
int gather_buffer_capacity = 0;

/** Device index within a chunk for this (chan, pol); -1 if absent. */
int vis_slot_for(const gpuvmem::ms::TimeSample& ts, int chan, int pol) {
  for (size_t vi = 0; vi < ts.visibilities().size(); ++vi) {
    const auto& v = ts.visibilities()[vi];
    if (v.chan == chan && v.pol == pol) return static_cast<int>(vi);
  }
  return -1;
}

void collect_chunks_for_chan_pol(const gpuvmem::ms::GPUField& gpu_field,
                                 const gpuvmem::ms::Field& host_field,
                                 int dd_id,
                                 int chan,
                                 int pol,
                                 std::vector<const gpuvmem::ms::GPUChunk*>& chunks_out,
                                 std::vector<int>& slots_out) {
  chunks_out.clear();
  slots_out.clear();
  if (gpu_field.baselines.size() != host_field.baselines().size()) return;
  for (size_t bi = 0; bi < gpu_field.baselines.size(); ++bi) {
    const gpuvmem::ms::GPUBaseline& gbl = gpu_field.baselines[bi];
    const gpuvmem::ms::Baseline& hbl = host_field.baselines()[bi];
    if (gbl.chunks.size() != hbl.time_samples().size()) continue;
    for (size_t ti = 0; ti < gbl.chunks.size(); ++ti) {
      const gpuvmem::ms::GPUChunk& ch = gbl.chunks[ti];
      if (ch.data_desc_id != dd_id || ch.empty()) continue;
      const gpuvmem::ms::TimeSample& ts = hbl.time_samples()[ti];
      const int slot = vis_slot_for(ts, chan, pol);
      if (slot < 0 || slot >= static_cast<int>(ch.count)) continue;
      chunks_out.push_back(&ch);
      slots_out.push_back(slot);
    }
  }
}

/** After gridding, GPU mirrors `gridded_ms` (few baselines), not native `ms`. */
const gpuvmem::ms::Field& host_field_for_gpu_field(const gpuvmem::ms::MSWithGPU& dw,
                                                   size_t field_index,
                                                   const gpuvmem::ms::GPUField& gf) {
  if (dw.gridded_ms.num_fields() > field_index) {
    const gpuvmem::ms::Field& cand = dw.gridded_ms.field(field_index);
    if (cand.baselines().size() == gf.baselines.size()) return cand;
  }
  return dw.ms.field(field_index);
}

void free_gather_buffers() {
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice(g + firstgpu);
    if (g < static_cast<int>(d_uvw_gather.size()) && d_uvw_gather[g]) {
      cudaFree(d_uvw_gather[g]);
      d_uvw_gather[g] = nullptr;
    }
    if (g < static_cast<int>(d_Vo_gather.size()) && d_Vo_gather[g]) {
      cudaFree(d_Vo_gather[g]);
      d_Vo_gather[g] = nullptr;
    }
    if (g < static_cast<int>(d_Vm_gather.size()) && d_Vm_gather[g]) {
      cudaFree(d_Vm_gather[g]);
      d_Vm_gather[g] = nullptr;
    }
    if (g < static_cast<int>(d_Vr_gather.size()) && d_Vr_gather[g]) {
      cudaFree(d_Vr_gather[g]);
      d_Vr_gather[g] = nullptr;
    }
    if (g < static_cast<int>(d_weight_gather.size()) && d_weight_gather[g]) {
      cudaFree(d_weight_gather[g]);
      d_weight_gather[g] = nullptr;
    }
    if (g < static_cast<int>(d_uvw_ptrs.size()) && d_uvw_ptrs[g]) {
      cudaFree(d_uvw_ptrs[g]);
      d_uvw_ptrs[g] = nullptr;
    }
    if (g < static_cast<int>(d_Vo_ptrs.size()) && d_Vo_ptrs[g]) {
      cudaFree(d_Vo_ptrs[g]);
      d_Vo_ptrs[g] = nullptr;
    }
    if (g < static_cast<int>(d_weight_ptrs.size()) && d_weight_ptrs[g]) {
      cudaFree(d_weight_ptrs[g]);
      d_weight_ptrs[g] = nullptr;
    }
  }
  gather_buffers_initialized = false;
  gather_buffer_capacity = 0;
}

void ensure_gather_buffers() {
  if (max_number_vis <= 0) return;
  if (gather_buffers_initialized && gather_buffer_capacity >= max_number_vis) return;

  if (gather_buffers_initialized) free_gather_buffers();

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
  gather_buffer_capacity = max_number_vis;
  gather_buffers_initialized = true;
}

void collect_chunks_for_baseline_chan_pol(const gpuvmem::ms::GPUField& gpu_field,
                                          const gpuvmem::ms::Field& host_field,
                                          size_t baseline_index,
                                          int dd_id,
                                          int chan,
                                          int pol,
                                          std::vector<const gpuvmem::ms::GPUChunk*>& chunks_out,
                                          std::vector<int>& slots_out) {
  chunks_out.clear();
  slots_out.clear();
  if (baseline_index >= gpu_field.baselines.size()) return;
  if (baseline_index >= host_field.baselines().size()) return;
  const gpuvmem::ms::GPUBaseline& gbl = gpu_field.baselines[baseline_index];
  const gpuvmem::ms::Baseline& hbl = host_field.baselines()[baseline_index];
  if (gbl.chunks.size() != hbl.time_samples().size()) return;
  for (size_t ti = 0; ti < gbl.chunks.size(); ++ti) {
    const gpuvmem::ms::GPUChunk& ch = gbl.chunks[ti];
    if (ch.data_desc_id != dd_id || ch.empty()) continue;
    const gpuvmem::ms::TimeSample& ts = hbl.time_samples()[ti];
    const int slot = vis_slot_for(ts, chan, pol);
    if (slot < 0 || slot >= static_cast<int>(ch.count)) continue;
    chunks_out.push_back(&ch);
    slots_out.push_back(slot);
  }
}

static const gpuvmem::ms::Antenna& antenna_for_baseline(
    const gpuvmem::ms::MeasurementSetMetadata& meta, int ant_id) {
  if (ant_id >= 0 && static_cast<size_t>(ant_id) < meta.num_antennas())
    return meta.antenna(static_cast<size_t>(ant_id));
  return meta.antenna(0);
}

static int primary_beam_type_int(gpuvmem::ms::PrimaryBeamType t) {
  return (t == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0;
}

}  // namespace

static bool use_chi2_chunked_path() {
  if (!g_datasets || g_datasets->empty()) return false;
  for (size_t d = 0; d < g_datasets->size(); d++)
    if ((*g_datasets)[d].gpu.num_fields() == 0) return false;
  return true;
}

// Chi2 on chunked MS: per baseline, per channel — baseline PB sqrt(A1*A2), FFT to device_V,
// gather that baseline's vis for (dd, chan, pol), degrid/interpolate, residual, accumulate.
__host__ float chi2_chunked(float* I,
                            const Image* grid_image,
                            VirtualImageProcessor* ip,
                            bool normalize,
                            float fg_scale) {
  if (!grid_image) return 0.0f;
  const long Mc = grid_image->getM();
  const long Nc = grid_image->getN();
  const ImagingGeometry geom = grid_image->imaging_geometry();
  bool fft_shift = true;
  cudaSetDevice(firstgpu);
  cudaDeviceProp chi2_dev_prop{};
  checkCudaErrors(cudaGetDeviceProperties(&chi2_dev_prop, firstgpu));
  float reduced_chi2 = 0.0f;

  static PillBox2D* degrid_kernel = NULL;
  static bool degrid_kernel_initialized = false;
  CKernel* ckernel = ip->getCKernel();
  bool use_gridding = (ckernel != NULL && ckernel->getGPUKernel() != NULL);
  if (use_gridding && !degrid_kernel_initialized) {
    degrid_kernel = new PillBox2D(1, 1);
    degrid_kernel->setGPUID(firstgpu);
    degrid_kernel->setSigmas(fabs(geom.deltau), fabs(geom.deltav));
    degrid_kernel->buildKernel();
    degrid_kernel_initialized = true;
  }

  ip->clipWNoise(I);

  ensure_gather_buffers();

  for (int d = 0; d < nMeasurementSets; d++) {
    gpuvmem::ms::MSWithGPU& dw = (*g_datasets)[d];
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    if (meta.num_antennas() == 0) continue;

    for (size_t f = 0; f < dw.gpu.num_fields(); f++) {
      const gpuvmem::ms::GPUField& gpu_field = dw.gpu.fields()[f];
      const gpuvmem::ms::Field& host_field = host_field_for_gpu_field(dw, f, gpu_field);
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

        std::vector<const gpuvmem::ms::GPUChunk*> chunks_sel;
        std::vector<int> slots_sel;

        for (int chan = 0; chan < nchan; chan++) {
          float nu = static_cast<float>(spw->frequency(chan));
          int gpu_idx = chan % num_gpus;
          cudaSetDevice(gpu_idx + firstgpu);

          /* Stokes mode only when user set -S/--stokes; otherwise MFS can have
             image_count==npol by coincidence (e.g. 2 terms + RR/LL) and must
             use the joint forward model, not per-pol image slices. */
          const bool stokes_imaging =
              !variables.stokes.empty() && image_count == npol;

          for (size_t bi = 0; bi < gpu_field.baselines.size(); ++bi) {
            const gpuvmem::ms::GPUBaseline& gbl = gpu_field.baselines[bi];
            const gpuvmem::ms::Antenna& a1 = antenna_for_baseline(meta, gbl.antenna1);
            const gpuvmem::ms::Antenna& a2 = antenna_for_baseline(meta, gbl.antenna2);
            const int pb1 = primary_beam_type_int(a1.primary_beam);
            const int pb2 = primary_beam_type_int(a2.primary_beam);

            if (!stokes_imaging) {
              computeImageToVisibilityGridBaseline(
                  {I, grid_image, ip}, fmeta, vars_gpu, gpu_idx, nu,
                  a1.antenna_diameter, a1.pb_factor, a1.pb_cutoff, pb1,
                  a2.antenna_diameter, a2.pb_factor, a2.pb_cutoff, pb2, fg_scale,
                  use_gridding ? degrid_kernel : nullptr, fft_shift);
            }

            for (int pol = 0; pol < npol; pol++) {
              if (!corr_type.empty()) {
                if (pol >= static_cast<int>(corr_type.size())) continue;
                const int ct = corr_type[static_cast<size_t>(pol)];
                if (!gpuvmem::ms::is_circular(ct) && !gpuvmem::ms::is_linear(ct)) continue;
              }

              if (stokes_imaging) {
                float* I_slice = I + static_cast<ptrdiff_t>(pol) * Mc * Nc;
                computeImageToVisibilityGridBaseline(
                    {I_slice, grid_image, ip}, fmeta, vars_gpu, gpu_idx, nu,
                    a1.antenna_diameter, a1.pb_factor, a1.pb_cutoff, pb1,
                    a2.antenna_diameter, a2.pb_factor, a2.pb_cutoff, pb2, fg_scale,
                    use_gridding ? degrid_kernel : nullptr, fft_shift);
              }

              collect_chunks_for_baseline_chan_pol(gpu_field, host_field, bi, dd_id,
                                                   chan, pol, chunks_sel, slots_sel);
            const int nch = static_cast<int>(chunks_sel.size());
            if (nch == 0) continue;
            if (nch > max_number_vis) {
              if (gpuvmem_cli_verbose()) {
                std::cerr << "WARNING: Chi2 gather nch=" << nch << " exceeds max_number_vis="
                          << max_number_vis << " (chan=" << chan << " pol=" << pol << " dd_id="
                          << dd_id << "); skipping term (raise max_number_vis / check sizing).\n";
              }
              continue;
            }

            bool uniform_slots = true;
            for (int si = 1; si < nch; ++si) {
              if (slots_sel[static_cast<size_t>(si)] !=
                  slots_sel[static_cast<size_t>(0)]) {
                uniform_slots = false;
                break;
              }
            }
            const int gather_slot =
                uniform_slots ? slots_sel[static_cast<size_t>(0)] : -1;

            std::vector<double3*> h_uvw_ptrs(static_cast<size_t>(nch));
            std::vector<cufftComplex*> h_Vo_ptrs(static_cast<size_t>(nch));
            std::vector<float*> h_weight_ptrs(static_cast<size_t>(nch));
            for (int c = 0; c < nch; c++) {
              const gpuvmem::ms::GPUChunk* ch = chunks_sel[static_cast<size_t>(c)];
              h_uvw_ptrs[static_cast<size_t>(c)] = ch->uvw;
              h_Vo_ptrs[static_cast<size_t>(c)] = ch->Vo;
              h_weight_ptrs[static_cast<size_t>(c)] = ch->weight;
            }

            checkCudaErrors(cudaMemcpy(d_uvw_ptrs[gpu_idx], h_uvw_ptrs.data(),
                                      nch * sizeof(double3*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_Vo_ptrs[gpu_idx], h_Vo_ptrs.data(),
                                      nch * sizeof(cufftComplex*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_weight_ptrs[gpu_idx], h_weight_ptrs.data(),
                                      nch * sizeof(float*),
                                      cudaMemcpyHostToDevice));

            if (uniform_slots) {
              gatherChunkAtOffset<<<(nch + 255) / 256, 256>>>(
                  d_uvw_gather[gpu_idx], d_Vo_gather[gpu_idx],
                  d_weight_gather[gpu_idx],
                  const_cast<double3 const* const*>(d_uvw_ptrs[gpu_idx]),
                  const_cast<cufftComplex const* const*>(d_Vo_ptrs[gpu_idx]),
                  const_cast<float const* const*>(d_weight_ptrs[gpu_idx]),
                  gather_slot, nch);
            } else {
              int* d_slots = nullptr;
              checkCudaErrors(cudaMalloc(&d_slots, static_cast<size_t>(nch) * sizeof(int)));
              checkCudaErrors(cudaMemcpy(d_slots, slots_sel.data(),
                                        static_cast<size_t>(nch) * sizeof(int),
                                        cudaMemcpyHostToDevice));
              gatherChunkAtOffsets<<<(nch + 255) / 256, 256>>>(
                  d_uvw_gather[gpu_idx], d_Vo_gather[gpu_idx],
                  d_weight_gather[gpu_idx],
                  const_cast<double3 const* const*>(d_uvw_ptrs[gpu_idx]),
                  const_cast<cufftComplex const* const*>(d_Vo_ptrs[gpu_idx]),
                  const_cast<float const* const*>(d_weight_ptrs[gpu_idx]),
                  d_slots, nch);
              cudaFree(d_slots);
            }
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
            const gpuvmem::CudaGrid<1> vis1d =
                (variables.blockSizeV >= 0)
                    ? gpuvmem::CudaGrid<1>::from_total(UVpow2,
                                                       variables.blockSizeV)
                    : gpuvmem::CudaGrid<1>::from_auto(UVpow2, chi2_dev_prop);
            if (use_gridding && degrid_kernel) {
              degriddingGPU<<<vis1d.blocks(), vis1d.threads()>>>(
                  d_uvw_gather[gpu_idx], d_Vm_gather[gpu_idx],
                  vars_gpu[gpu_idx].device_V, degrid_kernel->getGPUKernel(),
                  geom.deltau, geom.deltav, nch, Mc, Nc, degrid_kernel->getm(),
                  degrid_kernel->getn(), degrid_kernel->getSupportX(),
                  degrid_kernel->getSupportY());
            } else {
              bilinearInterpolateVisibility<<<vis1d.blocks(), vis1d.threads()>>>(
                  d_Vm_gather[gpu_idx], vars_gpu[gpu_idx].device_V,
                  d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx], geom.deltau, geom.deltav,
                  nch, Mc, Nc, fft_shift);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            residual<<<vis1d.blocks(), vis1d.threads()>>>(
                d_Vr_gather[gpu_idx], d_Vm_gather[gpu_idx], d_Vo_gather[gpu_idx],
                static_cast<long>(nch));
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_chi2, 0,
                                      sizeof(float) * max_number_vis));
            chi2Vector<<<vis1d.blocks(), vis1d.threads()>>>(
                vars_gpu[gpu_idx].device_chi2, d_Vr_gather[gpu_idx],
                d_weight_gather[gpu_idx], static_cast<long>(nch));
            checkCudaErrors(cudaDeviceSynchronize());

            const int threads_v = static_cast<int>(vis1d.threads().x);
            float result = deviceReduce<float>(
                vars_gpu[gpu_idx].device_chi2, nch, threads_v);
            float N_eff = normalize ? static_cast<float>(nch) : 0.0f;
            if (normalize && N_eff > 0.0f) result /= N_eff;
            reduced_chi2 += result;
            }
          }
        }
      }
    }
  }

  cudaSetDevice(firstgpu);
  if (reduced_chi2 <= 0.0f && gpuvmem_cli_verbose()) {
    static int chi2_zero_warn_prints = 0;
    if (chi2_zero_warn_prints < 2) {
      ++chi2_zero_warn_prints;
      size_t tv = 0, tc = 0;
      if (g_datasets && !g_datasets->empty()) {
        tv = (*g_datasets)[0].gpu.total_visibilities();
        tc = (*g_datasets)[0].gpu.total_chunk_count();
      }
      std::cerr << "WARNING: Chi2 chunked sum is zero (reduced_chi2=" << reduced_chi2
                << "). max_number_vis=" << max_number_vis << " dataset0 total_vis=" << tv
                << " total_chunks=" << tc
                << ". Check CORR_TYPE metadata, weights, GPU/host chunk alignment, or "
                   "max_number_vis after MS re-upload.\n";
    }
  }
  return 0.5f * reduced_chi2;
}

__host__ void dchi2_chunked(float* I,
                            float* dxi2,
                            float* result_dchi2,
                            const Image* grid_image,
                            VirtualImageProcessor* ip,
                            bool normalize,
                            float fg_scale) {
  if (!grid_image) return;
  const long Mc = grid_image->getM();
  const long Nc = grid_image->getN();
  const ImagingGeometry geom = grid_image->imaging_geometry();
  bool fft_shift = true;
  cudaSetDevice(firstgpu);
  cudaDeviceProp chi2_dev_prop{};
  checkCudaErrors(cudaGetDeviceProperties(&chi2_dev_prop, firstgpu));

  static PillBox2D* degrid_kernel = NULL;
  static bool degrid_kernel_initialized = false;
  CKernel* ckernel = ip->getCKernel();
  bool use_gridding = (ckernel != NULL && ckernel->getGPUKernel() != NULL);
  if (use_gridding && !degrid_kernel_initialized) {
    degrid_kernel = new PillBox2D(1, 1);
    degrid_kernel->setGPUID(firstgpu);
    degrid_kernel->setSigmas(fabs(geom.deltau), fabs(geom.deltav));
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
  static int dchi2_gather_capacity = 0;
  if (max_number_vis > 0 &&
      (!dchi2_gather_initialized || dchi2_gather_capacity < max_number_vis)) {
    if (dchi2_gather_initialized) {
      for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g + firstgpu);
        if (g < static_cast<int>(d_uvw_gather.size()) && d_uvw_gather[g]) cudaFree(d_uvw_gather[g]);
        if (g < static_cast<int>(d_Vo_gather.size()) && d_Vo_gather[g]) cudaFree(d_Vo_gather[g]);
        if (g < static_cast<int>(d_Vm_gather.size()) && d_Vm_gather[g]) cudaFree(d_Vm_gather[g]);
        if (g < static_cast<int>(d_Vr_gather.size()) && d_Vr_gather[g]) cudaFree(d_Vr_gather[g]);
        if (g < static_cast<int>(d_weight_gather.size()) && d_weight_gather[g])
          cudaFree(d_weight_gather[g]);
        if (g < static_cast<int>(d_uvw_ptrs.size()) && d_uvw_ptrs[g]) cudaFree(d_uvw_ptrs[g]);
        if (g < static_cast<int>(d_Vo_ptrs.size()) && d_Vo_ptrs[g]) cudaFree(d_Vo_ptrs[g]);
        if (g < static_cast<int>(d_weight_ptrs.size()) && d_weight_ptrs[g])
          cudaFree(d_weight_ptrs[g]);
      }
      dchi2_gather_initialized = false;
      dchi2_gather_capacity = 0;
    }
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
    dchi2_gather_capacity = max_number_vis;
    dchi2_gather_initialized = true;
  }

  for (int d = 0; d < nMeasurementSets; d++) {
    gpuvmem::ms::MSWithGPU& dw = (*g_datasets)[d];
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    if (meta.num_antennas() == 0) continue;

    for (size_t f = 0; f < dw.gpu.num_fields(); f++) {
      const gpuvmem::ms::GPUField& gpu_field = dw.gpu.fields()[f];
      const gpuvmem::ms::Field& host_field = host_field_for_gpu_field(dw, f, gpu_field);
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

        std::vector<const gpuvmem::ms::GPUChunk*> chunks_sel;
        std::vector<int> slots_sel;

        for (int chan = 0; chan < nchan; chan++) {
          float nu = static_cast<float>(spw->frequency(chan));
          int gpu_idx = chan % num_gpus;
          cudaSetDevice(gpu_idx + firstgpu);

          const bool stokes_imaging =
              !variables.stokes.empty() && image_count == npol;

          for (size_t bi = 0; bi < gpu_field.baselines.size(); ++bi) {
            const gpuvmem::ms::GPUBaseline& gbl = gpu_field.baselines[bi];
            const gpuvmem::ms::Antenna& a1 = antenna_for_baseline(meta, gbl.antenna1);
            const gpuvmem::ms::Antenna& a2 = antenna_for_baseline(meta, gbl.antenna2);
            const int pb1 = primary_beam_type_int(a1.primary_beam);
            const int pb2 = primary_beam_type_int(a2.primary_beam);

            if (!stokes_imaging) {
              computeImageToVisibilityGridBaseline(
                  {I, grid_image, ip}, fmeta, vars_gpu, gpu_idx, nu,
                  a1.antenna_diameter, a1.pb_factor, a1.pb_cutoff, pb1,
                  a2.antenna_diameter, a2.pb_factor, a2.pb_cutoff, pb2, fg_scale,
                  use_gridding ? degrid_kernel : nullptr, fft_shift);
            }

            for (int pol = 0; pol < npol; pol++) {
              if (!corr_type.empty()) {
                if (pol >= static_cast<int>(corr_type.size())) continue;
                const int ct = corr_type[static_cast<size_t>(pol)];
                if (!gpuvmem::ms::is_circular(ct) && !gpuvmem::ms::is_linear(ct)) continue;
              }

              if (stokes_imaging) {
                float* I_slice = I + static_cast<ptrdiff_t>(pol) * Mc * Nc;
                computeImageToVisibilityGridBaseline(
                    {I_slice, grid_image, ip}, fmeta, vars_gpu, gpu_idx, nu,
                    a1.antenna_diameter, a1.pb_factor, a1.pb_cutoff, pb1,
                    a2.antenna_diameter, a2.pb_factor, a2.pb_cutoff, pb2, fg_scale,
                    use_gridding ? degrid_kernel : nullptr, fft_shift);
              }

              collect_chunks_for_baseline_chan_pol(gpu_field, host_field, bi, dd_id,
                                                   chan, pol, chunks_sel, slots_sel);
            const int nch = static_cast<int>(chunks_sel.size());
            if (nch == 0) continue;
            if (nch > max_number_vis) {
              if (gpuvmem_cli_verbose()) {
                std::cerr << "WARNING: dChi2 gather nch=" << nch << " exceeds max_number_vis="
                          << max_number_vis << " (chan=" << chan << " pol=" << pol << " dd_id="
                          << dd_id << "); skipping.\n";
              }
              continue;
            }

            bool uniform_slots = true;
            for (int si = 1; si < nch; ++si) {
              if (slots_sel[static_cast<size_t>(si)] !=
                  slots_sel[static_cast<size_t>(0)]) {
                uniform_slots = false;
                break;
              }
            }
            const int gather_slot =
                uniform_slots ? slots_sel[static_cast<size_t>(0)] : -1;

            std::vector<double3*> h_uvw_ptrs(static_cast<size_t>(nch));
            std::vector<cufftComplex*> h_Vo_ptrs(static_cast<size_t>(nch));
            std::vector<float*> h_weight_ptrs(static_cast<size_t>(nch));
            for (int c = 0; c < nch; c++) {
              const gpuvmem::ms::GPUChunk* ch = chunks_sel[static_cast<size_t>(c)];
              h_uvw_ptrs[static_cast<size_t>(c)] = ch->uvw;
              h_Vo_ptrs[static_cast<size_t>(c)] = ch->Vo;
              h_weight_ptrs[static_cast<size_t>(c)] = ch->weight;
            }

            checkCudaErrors(cudaMemcpy(d_uvw_ptrs[gpu_idx], h_uvw_ptrs.data(),
                                      nch * sizeof(double3*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_Vo_ptrs[gpu_idx], h_Vo_ptrs.data(),
                                      nch * sizeof(cufftComplex*),
                                      cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_weight_ptrs[gpu_idx], h_weight_ptrs.data(),
                                      nch * sizeof(float*),
                                      cudaMemcpyHostToDevice));

            if (uniform_slots) {
              gatherChunkAtOffset<<<(nch + 255) / 256, 256>>>(
                  d_uvw_gather[gpu_idx], d_Vo_gather[gpu_idx],
                  d_weight_gather[gpu_idx],
                  const_cast<double3 const* const*>(d_uvw_ptrs[gpu_idx]),
                  const_cast<cufftComplex const* const*>(d_Vo_ptrs[gpu_idx]),
                  const_cast<float const* const*>(d_weight_ptrs[gpu_idx]),
                  gather_slot, nch);
            } else {
              int* d_slots = nullptr;
              checkCudaErrors(cudaMalloc(&d_slots, static_cast<size_t>(nch) * sizeof(int)));
              checkCudaErrors(cudaMemcpy(d_slots, slots_sel.data(),
                                        static_cast<size_t>(nch) * sizeof(int),
                                        cudaMemcpyHostToDevice));
              gatherChunkAtOffsets<<<(nch + 255) / 256, 256>>>(
                  d_uvw_gather[gpu_idx], d_Vo_gather[gpu_idx],
                  d_weight_gather[gpu_idx],
                  const_cast<double3 const* const*>(d_uvw_ptrs[gpu_idx]),
                  const_cast<cufftComplex const* const*>(d_Vo_ptrs[gpu_idx]),
                  const_cast<float const* const*>(d_weight_ptrs[gpu_idx]),
                  d_slots, nch);
              cudaFree(d_slots);
            }
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
            const gpuvmem::CudaGrid<1> vis1d =
                (variables.blockSizeV >= 0)
                    ? gpuvmem::CudaGrid<1>::from_total(UVpow2,
                                                       variables.blockSizeV)
                    : gpuvmem::CudaGrid<1>::from_auto(UVpow2, chi2_dev_prop);
            if (use_gridding && degrid_kernel) {
              degriddingGPU<<<vis1d.blocks(), vis1d.threads()>>>(
                  d_uvw_gather[gpu_idx], d_Vm_gather[gpu_idx],
                  vars_gpu[gpu_idx].device_V, degrid_kernel->getGPUKernel(),
                  geom.deltau, geom.deltav, nch, Mc, Nc, degrid_kernel->getm(),
                  degrid_kernel->getn(), degrid_kernel->getSupportX(),
                  degrid_kernel->getSupportY());
            } else {
              bilinearInterpolateVisibility<<<vis1d.blocks(), vis1d.threads()>>>(
                  d_Vm_gather[gpu_idx], vars_gpu[gpu_idx].device_V,
                  d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx], geom.deltau, geom.deltav,
                  nch, Mc, Nc, fft_shift);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            residual<<<vis1d.blocks(), vis1d.threads()>>>(d_Vr_gather[gpu_idx],
                                            d_Vm_gather[gpu_idx],
                                            d_Vo_gather[gpu_idx],
                                            static_cast<long>(nch));
            checkCudaErrors(cudaDeviceSynchronize());

            float N_eff = normalize ? static_cast<float>(nch) : 0.0f;
            if (normalize && N_eff <= 0.0f) N_eff = static_cast<float>(nch);

            checkCudaErrors(cudaMemset(vars_gpu[gpu_idx].device_dchi2, 0,
                                       sizeof(float) * Mc * Nc));
            if (ckernel && ckernel->getGCFGPU()) {
              DChi2Baseline<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, ckernel->getGCFGPU(),
                  vars_gpu[gpu_idx].device_dchi2, d_Vr_gather[gpu_idx],
                  d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx], Nc, nch,
                  fg_scale, noise_cut, fmeta.ref_xobs_pix, fmeta.ref_yobs_pix,
                  fmeta.phs_xobs_pix, fmeta.phs_yobs_pix, geom.delta_x_deg, geom.delta_y_deg,
                  a1.antenna_diameter, a1.pb_factor, a1.pb_cutoff, pb1,
                  a2.antenna_diameter, a2.pb_factor, a2.pb_cutoff, pb2, nu,
                  normalize, N_eff);
            } else {
              DChi2Baseline<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, vars_gpu[gpu_idx].device_dchi2,
                  d_Vr_gather[gpu_idx], d_uvw_gather[gpu_idx], d_weight_gather[gpu_idx],
                  Nc, nch, fg_scale, noise_cut, fmeta.ref_xobs_pix, fmeta.ref_yobs_pix,
                  fmeta.phs_xobs_pix, fmeta.phs_yobs_pix, geom.delta_x_deg, geom.delta_y_deg,
                  a1.antenna_diameter, a1.pb_factor, a1.pb_cutoff, pb1,
                  a2.antenna_diameter, a2.pb_factor, a2.pb_cutoff, pb2, nu,
                  normalize, N_eff);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            if (stokes_imaging) {
              AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(
                  result_dchi2, vars_gpu[gpu_idx].device_dchi2, Nc, Mc, pol);
            } else if (flag_opt == -1) {
              DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, Nc, Mc);
              DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, Nc, Mc);
            } else if (flag_opt % 2 == 0) {
              DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, Nc, Mc);
            } else {
              DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(
                  device_noise_image, result_dchi2, vars_gpu[gpu_idx].device_dchi2,
                  I, nu, nu_0, noise_cut, Nc, Mc);
            }
            checkCudaErrors(cudaDeviceSynchronize());
            }
          }
        }
      }
    }
  }
  cudaSetDevice(firstgpu);
}

__host__ float chi2(float* I,
                    const Image* grid_image,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  if (!use_chi2_chunked_path()) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::cerr << "WARNING: Chi2 returns 0 because the chunked visibility path is off. ";
      if (!g_datasets)
        std::cerr << "g_datasets is null (MFS::configure should set it).\n";
      else if (g_datasets->empty())
        std::cerr << "g_datasets is empty.\n";
      else {
        for (size_t d = 0; d < g_datasets->size(); ++d) {
          if ((*g_datasets)[d].gpu.num_fields() == 0)
            std::cerr << "dataset " << d << " has ChunkedVisibilityGPU.num_fields()==0 "
                         "(upload/gridding did not populate GPU buffers).\n";
        }
      }
    }
    return 0.0f;
  }
  if (!grid_image) {
    if (gpuvmem_cli_verbose())
      std::cerr << "WARNING: chi2: grid_image is null (Chi2::configureImage not called?).\n";
    return 0.0f;
  }
  return chi2_chunked(I, grid_image, ip, normalize, fg_scale);
}

__host__ void dchi2(float* I,
                    float* dxi2,
                    float* result_dchi2,
                    const Image* grid_image,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale) {
  if (!use_chi2_chunked_path())
    return;
  if (!grid_image) {
    if (gpuvmem_cli_verbose())
      std::cerr << "WARNING: dchi2: grid_image is null (Chi2::configureImage not called?).\n";
    return;
  }
  dchi2_chunked(I, dxi2, result_dchi2, grid_image, ip, normalize, fg_scale);
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
