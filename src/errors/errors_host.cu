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
#include "framework.cuh"
#include "ms/ms_with_gpu.h"
#include "ms/polarization.h"
#include "error.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <set>

namespace {

int vis_slot_for_err(const gpuvmem::ms::TimeSample& ts, int chan, int pol) {
  for (size_t vi = 0; vi < ts.visibilities().size(); ++vi) {
    const auto& v = ts.visibilities()[vi];
    if (v.chan == chan && v.pol == pol) return static_cast<int>(vi);
  }
  return -1;
}

void collect_chunks_for_chan_pol_err(const gpuvmem::ms::GPUField& gpu_field,
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
      const int slot = vis_slot_for_err(ts, chan, pol);
      if (slot < 0 || slot >= static_cast<int>(ch.count)) continue;
      chunks_out.push_back(&ch);
      slots_out.push_back(slot);
    }
  }
}

const gpuvmem::ms::Field& host_field_for_gpu_field_err(const gpuvmem::ms::MSWithGPU& dw,
                                                       size_t field_index,
                                                       const gpuvmem::ms::GPUField& gf) {
  if (dw.gridded_ms.num_fields() > field_index) {
    const gpuvmem::ms::Field& cand = dw.gridded_ms.field(field_index);
    if (cand.baselines().size() == gf.baselines.size()) return cand;
  }
  return dw.ms.field(field_index);
}

}  // namespace

// Extern variables
extern long M, N;
extern int num_gpus, firstgpu, nMeasurementSets, max_number_vis, image_count;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern float noise_cut, nu_0;
extern float* device_noise_image;
extern double DELTAX, DELTAY;
extern varsPerGPU* vars_gpu;
extern std::vector<gpuvmem::ms::MSWithGPU>* g_datasets;

// Shared gather buffers (same as chi2_chunked) - declared in chi2_host.cu
// These are in an anonymous namespace in chi2_host.cu, so we need to include
// the chi2_host.cu file or declare them here. For now, we'll declare them
// as extern in a shared header or include chi2_host.cu. Actually, since
// they're in anonymous namespace, we can't access them. Let's move them
// to a shared location or duplicate the initialization.

// For now, we'll use a separate namespace for errors to avoid conflicts
// The gather buffers will be initialized separately for errors if needed
namespace errors_gather {
std::vector<double3*> d_uvw_gather;
std::vector<cufftComplex*> d_Vo_gather;
std::vector<float*> d_weight_gather;
std::vector<double3**> d_uvw_ptrs;
std::vector<cufftComplex**> d_Vo_ptrs;
std::vector<float**> d_weight_ptrs;
bool gather_buffers_initialized = false;
int gather_buffer_capacity = 0;

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
  d_weight_gather.resize(num_gpus);
  d_uvw_ptrs.resize(num_gpus);
  d_Vo_ptrs.resize(num_gpus);
  d_weight_ptrs.resize(num_gpus);
  for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice(g + firstgpu);
    checkCudaErrors(cudaMalloc(&d_uvw_gather[g], max_number_vis * sizeof(double3)));
    checkCudaErrors(cudaMalloc(&d_Vo_gather[g], max_number_vis * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_weight_gather[g], max_number_vis * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_uvw_ptrs[g], max_number_vis * sizeof(double3*)));
    checkCudaErrors(cudaMalloc(&d_Vo_ptrs[g], max_number_vis * sizeof(cufftComplex*)));
    checkCudaErrors(cudaMalloc(&d_weight_ptrs[g], max_number_vis * sizeof(float*)));
  }
  gather_buffer_capacity = max_number_vis;
  gather_buffers_initialized = true;
}
}  // namespace errors_gather

static bool use_chi2_chunked_path() {
  if (!g_datasets || g_datasets->empty()) return false;
  for (size_t d = 0; d < g_datasets->size(); d++)
    if ((*g_datasets)[d].gpu.num_fields() == 0) return false;
  return true;
}

__host__ void calculateErrors_chunked(Image* image, float fg_scale) {
  float* errors = image->getErrorImage();
  cudaSetDevice(firstgpu);

  const int image_count_local = image->getImageCount();
  const bool stokes_imaging = (image_count_local != 2);
  int error_image_count =
      stokes_imaging ? image_count_local : (image_count_local + 2);  // MFS: +covariance, +ρ
  checkCudaErrors(
      cudaMalloc((void**)&errors, sizeof(float) * M * N * error_image_count));
  checkCudaErrors(
      cudaMemset(errors, 0, sizeof(float) * M * N * error_image_count));

  errors_gather::ensure_gather_buffers();

  for (int d = 0; d < nMeasurementSets; d++) {
    gpuvmem::ms::MSWithGPU& dw = (*g_datasets)[d];
    const gpuvmem::ms::MeasurementSetMetadata& meta = dw.ms.metadata();
    if (meta.num_antennas() == 0) continue;
    const gpuvmem::ms::Antenna& ant0 = meta.antenna(0);
    int primary_beam_int =
        (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0;

    for (size_t f = 0; f < dw.gpu.num_fields(); f++) {
      const gpuvmem::ms::GPUField& gpu_field = dw.gpu.fields()[f];
      const gpuvmem::ms::Field& host_field = host_field_for_gpu_field_err(dw, f, gpu_field);
      const gpuvmem::ms::FieldMetadata& fmeta = host_field.metadata();
      float ref_xobs = fmeta.ref_xobs_pix;
      float ref_yobs = fmeta.ref_yobs_pix;

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

          for (int pol = 0; pol < npol; pol++) {
            if (!corr_type.empty()) {
              if (pol >= static_cast<int>(corr_type.size())) continue;
              const int ct = corr_type[static_cast<size_t>(pol)];
              if (!gpuvmem::ms::is_circular(ct) && !gpuvmem::ms::is_linear(ct)) continue;
            }

            collect_chunks_for_chan_pol_err(gpu_field, host_field, dd_id, chan, pol,
                                            chunks_sel, slots_sel);
            const int nch = static_cast<int>(chunks_sel.size());
            if (nch == 0) continue;
            if (nch > max_number_vis) continue;

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

            std::vector<float*> h_weight_ptrs(static_cast<size_t>(nch));
            std::vector<double3*> h_uvw_ptrs(static_cast<size_t>(nch));
            std::vector<cufftComplex*> h_Vo_ptrs(static_cast<size_t>(nch));
            for (int c = 0; c < nch; c++) {
              const gpuvmem::ms::GPUChunk* ch = chunks_sel[static_cast<size_t>(c)];
              h_weight_ptrs[static_cast<size_t>(c)] = const_cast<float*>(ch->weight);
              h_uvw_ptrs[static_cast<size_t>(c)] = const_cast<double3*>(ch->uvw);
              h_Vo_ptrs[static_cast<size_t>(c)] = const_cast<cufftComplex*>(ch->Vo);
            }

            checkCudaErrors(cudaMemcpy(errors_gather::d_weight_ptrs[gpu_idx], h_weight_ptrs.data(),
                                        nch * sizeof(float*),
                                        cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(errors_gather::d_uvw_ptrs[gpu_idx], h_uvw_ptrs.data(),
                                        nch * sizeof(double3*),
                                        cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(errors_gather::d_Vo_ptrs[gpu_idx], h_Vo_ptrs.data(),
                                        nch * sizeof(cufftComplex*),
                                        cudaMemcpyHostToDevice));

            if (uniform_slots) {
              gatherChunkAtOffset<<<(nch + 255) / 256, 256>>>(
                  errors_gather::d_uvw_gather[gpu_idx], errors_gather::d_Vo_gather[gpu_idx],
                  errors_gather::d_weight_gather[gpu_idx],
                  const_cast<double3 const* const*>(errors_gather::d_uvw_ptrs[gpu_idx]),
                  const_cast<cufftComplex const* const*>(errors_gather::d_Vo_ptrs[gpu_idx]),
                  const_cast<float const* const*>(errors_gather::d_weight_ptrs[gpu_idx]),
                  gather_slot, nch);
            } else {
              int* d_slots = nullptr;
              checkCudaErrors(cudaMalloc(&d_slots, static_cast<size_t>(nch) * sizeof(int)));
              checkCudaErrors(cudaMemcpy(d_slots, slots_sel.data(),
                                        static_cast<size_t>(nch) * sizeof(int),
                                        cudaMemcpyHostToDevice));
              gatherChunkAtOffsets<<<(nch + 255) / 256, 256>>>(
                  errors_gather::d_uvw_gather[gpu_idx], errors_gather::d_Vo_gather[gpu_idx],
                  errors_gather::d_weight_gather[gpu_idx],
                  const_cast<double3 const* const*>(errors_gather::d_uvw_ptrs[gpu_idx]),
                  const_cast<cufftComplex const* const*>(errors_gather::d_Vo_ptrs[gpu_idx]),
                  const_cast<float const* const*>(errors_gather::d_weight_ptrs[gpu_idx]),
                  d_slots, nch);
              cudaFree(d_slots);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            float sum_weights = deviceReduce<float>(errors_gather::d_weight_gather[gpu_idx],
                                                    static_cast<long>(nch), 256);

            if (stokes_imaging) {
              if (pol < image_count_local) {
                stokes_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                    errors, pol, nu, device_noise_image, noise_cut,
                    ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                    ref_xobs, ref_yobs, DELTAX, DELTAY,
                    sum_weights, fg_scale, N, M, primary_beam_int);
                checkCudaErrors(cudaDeviceSynchronize());
              }
            } else {
              I_nu_0_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                  errors, image->getImage(), device_noise_image, noise_cut,
                  nu, nu_0, errors_gather::d_weight_gather[gpu_idx],
                  ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                  ref_xobs, ref_yobs, DELTAX, DELTAY,
                  sum_weights, fg_scale, N, M, primary_beam_int);
              checkCudaErrors(cudaDeviceSynchronize());

              alpha_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                  errors, image->getImage(), nu, nu_0, device_noise_image, noise_cut,
                  DELTAX, DELTAY, ref_xobs, ref_yobs,
                  ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                  sum_weights, fg_scale, N, M, primary_beam_int);
              checkCudaErrors(cudaDeviceSynchronize());

              covariance_Noise<<<numBlocksNN, threadsPerBlockNN>>>(
                  errors, image->getImage(), nu, nu_0, device_noise_image, noise_cut,
                  DELTAX, DELTAY, ref_xobs, ref_yobs,
                  ant0.antenna_diameter, ant0.pb_factor, ant0.pb_cutoff,
                  sum_weights, fg_scale, N, M, primary_beam_int);
              checkCudaErrors(cudaDeviceSynchronize());
            }
          }
        }
      }
    }
  }

  if (stokes_imaging)
    stokes_noise_reduction<<<numBlocksNN, threadsPerBlockNN>>>(
        errors, image_count_local, N, M);
  else
    noise_reduction<<<numBlocksNN, threadsPerBlockNN>>>(errors, N, M);
  checkCudaErrors(cudaDeviceSynchronize());
  image->setErrorImage(errors);
}

__host__ void calculateErrors(Image* image, float fg_scale) {
  if (!use_chi2_chunked_path())
    return;
  calculateErrors_chunked(image, fg_scale);
}

__host__ void precomputeNeff(bool normalize) {
  if (!normalize) {
    return;
  }
  // N_eff is handled in chi2_chunked path; legacy data/fields removed from
  // MSWithGPU so no legacy precompute.
}
