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
#include "framework.cuh"
#include "ms/measurement_set.h"
#include "ms/gpu_buffers.h"
#include "utils/physics_utils.cuh"
#include "utils/physics_utils.cuh"
#include "utils/complexOps.cuh"
#include "measurement_operator/measurement_operator_host.cuh"
#include "framework/cuda_grid.cuh"
#include "classes/image.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>
#include <omp.h>

// Extern variables
extern Vars variables;
extern varsPerGPU* vars_gpu;
extern int image_count;
extern long M, N;
extern dim3 threadsPerBlockNN, numBlocksNN;
extern int num_gpus, firstgpu;
extern unsigned int NearestPowerOf2(unsigned int x);

// Scatter degridded values (one per chunk) to chunk.Vm; ptrs point to chan start.
__global__ void scatter_Vm_to_chunks(cufftComplex* Vm_out,
                                     cufftComplex** chunk_Vm_ptrs,
                                     int npol,
                                     int num_chunks) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_chunks) return;
  cufftComplex v = Vm_out[i];
  cufftComplex* dst = chunk_Vm_ptrs[i];
  for (int p = 0; p < npol; p++) dst[p] = v;
}

__host__ gpuvmem::ms::MeasurementSet do_gridding(
    gpuvmem::ms::MeasurementSet& ms,
    gpuvmem::ms::ChunkedVisibilityGPU* gpu,
    double deltau,
    double deltav,
    long M,
    long N,
    CKernel* ckernel,
    int gridding) {
  gpuvmem::ms::MeasurementSet gridded_ms(ms.name() + ".gridded");
  if (!ckernel || !gpu) return gridded_ms;
  gridded_ms.metadata() = ms.metadata();
  const gpuvmem::ms::MeasurementSetMetadata& meta = ms.metadata();
  const double center_j = floor(N / 2.0);
  const double center_k = floor(M / 2.0);
  const int support_x = ckernel->getSupportX();
  const int support_y = ckernel->getSupportY();

  std::vector<float> g_weights(static_cast<size_t>(M * N));
  std::vector<float> g_weights_aux(static_cast<size_t>(M * N));
  std::vector<cufftComplex> g_Vo(static_cast<size_t>(M * N));
  std::vector<double3> g_uvw(static_cast<size_t>(M * N));
  const cufftComplex complex_zero = floatComplexZero();
  const double3 double3_zero = {0.0, 0.0, 0.0};

  for (size_t f = 0; f < ms.num_fields(); f++) {
    const gpuvmem::ms::Field& host_field = ms.field(f);
    gpuvmem::ms::Field& gfield = gridded_ms.add_field(host_field.metadata());
    gpuvmem::ms::Baseline& gbl = gfield.baseline(0, 0);

    std::set<int> dd_ids;
    for (const gpuvmem::ms::Baseline& bl : host_field.baselines()) {
      for (const gpuvmem::ms::TimeSample& ts : bl.time_samples())
        dd_ids.insert(ts.data_desc_id());
    }

    for (int dd_id : dd_ids) {
      const gpuvmem::ms::DataDescription* dd =
          meta.find_data_description(dd_id);
      if (!dd) continue;
      const gpuvmem::ms::SpectralWindow* spw =
          meta.find_spectral_window(dd->spectral_window_id());
      if (!spw) continue;
      const int nchan = dd->nchan();
      const int npol = dd->npol();
      if (nchan <= 0 || npol <= 0) continue;

      for (int chan = 0; chan < nchan; chan++) {
        const float freq = static_cast<float>(spw->frequency(chan));
        const float lambda = freq_to_wavelength(freq);

        for (int pol = 0; pol < npol; pol++) {
          std::vector<double3> uvw_list;
          std::vector<cufftComplex> Vo_list;
          std::vector<float> weight_list;

          for (const gpuvmem::ms::Baseline& bl : host_field.baselines()) {
            for (const gpuvmem::ms::TimeSample& ts : bl.time_samples()) {
              if (ts.data_desc_id() != dd_id) continue;
              for (const auto& v : ts.visibilities()) {
                if (v.chan == chan && v.pol == pol) {
                  uvw_list.push_back(ts.uvw());
                  Vo_list.push_back(v.Vo);
                  weight_list.push_back(v.imaging_weight * 0.5f);
                  break;
                }
              }
            }
          }

          const int num_vis = static_cast<int>(uvw_list.size());
          if (num_vis == 0) continue;

          std::fill(g_weights_aux.begin(), g_weights_aux.end(), 0.0f);
          std::fill(g_weights.begin(), g_weights.end(), 0.0f);
          std::fill(g_uvw.begin(), g_uvw.end(), double3_zero);
          std::fill(g_Vo.begin(), g_Vo.end(), complex_zero);

#pragma omp parallel for schedule(static) num_threads(gridding > 0 ? gridding : 1) \
    shared(g_weights, g_weights_aux, g_Vo, center_j, center_k, support_x, support_y)
          for (int z = 0; z < num_vis; z++) {
            const double3 uvw = uvw_list[z];
            const float w = weight_list[z];
            const cufftComplex Vo = Vo_list[z];
            const double u_lambda = metres_to_lambda(uvw.x, freq);
            const double v_lambda = metres_to_lambda(uvw.y, freq);

            for (int h = 0; h < 2; h++) {
              const double u_pos = (h == 0) ? u_lambda : -u_lambda;
              const double v_pos = (h == 0) ? v_lambda : -v_lambda;
              const float Vo_imag = (h == 0) ? Vo.y : -Vo.y;
              const double grid_pos_x = u_pos / deltau;
              const double grid_pos_y = v_pos / deltav;
              const int j = static_cast<int>(grid_pos_x + center_j + 0.5);
              const int k = static_cast<int>(grid_pos_y + center_k + 0.5);

              for (int m = -support_y; m <= support_y; m++) {
                for (int n = -support_x; n <= support_x; n++) {
                  const int shifted_j = j + n;
                  const int shifted_k = k + m;
                  const int kernel_j = n + support_x;
                  const int kernel_i = m + support_y;
                  if (shifted_k >= 0 && shifted_k < M && shifted_j >= 0 &&
                      shifted_j < N && kernel_i >= 0 &&
                      kernel_i < ckernel->getm() && kernel_j >= 0 &&
                      kernel_j < ckernel->getn()) {
                    const float ckernel_result =
                        ckernel->getKernelValue(kernel_i, kernel_j);
                    const float ckernel_result_sq =
                        ckernel_result * ckernel_result;
                    const int grid_idx =
                        static_cast<int>(N * shifted_k + shifted_j);
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

#pragma omp parallel for schedule(static) shared(g_weights, g_weights_aux, g_Vo, g_uvw, lambda, center_j, center_k)
          for (int grid_k = 0; grid_k < M; grid_k++) {
            for (int grid_j = 0; grid_j < N; grid_j++) {
              const int grid_idx = N * grid_k + grid_j;
              const double u_lambdas = (grid_j - center_j) * deltau;
              const double v_lambdas = (grid_k - center_k) * deltav;
              const double u_meters = u_lambdas * lambda;
              const double v_meters = v_lambdas * lambda;
              g_uvw[grid_idx].x = u_meters;
              g_uvw[grid_idx].y = v_meters;
              g_uvw[grid_idx].z = 0.0;

              const float ws = g_weights[grid_idx];
              const float aux_ws = g_weights_aux[grid_idx];
              if (aux_ws != 0.0f && ws != 0.0f) {
                const float weight_eff = ws * ws / aux_ws;
                g_Vo[grid_idx].x /= ws;
                g_Vo[grid_idx].y /= ws;
                g_weights[grid_idx] = weight_eff;
              } else {
                g_weights[grid_idx] = 0.0f;
                g_Vo[grid_idx].x = 0.0f;
                g_Vo[grid_idx].y = 0.0f;
              }
            }
          }

          for (int grid_k = 0; grid_k < M; grid_k++) {
            for (int grid_j = 0; grid_j < N; grid_j++) {
              const int grid_idx = N * grid_k + grid_j;
              const float weight = g_weights[grid_idx];
              if (weight > 0.0f) {
                gpuvmem::ms::TimeSample ts(dd_id, 0.0);
                ts.set_uvw(g_uvw[grid_idx]);
                ts.add_visibility(
                    chan, pol,
                    make_cuFloatComplex(g_Vo[grid_idx].x, g_Vo[grid_idx].y),
                    cufftComplex{0.f, 0.f}, cufftComplex{0.f, 0.f}, weight);
                gbl.add_time_sample(std::move(ts));
              }
            }
          }
        }
      }
    }
  }

  gpu->upload(gridded_ms);
  return gridded_ms;
}

__host__ void do_degridding(gpuvmem::ms::MeasurementSet& ms,
                            gpuvmem::ms::ChunkedVisibilityGPU* gpu,
                            int num_gpus,
                            int firstgpu,
                            int blockSizeV,
                            CKernel* ckernel,
                            float* I,
                            VirtualImageProcessor* ip,
                            const Image* grid_image) {
  if (!gpu || !ckernel || !I || !ip || !grid_image) return;
  if (gpu->num_fields() == 0) {
    if (!gpu->upload(ms)) return;
  }
  if (gpu->num_fields() == 0) return;

  const long Mc = grid_image->getM();
  const long Nc = grid_image->getN();
  const ImagingGeometry geo = grid_image->imaging_geometry();
  const double deltau = geo.deltau;
  const double deltav = geo.deltav;

  const gpuvmem::ms::MeasurementSetMetadata& meta = ms.metadata();
  if (meta.num_antennas() == 0) return;

  const gpuvmem::ms::Antenna& ant0 = meta.antenna(0);
  int primary_beam_int =
      (ant0.primary_beam == gpuvmem::ms::PrimaryBeamType::AiryDisk) ? 1 : 0;
  const float ant_diam = ant0.antenna_diameter;
  const float pb_factor = ant0.pb_factor;
  const float pb_cutoff = ant0.pb_cutoff;

  long UVpow2;
  bool fft_shift = true;
  int slot = 0;

  for (size_t f = 0; f < gpu->num_fields(); f++) {
    const gpuvmem::ms::GPUField& gpu_field = gpu->fields()[f];
    const gpuvmem::ms::Field& host_field = ms.field(f);
    const gpuvmem::ms::FieldMetadata& fmeta = host_field.metadata();

    std::set<int> dd_ids;
    for (const auto& bl : gpu_field.baselines)
      for (const auto& ch : bl.chunks) dd_ids.insert(ch.data_desc_id);

    for (int dd_id : dd_ids) {
      const gpuvmem::ms::DataDescription* dd =
          meta.find_data_description(dd_id);
      if (!dd) continue;
      const gpuvmem::ms::SpectralWindow& spw =
          meta.spectral_window(dd->spectral_window_id());
      int nchan = dd->nchan();
      int npol = dd->npol();
      if (nchan <= 0 || npol <= 0) continue;

      const bool stokes_imaging =
          !variables.stokes.empty() && image_count == npol;

      for (int chan = 0; chan < nchan; chan++) {
        float nu = static_cast<float>(spw.frequency(chan));
        int gpu_idx = (slot++) % num_gpus;
        cudaSetDevice(gpu_idx + firstgpu);

        std::vector<const gpuvmem::ms::GPUChunk*> chunks;
        for (const auto& bl : gpu_field.baselines)
          for (const auto& ch : bl.chunks)
            if (ch.data_desc_id == dd_id && ch.count > 0) chunks.push_back(&ch);

        if (chunks.empty()) continue;

        size_t num_chunks = chunks.size();
        std::vector<double3> h_uvw_m(num_chunks);
        for (size_t c = 0; c < num_chunks; c++) {
          checkCudaErrors(cudaMemcpy(
              &h_uvw_m[c], chunks[c]->uvw, sizeof(double3),
              cudaMemcpyDeviceToHost));
        }
        std::vector<double3> h_uvw_lambda(num_chunks);
        for (size_t c = 0; c < num_chunks; c++) {
          h_uvw_lambda[c].x = metres_to_lambda(h_uvw_m[c].x, nu);
          h_uvw_lambda[c].y = metres_to_lambda(h_uvw_m[c].y, nu);
          h_uvw_lambda[c].z = metres_to_lambda(h_uvw_m[c].z, nu);
        }

        double3* d_uvw = nullptr;
        checkCudaErrors(
            cudaMalloc(&d_uvw, num_chunks * sizeof(double3)));
        checkCudaErrors(cudaMemcpy(d_uvw, h_uvw_lambda.data(),
                                   num_chunks * sizeof(double3),
                                   cudaMemcpyHostToDevice));

        int nvis = static_cast<int>(num_chunks);
        UVpow2 = static_cast<long>(NearestPowerOf2(static_cast<unsigned int>(nvis)));
        cudaDeviceProp dev_prop{};
        checkCudaErrors(
            cudaGetDeviceProperties(&dev_prop, gpu_idx + firstgpu));
        const gpuvmem::CudaGrid<1> vis1d =
            (blockSizeV >= 0)
                ? gpuvmem::CudaGrid<1>::from_total(UVpow2, blockSizeV)
                : gpuvmem::CudaGrid<1>::from_auto(UVpow2, dev_prop);
        const dim3 blocks_vis = vis1d.blocks();
        const dim3 threads_vis = vis1d.threads();

        if (stokes_imaging) {
          for (int pol = 0; pol < npol; pol++) {
            float* I_slice = I + static_cast<ptrdiff_t>(pol) * Mc * Nc;
            computeImageToVisibilityGridBaseline(
                {I_slice, grid_image, ip}, fmeta, vars_gpu, gpu_idx, nu,
                ant_diam, pb_factor, pb_cutoff, primary_beam_int, ant_diam, pb_factor,
                pb_cutoff, primary_beam_int, 1.0f, ckernel, fft_shift);

            cufftComplex* d_Vm_out = nullptr;
            checkCudaErrors(
                cudaMalloc(&d_Vm_out, num_chunks * sizeof(cufftComplex)));
            degriddingGPU<<<blocks_vis, threads_vis>>>(
                d_uvw, d_Vm_out, vars_gpu[gpu_idx].device_V, ckernel->getGPUKernel(),
                deltau, deltav, nvis, Mc, Nc, ckernel->getm(), ckernel->getn(),
                ckernel->getSupportX(), ckernel->getSupportY());
            checkCudaErrors(cudaDeviceSynchronize());

            int offset = chan * npol + pol;
            std::vector<cufftComplex*> h_ptrs(num_chunks);
            for (size_t c = 0; c < num_chunks; c++)
              h_ptrs[c] = chunks[c]->Vm + offset;
            cufftComplex** d_ptrs = nullptr;
            checkCudaErrors(
                cudaMalloc(&d_ptrs, num_chunks * sizeof(cufftComplex*)));
            checkCudaErrors(cudaMemcpy(d_ptrs, h_ptrs.data(),
                                       num_chunks * sizeof(cufftComplex*),
                                       cudaMemcpyHostToDevice));
            scatter_Vm_to_chunks<<<(num_chunks + 255) / 256, 256>>>(
                d_Vm_out, d_ptrs, 1, static_cast<int>(num_chunks));
            checkCudaErrors(cudaDeviceSynchronize());

            cudaFree(d_Vm_out);
            cudaFree(d_ptrs);
          }
        } else {
          computeImageToVisibilityGridBaseline(
              {I, grid_image, ip}, fmeta, vars_gpu, gpu_idx, nu, ant_diam, pb_factor,
              pb_cutoff, primary_beam_int, ant_diam, pb_factor, pb_cutoff,
              primary_beam_int, 1.0f, ckernel, fft_shift);

          cufftComplex* d_Vm_out = nullptr;
          checkCudaErrors(
              cudaMalloc(&d_Vm_out, num_chunks * sizeof(cufftComplex)));
          degriddingGPU<<<blocks_vis, threads_vis>>>(
              d_uvw, d_Vm_out, vars_gpu[gpu_idx].device_V, ckernel->getGPUKernel(),
              deltau, deltav, nvis, Mc, Nc, ckernel->getm(), ckernel->getn(),
              ckernel->getSupportX(), ckernel->getSupportY());
          checkCudaErrors(cudaDeviceSynchronize());

          int offset = chan * npol;
          std::vector<cufftComplex*> h_ptrs(num_chunks);
          for (size_t c = 0; c < num_chunks; c++)
            h_ptrs[c] = chunks[c]->Vm + offset;
          cufftComplex** d_ptrs = nullptr;
          checkCudaErrors(
              cudaMalloc(&d_ptrs, num_chunks * sizeof(cufftComplex*)));
          checkCudaErrors(cudaMemcpy(d_ptrs, h_ptrs.data(),
                                     num_chunks * sizeof(cufftComplex*),
                                     cudaMemcpyHostToDevice));

          scatter_Vm_to_chunks<<<(num_chunks + 255) / 256, 256>>>(
              d_Vm_out, d_ptrs, npol, static_cast<int>(num_chunks));
          checkCudaErrors(cudaDeviceSynchronize());

          cudaFree(d_Vm_out);
          cudaFree(d_ptrs);
        }

        cudaFree(d_uvw);
      }
    }
  }

  gpu->compute_residuals();
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
