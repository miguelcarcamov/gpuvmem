/* GPU buffers for visibility data: upload/download to MeasurementSet. */
#include "ms/gpu_buffers.h"
#include "ms/visibility_model.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

namespace gpuvmem {
namespace ms {

VisibilityGPUBuffers::~VisibilityGPUBuffers() { clear(); }

void VisibilityGPUBuffers::clear() {
  if (uvw_) { cudaFree(uvw_); uvw_ = nullptr; }
  if (weight_) { cudaFree(weight_); weight_ = nullptr; }
  if (Vo_) { cudaFree(Vo_); Vo_ = nullptr; }
  if (Vm_) { cudaFree(Vm_); Vm_ = nullptr; }
  if (Vr_) { cudaFree(Vr_); Vr_ = nullptr; }
  nvis_ = 0;
  row_mapping_.clear();
}

bool VisibilityGPUBuffers::upload(const MeasurementSet& ms) {
  row_mapping_.clear();
  size_t nvis = 0;
  for (size_t f = 0; f < ms.num_fields(); f++) {
    const Field& field = ms.field(f);
    for (const Baseline& bl : field.baselines()) {
      for (size_t t = 0; t < bl.time_samples().size(); t++) {
        const TimeSample& ts = bl.time_samples()[t];
        nvis += ts.visibilities().size();
        for (size_t vi = 0; vi < ts.visibilities().size(); vi++) {
          RowMapping r;
          r.field_id = field.field_id();
          r.antenna1 = bl.antenna1();
          r.antenna2 = bl.antenna2();
          r.data_desc_id = ts.data_desc_id();
          r.time_sample_index = t;
          r.vis_index_in_sample = vi;
          row_mapping_.push_back(r);
        }
      }
    }
  }
  if (nvis == 0) { clear(); return true; }

  if (nvis != nvis_) {
    clear();
    nvis_ = nvis;
    if (cudaMalloc(&uvw_, nvis_ * sizeof(double3)) != cudaSuccess ||
        cudaMalloc(&weight_, nvis_ * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&Vo_, nvis_ * sizeof(cufftComplex)) != cudaSuccess ||
        cudaMalloc(&Vm_, nvis_ * sizeof(cufftComplex)) != cudaSuccess ||
        cudaMalloc(&Vr_, nvis_ * sizeof(cufftComplex)) != cudaSuccess) {
      clear();
      return false;
    }
  }

  std::vector<double3> h_uvw(nvis_);
  std::vector<float> h_weight(nvis_);
  std::vector<cufftComplex> h_Vo(nvis_), h_Vm(nvis_), h_Vr(nvis_);
  size_t idx = 0;
  for (size_t f = 0; f < ms.num_fields(); f++) {
    const Field& field = ms.field(f);
    for (const Baseline& bl : field.baselines()) {
      for (const TimeSample& ts : bl.time_samples()) {
        double3 u = ts.uvw();
        const auto& w = ts.weight();
        const auto& vis = ts.visibilities();
        for (size_t vi = 0; vi < vis.size(); vi++) {
          h_uvw[idx] = u;
          h_weight[idx] = vis[vi].imaging_weight;
          h_Vo[idx] = vis[vi].Vo;
          h_Vm[idx] = vis[vi].Vm;
          h_Vr[idx] = vis[vi].Vr;
          idx++;
        }
      }
    }
  }

  if (cudaMemcpy(uvw_, h_uvw.data(), nvis_ * sizeof(double3), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(weight_, h_weight.data(), nvis_ * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(Vo_, h_Vo.data(), nvis_ * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(Vm_, h_Vm.data(), nvis_ * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(Vr_, h_Vr.data(), nvis_ * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess) {
    return false;
  }
  return true;
}

bool VisibilityGPUBuffers::download(MeasurementSet* ms) {
  if (!ms || nvis_ == 0 || row_mapping_.size() != nvis_) return false;
  std::vector<cufftComplex> h_Vm(nvis_), h_Vr(nvis_);
  if (cudaMemcpy(h_Vm.data(), Vm_, nvis_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(h_Vr.data(), Vr_, nvis_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  size_t idx = 0;
  for (size_t f = 0; f < ms->num_fields(); f++) {
    Field& field = ms->field(f);
    for (Baseline& bl : field.baselines()) {
      for (size_t t = 0; t < bl.time_samples().size(); t++) {
        TimeSample& ts = bl.time_samples()[t];
        auto& vis = ts.visibilities();
        for (size_t vi = 0; vi < vis.size() && idx < nvis_; vi++, idx++) {
          vis[vi].Vm = h_Vm[idx];
          vis[vi].Vr = h_Vr[idx];
        }
      }
    }
  }
  return true;
}

// -----------------------------------------------------------------------------
// ChunkedVisibilityGPU
// -----------------------------------------------------------------------------

ChunkedVisibilityGPU::~ChunkedVisibilityGPU() { clear(); }

void ChunkedVisibilityGPU::clear() {
  if (uvw_base_) { cudaFree(uvw_base_); uvw_base_ = nullptr; }
  if (weight_base_) { cudaFree(weight_base_); weight_base_ = nullptr; }
  if (Vo_base_) { cudaFree(Vo_base_); Vo_base_ = nullptr; }
  if (Vm_base_) { cudaFree(Vm_base_); Vm_base_ = nullptr; }
  if (Vr_base_) { cudaFree(Vr_base_); Vr_base_ = nullptr; }
  total_vis_ = 0;
  fields_.clear();
}

bool ChunkedVisibilityGPU::upload(const MeasurementSet& ms) {
  clear();
  size_t nvis = 0;
  for (size_t f = 0; f < ms.num_fields(); f++) {
    const Field& field = ms.field(f);
    for (const Baseline& bl : field.baselines()) {
      for (const TimeSample& ts : bl.time_samples())
        nvis += ts.visibilities().size();
    }
  }
  if (nvis == 0) return true;

  total_vis_ = nvis;
  if (cudaMalloc(&uvw_base_, nvis * sizeof(double3)) != cudaSuccess ||
      cudaMalloc(&weight_base_, nvis * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&Vo_base_, nvis * sizeof(cufftComplex)) != cudaSuccess ||
      cudaMalloc(&Vm_base_, nvis * sizeof(cufftComplex)) != cudaSuccess ||
      cudaMalloc(&Vr_base_, nvis * sizeof(cufftComplex)) != cudaSuccess) {
    clear();
    return false;
  }

  std::vector<double3> h_uvw(nvis);
  std::vector<float> h_weight(nvis);
  std::vector<cufftComplex> h_Vo(nvis), h_Vm(nvis), h_Vr(nvis);
  size_t global_idx = 0;

  for (size_t f = 0; f < ms.num_fields(); f++) {
    const Field& field = ms.field(f);
    GPUField gpu_field;
    gpu_field.field_id = field.field_id();

    for (const Baseline& bl : field.baselines()) {
      GPUBaseline gpu_bl;
      gpu_bl.antenna1 = bl.antenna1();
      gpu_bl.antenna2 = bl.antenna2();

      for (const TimeSample& ts : bl.time_samples()) {
        const auto& vis = ts.visibilities();
        double3 u = ts.uvw();
        size_t chunk_start = global_idx;
        for (size_t vi = 0; vi < vis.size(); vi++) {
          h_uvw[global_idx] = u;
          h_weight[global_idx] = vis[vi].imaging_weight;
          h_Vo[global_idx] = vis[vi].Vo;
          h_Vm[global_idx] = vis[vi].Vm;
          h_Vr[global_idx] = vis[vi].Vr;
          global_idx++;
        }
        GPUChunk ch;
        ch.uvw = uvw_base_ + chunk_start;
        ch.weight = weight_base_ + chunk_start;
        ch.Vo = Vo_base_ + chunk_start;
        ch.Vm = Vm_base_ + chunk_start;
        ch.Vr = Vr_base_ + chunk_start;
        ch.count = vis.size();
        ch.data_desc_id = ts.data_desc_id();
        ch.time = ts.time();
        gpu_bl.chunks.push_back(ch);
      }
      gpu_field.baselines.push_back(std::move(gpu_bl));
    }
    fields_.push_back(std::move(gpu_field));
  }

  if (cudaMemcpy(uvw_base_, h_uvw.data(), nvis * sizeof(double3), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(weight_base_, h_weight.data(), nvis * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(Vo_base_, h_Vo.data(), nvis * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(Vm_base_, h_Vm.data(), nvis * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(Vr_base_, h_Vr.data(), nvis * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess) {
    clear();
    return false;
  }
  return true;
}

__global__ static void compute_Vr_kernel(const cufftComplex* Vo,
                                         const cufftComplex* Vm,
                                         cufftComplex* Vr,
                                         size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  cufftComplex vo = Vo[i];
  cufftComplex vm = Vm[i];
  Vr[i].x = vo.x - vm.x;
  Vr[i].y = vo.y - vm.y;
}

void ChunkedVisibilityGPU::compute_residuals() {
  if (total_vis_ == 0 || !Vo_base_ || !Vm_base_ || !Vr_base_) return;
  const int block = 256;
  int grid = static_cast<int>((total_vis_ + block - 1) / block);
  if (grid <= 0) return;
  compute_Vr_kernel<<<grid, block>>>(Vo_base_, Vm_base_, Vr_base_, total_vis_);
  cudaDeviceSynchronize();
}

void ChunkedVisibilityGPU::zero_model_and_residual() {
  if (total_vis_ == 0 || !Vm_base_ || !Vr_base_) return;
  cudaMemset(Vm_base_, 0, total_vis_ * sizeof(cufftComplex));
  cudaMemset(Vr_base_, 0, total_vis_ * sizeof(cufftComplex));
}

size_t ChunkedVisibilityGPU::max_chunk_count() const {
  size_t m = 0;
  for (const GPUField& gf : fields_) {
    for (const GPUBaseline& gb : gf.baselines) {
      for (const GPUChunk& ch : gb.chunks) {
        if (ch.count > m) m = ch.count;
      }
    }
  }
  return m;
}

bool ChunkedVisibilityGPU::download(MeasurementSet* ms) {
  if (!ms || total_vis_ == 0 || fields_.size() != ms->num_fields()) return false;
  std::vector<cufftComplex> h_Vm(total_vis_), h_Vr(total_vis_);
  if (cudaMemcpy(h_Vm.data(), Vm_base_, total_vis_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(h_Vr.data(), Vr_base_, total_vis_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  size_t idx = 0;
  for (size_t f = 0; f < fields_.size(); f++) {
    Field& field = ms->field(f);
    const GPUField& gf = fields_[f];
    size_t bl_idx = 0;
    for (Baseline& bl : field.baselines()) {
      if (bl_idx >= gf.baselines.size()) break;
      const GPUBaseline& gb = gf.baselines[bl_idx];
      for (size_t c = 0; c < gb.chunks.size(); c++) {
        TimeSample& ts = bl.time_samples()[c];
        auto& vis = ts.visibilities();
        for (size_t vi = 0; vi < vis.size() && idx < total_vis_; vi++, idx++) {
          vis[vi].Vm = h_Vm[idx];
          vis[vi].Vr = h_Vr[idx];
        }
      }
      bl_idx++;
    }
  }
  return true;
}

}  // namespace ms
}  // namespace gpuvmem
