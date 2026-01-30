#ifndef GPUVMEM_MS_GPU_BUFFERS_H
#define GPUVMEM_MS_GPU_BUFFERS_H

#include "ms/measurement_set.h"

#include <cufft.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace gpuvmem {
namespace ms {

/**
 * GPU device buffers for visibility data: hold uvw, weights, and
 * visibility columns (data, model, residual) on device for image synthesis.
 * Sync to/from host MeasurementSet.
 */
class VisibilityGPUBuffers {
 public:
  VisibilityGPUBuffers() = default;
  ~VisibilityGPUBuffers();

  VisibilityGPUBuffers(const VisibilityGPUBuffers&) = delete;
  VisibilityGPUBuffers& operator=(const VisibilityGPUBuffers&) = delete;
  VisibilityGPUBuffers(VisibilityGPUBuffers&&) noexcept = default;
  VisibilityGPUBuffers& operator=(VisibilityGPUBuffers&&) noexcept = default;

  /** Total number of visibility samples (across all baselines/time/channel/pol). */
  size_t num_visibilities() const { return nvis_; }

  /** Device pointer to UVW (length nvis_, double3). */
  double3* uvw_device() { return uvw_; }
  const double3* uvw_device() const { return uvw_; }

  /** Device pointer to weight (length nvis_). */
  float* weight_device() { return weight_; }
  const float* weight_device() const { return weight_; }

  /** Device pointer to observed data Vo (length nvis_). */
  cufftComplex* Vo_device() { return Vo_; }
  const cufftComplex* Vo_device() const { return Vo_; }

  /** Device pointer to model Vm (length nvis_). */
  cufftComplex* Vm_device() { return Vm_; }
  const cufftComplex* Vm_device() const { return Vm_; }

  /** Device pointer to residual Vr (length nvis_). */
  cufftComplex* Vr_device() { return Vr_; }
  const cufftComplex* Vr_device() const { return Vr_; }

  /**
   * Upload visibility data from ms to device. Flattens per-baseline/time/channel/pol
   * into contiguous arrays. Allocates device memory as needed.
   * Returns true on success.
   */
  bool upload(const MeasurementSet& ms);

  /**
   * Download visibility data from device to ms (e.g. after synthesis updates Vm/Vr).
   * ms layout must match the previous upload (same fields/baselines/time/channel/pol).
   * Returns true on success.
   */
  bool download(MeasurementSet* ms);

  /** Release all device memory. */
  void clear();

 private:
  size_t nvis_{0};
  double3* uvw_{nullptr};
  float* weight_{nullptr};
  cufftComplex* Vo_{nullptr};
  cufftComplex* Vm_{nullptr};
  cufftComplex* Vr_{nullptr};

  /** Flattened layout metadata for download: same order as upload. */
  struct RowMapping {
    int field_id{0};
    int antenna1{0};
    int antenna2{0};
    int data_desc_id{0};
    size_t time_sample_index{0};
    size_t vis_index_in_sample{0};
  };
  std::vector<RowMapping> row_mapping_;
};

// -----------------------------------------------------------------------------
// Chunked GPU structure: mirrors host (field → baseline → chunk) for intuitive
// per-chunk looping (e.g. per field, per baseline, per spectral window).
// -----------------------------------------------------------------------------

/**
 * One time sample's visibility data on device. Contiguous Vo, Vm, Vr, weight;
 * uvw is one double3 (same for all vis in this chunk) or repeated per vis.
 * weight = imaging_weight (used for imaging; weighting schemes modify this on host).
 */
struct GPUChunk {
  double3* uvw{nullptr};       // one double3 or length count (same value)
  float* weight{nullptr};      // imaging_weight per visibility
  cufftComplex* Vo{nullptr};
  cufftComplex* Vm{nullptr};
  cufftComplex* Vr{nullptr};
  size_t count{0};
  int data_desc_id{0};         // for spectral window / frequency context
  double time{0.0};

  bool empty() const { return count == 0 || !Vo; }
};

/**
 * One baseline on device: antenna indices (for PB lookup) + list of chunks
 * (one per time sample / data_desc).
 */
struct GPUBaseline {
  int antenna1{0};
  int antenna2{0};
  std::vector<GPUChunk> chunks;
};

/**
 * One field on device: field id + list of baselines.
 */
struct GPUField {
  int field_id{0};
  std::vector<GPUBaseline> baselines;
};

/**
 * GPU dataset with same structure as host: field → baseline → chunk.
 * Loop the same way as on CPU; each chunk has device pointers for that chunk only.
 *
 * Example (same loop as host):
 *   ChunkedVisibilityGPU gpu;
 *   gpu.upload(ms);
 *   for (const GPUField& field : gpu.fields())
 *     for (const GPUBaseline& bl : field.baselines())
 *       for (const GPUChunk& ch : bl.chunks) {
 *         // ch.Vo, ch.Vm, ch.Vr, ch.weight, ch.uvw are device ptrs for this chunk
 *         // ch.count, ch.data_desc_id (spectral window), bl.antenna1/2 (for PB)
 *         launch_kernel(ch.Vo, ch.Vm, ch.count, ...);
 *       }
 */
class ChunkedVisibilityGPU {
 public:
  ChunkedVisibilityGPU() = default;
  ~ChunkedVisibilityGPU();

  ChunkedVisibilityGPU(const ChunkedVisibilityGPU&) = delete;
  ChunkedVisibilityGPU& operator=(const ChunkedVisibilityGPU&) = delete;
  ChunkedVisibilityGPU(ChunkedVisibilityGPU&&) noexcept = default;
  ChunkedVisibilityGPU& operator=(ChunkedVisibilityGPU&&) noexcept = default;

  /** Same hierarchy as MeasurementSet: fields → baselines → chunks. */
  std::vector<GPUField>& fields() { return fields_; }
  const std::vector<GPUField>& fields() const { return fields_; }
  size_t num_fields() const { return fields_.size(); }

  /**
   * Upload from host MS. Builds field→baseline→chunk structure and copies
   * each chunk's data to device. Returns true on success.
   */
  bool upload(const MeasurementSet& ms);

  /**
   * Download Vm and Vr from device back to host ms (same structure).
   */
  bool download(MeasurementSet* ms);

  /** Compute Vr = Vo - Vm on device (call after filling Vm, e.g. after degridding). */
  void compute_residuals();

  /** Zero Vm and Vr on device (e.g. for clearRun). */
  void zero_model_and_residual();

  /** Max of chunk counts over all chunks (for sizing gather buffers). */
  size_t max_chunk_count() const;

  void clear();

 private:
  std::vector<GPUField> fields_;
  /** Owned device memory: one allocation per type, chunks point into these. */
  double3* uvw_base_{nullptr};
  float* weight_base_{nullptr};
  cufftComplex* Vo_base_{nullptr};
  cufftComplex* Vm_base_{nullptr};
  cufftComplex* Vr_base_{nullptr};
  size_t total_vis_{0};
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_GPU_BUFFERS_H
