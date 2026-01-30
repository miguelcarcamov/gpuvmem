#ifndef GPUVMEM_GRIDDER_CUH
#define GPUVMEM_GRIDDER_CUH

#include "ms/ms_with_gpu.h"

#include <vector>

class CKernel;
class VirtualImageProcessor;

/**
 * Gridder: performs gridding and degridding on one or multiple measurement sets.
 * Can be instanced once and grid() / degrid() called with a group of datasets
 * (or a single one). Works with the new MS model (MeasurementSet +
 * ChunkedVisibilityGPU).
 */
class Gridder {
 public:
  Gridder() = default;
  /** Optionally set convolution kernel and thread count for CPU gridding. */
  explicit Gridder(CKernel* ckernel, int num_threads = 1);

  /** Grid one or multiple datasets. Each MSWithGPU holds host .ms and device .gpu. */
  void grid(std::vector<gpuvmem::ms::MSWithGPU>& datasets);
  /** Grid a single dataset. */
  void grid(gpuvmem::ms::MSWithGPU& dataset);

  /** Degrid one or multiple datasets: compute model visibilities from image I. */
  void degrid(std::vector<gpuvmem::ms::MSWithGPU>& datasets,
              float* I,
              VirtualImageProcessor* ip);
  /** Degrid a single dataset. */
  void degrid(gpuvmem::ms::MSWithGPU& dataset,
              float* I,
              VirtualImageProcessor* ip);

  void set_kernel(CKernel* ckernel) { ckernel_ = ckernel; }
  void set_threads(int t) { num_threads_ = t; }
  int threads() const { return num_threads_; }

 private:
  CKernel* ckernel_{nullptr};
  int num_threads_{1};
};

#endif  // GPUVMEM_GRIDDER_CUH
