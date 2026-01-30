#ifndef GPUVMEM_MS_MS_WITH_GPU_H
#define GPUVMEM_MS_MS_WITH_GPU_H

#include "ms/beam_noise.h"
#include "ms/gpu_buffers.h"
#include "ms/measurement_set.h"

#include <string>

namespace gpuvmem {
namespace ms {

/**
 * One measurement set plus its GPU visibility buffers.
 * All operations use the native MS model (ms, gpu). Legacy view removed.
 */
struct MSWithGPU {
  std::string name;   /**< Input MS path */
  std::string oname;  /**< Output MS path (for write-back) */

  MeasurementSet ms;
  ChunkedVisibilityGPU gpu;

  /**
   * When gridding is used: host copy of gridded visibilities for synthesis.
   * Empty (num_fields()==0) when not gridded. Filled by Gridder::grid() via
   * do_gridding().
   */
  MeasurementSet gridded_ms;

  /**
   * Per-field beam attenuation image (M*N float), allocated in setDevice.
   * Size = ms.num_fields(); cleared in unSetDevice.
   */
  std::vector<float*> atten_image;

  /** Upload host ms to device gpu. Call after weighting (imaging_weight is used). */
  bool upload() { return gpu.upload(ms); }

  /** Download Vm and Vr from device to host ms (e.g. after synthesis). */
  bool download() { return gpu.download(&ms); }

  void clear() { gpu.clear(); }

  /**
   * Accumulate this dataset's contribution to beam and noise sums.
   * Caller zeros s_uu, s_vv, s_uv, sum_weights, total_visibilities before the
   * first dataset; then call beamNoiseFromSums once after all datasets.
   */
  void computeNoiseAndBeamContribution(double* s_uu,
                                        double* s_vv,
                                        double* s_uv,
                                        float* sum_weights,
                                        int* total_visibilities) const {
    gpuvmem::ms::computeNoiseAndBeamContribution(
        ms, s_uu, s_vv, s_uv, sum_weights, total_visibilities);
  }
};

}  // namespace ms
}  // namespace gpuvmem

#endif  // GPUVMEM_MS_MS_WITH_GPU_H
