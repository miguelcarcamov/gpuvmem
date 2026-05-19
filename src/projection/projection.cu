#include "projection/projection.hh"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cstdio>

namespace {

/**
 * 1D clamp over one image plane [plane_offset, plane_offset + plane_elems).
 * Uses a fixed launch shape so we never depend on ObjectiveFunction 2D dim3 (z slices,
 * stale zeros, etc.), which was triggering cudaErrorInvalidValue on some runs.
 */
__global__ void clamp_plane_minimum_1d_kernel(float* buffer, long plane_offset, long plane_elems,
                                              float floor_val) {
  const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= plane_elems) return;
  const long idx = plane_offset + tid;
  float v = buffer[idx];
  if (v < floor_val) buffer[idx] = floor_val;
}

void launch_clamp_plane_minimum(float* buffer, long N, long M, int image, unsigned /*blocks_x*/,
                                unsigned /*blocks_y*/, unsigned /*threads_x*/,
                                unsigned /*threads_y*/, float floor_val) {
  if (buffer == nullptr || N <= 0 || M <= 0) return;

  const long plane_elems = N * M;
  const long plane_offset = plane_elems * static_cast<long>(image);
  constexpr int kThreads = 256;
  if (plane_elems <= 0) return;

  unsigned int grid_x =
      static_cast<unsigned int>((plane_elems + static_cast<long>(kThreads) - 1L) /
                                static_cast<long>(kThreads));
  if (grid_x == 0u) grid_x = 1u;

  (void)cudaGetLastError();
  clamp_plane_minimum_1d_kernel<<<grid_x, kThreads>>>(buffer, plane_offset, plane_elems, floor_val);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace

Projection::~Projection() = default;

float Projection::positivityEta() const { return -1.f; }

float Projection::referenceValue(int /*image_index*/) const { return 0.f; }

float Projection::minimalValue(int /*image_index*/) const { return 0.f; }

void Projection::applyToImagePlane(float* /*buffer*/, long /*N*/, long /*M*/, int /*image*/,
                                   unsigned /*blocks_x*/, unsigned /*blocks_y*/, unsigned /*threads_x*/,
                                   unsigned /*threads_y*/, ProjectionApplyContext /*ctx*/) const {}

PositivityProjection::PositivityProjection(float eta, std::vector<float> xt_reference_per_image,
                                           std::vector<float> minimal_pixel_values_per_image)
    : eta_(eta),
      xt_reference_(std::move(xt_reference_per_image)),
      minimal_pixel_(std::move(minimal_pixel_values_per_image)) {}

float PositivityProjection::positivityEta() const { return eta_; }

float PositivityProjection::referenceValue(int image_index) const {
  if (image_index >= 0 && image_index < static_cast<int>(xt_reference_.size())) {
    return xt_reference_[static_cast<size_t>(image_index)];
  }
  // No per-image reference configured: use the same floor as the parameter step.
  return minimalValue(image_index);
}

float PositivityProjection::minimalValue(int image_index) const {
  if (image_index >= 0 && image_index < static_cast<int>(minimal_pixel_.size())) {
    return minimal_pixel_[static_cast<size_t>(image_index)];
  }
  return 0.0f;
}

void PositivityProjection::applyToImagePlane(float* buffer, long N, long M, int image,
                                             unsigned blocks_x, unsigned blocks_y, unsigned threads_x,
                                             unsigned threads_y, ProjectionApplyContext ctx) const {
  // Legacy MEM: only the first image plane used fused positivity kernels.
  if (image != 0 || buffer == nullptr) return;
  const float floor_val = (ctx == ProjectionApplyContext::kParameterStep)
                              ? minimalValue(0)
                              : referenceValue(0);
  launch_clamp_plane_minimum(buffer, N, M, image, blocks_x, blocks_y, threads_x, threads_y, floor_val);
}
