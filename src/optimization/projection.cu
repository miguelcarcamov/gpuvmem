#include "optimization/projection.hh"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cstdio>

namespace {

__global__ void scalar_replace_plane_kernel(float* buffer, long N, long M, int image, float compared,
                                            float replacement, int op_kind) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= M || j >= N) return;
  const long idx = N * M * image + N * i + j;
  float v = buffer[idx];
  bool cond = false;
  switch (op_kind) {
    case 0:
      cond = (v == compared);
      break;
    case 1:
      cond = (v != compared);
      break;
    case 2:
      cond = (v > compared);
      break;
    case 3:
      cond = (v >= compared);
      break;
    case 4:
      cond = (v < compared);
      break;
    case 5:
      cond = (v <= compared);
      break;
    default:
      return;
  }
  if (cond) buffer[idx] = replacement;
}

void launch_scalar_projection_on_plane(float* buffer, long N, long M, int image, unsigned blocks_x,
                                       unsigned blocks_y, unsigned threads_x, unsigned threads_y,
                                       float compared_value, float replacement_value, int op_kind) {
  dim3 threads(threads_x, threads_y);
  dim3 blocks(blocks_x, blocks_y);
  scalar_replace_plane_kernel<<<blocks, threads>>>(buffer, N, M, image, compared_value,
                                                    replacement_value, op_kind);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace

Projection::~Projection() = default;

float Projection::positivityEta() const { return -1.f; }

float Projection::referenceValue(int /*image_index*/) const { return 0.f; }

void Projection::applyToImagePlane(float* /*buffer*/, long /*N*/, long /*M*/, int /*image*/,
                                    unsigned /*blocks_x*/, unsigned /*blocks_y*/, unsigned /*threads_x*/,
                                    unsigned /*threads_y*/) const {}

PositivityProjection::PositivityProjection(float eta, std::vector<float> xt_reference_per_image)
    : eta_(eta), xt_reference_(std::move(xt_reference_per_image)) {}

float PositivityProjection::positivityEta() const { return eta_; }

float PositivityProjection::referenceValue(int image_index) const {
  if (image_index >= 0 && image_index < static_cast<int>(xt_reference_.size())) {
    return xt_reference_[static_cast<size_t>(image_index)];
  }
  return 0.0f;
}

ScalarReplaceProjection::ScalarReplaceProjection(float compared_value, float replacement_value,
                                                 ScalarProjectionOp op)
    : compared_value_(compared_value), replacement_value_(replacement_value), op_(op) {}

void ScalarReplaceProjection::applyToImagePlane(float* buffer, long N, long M, int image,
                                                unsigned blocks_x, unsigned blocks_y,
                                                unsigned threads_x, unsigned threads_y) const {
  launch_scalar_projection_on_plane(buffer, N, M, image, blocks_x, blocks_y, threads_x, threads_y,
                                   compared_value_, replacement_value_, static_cast<int>(op_));
}

float ScalarReplaceProjection::comparedValue() const { return compared_value_; }

float ScalarReplaceProjection::replacementValue() const { return replacement_value_; }

ScalarProjectionOp ScalarReplaceProjection::op() const { return op_; }

EqualTo::EqualTo(float compared_value, float replacement_value)
    : ScalarReplaceProjection(compared_value, replacement_value, ScalarProjectionOp::EqualTo) {}

NotEqualTo::NotEqualTo(float compared_value, float replacement_value)
    : ScalarReplaceProjection(compared_value, replacement_value, ScalarProjectionOp::NotEqualTo) {}

GreaterThan::GreaterThan(float compared_value, float replacement_value)
    : ScalarReplaceProjection(compared_value, replacement_value, ScalarProjectionOp::GreaterThan) {}

GreaterThanEqualTo::GreaterThanEqualTo(float compared_value, float replacement_value)
    : ScalarReplaceProjection(compared_value, replacement_value,
                              ScalarProjectionOp::GreaterThanEqualTo) {}

LessThan::LessThan(float compared_value, float replacement_value)
    : ScalarReplaceProjection(compared_value, replacement_value, ScalarProjectionOp::LessThan) {}

LessThanEqualTo::LessThanEqualTo(float compared_value, float replacement_value)
    : ScalarReplaceProjection(compared_value, replacement_value, ScalarProjectionOp::LessThanEqualTo) {
}

CompositeProjection::CompositeProjection() = default;

CompositeProjection::CompositeProjection(std::vector<std::unique_ptr<Projection>> parts)
    : parts_(std::move(parts)) {}

void CompositeProjection::append(std::unique_ptr<Projection> p) {
  if (p) parts_.push_back(std::move(p));
}

void CompositeProjection::applyToImagePlane(float* buffer, long N, long M, int image,
                                            unsigned blocks_x, unsigned blocks_y, unsigned threads_x,
                                            unsigned threads_y) const {
  for (const auto& up : parts_) {
    if (up) {
      up->applyToImagePlane(buffer, N, M, image, blocks_x, blocks_y, threads_x, threads_y);
    }
  }
}

float CompositeProjection::positivityEta() const {
  for (const auto& up : parts_) {
    if (!up) continue;
    if (dynamic_cast<const PositivityProjection*>(up.get()) != nullptr) {
      return up->positivityEta();
    }
  }
  return Projection::positivityEta();
}

float CompositeProjection::referenceValue(int image_index) const {
  for (const auto& up : parts_) {
    if (!up) continue;
    if (dynamic_cast<const PositivityProjection*>(up.get()) != nullptr) {
      return up->referenceValue(image_index);
    }
  }
  return Projection::referenceValue(image_index);
}
