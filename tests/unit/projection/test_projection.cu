#include <gtest/gtest.h>

#include "cuda_test_skip.hh"
#include "projection/projection.hh"

#include <cuda_runtime.h>
#include <vector>

using gpuvmem::test::cuda_device_available;

TEST(Projection, PositivityProjectionMetadata) {
  PositivityProjection proj(0.5f, {1.f, 2.f}, {0.01f, 0.02f});
  EXPECT_FLOAT_EQ(proj.positivityEta(), 0.5f);
  EXPECT_FLOAT_EQ(proj.referenceValue(0), 1.f);
  EXPECT_FLOAT_EQ(proj.referenceValue(1), 2.f);
  EXPECT_FLOAT_EQ(proj.minimalValue(0), 0.01f);
}

TEST(Projection, PositivityProjectionClampsPlaneParameterStep) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  const long N = 4;
  const long M = 4;
  std::vector<float> host(static_cast<size_t>(N * M), -1.f);
  float* device = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  PositivityProjection proj(-1.f, {}, {0.1f});
  proj.applyToImagePlane(device, N, M, 0, 1, 1, 256, 1,
                         ProjectionApplyContext::kParameterStep);

  ASSERT_EQ(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (float v : host) {
    EXPECT_GE(v, 0.1f);
  }
  cudaFree(device);
}

TEST(Projection, PositivityProjectionClampsPlaneLineSearchSample) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  const long N = 4;
  const long M = 4;
  std::vector<float> host(static_cast<size_t>(N * M), -1.f);
  float* device = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  // Empty reference vector: production falls back to minimal floor (see referenceValue).
  PositivityProjection proj(-1.f, {}, {0.1f});
  proj.applyToImagePlane(device, N, M, 0, 1, 1, 256, 1,
                         ProjectionApplyContext::kLineSearch1dSample);

  ASSERT_EQ(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (float v : host) {
    EXPECT_GE(v, 0.1f);
  }
  cudaFree(device);
}

TEST(Projection, PositivityProjectionClampsPlaneLineSearchUsesReference) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  const long N = 4;
  const long M = 4;
  std::vector<float> host(static_cast<size_t>(N * M), -1.f);
  float* device = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  PositivityProjection proj(-1.f, {0.25f}, {0.1f});
  proj.applyToImagePlane(device, N, M, 0, 1, 1, 256, 1,
                         ProjectionApplyContext::kLineSearch1dSample);

  ASSERT_EQ(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (float v : host) {
    EXPECT_GE(v, 0.25f);
  }
  cudaFree(device);
}
