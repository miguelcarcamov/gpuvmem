#pragma once

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace gpuvmem {
namespace test {

/** True only if at least one device is visible and usable (setDevice succeeds). */
inline bool cuda_device_available() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) return false;
  return cudaSetDevice(0) == cudaSuccess;
}

inline void skip_if_no_cuda() {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA device available";
  }
}

inline void require_cuda_device() {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA device available";
  }
}

}  // namespace test
}  // namespace gpuvmem
