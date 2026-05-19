#pragma once

#include "classes/fi.cuh"
#include "classes/objectivefunction.cuh"
#include "cuda_test_skip.hh"
#include "framework/cuda_grid.cuh"
#include "imaging_geometry.hh"

#include <cuda_runtime.h>
#include <vector>

namespace gpuvmem {
namespace test {

struct ObjectiveCudaHarness {
  ImagingGeometryParam geometry = dataset_geometry_presets().front();
  long n = 8;
  long m = 8;
  int images = 1;
  float* device_image = nullptr;
  float weights[4] = {1.f, 0.01f, 0.001f, 0.05f};

  /** Returns false when no usable CUDA device (caller should GTEST_SKIP). */
  bool init_cuda();
  void teardown_cuda();

  /** Sync n/m/images and legacy globals (deltau, deltav, nu_0) from geometry. */
  void apply_geometry();

  void init_launch_grid();
  void alloc_uniform_image(float value = 2.f);
  void alloc_noise_mask(float value = 1.f);

  ObjectiveFunction make_objective() const;
  void wire_production_weights(ObjectiveFunction& of) const;

  /** Attach/configure/add one Fi; returns false if λ==0 and term skipped. */
  bool add_term(ObjectiveFunction& of, Fi* term, int z_index, int image_index = 0,
                int image_to_add = 0, bool normalize = false) const;

  float eval_phi(ObjectiveFunction& of) const;
};

}  // namespace test
}  // namespace gpuvmem
