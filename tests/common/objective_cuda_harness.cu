#include "objective_cuda_harness.hh"

#include "legacy_imaging_globals.hh"

#include <helper_cuda.h>

extern long M, N;
extern int image_count, iter, firstgpu;
extern float noise_cut;
extern float* device_noise_image;
extern dim3 threadsPerBlockNN, numBlocksNN;

namespace gpuvmem {
namespace test {

bool ObjectiveCudaHarness::init_cuda() {
  if (!cuda_device_available()) {
    return false;
  }
  checkCudaErrors(cudaSetDevice(firstgpu));
  return true;
}

void ObjectiveCudaHarness::apply_geometry() {
  n = geometry.n;
  m = geometry.m;
  images = geometry.images;
  apply_legacy_imaging_globals(geometry);
}

void ObjectiveCudaHarness::teardown_cuda() {
  if (device_image) {
    cudaFree(device_image);
    device_image = nullptr;
  }
  if (device_noise_image) {
    cudaFree(device_noise_image);
    device_noise_image = nullptr;
  }
}

void ObjectiveCudaHarness::init_launch_grid() {
  M = m;
  N = n;
  image_count = images;
  iter = 1;
  const CudaGrid<2> grid =
      CudaGrid<2>::from_extents(n, m, dim3(16, 16, 1));
  threadsPerBlockNN = grid.threads();
  numBlocksNN = grid.blocks();
}

void ObjectiveCudaHarness::alloc_uniform_image(float value) {
  const size_t plane = static_cast<size_t>(m) * static_cast<size_t>(n) *
                       static_cast<size_t>(images);
  if (device_image) cudaFree(device_image);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&device_image),
                             sizeof(float) * plane));
  std::vector<float> host(plane, value);
  checkCudaErrors(cudaMemcpy(device_image, host.data(), sizeof(float) * plane,
                             cudaMemcpyHostToDevice));
}

void ObjectiveCudaHarness::alloc_noise_mask(float value) {
  const size_t plane = static_cast<size_t>(m) * static_cast<size_t>(n);
  if (device_noise_image) cudaFree(device_noise_image);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&device_noise_image),
                             sizeof(float) * plane));
  std::vector<float> host(plane, value);
  checkCudaErrors(cudaMemcpy(device_noise_image, host.data(), sizeof(float) * plane,
                             cudaMemcpyHostToDevice));
}

ObjectiveFunction ObjectiveCudaHarness::make_objective() const {
  ObjectiveFunction of;
  of.setGridDimensions(n, m, images);
  of.configure(n, m, images);
  const CudaGrid<2> grid =
      CudaGrid<2>::from_extents(n, m, dim3(16, 16, 1));
  of.setThreadsPerBlockNN(grid.threads());
  of.setNumBlocksNN(grid.blocks());
  of.setPrimaryCudaDevice(firstgpu);
  return of;
}

void ObjectiveCudaHarness::wire_production_weights(ObjectiveFunction& of) const {
  of.setRegularizationWeights(weights, 4);
}

bool ObjectiveCudaHarness::add_term(ObjectiveFunction& of, Fi* term, int z_index,
                                    int image_index, int image_to_add,
                                    bool normalize) const {
  term->attachToObjectiveFunction(&of);
  term->configure(z_index, image_index, image_to_add, normalize);
  term->setIteration(1);
  const bool had = term->getPenalizationFactor() != 0.f;
  of.addFi(term);
  return had;
}

float ObjectiveCudaHarness::eval_phi(ObjectiveFunction& of) const {
  return of.calcFunction(device_image);
}

}  // namespace test
}  // namespace gpuvmem
