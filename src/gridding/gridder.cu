#include "gridding/gridder.cuh"
#include "framework.cuh"
#include "classes/image.cuh"

#include <ckernel.cuh>

extern double deltau, deltav;
extern long M, N;
extern int num_gpus, firstgpu;
extern Vars variables;

Gridder::Gridder(CKernel* ckernel, int num_threads)
    : ckernel_(ckernel), num_threads_(num_threads) {}

void Gridder::grid(std::vector<gpuvmem::ms::MSWithGPU>& datasets) {
  for (auto& dw : datasets) {
    grid(dw);
  }
}

void Gridder::grid(gpuvmem::ms::MSWithGPU& dataset) {
  dataset.gridded_ms =
      do_gridding(dataset.ms, &dataset.gpu, deltau, deltav, M, N, ckernel_,
                  num_threads_);
}

void Gridder::degrid(std::vector<gpuvmem::ms::MSWithGPU>& datasets,
                     float* I,
                     VirtualImageProcessor* ip,
                     const Image* grid_image) {
  for (auto& dw : datasets) {
    degrid(dw, I, ip, grid_image);
  }
}

void Gridder::degrid(gpuvmem::ms::MSWithGPU& dataset,
                     float* I,
                     VirtualImageProcessor* ip,
                     const Image* grid_image) {
  do_degridding(dataset.ms, &dataset.gpu, num_gpus, firstgpu,
                variables.blockSizeV, ckernel_, I, ip, grid_image);
}
