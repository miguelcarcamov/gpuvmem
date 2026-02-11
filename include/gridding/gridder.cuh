#ifndef GRIDDER_CUH
#define GRIDDER_CUH

#include "framework.cuh"
#include "gridding/gridding_host.cuh"
#include <vector>

class CKernel;

class Gridder {
 public:
  Gridder(CKernel* ckernel, int num_threads);
  void grid(std::vector<gpuvmem::ms::MSWithGPU>& datasets);
  void grid(gpuvmem::ms::MSWithGPU& dataset);
  void degrid(std::vector<gpuvmem::ms::MSWithGPU>& datasets, float* I,
              VirtualImageProcessor* ip);
  void degrid(gpuvmem::ms::MSWithGPU& dataset, float* I,
              VirtualImageProcessor* ip);

 private:
  CKernel* ckernel_;
  int num_threads_;
};

#endif  // GRIDDER_CUH
