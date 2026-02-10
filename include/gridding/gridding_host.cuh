#ifndef GRIDDING_HOST_CUH
#define GRIDDING_HOST_CUH

#include "framework.cuh"
#include <vector>

// Gridding functions
__host__ void do_gridding(std::vector<Field>& fields,
                          MSData* data,
                          double deltau,
                          double deltav,
                          int M,
                          int N,
                          CKernel* ckernel,
                          int gridding);

__host__ void degridding(std::vector<Field>& fields,
                         MSData data,
                         double deltau,
                         double deltav,
                         int num_gpus,
                         int firstgpu,
                         int blockSizeV,
                         long M,
                         long N,
                         CKernel* ckernel,
                         float* I,
                         VirtualImageProcessor* ip,
                         MSDataset& dataset);

__host__ void griddedTogrid(std::vector<cufftComplex>& Vm_gridded,
                            std::vector<cufftComplex> Vm_gridded_sp,
                            std::vector<double3> uvw_gridded_sp,
                            double deltau,
                            double deltav,
                            float freq,
                            long M,
                            long N,
                            int numvis);

#endif  // GRIDDING_HOST_CUH
