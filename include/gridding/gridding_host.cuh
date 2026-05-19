#ifndef GRIDDING_HOST_CUH
#define GRIDDING_HOST_CUH

#include "framework.cuh"
#include "ms/measurement_set.h"
#include "ms/gpu_buffers.h"
#include <vector>

namespace gpuvmem {
namespace ms {
class MeasurementSet;
class ChunkedVisibilityGPU;
}  // namespace ms
}  // namespace gpuvmem

class Image;

// Gridding functions - New MS API
__host__ gpuvmem::ms::MeasurementSet do_gridding(
    gpuvmem::ms::MeasurementSet& ms,
    gpuvmem::ms::ChunkedVisibilityGPU* gpu,
    double deltau,
    double deltav,
    long M,
    long N,
    CKernel* ckernel,
    int gridding);

__host__ void do_degridding(gpuvmem::ms::MeasurementSet& ms,
                            gpuvmem::ms::ChunkedVisibilityGPU* gpu,
                            int num_gpus,
                            int firstgpu,
                            int blockSizeV,
                            CKernel* ckernel,
                            float* I,
                            VirtualImageProcessor* ip,
                            const Image* grid_image);

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
