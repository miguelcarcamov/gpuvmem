#ifndef CHI2_HOST_CUH
#define CHI2_HOST_CUH

#include "framework.cuh"

class Image;

__host__ float chi2(float* I,
                    const Image* grid_image,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale);

__host__ void dchi2(float* I,
                    float* dxi2,
                    float* result_dchi2,
                    const Image* grid_image,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale);

// Add gradient contribution from single-plane dgi to multi-plane dphi
__host__ void linkAddToDPhi(float* dphi, float* dgi, int index);

#endif  // CHI2_HOST_CUH
