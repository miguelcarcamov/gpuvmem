#ifndef CHI2_HOST_CUH
#define CHI2_HOST_CUH

#include "framework.cuh"

__host__ float chi2(float* I,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale);

__host__ void dchi2(float* I,
                    float* dxi2,
                    float* result_dchi2,
                    VirtualImageProcessor* ip,
                    bool normalize,
                    float fg_scale);

#endif  // CHI2_HOST_CUH
