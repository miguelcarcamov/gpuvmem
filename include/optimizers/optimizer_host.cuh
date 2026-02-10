#ifndef OPTIMIZER_HOST_CUH
#define OPTIMIZER_HOST_CUH

#include <cuda_runtime.h>

// Optimizer host wrapper functions
__host__ void linkRestartDGi(float* dgi);

__host__ void linkAddToDPhi(float* dphi, float* dgi, int index);

__host__ void defaultNewP(float* p, float* xi, float xmin, int image);

__host__ void particularNewP(float* p, float* xi, float xmin, int image);

__host__ void defaultEvaluateXt(float* xt,
                                float* pcom,
                                float* xicom,
                                float x,
                                int image);

__host__ void particularEvaluateXt(float* xt,
                                   float* pcom,
                                   float* xicom,
                                   float x,
                                   int image);

#endif  // OPTIMIZER_HOST_CUH
