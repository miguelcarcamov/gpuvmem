#ifndef LINESEARCH_KERNELS_CUH
#define LINESEARCH_KERNELS_CUH

#include <cuda_runtime.h>

// Kernel: Update point without positivity constraint
__global__ void newPNoPositivity(float* p,
                                 float* xi,
                                 float xmin,
                                 long N,
                                 long M,
                                 int image);

// Kernel: Evaluate xt = pcom + x*xicom without positivity constraint
__global__ void evaluateXtNoPositivity(float* xt,
                                       float* pcom,
                                       float* xicom,
                                       float x,
                                       long N,
                                       long M,
                                       int image);

// Kernel: Update point with positivity constraint (clip to min_pixel_value)
__global__ void newP(float* p, float* xi, float xmin, long N, long M,
                     float min_pixel_value, float eta, int image);

// Kernel: Evaluate xt = pcom + x*xicom with positivity constraint
__global__ void evaluateXt(float* xt, float* pcom, float* xicom, float x,
                          long N, long M, float initial_value, float eta, int image);

#endif  // LINESEARCH_KERNELS_CUH
