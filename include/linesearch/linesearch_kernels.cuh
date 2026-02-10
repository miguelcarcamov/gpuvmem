#ifndef LINESEARCH_KERNELS_CUH
#define LINESEARCH_KERNELS_CUH

#include <cuda_runtime.h>

__global__ void newP(float* p,
                     float* xi,
                     float xmin,
                     float MINPIX,
                     float eta,
                     long N);

__global__ void newP(float* p,
                     float* xi,
                     float xmin,
                     long N,
                     long M,
                     float MINPIX,
                     float eta,
                     int image);

__global__ void newPNoPositivity(float* p,
                                 float* xi,
                                 float xmin,
                                 long N,
                                 long M,
                                 int image);

__global__ void evaluateXt(float* xt,
                           float* pcom,
                           float* xicom,
                           float x,
                           float MINPIX,
                           float eta,
                           long N);

__global__ void evaluateXt(float* xt,
                           float* pcom,
                           float* xicom,
                           float x,
                           long N,
                           long M,
                           float MINPIX,
                           float eta,
                           int image);

__global__ void evaluateXtNoPositivity(float* xt,
                                       float* pcom,
                                       float* xicom,
                                       float x,
                                       long N,
                                       long M,
                                       int image);

#endif  // LINESEARCH_KERNELS_CUH
