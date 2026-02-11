#ifndef GRIDDING_KERNELS_CUH
#define GRIDDING_KERNELS_CUH

#include <cufft.h>
#include <cuda_runtime.h>

struct float3;
struct double3;

__global__ void do_griddingGPU(float3* uvw,
                               cufftComplex* Vo,
                               cufftComplex* Vo_g,
                               float* w,
                               float* w_g,
                               int* count,
                               double deltau,
                               double deltav,
                               int visibilities,
                               int M,
                               int N);

__global__ void degriddingGPU(double3* uvw,
                              cufftComplex* Vm,
                              cufftComplex* Vm_g,
                              float* kernel,
                              double deltau,
                              double deltav,
                              int visibilities,
                              int M,
                              int N,
                              int kernel_m,
                              int kernel_n,
                              int supportX,
                              int supportY);

__global__ void applyHermitianSymmetry(double3* UVW,
                                       cufftComplex* Vo,
                                       int numVisibilities);

__global__ void convertUVWToLambda(double3* UVW,
                                   float freq,
                                   int numVisibilities);

#endif  // GRIDDING_KERNELS_CUH
