#ifndef IDFT_KERNELS_CUH
#define IDFT_KERNELS_CUH

#include <cufft.h>
#include <cuda_runtime.h>
#include "framework.cuh"

struct double3;

// Device function to compute IDFT for a single pixel
// This can be called from both the idft kernel and DChi2 kernel
__device__ float computeIdftPixel(int i,
                                   int j,
                                   cufftComplex* visibilities,
                                   double3* uvw,
                                   float* weights,
                                   long N,
                                   long numVisibilities,
                                   float phs_xobs,
                                   float phs_yobs,
                                   double DELTAX,
                                   double DELTAY);

// Inverse Direct Fourier Transform kernel
// Computes IDFT: I(x,y) = Σ_v w_v * V_r(v) * exp(2πi * (u*x + v*y + w*(z-1)))
// This is the adjoint of the forward DFT used in the measurement operator
__global__ void idft(float* output_image,
                     cufftComplex* visibilities,
                     double3* uvw,
                     float* weights,
                     long N,
                     long M,
                     long numVisibilities,
                     float phs_xobs,
                     float phs_yobs,
                     double DELTAX,
                     double DELTAY);

#endif  // IDFT_KERNELS_CUH
