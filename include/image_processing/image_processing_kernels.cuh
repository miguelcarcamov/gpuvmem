#ifndef IMAGE_PROCESSING_KERNELS_CUH
#define IMAGE_PROCESSING_KERNELS_CUH

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void clipWNoise(cufftComplex* fg_image,
                           float* noise,
                           float* I,
                           long N,
                           float noise_cut,
                           float MINPIX,
                           float eta);

__global__ void clip2IWNoise(float* noise,
                             float* I,
                             long N,
                             long M,
                             float noise_cut,
                             float MINPIX,
                             float alpha_start,
                             float eta,
                             float threshold,
                             float alpha_n_sigma,
                             int schedule);

__global__ void clip2I(float* I, long N, float MINPIX);

__global__ void normalizeImageKernel(float* image,
                                     float normalization_factor,
                                     long N);

__global__ void substraction(float* x,
                             cufftComplex* xc,
                             float* gc,
                             float lambda,
                             long N);

__global__ void projection(float* px, float* x, float MINPIX, long N);

__global__ void normVectorCalculation(float* normVector, float* gc, long N);

__global__ void copyImage(cufftComplex* p, float* device_xt, long N);

__global__ void clipStokesWNoise(float* I,
                                 int nplanes,
                                 long N,
                                 long M,
                                 float* noise,
                                 float noise_cut,
                                 float MINPIX,
                                 float eta);

__global__ void copyItoInu(cufftComplex* image, const float* I, long M, long N);

#endif  // IMAGE_PROCESSING_KERNELS_CUH
