#ifndef COMPLEXOPS_CUH
#define COMPLEXOPS_CUH

#include <cufft.h>
#include <math_constants.h>

__host__ __device__ cufftComplex floatComplexZero();
__host__ __device__ cufftDoubleComplex doubleComplexZero();
__host__ __device__ float amplitude(cufftComplex c);
__host__ __device__ double amplitude(cufftDoubleComplex c);
__host__ __device__ float phaseDegrees(cufftComplex c);
__host__ __device__ double phaseDegrees(cufftDoubleComplex c);
__host__ __device__ cufftComplex mulComplexReal(cufftComplex c1, float c2);
__host__ __device__ cufftDoubleComplex mulComplexReal(cufftDoubleComplex c1,
                                                      float c2);
__host__ __device__ cufftComplex divComplexReal(cufftComplex c1, float c2);
__host__ __device__ cufftDoubleComplex divComplexReal(cufftDoubleComplex c1,
                                                      double c2);
__global__ void mulArrayComplexComplex(cufftComplex *c1, cufftComplex *c2,
                                       int M, int N);
__global__ void mulArrayComplexComplex(cufftDoubleComplex *c1,
                                       cufftDoubleComplex *c2, int M, int N);

#endif
