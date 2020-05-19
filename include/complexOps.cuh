#ifndef COMPLEXOPS_CUH
#define COMPLEXOPS_CUH
#include <cufft.h>

__host__ __device__ cufftComplex multRealComplex(cufftComplex c1, float c2);
__host__ __device__ cufftComplex multComplexComplex(cufftComplex c1, cufftComplex c2);
__host__ __device__ cufftComplex divComplexComplex(cufftComplex c1, cufftComplex c2);
__host__ __device__ cufftComplex addComplexComplex(cufftComplex c1, cufftComplex c2);
__host__ __device__ cufftComplex subComplexComplex(cufftComplex c1, cufftComplex c2);
__host__ __device__ cufftComplex ConjComplex(cufftComplex c1);

#endif
