#ifndef FFT_KERNELS_CUH
#define FFT_KERNELS_CUH

#include <cufft.h>

// fftshift_swap and ifftshift_swap are implemented in fft_kernels.cu
// (not declared here since they're only used internally)

__global__ void fftshift_2D(cufftComplex* __restrict__ data, int N1, int N2);
__global__ void ifftshift_2D(cufftComplex* __restrict__ data, int N1, int N2);

#endif  // FFT_KERNELS_CUH
