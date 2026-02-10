#ifndef FFT_HOST_CUH
#define FFT_HOST_CUH

#include "framework.cuh"
#include <cufft.h>

__host__ void initFFT(varsPerGPU* vars_gpu,
                      long M,
                      long N,
                      int firstgpu,
                      int num_gpus);

__host__ void FFT2D(cufftComplex* output_data,
                    cufftComplex* input_data,
                    cufftHandle plan,
                    int M,
                    int N,
                    int direction,
                    bool shift);

#endif  // FFT_HOST_CUH
