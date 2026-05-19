#ifndef PADDING_CUH
#define PADDING_CUH

#include <cufft.h>

template <class T>
__global__ void paddingKernel(T* dest,
                              T* src,
                              int fft_M,
                              int fft_N,
                              int m,
                              int n,
                              int centerKernel_x,
                              int centerKernel_y);

template <class T>
__global__ void paddingData(T* dest,
                            T* src,
                            int fft_M,
                            int fft_N,
                            int M,
                            int N,
                            int m,
                            int n,
                            int centerKernel_x,
                            int centerKernel_y);

template <class T>
__global__ void depaddingData(T* dest,
                              T* src,
                              int fft_M,
                              int fft_N,
                              int M,
                              int N,
                              int centerKernel_x,
                              int centerKernel_y);

template <class TD, class T>
__host__ TD* convolutionComplexRealFFT(TD* data,
                                       T* kernel,
                                       int M,
                                       int N,
                                       int m,
                                       int n,
                                       bool isDataOnFourier,
                                       bool isDataOnDevice,
                                       bool isKernelOnDevice);

#endif  // PADDING_CUH
