#ifndef REDUCTION_HOST_CUH
#define REDUCTION_HOST_CUH

template <class T>
__host__ T reduceCPU(T* data, int size);

template <class T>
__host__ T deviceReduce(T* in, long N, int input_threads);

__host__ float deviceMaxReduce(float* in, long N, int input_threads);
__host__ float deviceMinReduce(float* in, long N, int input_threads);

#endif  // REDUCTION_HOST_CUH
