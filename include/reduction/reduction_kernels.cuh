#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Forward declaration of SharedMemory template (defined in .cu file)
template <class T>
struct SharedMemory;

// Template specializations
template <>
struct SharedMemory<double>;

// Reduction kernels
template <class T, int blockSize, bool nIsPow2>
__global__ void reduceSumKernel(T* g_idata, T* g_odata, unsigned int n);

template <int blockSize, bool nIsPow2>
__global__ void reduceMinKernel(float* g_idata, float* g_odata, unsigned int n);

template <int blockSize, bool nIsPow2>
__global__ void reduceMaxKernel(float* g_idata, float* g_odata, unsigned int n);

#endif  // REDUCTION_KERNELS_CUH
