#include <cuda_runtime.h>

extern long N, M;
extern dim3 numBlocksNN, threadsPerBlockNN;
extern int firstgpu;

__global__ void AddToDPhiStub(float* dphi, float* dgi, long n, long m, int index) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= m || j >= n) return;
  dphi[static_cast<size_t>(n) * static_cast<size_t>(m) * static_cast<size_t>(index) +
       static_cast<size_t>(n) * static_cast<size_t>(i) + static_cast<size_t>(j)] +=
      dgi[static_cast<size_t>(n) * static_cast<size_t>(i) + static_cast<size_t>(j)];
}

__host__ void linkAddToDPhi(float* dphi, float* dgi, int index) {
  cudaSetDevice(firstgpu);
  AddToDPhiStub<<<numBlocksNN, threadsPerBlockNN>>>(dphi, dgi, N, M, index);
  cudaDeviceSynchronize();
}
