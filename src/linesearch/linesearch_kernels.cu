#include "linesearch/linesearch_kernels.cuh"
#include "framework.cuh"

// Kernel: Update point without positivity constraint
__global__ void newPNoPositivity(float* p,
                                 float* xi,
                                 float xmin,
                                 long N,
                                 long M,
                                 int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  // Bounds check to prevent illegal memory access
  if (i >= M || j >= N) {
    return;
  }

  const long idx = N * M * image + N * i + j;
  xi[idx] *= xmin;
  p[idx] += xi[idx];
}

// Kernel: Evaluate xt = pcom + x*xicom without positivity constraint
__global__ void evaluateXtNoPositivity(float* xt,
                                       float* pcom,
                                       float* xicom,
                                       float x,
                                       long N,
                                       long M,
                                       int image) {
  const int j = threadIdx.x + blockDim.x * blockIdx.x;
  const int i = threadIdx.y + blockDim.y * blockIdx.y;

  if (i >= M || j >= N) {
    return;
  }

  xt[N * M * image + N * i + j] =
      pcom[N * M * image + N * i + j] + x * xicom[N * M * image + N * i + j];
}
