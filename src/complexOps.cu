#include "complexOps.cuh"

__host__ __device__ cufftComplex floatComplexZero() {
  cufftComplex zero = make_cuFloatComplex(0.0f, 0.0f);
  return zero;
};

__host__ __device__ cufftDoubleComplex doubleComplexZero() {
  cufftDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  return zero;
};

__host__ __device__ float amplitude(cufftComplex c) {
  float amp = cuCabsf(c);
  return amp;
};

__host__ __device__ double amplitude(cufftDoubleComplex c) {
  double amp = cuCabs(c);
  return amp;
};

__host__ __device__ float phaseDegrees(cufftComplex c) {
  float phase = atan2f(c.y, c.x) * 180.0f / CUDART_PI_F;
  return phase;
};

__host__ __device__ double phaseDegrees(cufftDoubleComplex c) {
  double phase = atan2(c.y, c.x) * 180.0 / CUDART_PI;
  return phase;
};

__host__ __device__ cufftComplex mulComplexReal(cufftComplex c1, float c2) {
  cufftComplex result;
  result = cuCmulf(c1, make_cuFloatComplex(c2, 0.0f));
  result.x = c1.x * c2;
  result.y = c1.y * c2;

  return result;
};

__host__ __device__ cufftDoubleComplex mulComplexReal(cufftDoubleComplex c1,
                                                      float c2) {
  cufftDoubleComplex result;
  result = cuCmul(c1, make_cuDoubleComplex(c2, 0.0));
  return result;
};

__host__ __device__ cufftComplex divComplexReal(cufftComplex c1, float c2) {
  cufftComplex result;

  result = cuCdivf(c1, make_cuFloatComplex(c2, 0.0));
  return result;
};

__host__ __device__ cufftDoubleComplex divComplexReal(cufftDoubleComplex c1,
                                                      double c2) {
  cufftDoubleComplex result;
  result = cuCdiv(c1, make_cuDoubleComplex(c2, 0.0));
  return result;
};

__global__ void mulArrayComplexComplex(cufftComplex *c1, cufftComplex *c2,
                                       int M, int N) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < M && j < N) {
    c1[N * i + j] = cuCmulf(c1[N * i + j], c2[N * i + j]);
  }
};

__global__ void mulArrayComplexComplex(cufftDoubleComplex *c1,
                                       cufftDoubleComplex *c2, int M, int N) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < M && j < N) {
    c1[N * i + j] = cuCmul(c1[N * i + j], c2[N * i + j]);
  }
};
