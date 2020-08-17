#ifndef SINC2D_CUH
#define SINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Sinc2D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__host__ __device__ float GCF_fn(float amp, float nu, float w);
__device__ float constructGCF(float amp, float x0, float y0, float sigma_x, float sigma_y, float w, int M, int N);
Sinc2D() : CKernel(){
};
Sinc2D(int m, int n) : CKernel(m, n){
};
Sinc2D(int m, int n, float w1) : CKernel(m, n, w1){
};
};

#endif
