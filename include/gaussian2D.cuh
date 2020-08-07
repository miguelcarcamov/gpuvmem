#ifndef GAUSSIAN2D_CUH
#define GAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Gaussian2D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
Gaussian2D() : CKernel(){
};
Gaussian2D(int M, int N) : CKernel(M, N){
};
Gaussian2D(int M, int N, float w1) : CKernel(M, N, w1){
};
};

#endif
