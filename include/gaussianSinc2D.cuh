#ifndef GAUSSIANSINC2D_CUH
#define GAUSSIANSINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class GaussianSinc2D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
GaussianSinc2D() : CKernel(){
};
GaussianSinc2D(int m, int n) : CKernel(m, n){
};
GaussianSinc2D(int m, int n, float w1, float w2) : CKernel(m, n, w1, w2){
};
};

#endif
