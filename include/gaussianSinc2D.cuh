#ifndef GAUSSIANSINC2D_CUH
#define GAUSSIANSINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class GaussianSinc2D : public CKernel {
public:
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__device__ float buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w){return 1.0f;};
__host__ __device__ GaussianSinc2D() : CKernel(){
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2.0f;
};
__host__ __device__ GaussianSinc2D(int m, int n) : CKernel(m, n){
        this->w1 = 2.52;
        this->w2 = 1.55;
        this->alpha = 2.0f;
};
__host__ __device__ GaussianSinc2D(int m, int n, float w1, float w2) : CKernel(m, n, w1, w2){
        this->alpha = 2.0f;
};
};

#endif
