#ifndef GAUSSIAN2D_CUH
#define GAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Gaussian2D : public CKernel {
public:
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__device__ float buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w);
__host__ __device__ Gaussian2D() : CKernel(){
        this->alpha=2.0f;
        this->w1=1.0f;
};
__host__ __device__ Gaussian2D(int m, int n) : CKernel(m, n){
        this->alpha=2.0f;
        this->w1=1.0f;
};
__host__ __device__ Gaussian2D(int m, int n, float w1) : CKernel(m, n, w1){
        this->alpha=2.0f;
};
};

#endif
