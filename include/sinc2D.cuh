#ifndef SINC2D_CUH
#define SINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Sinc2D : public CKernel {
public:
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__device__ float GCF_fn(float amp, float nu, float w);
__device__ float buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha);
__host__ __device__ Sinc2D() : CKernel(){
        this->w1 =1.55f;
};
__host__ __device__ Sinc2D(int m, int n) : CKernel(m, n){
        this->w1 =1.55f;
};
__host__ __device__ Sinc2D(int m, int n, float w1) : CKernel(m, n, w1){
};
};

#endif
