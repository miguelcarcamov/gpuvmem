#ifndef PSWF_12D_CUH
#define PSWF_12D_CUH

#include "framework.cuh"
#include "functions.cuh"

class PSWF_12D : public CKernel {
public:
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__device__ float buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w);
__host__ __device__ PSWF_12D() : CKernel(){
        this->w1 = 6;
};
__host__ __device__ PSWF_12D(int m, int n) : CKernel(m, n){
        this->w1 = 6;
};
__host__ __device__ PSWF_12D(int m, int n, float w1) : CKernel(m, n, w1){
};
};

#endif
