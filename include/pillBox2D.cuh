#ifndef PILLBOX2D_CUH
#define PILLBOX2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class PillBox2D : public CKernel {
public:
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__device__ float buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w);
__host__ __device__ PillBox2D() : CKernel(){
        this->setmn(1, 1);

};
__host__ __device__ PillBox2D(int m, int n) : CKernel(m, n){
};
};

#endif
