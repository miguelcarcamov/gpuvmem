#ifndef GAUSSIAN2D_CUH
#define GAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Gaussian2D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
Gaussian2D() : CKernel(){
};
Gaussian2D(int m, int n) : CKernel(m, n){
};
Gaussian2D(int m, int n, float w1) : CKernel(m, n, w1){
};
};

#endif
