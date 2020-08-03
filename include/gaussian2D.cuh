#ifndef GAUSSIAN2D_CUH
#define GAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Gaussian2D : public CKernel {
public:
__host__ __device__ float run(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y);
Gaussian2D() : CKernel(){
};
Gaussian2D(int M, int N) : CKernel(M, N){
};
};

#endif
