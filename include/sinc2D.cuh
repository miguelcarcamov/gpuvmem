#ifndef SINC2D_CUH
#define SINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class Sinc2D : public CKernel {
public:
__host__ __device__ float run(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y);
Sinc2D() : CKernel(){
};
Sinc2D(int M, int N) : CKernel(M, N){
};
};

#endif
