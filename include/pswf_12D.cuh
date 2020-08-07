#ifndef PSWF_12D_CUH
#define PSWF_12D_CUH

#include "framework.cuh"
#include "functions.cuh"

class PSWF_12D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
PSWF_12D() : CKernel(){
};
PSWF_12D(int M, int N) : CKernel(M, N){
};
};

#endif
