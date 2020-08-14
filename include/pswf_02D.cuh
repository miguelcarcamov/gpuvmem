#ifndef PSWF_02D_CUH
#define PSWF_02D_CUH

#include "framework.cuh"
#include "functions.cuh"

class PSWF_02D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
PSWF_02D() : CKernel(){
};
PSWF_02D(int m, int n) : CKernel(m, n){
};
PSWF_02D(int m, int n, float w1) : CKernel(m, n, w1){
};
};

#endif
