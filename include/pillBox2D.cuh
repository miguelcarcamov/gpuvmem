#ifndef PILLBOX2D_CUH
#define PILLBOX2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class PillBox2D : public CKernel {
public:
__host__ __device__ void constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
PillBox2D() : CKernel(){
};
PillBox2D(int m, int n) : CKernel(m, n){
};
};

#endif
