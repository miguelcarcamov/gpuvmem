#ifndef ELLIPTICGAUSSIAN2D_CUH
#define ELLIPTICGAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class EllipticalGaussian2D : public CKernel{
public:
__host__ __device__ float run(float M, float N);
EllipticalGaussian2D(){};
EllipticalGaussian2D(float dx, float dy, int M, int N): CKernel(dx, dy, M, N){};
};

#endif
