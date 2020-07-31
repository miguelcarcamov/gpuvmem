#ifndef ELLIPTICGAUSSIAN2D_CUH
#define ELLIPTICGAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"

class EllipticalGaussian2D : public CKernel{
public:
__host__ __device__ float run(float M, float N);
};

#endif
