#ifndef GAUSSIANSINC2D_CUH
#define GAUSSIANSINC2D_CUH

#include "framework.cuh"
#include "functions.cuh"
#include "gaussian2D.cuh"
#include "sinc2D.cuh"

__host__ float gaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha);
__host__ float gaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha);
class GaussianSinc2D : public CKernel {
private:
float alpha;
float w2;
public:
__host__ GaussianSinc2D();
__host__ GaussianSinc2D(int m, int n);
__host__ GaussianSinc2D(int m, int n, float w, float w2);
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__host__ void buildKernel();
__host__ float GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha){return 1.0f;};
};

#endif
