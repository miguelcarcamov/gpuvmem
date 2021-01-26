#ifndef GAUSSIAN2D_CUH
#define GAUSSIAN2D_CUH

#include "framework.cuh"
#include "functions.cuh"
__host__ float gaussian1D(float amp, float x, float x0, float sigma, float w, float alpha);
__host__ float gaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha);
class Gaussian2D : public CKernel {
private:
float alpha;
public:
__host__ Gaussian2D();
__host__ Gaussian2D(int m, int n);
__host__ Gaussian2D(int m, int n, float w);
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__host__ void buildKernel();
__host__ float GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha);
};

#endif
