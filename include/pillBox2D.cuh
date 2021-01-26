#ifndef PILLBOX2D_CUH
#define PILLBOX2D_CUH

#include "framework.cuh"
#include "functions.cuh"

__host__ float pillBox1D(float amp, float x, float limit);
__host__ float pillBox2D(float amp, float x, float y, float limit_x, float limit_y);
class PillBox2D : public CKernel {
public:
__host__ PillBox2D();
__host__ PillBox2D(int m, int n);
__host__ void buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y);
__host__ void buildKernel();
__host__ void buildGCF() override;
__host__ void buildGCF(float amp, float x0, float y0, float sigma_x, float sigma_y) override;
__host__ float GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y) override;
};

#endif
