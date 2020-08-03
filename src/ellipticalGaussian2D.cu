#include "ellipticalGaussian2D.cuh"

__host__ __device__ float EllipticalGaussian2D::run(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y)
{
        float value = ellipticalGaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->angle);
        return value;
};

namespace {
CKernel* CreateCKernel()
{
        return new EllipticalGaussian2D;
}
const int CKERNELID = 1;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
