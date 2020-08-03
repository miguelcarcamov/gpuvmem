#include "gaussian2D.cuh"

__host__ __device__ float Gaussian2D::run(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y)
{
        float value = gaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1, this->alpha);
        return value;
};

namespace {
CKernel* CreateCKernel()
{
        return new Gaussian2D;
}
const int CKERNELID = 2;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
