#include "sinc2D.cuh"

__host__ __device__ float Sinc2D::run(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y)
{
        float value = sinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1);
        return value;
};

namespace {
CKernel* CreateCKernel()
{
        return new Sinc2D;
}
const int CKERNELID = 3;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
