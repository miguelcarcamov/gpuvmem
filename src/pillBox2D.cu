#include "pillBox2D.cuh"

__host__ __device__ float PillBox2D::run(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y)
{
        float limit_x = this->M * sigma_x / 2;
        float limit_y = this->N * sigma_y / 2;
        float value = pillBox2D(amp, x, y, limit_x, limit_y);
        return value;
};

namespace {
CKernel* CreateCKernel()
{
        return new PillBox2D;
}
const int CKERNELID = 0;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
