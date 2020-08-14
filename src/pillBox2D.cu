#include "pillBox2D.cuh"

__host__ __device__ void PillBox2D::constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        float limit_x = this->m * sigma_x / 2;
        float limit_y = this->n * sigma_y / 2;

        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = pillBox2D(amp, x, y, limit_x, limit_y);
                }
        }
};

namespace {
CKernel* CreateCKernel()
{
        return new PillBox2D;
}
const int CKERNELID = 0;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
