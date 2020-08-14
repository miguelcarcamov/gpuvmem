#include "ellipticalGaussian2D.cuh"

__host__ __device__ void EllipticalGaussian2D::constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = ellipticalGaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->angle);
                }
        }
};

namespace {
CKernel* CreateCKernel()
{
        return new EllipticalGaussian2D;
}
const int CKERNELID = 1;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
