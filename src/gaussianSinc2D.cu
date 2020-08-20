#include "gaussianSinc2D.cuh"

__host__ __device__ void GaussianSinc2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussianSinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1, this->w2, this->alpha);
                }
        }

};

namespace {
CKernel* CreateCKernel()
{
        return new GaussianSinc2D;
}
const int CKERNELID = 4;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
