#include "gaussian2D.cuh"

__host__ __device__ void Gaussian2D::constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{

        float x, y;
        for(int i=0; i<this->M; i++) {
                for(int j=0; j<this->N; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->N*i+j] = gaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1, this->alpha);
                }
        }
};


namespace {
CKernel* CreateCKernel()
{
        return new Gaussian2D;
}
const int CKERNELID = 2;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
