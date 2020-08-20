#include "pswf_12D.cuh"

__host__ __device__ void PSWF_12D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        float x, y;
        float val;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1);
                        this->kernel[this->n * i + j] = val;
                }
        }

};

__device__ float PSWF_12D::buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w)
{
        float val = pswf_12D(amp, x, y, x0, y0, sigma_x, sigma_y, w);
        return 1.0f/val;
}


namespace {
CKernel* CreateCKernel()
{
        return new PSWF_12D;
}
const int CKERNELID = 6;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
