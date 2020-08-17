#include "sinc2D.cuh"

__host__ __device__ void Sinc2D::constructKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = sinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1);
                }
        }

};

__host__ __device__ float Sinc2D::GCF_fn(float amp, float nu, float w)
{
        if(fabs(nu) < w)
                return amp*1.0f;
        else
                return 0.0f;
};

__device__ float Sinc2D::constructGCF(float amp, float x0, float y0, float sigma_x, float sigma_y, float w, int M, int N)
{
        const int i = threadIdx.y + blockDim.y * blockIdx.y;
        const int j = threadIdx.x + blockDim.x * blockIdx.x;

        float x = (j - x0) * sigma_x;
        float y = (i - y0) * sigma_y;

        float radius = distance(x, 0, y, 0);
        float val = GCF_fn(amp, radius, w);

        return val;

};

namespace {
CKernel* CreateCKernel()
{
        return new Sinc2D;
}
const int CKERNELID = 3;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
