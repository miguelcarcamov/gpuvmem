#include "sinc2D.cuh"

__host__ void Sinc2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = sinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1);
                }
        }
        this->copyKerneltoGPU();

};

__device__ float Sinc2D::GCF_fn(float amp, float nu, float w)
{
        if(fabs(nu) < w)
                return amp;
        else
                return 0.0f;
};

__device__ float Sinc2D::buildGCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float distance_x = distance(x, y, x0, y0) * sigma_x;
        float distance_y = distance(x, y, x0, y0) * sigma_y;
        return GCF_fn(amp, distance_x, w*sigma_x)*GCF_fn(amp, distance_y, w*sigma_y);
};

namespace {
CKernel* CreateCKernel()
{
        return new Sinc2D;
}

const std::string name = "Sinc2D";
const bool RegisteredSinc2D = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
