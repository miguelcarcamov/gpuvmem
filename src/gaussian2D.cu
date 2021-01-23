#include "gaussian2D.cuh"

__host__ void Gaussian2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w1, this->alpha);
                }
        }
        this->copyKerneltoGPU();
};

__host__ float Gaussian2D::GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        return gaussian2D(amp, PI*x, PI*y, PI*x0, PI*y0, sigma_x, sigma_y, 2.0f*w, alpha);
}


namespace {
CKernel* CreateCKernel()
{
        return new Gaussian2D;
}

const std::string name = "Gaussian2D";
const bool RegisteredGaussian2D = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
