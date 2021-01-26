#include "gaussianSinc2D.cuh"

__host__ float gaussianSinc1D(float amp, float x, float x0, float sigma, float w1, float w2, float alpha)
{
        return amp*gaussian1D(1.0f, x, x0, sigma, w1, alpha)*sinc1D(1.0f, x, x0, sigma, w2);
};

__host__ float gaussianSinc2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w1, float w2, float alpha)
{
        float G = gaussian2D(1.0f, x, y, x0, y0, sigma_x, sigma_y, w1, alpha);
        float S = sinc2D(1.0f, x, x0, y, y0, sigma_x, sigma_y, w2);
        return amp*G*S;
};

__host__ GaussianSinc2D::GaussianSinc2D() : CKernel(){
        this->w = 2.52;
        this->w2 = 1.55;
        this->alpha = 2.0f;
        this->name = "Gaussian Sinc";
};
__host__ GaussianSinc2D::GaussianSinc2D(int m, int n) : CKernel(m, n){
        this->w = 2.52;
        this->w2 = 1.55;
        this->alpha = 2.0f;
        this->name = "Gaussian Sinc";
};
__host__ GaussianSinc2D::GaussianSinc2D(int m, int n, float w, float w2) : CKernel(m, n, w){
        this->w2 = w2;
        this->alpha = 2.0f;
        this->name = "Gaussian Sinc";
};

__host__ void GaussianSinc2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussianSinc2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w, this->w2, this->alpha);
                }
        }
        this->copyKerneltoGPU();

};

__host__ void GaussianSinc2D::buildKernel()
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussianSinc2D(this->amp, x, y, this->x0, this->y0, this->sigma_x, this->sigma_y, this->w, this->w2, this->alpha);
                }
        }
        this->copyKerneltoGPU();

};

namespace {
CKernel* CreateCKernel()
{
        return new GaussianSinc2D;
}

const std::string name = "GaussianSinc2D";
const bool RegisteredGaussianSinc2D = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
