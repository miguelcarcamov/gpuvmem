#include "gaussian2D.cuh"

__host__ float gaussian1D(float amp, float x, float x0, float sigma, float w, float alpha)
{
        float radius_x = distance(x, 0.0f, x0, 0.0f);
        float val = radius_x/(w*sigma);
        float val_alpha = powf(val, alpha);
        float G = amp*expf(-val_alpha);

        return G;
};

__host__ float gaussian2D(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        float radius_x = distance(x, 0.0f, x0, 0.0f);
        float radius_y = distance(0.0f, y, 0.0f, y0);

        if(radius_x < w*sigma_x && radius_y < w*sigma_y) {
                float fx = radius_x/(w*sigma_x);
                float fy = radius_y/(w*sigma_y);

                float val_x = powf(fx, alpha);
                float val_y = powf(fy, alpha);
                float G = amp*expf(-1.0f*(val_x+val_y));

                return G;
        }else
                return 0.0f;
};

__host__ Gaussian2D::Gaussian2D() : CKernel(){
        this->alpha=2.0f;
        this->w=1.0f;
        this->name="Gaussian";
};
__host__ Gaussian2D::Gaussian2D(int m, int n) : CKernel(m, n){
        this->alpha=2.0f;
        this->w=1.0f;
        this->name="Gaussian";
};
__host__ Gaussian2D::Gaussian2D(int m, int n, float w) : CKernel(m, n, w){
        this->alpha=2.0f;
        this->name="Gaussian";
};


__host__ void Gaussian2D::buildKernel(float amp, float x0, float y0, float sigma_x, float sigma_y)
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*sigma_y;
                        x = (j-this->support_x)*sigma_x;
                        this->kernel[this->n*i+j] = gaussian2D(amp, x, y, x0, y0, sigma_x, sigma_y, this->w, this->alpha);
                }
        }
        this->copyKerneltoGPU();
};

__host__ void Gaussian2D::buildKernel()
{
        this->setKernelMemory();
        float x, y;
        for(int i=0; i<this->m; i++) {
                for(int j=0; j<this->n; j++) {
                        y = (i-this->support_y)*this->sigma_y;
                        x = (j-this->support_x)*this->sigma_x;
                        this->kernel[this->n*i+j] = gaussian2D(this->amp, x, y, this->x0, this->y0, this->sigma_x, this->sigma_y, this->w, this->alpha);
                }
        }
        this->copyKerneltoGPU();
};

__host__ float Gaussian2D::GCF(float amp, float x, float y, float x0, float y0, float sigma_x, float sigma_y, float w, float alpha)
{
        return gaussian2D(amp, PI*x, PI*y, PI*x0, PI*y0, sigma_x, sigma_y, 2.0f*w, alpha);
};


namespace {
CKernel* CreateCKernel()
{
        return new Gaussian2D;
}

const std::string name = "Gaussian2D";
const bool RegisteredGaussian2D = registerCreationFunction<CKernel, std::string>(name, CreateCKernel);
};
